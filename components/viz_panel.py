""""Visualization panel — control panel (left), scatter plot (right), molecule selection + multi-objective optimization with clean per-point outlines and stable interactions."""

import time
import streamlit as st
import pandas as pd
import numpy as np

from utils.theme import NORD
from utils.plotting import (
    build_scatter,                 # unchanged signature
    SCATTER_COLUMNS, PROPERTY_LABELS,
)
from components.molecule_card import show_molecule_dialog


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

LOWER_IS_BETTER_DEFAULTS = {
    "Hepatotoxicity probability",
    "MolLogP_unitless", "MolWt (g/mol)", "TPSA (Ang^2)",
}

ALL_PROPERTIES = [
    "binding_probability",
    "Hepatotoxicity probability",
    "Caco2",
    "Half_Life (h)",
    "LD50 (nM)",
    "MolLogP_unitless",
    "MolWt (g/mol)",
    "TPSA (Ang^2)",
    "hERG (nM)",
]


def _label(col: str) -> str:
    return PROPERTY_LABELS.get(col, col.replace("_", " ").title())


# ──────────────────────────────────────────────────────────────────────────────
# Data prep
# ──────────────────────────────────────────────────────────────────────────────

def _get_ids_in_plot_order(fig, df_scored):
    """Return IDs in the same order as the first trace points."""
    if fig.data and hasattr(fig.data[0], "customdata") and fig.data[0].customdata is not None:
        out = []
        for row in fig.data[0].customdata:
            if isinstance(row, (list, tuple)) and len(row) > 0:
                out.append(row[0])
            else:
                out.append(row)
        return out
    return df_scored["id"].tolist()


def _add_topk_outer_rings(fig, ids_in_plot_order, topk_ids, size_boost=6, line_width=3):
    """
    Draws a separate transparent markers trace with a gold outline, slightly larger than the base points.
    - size_boost: how many px larger than the base marker size
    - line_width: gold ring thickness
    """
    if not fig.data:
        return
    base = fig.data[0]
    if not hasattr(base, "x") or base.x is None or not hasattr(base, "y") or base.y is None:
        return

    id_to_idx = {mid: i for i, mid in enumerate(ids_in_plot_order)}
    ok_ids = [mid for mid in (topk_ids or []) if mid in id_to_idx]
    if not ok_ids:
        return

    # Determine base marker size (scalar or array)
    base_size = 9
    try:
        if hasattr(base, "marker") and base.marker is not None and getattr(base.marker, "size", None) is not None:
            if isinstance(base.marker.size, (list, tuple, np.ndarray)):
                # If array, just pick a reasonable default; halo must be same for all topK for clarity.
                base_size = int(np.median(base.marker.size)) if len(base.marker.size) > 0 else 9
            else:
                base_size = int(base.marker.size)
    except Exception:
        pass

    halo_size = max(3, base_size + int(size_boost))

    x_halo = [base.x[id_to_idx[mid]] for mid in ok_ids]
    y_halo = [base.y[id_to_idx[mid]] for mid in ok_ids]

    # Add a transparent marker with a gold outline; sits above the base trace
    fig.add_scattergl(
        x=x_halo, y=y_halo, mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(
            size=halo_size,
            color="rgba(0,0,0,0)",
            line=dict(color="#EBCB8B", width=line_width)
        )
    )

def _extract_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Pull nested properties / delta_properties to top-level columns."""
    if "properties" in df.columns:
        props_df = pd.json_normalize(df["properties"])
        for col in props_df.columns:
            if col not in df.columns:
                df[col] = props_df[col]

    if "delta_properties" in df.columns:
        delta_df = pd.json_normalize(df["delta_properties"])
        delta_df.columns = [f"delta_{c}" for c in delta_df.columns]
        for col in delta_df.columns:
            if col not in df.columns:
                df[col] = delta_df[col]
    return df


def _best_direction_default(prop: str) -> str:
    return "Lower is better" if prop in LOWER_IS_BETTER_DEFAULTS else "Higher is better"


def _score_column_for(prop: str, df: pd.DataFrame) -> str:
    """Which column to score for a given objective.

    - binding_probability uses ABSOLUTE value (not delta), by design.
    - Other properties prefer delta_ if present, else absolute.
    """
    if prop == "binding_probability":
        return "binding_probability"
    delta_col = f"delta_{prop}"
    return delta_col if delta_col in df.columns else prop


def _rank_sum_score(df: pd.DataFrame, props: list[str], directions: list[str]) -> pd.Series:
    """Rank-sum (higher is better after scaling 0..10)."""
    n = len(df)
    if n == 0 or not props:
        return pd.Series(5.0, index=df.index, dtype=float)  # flat color when no objectives

    score = pd.Series(0.0, index=df.index, dtype=float)
    for prop, direction in zip(props, directions):
        col = _score_column_for(prop, df)
        if col not in df.columns:
            continue
        ascending = (direction == "Lower is better")
        ranks = df[col].rank(ascending=ascending, method="min")
        score += (n + 1 - ranks)

    if score.max() > score.min():
        score = (score - score.min()) / (score.max() - score.min()) * 10.0
    else:
        score[:] = 5.0
    return score


def _objective_matrix(df: pd.DataFrame, props: list[str], directions: list[str]) -> np.ndarray:
    """Build matrix X for Pareto: smaller is better along each column."""
    if not props:
        return np.zeros((len(df), 0))

    cols = []
    for prop, direction in zip(props, directions):
        col = _score_column_for(prop, df)
        if col not in df.columns:
            cols.append(pd.Series(0.0, index=df.index))
            continue
        s = df[col].astype(float)
        if direction == "Higher is better":  # convert to "smaller is better"
            s = -s
        cols.append(s)
    return np.vstack([c.values for c in cols]).T


def _pareto_mask_multiobj(X: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-efficient points for 'smaller is better' X."""
    n = X.shape[0]
    if n == 0 or X.shape[1] == 0:
        return np.zeros((n,), dtype=bool)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        xi = X[i]
        for j in range(n):
            if i == j or dominated[i]:
                continue
            xj = X[j]
            if np.all(xj <= xi) and np.any(xj < xi):
                dominated[i] = True
    return ~dominated


def _nondominated_sort(X: np.ndarray) -> list[list[int]]:
    """
    Fast (O(N^2)) non-dominated sorting (NSGA-II style).
    X: ndarray shape (n_points, n_obj), smaller is better along every column.

    Returns: list of fronts; each front is a list of row indices (0-based).
    """
    if X.size == 0 or X.shape[1] == 0:
        return []

    n = X.shape[0]
    S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
    n_dom = np.zeros(n, dtype=int)  # n_dom[i] = count of how many points dominate i

    # For each pair, check dominance
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi, xj = X[i], X[j]
            # i dominates j if: all xi <= xj AND at least one xi < xj
            if np.all(xi <= xj) and np.any(xi < xj):
                S[i].append(j)
            # j dominates i if: all xj <= xi AND at least one xj < xi
            elif np.all(xj <= xi) and np.any(xj < xi):
                n_dom[i] += 1

    fronts = []
    current_front = [i for i in range(n) if n_dom[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front = []
        for p in current_front:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        current_front = next_front

    return fronts


def _apply_binding_guardrail(df: pd.DataFrame) -> pd.DataFrame:
    """Filter by binding_probability >= threshold if guardrail is enabled."""
    if "binding_probability" not in df.columns:
        return df
    if not st.session_state.get("bp_guard_enabled", False):
        return df
    try:
        thr = float(st.session_state.get("bp_guard_value", 0.0))
    except Exception:
        thr = 0.0
    return df[df["binding_probability"] >= thr].copy()


def _compute_scores_and_pareto(df: pd.DataFrame, props: list[str], directions: list[str]) -> pd.DataFrame:
    """Compute opt_score and Pareto flag given objectives/directions and guardrail."""
    df_local = _apply_binding_guardrail(df)
    df_local["opt_score"] = _rank_sum_score(df_local, props, directions)
    X = _objective_matrix(df_local, props, directions)
    mask = _pareto_mask_multiobj(X)
    df_local["pareto"] = False
    df_local.loc[df_local.index[mask], "pareto"] = True

    out = df.copy()
    out["opt_score"] = 5.0
    out["pareto"] = False
    out.loc[df_local.index, "opt_score"] = df_local["opt_score"]
    out.loc[df_local.index, "pareto"] = df_local["pareto"]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Session helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_full_scored_df() -> pd.DataFrame:
    parent_idx = st.session_state.selected_parent_index
    all_children = st.session_state.children_sets[parent_idx]
    all_df = pd.DataFrame(all_children)
    if all_df.empty:
        return all_df
    all_df = _extract_properties(all_df)

    obj_props = st.session_state.get("obj_props", [])
    obj_dirs = st.session_state.get("obj_dirs", {})
    directions = [obj_dirs.get(p, _best_direction_default(p)) for p in obj_props]

    all_df = _compute_scores_and_pareto(all_df, obj_props, directions)
    return all_df


def _get_filtered_df() -> pd.DataFrame:
    all_df = _get_full_scored_df()
    if all_df.empty:
        return all_df
    displayed_children = st.session_state.children_data
    displayed_ids = {c["id"] for c in displayed_children}
    df = all_df[all_df["id"].isin(displayed_ids)].copy()
    df["__ord"] = df.index
    df = df.sort_values("__ord").drop(columns="__ord")
    return df


def _axes_changed(new_x: str, new_y: str) -> bool:
    px = st.session_state.get("_prev_x_axis")
    py = st.session_state.get("_prev_y_axis")
    changed = (px != new_x) or (py != new_y)
    st.session_state["_prev_x_axis"] = new_x
    st.session_state["_prev_y_axis"] = new_y
    return changed


def _compute_and_cache_highlights(df: pd.DataFrame):
    """Compute Pareto + Top-K with multi-front support; store lists; return them with scored df."""
    obj_props = st.session_state.get("obj_props", [])
    obj_dirs = st.session_state.get("obj_dirs", {})
    directions = [obj_dirs.get(p, _best_direction_default(p)) for p in obj_props]
    k = max(1, int(st.session_state.get("top_k", 1)))

    df2 = df.copy()
    df2 = _compute_scores_and_pareto(df2, obj_props, directions)

    # Case A: No objectives selected
    if not obj_props:
        st.session_state.pareto_ids = []
        st.session_state.topk_ids = []
        return [], [], df2

    # Case B: Single objective
    if len(obj_props) == 1:
        prop = obj_props[0]
        direction = directions[0]

        # Get scoring column
        score_col = _score_column_for(prop, df2)

        # Ensure tie-breaker columns exist
        for col in ["binding_probability", "TPSA (Ang^2)"]:
            if col not in df2.columns:
                df2[col] = 0.0

        # Sort by property (direction-aware) + tie-breakers
        ascending = (direction == "Lower is better")
        sorted_df = df2.sort_values(
            by=[score_col, "binding_probability", "TPSA (Ang^2)"],
            ascending=[ascending, False, True],
            kind="mergesort",
        )

        # Top-K molecules
        topk_ids = sorted_df["id"].head(k).tolist()

        # Pareto = all molecules with best value (handles ties)
        best_val = sorted_df[score_col].iloc[0]
        if direction == "Lower is better":
            pareto_ids = sorted_df[sorted_df[score_col] == best_val]["id"].tolist()
        else:
            pareto_ids = sorted_df[sorted_df[score_col] == best_val]["id"].tolist()

        # Mark pareto column
        df2["pareto"] = df2["id"].isin(pareto_ids)

        st.session_state.pareto_ids = pareto_ids
        st.session_state.topk_ids = topk_ids
        return pareto_ids, topk_ids, df2

    # Case C: Multiple objectives (≥2)
    # Build objective matrix
    X = _objective_matrix(df2, obj_props, directions)

    # Non-dominated sorting to get all fronts
    fronts = _nondominated_sort(X)

    if not fronts:
        st.session_state.pareto_ids = []
        st.session_state.topk_ids = []
        return [], [], df2

    # First front = Pareto-optimal (semantic meaning)
    first_front_indices = fronts[0]
    pareto_ids = df2.iloc[first_front_indices]["id"].tolist()
    df2["pareto"] = df2["id"].isin(pareto_ids)

    # Fill up to K by stacking fronts
    def _front_sorted_ids(indices: list[int]) -> list[str]:
        """Sort molecules within a front by deterministic tie-breaker."""
        sub = df2.iloc[indices].copy()
        for col in ["binding_probability", "TPSA (Ang^2)"]:
            if col not in sub.columns:
                sub[col] = 0.0
        sub = sub.sort_values(
            by=["opt_score", "binding_probability", "TPSA (Ang^2)"],
            ascending=[False, False, True],
            kind="mergesort"
        )
        return sub["id"].tolist()

    pick_ids = []
    for front_indices in fronts:
        sorted_ids = _front_sorted_ids(front_indices)
        pick_ids.extend(sorted_ids)
        if len(pick_ids) >= k:
            break

    # Truncate to exactly K
    topk_ids = pick_ids[:k]

    st.session_state.pareto_ids = pareto_ids
    st.session_state.topk_ids = topk_ids
    return pareto_ids, topk_ids, df2


# ──────────────────────────────────────────────────────────────────────────────
# Per-point outline utilities (no shapes; no extra traces)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_outlines_to_main_trace(fig, ids_in_plot_order, topk_ids, tracked_ids):
    """
    Apply per-point marker line arrays on the main scatter trace ONLY:
      - Top-K: gold (#EBCB8B, width 3)
      - Tracked (manual): blue (#5E81AC, width 3)
      - Others: width 0
    No Pareto-only rings.
    """
    if not fig.data:
        return
    tr = fig.data[0]

    topk_set = set(topk_ids or [])
    tracked_set = set(tracked_ids or [])

    line_colors = []
    line_widths = []
    for mid in ids_in_plot_order:
        if mid in topk_set:
            line_colors.append("#EBCB8B")
            line_widths.append(3)
        elif mid in tracked_set:
            line_colors.append("#5E81AC")
            line_widths.append(3)
        else:
            line_colors.append("rgba(0,0,0,0)")
            line_widths.append(0)

    if not hasattr(tr, "marker") or tr.marker is None:
        tr.marker = {}
    tr.marker.update(line=dict(color=line_colors, width=line_widths))

# put this near the other small helpers
def _ids_in_plot_order(fig, df_scored):
    """
    Return the list of molecule IDs in the same order as the main trace.
    Prefers trace.customdata (first element per point) if available; otherwise
    falls back to df_scored order.
    """
    if fig.data and hasattr(fig.data[0], "customdata") and fig.data[0].customdata is not None:
        out = []
        for row in fig.data[0].customdata:
            if isinstance(row, (list, tuple)) and len(row) > 0:
                out.append(row[0])
            else:
                out.append(row)
        return out
    return df_scored["id"].tolist()


# ──────────────────────────────────────────────────────────────────────────────
# UI: Control panel
# ──────────────────────────────────────────────────────────────────────────────

def render_control_panel():
    parent = st.session_state.parent_data
    parents = st.session_state.parents_data

    # ── Logo + title ──
    st.markdown('<div class="moldelta-logo">MolDelta</div>', unsafe_allow_html=True)
    target_name = parent.get("target_name", "Target")
    parent_smiles = parent.get("smiles", "")
    st.markdown(
        f'<div class="target-title">{target_name}</div>'
        f'<div class="lead-opt-subtitle"><span class="lead-opt-label">Lead Optimization:</span> {parent_smiles}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Parent selector ──
    parent_names = [p.get("name", f"Parent {i+1}") for i, p in enumerate(parents)]
    selected_parent = st.selectbox(
        "Parent Molecule",
        parent_names,
        index=st.session_state.selected_parent_index,
        key="parent_selector"
    )
    new_idx = parent_names.index(selected_parent)
    if new_idx != st.session_state.selected_parent_index:
        st.session_state.selected_parent_index = new_idx
        st.session_state.parent_data = parents[new_idx]
        st.session_state.children_data = st.session_state.children_sets[new_idx]
        # Reset highlights/state to avoid bleed-through
        st.session_state.tracked_ids = set()
        st.session_state.pareto_ids = []
        st.session_state.topk_ids = []
        st.session_state.selected_molecule_id = None
        st.session_state.flow_step = "WELCOME"
        st.session_state.enumeration_completed = False
        st.rerun()

    # ── Number of molecules ──
    max_children = len(st.session_state.children_sets[st.session_state.selected_parent_index])
    num_options = [n for n in [10, 20, 30, 40, 50] if n <= max_children]
    num_options.append(max_children)
    num_options = sorted(set(num_options))
    display_options = [str(n) if n != max_children else f"All ({max_children})" for n in num_options]
    selected_num = st.selectbox(
        "# Molecules",
        display_options,
        index=min(1, len(display_options) - 1),
        key="num_selector"
    )
    new_num = max_children if selected_num.startswith("All") else int(selected_num)
    if new_num != st.session_state.get("num_to_generate", 20):
        st.session_state.num_to_generate = new_num
        st.session_state.selected_molecule_id = None

    # ── Generate button ──
    st.markdown("")
    st.markdown('<div class="generate-btn">', unsafe_allow_html=True)
    if st.button("Generate Children", use_container_width=True):
        st.session_state.flow_step = "ENUMERATING"
        st.session_state.enumeration_completed = False
        st.session_state.selected_molecule_id = None
        st.session_state.tracked_ids = set()
        st.session_state.pareto_ids = []
        st.session_state.topk_ids = []
        all_children = st.session_state.children_sets[st.session_state.selected_parent_index]
        st.session_state.children_data = all_children[:st.session_state.num_to_generate]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Advanced Optimization (collapsible) ──
    st.markdown("---")
    with st.expander("⚙️ Advanced Optimization", expanded=False):
        st.caption("Optimization objectives. Select properties; each can be Higher/Lower.")

        # Map property keys to human-readable labels
        property_labels_list = [_label(p) for p in ALL_PROPERTIES]
        label_to_key = {_label(p): p for p in ALL_PROPERTIES}

        default_obj = st.session_state.get("obj_props", [])
        default_labels = [_label(p) for p in default_obj if p in ALL_PROPERTIES]

        selected_labels = st.multiselect(
            "Objectives",
            options=property_labels_list,
            default=default_labels,
            key="obj_props_selector",
        )

        # Convert selected labels back to property keys
        obj_props = [label_to_key[lbl] for lbl in selected_labels]

        obj_dirs_state = st.session_state.get("obj_dirs", {})
        new_dirs = {}
        for prop in obj_props:
            current = obj_dirs_state.get(prop, _best_direction_default(prop))
            new_dirs[prop] = st.radio(
                f"{_label(prop)}",
                ["Higher is better", "Lower is better"],
                horizontal=True,
                index=0 if current == "Higher is better" else 1,
                key=f"obj_dir_{prop}",
            )

        # Top-K slider
        top_k_widget_val = st.slider(
            "Highlight top-K within Pareto",
            min_value=1, max_value=10,
            value=int(st.session_state.get("top_k", 1)),
            key="top_k_slider"
        )

        # Binding guardrail
        st.markdown("")
        st.caption("Binding probability guardrail: Keep only molecules with binding_probability ≥ threshold")
        prev_enabled = bool(st.session_state.get("bp_guard_enabled", False))
        prev_value = float(st.session_state.get("bp_guard_value", 0.50))

        cols_bp = st.columns([2, 2])
        with cols_bp[0]:
            enabled_widget = st.checkbox(
                "Enable guardrail",
                value=prev_enabled,
                key="bp_guard_enabled_widget"
            )
        with cols_bp[1]:
            txt_val = st.text_input(
                "Threshold (0.00–1.00)",
                value=f"{prev_value:.2f}",
                key="bp_guard_text"
            )
            try:
                txt_float = float(txt_val)
            except Exception:
                txt_float = prev_value
            txt_float = max(0.0, min(1.0, txt_float))

        # Update state keys AFTER widgets exist
        st.session_state["bp_guard_enabled"] = bool(enabled_widget)
        st.session_state["bp_guard_value"] = float(txt_float)

        # Detect changes
        changed = (
            set(obj_props) != set(st.session_state.get("obj_props", []))
            or any(st.session_state.get("obj_dirs", {}).get(k) != v for k, v in new_dirs.items())
            or int(top_k_widget_val) != int(st.session_state.get("top_k", 1))
            or bool(enabled_widget) != prev_enabled
            or abs(float(txt_float) - float(prev_value)) > 1e-12
        )

        st.session_state.obj_props = obj_props
        st.session_state.obj_dirs = new_dirs
        st.session_state.top_k = int(top_k_widget_val)

        if changed:
            st.session_state.pareto_ids = []
            st.session_state.topk_ids = []
            st.session_state.selected_molecule_id = None
            st.rerun()

    # ── Tracked controls ──
    st.markdown("---")
    cols_trk = st.columns(2)
    with cols_trk[0]:
        if st.button("Clear Tracked", use_container_width=True):
            st.session_state.tracked_ids = set()
            st.session_state.selected_molecule_id = None
            st.rerun()
    with cols_trk[1]:
        sel = st.session_state.get("selected_molecule_id")
        tracked = set(st.session_state.get("tracked_ids", set()))
        label = "Track Selected" if (sel and sel not in tracked) else "Untrack Selected"
        if st.button(label, use_container_width=True, disabled=sel is None):
            s = set(tracked)
            if sel in s:
                s.remove(sel)
            else:
                s.add(sel)
            st.session_state.tracked_ids = s
            st.rerun()

    # ── Export ──
    if st.session_state.enumeration_completed:
        st.markdown("---")
        df = _get_filtered_df()
        if not df.empty:
            drop_cols = ["generation", "reaction_type", "modification",
                         "binding_affinity_log10",
                         "is_known_binder", "literature_source"]
            export_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
            export_df["pareto"] = df["pareto"]

            # Add column to mark molecules highlighted by Top-K selector
            topk_ids = set(st.session_state.get("topk_ids", []))
            export_df["top_k_highlighted"] = export_df["id"].isin(topk_ids)

            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("Export CSV", data=csv_data,
                               file_name="moldelta_results.csv", mime="text/csv",
                               use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# UI: Visualization panel
# ──────────────────────────────────────────────────────────────────────────────

def render_viz_panel():
    step = st.session_state.flow_step
    if step == "WELCOME":
        _render_welcome_viz()
    elif step == "ENUMERATING":
        _render_animated_enumeration()
    elif step in ("RESULTS", "FILTERING"):
        _render_results_viz()


def _render_welcome_viz():
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        height=550, margin=dict(l=60, r=60, t=20, b=60),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
        transition={'duration': 0}, uirevision="stay"
    )
    fig.add_annotation(
        text="<b>Ready for Analysis</b><br><br>Select a parent molecule and click <b>Generate Children</b> to begin",
        xref="paper", yref="paper", x=0.5, y=0.5, xanchor="center", yanchor="middle",
        showarrow=False, font=dict(size=18, color=NORD['snow_1']), align="center"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def _render_animated_enumeration():
    import time
    from plotly import graph_objects as go

    all_df = _get_full_scored_df()
    if all_df.empty:
        st.warning("No children data available.")
        st.session_state.flow_step = "WELCOME"
        return

    displayed_children = st.session_state.children_data
    displayed_ids = {c["id"] for c in displayed_children}
    df = all_df[all_df["id"].isin(displayed_ids)].copy()

    # Stable order → prevents jitter
    df["__ord"] = df.index
    df = df.sort_values("__ord").drop(columns="__ord")

    x_axis = st.session_state.get("x_axis", "Half_Life (h)")
    y_axis = st.session_state.get("y_axis", "hERG (nM)")

    # Fixed axes for the whole animation
    x_min, x_max = df[x_axis].min(), df[x_axis].max()
    y_min, y_max = df[y_axis].min(), df[y_axis].max()
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 0.5
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.5
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]

    chart_placeholder = st.empty()
    overlay_placeholder = st.empty()

    total = len(df)
    batch_size = max(1, total // 20)

    # Empty chart first (no transitions)
    fig_empty = build_scatter(
        df.iloc[:0], x_col=x_axis, y_col=y_axis,
        color_col="opt_score", selected_id=None,
        animate_transition=False
    )
    fig_empty.update_xaxes(range=x_range, autorange=False)
    fig_empty.update_yaxes(range=y_range, autorange=False)
    fig_empty.update_layout(
        height=550, margin=dict(l=60, r=60, t=20, b=60),
        showlegend=True, transition={'duration': 0}, uirevision="stay"
    )
    chart_placeholder.plotly_chart(fig_empty, use_container_width=True, config={'displayModeBar': False})

    # Reveal batches
    for i in range(batch_size, total + 1, batch_size):
        end_idx = min(i, total)
        subset = df.iloc[:end_idx]

        pareto_ids, topk_ids, subset_scored = _compute_and_cache_highlights(subset)

        # Base scatter
        fig = build_scatter(
            subset_scored, x_col=x_axis, y_col=y_axis,
            color_col="opt_score", selected_id=None,
            animate_transition=False
        )
        fig.update_xaxes(range=x_range, autorange=False)
        fig.update_yaxes(range=y_range, autorange=False)
        fig.update_layout(
            height=550, margin=dict(l=60, r=60, t=20, b=60),
            showlegend=True, transition={'duration': 0}, uirevision="stay"
        )

        # ---- Top-K gold halo (visible ring) — SVG layer, on top ----
        if topk_ids:
            tk = subset_scored[subset_scored["id"].isin(topk_ids)]
            fig.add_trace(go.Scatter(
                x=tk[x_axis], y=tk[y_axis], mode="markers", hoverinfo="skip",
                showlegend=False, cliponaxis=False,
                marker=dict(
                    symbol="circle-open",
                    size=20,                 # bigger, clearly offset
                    line=dict(color="#EBCB8B", width=4)
                ),
                name=""
            ))

        chart_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # UI overlay
        current_best = subset_scored["opt_score"].max() if "opt_score" in subset_scored.columns else 0
        progress_percent = (end_idx / total) * 100
        overlay_placeholder.markdown(
            f"""
            <div class="enum-overlay" style="
                position: fixed; top: 50%; left: 60%; transform: translate(-50%, -50%);
                background: linear-gradient(135deg, rgba(46, 52, 64, 0.95) 0%, rgba(59, 66, 82, 0.95) 100%);
                border: 2px solid #4C566A; border-radius: 16px; padding: 2rem 3rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6); z-index: 9999;
                min-width: 400px; max-width: 480px; text-align: center;">
                <div style="font-size: 1.4rem; font-weight: 600; color: #ECEFF4; margin-bottom: 0.8rem;">
                    Optimizing candidates...
                </div>
                <div style="font-size: 0.95rem; color: #D8DEE9; margin-bottom: 1.2rem;">
                    Molecules processed: {end_idx} / {total}
                </div>
                <div style="width: 100%; height: 8px; background: #3B4252; border-radius: 4px; overflow: hidden; margin-bottom: 0.5rem;">
                    <div style="height: 100%; width: {progress_percent}%; background: linear-gradient(90deg, #5E81AC 0%, #81A1C1 100%); border-radius: 4px; transition: width 0.3s ease;"></div>
                </div>
                <div style="font-size: 0.9rem; color: #88C0D0; margin-top: 0.3rem;">
                    Best score: {current_best:.1f}/10
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.08)

    overlay_placeholder.empty()
    st.session_state.enumeration_completed = True
    st.session_state.flow_step = "RESULTS"
    st.rerun()

def _resolve_clicked_id(event, df_scored: pd.DataFrame):
    """Extract clicked molecule id from a plotly selection event, robustly."""
    if not event or not hasattr(event, "selection") or not event.selection:
        return None
    points = event.selection.get("points", [])
    point_indices = event.selection.get("point_indices", [])

    if points:
        point = points[0]
        if "customdata" in point:
            data = point["customdata"]
            return data[0] if isinstance(data, (list, tuple)) else data
        idx = point.get("point_index", point.get("pointIndex"))
        if idx is not None and idx < len(df_scored):
            return df_scored.iloc[idx]["id"]
    elif point_indices:
        idx = point_indices[0]
        if idx < len(df_scored):
            return df_scored.iloc[idx]["id"]
    return None


def _render_results_viz():
    from plotly import graph_objects as go

    if not st.session_state.get("enumeration_completed", False):
        _render_welcome_viz()
        return

    df = _get_filtered_df()
    if df.empty:
        st.warning("No molecules to display.")
        return

    pareto_ids, topk_ids, df_scored = _compute_and_cache_highlights(df)

    x_axis = st.session_state.get("x_axis", "Half_Life (h)")
    y_axis = st.session_state.get("y_axis", "hERG (nM)")

    tracked_ids = set(st.session_state.get("tracked_ids", set()))

    animate_axes = _axes_changed(x_axis, y_axis)

    # Base scatter
    fig = build_scatter(
        df_scored, x_col=x_axis, y_col=y_axis,
        color_col="opt_score",
        selected_id=None,
        animate_transition=bool(animate_axes)
    )
    fig.update_layout(
        height=550, margin=dict(l=60, r=60, t=20, b=60),
        showlegend=True,
        transition={'duration': 0 if not animate_axes else 200},
        uirevision="stay"
    )

    # ---- Top-K gold halo (SVG) ----
    if topk_ids:
        tk = df_scored[df_scored["id"].isin(topk_ids)]
        fig.add_trace(go.Scatter(
            x=tk[x_axis], y=tk[y_axis], mode="markers", hoverinfo="skip",
            showlegend=False, cliponaxis=False,
            marker=dict(
                symbol="circle-open",
                size=20,
                line=dict(color="#EBCB8B", width=4)
            ),
            name=""
        ))

    # ---- Tracked (manual) blue inner ring (SVG) ----
    if tracked_ids:
        tr = df_scored[df_scored["id"].isin(tracked_ids)]
        fig.add_trace(go.Scatter(
            x=tr[x_axis], y=tr[y_axis], mode="markers", hoverinfo="skip",
            showlegend=False, cliponaxis=False,
            marker=dict(
                symbol="circle-open",
                size=14,  # smaller than Top-K halo
                line=dict(color="#88C0D0", width=3)
            ),
            name=""
        ))

    event = st.plotly_chart(
        fig, key="main_scatter", on_select="rerun",
        use_container_width=True, config={'displayModeBar': False}
    )

    # ── Axis selectors (side-by-side below graph) ──
    axis_options = SCATTER_COLUMNS
    axis_labels = [_label(c) for c in axis_options]

    axis_cols = st.columns(2)
    with axis_cols[0]:
        current_x = st.session_state.get("x_axis", "Half_Life (h)")
        x_idx = axis_options.index(current_x) if current_x in axis_options else 0
        x_label = st.selectbox("X-Axis", axis_labels, index=x_idx, key="x_axis_selector")
        new_x = axis_options[axis_labels.index(x_label)]
        if new_x != st.session_state.get("x_axis"):
            st.session_state.x_axis = new_x
            st.session_state.selected_molecule_id = None
            st.rerun()

    with axis_cols[1]:
        current_y = st.session_state.get("y_axis", "hERG (nM)")
        y_idx = axis_options.index(current_y) if current_y in axis_options else 0
        y_label = st.selectbox("Y-Axis", axis_labels, index=y_idx, key="y_axis_selector")
        new_y = axis_options[axis_labels.index(y_label)]
        if new_y != st.session_state.get("y_axis"):
            st.session_state.y_axis = new_y
            st.session_state.selected_molecule_id = None
            st.rerun()

    # Manual selection toggling (track/untrack)
    clicked_id = _resolve_clicked_id(event, df_scored)
    if clicked_id:
        if clicked_id in tracked_ids:
            tracked_ids.remove(clicked_id)
            st.session_state.tracked_ids = set(tracked_ids)
            st.session_state.selected_molecule_id = None
        else:
            tracked_ids.add(clicked_id)
            st.session_state.tracked_ids = set(tracked_ids)
            st.session_state.selected_molecule_id = clicked_id
        st.rerun()

    # Open molecule dialog only on new selection
    sel_id = st.session_state.get("selected_molecule_id", None)
    if sel_id and sel_id in df_scored["id"].values:
        mol_row = df_scored[df_scored["id"] == sel_id].iloc[0].to_dict()
        parent = st.session_state.parent_data
        show_molecule_dialog(mol_row, parent)
