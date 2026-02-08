"""Visualization panel — control panel (left), scatter plot (right), molecule selection."""

import time
import streamlit as st
import pandas as pd

from utils.theme import NORD
from utils.plotting import (
    build_scatter,
    SCATTER_COLUMNS, PROPERTY_LABELS,
)
from components.molecule_card import show_molecule_dialog


# Properties where lower values are better
LOWER_IS_BETTER = {
    "Hepatotoxicity probability", "hERG (nM)",
    "MolLogP_unitless", "MolWt (g/mol)", "TPSA (Ang^2)",
}

ALL_PROPERTIES = [
    "binding_probability",
    "Hepatotoxicity probability",
    "Caco2",
    "Half_Life (h)",
    "LD50 (nM)",
    "hERG (nM)",
    "MolLogP_unitless",
    "MolWt (g/mol)",
    "TPSA (Ang^2)",
]


def _label(col):
    return PROPERTY_LABELS.get(col, col.replace("_", " ").title())


def _extract_properties(df):
    """Extract nested properties and delta_properties into top-level columns."""
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


def _compute_opt_score(df):
    """Compute optimization score by ranking children on each property's delta.

    For each property, rank all children by delta in the improvement direction.
    A child's score = sum of (N - rank) across all 8 properties, scaled to 0-10.
    The child that outranks siblings on the most properties gets the highest score.
    """
    n = len(df)
    if n == 0:
        df["opt_score"] = pd.Series(dtype=float)
        return df

    df["opt_score"] = 0.0

    for prop in ALL_PROPERTIES:
        delta_col = f"delta_{prop}"
        if delta_col not in df.columns:
            continue

        # Rank by delta: best improvement = rank 1
        # For higher-is-better: highest delta is best → ascending=False
        # For lower-is-better: most negative delta is best → ascending=True
        ascending = prop in LOWER_IS_BETTER
        ranks = df[delta_col].rank(ascending=ascending, method="min")

        # Convert rank to points: rank 1 gets (n) points, rank n gets 1 point
        df["opt_score"] += (n + 1 - ranks)

    # Scale to 0-10
    max_score = df["opt_score"].max()
    min_score = df["opt_score"].min()
    score_range = max_score - min_score
    if score_range > 0:
        df["opt_score"] = ((df["opt_score"] - min_score) / score_range) * 10
    else:
        df["opt_score"] = 5.0

    return df


def _get_full_scored_df():
    """Compute opt_score on the FULL children set for absolute normalization."""
    parent_idx = st.session_state.selected_parent_index
    all_children = st.session_state.children_sets[parent_idx]
    all_df = pd.DataFrame(all_children)

    if all_df.empty:
        return all_df

    all_df = _extract_properties(all_df)
    all_df = _compute_opt_score(all_df)
    return all_df


def _get_filtered_df():
    """Prepare displayed children with absolute opt_scores (normalized against full set)."""
    all_df = _get_full_scored_df()
    if all_df.empty:
        return all_df

    # Filter to only the displayed children
    displayed_children = st.session_state.children_data
    displayed_ids = {c["id"] for c in displayed_children}
    df = all_df[all_df["id"].isin(displayed_ids)].copy()

    # Sort by opt_score descending (best first)
    df = df.sort_values("opt_score", ascending=False)
    return df


def render_control_panel():
    """Render the left control panel with selectors, generate button, and axis controls."""
    parent = st.session_state.parent_data
    parents = st.session_state.parents_data

    # ── Logo ──
    st.markdown('<div class="moldelta-logo">MolDelta</div>', unsafe_allow_html=True)

    # ── Title area ──
    target_name = parent.get("target_name", "Target")
    parent_smiles = parent.get("smiles", "")
    st.markdown(
        f'<div class="target-title">{target_name}</div>'
        f'<div class="lead-opt-subtitle">'
        f'<span class="lead-opt-label">Lead Optimization:</span> {parent_smiles}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Parent selector ──
    parent_names = [p.get("name", f"Parent {i+1}") for i, p in enumerate(parents)]
    selected_parent = st.selectbox(
        "Parent Molecule",
        parent_names,
        index=st.session_state.selected_parent_index,
        key="parent_selector",
    )
    new_idx = parent_names.index(selected_parent)
    if new_idx != st.session_state.selected_parent_index:
        st.session_state.selected_parent_index = new_idx
        st.session_state.parent_data = parents[new_idx]
        st.session_state.children_data = st.session_state.children_sets[new_idx]
        st.session_state.flow_step = "WELCOME"
        st.session_state.enumeration_completed = False
        st.session_state.selected_molecule_id = None
        st.session_state.top_ids = []
        st.rerun()

    # ── Number of molecules selector ──
    max_children = len(st.session_state.children_sets[st.session_state.selected_parent_index])
    num_options = [n for n in [10, 20, 30, 40, 50] if n <= max_children]
    num_options.append(max_children)
    num_options = sorted(set(num_options))
    display_options = [str(n) if n != max_children else f"All ({max_children})" for n in num_options]
    selected_num = st.selectbox(
        "# Molecules",
        display_options,
        index=min(1, len(display_options) - 1),
        key="num_selector",
    )
    new_num = max_children if selected_num.startswith("All") else int(selected_num)
    if new_num != st.session_state.num_to_generate:
        st.session_state.num_to_generate = new_num
        st.session_state.selected_molecule_id = None

    # ── Generate button ──
    st.markdown("")
    st.markdown('<div class="generate-btn">', unsafe_allow_html=True)
    if st.button("Generate Children", use_container_width=True):
        st.session_state.flow_step = "ENUMERATING"
        st.session_state.enumeration_completed = False
        st.session_state.selected_molecule_id = None
        st.session_state.best_molecule_id = None
        st.session_state.original_best_id = None
        all_children = st.session_state.children_sets[st.session_state.selected_parent_index]
        st.session_state.children_data = all_children[:st.session_state.num_to_generate]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Axis selectors (always visible so user can set before generating) ──
    st.markdown("---")
    axis_options = SCATTER_COLUMNS
    axis_labels = [_label(c) for c in axis_options]

    current_x = st.session_state.get("x_axis", "Half_Life (h)")
    x_idx = axis_options.index(current_x) if current_x in axis_options else 0
    x_label = st.selectbox("X-Axis", axis_labels, index=x_idx, key="x_axis_selector")
    new_x = axis_options[axis_labels.index(x_label)]
    if new_x != st.session_state.x_axis:
        st.session_state.x_axis = new_x
        st.session_state.selected_molecule_id = None
        st.rerun()

    current_y = st.session_state.get("y_axis", "hERG (nM)")
    y_idx = axis_options.index(current_y) if current_y in axis_options else 0
    y_label = st.selectbox("Y-Axis", axis_labels, index=y_idx, key="y_axis_selector")
    new_y = axis_options[axis_labels.index(y_label)]
    if new_y != st.session_state.y_axis:
        st.session_state.y_axis = new_y
        st.session_state.selected_molecule_id = None
        st.rerun()

    # ── Export button (only after enumeration) ──
    if st.session_state.enumeration_completed:
        st.markdown("---")
        df = _get_filtered_df()
        if not df.empty:
            drop_cols = ["generation", "reaction_type", "modification",
                         "binding_affinity_log10",
                         "is_known_binder", "literature_source"]
            export_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Export CSV",
                data=csv_data,
                file_name="moldelta_results.csv",
                mime="text/csv",
                use_container_width=True,
            )


def render_viz_panel():
    """Render the right visualization panel (chart + click handling)."""
    step = st.session_state.flow_step

    if step == "WELCOME":
        _render_welcome_viz()
    elif step == "ENUMERATING":
        _render_animated_enumeration()
    elif step in ("RESULTS", "FILTERING"):
        _render_results_viz()


def _render_welcome_viz():
    """Welcome state — show placeholder prompting the user to generate."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        height=700,
        margin=dict(l=60, r=60, t=20, b=60),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )

    fig.add_annotation(
        text="<b>Ready for Analysis</b><br><br>"
             "Select a parent molecule and click <b>Generate Children</b> to begin",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font=dict(size=18, color=NORD['snow_1']),
        align="center"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={'displayModeBar': False}
    )


def _render_animated_enumeration():
    """Show animated enumeration with molecules appearing one by one."""
    # Compute opt_score against FULL set for absolute normalization
    all_df = _get_full_scored_df()

    if all_df.empty:
        st.warning("No children data available.")
        st.session_state.flow_step = "WELCOME"
        return

    # Filter to displayed children only
    displayed_children = st.session_state.children_data
    displayed_ids = {c["id"] for c in displayed_children}
    df = all_df[all_df["id"].isin(displayed_ids)].copy()

    # Sort by opt_score ascending (worst first, best appears last)
    df = df.sort_values("opt_score", ascending=True)

    # Get axes
    x_axis = st.session_state.get("x_axis", "Half_Life (h)")
    y_axis = st.session_state.get("y_axis", "hERG (nM)")

    # Calculate fixed axis ranges
    x_min, x_max = df[x_axis].min(), df[x_axis].max()
    y_min, y_max = df[y_axis].min(), df[y_axis].max()
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    x_range = [x_min - x_padding, x_max + x_padding]
    y_range = [y_min - y_padding, y_max + y_padding]

    chart_placeholder = st.empty()
    overlay_placeholder = st.empty()

    total = len(df)
    batch_size = max(1, total // 20)

    # Render initial empty chart
    fig_empty = build_scatter(
        df.iloc[:0],
        x_col=x_axis,
        y_col=y_axis,
        color_col="opt_score",
        selected_id=None,
        animate_transition=False
    )
    fig_empty.update_xaxes(range=x_range, autorange=False)
    fig_empty.update_yaxes(range=y_range, autorange=False)
    fig_empty.update_layout(
        height=700,
        margin=dict(l=60, r=60, t=20, b=60),
        showlegend=True,
    )
    chart_placeholder.plotly_chart(
        fig_empty,
        use_container_width=True,
        config={'displayModeBar': False}
    )

    # Show molecules in batches
    for i in range(batch_size, total + 1, batch_size):
        end_idx = min(i, total)
        subset = df.iloc[:end_idx]

        fig = build_scatter(
            subset,
            x_col=x_axis,
            y_col=y_axis,
            color_col="opt_score",
            selected_id=None,
            animate_transition=False
        )

        fig.update_xaxes(range=x_range, autorange=False)
        fig.update_yaxes(range=y_range, autorange=False)
        fig.update_layout(
            height=700,
            margin=dict(l=60, r=60, t=20, b=60),
            showlegend=True,
        )

        chart_placeholder.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False}
        )

        current_best = subset["opt_score"].max() if "opt_score" in subset.columns else 0
        progress_percent = (end_idx / total) * 100

        overlay_placeholder.markdown(
            f"""
            <div class="enum-overlay" style="
                position: fixed;
                top: 50%;
                left: 60%;
                transform: translate(-50%, -50%);
                background: linear-gradient(135deg, rgba(46, 52, 64, 0.95) 0%, rgba(59, 66, 82, 0.95) 100%);
                border: 2px solid #4C566A;
                border-radius: 16px;
                padding: 2rem 3rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
                z-index: 9999;
                min-width: 400px;
                max-width: 480px;
                text-align: center;
            ">
                <div style="font-size: 1.4rem; font-weight: 600; color: #ECEFF4; margin-bottom: 0.8rem;">
                    Optimizing sequences...
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

    # Remove overlay
    overlay_placeholder.empty()

    # Find and mark best molecule (opt_score = 10)
    if not df.empty:
        best_molecule = df.nlargest(1, "opt_score")["id"].tolist()
        st.session_state.top_ids = best_molecule

        fig = build_scatter(
            df,
            x_col=x_axis,
            y_col=y_axis,
            color_col="opt_score",
            selected_id=None,
            top_ids=best_molecule
        )

        fig.update_xaxes(range=x_range, autorange=False)
        fig.update_yaxes(range=y_range, autorange=False)
        fig.update_layout(
            height=700,
            margin=dict(l=60, r=60, t=20, b=60),
            showlegend=True,
        )

        chart_placeholder.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False},
        )

    # Transition to results
    st.session_state.enumeration_completed = True
    st.session_state.flow_step = "RESULTS"
    st.rerun()


def _render_results_viz():
    """Results state — interactive scatter plot with click-to-select."""
    if not st.session_state.get("enumeration_completed", False):
        _render_welcome_viz()
        return

    parent = st.session_state.parent_data
    df = _get_filtered_df()

    if df.empty:
        st.warning("No molecules to display.")
        return

    # Get axes
    x_axis = st.session_state.get("x_axis", "Half_Life (h)")
    y_axis = st.session_state.get("y_axis", "hERG (nM)")

    top_ids = st.session_state.get("top_ids", [])
    selected_id = st.session_state.selected_molecule_id

    fig = build_scatter(df, x_col=x_axis, y_col=y_axis,
                        color_col="opt_score", selected_id=selected_id,
                        top_ids=top_ids,
                        animate_transition=True)

    fig.update_layout(
        height=700,
        margin=dict(l=60, r=60, t=20, b=60),
        showlegend=True,
    )

    event = st.plotly_chart(
        fig, key="main_scatter",
        on_select="rerun",
        use_container_width=True,
        config={'displayModeBar': False}
    )

    # Handle click events
    if event and hasattr(event, "selection") and event.selection:
        points = event.selection.get("points", [])
        point_indices = event.selection.get("point_indices", [])

        if points:
            point = points[0]
            if "customdata" in point:
                clicked_data = point["customdata"]
                new_id = clicked_data[0] if isinstance(clicked_data, (list, tuple)) else clicked_data
                if new_id != st.session_state.selected_molecule_id:
                    st.session_state.selected_molecule_id = new_id
                    st.rerun()
            elif "point_index" in point or "pointIndex" in point:
                idx = point.get("point_index", point.get("pointIndex"))
                if idx is not None and idx < len(df):
                    new_id = df.iloc[idx]["id"]
                    if new_id != st.session_state.selected_molecule_id:
                        st.session_state.selected_molecule_id = new_id
                        st.rerun()
        elif point_indices:
            idx = point_indices[0]
            if idx < len(df):
                new_id = df.iloc[idx]["id"]
                if new_id != st.session_state.selected_molecule_id:
                    st.session_state.selected_molecule_id = new_id
                    st.rerun()

    # Selected molecule dialog - only show if molecule is selected
    if selected_id and selected_id in df["id"].values:
        mol_row = df[df["id"] == selected_id].iloc[0].to_dict()
        show_molecule_dialog(mol_row, parent)
