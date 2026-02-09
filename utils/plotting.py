"""Plotly chart constructors with consistent Nord theming + multi-ring highlights."""

import plotly.graph_objects as go
import pandas as pd

from utils.theme import NORD, plotly_layout_defaults

# Human-readable labels for property columns
PROPERTY_LABELS = {
    "binding_probability": "Binding Strength",
    "Hepatotoxicity probability": "Hepatotoxicity Risk Score",
    "Caco2": "Gut Absorption",
    "Half_Life (h)": "Half-Life",
    "LD50 (nM)": "Safety Margin (LD50)",
    "hERG (nM)": "Heart Safety (hERG)",
    "MolLogP_unitless": "Drug-likeness (LogP)",
    "MolWt (g/mol)": "Molecular Weight",
    "TPSA (Ang^2)": "Cell Permeability (TPSA)",
    "opt_score": "Optimization Score",
}

# Columns available for scatter axes (excludes opt_score)
SCATTER_COLUMNS = [
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


def _ring_trace(df, ids, x_col, y_col, color, size, name):
    """Add a ring overlay trace for a set of ids."""
    if not ids:
        return None
    sub = df[df["id"].isin(ids)]
    if sub.empty:
        return None
    return go.Scatter(
        x=sub[x_col],
        y=sub[y_col],
        mode="markers",
        name=name,
        marker=dict(
            size=size,
            color="rgba(0,0,0,0)",
            line=dict(width=3, color=color),
        ),
        showlegend=False,
        hoverinfo="skip",
    )


def build_scatter(df, x_col="Half_Life (h)", y_col="hERG (nM)",
                  color_col="opt_score", selected_id=None,
                  topk_ids=None, tracked_ids=None,
                  animate_transition=False):
    import plotly.graph_objects as go

    topk_ids = set(topk_ids or [])
    tracked_ids = set(tracked_ids or [])

    fig = go.Figure()

    if len(df) > 0:
        colorbar_config = dict(
            title=dict(text=_label(color_col), font=dict(size=11), side="right"),
            thickness=12, len=0.6, x=1.0, xanchor="left", xpad=10,
        )
        if color_col == "opt_score":
            colorbar_config.update({"tickmode": "array",
                                    "tickvals": [1, 5, 10],
                                    "ticktext": ["1", "5", "10"]})

        # Base points
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[y_col], mode="markers", name="Candidates",
            showlegend=False,
            marker=dict(
                size=11,
                color=df[color_col] if color_col in df.columns else None,
                colorscale=[[0.0, NORD["aurora_red"]],
                            [0.5, NORD["aurora_yellow"]],
                            [1.0, NORD["aurora_green"]]],
                showscale=True, colorbar=colorbar_config,
                line=dict(width=1, color=NORD["bg_light"]),
                cmin=0 if color_col == "opt_score" else None,
                cmax=10 if color_col == "opt_score" else None,
            ),
            text=df.get("name"),
            customdata=df["id"].values,
            hovertemplate=(f"<b>%{{text}}</b><br>"
                           f"{_label(x_col)}: %{{x:.2f}}<br>"
                           f"{_label(y_col)}: %{{y:.2f}}<br>"
                           "<extra></extra>")
        ))

    # Small cyan rings for manually tracked
    if tracked_ids:
        tr = df[df["id"].isin(tracked_ids)]
        if len(tr):
            fig.add_trace(go.Scatter(
                x=tr[x_col], y=tr[y_col], mode="markers", name="Tracked",
                marker=dict(size=17, color="rgba(0,0,0,0)",
                            line=dict(width=2.2, color=NORD["frost_1"])),
                showlegend=False, hoverinfo="skip",
            ))

    # Gold rings for Top-K within Pareto
    if topk_ids:
        tk = df[df["id"].isin(topk_ids)]
        if len(tk):
            fig.add_trace(go.Scatter(
                x=tk[x_col], y=tk[y_col], mode="markers", name="Top-K (Pareto)",
                marker=dict(size=21, color="rgba(0,0,0,0)",
                            line=dict(width=3, color=NORD["aurora_yellow"])),
                showlegend=False, hoverinfo="skip",
            ))

    # Blue ring for selected (on top)
    if selected_id is not None and selected_id in df["id"].values:
        sel = df[df["id"] == selected_id].iloc[0]
        fig.add_trace(go.Scatter(
            x=[sel[x_col]], y=[sel[y_col]], mode="markers", name="Selected",
            marker=dict(size=24, color="rgba(0,0,0,0)",
                        line=dict(width=3, color=NORD["frost_2"])),
            showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        **plotly_layout_defaults(),
        xaxis_title=_label(x_col), yaxis_title=_label(y_col),
        height=650, legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, font=dict(size=11)),
        dragmode="pan",
        transition={'duration': 600, 'easing': 'cubic-in-out'} if animate_transition else None,
    )
    return fig
