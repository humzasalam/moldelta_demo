"""Plotly chart constructors with consistent Nord theming."""

import plotly.graph_objects as go
import pandas as pd

from utils.theme import NORD, plotly_layout_defaults

# Human-readable labels for property columns
PROPERTY_LABELS = {
    "binding_probability": "Binding Probability",
    "Hepatotoxicity probability": "Hepatotoxicity",
    "Caco2": "Caco-2 Permeability",
    "Half_Life (h)": "Half-Life (h)",
    "LD50 (nM)": "LD50 (nM)",
    "hERG (nM)": "hERG (nM)",
    "MolLogP_unitless": "LogP",
    "MolWt (g/mol)": "Mol. Weight (g/mol)",
    "TPSA (Ang^2)": "TPSA (A\u00b2)",
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


def build_scatter(df, x_col="Half_Life (h)", y_col="hERG (nM)",
                  color_col="opt_score", selected_id=None, top_ids=None,
                  animate_transition=False):
    """Build the main interactive scatter plot of children molecules.

    Args:
        df: DataFrame with children data.
        x_col: Column for X axis.
        y_col: Column for Y axis.
        color_col: Column for dot coloring.
        selected_id: ID of the currently selected molecule (blue ring).
        top_ids: List of IDs of top molecules from optimization (purple rings).
        animate_transition: If True, add smooth transition animations for axis changes.

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()

    if len(df) > 0:
        colorbar_config = dict(
            title=dict(
                text=_label(color_col),
                font=dict(size=11),
                side="right",
            ),
            thickness=12,
            len=0.6,
            x=1.0,
            xanchor="left",
            xpad=10,
        )

        if color_col == "opt_score":
            colorbar_config.update({
                "tickmode": "array",
                "tickvals": [1, 5, 10],
                "ticktext": ["1", "5", "10"],
            })

        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            name="Candidates",
            showlegend=False,
            marker=dict(
                size=11,
                color=df[color_col] if color_col in df.columns else None,
                colorscale=[
                    [0.0, NORD["aurora_red"]],
                    [0.5, NORD["aurora_yellow"]],
                    [1.0, NORD["aurora_green"]],
                ],
                showscale=True,
                colorbar=colorbar_config,
                line=dict(width=1, color=NORD["bg_light"]),
                cmin=0 if color_col == "opt_score" else None,
                cmax=10 if color_col == "opt_score" else None,
            ),
            text=df["name"],
            customdata=df["id"].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{_label(x_col)}: %{{x:.2f}}<br>"
                f"{_label(y_col)}: %{{y:.2f}}<br>"
                "<extra></extra>"
            ),
        ))

    # Selected molecule highlight ring (blue)
    if selected_id is not None and selected_id in df["id"].values:
        sel = df[df["id"] == selected_id].iloc[0]
        fig.add_trace(go.Scatter(
            x=[sel[x_col]],
            y=[sel[y_col]],
            mode="markers",
            name="Selected",
            marker=dict(
                size=20,
                color="rgba(0,0,0,0)",
                line=dict(width=3, color=NORD["frost_1"]),
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Best molecule from optimization (purple ring)
    if top_ids:
        top_in_df = [tid for tid in top_ids if tid in df["id"].values and tid != selected_id]
        if top_in_df:
            top_rows = df[df["id"].isin(top_in_df)]
            fig.add_trace(go.Scatter(
                x=top_rows[x_col],
                y=top_rows[y_col],
                mode="markers",
                name="Best",
                marker=dict(
                    size=22,
                    color="rgba(0,0,0,0)",
                    line=dict(width=2.5, color="#B48EAD"),
                ),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        **plotly_layout_defaults(),
        xaxis_title=_label(x_col),
        yaxis_title=_label(y_col),
        height=650,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        dragmode="pan",
        transition={
            'duration': 600,
            'easing': 'cubic-in-out'
        } if animate_transition else None,
    )

    return fig
