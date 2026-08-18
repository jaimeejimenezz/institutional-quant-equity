from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

NAVY = "#17324D"
SLATE = "#6B7C8F"
RED = "#A05A55"
GRID = "#E5E9ED"
TEXT = "#27323C"


def validation_suite_figure(summary: pd.DataFrame) -> go.Figure:
    """Show passing and failed checks by validation suite."""
    ordered = summary.sort_values("checks", ascending=True)

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=ordered["passed"],
            y=ordered["suite"],
            orientation="h",
            name="Passed",
            marker_color=NAVY,
            text=ordered["passed"].astype(str),
            textposition="inside",
        )
    )
    figure.add_trace(
        go.Bar(
            x=ordered["failed"],
            y=ordered["suite"],
            orientation="h",
            name="Attention",
            marker_color=RED,
            text=ordered["failed"].astype(str),
            textposition="inside",
        )
    )

    figure.update_layout(
        template="plotly_white",
        barmode="stack",
        height=430,
        margin=dict(l=150, r=25, t=25, b=50),
        font=dict(color=TEXT, size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(
            title="Controls",
            gridcolor=GRID,
            rangemode="tozero",
        ),
        yaxis=dict(title=""),
    )
    return figure


def violation_figure(summary: pd.DataFrame) -> go.Figure:
    """Show recorded violations by validation suite."""
    ordered = summary.sort_values("violations", ascending=True)
    colors = [
        NAVY if int(value) == 0 else RED
        for value in ordered["violations"].tolist()
    ]

    figure = go.Figure(
        go.Bar(
            x=ordered["violations"],
            y=ordered["suite"],
            orientation="h",
            marker_color=colors,
            text=ordered["violations"].astype(str),
            textposition="outside",
            name="Violations",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=150, r=35, t=25, b=50),
        font=dict(color=TEXT, size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title="Recorded violations",
            gridcolor=GRID,
            rangemode="tozero",
        ),
        yaxis=dict(title=""),
    )
    return figure
