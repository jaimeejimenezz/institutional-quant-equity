from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from quant_equity.reporting.dashboard_catalog import strategy_label

_SELECTED = "#17324D"
_BENCHMARK = "#6B7680"
_BASELINE = "#B08D57"
_GRID = "#E3E7EB"
_TEXT = "#17212B"
_MUTED = "#66717D"
_BACKGROUND = "#FFFFFF"


def performance_figure(performance_index: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for (series, role), group in performance_index.groupby(["series", "role"], sort=False):
        color, width, dash = _line_style(str(role))
        label = "SPY" if series == "SPY" else strategy_label(str(series))
        ordered = group.sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=ordered["date"],
                y=ordered["index_value"],
                mode="lines",
                name=label,
                line={"color": color, "width": width, "dash": dash},
                hovertemplate="%{y:.1f}<extra></extra>",
            )
        )

    return _style_figure(
        fig,
        yaxis_title="Growth of 100",
        y_tickformat=".0f",
        height=420,
    )


def drawdown_figure(drawdowns: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for (series, role), group in drawdowns.groupby(["series", "role"], sort=False):
        color = _SELECTED if role == "selected" else _BENCHMARK
        width = 2.4 if role == "selected" else 1.8
        dash = "solid" if role == "selected" else "dot"
        label = "SPY" if series == "SPY" else strategy_label(str(series))
        ordered = group.sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=ordered["date"],
                y=ordered["drawdown"],
                mode="lines",
                name=label,
                line={"color": color, "width": width, "dash": dash},
                hovertemplate="%{y:.1%}<extra></extra>",
            )
        )

    min_drawdown = float(drawdowns["drawdown"].min()) if not drawdowns.empty else -0.1
    fig = _style_figure(
        fig,
        yaxis_title="Drawdown",
        y_tickformat=".0%",
        height=320,
    )
    fig.update_yaxes(range=[min_drawdown * 1.08, 0.01])
    return fig


def sector_exposure_figure(exposure: pd.DataFrame) -> go.Figure:
    ordered = exposure.sort_values("sector_weight", ascending=True).copy()
    ordered["label"] = ordered["sector_weight"].map(lambda value: f"{value:.1%}")

    max_weight = float(ordered["sector_weight"].max()) if not ordered.empty else 0.0
    fig = go.Figure(
        go.Bar(
            x=ordered["sector_weight"],
            y=ordered["sector"],
            orientation="h",
            marker={"color": _SELECTED},
            text=ordered["label"],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x:.1%}<extra></extra>",
        )
    )
    fig = _style_figure(
        fig,
        yaxis_title=None,
        y_tickformat=None,
        height=340,
        show_legend=False,
        x_tickformat=".0%",
    )
    fig.update_xaxes(range=[0.0, max_weight * 1.18 if max_weight > 0 else 1.0])
    return fig


def _line_style(role: str) -> tuple[str, float, str]:
    if role == "selected":
        return _SELECTED, 2.8, "solid"
    if role == "benchmark":
        return _BENCHMARK, 2.1, "solid"
    return _BASELINE, 1.9, "dot"


def _style_figure(
    fig: go.Figure,
    *,
    yaxis_title: str | None,
    y_tickformat: str | None,
    height: int,
    show_legend: bool = True,
    x_tickformat: str | None = None,
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 20, "r": 24, "t": 16, "b": 16},
        paper_bgcolor=_BACKGROUND,
        plot_bgcolor=_BACKGROUND,
        font={"color": _TEXT, "size": 12},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 12, "color": _TEXT},
            "traceorder": "normal",
        },
        showlegend=show_legend,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        title=None,
        tickfont={"size": 11, "color": _MUTED},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=_GRID,
        gridwidth=1,
        zeroline=False,
        title=yaxis_title,
        tickfont={"size": 11, "color": _MUTED},
        tickformat=y_tickformat,
    )
    if x_tickformat is not None:
        fig.update_xaxes(tickformat=x_tickformat)
    return fig
