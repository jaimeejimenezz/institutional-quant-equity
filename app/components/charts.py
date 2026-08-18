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


_ALPHA_HELD = "#17324D"
_ALPHA_NOT_HELD = "#AAB3BC"
_COMPONENT_COLORS = {
    "Technical composite": "#4F6B84",
    "Elastic Net": "#2F6B5F",
    "LightGBM ranker": "#B08D57",
}


def alpha_ranking_figure(snapshot: pd.DataFrame, *, top_n: int = 15) -> go.Figure:
    ordered = snapshot.nsmallest(top_n, "rank").sort_values("rank", ascending=False)
    colors = [
        _ALPHA_HELD if float(weight) > 0.0 else _ALPHA_NOT_HELD
        for weight in ordered["selected_weight"]
    ]
    labels = ordered["percentile_score"].map(lambda value: f"{value:.0%}")

    fig = go.Figure(
        go.Bar(
            x=ordered["percentile_score"],
            y=ordered["ticker"],
            orientation="h",
            marker={"color": colors},
            text=labels,
            textposition="outside",
            cliponaxis=False,
            customdata=ordered[["rank", "sector", "selected_weight"]],
            hovertemplate=(
                "Rank %{customdata[0]}<br>"
                "%{customdata[1]}<br>"
                "Signal %{x:.1%}<br>"
                "Portfolio weight %{customdata[2]:.2%}<extra></extra>"
            ),
        )
    )
    fig = _style_figure(
        fig,
        yaxis_title=None,
        y_tickformat=None,
        height=430,
        show_legend=False,
        x_tickformat=".0%",
    )
    fig.update_xaxes(range=[0.0, 1.08])
    return fig


def component_contribution_figure(
    snapshot: pd.DataFrame,
    *,
    top_n: int = 8,
) -> go.Figure:
    ordered = snapshot.nsmallest(top_n, "rank").sort_values("rank", ascending=False)
    components = (
        ("Technical composite", "composite_contribution"),
        ("Elastic Net", "elastic_net_contribution"),
        ("LightGBM ranker", "lightgbm_ranker_contribution"),
    )

    fig = go.Figure()
    for label, column in components:
        fig.add_trace(
            go.Bar(
                x=ordered[column],
                y=ordered["ticker"],
                orientation="h",
                name=label,
                marker={"color": _COMPONENT_COLORS[label]},
                hovertemplate=f"{label}: %{{x:.3f}}<extra></extra>",
            )
        )

    fig = _style_figure(
        fig,
        yaxis_title=None,
        y_tickformat=None,
        height=360,
        show_legend=True,
    )
    fig.update_layout(barmode="stack", hovermode="y unified")
    fig.update_xaxes(title="Weighted component contribution")
    return fig


_PORTFOLIO_PREVIOUS = "#AAB3BC"
_PORTFOLIO_POSITIVE = "#2F6B5F"
_PORTFOLIO_NEGATIVE = "#9A5A55"
_METHOD_OTHER = "#AAB3BC"


def portfolio_weight_change_figure(
    snapshot: pd.DataFrame,
    *,
    top_n: int = 15,
) -> go.Figure:
    ordered = snapshot.copy()
    ordered["weight_delta"] = ordered["weight"].astype(float) - ordered["previous_weight"].astype(
        float
    )
    ordered["absolute_delta"] = ordered["weight_delta"].abs()
    ordered = ordered.nlargest(top_n, "absolute_delta").sort_values("weight_delta")
    colors = [
        _PORTFOLIO_POSITIVE if value >= 0.0 else _PORTFOLIO_NEGATIVE
        for value in ordered["weight_delta"]
    ]
    labels = ordered["weight_delta"].map(lambda value: f"{value:+.2%}")

    fig = go.Figure(
        go.Bar(
            x=ordered["weight_delta"],
            y=ordered["ticker"],
            orientation="h",
            marker={"color": colors},
            text=labels,
            textposition="outside",
            cliponaxis=False,
            customdata=ordered[["weight", "previous_weight", "sector"]],
            hovertemplate=(
                "%{customdata[2]}<br>"
                "Current %{customdata[0]:.2%}<br>"
                "Previous %{customdata[1]:.2%}<br>"
                "Change %{x:+.2%}<extra></extra>"
            ),
        )
    )
    max_abs = float(ordered["weight_delta"].abs().max()) if not ordered.empty else 0.01
    bound = max(max_abs * 1.30, 0.01)
    fig = _style_figure(
        fig,
        yaxis_title=None,
        y_tickformat=None,
        height=420,
        show_legend=False,
        x_tickformat="+.0%",
    )
    fig.update_xaxes(range=[-bound, bound], zeroline=True, zerolinecolor=_GRID)
    return fig


def portfolio_sector_comparison_figure(exposure: pd.DataFrame) -> go.Figure:
    ordered = exposure.sort_values("current_weight", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ordered["previous_weight"],
            y=ordered["sector"],
            orientation="h",
            name="Previous",
            marker={"color": _PORTFOLIO_PREVIOUS},
            hovertemplate="Previous %{x:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=ordered["current_weight"],
            y=ordered["sector"],
            orientation="h",
            name="Current",
            marker={"color": _SELECTED},
            text=ordered["current_weight"].map(lambda value: f"{value:.1%}"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Current %{x:.1%}<extra></extra>",
        )
    )
    maximum = float(ordered[["current_weight", "previous_weight"]].max().max())
    fig = _style_figure(
        fig,
        yaxis_title=None,
        y_tickformat=None,
        height=420,
        show_legend=True,
        x_tickformat=".0%",
    )
    fig.update_layout(barmode="group", hovermode="y unified")
    fig.update_xaxes(range=[0.0, maximum * 1.20 if maximum > 0.0 else 1.0])
    return fig


def turnover_history_figure(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for (method, role), group in history.groupby(["method", "role"], sort=False):
        color = _SELECTED if role == "selected" else _BASELINE
        width = 2.5 if role == "selected" else 1.8
        dash = "solid" if role == "selected" else "dot"
        ordered = group.sort_values("as_of_date")
        fig.add_trace(
            go.Scatter(
                x=ordered["as_of_date"],
                y=ordered["one_way_turnover"],
                mode="lines",
                name=strategy_label(str(method)),
                line={"color": color, "width": width, "dash": dash},
                hovertemplate="%{y:.1%}<extra></extra>",
            )
        )
    return _style_figure(
        fig,
        yaxis_title="One-way turnover",
        y_tickformat=".0%",
        height=340,
    )


def realized_drift_figure(positions: pd.DataFrame, *, top_n: int = 12) -> go.Figure:
    ordered = positions.copy()
    ordered["absolute_drift"] = ordered["weight_drift"].astype(float).abs()
    ordered = ordered.nlargest(top_n, "absolute_drift").sort_values("weight_drift")
    colors = [
        _PORTFOLIO_POSITIVE if value >= 0.0 else _PORTFOLIO_NEGATIVE
        for value in ordered["weight_drift"]
    ]
    fig = go.Figure(
        go.Bar(
            x=ordered["weight_drift"],
            y=ordered["ticker"],
            orientation="h",
            marker={"color": colors},
            text=ordered["weight_drift"].map(lambda value: f"{value:+.2%}"),
            textposition="outside",
            cliponaxis=False,
            customdata=ordered[["actual_weight", "target_weight"]],
            hovertemplate=(
                "Actual %{customdata[0]:.2%}<br>"
                "Target %{customdata[1]:.2%}<br>"
                "Drift %{x:+.2%}<extra></extra>"
            ),
        )
    )
    max_abs = float(ordered["weight_drift"].abs().max()) if not ordered.empty else 0.01
    bound = max(max_abs * 1.35, 0.005)
    fig = _style_figure(
        fig,
        yaxis_title=None,
        y_tickformat=None,
        height=340,
        show_legend=False,
        x_tickformat="+.1%",
    )
    fig.update_xaxes(range=[-bound, bound], zeroline=True, zerolinecolor=_GRID)
    return fig


def portfolio_method_comparison_figure(comparison: pd.DataFrame) -> go.Figure:
    short_labels = {
        "score_weighted": "Score Weighted",
        "top_n_equal_weight": "Top-N",
        "median_mad_de": "Median-MAD",
        "alpha_risk_turnover": "Alpha-Risk",
        "cvar": "CVaR",
    }

    fig = go.Figure()
    for role, group in comparison.groupby("role", sort=False):
        selected = role == "selected"
        labels = group["method"].map(
            lambda method: short_labels.get(str(method), strategy_label(str(method)))
        )
        full_labels = group["method"].map(lambda method: strategy_label(str(method)))
        customdata = pd.DataFrame(
            {
                "label": full_labels,
                "effective_positions": group["effective_positions"],
                "maximum_sector_weight": group["maximum_sector_weight"],
            }
        )

        fig.add_trace(
            go.Scatter(
                x=group["one_way_turnover"],
                y=group["predicted_volatility"],
                mode="markers+text",
                text=labels,
                textposition="top center",
                name="Selected" if selected else "Other methods",
                marker={
                    "color": _SELECTED if selected else _METHOD_OTHER,
                    "size": 16 if selected else 12,
                    "line": {"width": 1, "color": _BACKGROUND},
                },
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "Turnover %{x:.1%}<br>"
                    "Predicted vol %{y:.1%}<br>"
                    "Effective positions %{customdata[1]:.1f}<br>"
                    "Max sector %{customdata[2]:.1%}<extra></extra>"
                ),
            )
        )
    fig = _style_figure(
        fig,
        yaxis_title="Predicted volatility",
        y_tickformat=".1%",
        height=360,
        show_legend=False,
        x_tickformat=".0%",
    )
    fig.update_xaxes(title="One-way turnover")
    return fig
