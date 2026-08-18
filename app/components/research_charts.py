from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from quant_equity.reporting.dashboard_research import model_label

_NAVY = "#17324D"
_SLATE = "#4F6B84"
_GREEN = "#2F6B5F"
_GOLD = "#B08D57"
_RED = "#9A5A55"
_GREY = "#AAB3BC"
_GRID = "#E3E7EB"
_TEXT = "#17212B"
_MUTED = "#66717D"
_BACKGROUND = "#FFFFFF"

_MODEL_COLORS: dict[str, str] = {
    "technical_equal_weight_composite": _SLATE,
    "elastic_net": _GREEN,
    "lightgbm_ranker": _GOLD,
}

_MODEL_TEXT_POSITIONS: dict[str, str] = {
    "momentum_3m": "bottom center",
    "technical_equal_weight_composite": "middle left",
    "ridge": "top center",
    "elastic_net": "bottom center",
    "lightgbm_regressor": "bottom center",
    "lightgbm_ranker": "top center",
}

_SECTOR_LABELS: dict[str, str] = {
    "Communication Services": "Comm. Services",
    "Consumer Discretionary": "Consumer Disc.",
    "Consumer Staples": "Consumer Staples",
    "Energy": "Energy",
    "Financials": "Financials",
    "Health Care": "Health Care",
    "Industrials": "Industrials",
    "Information Technology": "Info. Tech.",
    "Materials": "Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}

_ENSEMBLE_CHART_LABELS: dict[str, str] = {
    "core_percentile_ensemble": "Core",
    "equal_percentile_ensemble": "Equal",
    "validation_weighted_ensemble": "Validation-weighted",
}


def model_comparison_figure(summary: pd.DataFrame) -> go.Figure:
    frame = summary.copy()
    frame = frame[pd.to_numeric(frame["mean_ic"], errors="coerce").notna()].copy()
    frame["mean_ic"] = pd.to_numeric(frame["mean_ic"], errors="coerce")
    frame["annualized_ic_ir"] = pd.to_numeric(frame["annualized_ic_ir"], errors="coerce")
    frame["model_label"] = frame["model_name"].astype(str).map(model_label)
    frame["color"] = frame["model_name"].astype(str).map(_MODEL_COLORS).fillna(_GREY)
    frame["text_position"] = (
        frame["model_name"].astype(str).map(_MODEL_TEXT_POSITIONS).fillna("top center")
    )

    fig = go.Figure(
        go.Scatter(
            x=frame["mean_ic"],
            y=frame["annualized_ic_ir"],
            mode="markers+text",
            text=frame["model_label"],
            textposition=frame["text_position"],
            cliponaxis=False,
            marker={
                "size": 12,
                "color": frame["color"],
                "line": {"width": 0.8, "color": _BACKGROUND},
            },
            customdata=frame[
                [
                    "positive_ic_ratio",
                    "mean_top_bottom_spread",
                    "mean_top_quintile_precision",
                ]
            ],
            hovertemplate=(
                "%{text}<br>Mean IC %{x:.3f}<br>IC IR %{y:.2f}<br>"
                "Positive IC months %{customdata[0]:.1%}<br>"
                "Top-bottom spread %{customdata[1]:.2%}<br>"
                "Top-quintile precision %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0.0, line_width=1, line_color=_GRID)
    fig.add_vline(x=0.0, line_width=1, line_color=_GRID)
    fig = _style_figure(fig, height=400, show_legend=False)
    fig.update_layout(margin={"l": 55, "r": 72, "t": 34, "b": 52})
    fig.update_xaxes(title="Mean monthly Spearman IC", tickformat=".3f")
    fig.update_yaxes(title="Annualized IC information ratio", tickformat=".2f")
    _apply_axis_padding(fig, frame["mean_ic"], frame["annualized_ic_ir"])
    return fig


def monthly_research_figure(
    monthly: pd.DataFrame,
    *,
    metric: str,
    yaxis_title: str,
    tickformat: str,
) -> go.Figure:
    fig = go.Figure()
    for model_name, group in monthly.groupby("model_name", sort=False):
        ordered = group.sort_values("as_of_date")
        fig.add_trace(
            go.Scatter(
                x=ordered["as_of_date"],
                y=ordered[metric],
                mode="lines",
                name=model_label(str(model_name)),
                line={
                    "color": _MODEL_COLORS.get(str(model_name), _GREY),
                    "width": 1.8,
                },
                hovertemplate=f"%{{y:{tickformat}}}<extra></extra>",
            )
        )
    fig.add_hline(y=0.0, line_width=1, line_color=_GRID)
    fig = _style_figure(fig, height=330, show_legend=True)
    fig.update_layout(margin={"l": 55, "r": 24, "t": 38, "b": 42})
    fig.update_yaxes(title=yaxis_title, tickformat=tickformat)
    return fig


def stability_heatmap_figure(
    matrix: pd.DataFrame,
    *,
    height: int,
) -> go.Figure:
    values = matrix.astype(float)
    finite = values.stack().dropna().abs()
    bound = max(float(finite.max()) if not finite.empty else 0.05, 0.05)
    text = values.map(lambda value: "" if pd.isna(value) else f"{float(value):+.3f}")

    full_columns = [str(column) for column in values.columns]
    display_columns = [_SECTOR_LABELS.get(column, column) for column in full_columns]
    customdata = [full_columns for _ in values.index]

    fig = go.Figure(
        go.Heatmap(
            z=values.to_numpy(),
            x=display_columns,
            y=[str(index) for index in values.index],
            zmin=-bound,
            zmax=bound,
            zmid=0.0,
            colorscale=[
                [0.0, _RED],
                [0.5, "#F3F5F7"],
                [1.0, _NAVY],
            ],
            text=text.to_numpy(),
            texttemplate="%{text}",
            textfont={"size": 10},
            customdata=customdata,
            hovertemplate=(
                "%{y}<br>%{customdata}<br>Mean IC %{z:+.3f}<extra></extra>"
            ),
            colorbar={"title": "IC", "thickness": 10, "len": 0.72},
        )
    )
    fig = _style_figure(fig, height=height, show_legend=False)
    sector_like = any(column in _SECTOR_LABELS for column in full_columns)
    fig.update_layout(
        margin={
            "l": 105,
            "r": 55,
            "t": 22,
            "b": 88 if sector_like else 58,
        }
    )
    fig.update_xaxes(
        side="bottom",
        tickangle=-25 if sector_like else 0,
        tickfont={"size": 10, "color": _MUTED},
    )
    fig.update_yaxes(showgrid=False, title=None, tickfont={"size": 10, "color": _MUTED})
    return fig


def feature_importance_figure(feature_importance: pd.DataFrame) -> go.Figure:
    ordered = feature_importance.sort_values("mean_gain_share", ascending=True)
    colors = ordered["family"].map(
        {"Technical": _NAVY, "Fundamental": _GOLD, "Other": _GREY}
    )
    fig = go.Figure(
        go.Bar(
            x=ordered["mean_gain_share"],
            y=ordered["feature_label"],
            orientation="h",
            marker={"color": colors},
            text=ordered["mean_gain_share"].map(lambda value: f"{float(value):.1%}"),
            textposition="outside",
            cliponaxis=False,
            customdata=ordered[["feature", "family", "folds_used"]],
            hovertemplate=(
                "%{customdata[0]}<br>%{customdata[1]}<br>"
                "Mean gain share %{x:.2%}<br>Folds used %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig = _style_figure(fig, height=520, show_legend=False)
    fig.update_layout(margin={"l": 30, "r": 48, "t": 20, "b": 48})
    fig.update_xaxes(title="Mean gain share", tickformat=".0%")
    fig.update_yaxes(showgrid=False, title=None, tickfont={"size": 10, "color": _MUTED})
    maximum = float(ordered["mean_gain_share"].max()) if not ordered.empty else 0.1
    fig.update_xaxes(range=[0.0, maximum * 1.24])
    return fig


def ensemble_candidate_figure(summary: pd.DataFrame) -> go.Figure:
    frame = summary.copy()
    frame["model_label"] = frame["model_name"].astype(str).map(model_label)
    frame["chart_label"] = (
        frame["model_name"].astype(str).map(_ENSEMBLE_CHART_LABELS).fillna(frame["model_label"])
    )
    text_positions = ["middle left", "top center", "bottom center"][: len(frame)]
    fig = go.Figure(
        go.Scatter(
            x=frame["mean_ic"],
            y=frame["annualized_ic_ir"],
            mode="markers+text",
            text=frame["chart_label"],
            textposition=text_positions,
            cliponaxis=False,
            marker={"size": 13, "color": [_NAVY, _SLATE, _GOLD][: len(frame)]},
            customdata=frame[
                [
                    "model_label",
                    "positive_ic_ratio",
                    "mean_top_bottom_spread",
                    "mean_top_quintile_turnover",
                ]
            ],
            hovertemplate=(
                "%{customdata[0]}<br>Mean IC %{x:.3f}<br>IC IR %{y:.2f}<br>"
                "Positive IC months %{customdata[1]:.1%}<br>"
                "Top-bottom spread %{customdata[2]:.2%}<br>"
                "Top-quintile turnover %{customdata[3]:.1%}<extra></extra>"
            ),
        )
    )
    fig = _style_figure(fig, height=340, show_legend=False)
    fig.update_layout(margin={"l": 55, "r": 58, "t": 32, "b": 48})
    fig.update_xaxes(title="Mean monthly IC", tickformat=".3f")
    fig.update_yaxes(title="Annualized IC IR", tickformat=".2f")
    _apply_axis_padding(fig, frame["mean_ic"], frame["annualized_ic_ir"], fraction=0.18)
    return fig


def ensemble_correlation_figure(matrix: pd.DataFrame) -> go.Figure:
    values = matrix.astype(float)
    text = values.map(lambda value: f"{float(value):.2f}")
    x_labels = [
        "Technical<br>Composite" if value == "Technical Composite" else value
        for value in values.columns
    ]
    x_labels = [
        "LightGBM<br>Ranker" if value == "LightGBM Ranker" else value for value in x_labels
    ]
    fig = go.Figure(
        go.Heatmap(
            z=values.to_numpy(),
            x=x_labels,
            y=list(values.index),
            zmin=0.0,
            zmax=1.0,
            colorscale=[[0.0, "#F3F5F7"], [1.0, _NAVY]],
            text=text.to_numpy(),
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>Mean Spearman %{z:.2f}<extra></extra>",
            showscale=False,
        )
    )
    fig = _style_figure(fig, height=340, show_legend=False)
    fig.update_layout(margin={"l": 82, "r": 24, "t": 20, "b": 62})
    fig.update_xaxes(tickangle=0, tickfont={"size": 10, "color": _MUTED})
    fig.update_yaxes(showgrid=False, title=None, tickfont={"size": 10, "color": _MUTED})
    return fig


def _apply_axis_padding(
    fig: go.Figure,
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    fraction: float = 0.14,
) -> None:
    x = pd.to_numeric(x_values, errors="coerce").dropna()
    y = pd.to_numeric(y_values, errors="coerce").dropna()
    if not x.empty:
        x_span = max(float(x.max() - x.min()), 0.01)
        fig.update_xaxes(
            range=[
                float(x.min()) - x_span * fraction,
                float(x.max()) + x_span * fraction,
            ]
        )
    if not y.empty:
        y_span = max(float(y.max() - y.min()), 0.10)
        fig.update_yaxes(
            range=[
                float(y.min()) - y_span * fraction,
                float(y.max()) + y_span * fraction,
            ]
        )


def _style_figure(
    fig: go.Figure,
    *,
    height: int,
    show_legend: bool,
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 20, "r": 28, "t": 22, "b": 20},
        paper_bgcolor=_BACKGROUND,
        plot_bgcolor=_BACKGROUND,
        font={"color": _TEXT, "size": 12},
        hovermode="closest",
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 11, "color": _TEXT},
        },
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont={"size": 11, "color": _MUTED},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=_GRID,
        gridwidth=1,
        zeroline=False,
        tickfont={"size": 11, "color": _MUTED},
    )
    return fig
