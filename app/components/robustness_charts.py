from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from quant_equity.reporting.dashboard_robustness import (
    humanize_identifier,
    strategy_chart_label,
)

NAVY = "#17324D"
SLATE = "#6B7C8F"
STEEL = "#9AA8B5"
GOLD = "#A98645"
RED = "#A05A55"
GRID = "#E5E9ED"
TEXT = "#27323C"


def _style(
    figure: go.Figure,
    *,
    height: int = 360,
    x_title: str = "",
    y_title: str = "",
) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=55, r=35, t=30, b=55),
        font=dict(color=TEXT, size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(
            title=x_title,
            gridcolor=GRID,
            zerolinecolor=GRID,
            showline=False,
        ),
        yaxis=dict(
            title=y_title,
            gridcolor=GRID,
            zerolinecolor=GRID,
            showline=False,
        ),
    )
    return figure


def signal_bootstrap_figure(data: pd.DataFrame) -> go.Figure:
    row = data.iloc[0]
    labels = ["Mean IC", "Top-bottom spread"]
    observed = [
        float(row["observed_mean_ic"]),
        float(row["observed_mean_top_bottom_spread"]),
    ]
    lower = [
        float(row["mean_ic_ci_lower"]),
        float(row["top_bottom_spread_ci_lower"]),
    ]
    upper = [
        float(row["mean_ic_ci_upper"]),
        float(row["top_bottom_spread_ci_upper"]),
    ]

    figure = go.Figure(
        go.Scatter(
            x=observed,
            y=labels,
            mode="markers",
            marker=dict(size=11, color=NAVY),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[u - o for u, o in zip(upper, observed, strict=True)],
                arrayminus=[
                    o - lower_value
                    for o, lower_value in zip(observed, lower, strict=True)
                ],
                color=SLATE,
                thickness=1.5,
            ),
            name="Observed with 95% CI",
        )
    )
    figure.add_vline(x=0, line_width=1, line_color=STEEL)
    return _style(
        figure,
        height=290,
        x_title="Cross-sectional research statistic",
    )


def strategy_return_bootstrap_figure(data: pd.DataFrame) -> go.Figure:
    ordered = data.sort_values("observed_annualized_return", ascending=True).copy()
    labels = ordered["strategy_name"].map(strategy_chart_label)
    observed = ordered["observed_annualized_return"].astype(float)
    lower = ordered["annualized_return_ci_lower"].astype(float)
    upper = ordered["annualized_return_ci_upper"].astype(float)

    figure = go.Figure(
        go.Scatter(
            x=observed,
            y=labels,
            mode="markers",
            marker=dict(size=10, color=NAVY),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(upper - observed),
                arrayminus=(observed - lower),
                color=SLATE,
                thickness=1.3,
            ),
            name="Observed with 95% CI",
        )
    )
    figure.update_xaxes(tickformat=".0%")
    return _style(figure, height=340, x_title="Annualized return")


def horizon_sensitivity_figure(data: pd.DataFrame) -> go.Figure:
    ordered = data.sort_values("horizon_sessions")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=ordered["horizon_sessions"].astype(str),
            y=ordered["mean_ic"],
            name="Mean IC",
            marker_color=NAVY,
            text=ordered["mean_ic"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ordered["horizon_sessions"].astype(str),
            y=ordered["mean_top_bottom_spread"],
            name="T-B spread",
            mode="lines+markers",
            line=dict(color=GOLD, width=2),
            marker=dict(size=8),
            yaxis="y2",
        )
    )
    _style(
        figure,
        height=340,
        x_title="Forward horizon (sessions)",
        y_title="Mean IC",
    )
    figure.update_layout(
        yaxis2=dict(
            title="Top-bottom spread",
            overlaying="y",
            side="right",
            tickformat=".1%",
            showgrid=False,
        )
    )
    return figure


def rebalance_sensitivity_figure(data: pd.DataFrame) -> go.Figure:
    ordered = data.sort_values("quarterly_minus_monthly_cagr")
    labels = ordered["strategy_name"].map(strategy_chart_label)
    values = ordered["quarterly_minus_monthly_cagr"].astype(float)

    colors = [NAVY if value >= 0 else RED for value in values]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.1%}" for value in values],
            textposition="outside",
            name="Quarterly minus monthly CAGR",
        )
    )
    figure.add_vline(x=0, line_width=1, line_color=STEEL)
    figure.update_xaxes(tickformat=".1%")
    _style(figure, height=340, x_title="Quarterly minus monthly CAGR")
    figure.update_layout(margin=dict(l=95, r=35, t=30, b=55))
    return figure


def rolling_window_figure(data: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    palette = [NAVY, GOLD, SLATE, STEEL, RED]
    for index, (strategy, group) in enumerate(data.groupby("strategy_name", sort=False)):
        ordered = group.sort_values("window_months")
        figure.add_trace(
            go.Scatter(
                x=ordered["window_months"],
                y=ordered["mean_sharpe"],
                mode="lines+markers",
                name=strategy_chart_label(strategy),
                line=dict(color=palette[index % len(palette)], width=2),
                marker=dict(size=7),
            )
        )
    return _style(
        figure,
        height=360,
        x_title="Evaluation window (months)",
        y_title="Mean rolling Sharpe",
    )


def feature_ablation_figure(data: pd.DataFrame) -> go.Figure:
    labels = data["scenario"].map(humanize_identifier)

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=labels,
            y=data["mean_ic"],
            name="Mean IC",
            marker_color=NAVY,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=data["mean_top_bottom_spread"],
            mode="lines+markers",
            name="T-B spread",
            line=dict(color=GOLD, width=2),
            marker=dict(size=8),
            yaxis="y2",
        )
    )
    _style(figure, height=330, y_title="Mean IC")
    figure.update_layout(
        yaxis2=dict(
            title="Top-bottom spread",
            overlaying="y",
            side="right",
            tickformat=".1%",
            showgrid=False,
        )
    )
    return figure


def universe_exclusion_figure(data: pd.DataFrame) -> go.Figure:
    filtered = data.loc[~data["is_baseline"].astype(bool)].copy()
    filtered = filtered.sort_values("cagr_difference_vs_full")
    labels = filtered["excluded_group"].fillna(filtered["strategy_name"]).astype(str)
    values = filtered["cagr_difference_vs_full"].astype(float)

    colors = [NAVY if value >= 0 else RED for value in values]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.1%}" for value in values],
            textposition="outside",
            name="CAGR difference vs full",
        )
    )
    figure.add_vline(x=0, line_width=1, line_color=STEEL)
    figure.update_xaxes(tickformat=".1%")
    _style(
        figure,
        height=470,
        x_title="CAGR difference vs full frozen universe",
    )
    figure.update_layout(margin=dict(l=185, r=45, t=30, b=55))
    return figure


def regime_heatmap_figure(data: pd.DataFrame) -> go.Figure:
    pivot = data.pivot(
        index="strategy_name",
        columns="regime",
        values="sharpe_ratio",
    )
    pivot.index = [strategy_chart_label(value) for value in pivot.index]

    warning_by_regime = (
        data.groupby("regime")["short_sample_warning"]
        .apply(lambda values: values.astype(bool).any())
        .to_dict()
    )
    display_columns = []
    for regime in pivot.columns:
        label = humanize_identifier(regime)
        if warning_by_regime.get(regime, False):
            label = f"{label} *"
        display_columns.append(label)

    figure = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=display_columns,
            y=pivot.index.tolist(),
            text=[[f"{value:+.2f}" for value in row] for row in pivot.values],
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#B87A73"],
                [0.5, "#F4F5F6"],
                [1.0, "#355A78"],
            ],
            zmin=-3,
            zmax=3,
            zmid=0,
            colorbar=dict(title="Sharpe", thickness=12),
            hovertemplate="%{y}<br>%{x}<br>Sharpe %{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=90, r=35, t=25, b=100),
        font=dict(color=TEXT, size=11),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_xaxes(tickangle=-28)
    return figure
