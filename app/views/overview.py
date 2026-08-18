from __future__ import annotations

import pandas as pd
import streamlit as st
from components.charts import (
    drawdown_figure,
    performance_figure,
    sector_exposure_figure,
)
from components.formatting import (
    format_bps,
    format_currency,
    format_percent,
    format_ratio,
)
from components.shell import render_page_header
from data_access import dashboard_context, load_dashboard_source

from quant_equity.reporting.dashboard_catalog import DEFAULT_STRATEGY, strategy_label
from quant_equity.reporting.dashboard_metrics import (
    build_drawdown_series,
    build_performance_index,
    latest_strategy_row,
    portfolio_dates,
    portfolio_snapshot,
    sector_exposure,
    strategy_summary,
)

strategy = st.session_state.get("dashboard_strategy", DEFAULT_STRATEGY)
context = dashboard_context()

summary_table = load_dashboard_source("performance_summary")
net_daily = load_dashboard_source("performance_net_daily")
benchmark = load_dashboard_source("benchmark_spy")
target_weights = load_dashboard_source("target_weights")
portfolio_risk = load_dashboard_source("portfolio_risk")
execution_summary = load_dashboard_source("execution_summary")
execution_costs = load_dashboard_source("execution_cost_components")
signal_bootstrap = load_dashboard_source("signal_bootstrap")
robustness_inventory = load_dashboard_source("robustness_inventory")
robustness_coverage = load_dashboard_source("robustness_coverage")

summary = strategy_summary(summary_table, strategy)
performance_index = build_performance_index(net_daily, benchmark, strategy)
drawdowns = build_drawdown_series(performance_index)
available_portfolio_dates = portfolio_dates(target_weights, strategy)

if not available_portfolio_dates:
    raise ValueError(f"No portfolio dates found for {strategy!r}.")

latest_portfolio_date = available_portfolio_dates[-1]

portfolio = portfolio_snapshot(
    target_weights,
    strategy,
    latest_portfolio_date,
)

portfolio = portfolio.loc[
    portfolio["weight"].astype(float) > 1e-10
].copy()

sectors = sector_exposure(portfolio)
risk = latest_strategy_row(
    portfolio_risk,
    strategy,
    strategy_column="method",
    date_column="as_of_date",
)
execution = latest_strategy_row(
    execution_summary,
    strategy,
    strategy_column="strategy_name",
)
costs = latest_strategy_row(
    execution_costs,
    strategy,
    strategy_column="strategy_name",
)
bootstrap = signal_bootstrap.iloc[0]
portfolio_bootstrap_table = load_dashboard_source("bootstrap_strategy")
portfolio_bootstrap = latest_strategy_row(
    portfolio_bootstrap_table,
    strategy,
    strategy_column="strategy_name",
)


def _inject_overview_css() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            min-width: 270px;
            max-width: 270px;
        }
        .block-container {
            padding-top: 3.6rem;
            padding-left: 2.0rem;
            padding-right: 2.0rem;
            max-width: 1360px;
        }
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #D7DDE4;
            border-radius: 10px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem;
            color: #66717D;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.75rem;
            line-height: 1.1;
        }
        .iq-mini-label {
            color: #66717D;
            font-size: 0.82rem;
            margin-bottom: 0.15rem;
        }
        .iq-mini-value {
            color: #17212B;
            font-size: 1.40rem;
            font-weight: 600;
            margin-bottom: 0.90rem;
        }
        .iq-stat-caption {
            color: #66717D;
            font-size: 0.83rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_row(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items), gap="small")
    for column, (label, value) in zip(columns, items, strict=True):
        column.metric(label, value, border=True)


def _render_stat_grid(items: list[tuple[str, str]], *, columns_count: int = 2) -> None:
    for start in range(0, len(items), columns_count):
        row = items[start : start + columns_count]
        columns = st.columns(columns_count, gap="small")
        for column, (label, value) in zip(columns, row, strict=False):
            column.markdown(
                f"<div class='iq-mini-label'>{label}</div><div class='iq-mini-value'>{value}</div>",
                unsafe_allow_html=True,
            )


_inject_overview_css()

render_page_header(
    "Executive Overview",
    "Portfolio performance, current exposures, risk, implementation, and research evidence.",
    eyebrow="Institutional Quant Equity Research Platform",
)

horizon = context["horizon_sessions"]
horizon_text = f"{horizon}-session horizon" if horizon is not None else "Model horizon"
st.caption(
    f"{strategy_label(strategy)}  |  OOS signal period "
    f"{context['oos_start_date'].date().isoformat()} → "
    f"{context['last_signal_date'].date().isoformat()}  |  "
    f"Evaluated through {context['last_evaluated_date'].date().isoformat()}  |  "
    f"{context['universe_size']} securities  |  {horizon_text}  |  "
    "Monthly rebalance  |  Net of modeled transaction costs"
)

_render_kpi_row(
    [
        ("Net CAGR", format_percent(summary.net_cagr)),
        ("Sharpe", format_ratio(summary.net_sharpe_ratio)),
        ("Max drawdown", format_percent(summary.net_maximum_drawdown)),
    ]
)
_render_kpi_row(
    [
        ("Alpha vs SPY", format_percent(summary.net_annualized_alpha_vs_spy, signed=True)),
        ("Beta vs SPY", format_ratio(summary.net_beta_vs_spy)),
        ("Mean turnover", format_percent(summary.mean_one_way_turnover)),
    ]
)

performance_column, portfolio_column = st.columns([1.75, 1.10], gap="medium")

with performance_column:
    with st.container(border=True):
        st.subheader("Cumulative performance")
        st.caption(
            "Growth of 100 from the first common OOS trading day. "
            "The dashed line is the simple Top-N equal-weight baseline when applicable."
        )
        st.plotly_chart(
            performance_figure(performance_index),
            width="stretch",
            config={"displayModeBar": False},
        )

with portfolio_column:
    with st.container(border=True):
        st.subheader("Current portfolio")
        snapshot_date = pd.Timestamp(portfolio["as_of_date"].iloc[0]).date().isoformat()
        st.caption(f"Target weights as of {snapshot_date}")

        _render_stat_grid(
            [
                ("Positions", f"{len(portfolio)}"),
                ("Largest weight", format_percent(float(portfolio["weight"].max()))),
                ("Largest sector", format_percent(float(sectors["sector_weight"].max()))),
                (
                    "Weight in top 5",
                    format_percent(float(portfolio["weight"].head(5).sum())),
                ),
            ]
        )

        holdings = portfolio.loc[
            :,
            ["ticker", "sector", "weight", "weight_delta"],
        ].head(6)

        holdings = holdings.assign(
            Weight=holdings["weight"].astype(float).map(lambda value: f"{value:.2%}"),
            Delta=holdings["weight_delta"]
            .astype(float)
            .map(lambda value: f"{value:+.2%}"),
        ).rename(columns={"ticker": "Ticker", "sector": "Sector"})
        st.dataframe(
            holdings[["Ticker", "Sector", "Weight", "Delta"]],
            width="stretch",
            hide_index=True,
            height=250,
        )

risk_column, exposure_column = st.columns([1.05, 1.35], gap="medium")

with risk_column:
    with st.container(border=True):
        st.subheader("Risk & implementation")
        _render_stat_grid(
            [
                ("Predicted volatility", format_percent(float(risk["predicted_volatility"]))),
                ("Portfolio beta", format_ratio(float(risk["portfolio_beta_vs_spy"]))),
                (
                    "Effective positions",
                    format_ratio(float(risk["effective_positions"]), digits=1),
                ),
                (
                    "Max liquidation",
                    f"{float(risk['maximum_liquidation_days']):.2f} days",
                ),
            ]
        )
        st.divider()
        _render_stat_grid(
            [
                (
                    "Transaction costs",
                    format_currency(float(execution["total_transaction_cost"])),
                ),
                ("Effective cost", format_bps(float(costs["effective_cost_bps"]))),
                ("Rebalances", f"{int(execution['rebalances'])}"),
                ("Mean holdings", f"{float(execution['mean_holdings']):.1f}"),
            ]
        )

with exposure_column:
    with st.container(border=True):
        st.subheader("Sector exposure")
        st.caption("Current target-weight allocation across sectors.")
        st.plotly_chart(
            sector_exposure_figure(sectors),
            width="stretch",
            config={"displayModeBar": False},
        )

with st.container(border=True):
    st.subheader("Drawdown profile")
    st.caption("Selected strategy versus SPY over the common evaluated period.")
    st.plotly_chart(
        drawdown_figure(drawdowns),
        width="stretch",
        config={"displayModeBar": False},
    )

with st.container(border=True):
    st.subheader("Research & robustness evidence")
    st.caption(
        "Signal evidence is common to all construction methods. "
        "Portfolio bootstrap evidence changes with the selected method."
    )

    signal_column, portfolio_evidence_column = st.columns(2, gap="medium")

    with signal_column:
        st.markdown("#### Frozen alpha signal")
        st.caption("Predictive evidence for the shared cross-sectional ranking signal.")
        _render_stat_grid(
            [
                (
                    "Mean signal IC",
                    format_ratio(float(bootstrap["observed_mean_ic"]), digits=3),
                ),
                (
                    "IC 95% CI",
                    (
                        f"[{float(bootstrap['mean_ic_ci_lower']):.3f}, "
                        f"{float(bootstrap['mean_ic_ci_upper']):.3f}]"
                    ),
                ),
                (
                    "P(IC > 0)",
                    format_percent(float(bootstrap["probability_mean_ic_positive"])),
                ),
                (
                    "Top-bottom spread",
                    format_percent(float(bootstrap["observed_mean_top_bottom_spread"])),
                ),
            ]
        )

    with portfolio_evidence_column:
        st.markdown(f"#### {strategy_label(strategy)} portfolio")
        st.caption("Monthly-return bootstrap evidence for the selected construction method.")
        _render_stat_grid(
            [
                (
                    "Observed annualized return",
                    format_percent(float(portfolio_bootstrap["observed_annualized_return"])),
                ),
                (
                    "Return 95% CI",
                    (
                        "["
                        + format_percent(float(portfolio_bootstrap["annualized_return_ci_lower"]))
                        + ", "
                        + format_percent(float(portfolio_bootstrap["annualized_return_ci_upper"]))
                        + "]"
                    ),
                ),
                (
                    "Observed Sharpe",
                    format_ratio(float(portfolio_bootstrap["observed_sharpe"])),
                ),
                (
                    "Sharpe 95% CI",
                    (
                        f"[{format_ratio(float(portfolio_bootstrap['sharpe_ci_lower']))}, "
                        f"{format_ratio(float(portfolio_bootstrap['sharpe_ci_upper']))}]"
                    ),
                ),
            ]
        )

    st.divider()
    passed = int((robustness_inventory["suite_status"].astype(str) == "PASS").sum())
    audit_columns = st.columns(3, gap="small")
    audit_columns[0].metric(
        "Robustness audit",
        f"{passed}/{len(robustness_inventory)} PASS",
    )
    audit_columns[1].metric(
        "Deferred dimensions",
        str(len(context["deferred_dimensions"])),
    )
    audit_columns[2].metric(
        "Signal months",
        str(context["signal_dates"]),
    )

    deferred = robustness_coverage.loc[
        robustness_coverage["status"].astype(str) == "DEFERRED_LIMITATION",
        "dimension",
    ].astype(str)
    if not deferred.empty:
        st.caption(
            "Documented limitation: "
            + ", ".join(deferred.tolist())
            + ". This is disclosed rather than treated as a failed validation."
        )
