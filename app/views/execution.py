from __future__ import annotations

import pandas as pd
import streamlit as st
from components.charts import (
    capacity_metric_figure,
    execution_cost_breakdown_figure,
    execution_method_comparison_figure,
    rebalance_cost_history_figure,
    transaction_cost_sensitivity_figure,
)
from components.formatting import (
    format_bps,
    format_currency,
    format_percent,
)
from components.shell import render_page_header
from data_access import load_dashboard_source

from quant_equity.reporting.dashboard_catalog import DEFAULT_STRATEGY, strategy_label
from quant_equity.reporting.dashboard_metrics import (
    capacity_curve,
    execution_cost_breakdown,
    execution_cost_row,
    execution_dates,
    execution_method_comparison,
    execution_summary_row,
    execution_trade_snapshot,
    rebalance_execution_history,
    transaction_cost_sensitivity_curve,
)

strategy = st.session_state.get("dashboard_strategy", DEFAULT_STRATEGY)

trades = load_dashboard_source("trades")
execution_summary = load_dashboard_source("execution_summary")
cost_components = load_dashboard_source("execution_cost_components")
cost_sensitivity = load_dashboard_source("cost_sensitivity")
capacity = load_dashboard_source("capacity")

summary = execution_summary_row(execution_summary, strategy)
cost_row = execution_cost_row(cost_components, strategy)
breakdown = execution_cost_breakdown(cost_components, strategy)
history = rebalance_execution_history(trades, strategy)
capacity_data = capacity_curve(capacity, strategy)
sensitivity = transaction_cost_sensitivity_curve(cost_sensitivity, strategy)
method_comparison = execution_method_comparison(
    execution_summary,
    cost_components,
    strategy,
)

available_execution_dates = execution_dates(trades, strategy)
if not available_execution_dates:
    raise ValueError(f"No execution dates found for {strategy!r}.")

render_page_header(
    "Execution & Capacity",
    "Trading activity, transaction costs, implementation sensitivity, and scalability.",
    eyebrow="Institutional Quant Equity Research Platform",
)

st.caption(
    "All execution figures are modeled research outputs. "
    "They are not broker fills or live market execution records."
)

control_left, control_right = st.columns([1.0, 2.2], gap="medium")
with control_left:
    selected_execution_date = st.selectbox(
        "Execution date",
        options=list(reversed(available_execution_dates)),
        index=0,
        format_func=lambda value: pd.Timestamp(value).date().isoformat(),
    )
with control_right:
    st.caption("Selected construction method")
    st.markdown(f"**{strategy_label(strategy)}**")

selected_execution_date = pd.Timestamp(selected_execution_date)
trade_snapshot = execution_trade_snapshot(
    trades,
    strategy,
    selected_execution_date,
)

latest_cost = float(trade_snapshot["total_execution_cost"].sum())
latest_notional = float(trade_snapshot["trade_notional"].abs().sum())
latest_bps = latest_cost / latest_notional * 10_000.0 if latest_notional > 0.0 else float("nan")

top_kpis = st.columns(3, gap="small")
top_kpis[0].metric(
    "Total modeled cost",
    format_currency(float(summary["total_transaction_cost"])),
    border=True,
)
top_kpis[1].metric(
    "Effective cost",
    format_bps(float(cost_row["effective_cost_bps"])),
    border=True,
)
top_kpis[2].metric(
    "Mean turnover",
    format_percent(float(summary["mean_one_way_turnover"])),
    border=True,
)

bottom_kpis = st.columns(3, gap="small")
bottom_kpis[0].metric(
    "Rebalances",
    f"{int(summary['rebalances'])}",
    border=True,
)
bottom_kpis[1].metric(
    "Latest rebalance cost",
    format_currency(latest_cost),
    border=True,
)
bottom_kpis[2].metric(
    "Latest effective cost",
    format_bps(latest_bps),
    border=True,
)

cost_column, history_column = st.columns([1.0, 1.4], gap="medium")

with cost_column:
    with st.container(border=True):
        st.subheader("Cost decomposition")
        st.caption("Aggregate modeled implementation cost by execution component.")
        st.plotly_chart(
            execution_cost_breakdown_figure(breakdown),
            width="stretch",
            config={"displayModeBar": False},
        )

with history_column:
    with st.container(border=True):
        st.subheader("Rebalance execution-cost history")
        st.caption("Effective cost normalized by gross traded notional at each rebalance.")
        st.plotly_chart(
            rebalance_cost_history_figure(history),
            width="stretch",
            config={"displayModeBar": False},
        )

with st.container(border=True):
    st.subheader("Trade blotter")
    st.caption(
        f"Largest modeled trades by execution cost on {selected_execution_date.date().isoformat()}."
    )

    blotter = trade_snapshot.loc[
        :,
        [
            "ticker",
            "side",
            "trade_notional",
            "total_execution_cost",
            "effective_cost_bps",
            "order_adv_fraction",
        ],
    ].head(30)
    blotter = blotter.rename(
        columns={
            "ticker": "Ticker",
            "side": "Side",
            "trade_notional": "Notional",
            "total_execution_cost": "Execution cost",
            "effective_cost_bps": "Cost",
            "order_adv_fraction": "Order / ADV",
        }
    )
    blotter["Notional"] = blotter["Notional"].map(lambda value: format_currency(abs(float(value))))
    blotter["Execution cost"] = blotter["Execution cost"].map(
        lambda value: format_currency(float(value))
    )
    blotter["Cost"] = blotter["Cost"].map(lambda value: format_bps(float(value)))
    blotter["Order / ADV"] = blotter["Order / ADV"].map(
        lambda value: format_percent(float(value), digits=2)
    )
    st.dataframe(
        blotter,
        width="stretch",
        hide_index=True,
        height=430,
    )

with st.container(border=True):
    st.subheader("Capacity analysis")
    st.caption("Sensitivity to portfolio capital under the modeled market-impact framework.")

    capacity_left, capacity_right = st.columns(2, gap="medium")
    with capacity_left:
        st.plotly_chart(
            capacity_metric_figure(
                capacity_data,
                metric="effective_cost_bps",
                title="Effective cost (bps)",
                tickformat=".2f",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    with capacity_right:
        st.plotly_chart(
            capacity_metric_figure(
                capacity_data,
                metric="net_cagr",
                title="Net CAGR",
                tickformat=".1%",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

    capacity_table = capacity_data.loc[
        :,
        [
            "capital",
            "net_cagr",
            "net_sharpe_ratio",
            "effective_cost_bps",
            "maximum_order_adv_fraction",
        ],
    ].rename(
        columns={
            "capital": "Capital",
            "net_cagr": "Net CAGR",
            "net_sharpe_ratio": "Sharpe",
            "effective_cost_bps": "Cost",
            "maximum_order_adv_fraction": "Max order / ADV",
        }
    )
    capacity_table["Capital"] = capacity_table["Capital"].map(
        lambda value: format_currency(float(value))
    )
    capacity_table["Net CAGR"] = capacity_table["Net CAGR"].map(lambda value: f"{float(value):.1%}")
    capacity_table["Sharpe"] = capacity_table["Sharpe"].map(lambda value: f"{float(value):.2f}")
    capacity_table["Cost"] = capacity_table["Cost"].map(lambda value: format_bps(float(value)))
    capacity_table["Max order / ADV"] = capacity_table["Max order / ADV"].map(
        lambda value: f"{float(value):.2%}"
    )
    st.dataframe(
        capacity_table,
        width="stretch",
        hide_index=True,
        height=220,
    )

sensitivity_column, comparison_column = st.columns([1.15, 1.0], gap="medium")

with sensitivity_column:
    with st.container(border=True):
        st.subheader("Transaction-cost sensitivity")
        st.caption("Portfolio performance under alternative modeled cost assumptions.")
        st.plotly_chart(
            transaction_cost_sensitivity_figure(sensitivity),
            width="stretch",
            config={"displayModeBar": False},
        )

with comparison_column:
    with st.container(border=True):
        st.subheader("Method execution comparison")
        st.caption("Mean turnover versus aggregate effective execution cost.")
        st.plotly_chart(
            execution_method_comparison_figure(method_comparison),
            width="stretch",
            config={"displayModeBar": False},
        )

st.caption(
    "Capacity and transaction-cost sensitivity are research stress tests. "
    "They do not guarantee executable capacity at future market conditions."
)
