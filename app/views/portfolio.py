from __future__ import annotations

import pandas as pd
import streamlit as st
from components.charts import (
    portfolio_method_comparison_figure,
    portfolio_sector_comparison_figure,
    portfolio_weight_change_figure,
    realized_drift_figure,
    turnover_history_figure,
)
from components.formatting import format_currency, format_percent, format_ratio
from components.shell import render_page_header
from data_access import load_dashboard_source

from quant_equity.reporting.dashboard_catalog import DEFAULT_STRATEGY, strategy_label
from quant_equity.reporting.dashboard_metrics import (
    enrich_portfolio_snapshot,
    portfolio_dates,
    portfolio_diagnostics_row,
    portfolio_method_comparison,
    portfolio_risk_row,
    portfolio_sector_changes,
    portfolio_snapshot,
    realized_positions_for_signal,
    turnover_history,
)

strategy = st.session_state.get("dashboard_strategy", DEFAULT_STRATEGY)
target_weights = load_dashboard_source("target_weights")
positions_daily = load_dashboard_source("positions_daily")
portfolio_diagnostics = load_dashboard_source("portfolio_diagnostics")
portfolio_risk = load_dashboard_source("portfolio_risk")
security_risk = load_dashboard_source("security_risk")

available_dates = portfolio_dates(target_weights, strategy)
if not available_dates:
    raise ValueError(f"No portfolio dates found for {strategy!r}.")

render_page_header(
    "Portfolio Construction",
    "Target allocation, concentration, turnover, sector structure, and realized drift.",
    eyebrow="Institutional Quant Equity Research Platform",
)

st.caption(
    "Target weights are frozen monthly construction outputs. Realized drift is shown "
    "at the final evaluated day associated with the selected signal date."
)

control_left, control_right = st.columns([1.0, 2.2], gap="medium")
with control_left:
    selected_date = st.selectbox(
        "Portfolio date",
        options=list(reversed(available_dates)),
        index=0,
        format_func=lambda value: pd.Timestamp(value).date().isoformat(),
    )
with control_right:
    st.caption("Selected construction method")
    st.markdown(f"**{strategy_label(strategy)}**")

selected_date = pd.Timestamp(selected_date)
snapshot = portfolio_snapshot(target_weights, strategy, selected_date)
snapshot = enrich_portfolio_snapshot(snapshot, security_risk)
sector_changes = portfolio_sector_changes(snapshot)
diagnostics = portfolio_diagnostics_row(
    portfolio_diagnostics,
    strategy,
    selected_date,
)
risk = portfolio_risk_row(portfolio_risk, strategy, selected_date)
realized = realized_positions_for_signal(positions_daily, strategy, selected_date)
history = turnover_history(portfolio_diagnostics, strategy)
comparison = portfolio_method_comparison(
    portfolio_diagnostics,
    portfolio_risk,
    strategy,
    selected_date,
)

positive = snapshot.loc[snapshot["weight"].astype(float) > 1e-10].copy()
turnover = float(diagnostics["one_way_turnover"])

kpi_top = st.columns(3, gap="small")
kpi_top[0].metric("Positions", f"{int(diagnostics['positions'])}", border=True)
kpi_top[1].metric(
    "Effective positions",
    format_ratio(float(diagnostics["effective_positions"]), digits=1),
    border=True,
)
kpi_top[2].metric(
    "Maximum weight",
    format_percent(float(diagnostics["maximum_weight"])),
    border=True,
)

kpi_bottom = st.columns(3, gap="small")
kpi_bottom[0].metric(
    "Maximum sector",
    format_percent(float(diagnostics["maximum_sector_weight"])),
    border=True,
)
kpi_bottom[1].metric("Rebalance turnover", format_percent(turnover), border=True)
kpi_bottom[2].metric(
    "Predicted volatility",
    format_percent(float(risk["predicted_volatility"])),
    border=True,
)

changes_column, sector_column = st.columns([1.45, 1.10], gap="medium")

with changes_column:
    with st.container(border=True):
        st.subheader("Rebalance changes")
        st.caption(
            "Largest target-weight changes versus the previous monthly rebalance. "
            "Green increases exposure; red reduces it."
        )
        st.plotly_chart(
            portfolio_weight_change_figure(snapshot),
            width="stretch",
            config={"displayModeBar": False},
        )

with sector_column:
    with st.container(border=True):
        st.subheader("Sector allocation")
        st.caption("Current target weights versus the previous rebalance.")
        st.plotly_chart(
            portfolio_sector_comparison_figure(sector_changes),
            width="stretch",
            config={"displayModeBar": False},
        )

with st.container(border=True):
    st.subheader("Target holdings")
    st.caption(
        f"Portfolio target as of {selected_date.date().isoformat()}. "
        "Weight change is measured against the previous rebalance."
    )
    holdings = positive.loc[
        :,
        [
            "ticker",
            "sector",
            "weight",
            "previous_weight",
            "weight_delta",
            "beta_vs_spy",
            "average_dollar_volume",
        ],
    ].rename(
        columns={
            "ticker": "Ticker",
            "sector": "Sector",
            "weight": "Target weight",
            "previous_weight": "Previous weight",
            "weight_delta": "Change",
            "beta_vs_spy": "Beta",
            "average_dollar_volume": "ADV",
        }
    )
    holdings["Target weight"] = holdings["Target weight"].map(lambda value: f"{float(value):.2%}")
    holdings["Previous weight"] = holdings["Previous weight"].map(
        lambda value: f"{float(value):.2%}"
    )
    holdings["Change"] = holdings["Change"].map(lambda value: f"{float(value):+.2%}")
    holdings["Beta"] = holdings["Beta"].map(lambda value: f"{float(value):.2f}")
    holdings["ADV"] = holdings["ADV"].map(lambda value: format_currency(float(value)))
    st.dataframe(holdings, width="stretch", hide_index=True, height=520)

realized_column, turnover_column = st.columns([1.15, 1.0], gap="medium")

with realized_column:
    with st.container(border=True):
        realized_date = pd.Timestamp(realized["date"].iloc[0]).date().isoformat()
        st.subheader("Realized weight drift")
        st.caption(
            f"Actual versus target weights on {realized_date}, the final evaluated "
            "day linked to the selected signal."
        )
        st.plotly_chart(
            realized_drift_figure(realized),
            width="stretch",
            config={"displayModeBar": False},
        )

with turnover_column:
    with st.container(border=True):
        st.subheader("Turnover history")
        st.caption("Monthly one-way turnover for the selected method and the Top-N baseline.")
        st.plotly_chart(
            turnover_history_figure(history),
            width="stretch",
            config={"displayModeBar": False},
        )

with st.container(border=True):
    st.subheader("Construction-method comparison")
    st.caption("Selected-date implementation trade-off between turnover and predicted risk.")
    st.plotly_chart(
        portfolio_method_comparison_figure(comparison),
        width="stretch",
        config={"displayModeBar": False},
    )

    st.markdown("**Method diagnostics**")
    table = comparison.loc[
        :,
        [
            "method",
            "positions",
            "effective_positions",
            "maximum_sector_weight",
            "one_way_turnover",
            "predicted_volatility",
            "portfolio_beta_vs_spy",
        ],
    ].copy()
    table["method"] = table["method"].map(strategy_label)
    table = table.rename(
        columns={
            "method": "Method",
            "positions": "Positions",
            "effective_positions": "Effective",
            "maximum_sector_weight": "Max sector",
            "one_way_turnover": "Turnover",
            "predicted_volatility": "Pred. vol",
            "portfolio_beta_vs_spy": "Beta",
        }
    )
    table["Effective"] = table["Effective"].map(lambda value: f"{float(value):.1f}")
    table["Max sector"] = table["Max sector"].map(lambda value: f"{float(value):.1%}")
    table["Turnover"] = table["Turnover"].map(lambda value: f"{float(value):.1%}")
    table["Pred. vol"] = table["Pred. vol"].map(lambda value: f"{float(value):.1%}")
    table["Beta"] = table["Beta"].map(lambda value: f"{float(value):.2f}")
    st.dataframe(table, width="stretch", hide_index=True, height=225)

st.caption(
    "Target construction diagnostics are evaluated at the signal date; realized holdings "
    "are displayed separately to avoid mixing ex-ante targets with subsequent market drift."
)
