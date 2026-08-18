from __future__ import annotations

import pandas as pd
import streamlit as st
from components.charts import (
    covariance_diagnostics_figure,
    reference_risk_contribution_figure,
    risk_history_figure,
    risk_method_comparison_figure,
    security_risk_map_figure,
)
from components.formatting import format_currency, format_percent, format_ratio
from components.shell import render_page_header
from data_access import load_dashboard_source

from quant_equity.reporting.dashboard_catalog import DEFAULT_STRATEGY, strategy_label
from quant_equity.reporting.dashboard_metrics import (
    covariance_history,
    covariance_snapshot,
    current_security_risk,
    portfolio_diagnostics_row,
    reference_risk_contribution_snapshot,
    risk_dates,
    risk_history,
    risk_method_comparison,
    risk_summary_row,
)

strategy = st.session_state.get("dashboard_strategy", DEFAULT_STRATEGY)


def _format_small_percent(value: float) -> str:
    if 0.0 < abs(float(value)) < 0.0001:
        return "<0.01%"
    return format_percent(float(value), digits=2)


def _format_weight(value: float) -> str:
    if 0.0 < float(value) < 0.0001:
        return "<0.01%"
    return f"{float(value):.2%}"


portfolio_risk = load_dashboard_source("portfolio_risk")
portfolio_diagnostics = load_dashboard_source("portfolio_diagnostics")
security_risk = load_dashboard_source("security_risk")
target_weights = load_dashboard_source("target_weights")
covariance = load_dashboard_source("covariance_diagnostics")
reference_contributions = load_dashboard_source("risk_contributions")

available_dates = risk_dates(portfolio_risk, strategy)
if not available_dates:
    raise ValueError(f"No risk dates found for {strategy!r}.")

render_page_header(
    "Risk Analytics",
    "Ex-ante portfolio risk, security exposures, covariance diagnostics, and liquidity.",
    eyebrow="Institutional Quant Equity Research Platform",
)

st.caption(
    "Portfolio-level risk metrics respond to the selected construction method. "
    "Covariance diagnostics are common risk-model inputs across methods."
)

control_left, control_right = st.columns([1.0, 2.2], gap="medium")
with control_left:
    selected_date = st.selectbox(
        "Risk date",
        options=list(reversed(available_dates)),
        index=0,
        format_func=lambda value: pd.Timestamp(value).date().isoformat(),
    )
with control_right:
    st.caption("Selected construction method")
    st.markdown(f"**{strategy_label(strategy)}**")

selected_date = pd.Timestamp(selected_date)
risk = risk_summary_row(portfolio_risk, strategy, selected_date)
diagnostic = portfolio_diagnostics_row(
    portfolio_diagnostics,
    strategy,
    selected_date,
)
active_positions = int(diagnostic["positions"])
history = risk_history(portfolio_risk, strategy)
securities = current_security_risk(
    target_weights,
    security_risk,
    strategy,
    selected_date,
    active_positions=active_positions,
)
covariance_row = covariance_snapshot(covariance, selected_date)
covariance_series = covariance_history(covariance)
comparison = risk_method_comparison(portfolio_risk, strategy, selected_date)
reference = reference_risk_contribution_snapshot(
    reference_contributions,
    selected_date,
)

kpi_top = st.columns(3, gap="small")
kpi_top[0].metric(
    "Predicted volatility",
    format_percent(float(risk["predicted_volatility"])),
    border=True,
)
kpi_top[1].metric(
    "Beta vs SPY",
    format_ratio(float(risk["portfolio_beta_vs_spy"])),
    border=True,
)
kpi_top[2].metric(
    "Effective positions",
    format_ratio(float(risk["effective_positions"]), digits=1),
    border=True,
)

kpi_bottom = st.columns(3, gap="small")
kpi_bottom[0].metric(
    "Maximum sector",
    format_percent(float(risk["maximum_sector_weight"])),
    border=True,
)
kpi_bottom[1].metric(
    "Maximum active sector",
    format_percent(float(risk["maximum_active_sector_weight"])),
    border=True,
)
kpi_bottom[2].metric(
    "Concentration HHI",
    format_ratio(float(risk["concentration_hhi"]), digits=3),
    border=True,
)

volatility_column, beta_column = st.columns(2, gap="medium")
with volatility_column:
    with st.container(border=True):
        st.subheader("Predicted volatility history")
        st.caption("Selected method versus the Top-N equal-weight baseline.")
        st.plotly_chart(
            risk_history_figure(
                history,
                metric="predicted_volatility",
                title="Predicted volatility",
                tickformat=".0%",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

with beta_column:
    with st.container(border=True):
        st.subheader("Portfolio beta history")
        st.caption("Ex-ante portfolio beta versus SPY at each signal date.")
        st.plotly_chart(
            risk_history_figure(
                history,
                metric="portfolio_beta_vs_spy",
                title="Beta vs SPY",
                tickformat=".2f",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

map_column, liquidity_column = st.columns([1.45, 1.0], gap="medium")
with map_column:
    with st.container(border=True):
        st.subheader("Current holdings risk map")
        st.caption("Security volatility versus market beta. Marker size reflects target weight.")
        st.plotly_chart(
            security_risk_map_figure(securities),
            width="stretch",
            config={"displayModeBar": False},
        )

with liquidity_column:
    with st.container(border=True):
        st.subheader("Liquidity & concentration")
        st.caption("Capacity indicators are expressed relative to recent average dollar volume.")

        liquidity_metrics = [
            (
                "Max position / ADV",
                _format_small_percent(float(risk["maximum_position_adv_fraction"])),
            ),
            (
                "Weighted position / ADV",
                _format_small_percent(float(risk["weighted_position_adv_fraction"])),
            ),
            (
                "Maximum liquidation",
                f"{float(risk['maximum_liquidation_days']):.3f} days",
            ),
            (
                "Weighted liquidation",
                f"{float(risk['weighted_liquidation_days']):.3f} days",
            ),
            ("Active positions", f"{active_positions}"),
            (
                "Risk contribution total",
                format_percent(float(risk["risk_contribution_sum"])),
            ),
        ]

        for label, value in liquidity_metrics:
            st.markdown(
                (
                    "<div style='display:flex;justify-content:space-between;"
                    "align-items:baseline;padding:0.32rem 0;"
                    "border-bottom:1px solid #E3E7EB;'>"
                    f"<span style='color:#66717D;font-size:0.82rem;'>{label}</span>"
                    f"<strong style='font-size:1.05rem;'>{value}</strong>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

with st.container(border=True):
    st.subheader("Security risk detail")
    table = securities.loc[
        :,
        [
            "ticker",
            "sector",
            "weight",
            "annualized_volatility",
            "annualized_downside_volatility",
            "beta_vs_spy",
            "correlation_vs_spy",
            "average_dollar_volume",
        ],
    ].rename(
        columns={
            "ticker": "Ticker",
            "sector": "Sector",
            "weight": "Weight",
            "annualized_volatility": "Volatility",
            "annualized_downside_volatility": "Downside vol",
            "beta_vs_spy": "Beta",
            "correlation_vs_spy": "SPY corr.",
            "average_dollar_volume": "ADV",
        }
    )
    table["Weight"] = table["Weight"].map(_format_weight)
    table["Volatility"] = table["Volatility"].map(lambda value: f"{float(value):.1%}")
    table["Downside vol"] = table["Downside vol"].map(lambda value: f"{float(value):.1%}")
    table["Beta"] = table["Beta"].map(lambda value: f"{float(value):.2f}")
    table["SPY corr."] = table["SPY corr."].map(lambda value: f"{float(value):.2f}")
    table["ADV"] = table["ADV"].map(lambda value: format_currency(float(value)))
    st.dataframe(table, width="stretch", hide_index=True, height=460)

covariance_column, comparison_column = st.columns([1.15, 1.0], gap="medium")

with covariance_column:
    with st.container(border=True):
        st.subheader("Covariance model diagnostics")
        st.caption("Common shrinkage-covariance inputs used by the ex-ante portfolio risk model.")
        cov_top = st.columns(2, gap="small")
        cov_top[0].metric(
            "Shrinkage",
            format_ratio(float(covariance_row["shrinkage"]), digits=3),
        )
        cov_top[1].metric(
            "Mean correlation",
            format_ratio(
                float(covariance_row["mean_pairwise_correlation"]),
                digits=3,
            ),
        )
        cov_bottom = st.columns(2, gap="small")
        cov_bottom[0].metric(
            "Max correlation",
            format_ratio(
                float(covariance_row["maximum_pairwise_correlation"]),
                digits=3,
            ),
        )
        cov_bottom[1].metric(
            "Condition number",
            f"{float(covariance_row['shrinkage_condition_number']):,.1f}",
        )
        st.plotly_chart(
            covariance_diagnostics_figure(covariance_series),
            width="stretch",
            config={"displayModeBar": False},
        )

with comparison_column:
    with st.container(border=True):
        st.subheader("Method risk comparison")
        st.caption("Selected-date beta and predicted-volatility trade-off.")
        st.plotly_chart(
            risk_method_comparison_figure(comparison),
            width="stretch",
            config={"displayModeBar": False},
        )

with st.container(border=True):
    st.subheader("Reference portfolio risk decomposition")
    st.warning(
        "This stored contribution artifact has no construction-method identifier. "
        "It is shown only as a reference risk-model diagnostic and is not attributed "
        "to the selected portfolio method."
    )

    st.plotly_chart(
        reference_risk_contribution_figure(reference),
        width="stretch",
        config={"displayModeBar": False},
    )

    reference_table = reference.loc[
        :,
        [
            "ticker",
            "sector",
            "weight",
            "risk_contribution_share",
            "annualized_volatility",
            "beta_vs_spy",
            "liquidation_days",
        ],
    ].head(12)
    reference_table = reference_table.rename(
        columns={
            "ticker": "Ticker",
            "sector": "Sector",
            "weight": "Weight",
            "risk_contribution_share": "Risk share",
            "annualized_volatility": "Volatility",
            "beta_vs_spy": "Beta",
            "liquidation_days": "Liquidation",
        }
    )
    reference_table["Weight"] = reference_table["Weight"].map(lambda value: f"{float(value):.1%}")
    reference_table["Risk share"] = reference_table["Risk share"].map(
        lambda value: f"{float(value):.1%}"
    )
    reference_table["Volatility"] = reference_table["Volatility"].map(
        lambda value: f"{float(value):.1%}"
    )
    reference_table["Beta"] = reference_table["Beta"].map(lambda value: f"{float(value):.2f}")
    reference_table["Liquidation"] = reference_table["Liquidation"].map(
        lambda value: f"{float(value):.3f} d"
    )
    st.dataframe(
        reference_table,
        width="stretch",
        hide_index=True,
        height=410,
    )

st.caption(
    "Risk metrics are ex-ante estimates at the selected signal date. "
    "They should not be interpreted as realized future volatility."
)
