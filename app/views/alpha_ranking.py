from __future__ import annotations

import pandas as pd
import streamlit as st
from components.charts import alpha_ranking_figure, component_contribution_figure
from components.formatting import format_currency, format_percent, format_ratio
from components.shell import render_page_header
from data_access import load_dashboard_source

from quant_equity.reporting.dashboard_catalog import DEFAULT_STRATEGY, strategy_label
from quant_equity.reporting.dashboard_metrics import (
    build_alpha_snapshot,
    ensemble_weights,
    parse_model_contributions,
    signal_dates,
)

strategy = st.session_state.get("dashboard_strategy", DEFAULT_STRATEGY)
alpha_signal = load_dashboard_source("alpha_signal")
security_risk = load_dashboard_source("security_risk")
target_weights = load_dashboard_source("target_weights")

dates = signal_dates(alpha_signal)
if not dates:
    raise ValueError("The frozen alpha signal contains no valid dates.")

render_page_header(
    "Alpha & Security Ranking",
    "Cross-sectional signal strength, ensemble composition, and security-level risk context.",
    eyebrow="Institutional Quant Equity Research Platform",
)

st.caption(
    "The ranking is the frozen full-model signal shared by all portfolio methods. "
    "The portfolio selector only changes the holding and weight overlay."
)

control_left, control_right = st.columns([1.0, 2.2], gap="medium")
with control_left:
    selected_date = st.selectbox(
        "Signal date",
        options=list(reversed(dates)),
        index=0,
        format_func=lambda value: pd.Timestamp(value).date().isoformat(),
    )
with control_right:
    st.caption("Selected portfolio method")
    st.markdown(f"**{strategy_label(strategy)}**")

snapshot = build_alpha_snapshot(
    alpha_signal,
    security_risk,
    target_weights,
    strategy,
    pd.Timestamp(selected_date),
)
weights = ensemble_weights(snapshot)

top_quintile = max(1, int(round(len(snapshot) * 0.20)))
top_ten = snapshot.nsmallest(10, "rank")
selected_positions = int((snapshot["selected_weight"] > 0.0).sum())
top_ten_weight = float(top_ten["selected_weight"].sum())

kpi_columns = st.columns(4, gap="small")
kpi_columns[0].metric("Cross-section", f"{len(snapshot)}", border=True)
kpi_columns[1].metric("Top quintile", f"{top_quintile}", border=True)
kpi_columns[2].metric("Selected holdings", f"{selected_positions}", border=True)
kpi_columns[3].metric(
    "Portfolio weight in top 10",
    format_percent(top_ten_weight),
    border=True,
)

ranking_column, ensemble_column = st.columns([1.45, 1.10], gap="medium")

with ranking_column:
    with st.container(border=True):
        st.subheader("Top-ranked securities")
        st.caption(
            "Navy bars are held by the selected portfolio method; gray bars are signal-only."
        )
        st.plotly_chart(
            alpha_ranking_figure(snapshot),
            width="stretch",
            config={"displayModeBar": False},
        )

with ensemble_column:
    with st.container(border=True):
        st.subheader("Ensemble mix")
        st.caption("Fold-specific validation weights used by the frozen signal.")

        for label, value in weights.items():
            st.markdown(
                (
                    "<div style='display:flex;justify-content:space-between;"
                    "align-items:center;padding:0.18rem 0;'>"
                    f"<span style='color:#66717D;font-size:0.84rem;'>{label}</span>"
                    f"<strong style='font-size:1.05rem;'>{format_percent(value)}</strong>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        st.plotly_chart(
            component_contribution_figure(snapshot),
            width="stretch",
            config={"displayModeBar": False},
        )

with st.container(border=True):
    st.subheader("Full security ranking")
    st.caption(
        "Signal metrics are common to every construction method. Portfolio weight reflects "
        f"the selected {strategy_label(strategy)} construction."
    )

    table = snapshot.loc[
        :,
        [
            "rank",
            "ticker",
            "sector",
            "percentile_score",
            "selected_weight",
            "annualized_volatility",
            "beta_vs_spy",
            "average_dollar_volume",
            "composite_contribution",
            "elastic_net_contribution",
            "lightgbm_ranker_contribution",
        ],
    ].rename(
        columns={
            "rank": "Rank",
            "ticker": "Ticker",
            "sector": "Sector",
            "percentile_score": "Signal",
            "selected_weight": "Portfolio weight",
            "annualized_volatility": "Volatility",
            "beta_vs_spy": "Beta",
            "average_dollar_volume": "ADV",
            "composite_contribution": "Technical",
            "elastic_net_contribution": "Elastic Net",
            "lightgbm_ranker_contribution": "LightGBM",
        }
    )

    table["Signal"] = table["Signal"].map(lambda value: f"{float(value):.1%}")
    table["Portfolio weight"] = table["Portfolio weight"].map(lambda value: f"{float(value):.2%}")
    table["Volatility"] = table["Volatility"].map(lambda value: f"{float(value):.1%}")
    table["Beta"] = table["Beta"].map(lambda value: f"{float(value):.2f}")
    table["ADV"] = table["ADV"].map(lambda value: format_currency(float(value)))
    for column in ("Technical", "Elastic Net", "LightGBM"):
        table[column] = table[column].map(lambda value: f"{float(value):.3f}")

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=560,
    )

with st.container(border=True):
    st.subheader("Security drill-down")
    tickers = snapshot.sort_values("rank")["ticker"].astype(str).tolist()
    selected_ticker = st.selectbox("Security", options=tickers, index=0)
    security = snapshot.loc[snapshot["ticker"].astype(str) == selected_ticker].iloc[0]

    first_metrics = st.columns(3, gap="small")
    first_metrics[0].metric("Rank", f"#{int(security['rank'])}", border=True)
    first_metrics[1].metric(
        "Signal",
        format_percent(float(security["percentile_score"])),
        border=True,
    )
    first_metrics[2].metric(
        "Portfolio weight",
        format_percent(float(security["selected_weight"])),
        border=True,
    )

    second_metrics = st.columns(3, gap="small")
    second_metrics[0].metric(
        "Volatility",
        format_percent(float(security["annualized_volatility"])),
        border=True,
    )
    second_metrics[1].metric(
        "Beta",
        format_ratio(float(security["beta_vs_spy"])),
        border=True,
    )
    second_metrics[2].metric(
        "ADV",
        format_currency(float(security["average_dollar_volume"])),
        border=True,
    )

    st.caption(
        f"{selected_ticker} · {security['sector']} · signal date "
        f"{pd.Timestamp(selected_date).date().isoformat()}"
    )

    components = parse_model_contributions(str(security["model_contributions"]))
    component_table = components.rename(
        columns={
            "component": "Component",
            "percentile": "Component percentile",
            "weight": "Ensemble weight",
            "contribution": "Weighted contribution",
        }
    )
    st.dataframe(
        component_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Component percentile": st.column_config.NumberColumn(format="percent"),
            "Ensemble weight": st.column_config.NumberColumn(format="percent"),
            "Weighted contribution": st.column_config.NumberColumn(format="%.3f"),
        },
    )
