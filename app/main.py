from __future__ import annotations

import streamlit as st
from data_access import dashboard_context

from quant_equity.reporting.dashboard_catalog import (
    DEFAULT_STRATEGY,
    STRATEGY_ORDER,
    strategy_label,
)

st.set_page_config(
    page_title="Institutional Quant Equity Research",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dashboard_strategy" not in st.session_state:
    st.session_state.dashboard_strategy = DEFAULT_STRATEGY

pages = {
    "PORTFOLIO": [
        st.Page("views/overview.py", title="Overview", default=True),
        st.Page("views/alpha_ranking.py", title="Alpha & Ranking"),
        st.Page("views/portfolio.py", title="Portfolio"),
    ],
    "RISK & EXECUTION": [
        st.Page("views/risk.py", title="Risk"),
        st.Page("views/execution.py", title="Execution & Capacity"),
    ],
    "RESEARCH": [
        st.Page("views/model_research.py", title="Models & Factors"),
        st.Page("views/robustness.py", title="Robustness"),
    ],
    "SYSTEM": [
        st.Page("views/data_quality.py", title="Data Quality"),
    ],
}

navigation = st.navigation(pages, position="sidebar", expanded=True)

with st.sidebar:
    st.markdown("### INSTITUTIONAL QUANT EQUITY")
    st.caption("Systematic US Equity Research")

    selected_strategy = st.selectbox(
        "Portfolio method",
        options=list(STRATEGY_ORDER),
        index=list(STRATEGY_ORDER).index(st.session_state.dashboard_strategy),
        format_func=strategy_label,
        key="dashboard_strategy_selector",
    )
    st.session_state.dashboard_strategy = selected_strategy

    context = dashboard_context()
    st.divider()
    st.caption("RESEARCH CONTEXT")
    st.text("OOS period      2020-2026")
    st.text("Universe        50 US equities")
    st.text("Horizon         21 sessions")
    st.text("Rebalance       Monthly")
    st.text(f"Robustness      {context['robustness_passed']}/{context['robustness_total']} PASS")
    st.caption("Results are research outputs, not live trading recommendations.")

navigation.run()
