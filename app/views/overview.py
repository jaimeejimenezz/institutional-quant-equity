from __future__ import annotations

import streamlit as st
from components.shell import (
    render_foundation_notice,
    render_page_header,
    render_validation_summary,
)
from data_access import dashboard_context, load_dashboard_validation

render_page_header(
    "Executive Overview",
    "Portfolio performance, implementation quality, and research evidence.",
    eyebrow="Institutional Quant Equity Research Platform",
)

context = dashboard_context()
validation = load_dashboard_validation()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last signal", context["last_signal_date"].date().isoformat())
col2.metric("Last evaluated day", context["last_evaluated_date"].date().isoformat())
col3.metric("Benchmark freshness", context["benchmark_latest_date"].date().isoformat())
col4.metric(
    "Robustness suites",
    f"{context['robustness_passed']}/{context['robustness_total']}",
)

render_validation_summary(validation)

if context["deferred_dimensions"]:
    st.warning(
        "Documented methodological limitation: " + ", ".join(context["deferred_dimensions"]) + "."
    )

render_foundation_notice(
    "overview",
    (
        "performance_net_daily",
        "performance_gross_daily",
        "performance_summary",
        "benchmark_spy",
    ),
)
