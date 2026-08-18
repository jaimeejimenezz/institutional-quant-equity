from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from quant_equity.reporting.dashboard_data import latest_date, read_dashboard_source
from quant_equity.reporting.dashboard_validation import run_dashboard_validation


@st.cache_data(show_spinner=False)
def load_dashboard_source(source_id: str) -> pd.DataFrame:
    return read_dashboard_source(source_id)


@st.cache_data(show_spinner=False)
def load_dashboard_validation() -> pd.DataFrame:
    return run_dashboard_validation()


@st.cache_data(show_spinner=False)
def dashboard_context() -> dict[str, Any]:
    alpha = load_dashboard_source("alpha_signal")
    performance = load_dashboard_source("performance_net_daily")
    benchmark = load_dashboard_source("benchmark_spy")
    robustness = load_dashboard_source("robustness_inventory")
    coverage = load_dashboard_source("robustness_coverage")

    passed_suites = int((robustness["suite_status"].astype(str) == "PASS").sum())
    total_suites = int(len(robustness))
    deferred = coverage[coverage["status"].astype(str) == "DEFERRED_LIMITATION"]

    return {
        "last_signal_date": latest_date(alpha, "as_of_date"),
        "last_evaluated_date": latest_date(performance, "date"),
        "benchmark_latest_date": latest_date(benchmark, "date"),
        "robustness_passed": passed_suites,
        "robustness_total": total_suites,
        "deferred_dimensions": deferred["dimension"].astype(str).tolist(),
    }
