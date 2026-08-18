from __future__ import annotations

from components.shell import (
    render_foundation_notice,
    render_page_header,
    render_validation_summary,
)
from data_access import load_dashboard_validation

render_page_header(
    "Data Quality & Controls",
    "Point-in-time integrity, leakage controls, readiness checks, and system validation.",
)

validation = load_dashboard_validation()
render_validation_summary(validation)

render_foundation_notice(
    "data quality",
    (
        "leakage_checks",
        "panel_readiness",
        "walk_forward_readiness",
        "risk_checks",
        "covariance_checks",
        "portfolio_checks",
        "execution_checks",
        "robustness_inventory",
    ),
)
