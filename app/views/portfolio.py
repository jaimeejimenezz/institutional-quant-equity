from __future__ import annotations

from components.shell import render_foundation_notice, render_page_header

render_page_header(
    "Portfolio",
    "Target weights, realized positions, sector exposure, concentration, and drift.",
)
render_foundation_notice(
    "portfolio",
    ("target_weights", "positions_daily", "portfolio_diagnostics", "portfolio_risk"),
)
