from __future__ import annotations

from components.shell import render_foundation_notice, render_page_header

render_page_header(
    "Execution & Capacity",
    "Trades, turnover, transaction-cost decomposition, sensitivity, and capacity.",
)
render_foundation_notice(
    "execution",
    ("trades", "execution_summary", "execution_cost_components", "cost_sensitivity", "capacity"),
)
