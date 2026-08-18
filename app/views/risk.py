from __future__ import annotations

from components.shell import render_foundation_notice, render_page_header

render_page_header(
    "Risk",
    "Ex-ante volatility, beta, risk contributions, covariance, and liquidity.",
)
render_foundation_notice(
    "risk",
    ("security_risk", "portfolio_risk", "risk_contributions", "covariance_diagnostics"),
)
