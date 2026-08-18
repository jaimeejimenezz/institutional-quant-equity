from __future__ import annotations

from components.shell import render_foundation_notice, render_page_header

render_page_header(
    "Models & Factors",
    "Out-of-sample model evidence, IC, spreads, stability, and feature importance.",
)
render_foundation_notice(
    "model research",
    (
        "model_summary",
        "model_monthly",
        "model_yearly",
        "model_sector",
        "feature_importance",
        "ensemble_summary",
    ),
)
