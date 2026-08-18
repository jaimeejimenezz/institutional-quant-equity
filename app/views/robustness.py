from __future__ import annotations

from components.shell import render_foundation_notice, render_page_header

render_page_header(
    "Robustness",
    "Bootstrap evidence, regimes, horizons, windows, exclusions, and ablations.",
)
render_foundation_notice(
    "robustness",
    (
        "robustness_inventory",
        "robustness_coverage",
        "bootstrap_strategy",
        "signal_bootstrap",
        "feature_family_ablation",
        "economic_ablation",
        "construction_ablation",
        "horizon_sensitivity",
        "rebalance_sensitivity",
        "rolling_window_sensitivity",
        "universe_exclusions",
        "regime_performance",
    ),
)
