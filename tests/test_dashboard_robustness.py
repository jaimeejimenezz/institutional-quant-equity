from __future__ import annotations

import pandas as pd

from quant_equity.reporting.dashboard_robustness import (
    coverage_table,
    humanize_identifier,
    robustness_headline,
    strategy_bootstrap_table,
    strategy_chart_label,
    strategy_display_label,
)


def test_robustness_headline_counts_pass_and_deferred() -> None:
    inventory = pd.DataFrame({"suite_status": ["PASS", "PASS", "PASS"]})
    coverage = pd.DataFrame(
        {"status": ["COMPLETE", "COMPLETE", "DEFERRED_LIMITATION"]}
    )
    signal = pd.DataFrame(
        {
            "bootstrap_replications": [10000],
            "months": [77],
        }
    )

    headline = robustness_headline(inventory, coverage, signal)

    assert headline["passed_suites"] == 3
    assert headline["total_suites"] == 3
    assert headline["complete_dimensions"] == 2
    assert headline["deferred_dimensions"] == 1
    assert headline["bootstrap_replications"] == 10000
    assert headline["oos_months"] == 77


def test_coverage_table_is_compact() -> None:
    coverage = pd.DataFrame(
        {
            "dimension": ["expanded_universe"],
            "category": ["universe"],
            "status": ["DEFERRED_LIMITATION"],
            "note": ["Requires new securities."],
        }
    )

    result = coverage_table(coverage)

    assert result.columns.tolist() == ["Dimension", "Category", "Status"]
    assert result.loc[0, "Status"] == "Documented limitation"


def test_strategy_labels_preserve_acronyms_and_have_chart_variants() -> None:
    assert strategy_display_label("cvar") == "CVaR"
    assert strategy_display_label("median_mad_de") == "Median-MAD DE"
    assert strategy_chart_label("median_mad_de") == "Median-MAD"
    assert strategy_chart_label("alpha_risk_turnover") == "Alpha-Risk"
    assert humanize_identifier("no_lightgbm") == "No LightGBM"


def test_bootstrap_table_combines_confidence_intervals() -> None:
    data = pd.DataFrame(
        {
            "strategy_name": ["cvar"],
            "observed_annualized_return": [0.20],
            "annualized_return_ci_lower": [0.04],
            "annualized_return_ci_upper": [0.39],
            "observed_sharpe": [1.08],
            "sharpe_ci_lower": [0.29],
            "sharpe_ci_upper": [1.95],
            "probability_excess_annualized_return_positive": [0.96],
        }
    )

    result = strategy_bootstrap_table(data)

    assert result.loc[0, "Strategy"] == "CVaR"
    assert result.loc[0, "Return 95% CI"] == "4.0% to 39.0%"
    assert result.loc[0, "Sharpe 95% CI"] == "0.29 to 1.95"
