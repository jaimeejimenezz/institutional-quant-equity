from __future__ import annotations

import pandas as pd

from quant_equity.reporting.dashboard_data_quality import (
    failed_controls,
    quality_headline,
    standardize_check_table,
    suite_summary,
)


def test_suite_summary_counts_standard_checks() -> None:
    sources = {
        "leakage_checks": pd.DataFrame(
            {
                "check": ["a", "b"],
                "status": ["PASS", "PASS"],
                "violations": [0, 0],
                "description": ["A", "B"],
            }
        ),
        "risk_checks": pd.DataFrame(
            {
                "check": ["c"],
                "status": ["PASS"],
                "violations": [0],
                "description": ["C"],
            }
        ),
    }

    summary = suite_summary(sources)

    assert summary["checks"].sum() == 3
    assert summary["passed"].sum() == 3
    assert summary["failed"].sum() == 0
    assert set(summary["status"]) == {"PASS"}


def test_robustness_inventory_is_normalized() -> None:
    frame = pd.DataFrame(
        {
            "suite": ["calendar_years"],
            "category": ["temporal"],
            "checks": [3],
            "passed_checks": [3],
            "failed_checks": [0],
            "suite_status": ["PASS"],
        }
    )

    result = standardize_check_table("robustness_inventory", frame)

    assert result.loc[0, "Check"] == "Calendar Years"
    assert result.loc[0, "Status"] == "PASS"
    assert result.loc[0, "Violations"] == 0


def test_failed_controls_returns_non_passing_rows() -> None:
    sources = {
        "portfolio_checks": pd.DataFrame(
            {
                "check": ["weights", "sector"],
                "status": ["PASS", "FAIL"],
                "violations": [0, 2],
                "description": ["Weights valid", "Sector cap exceeded"],
            }
        )
    }

    result = failed_controls(sources)

    assert len(result) == 1
    assert result.loc[0, "Status"] == "FAIL"
    assert result.loc[0, "Violations"] == 2


def test_quality_headline_aggregates_suites() -> None:
    summary = pd.DataFrame(
        {
            "status": ["PASS", "ATTENTION"],
            "checks": [5, 4],
            "passed": [5, 3],
            "failed": [0, 1],
            "violations": [0, 2],
        }
    )

    headline = quality_headline(summary)

    assert headline["suites"] == 2
    assert headline["passing_suites"] == 1
    assert headline["checks"] == 9
    assert headline["failed_checks"] == 1
    assert headline["violations"] == 2
