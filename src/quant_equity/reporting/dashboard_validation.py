from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_equity.reporting.dashboard_catalog import (
    DASHBOARD_SOURCES,
    STRATEGY_ORDER,
    source_path,
)
from quant_equity.reporting.dashboard_data import read_dashboard_source


@dataclass(frozen=True)
class DashboardValidationResult:
    check: str
    status: str
    violations: int
    description: str


def validate_source_contracts() -> list[DashboardValidationResult]:
    results: list[DashboardValidationResult] = []

    for source_id, source in DASHBOARD_SOURCES.items():
        path = source_path(source_id)
        if not path.exists():
            results.append(
                DashboardValidationResult(
                    check=f"source_exists:{source_id}",
                    status="FAIL",
                    violations=1,
                    description=f"Missing canonical dashboard source: {path}",
                )
            )
            continue

        frame = read_dashboard_source(source_id)
        missing_columns = sorted(set(source.required_columns) - set(frame.columns))
        results.append(
            DashboardValidationResult(
                check=f"source_schema:{source_id}",
                status="PASS" if not missing_columns else "FAIL",
                violations=len(missing_columns),
                description=(
                    "Required columns are present."
                    if not missing_columns
                    else f"Missing required columns: {missing_columns}"
                ),
            )
        )

    return results


def validate_strategy_coverage() -> list[DashboardValidationResult]:
    expected = set(STRATEGY_ORDER)
    checks = (
        ("performance_net_daily", "strategy_name"),
        ("performance_summary", "strategy_name"),
        ("target_weights", "method"),
        ("positions_daily", "strategy_name"),
        ("portfolio_diagnostics", "method"),
        ("portfolio_risk", "method"),
        ("trades", "strategy_name"),
        ("execution_summary", "strategy_name"),
        ("capacity", "strategy_name"),
    )

    results: list[DashboardValidationResult] = []
    for source_id, column in checks:
        frame = read_dashboard_source(source_id)
        observed = set(frame[column].dropna().astype(str).unique())
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        violations = len(missing) + len(unexpected)

        results.append(
            DashboardValidationResult(
                check=f"strategy_coverage:{source_id}",
                status="PASS" if violations == 0 else "FAIL",
                violations=violations,
                description=(
                    "Strategy universe matches the canonical five-method contract."
                    if violations == 0
                    else f"Missing={missing}; unexpected={unexpected}"
                ),
            )
        )

    return results


def validate_robustness_status() -> list[DashboardValidationResult]:
    inventory = read_dashboard_source("robustness_inventory")
    coverage = read_dashboard_source("robustness_coverage")

    failed_suites = int((inventory["suite_status"].astype(str) != "PASS").sum())
    allowed_coverage = {"COMPLETE", "DEFERRED_LIMITATION"}
    invalid_coverage = int((~coverage["status"].astype(str).isin(allowed_coverage)).sum())

    deferred = coverage[coverage["status"].astype(str) == "DEFERRED_LIMITATION"]
    expanded_universe_deferred = (
        len(deferred) == 1 and str(deferred.iloc[0]["dimension"]) == "expanded_universe"
    )

    return [
        DashboardValidationResult(
            check="robustness_suites",
            status="PASS" if failed_suites == 0 else "FAIL",
            violations=failed_suites,
            description="Every audited robustness suite must pass.",
        ),
        DashboardValidationResult(
            check="robustness_coverage_status",
            status="PASS" if invalid_coverage == 0 else "FAIL",
            violations=invalid_coverage,
            description=("Coverage may only be COMPLETE or an explicitly documented limitation."),
        ),
        DashboardValidationResult(
            check="expanded_universe_limitation",
            status="PASS" if expanded_universe_deferred else "FAIL",
            violations=0 if expanded_universe_deferred else 1,
            description=("The only deferred robustness dimension must be the expanded universe."),
        ),
    ]


def run_dashboard_validation() -> pd.DataFrame:
    results = [
        *validate_source_contracts(),
        *validate_strategy_coverage(),
        *validate_robustness_status(),
    ]
    return pd.DataFrame(
        {
            "check": [result.check for result in results],
            "status": [result.status for result in results],
            "violations": [result.violations for result in results],
            "description": [result.description for result in results],
        }
    )
