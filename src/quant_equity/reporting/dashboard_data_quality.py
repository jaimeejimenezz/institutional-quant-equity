from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

SUITE_LABELS: dict[str, str] = {
    "leakage_checks": "Modeling leakage",
    "panel_readiness": "Modeling panel",
    "walk_forward_readiness": "Walk-forward validation",
    "risk_checks": "Risk estimates",
    "covariance_checks": "Covariance model",
    "portfolio_checks": "Portfolio construction",
    "execution_checks": "Execution engine",
    "robustness_inventory": "Robustness evaluation",
}


def suite_label(source_id: str) -> str:
    """Return the institutional display label for a validation source."""
    return SUITE_LABELS.get(source_id, source_id.replace("_", " ").title())


def standardize_check_table(
    source_id: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize validation artifacts to a common dashboard schema."""
    if source_id == "robustness_inventory":
        result = frame.loc[
            :,
            [
                "suite",
                "category",
                "checks",
                "passed_checks",
                "failed_checks",
                "suite_status",
            ],
        ].copy()
        result = result.rename(
            columns={
                "suite": "Check",
                "category": "Category",
                "checks": "Subchecks",
                "passed_checks": "Passed",
                "failed_checks": "Violations",
                "suite_status": "Status",
            }
        )
        result["Description"] = (
            "Final robustness evaluation suite and artifact coverage check."
        )
        result["Check"] = result["Check"].astype(str).str.replace("_", " ").str.title()
        result["Category"] = (
            result["Category"].astype(str).str.replace("_", " ").str.title()
        )
        return result[
            [
                "Check",
                "Category",
                "Status",
                "Violations",
                "Subchecks",
                "Passed",
                "Description",
            ]
        ]

    result = frame.loc[:, ["check", "status", "violations", "description"]].copy()
    result = result.rename(
        columns={
            "check": "Check",
            "status": "Status",
            "violations": "Violations",
            "description": "Description",
        }
    )
    result["Check"] = result["Check"].astype(str).str.replace("_", " ").str.title()
    result["Category"] = suite_label(source_id)
    result["Subchecks"] = pd.NA
    result["Passed"] = pd.NA
    return result[
        [
            "Check",
            "Category",
            "Status",
            "Violations",
            "Subchecks",
            "Passed",
            "Description",
        ]
    ]


def suite_summary(
    sources: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate validation status across all dashboard control artifacts."""
    rows: list[dict[str, object]] = []

    for source_id, frame in sources.items():
        if source_id == "robustness_inventory":
            statuses = frame["suite_status"].astype(str).str.upper()
            violations = pd.to_numeric(
                frame["failed_checks"],
                errors="coerce",
            ).fillna(0)
            checks = int(len(frame))
        else:
            statuses = frame["status"].astype(str).str.upper()
            violations = pd.to_numeric(
                frame["violations"],
                errors="coerce",
            ).fillna(0)
            checks = int(len(frame))

        passed = int(statuses.eq("PASS").sum())
        failed = int(checks - passed)

        rows.append(
            {
                "source_id": source_id,
                "suite": suite_label(source_id),
                "checks": checks,
                "passed": passed,
                "failed": failed,
                "violations": int(violations.sum()),
                "status": "PASS" if failed == 0 else "ATTENTION",
            }
        )

    return pd.DataFrame(rows)


def quality_headline(summary: pd.DataFrame) -> dict[str, int]:
    """Return the headline data-quality metrics."""
    suites = int(len(summary))
    passing_suites = int(summary["status"].eq("PASS").sum())
    checks = int(summary["checks"].sum())
    passed_checks = int(summary["passed"].sum())
    failed_checks = int(summary["failed"].sum())
    violations = int(summary["violations"].sum())

    return {
        "suites": suites,
        "passing_suites": passing_suites,
        "checks": checks,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "violations": violations,
    }


def failed_controls(
    sources: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Collect any non-passing controls for escalation."""
    frames: list[pd.DataFrame] = []

    for source_id, frame in sources.items():
        standardized = standardize_check_table(source_id, frame)
        failed = standardized.loc[
            ~standardized["Status"].astype(str).str.upper().eq("PASS")
        ].copy()
        if failed.empty:
            continue
        failed.insert(0, "Suite", suite_label(source_id))
        frames.append(failed)

    if not frames:
        return pd.DataFrame(
            columns=[
                "Suite",
                "Check",
                "Category",
                "Status",
                "Violations",
                "Description",
            ]
        )

    return pd.concat(frames, ignore_index=True)[
        [
            "Suite",
            "Check",
            "Category",
            "Status",
            "Violations",
            "Description",
        ]
    ]
