"""Point-in-time and leakage audit for the master modeling panel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.models.modeling_panel import (
    MODEL_FEATURE_COLUMNS,
    TARGET_METADATA_COLUMNS,
    TARGET_VALUE_COLUMNS,
)


class ModelingPanelAuditError(ValueError):
    """Raised when the modeling-panel audit cannot be completed."""


@dataclass(frozen=True)
class ModelingPanelAuditResult:
    """Result of the complete temporal audit."""

    is_valid: bool
    summary: dict[str, Any]
    checks: pd.DataFrame
    issues: tuple[str, ...]


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require expected dataset columns."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise ModelingPanelAuditError(
            f"{dataset_name} is missing columns: " + ", ".join(missing) + "."
        )


def _normalize_date(
    data: pd.DataFrame,
    column: str,
) -> None:
    """Normalize one date column in place."""
    data[column] = pd.to_datetime(
        data[column],
        errors="coerce",
    ).dt.normalize()


def _record_check(
    rows: list[dict[str, Any]],
    issues: list[str],
    *,
    name: str,
    violations: int,
    description: str,
) -> None:
    """Record one blocking audit check."""
    status = "PASS" if violations == 0 else "FAIL"

    rows.append(
        {
            "check": name,
            "status": status,
            "violations": int(violations),
            "description": description,
        }
    )

    if violations:
        issues.append(f"{name}: {violations} violation(s).")


def _equal_or_both_missing(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """Compare values while treating paired missing values as equal."""
    return left.eq(right) | (left.isna() & right.isna())


def audit_modeling_panel(
    panel: pd.DataFrame,
    rebalance_calendar: pd.DataFrame,
    ttm_snapshots: pd.DataFrame,
    *,
    expected_horizon_sessions: int,
) -> ModelingPanelAuditResult:
    """Audit point-in-time correctness and target separation."""
    if expected_horizon_sessions < 1:
        raise ModelingPanelAuditError("expected_horizon_sessions must be positive.")

    panel_required = (
        "as_of_date",
        "ticker",
        "sector",
        "technical_latest_market_date",
        *MODEL_FEATURE_COLUMNS,
        *TARGET_METADATA_COLUMNS,
        *TARGET_VALUE_COLUMNS,
        "has_target",
        "sample_role",
    )

    calendar_required = (
        "as_of_date",
        "first_future_date",
        "target_end_date",
        "horizon_sessions",
        "has_full_horizon",
    )

    ttm_required = (
        "as_of_date",
        "ticker",
        "canonical_metric",
        "latest_quarter_end",
        "quarter_count",
        "latest_component_available_date",
    )

    _require_columns(
        panel,
        panel_required,
        dataset_name="Modeling panel",
    )

    _require_columns(
        rebalance_calendar,
        calendar_required,
        dataset_name="Rebalance calendar",
    )

    _require_columns(
        ttm_snapshots,
        ttm_required,
        dataset_name="TTM snapshots",
    )

    frame = panel.copy()

    calendar = rebalance_calendar.loc[
        :,
        calendar_required,
    ].copy()

    ttm = ttm_snapshots.loc[
        :,
        ttm_required,
    ].copy()

    for column in (
        "as_of_date",
        "technical_latest_market_date",
        "first_future_date",
        "target_end_date",
    ):
        _normalize_date(
            frame,
            column,
        )

    for column in (
        "as_of_date",
        "first_future_date",
        "target_end_date",
    ):
        _normalize_date(
            calendar,
            column,
        )

    for column in (
        "as_of_date",
        "latest_quarter_end",
        "latest_component_available_date",
    ):
        _normalize_date(
            ttm,
            column,
        )

    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()

    frame["sector"] = frame["sector"].astype("string").str.strip()

    ttm["ticker"] = ttm["ticker"].astype("string").str.strip().str.upper()

    frame["horizon_sessions"] = pd.to_numeric(
        frame["horizon_sessions"],
        errors="coerce",
    )

    frame["has_target"] = pd.to_numeric(
        frame["has_target"],
        errors="coerce",
    )

    calendar["horizon_sessions"] = pd.to_numeric(
        calendar["horizon_sessions"],
        errors="coerce",
    )

    ttm["quarter_count"] = pd.to_numeric(
        ttm["quarter_count"],
        errors="coerce",
    )

    checks: list[dict[str, Any]] = []

    issues: list[str] = []

    duplicate_panel_rows = int(
        frame.duplicated(
            [
                "as_of_date",
                "ticker",
            ],
            keep=False,
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="unique_panel_keys",
        violations=duplicate_panel_rows,
        description=("Every modeling row must have a unique as_of_date-ticker key."),
    )

    invalid_identifiers = int(
        (
            frame["as_of_date"].isna()
            | frame["ticker"].isna()
            | frame["ticker"].eq("")
            | frame["sector"].isna()
            | frame["sector"].eq("")
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="valid_panel_identifiers",
        violations=invalid_identifiers,
        description=("Dates, tickers and sectors must be valid."),
    )

    technical_future = int(
        (
            frame["technical_latest_market_date"].isna()
            | frame["technical_latest_market_date"].gt(frame["as_of_date"])
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="technical_point_in_time",
        violations=technical_future,
        description=("Technical market information must not extend beyond as_of_date."),
    )

    invalid_has_target = int(
        (
            frame["has_target"].isna()
            | ~frame["has_target"].isin(
                [
                    0,
                    1,
                ]
            )
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="binary_has_target",
        violations=invalid_has_target,
        description=("has_target must contain only 0 or 1."),
    )

    target_value_presence = frame.loc[
        :,
        TARGET_VALUE_COLUMNS,
    ].notna()

    target_metadata_presence = frame.loc[
        :,
        TARGET_METADATA_COLUMNS,
    ].notna()

    all_target_data = target_value_presence.all(axis=1) & target_metadata_presence.all(axis=1)

    any_target_data = target_value_presence.any(axis=1) | target_metadata_presence.any(axis=1)

    has_target = frame["has_target"].eq(1)

    target_tuple_violations = int(
        ((has_target & ~all_target_data) | (~has_target & any_target_data)).sum()
    )

    _record_check(
        checks,
        issues,
        name="complete_target_tuples",
        violations=target_tuple_violations,
        description=("Modeling rows need complete targets and inference rows must contain none."),
    )

    role_violations = int(
        (
            (has_target & frame["sample_role"].ne("modeling"))
            | (~has_target & frame["sample_role"].ne("inference_only"))
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="sample_role_consistency",
        violations=role_violations,
        description=("sample_role must agree with target availability."),
    )

    target_start_violations = int(
        (
            has_target
            & (
                frame["first_future_date"].isna()
                | frame["first_future_date"].le(frame["as_of_date"])
            )
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="target_starts_after_as_of",
        violations=target_start_violations,
        description=("The target must begin strictly after as_of_date."),
    )

    target_end_violations = int(
        (
            has_target
            & (
                frame["target_end_date"].isna()
                | frame["target_end_date"].lt(frame["first_future_date"])
            )
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="valid_target_end",
        violations=target_end_violations,
        description=("The target end must not precede the target start."),
    )

    horizon_violations = int(
        (has_target & frame["horizon_sessions"].ne(expected_horizon_sessions)).sum()
    )

    _record_check(
        checks,
        issues,
        name="target_horizon",
        violations=horizon_violations,
        description=(
            "Every modeling target must use the configured "
            f"{expected_horizon_sessions}-session horizon."
        ),
    )

    metadata_columns = [
        "first_future_date",
        "target_end_date",
        "horizon_sessions",
        "has_target",
    ]

    metadata_nunique = frame.groupby(
        "as_of_date",
        dropna=False,
    )[metadata_columns].nunique(dropna=False)

    inconsistent_date_metadata = int((metadata_nunique > 1).any(axis=1).sum())

    _record_check(
        checks,
        issues,
        name="cross_section_target_consistency",
        violations=inconsistent_date_metadata,
        description=("All companies on the same rebalance date must share target timing metadata."),
    )

    panel_dates = frame.groupby(
        "as_of_date",
        as_index=False,
    ).agg(
        panel_has_target=(
            "has_target",
            "first",
        ),
        panel_first_future_date=(
            "first_future_date",
            "first",
        ),
        panel_target_end_date=(
            "target_end_date",
            "first",
        ),
        panel_horizon_sessions=(
            "horizon_sessions",
            "first",
        ),
    )

    calendar_duplicates = int(
        calendar.duplicated(
            [
                "as_of_date",
            ],
            keep=False,
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="unique_calendar_dates",
        violations=calendar_duplicates,
        description=("The rebalance calendar must contain one row per as_of_date."),
    )

    calendar_compare = calendar.rename(
        columns={
            "first_future_date": ("calendar_first_future_date"),
            "target_end_date": ("calendar_target_end_date"),
            "horizon_sessions": ("calendar_horizon_sessions"),
            "has_full_horizon": ("calendar_has_full_horizon"),
        }
    )

    date_alignment = panel_dates.merge(
        calendar_compare,
        on="as_of_date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_calendar_dates = int(date_alignment["_merge"].ne("both").sum())

    _record_check(
        checks,
        issues,
        name="calendar_date_alignment",
        violations=missing_calendar_dates,
        description=("Every panel date must exist in the rebalance calendar."),
    )

    aligned_dates = date_alignment.loc[date_alignment["_merge"].eq("both")].copy()

    horizon_status_violations = int(
        (
            aligned_dates["panel_has_target"]
            .eq(1)
            .ne(aligned_dates["calendar_has_full_horizon"].astype(bool))
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="calendar_target_availability",
        violations=horizon_status_violations,
        description=("Panel target availability must match the rebalance calendar."),
    )

    modeling_dates = aligned_dates.loc[aligned_dates["panel_has_target"].eq(1)]

    future_date_mismatch = int(
        (
            ~_equal_or_both_missing(
                modeling_dates["panel_first_future_date"],
                modeling_dates["calendar_first_future_date"],
            )
            | ~_equal_or_both_missing(
                modeling_dates["panel_target_end_date"],
                modeling_dates["calendar_target_end_date"],
            )
            | modeling_dates["panel_horizon_sessions"].ne(
                modeling_dates["calendar_horizon_sessions"]
            )
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="calendar_target_metadata",
        violations=future_date_mismatch,
        description=("Stored modeling targets must preserve the calendar timing exactly."),
    )

    forbidden_feature_columns = [
        column
        for column in (MODEL_FEATURE_COLUMNS)
        if (
            "target" in column.lower()
            or "label_top" in column.lower()
            or "future_date" in column.lower()
        )
    ]

    _record_check(
        checks,
        issues,
        name="feature_target_separation",
        violations=len(forbidden_feature_columns),
        description=("Predictor columns must not contain targets or future-return metadata."),
    )

    numeric_features = frame.loc[
        :,
        MODEL_FEATURE_COLUMNS,
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    infinite_features = int(np.isinf(numeric_features.to_numpy(dtype=float)).sum())

    _record_check(
        checks,
        issues,
        name="finite_model_features",
        violations=infinite_features,
        description=("Model features may be missing, but they must never be infinite."),
    )

    duplicate_ttm = int(
        ttm.duplicated(
            [
                "as_of_date",
                "ticker",
                "canonical_metric",
            ],
            keep=False,
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="unique_ttm_snapshots",
        violations=duplicate_ttm,
        description=("TTM snapshots must be unique by date, ticker and metric."),
    )

    ttm_future_availability = int(
        (
            ttm["latest_component_available_date"].isna()
            | ttm["latest_component_available_date"].gt(ttm["as_of_date"])
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="ttm_point_in_time",
        violations=ttm_future_availability,
        description=("All TTM components must have been available by as_of_date."),
    )

    ttm_future_periods = int(
        (ttm["latest_quarter_end"].isna() | ttm["latest_quarter_end"].gt(ttm["as_of_date"])).sum()
    )

    _record_check(
        checks,
        issues,
        name="ttm_period_end",
        violations=ttm_future_periods,
        description=("The latest accounting quarter must not end after as_of_date."),
    )

    invalid_quarter_counts = int((ttm["quarter_count"].isna() | ttm["quarter_count"].ne(4)).sum())

    _record_check(
        checks,
        issues,
        name="ttm_four_quarters",
        violations=invalid_quarter_counts,
        description=("Every TTM observation must contain exactly four quarters."),
    )

    panel_keys = (
        frame.loc[
            :,
            [
                "as_of_date",
                "ticker",
            ],
        ]
        .drop_duplicates()
        .assign(_in_panel=1)
    )

    ttm_keys = ttm.loc[
        :,
        [
            "as_of_date",
            "ticker",
        ],
    ].drop_duplicates()

    ttm_alignment = ttm_keys.merge(
        panel_keys,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    unmatched_ttm_keys = int(ttm_alignment["_in_panel"].isna().sum())

    _record_check(
        checks,
        issues,
        name="ttm_panel_alignment",
        violations=unmatched_ttm_keys,
        description=("Every TTM date-ticker key must belong to the master modeling panel."),
    )

    checks_frame = pd.DataFrame(checks)

    cross_section_sizes = frame.groupby("as_of_date")["ticker"].nunique()

    summary = {
        "panel_rows": len(frame),
        "panel_dates": int(frame["as_of_date"].nunique()),
        "panel_tickers": int(frame["ticker"].nunique()),
        "panel_sectors": int(frame["sector"].nunique()),
        "candidate_model_features": len(MODEL_FEATURE_COLUMNS),
        "modeling_rows": int(has_target.sum()),
        "inference_only_rows": int((~has_target).sum()),
        "minimum_cross_section_size": int(cross_section_sizes.min()),
        "maximum_cross_section_size": int(cross_section_sizes.max()),
        "ttm_rows": len(ttm),
        "ttm_metrics": int(ttm["canonical_metric"].nunique()),
        "audit_checks": len(checks_frame),
        "failed_checks": int(checks_frame["status"].eq("FAIL").sum()),
    }

    return ModelingPanelAuditResult(
        is_valid=not issues,
        summary=summary,
        checks=checks_frame,
        issues=tuple(issues),
    )


def write_modeling_panel_audit_report(
    result: ModelingPanelAuditResult,
    path: Path,
) -> Path:
    """Write the Step 11 modeling-panel quality report."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = "PASS" if result.is_valid else "FAIL"

    summary = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in result.summary.items()
        ]
    )

    lines = [
        "# Modeling Panel Quality Report",
        "",
        "## Status",
        "",
        f"**{status}**",
        "",
        "## Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Leakage and point-in-time checks",
        "",
        "```text",
        result.checks.to_string(index=False),
        "```",
        "",
        "## Blocking issues",
        "",
    ]

    if result.issues:
        lines.extend(f"- {issue}" for issue in result.issues)
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Modeling boundary",
            "",
            (
                "The master panel does not perform "
                "training-sample imputation, fitted scaling, "
                "PCA, model-based feature selection or "
                "hyperparameter optimization."
            ),
            "",
            (
                "Any transformation that learns parameters "
                "from observations must be fitted inside the "
                "training portion of each walk-forward fold."
            ),
            "",
            (
                "Cross-sectional technical and fundamental "
                "scores are date-local transformations and "
                "do not use observations from future dates."
            ),
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path
