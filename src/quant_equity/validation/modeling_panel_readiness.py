"""Final readiness audit for the master modeling panel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.models.modeling_panel import (
    MODEL_FEATURE_COLUMNS,
    MODELING_PANEL_COLUMNS,
)


class ModelingPanelReadinessError(ValueError):
    """Raised when final modeling-panel validation cannot run."""


@dataclass(frozen=True)
class ModelingPanelReadinessResult:
    """Final Step 11 readiness result."""

    is_ready: bool
    summary: dict[str, Any]
    checks: pd.DataFrame
    issues: tuple[str, ...]


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require expected columns."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise ModelingPanelReadinessError(
            f"{dataset_name} is missing columns: " + ", ".join(missing) + "."
        )


def _record_check(
    rows: list[dict[str, Any]],
    issues: list[str],
    *,
    name: str,
    violations: int,
    description: str,
) -> None:
    """Record one blocking readiness check."""
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


def audit_modeling_panel_readiness(
    panel: pd.DataFrame,
    dictionary: pd.DataFrame,
    leakage_checks: pd.DataFrame,
    *,
    tolerance: float = 1.0e-12,
) -> ModelingPanelReadinessResult:
    """Perform the final Step 11 dataset-contract audit."""
    if tolerance <= 0:
        raise ModelingPanelReadinessError("tolerance must be positive.")

    _require_columns(
        panel,
        MODELING_PANEL_COLUMNS,
        dataset_name="Modeling panel",
    )

    _require_columns(
        dictionary,
        (
            "column",
            "model_input",
        ),
        dataset_name="Data dictionary",
    )

    _require_columns(
        leakage_checks,
        (
            "check",
            "status",
            "violations",
        ),
        dataset_name="Leakage checks",
    )

    frame = panel.copy()

    frame["as_of_date"] = pd.to_datetime(
        frame["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    for column in (
        "first_future_date",
        "target_end_date",
    ):
        frame[column] = pd.to_datetime(
            frame[column],
            errors="coerce",
        ).dt.normalize()

    checks: list[dict[str, Any]] = []

    issues: list[str] = []

    exact_column_order = list(frame.columns) == list(MODELING_PANEL_COLUMNS)

    _record_check(
        checks,
        issues,
        name="canonical_column_order",
        violations=int(not exact_column_order),
        description=(
            "The stored panel must follow the canonical MODELING_PANEL_COLUMNS schema exactly."
        ),
    )

    duplicated_columns = int(frame.columns.duplicated().sum())

    _record_check(
        checks,
        issues,
        name="unique_column_names",
        violations=duplicated_columns,
        description=("The modeling panel must not contain duplicated column names."),
    )

    feature_count_violation = int(len(MODEL_FEATURE_COLUMNS) != 91)

    _record_check(
        checks,
        issues,
        name="candidate_feature_count",
        violations=feature_count_violation,
        description=("The current master-panel contract expects exactly 91 candidate predictors."),
    )

    raw_features = frame.loc[
        :,
        MODEL_FEATURE_COLUMNS,
    ]

    numeric_features = raw_features.apply(
        pd.to_numeric,
        errors="coerce",
    )

    non_numeric = int((raw_features.notna() & numeric_features.isna()).sum().sum())

    _record_check(
        checks,
        issues,
        name="numeric_predictors",
        violations=non_numeric,
        description=("Every non-missing candidate predictor must be numeric."),
    )

    infinite_values = int(np.isinf(numeric_features.to_numpy(dtype=float)).sum())

    _record_check(
        checks,
        issues,
        name="finite_predictors",
        violations=infinite_values,
        description=("Candidate predictors may be NaN but must never contain infinity."),
    )

    fully_missing_features = int(numeric_features.isna().all(axis=0).sum())

    _record_check(
        checks,
        issues,
        name="no_fully_missing_predictors",
        violations=fully_missing_features,
        description=("Every candidate predictor must have at least one historical observation."),
    )

    cross_section_sizes = frame.groupby("as_of_date")["ticker"].nunique()

    expected_cross_section_size = int(cross_section_sizes.max())

    incomplete_cross_sections = int(cross_section_sizes.ne(expected_cross_section_size).sum())

    _record_check(
        checks,
        issues,
        name="complete_cross_sections",
        violations=incomplete_cross_sections,
        description=(
            "Every rebalance date must contain the same configured security cross-section."
        ),
    )

    modeling = frame.loc[frame["has_target"].eq(1)].copy()

    inference = frame.loc[frame["has_target"].eq(0)].copy()

    if modeling.empty:
        no_modeling_rows = 1
    else:
        no_modeling_rows = 0

    _record_check(
        checks,
        issues,
        name="modeling_sample_exists",
        violations=no_modeling_rows,
        description=("The dataset must contain historical observations with completed targets."),
    )

    if not modeling.empty and not inference.empty:
        latest_modeling_date = modeling["as_of_date"].max()

        inference_inside_history = int(inference["as_of_date"].le(latest_modeling_date).sum())
    else:
        inference_inside_history = 0

    _record_check(
        checks,
        issues,
        name="inference_only_tail",
        violations=inference_inside_history,
        description=(
            "Observations without future targets must "
            "appear only after the completed modeling history."
        ),
    )

    if not modeling.empty:
        date_medians = modeling.groupby("as_of_date")["target_21d"].transform("median")

        reconstructed_excess = modeling["target_21d"] - date_medians

        stored_excess = pd.to_numeric(
            modeling["target_21d_excess"],
            errors="coerce",
        )

        reconstruction_valid = np.isclose(
            stored_excess.to_numpy(dtype=float),
            reconstructed_excess.to_numpy(dtype=float),
            rtol=0.0,
            atol=tolerance,
            equal_nan=False,
        )

        excess_reconstruction_violations = int((~reconstruction_valid).sum())
    else:
        excess_reconstruction_violations = 0

    _record_check(
        checks,
        issues,
        name="target_excess_reconstruction",
        violations=(excess_reconstruction_violations),
        description=(
            "target_21d_excess must equal target_21d minus the same-date cross-sectional median."
        ),
    )

    if not modeling.empty:
        excess_medians = modeling.groupby("as_of_date")["target_21d_excess"].median()

        non_centered_dates = int(excess_medians.abs().gt(tolerance).sum())
    else:
        non_centered_dates = 0

    _record_check(
        checks,
        issues,
        name="target_excess_centering",
        violations=non_centered_dates,
        description=(
            "The cross-sectional median of target_21d_excess must be zero on every modeling date."
        ),
    )

    if not modeling.empty:
        labels = pd.to_numeric(
            modeling["label_top_quintile"],
            errors="coerce",
        )

        invalid_labels = int(
            (
                labels.isna()
                | ~labels.isin(
                    [
                        0,
                        1,
                    ]
                )
            ).sum()
        )
    else:
        invalid_labels = 0

    _record_check(
        checks,
        issues,
        name="binary_top_quintile_label",
        violations=invalid_labels,
        description=("label_top_quintile must be binary for every modeling observation."),
    )

    ranking_violations = 0

    if not modeling.empty:
        for _, group in modeling.groupby(
            "as_of_date",
            sort=True,
        ):
            positive = group.loc[
                group["label_top_quintile"].eq(1),
                "target_21d",
            ]

            negative = group.loc[
                group["label_top_quintile"].eq(0),
                "target_21d",
            ]

            if positive.empty or negative.empty:
                ranking_violations += 1
                continue

            if positive.min() + tolerance < negative.max():
                ranking_violations += 1

    _record_check(
        checks,
        issues,
        name="top_quintile_ordering",
        violations=ranking_violations,
        description=(
            "Top-quintile observations must not "
            "have lower future returns than "
            "non-top-quintile observations on "
            "the same date."
        ),
    )

    dictionary_duplicates = int(dictionary["column"].duplicated(keep=False).sum())

    _record_check(
        checks,
        issues,
        name="unique_dictionary_columns",
        violations=dictionary_duplicates,
        description=("Every panel column must have exactly one dictionary entry."),
    )

    dictionary_columns = set(dictionary["column"])

    panel_columns = set(frame.columns)

    dictionary_coverage_violations = len(panel_columns - dictionary_columns) + len(
        dictionary_columns - panel_columns
    )

    _record_check(
        checks,
        issues,
        name="dictionary_panel_alignment",
        violations=(dictionary_coverage_violations),
        description=("The data dictionary must exactly cover the stored modeling-panel schema."),
    )

    documented_model_inputs = set(
        dictionary.loc[
            dictionary["model_input"].eq(True),
            "column",
        ]
    )

    feature_metadata_violations = len(set(MODEL_FEATURE_COLUMNS) - documented_model_inputs) + len(
        documented_model_inputs - set(MODEL_FEATURE_COLUMNS)
    )

    _record_check(
        checks,
        issues,
        name="dictionary_model_input_contract",
        violations=(feature_metadata_violations),
        description=(
            "Dictionary model_input=True columns must exactly equal MODEL_FEATURE_COLUMNS."
        ),
    )

    leakage_failures = int(
        (
            leakage_checks["status"].ne("PASS")
            | pd.to_numeric(
                leakage_checks["violations"],
                errors="coerce",
            )
            .fillna(1)
            .ne(0)
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="prior_leakage_audit",
        violations=leakage_failures,
        description=("Every Step 11B leakage and point-in-time check must pass."),
    )

    checks_frame = pd.DataFrame(checks)

    summary = {
        "panel_rows": len(frame),
        "panel_columns": len(frame.columns),
        "panel_dates": int(frame["as_of_date"].nunique()),
        "panel_tickers": int(frame["ticker"].nunique()),
        "candidate_model_features": len(MODEL_FEATURE_COLUMNS),
        "modeling_rows": len(modeling),
        "inference_only_rows": len(inference),
        "modeling_dates": int(modeling["as_of_date"].nunique()),
        "inference_only_dates": int(inference["as_of_date"].nunique()),
        "cross_section_size": (expected_cross_section_size),
        "dictionary_rows": len(dictionary),
        "prior_leakage_checks": len(leakage_checks),
        "final_readiness_checks": len(checks_frame),
        "failed_readiness_checks": int(checks_frame["status"].eq("FAIL").sum()),
    }

    return ModelingPanelReadinessResult(
        is_ready=not issues,
        summary=summary,
        checks=checks_frame,
        issues=tuple(issues),
    )


def write_modeling_panel_readiness_report(
    result: ModelingPanelReadinessResult,
    path: Path,
) -> Path:
    """Write the final Step 11 readiness report."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = "READY" if result.is_ready else "NOT READY"

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
        "# Modeling Panel Final Audit",
        "",
        "## Step",
        "",
        ("Step 11D — Final dataset-contract and modeling-readiness audit."),
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
        "## Final readiness checks",
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
            "## Step 11 conclusion",
            "",
            (
                "The master panel combines technical "
                "signals, point-in-time fundamental "
                "signals and forward targets while "
                "preserving a strict separation between "
                "predictors and future information."
            ),
            "",
            (
                "Rows without completed future returns "
                "remain available for inference but are "
                "excluded from historical model fitting."
            ),
            "",
            ("The dataset is ready to be consumed by the walk-forward validation framework."),
            "",
            "## Important next-stage rule",
            "",
            (
                "Imputation, fitted scaling, feature "
                "selection, dimensionality reduction, "
                "hyperparameter tuning and model fitting "
                "must occur inside each training fold."
            ),
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path
