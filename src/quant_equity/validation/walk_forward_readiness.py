"""Final structural audit for definitive walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_equity.models.modeling_panel import (
    MODEL_FEATURE_COLUMNS,
)
from quant_equity.validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardFold,
    split_panel_by_fold,
    walk_forward_folds_to_frame,
)


class WalkForwardReadinessError(ValueError):
    """Raised when the final walk-forward audit cannot run."""


@dataclass(frozen=True)
class WalkForwardReadinessResult:
    """Result of the Step 12C walk-forward audit."""

    is_ready: bool
    summary: dict[str, Any]
    checks: pd.DataFrame
    issues: tuple[str, ...]


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


def _require_columns(
    data: pd.DataFrame,
    required: tuple[str, ...],
    *,
    name: str,
) -> None:
    """Require columns used by the audit."""
    missing = sorted(set(required).difference(data.columns))

    if missing:
        raise WalkForwardReadinessError(f"{name} is missing columns: " + ", ".join(missing) + ".")


def audit_walk_forward_readiness(
    panel: pd.DataFrame,
    folds: tuple[
        WalkForwardFold,
        ...,
    ],
    fold_metadata: pd.DataFrame,
    preprocessing_audit: pd.DataFrame,
    *,
    config: WalkForwardConfig,
    tolerance: float = 1.0e-10,
) -> WalkForwardReadinessResult:
    """Audit the complete Step 12 walk-forward contract."""
    if tolerance <= 0:
        raise WalkForwardReadinessError("tolerance must be positive.")

    _require_columns(
        panel,
        (
            "as_of_date",
            "ticker",
            "target_end_date",
            "has_target",
        ),
        name="Modeling panel",
    )

    _require_columns(
        fold_metadata,
        (
            "fold_id",
            "mode",
            "train_start_date",
            "train_end_date",
            "validation_start_date",
            "validation_end_date",
            "test_date",
            "train_date_count",
            "validation_date_count",
            "purged_date_count",
            "test_rows",
        ),
        name="Fold metadata",
    )

    _require_columns(
        preprocessing_audit,
        (
            "fold_id",
            "test_date",
            "candidate_features",
            "active_features",
            "unavailable_features",
            "train_missing_after",
            "validation_missing_after",
            "test_missing_after",
            "train_non_finite_after",
            "validation_non_finite_after",
            "test_non_finite_after",
            "max_abs_train_scaled_mean",
            "max_abs_train_scaled_std_error",
        ),
        name="Preprocessing audit",
    )

    frame = panel.copy()

    frame["as_of_date"] = pd.to_datetime(
        frame["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    frame["target_end_date"] = pd.to_datetime(
        frame["target_end_date"],
        errors="coerce",
    ).dt.normalize()

    metadata = fold_metadata.copy()

    preprocessing = preprocessing_audit.copy()

    date_columns = (
        "train_start_date",
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
        "test_date",
    )

    for column in date_columns:
        if column in metadata:
            metadata[column] = pd.to_datetime(
                metadata[column],
                errors="coerce",
            ).dt.normalize()

    preprocessing["test_date"] = pd.to_datetime(
        preprocessing["test_date"],
        errors="coerce",
    ).dt.normalize()

    checks: list[dict[str, Any]] = []

    issues: list[str] = []

    # 1. Folds must exist.
    _record_check(
        checks,
        issues,
        name="folds_exist",
        violations=int(len(folds) == 0),
        description=("At least one definitive walk-forward fold must exist."),
    )

    fold_ids = [fold.fold_id for fold in folds]

    # 2. Fold IDs must be sequential.
    expected_ids = list(
        range(
            1,
            len(folds) + 1,
        )
    )

    sequential_id_violations = int(fold_ids != expected_ids)

    _record_check(
        checks,
        issues,
        name="sequential_fold_ids",
        violations=sequential_id_violations,
        description=("Fold IDs must be sequential and deterministic."),
    )

    test_dates = [pd.Timestamp(fold.test_date) for fold in folds]

    # 3. Every test date must appear once.
    duplicated_test_dates = len(test_dates) - len(set(test_dates))

    _record_check(
        checks,
        issues,
        name="unique_test_dates",
        violations=duplicated_test_dates,
        description=("Every out-of-sample month must be tested exactly once."),
    )

    # 4. Test months must move strictly forward.
    increasing_violations = sum(
        int(current <= previous)
        for previous, current in zip(
            test_dates,
            test_dates[1:],
            strict=False,
        )
    )

    _record_check(
        checks,
        issues,
        name="strictly_increasing_test_dates",
        violations=increasing_violations,
        description=("Out-of-sample test dates must advance strictly through time."),
    )

    modeling = frame.loc[frame["has_target"].eq(1)].copy()

    modeling_dates = [pd.Timestamp(date) for date in sorted(modeling["as_of_date"].unique())]

    # 5. Once OOS evaluation starts, no completed month is skipped.
    if test_dates:
        expected_test_dates = [date for date in modeling_dates if date >= test_dates[0]]

        test_coverage_violations = len(set(expected_test_dates) - set(test_dates)) + len(
            set(test_dates) - set(expected_test_dates)
        )
    else:
        test_coverage_violations = 1

    _record_check(
        checks,
        issues,
        name="complete_oos_date_coverage",
        violations=test_coverage_violations,
        description=(
            "Every completed modeling month from the first OOS date onward must be tested."
        ),
    )

    partition_overlap_violations = 0
    chronological_violations = 0
    minimum_train_violations = 0
    validation_size_violations = 0
    maturity_violations = 0
    target_availability_violations = 0
    test_cross_section_violations = 0

    cross_section_sizes = modeling.groupby("as_of_date")["ticker"].nunique()

    expected_cross_section_size = int(cross_section_sizes.max())

    for fold in folds:
        train_dates = set(fold.train_dates)

        validation_dates = set(fold.validation_dates)

        test_set = {fold.test_date}

        # 6. No date may exist in two partitions.
        if (
            not train_dates.isdisjoint(validation_dates)
            or not train_dates.isdisjoint(test_set)
            or not validation_dates.isdisjoint(test_set)
        ):
            partition_overlap_violations += 1

        # 7. Train < validation < test.
        if (
            not fold.train_dates
            or not fold.validation_dates
            or fold.train_end_date >= fold.validation_start_date
            or fold.validation_end_date >= fold.test_date
        ):
            chronological_violations += 1

        # 8. Minimum training history.
        if len(fold.train_dates) < config.min_train_dates:
            minimum_train_violations += 1

        # 9. Validation length.
        if len(fold.validation_dates) != config.validation_dates:
            validation_size_violations += 1

        train, validation, test = split_panel_by_fold(
            frame,
            fold,
        )

        # 10. Test month must contain full universe.
        if (
            len(test) != expected_cross_section_size
            or test["ticker"].nunique() != expected_cross_section_size
        ):
            test_cross_section_violations += 1

        # 11. Test observations must have known labels
        # because this is historical OOS evaluation.
        if not test["has_target"].eq(1).all():
            target_availability_violations += 1

        # 12. Every fitting label must already be known.
        if (
            train["target_end_date"].gt(fold.test_date).any()
            or validation["target_end_date"].gt(fold.test_date).any()
        ):
            maturity_violations += 1

    _record_check(
        checks,
        issues,
        name="disjoint_fold_partitions",
        violations=partition_overlap_violations,
        description=(
            "Train, validation and test dates must be mutually exclusive inside each fold."
        ),
    )

    _record_check(
        checks,
        issues,
        name="chronological_partition_order",
        violations=chronological_violations,
        description=("Every fold must satisfy train < validation < test."),
    )

    _record_check(
        checks,
        issues,
        name="minimum_training_history",
        violations=minimum_train_violations,
        description=("Every fold must satisfy the configured minimum training history."),
    )

    _record_check(
        checks,
        issues,
        name="validation_window_size",
        violations=validation_size_violations,
        description=("Every fold must contain the configured number of validation dates."),
    )

    _record_check(
        checks,
        issues,
        name="complete_test_cross_sections",
        violations=test_cross_section_violations,
        description=("Every OOS month must contain the complete security universe."),
    )

    _record_check(
        checks,
        issues,
        name="historical_test_targets_exist",
        violations=target_availability_violations,
        description=("Historical OOS test rows must have completed targets for evaluation."),
    )

    _record_check(
        checks,
        issues,
        name="fitting_label_maturity",
        violations=maturity_violations,
        description=("Train and validation targets must be fully known by the test date."),
    )

    # 13. Expanding history may never forget an old train date.
    expanding_violations = 0

    if config.mode == "expanding" and folds:
        first_train_start = folds[0].train_start_date

        for index, fold in enumerate(folds):
            if fold.train_start_date != first_train_start:
                expanding_violations += 1

            if index > 0:
                previous_train = set(folds[index - 1].train_dates)

                current_train = set(fold.train_dates)

                if not previous_train.issubset(current_train):
                    expanding_violations += 1

    _record_check(
        checks,
        issues,
        name="expanding_training_history",
        violations=expanding_violations,
        description=("Expanding mode must preserve previously eligible training history."),
    )

    # 14. Stored 12A metadata must match regenerated folds.
    regenerated_metadata = walk_forward_folds_to_frame(
        frame,
        folds,
        mode=config.mode,
    )

    metadata_core = (
        "fold_id",
        "mode",
        "train_start_date",
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
        "test_date",
        "train_date_count",
        "validation_date_count",
        "purged_date_count",
        "test_rows",
    )

    stored_core = (
        metadata.loc[
            :,
            metadata_core,
        ]
        .sort_values("fold_id")
        .reset_index(drop=True)
    )

    expected_core = (
        regenerated_metadata.loc[
            :,
            metadata_core,
        ]
        .sort_values("fold_id")
        .reset_index(drop=True)
    )

    for column in date_columns:
        if column in stored_core:
            stored_core[column] = pd.to_datetime(stored_core[column]).dt.normalize()

            expected_core[column] = pd.to_datetime(expected_core[column]).dt.normalize()

    metadata_alignment_violations = int(
        len(stored_core) != len(expected_core) or not stored_core.equals(expected_core)
    )

    _record_check(
        checks,
        issues,
        name="stored_fold_metadata_alignment",
        violations=metadata_alignment_violations,
        description=(
            "Persisted Step 12A fold metadata must match the regenerated walk-forward contract."
        ),
    )

    # 15. Preprocessing audit must correspond one-to-one with folds.
    expected_fold_map = pd.DataFrame(
        {
            "fold_id": [fold.fold_id for fold in folds],
            "test_date": [fold.test_date for fold in folds],
        }
    )

    stored_fold_map = (
        preprocessing[
            [
                "fold_id",
                "test_date",
            ]
        ]
        .sort_values("fold_id")
        .reset_index(drop=True)
    )

    expected_fold_map = expected_fold_map.sort_values("fold_id").reset_index(drop=True)

    preprocessing_alignment_violations = int(
        len(stored_fold_map) != len(expected_fold_map)
        or not stored_fold_map.equals(expected_fold_map)
    )

    _record_check(
        checks,
        issues,
        name="preprocessing_fold_alignment",
        violations=preprocessing_alignment_violations,
        description=("Step 12B preprocessing results must align one-to-one with the OOS folds."),
    )

    # 16. Candidate/active feature contract.
    expected_feature_count = len(MODEL_FEATURE_COLUMNS)

    feature_contract_violations = int(
        (
            preprocessing["candidate_features"].ne(expected_feature_count)
            | (preprocessing["active_features"] + preprocessing["unavailable_features"]).ne(
                expected_feature_count
            )
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="preprocessing_feature_contract",
        violations=feature_contract_violations,
        description=(
            "Every fold must account for all candidate features using training-only availability."
        ),
    )

    # 17. No missing transformed values.
    missing_after = int(
        preprocessing[
            [
                "train_missing_after",
                "validation_missing_after",
                "test_missing_after",
            ]
        ]
        .sum()
        .sum()
    )

    _record_check(
        checks,
        issues,
        name="no_missing_after_preprocessing",
        violations=missing_after,
        description=("No active predictor may remain missing after fold-local preprocessing."),
    )

    # 18. No infinite transformed values.
    non_finite_after = int(
        preprocessing[
            [
                "train_non_finite_after",
                "validation_non_finite_after",
                "test_non_finite_after",
            ]
        ]
        .sum()
        .sum()
    )

    _record_check(
        checks,
        issues,
        name="finite_preprocessed_features",
        violations=non_finite_after,
        description=("Preprocessed features must not contain positive or negative infinity."),
    )

    # 19. Training scaling must reconstruct mean 0/std 1.
    scaling_violations = int(
        (
            preprocessing["max_abs_train_scaled_mean"].gt(tolerance)
            | preprocessing["max_abs_train_scaled_std_error"].gt(tolerance)
        ).sum()
    )

    _record_check(
        checks,
        issues,
        name="training_scaling_contract",
        violations=scaling_violations,
        description=(
            "Non-constant continuous training "
            "features must be centered and scaled "
            "using training-only statistics."
        ),
    )

    checks_frame = pd.DataFrame(checks)

    summary = {
        "folds": len(folds),
        "mode": config.mode,
        "minimum_train_dates": (config.min_train_dates),
        "validation_dates": (config.validation_dates),
        "first_test_date": (test_dates[0] if test_dates else None),
        "last_test_date": (test_dates[-1] if test_dates else None),
        "oos_test_dates": len(set(test_dates)),
        "expected_cross_section_size": (expected_cross_section_size),
        "candidate_features": (expected_feature_count),
        "minimum_active_features": int(preprocessing["active_features"].min()),
        "maximum_active_features": int(preprocessing["active_features"].max()),
        "maximum_purged_dates": int(metadata["purged_date_count"].max()),
        "readiness_checks": len(checks_frame),
        "failed_readiness_checks": int(checks_frame["status"].eq("FAIL").sum()),
    }

    return WalkForwardReadinessResult(
        is_ready=not issues,
        summary=summary,
        checks=checks_frame,
        issues=tuple(issues),
    )


def write_walk_forward_readiness_report(
    result: WalkForwardReadinessResult,
    config: WalkForwardConfig,
    path: Path,
) -> Path:
    """Write the definitive Step 12 walk-forward validation report."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = "READY FOR MODEL TRAINING" if result.is_ready else "NOT READY"

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
        "# Definitive Walk-Forward Validation",
        "",
        "## Step",
        "",
        ("Step 12D — Final documentation and approval of the definitive walk-forward protocol."),
        "",
        "## Status",
        "",
        f"**{status}**",
        "",
        "## Configuration",
        "",
        "```text",
        f"mode: {config.mode}",
        f"minimum training dates: {config.min_train_dates}",
        f"validation dates: {config.validation_dates}",
        "test window: 1 monthly cross-section",
        "```",
        "",
        "## Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Temporal protocol",
        "",
        ("- Every security from the same as_of_date remains in the same partition."),
        ("- Training always precedes validation and validation always precedes test."),
        ("- The test set contains exactly one historical monthly cross-section."),
        (
            "- Historical observations whose forward "
            "21-session target had not matured by the "
            "test date are purged from fitting."
        ),
        ("- Every historical OOS month from the first test date onward is evaluated exactly once."),
        "",
        "## Preprocessing protocol",
        "",
        ("- Feature availability is determined using training data only."),
        ("- Missing-value imputation parameters are estimated using training data only."),
        ("- Scaling parameters are estimated using training data only."),
        ("- Validation and test data are transformed without refitting preprocessing parameters."),
        ("- Missing-indicator features retain their binary 0/1 interpretation."),
        "",
        "## Hyperparameter rule",
        "",
        (
            "Hyperparameters and model-selection decisions "
            "must use training and validation information "
            "only. The monthly test cross-section must "
            "remain untouched until final OOS prediction."
        ),
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
            "## Artifacts",
            "",
            ("- `data/processed/modeling_panel.parquet`"),
            ("- `data/processed/walk_forward_folds.parquet`"),
            ("- `data/processed/walk_forward_preprocessing.parquet`"),
            ("- `reports/tables/walk_forward_preprocessing_audit.csv`"),
            ("- `reports/tables/walk_forward_readiness_checks.csv`"),
            "",
            "## Step 12 conclusion",
            "",
            ("The definitive validation framework is approved for out-of-sample model training."),
            "",
            (
                "All models evaluated from Step 13 onward "
                "must consume this temporal protocol rather "
                "than creating independent train/test splits."
            ),
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path
