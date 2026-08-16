"""Tests for final walk-forward readiness."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from quant_equity.validation import (
    WalkForwardConfig,
    audit_walk_forward_readiness,
    build_walk_forward_folds,
    walk_forward_folds_to_frame,
)


def make_panel(
    periods: int = 20,
) -> pd.DataFrame:
    """Create a synthetic historical panel."""
    dates = pd.date_range(
        "2014-01-31",
        periods=periods,
        freq="ME",
    )

    rows = []

    for date in dates:
        for ticker in (
            "AAA",
            "BBB",
        ):
            rows.append(
                {
                    "as_of_date": date,
                    "ticker": ticker,
                    "target_end_date": (date + pd.Timedelta(days=20)),
                    "has_target": 1,
                }
            )

    return pd.DataFrame(rows)


def make_inputs():
    """Build a valid synthetic readiness bundle."""
    panel = make_panel()

    config = WalkForwardConfig(
        min_train_dates=5,
        validation_dates=2,
        mode="expanding",
    )

    folds = build_walk_forward_folds(
        panel,
        config=config,
    )

    metadata = walk_forward_folds_to_frame(
        panel,
        folds,
        mode=config.mode,
    )

    preprocessing = pd.DataFrame(
        {
            "fold_id": [fold.fold_id for fold in folds],
            "test_date": [fold.test_date for fold in folds],
            "candidate_features": [91] * len(folds),
            "active_features": [91] * len(folds),
            "unavailable_features": [0] * len(folds),
            "train_missing_after": [0] * len(folds),
            "validation_missing_after": [0] * len(folds),
            "test_missing_after": [0] * len(folds),
            "train_non_finite_after": [0] * len(folds),
            "validation_non_finite_after": [0] * len(folds),
            "test_non_finite_after": [0] * len(folds),
            "max_abs_train_scaled_mean": [0.0] * len(folds),
            "max_abs_train_scaled_std_error": [0.0] * len(folds),
        }
    )

    return (
        panel,
        config,
        folds,
        metadata,
        preprocessing,
    )


def run_audit(
    panel=None,
    config=None,
    folds=None,
    metadata=None,
    preprocessing=None,
):
    """Run readiness with default valid inputs."""
    (
        default_panel,
        default_config,
        default_folds,
        default_metadata,
        default_preprocessing,
    ) = make_inputs()

    return audit_walk_forward_readiness(
        (default_panel if panel is None else panel),
        (default_folds if folds is None else folds),
        (default_metadata if metadata is None else metadata),
        (default_preprocessing if preprocessing is None else preprocessing),
        config=(default_config if config is None else config),
    )


def test_valid_walk_forward_is_ready() -> None:
    """A valid walk-forward setup must pass."""
    result = run_audit()

    assert result.is_ready

    assert (result.checks["status"] == "PASS").all()


def test_partition_overlap_fails() -> None:
    """Train and validation dates may never overlap."""
    (
        _,
        _,
        folds,
        _,
        _,
    ) = make_inputs()

    first = folds[0]

    broken = replace(
        first,
        validation_dates=(
            first.train_dates[-1],
            first.validation_dates[-1],
        ),
    )

    broken_folds = (
        broken,
        *folds[1:],
    )

    result = run_audit(folds=broken_folds)

    check = result.checks.set_index("check").loc["disjoint_fold_partitions"]

    assert check["violations"] > 0


def test_unmatured_training_target_fails() -> None:
    """A training label ending after test must fail."""
    (
        panel,
        _,
        folds,
        _,
        _,
    ) = make_inputs()

    first = folds[0]

    train_date = first.train_dates[-1]

    panel.loc[
        panel["as_of_date"].eq(train_date),
        "target_end_date",
    ] = first.test_date + pd.Timedelta(days=5)

    result = run_audit(panel=panel)

    check = result.checks.set_index("check").loc["fitting_label_maturity"]

    assert check["violations"] > 0


def test_duplicate_test_date_fails() -> None:
    """Each OOS month must be tested once."""
    (
        _,
        _,
        folds,
        _,
        _,
    ) = make_inputs()

    second = replace(
        folds[1],
        test_date=folds[0].test_date,
    )

    broken_folds = (
        folds[0],
        second,
        *folds[2:],
    )

    result = run_audit(folds=broken_folds)

    check = result.checks.set_index("check").loc["unique_test_dates"]

    assert check["violations"] > 0


def test_stale_fold_metadata_fails() -> None:
    """Stored fold metadata must match regenerated folds."""
    (
        _,
        _,
        _,
        metadata,
        _,
    ) = make_inputs()

    metadata.loc[
        metadata.index[0],
        "test_rows",
    ] = 999

    result = run_audit(metadata=metadata)

    check = result.checks.set_index("check").loc["stored_fold_metadata_alignment"]

    assert check["violations"] == 1


def test_missing_after_preprocessing_fails() -> None:
    """No missing values may remain after transformation."""
    (
        _,
        _,
        _,
        _,
        preprocessing,
    ) = make_inputs()

    preprocessing.loc[
        preprocessing.index[0],
        "test_missing_after",
    ] = 1

    result = run_audit(preprocessing=preprocessing)

    check = result.checks.set_index("check").loc["no_missing_after_preprocessing"]

    assert check["violations"] == 1


def test_bad_training_scaling_fails() -> None:
    """Incorrect training scaling must be detected."""
    (
        _,
        _,
        _,
        _,
        preprocessing,
    ) = make_inputs()

    preprocessing.loc[
        preprocessing.index[0],
        "max_abs_train_scaled_mean",
    ] = 0.5

    result = run_audit(preprocessing=preprocessing)

    check = result.checks.set_index("check").loc["training_scaling_contract"]

    assert check["violations"] == 1
