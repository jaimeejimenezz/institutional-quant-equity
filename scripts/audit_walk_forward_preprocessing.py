"""Audit leakage-safe preprocessing across all walk-forward folds."""

from __future__ import annotations

import logging

import pandas as pd

from quant_equity.config import (
    PROJECT_ROOT,
    REPORTS_DIR,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.models.modeling_panel import (
    MODEL_FEATURE_COLUMNS,
)
from quant_equity.validation import (
    WalkForwardConfig,
    audit_fold_preprocessing,
    build_walk_forward_folds,
)


PANEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_panel.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "walk_forward_preprocessing.parquet"
)

TABLE_PATH = (
    REPORTS_DIR
    / "tables"
    / "walk_forward_preprocessing_audit.csv"
)


def main() -> None:
    """Execute Step 12B."""
    configure_logging()

    logger = logging.getLogger(
        "quant_equity"
    )

    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Modeling panel not found: {PANEL_PATH}"
        )

    panel = pd.read_parquet(
        PANEL_PATH
    )

    config = WalkForwardConfig(
        min_train_dates=60,
        validation_dates=12,
        mode="expanding",
    )

    folds = build_walk_forward_folds(
        panel,
        config=config,
    )

    audit = audit_fold_preprocessing(
        panel,
        folds,
        feature_columns=MODEL_FEATURE_COLUMNS,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    audit.to_csv(
        TABLE_PATH,
        index=False,
    )

    transformed_missing = int(
        audit[
            [
                "train_missing_after",
                "validation_missing_after",
                "test_missing_after",
            ]
        ]
        .sum()
        .sum()
    )

    transformed_non_finite = int(
        audit[
            [
                "train_non_finite_after",
                "validation_non_finite_after",
                "test_non_finite_after",
            ]
        ]
        .sum()
        .sum()
    )

    logger.info(
        "Walk-forward preprocessing audit completed."
    )

    print()
    print(
        "Institutional Quant Equity Research Platform"
    )
    print(
        "Fold-local preprocessing audit - Step 12B"
    )
    print(
        "------------------------------------------------"
    )

    print(
        f"folds: {len(audit)}"
    )

    print(
        "candidate_features: "
        f"{len(MODEL_FEATURE_COLUMNS)}"
    )

    print(
        "minimum_active_features: "
        f"{audit['active_features'].min()}"
    )

    print(
        "maximum_active_features: "
        f"{audit['active_features'].max()}"
    )

    print(
        "folds_with_unavailable_features: "
        f"{audit['unavailable_features'].gt(0).sum()}"
    )

    print(
        "maximum_unavailable_features: "
        f"{audit['unavailable_features'].max()}"
    )

    print(
        "maximum_constant_continuous_features: "
        f"{audit['constant_continuous_features'].max()}"
    )

    print(
        "total_train_missing_before: "
        f"{audit['train_missing_before'].sum()}"
    )

    print(
        "total_validation_missing_before: "
        f"{audit['validation_missing_before'].sum()}"
    )

    print(
        "total_test_missing_before: "
        f"{audit['test_missing_before'].sum()}"
    )

    print(
        "transformed_missing_values: "
        f"{transformed_missing}"
    )

    print(
        "transformed_non_finite_values: "
        f"{transformed_non_finite}"
    )

    print(
        "max_abs_train_scaled_mean: "
        f"{audit['max_abs_train_scaled_mean'].max():.3e}"
    )

    print(
        "max_abs_train_scaled_std_error: "
        f"{audit['max_abs_train_scaled_std_error'].max():.3e}"
    )

    print()
    print(
        "First folds:"
    )

    preview = [
        "fold_id",
        "test_date",
        "active_features",
        "unavailable_features",
        "constant_continuous_features",
        "train_missing_before",
        "validation_missing_before",
        "test_missing_before",
        "test_missing_after",
    ]

    print(
        audit[
            preview
        ]
        .head()
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Parquet: {OUTPUT_PATH}"
    )

    print(
        f"CSV: {TABLE_PATH}"
    )

    if transformed_missing != 0:
        raise RuntimeError(
            "Missing values remain after preprocessing."
        )

    if transformed_non_finite != 0:
        raise RuntimeError(
            "Non-finite values remain after preprocessing."
        )

    print()
    print(
        "Fold-local preprocessing: OK"
    )


if __name__ == "__main__":
    main()