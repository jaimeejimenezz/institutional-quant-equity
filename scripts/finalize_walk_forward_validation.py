"""Finalize and approve the definitive walk-forward framework."""

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
from quant_equity.validation import (
    WalkForwardConfig,
    WalkForwardReadinessError,
    audit_walk_forward_readiness,
    build_walk_forward_folds,
    write_walk_forward_readiness_report,
)

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_panel.parquet"

FOLD_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "walk_forward_folds.parquet"

PREPROCESSING_PATH = PROJECT_ROOT / "data" / "processed" / "walk_forward_preprocessing.parquet"

CHECKS_PATH = REPORTS_DIR / "tables" / "walk_forward_readiness_checks.csv"

REPORT_PATH = REPORTS_DIR / "validation" / "walk_forward_validation_report.md"


def main() -> None:
    """Execute Step 12D."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        PANEL_PATH,
        FOLD_METADATA_PATH,
        PREPROCESSING_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    panel = pd.read_parquet(PANEL_PATH)

    metadata = pd.read_parquet(FOLD_METADATA_PATH)

    preprocessing = pd.read_parquet(PREPROCESSING_PATH)

    config = WalkForwardConfig(
        min_train_dates=60,
        validation_dates=12,
        mode="expanding",
    )

    folds = build_walk_forward_folds(
        panel,
        config=config,
    )

    result = audit_walk_forward_readiness(
        panel,
        folds,
        metadata,
        preprocessing,
        config=config,
    )

    CHECKS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.checks.to_csv(
        CHECKS_PATH,
        index=False,
    )

    write_walk_forward_readiness_report(
        result,
        config,
        REPORT_PATH,
    )

    logger.info("Definitive walk-forward validation finalized.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Definitive walk-forward validation - Step 12D")
    print("------------------------------------------------")

    for key, value in result.summary.items():
        if isinstance(
            value,
            pd.Timestamp,
        ):
            value = value.date()

        print(f"{key}: {value}")

    print()
    print(f"Checks table: {CHECKS_PATH}")

    print(f"Final report: {REPORT_PATH}")

    print()

    if not result.is_ready:
        print("Walk-forward framework: NOT READY")

        for issue in result.issues:
            print(f"- {issue}")

        raise WalkForwardReadinessError("Definitive walk-forward validation failed.")

    print("Walk-forward framework: READY FOR MODEL TRAINING")

    print()


if __name__ == "__main__":
    main()
