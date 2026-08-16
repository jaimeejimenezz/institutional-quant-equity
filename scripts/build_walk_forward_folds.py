"""Build the definitive purged walk-forward fold calendar."""

from __future__ import annotations

import logging

import pandas as pd

from quant_equity.config import (
    PROJECT_ROOT,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.validation import (
    WalkForwardConfig,
    build_walk_forward_folds,
    walk_forward_folds_to_frame,
)

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_panel.parquet"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "walk_forward_folds.parquet"


def main() -> None:
    """Execute Step 12A."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Modeling panel not found: {PANEL_PATH}")

    panel = pd.read_parquet(PANEL_PATH)

    config = WalkForwardConfig(
        min_train_dates=60,
        validation_dates=12,
        mode="expanding",
    )

    folds = build_walk_forward_folds(
        panel,
        config=config,
    )

    if not folds:
        raise RuntimeError("No walk-forward folds could be generated.")

    metadata = walk_forward_folds_to_frame(
        panel,
        folds,
        mode=config.mode,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    logger.info("Walk-forward fold generation completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Purged walk-forward validation - Step 12A")
    print("------------------------------------------------")

    print(f"folds: {len(metadata)}")

    print(f"mode: {config.mode}")

    print(f"minimum_train_dates: {config.min_train_dates}")

    print(f"validation_dates: {config.validation_dates}")

    print(f"first_test_date: {metadata['test_date'].min().date()}")

    print(f"last_test_date: {metadata['test_date'].max().date()}")

    print(f"minimum_train_date_count: {metadata['train_date_count'].min()}")

    print(f"maximum_train_date_count: {metadata['train_date_count'].max()}")

    print(f"validation_date_count: {metadata['validation_date_count'].min()}")

    print(f"minimum_test_rows: {metadata['test_rows'].min()}")

    print(f"maximum_test_rows: {metadata['test_rows'].max()}")

    print(f"maximum_purged_dates: {metadata['purged_date_count'].max()}")

    maturity_violations = int(
        (metadata["max_train_target_end"] > metadata["test_date"]).sum()
        + (metadata["max_validation_target_end"] > metadata["test_date"]).sum()
    )

    print(f"label_maturity_violations: {maturity_violations}")

    print()
    print("First folds:")

    preview_columns = [
        "fold_id",
        "train_start_date",
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
        "test_date",
        "train_date_count",
        "purged_date_count",
        "test_rows",
    ]

    print(metadata[preview_columns].head().to_string(index=False))

    print()
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
