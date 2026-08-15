"""Build the master technical-fundamental modeling panel."""

from __future__ import annotations

import logging

import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.models import (
    FUNDAMENTAL_GLOBAL_PANEL_COLUMNS,
    FUNDAMENTAL_MISSING_PANEL_COLUMNS,
    FUNDAMENTAL_SECTOR_PANEL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TECHNICAL_PANEL_MODEL_COLUMNS,
    build_modeling_panel,
    write_modeling_panel,
)

TECHNICAL_PATH = PROCESSED_DATA_DIR / "features_technical_monthly.parquet"

FUNDAMENTAL_PATH = PROCESSED_DATA_DIR / "features_fundamental_monthly.parquet"

LABELS_PATH = PROCESSED_DATA_DIR / "labels_monthly.parquet"

OUTPUT_PATH = PROCESSED_DATA_DIR / "modeling_panel.parquet"


def main() -> None:
    """Execute Step 11A."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        TECHNICAL_PATH,
        FUNDAMENTAL_PATH,
        LABELS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    technical = pd.read_parquet(TECHNICAL_PATH)

    fundamental = pd.read_parquet(FUNDAMENTAL_PATH)

    labels = pd.read_parquet(LABELS_PATH)

    panel = build_modeling_panel(
        technical,
        fundamental,
        labels,
    )

    write_modeling_panel(
        panel,
        OUTPUT_PATH,
    )

    duplicates = int(
        panel.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    market_violations = int(panel["technical_latest_market_date"].gt(panel["as_of_date"]).sum())

    target_rows = int(panel["has_target"].sum())

    inference_rows = len(panel) - target_rows

    target_date_violations = int(
        (panel["has_target"].eq(1) & panel["first_future_date"].le(panel["as_of_date"])).sum()
    )

    logger.info("Master modeling panel completed.")

    logger.info(
        "Panel rows: %s",
        len(panel),
    )

    print()
    print("Institutional Quant Equity Research Platform")

    print("Master modeling panel - Step 11A")

    print("------------------------------------------------")

    print(f"Rows: {len(panel)}")

    print(f"Dates: {panel['as_of_date'].nunique()}")

    print(f"Tickers: {panel['ticker'].nunique()}")

    print(f"Sectors: {panel['sector'].nunique()}")

    print()

    print(f"Technical model features: {len(TECHNICAL_PANEL_MODEL_COLUMNS)}")

    print(f"Fundamental global scores: {len(FUNDAMENTAL_GLOBAL_PANEL_COLUMNS)}")

    print(f"Fundamental sector scores: {len(FUNDAMENTAL_SECTOR_PANEL_COLUMNS)}")

    print(f"Fundamental missing flags: {len(FUNDAMENTAL_MISSING_PANEL_COLUMNS)}")

    print(f"Total candidate model features: {len(MODEL_FEATURE_COLUMNS)}")

    print()

    print(f"Rows with targets: {target_rows}")

    print(f"Inference-only rows: {inference_rows}")

    print(f"Target dates: {panel.loc[panel['has_target'].eq(1), 'as_of_date'].nunique()}")

    print()

    print(f"Duplicate date-ticker rows: {duplicates}")

    print(f"Future market-data violations: {market_violations}")

    print(f"Invalid target-start dates: {target_date_violations}")

    print()

    print(f"Mean technical missing count: {panel['technical_missing_count'].mean():.4f}")

    print(
        "Mean fundamental global missing count: "
        f"{panel['fundamental_global_missing_count'].mean():.4f}"
    )

    print(
        "Mean fundamental sector missing count: "
        f"{panel['fundamental_sector_missing_count'].mean():.4f}"
    )

    print()

    print(f"Modeling panel: {OUTPUT_PATH}")

    print()

    print("Master modeling panel: OK")


if __name__ == "__main__":
    main()
