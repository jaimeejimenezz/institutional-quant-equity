"""Generate modeling-panel documentation and coverage."""

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
from quant_equity.reporting import (
    write_modeling_panel_dictionary_artifacts,
)

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_panel.parquet"

DICTIONARY_CSV_PATH = REPORTS_DIR / "tables" / "modeling_panel_data_dictionary.csv"

COVERAGE_PATH = REPORTS_DIR / "tables" / "modeling_panel_feature_coverage.csv"

DOCUMENTATION_PATH = PROJECT_ROOT / "docs" / "DATA_DICTIONARY.md"


def main() -> None:
    """Execute Step 11C."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {PANEL_PATH}")

    panel = pd.read_parquet(PANEL_PATH)

    dictionary, coverage = write_modeling_panel_dictionary_artifacts(
        panel,
        dictionary_csv_path=(DICTIONARY_CSV_PATH),
        coverage_csv_path=(COVERAGE_PATH),
        documentation_path=(DOCUMENTATION_PATH),
    )

    predictor_dictionary = dictionary.loc[dictionary["model_input"].eq(True)]

    group_summary = (
        predictor_dictionary.groupby("feature_group")
        .agg(
            features=(
                "column",
                "count",
            ),
            mean_coverage=(
                "overall_coverage",
                "mean",
            ),
            latest_coverage=(
                "latest_coverage",
                "mean",
            ),
            minimum_coverage=(
                "overall_coverage",
                "min",
            ),
        )
        .reset_index()
    )

    undocumented = set(panel.columns) - set(dictionary["column"])

    duplicate_dictionary_columns = int(dictionary["column"].duplicated().sum())

    logger.info("Modeling panel documentation completed.")

    print()
    print("Institutional Quant Equity Research Platform")

    print("Modeling panel documentation - Step 11C")

    print("------------------------------------------------")

    print(f"Panel columns: {len(panel.columns)}")

    print(f"Documented columns: {len(dictionary)}")

    print(f"Candidate model features: {len(MODEL_FEATURE_COLUMNS)}")

    print(f"Coverage rows: {len(coverage)}")

    print(f"Undocumented panel columns: {len(undocumented)}")

    print(f"Duplicate dictionary columns: {duplicate_dictionary_columns}")

    print()

    print("Predictor coverage by group:")

    display = group_summary.copy()

    for column in (
        "mean_coverage",
        "latest_coverage",
        "minimum_coverage",
    ):
        display[column] = display[column].map(lambda value: f"{value:.2%}")

    print(display.to_string(index=False))

    print()

    print(f"Dictionary CSV: {DICTIONARY_CSV_PATH}")

    print(f"Coverage table: {COVERAGE_PATH}")

    print(f"Data dictionary: {DOCUMENTATION_PATH}")

    print()

    print("Modeling panel documentation: OK")


if __name__ == "__main__":
    main()
