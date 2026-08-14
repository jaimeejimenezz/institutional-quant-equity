"""Build processed monthly fundamental features."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from quant_equity.config import (
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.features import (
    FUNDAMENTAL_FACTOR_COLUMNS,
    FundamentalTransformConfig,
    build_processed_fundamental_features,
)
from quant_equity.logging_config import (
    configure_logging,
)

INPUT_PATH = (
    PROJECT_ROOT / "data" / "interim" / "fundamental_factors_raw_with_growth_monthly.parquet"
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_fundamental_monthly.parquet"

COVERAGE_PATH = REPORTS_DIR / "tables" / "fundamental_processed_feature_coverage.csv"

REPORT_PATH = REPORTS_DIR / "data_quality" / "fundamental_processed_feature_report.md"


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write Parquet atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(".tmp.parquet")

    temporary.unlink(missing_ok=True)

    data.to_parquet(
        temporary,
        index=False,
    )

    temporary.replace(path)


def _processed_feature_columns() -> list[str]:
    """Return transformed model feature names."""
    columns = []

    for factor in FUNDAMENTAL_FACTOR_COLUMNS:
        columns.extend(
            [
                f"{factor}_zscore",
                f"{factor}_sector_zscore",
                f"{factor}_missing",
            ]
        )

    return columns


def _build_coverage(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate latest processed feature coverage."""
    latest_date = data["as_of_date"].max()

    latest = data.loc[data["as_of_date"].eq(latest_date)]

    total = int(latest["ticker"].nunique())

    rows = []

    for factor in FUNDAMENTAL_FACTOR_COLUMNS:
        for version in (
            "zscore",
            "sector_zscore",
        ):
            column = f"{factor}_{version}"

            available = int(latest[column].notna().sum())

            rows.append(
                {
                    "as_of_date": (latest_date),
                    "factor": factor,
                    "version": version,
                    "companies_available": (available),
                    "companies_total": total,
                    "coverage_ratio": (available / total if total else np.nan),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "version",
                "coverage_ratio",
                "factor",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def _count_infinite_values(
    data: pd.DataFrame,
) -> int:
    """Count infinities in transformed numeric features."""
    columns = []

    for factor in FUNDAMENTAL_FACTOR_COLUMNS:
        columns.extend(
            [
                f"{factor}_winsorized",
                f"{factor}_zscore",
                f"{factor}_sector_zscore",
            ]
        )

    return int(np.isinf(data[columns].to_numpy(dtype=float)).sum())


def _write_report(
    data: pd.DataFrame,
    coverage: pd.DataFrame,
    path: Path,
) -> None:
    """Write Step 10D quality report."""
    duplicates = int(
        data.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    infinite_values = _count_infinite_values(data)

    raw_count = len(FUNDAMENTAL_FACTOR_COLUMNS)

    missing_flags = len(FUNDAMENTAL_FACTOR_COLUMNS)

    global_scores = len(FUNDAMENTAL_FACTOR_COLUMNS)

    sector_scores = len(FUNDAMENTAL_FACTOR_COLUMNS)

    lines = [
        "# Processed fundamental feature report",
        "",
        "## Step",
        "",
        ("Step 10D — Cross-sectional and sector fundamental transformations."),
        "",
        "## Summary",
        "",
        f"- Rows: {len(data)}",
        (f"- Rebalance dates: {data['as_of_date'].nunique()}"),
        (f"- Companies: {data['ticker'].nunique()}"),
        (f"- Raw fundamental factors: {raw_count}"),
        (f"- Global z-score features: {global_scores}"),
        (f"- Sector z-score features: {sector_scores}"),
        (f"- Missingness indicators: {missing_flags}"),
        (f"- Duplicate date-ticker rows: {duplicates}"),
        (f"- Infinite transformed values: {infinite_values}"),
        "",
        "## Latest coverage",
        "",
        "```text",
        coverage.to_string(index=False),
        "```",
        "",
        "## Methodology",
        "",
        ("- Winsorization is performed independently within each rebalance date."),
        ("- Global z-scores are calculated only using companies in the same rebalance date."),
        ("- Sector z-scores are calculated only using companies in the same date and sector."),
        ("- Missing raw factors remain missing. No imputation is performed."),
        ("- A binary missingness indicator is created for every fundamental factor."),
        ("- A factor with zero cross-sectional variation receives a neutral z-score of zero."),
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Execute Step 10D."""
    project_config = load_config()

    configure_logging()

    logger = logging.getLogger("quant_equity")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {INPUT_PATH}")

    raw = pd.read_parquet(INPUT_PATH)

    transform_config = FundamentalTransformConfig.from_mapping(
        project_config["fundamental_factors"]["transform"]
    )

    processed = build_processed_fundamental_features(
        raw,
        config=transform_config,
    )

    _write_parquet_atomically(
        processed,
        OUTPUT_PATH,
    )

    coverage = _build_coverage(processed)

    COVERAGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage.to_csv(
        COVERAGE_PATH,
        index=False,
    )

    _write_report(
        processed,
        coverage,
        REPORT_PATH,
    )

    duplicates = int(
        processed.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    infinite_values = _count_infinite_values(processed)

    latest_date = processed["as_of_date"].max()

    latest = processed.loc[processed["as_of_date"].eq(latest_date)]

    global_columns = [f"{factor}_zscore" for factor in (FUNDAMENTAL_FACTOR_COLUMNS)]

    sector_columns = [f"{factor}_sector_zscore" for factor in (FUNDAMENTAL_FACTOR_COLUMNS)]

    global_values_available = int(latest[global_columns].notna().sum().sum())

    sector_values_available = int(latest[sector_columns].notna().sum().sum())

    logger.info("Processed fundamental features completed.")

    print()
    print("Institutional Quant Equity Research Platform")

    print("Processed fundamental features - Step 10D")

    print("------------------------------------------------")

    print(f"Rows: {len(processed)}")

    print(f"Rebalance dates: {processed['as_of_date'].nunique()}")

    print(f"Companies: {processed['ticker'].nunique()}")

    print(f"Raw fundamental factors: {len(FUNDAMENTAL_FACTOR_COLUMNS)}")

    print(f"Global z-score features: {len(FUNDAMENTAL_FACTOR_COLUMNS)}")

    print(f"Sector z-score features: {len(FUNDAMENTAL_FACTOR_COLUMNS)}")

    print(f"Missingness indicators: {len(FUNDAMENTAL_FACTOR_COLUMNS)}")

    print(f"Duplicate date-ticker rows: {duplicates}")

    print(f"Infinite transformed values: {infinite_values}")

    print()

    print(
        "Latest global z-score values available: "
        f"{global_values_available}/"
        f"{len(latest) * len(global_columns)}"
    )

    print(
        "Latest sector z-score values available: "
        f"{sector_values_available}/"
        f"{len(latest) * len(sector_columns)}"
    )

    print()

    print("Latest processed feature coverage:")

    print(coverage.to_string(index=False))

    print()

    print(f"Output: {OUTPUT_PATH}")

    print(f"Coverage: {COVERAGE_PATH}")

    print(f"Quality report: {REPORT_PATH}")

    print()

    print("Processed fundamental features: OK")


if __name__ == "__main__":
    main()
