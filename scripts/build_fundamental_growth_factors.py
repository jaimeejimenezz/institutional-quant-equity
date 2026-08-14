"""Build raw fundamental growth factors."""

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
    GROWTH_FACTOR_COLUMNS,
    RAW_FACTOR_COLUMNS,
    FundamentalGrowthConfig,
    build_fundamental_growth_factors,
)
from quant_equity.logging_config import (
    configure_logging,
)

INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_factors_raw_monthly.parquet"

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "interim" / "fundamental_factors_raw_with_growth_monthly.parquet"
)

COVERAGE_PATH = REPORTS_DIR / "tables" / "fundamental_growth_factor_coverage.csv"

REPORT_PATH = REPORTS_DIR / "data_quality" / "fundamental_growth_factor_report.md"


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


def _build_coverage(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate latest growth-factor coverage."""
    latest_date = data["as_of_date"].max()

    latest = data.loc[data["as_of_date"].eq(latest_date)]

    total = int(latest["ticker"].nunique())

    rows = []

    for factor in GROWTH_FACTOR_COLUMNS:
        available = int(latest[factor].notna().sum())

        rows.append(
            {
                "as_of_date": latest_date,
                "factor": factor,
                "companies_available": available,
                "companies_total": total,
                "coverage_ratio": (available / total if total else np.nan),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "coverage_ratio",
                "factor",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def _write_report(
    data: pd.DataFrame,
    coverage: pd.DataFrame,
    path: Path,
) -> None:
    """Write Step 10C report."""
    factor_columns = list(RAW_FACTOR_COLUMNS) + list(GROWTH_FACTOR_COLUMNS)

    infinite_values = int(np.isinf(data[factor_columns].to_numpy(dtype=float)).sum())

    lines = [
        "# Fundamental growth factor report",
        "",
        "## Step",
        "",
        "Step 10C — Fundamental growth and acceleration.",
        "",
        "## Summary",
        "",
        f"- Rows: {len(data)}",
        (f"- Rebalance dates: {data['as_of_date'].nunique()}"),
        (f"- Companies: {data['ticker'].nunique()}"),
        (f"- Existing raw factors: {len(RAW_FACTOR_COLUMNS)}"),
        (f"- New growth factors: {len(GROWTH_FACTOR_COLUMNS)}"),
        (f"- Total raw fundamental factors: {len(factor_columns)}"),
        (f"- Infinite factor values: {infinite_values}"),
        "",
        "## Latest growth coverage",
        "",
        "```text",
        coverage.to_string(index=False),
        "```",
        "",
        "## Methodology",
        "",
        ("- Revenue and asset growth use current / prior-year value - 1."),
        (
            "- Net-income and operating-cash-flow "
            "growth use change divided by the "
            "absolute prior-year value because "
            "these metrics may be negative."
        ),
        (
            "- Acceleration equals current YoY growth "
            "minus the YoY growth observed one year earlier."
        ),
        (
            "- No missing-value imputation, winsorization "
            "or standardization is performed in Step 10C."
        ),
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
    """Execute Step 10C."""
    config = load_config()

    configure_logging()

    logger = logging.getLogger("quant_equity")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {INPUT_PATH}")

    raw = pd.read_parquet(INPUT_PATH)

    growth_config = FundamentalGrowthConfig.from_mapping(config["fundamental_factors"]["growth"])

    result = build_fundamental_growth_factors(
        raw,
        config=growth_config,
    )

    _write_parquet_atomically(
        result,
        OUTPUT_PATH,
    )

    coverage = _build_coverage(result)

    COVERAGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage.to_csv(
        COVERAGE_PATH,
        index=False,
    )

    _write_report(
        result,
        coverage,
        REPORT_PATH,
    )

    all_factor_columns = list(RAW_FACTOR_COLUMNS) + list(GROWTH_FACTOR_COLUMNS)

    infinite_values = int(np.isinf(result[all_factor_columns].to_numpy(dtype=float)).sum())

    latest_date = result["as_of_date"].max()

    logger.info("Fundamental growth factors completed.")

    print()
    print("Institutional Quant Equity Research Platform")

    print("Fundamental growth factors - Step 10C")

    print("------------------------------------------------")

    print(f"Rows: {len(result)}")

    print(f"Rebalance dates: {result['as_of_date'].nunique()}")

    print(f"Companies: {result['ticker'].nunique()}")

    print(f"Existing raw factors: {len(RAW_FACTOR_COLUMNS)}")

    print(f"New growth factors: {len(GROWTH_FACTOR_COLUMNS)}")

    print(f"Total raw fundamental factors: {len(all_factor_columns)}")

    print(f"Infinite factor values: {infinite_values}")

    print()

    print(f"Latest growth coverage ({latest_date.date()}):")

    print(coverage.to_string(index=False))

    print()

    print(f"Output: {OUTPUT_PATH}")

    print(f"Coverage: {COVERAGE_PATH}")

    print(f"Quality report: {REPORT_PATH}")

    print()

    print("Fundamental growth factors: OK")


if __name__ == "__main__":
    main()
