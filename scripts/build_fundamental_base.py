"""Build the monthly point-in-time fundamental input base."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from quant_equity.config import (
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.data import (
    load_universe,
)
from quant_equity.features import (
    FundamentalBaseConfig,
    build_monthly_fundamental_base,
)
from quant_equity.logging_config import (
    configure_logging,
)

PIT_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_snapshots_pit.parquet"

TTM_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_ttm_pit.parquet"

MARKET_PATH = PROJECT_ROOT / "data" / "processed" / "market_daily.parquet"

REBALANCE_PATH = PROJECT_ROOT / "data" / "processed" / "rebalance_calendar.parquet"

OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_base_monthly.parquet"

COVERAGE_PATH = REPORTS_DIR / "tables" / "fundamental_base_coverage.csv"

REPORT_PATH = REPORTS_DIR / "data_quality" / "fundamental_base_report.md"


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


def _fundamental_value_columns(
    data: pd.DataFrame,
) -> list[str]:
    """Identify raw fundamental input columns."""
    excluded = {
        "as_of_date",
        "ticker",
        "company_name",
        "sector",
        "industry",
        "cik",
        "close_price",
        "fundamental_input_count",
        "fundamental_missing_count",
    }

    return [
        column
        for column in data.columns
        if (column not in excluded and not column.endswith("_available_date"))
    ]


def _build_coverage(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate latest cross-sectional coverage."""
    latest_date = data["as_of_date"].max()

    latest = data.loc[data["as_of_date"].eq(latest_date)].copy()

    company_count = latest["ticker"].nunique()

    rows: list[dict[str, object]] = []

    for column in _fundamental_value_columns(data):
        available = int(latest[column].notna().sum())

        coverage_ratio = available / company_count if company_count > 0 else float("nan")

        rows.append(
            {
                "as_of_date": latest_date,
                "input": column,
                "companies_available": available,
                "companies_total": company_count,
                "coverage_ratio": coverage_ratio,
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        [
            "coverage_ratio",
            "input",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)


def _count_future_violations(
    data: pd.DataFrame,
) -> int:
    """Count source dates after as_of_date."""
    violations = 0

    date_columns = [column for column in data.columns if column.endswith("_available_date")]

    for column in date_columns:
        source_date = pd.to_datetime(
            data[column],
            errors="coerce",
        )

        invalid = source_date.notna() & (source_date > data["as_of_date"])

        violations += int(invalid.sum())

    return violations


def _write_report(
    data: pd.DataFrame,
    coverage: pd.DataFrame,
    path: Path,
) -> None:
    """Write Step 10A quality report."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    duplicates = int(
        data.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    missing_close = int(data["close_price"].isna().sum())

    future_violations = _count_future_violations(data)

    lines = [
        "# Fundamental base quality report",
        "",
        "## Step",
        "",
        "Step 10A — Monthly fundamental input base.",
        "",
        "## Summary",
        "",
        (f"- Rows: {len(data)}"),
        (f"- Rebalance dates: {data['as_of_date'].nunique()}"),
        (f"- Companies: {data['ticker'].nunique()}"),
        (f"- First date: {data['as_of_date'].min().date()}"),
        (f"- Last date: {data['as_of_date'].max().date()}"),
        (f"- Duplicate date-ticker rows: {duplicates}"),
        (f"- Missing close prices: {missing_close}"),
        (f"- Fundamental input columns: {len(_fundamental_value_columns(data))}"),
        (f"- Future-information violations: {future_violations}"),
        (f"- Mean available fundamentals per row: {data['fundamental_input_count'].mean():.2f}"),
        "",
        "## Latest coverage",
        "",
        "```text",
        coverage.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        (
            "Missing accounting inputs are preserved as "
            "missing values. No imputation is performed "
            "in Step 10A."
        ),
        "",
        (
            "All accounting source availability dates must "
            "be less than or equal to the corresponding "
            "rebalance date."
        ),
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Execute Step 10A."""
    project_config = load_config()

    configure_logging()

    logger = logging.getLogger("quant_equity")

    required_paths = (
        PIT_PATH,
        TTM_PATH,
        MARKET_PATH,
        REBALANCE_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    universe = load_universe(str(project_config["universe"]["version"]))

    pit = pd.read_parquet(PIT_PATH)

    ttm = pd.read_parquet(TTM_PATH)

    market = pd.read_parquet(MARKET_PATH)

    rebalance = pd.read_parquet(REBALANCE_PATH)

    config = FundamentalBaseConfig.from_mapping(
        project_config["sec_fundamentals"]["quarterly_reconstruction"]
    )

    fundamental_base = build_monthly_fundamental_base(
        pit_snapshots=pit,
        ttm_snapshots=ttm,
        market_daily=market,
        universe=universe,
        rebalance_calendar=rebalance,
        config=config,
    )

    _write_parquet_atomically(
        fundamental_base,
        OUTPUT_PATH,
    )

    coverage = _build_coverage(fundamental_base)

    COVERAGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage.to_csv(
        COVERAGE_PATH,
        index=False,
    )

    _write_report(
        fundamental_base,
        coverage,
        REPORT_PATH,
    )

    future_violations = _count_future_violations(fundamental_base)

    duplicates = int(
        fundamental_base.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    missing_close = int(fundamental_base["close_price"].isna().sum())

    logger.info("Fundamental monthly base completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Fundamental monthly base - Step 10A")
    print("------------------------------------------------")

    print(f"Rows: {len(fundamental_base)}")

    print(f"Rebalance dates: {fundamental_base['as_of_date'].nunique()}")

    print(f"Companies: {fundamental_base['ticker'].nunique()}")

    print(f"Fundamental input columns: {len(_fundamental_value_columns(fundamental_base))}")

    print(f"Duplicate date-ticker rows: {duplicates}")

    print(f"Missing close prices: {missing_close}")

    print(f"Future-information violations: {future_violations}")

    print()

    print("Latest fundamental coverage:")

    print(coverage.to_string(index=False))

    print()

    print(f"Output: {OUTPUT_PATH}")

    print(f"Coverage: {COVERAGE_PATH}")

    print(f"Quality report: {REPORT_PATH}")

    print()

    print("Fundamental monthly base: OK")


if __name__ == "__main__":
    main()
