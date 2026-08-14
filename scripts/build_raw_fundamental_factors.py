"""Build raw monthly fundamental factors."""

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
    RAW_FACTOR_COLUMNS,
    FundamentalFactorConfig,
    build_raw_fundamental_factors,
)
from quant_equity.logging_config import (
    configure_logging,
)

BASE_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_base_monthly.parquet"

PIT_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_snapshots_pit.parquet"

OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_factors_raw_monthly.parquet"

COVERAGE_PATH = REPORTS_DIR / "tables" / "fundamental_raw_factor_coverage.csv"

REPORT_PATH = REPORTS_DIR / "data_quality" / "fundamental_raw_factor_report.md"


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
    """Build latest factor coverage table."""
    latest_date = data["as_of_date"].max()

    latest = data.loc[data["as_of_date"].eq(latest_date)]

    total = int(latest["ticker"].nunique())

    rows = []

    for factor in RAW_FACTOR_COLUMNS:
        available = int(latest[factor].notna().sum())

        rows.append(
            {
                "as_of_date": latest_date,
                "factor": factor,
                "companies_available": (available),
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
    """Write raw-factor quality report."""
    latest_date = data["as_of_date"].max()

    latest = data.loc[data["as_of_date"].eq(latest_date)]

    source_counts = (
        latest["valuation_share_count_source"].fillna("missing").value_counts(dropna=False)
    )

    valuation_coverage = int(latest["market_cap_proxy"].notna().sum())

    infinite_values = int(
        np.isinf(
            data.loc[
                :,
                list(RAW_FACTOR_COLUMNS),
            ].to_numpy(dtype=float)
        ).sum()
    )

    lines = [
        "# Raw fundamental factor report",
        "",
        "## Step",
        "",
        "Step 10B — Raw fundamental factors.",
        "",
        "## Summary",
        "",
        f"- Rows: {len(data)}",
        (f"- Rebalance dates: {data['as_of_date'].nunique()}"),
        (f"- Companies: {data['ticker'].nunique()}"),
        (f"- Raw factors: {len(RAW_FACTOR_COLUMNS)}"),
        (f"- Latest market-cap proxy coverage: {valuation_coverage}/{latest['ticker'].nunique()}"),
        (f"- Infinite factor values: {infinite_values}"),
        "",
        "## Share-count sources on latest date",
        "",
        "```text",
        source_counts.to_string(),
        "```",
        "",
        "## Latest factor coverage",
        "",
        "```text",
        coverage.to_string(index=False),
        "```",
        "",
        "## Important conventions",
        "",
        (
            "- Market capitalization is a proxy equal to "
            "historical close price multiplied by the latest "
            "point-in-time share count."
        ),
        ("- shares_outstanding is preferred. Quarterly diluted_shares is used only as a fallback."),
        (
            "- CAPEX is stored as a positive cash outflow, "
            "therefore FCF = operating cash flow - CAPEX."
        ),
        ("- Missing debt components are not assumed to be zero."),
        ("- No winsorization, standardization or imputation is performed in Step 10B."),
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
    """Execute Step 10B."""
    project_config = load_config()

    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        BASE_PATH,
        PIT_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    base = pd.read_parquet(BASE_PATH)

    pit = pd.read_parquet(PIT_PATH)

    factor_config = FundamentalFactorConfig.from_mapping(project_config["fundamental_factors"])

    factors = build_raw_fundamental_factors(
        fundamental_base=base,
        pit_snapshots=pit,
        config=factor_config,
    )

    _write_parquet_atomically(
        factors,
        OUTPUT_PATH,
    )

    coverage = _build_coverage(factors)

    COVERAGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage.to_csv(
        COVERAGE_PATH,
        index=False,
    )

    _write_report(
        factors,
        coverage,
        REPORT_PATH,
    )

    latest_date = factors["as_of_date"].max()

    latest = factors.loc[factors["as_of_date"].eq(latest_date)]

    source_counts = (
        latest["valuation_share_count_source"].fillna("missing").value_counts(dropna=False)
    )

    valuation_coverage = int(latest["market_cap_proxy"].notna().sum())

    infinite_values = int(
        np.isinf(
            factors.loc[
                :,
                list(RAW_FACTOR_COLUMNS),
            ].to_numpy(dtype=float)
        ).sum()
    )

    logger.info("Raw fundamental factors completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Raw fundamental factors - Step 10B")
    print("------------------------------------------------")

    print(f"Rows: {len(factors)}")

    print(f"Rebalance dates: {factors['as_of_date'].nunique()}")

    print(f"Companies: {factors['ticker'].nunique()}")

    print(f"Raw factors: {len(RAW_FACTOR_COLUMNS)}")

    print(f"Latest market-cap proxy coverage: {valuation_coverage}/{latest['ticker'].nunique()}")

    print(f"Infinite factor values: {infinite_values}")

    print()

    print("Latest valuation share-count sources:")

    print(source_counts.to_string())

    print()

    print("Latest raw factor coverage:")

    print(coverage.to_string(index=False))

    print()

    print(f"Output: {OUTPUT_PATH}")

    print(f"Coverage: {COVERAGE_PATH}")

    print(f"Quality report: {REPORT_PATH}")

    print()

    print("Raw fundamental factors: OK")


if __name__ == "__main__":
    main()
