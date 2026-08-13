"""Build point-in-time quarterly and TTM fundamental datasets."""

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
    SECPointInTimeConfig,
    SECQuarterlyReconstructionConfig,
    build_quarterly_fundamental_events,
    build_ttm_point_in_time_snapshots,
)
from quant_equity.logging_config import (
    configure_logging,
)

CANONICAL_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_events_canonical.parquet"

REBALANCE_PATH = PROJECT_ROOT / "data" / "processed" / "rebalance_calendar.parquet"

QUARTERLY_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_quarterly_events.parquet"

TTM_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_ttm_pit.parquet"

METHODS_PATH = REPORTS_DIR / "tables" / "fundamental_quarterly_methods.csv"

TTM_COVERAGE_PATH = REPORTS_DIR / "tables" / "fundamental_ttm_coverage.csv"


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


def _load_rebalance_dates() -> pd.Series:
    """Load project rebalance dates."""
    data = pd.read_parquet(REBALANCE_PATH)

    column = "as_of_date" if "as_of_date" in data.columns else "rebalance_date"

    return (
        pd.to_datetime(data[column])
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )


def main() -> None:
    """Run Step 9E quarterly and TTM reconstruction."""
    config = load_config()

    configure_logging()

    logger = logging.getLogger("quant_equity")

    canonical = pd.read_parquet(CANONICAL_PATH)

    rebalance_dates = _load_rebalance_dates()

    pit_config = SECPointInTimeConfig.from_mapping(config["sec_fundamentals"]["point_in_time"])

    quarterly_config = SECQuarterlyReconstructionConfig.from_mapping(
        config["sec_fundamentals"]["quarterly_reconstruction"]
    )

    quarterly = build_quarterly_fundamental_events(
        canonical,
        pit_config=pit_config,
        config=quarterly_config,
    )

    logger.info(
        "Quarterly fundamental events: %s",
        len(quarterly),
    )

    ttm = build_ttm_point_in_time_snapshots(
        quarterly,
        rebalance_dates,
        config=quarterly_config,
    )

    _write_parquet_atomically(
        quarterly,
        QUARTERLY_PATH,
    )

    _write_parquet_atomically(
        ttm,
        TTM_PATH,
    )

    method_counts = (
        quarterly.groupby(
            [
                "canonical_metric",
                "source_method",
            ],
            as_index=False,
        )
        .agg(
            observations=(
                "quarter_value",
                "size",
            ),
            companies=(
                "ticker",
                "nunique",
            ),
        )
        .sort_values(
            [
                "canonical_metric",
                "source_method",
            ]
        )
    )

    latest_date = ttm["as_of_date"].max()

    latest = ttm.loc[ttm["as_of_date"].eq(latest_date)]

    coverage = (
        latest.groupby(
            "canonical_metric",
            as_index=False,
        )["ticker"]
        .nunique()
        .rename(columns={"ticker": ("companies")})
        .sort_values("canonical_metric")
    )

    METHODS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    method_counts.to_csv(
        METHODS_PATH,
        index=False,
    )

    coverage.to_csv(
        TTM_COVERAGE_PATH,
        index=False,
    )

    future_violations = int((ttm["latest_component_available_date"] > ttm["as_of_date"]).sum())

    print()
    print("Institutional Quant Equity Research Platform")
    print("Quarterly + TTM fundamentals - Step 9E")
    print("------------------------------------------------")

    print(f"Canonical rows: {len(canonical)}")

    print(f"Quarterly events: {len(quarterly)}")

    print(f"TTM snapshot rows: {len(ttm)}")

    print(f"TTM metrics: {ttm['canonical_metric'].nunique()}")

    print(f"Companies with TTM data: {ttm['ticker'].nunique()}")

    print(f"Future-information violations: {future_violations}")

    print()
    print("Quarter reconstruction methods:")

    print(method_counts.to_string(index=False))

    print()

    print(f"Latest TTM coverage ({latest_date.date()}):")

    print(coverage.to_string(index=False))

    print()

    print(f"Quarterly output: {QUARTERLY_PATH}")

    print(f"TTM output: {TTM_PATH}")

    print()

    print("Quarterly + TTM fundamentals: OK")


if __name__ == "__main__":
    main()
