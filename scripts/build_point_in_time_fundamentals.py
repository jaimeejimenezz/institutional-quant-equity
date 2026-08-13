"""Build monthly point-in-time SEC fundamental snapshots."""

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
    build_point_in_time_coverage,
    build_point_in_time_snapshots,
)
from quant_equity.logging_config import (
    configure_logging,
)

CANONICAL_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_events_canonical.parquet"

REBALANCE_PATH = PROJECT_ROOT / "data" / "processed" / "rebalance_calendar.parquet"

OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_snapshots_pit.parquet"

COVERAGE_PATH = REPORTS_DIR / "tables" / "fundamental_pit_coverage.csv"


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write Parquet atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(".tmp.parquet")

    temporary_path.unlink(missing_ok=True)

    data.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)


def _load_rebalance_dates(
    path: Path,
) -> pd.Series:
    """Load monthly project rebalance dates."""
    if not path.exists():
        raise FileNotFoundError(f"Rebalance calendar not found: {path}")

    data = pd.read_parquet(path)

    if "as_of_date" in data.columns:
        column = "as_of_date"
    elif "rebalance_date" in data.columns:
        column = "rebalance_date"
    else:
        raise ValueError("Rebalance calendar requires as_of_date or rebalance_date.")

    dates = pd.to_datetime(
        data[column],
        errors="coerce",
    ).dt.normalize()

    if dates.isna().any():
        raise ValueError("Rebalance calendar contains invalid dates.")

    return dates.drop_duplicates().sort_values().reset_index(drop=True)


def main() -> None:
    """Run Step 9D point-in-time reconstruction."""
    config = load_config()

    configure_logging()

    logger = logging.getLogger("quant_equity")

    if not CANONICAL_PATH.exists():
        raise FileNotFoundError(f"Canonical fundamentals not found: {CANONICAL_PATH}")

    canonical = pd.read_parquet(CANONICAL_PATH)

    rebalance_dates = _load_rebalance_dates(REBALANCE_PATH)

    pit_config = SECPointInTimeConfig.from_mapping(
        config["sec_fundamentals"].get(
            "point_in_time",
            {},
        )
    )

    logger.info(
        "Canonical SEC rows: %s",
        len(canonical),
    )

    logger.info(
        "Rebalance dates: %s",
        len(rebalance_dates),
    )

    snapshots = build_point_in_time_snapshots(
        canonical,
        rebalance_dates,
        config=pit_config,
    )

    coverage = build_point_in_time_coverage(snapshots)

    _write_parquet_atomically(
        snapshots,
        OUTPUT_PATH,
    )

    COVERAGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage.to_csv(
        COVERAGE_PATH,
        index=False,
    )

    latest_date = snapshots["as_of_date"].max()

    latest = snapshots.loc[snapshots["as_of_date"].eq(latest_date)]

    future_information = int((snapshots["available_date"] > snapshots["as_of_date"]).sum())

    print()
    print("Institutional Quant Equity Research Platform")
    print("Point-in-time fundamentals - Step 9D")
    print("------------------------------------------------")

    print(f"Canonical input rows: {len(canonical)}")

    print(f"Rebalance dates: {len(rebalance_dates)}")

    print(f"Snapshot rows: {len(snapshots)}")

    print(f"Companies represented: {snapshots['ticker'].nunique()}")

    print(f"Canonical metrics: {snapshots['canonical_metric'].nunique()}")

    print(f"First snapshot date: {snapshots['as_of_date'].min().date()}")

    print(f"Last snapshot date: {snapshots['as_of_date'].max().date()}")

    print(f"Future-information violations: {future_information}")

    print()

    print(f"Latest snapshot coverage ({latest_date.date()}):")

    latest_coverage = (
        latest.groupby(
            [
                "canonical_metric",
                "duration_class",
            ]
        )["ticker"]
        .nunique()
        .rename("companies")
        .reset_index()
    )

    print(latest_coverage.to_string(index=False))

    print()

    print(f"Output: {OUTPUT_PATH}")

    print(f"Coverage: {COVERAGE_PATH}")

    print()

    print("Point-in-time fundamentals: OK")


if __name__ == "__main__":
    main()
