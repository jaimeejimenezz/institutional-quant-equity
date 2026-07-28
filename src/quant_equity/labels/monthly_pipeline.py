"""Pipeline for building and storing monthly labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_equity.config import PROCESSED_DATA_DIR
from quant_equity.labels.monthly import (
    build_forward_return_labels,
    build_rebalance_calendar,
)

DEFAULT_MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

DEFAULT_REBALANCE_CALENDAR_PATH = PROCESSED_DATA_DIR / "rebalance_calendar.parquet"

DEFAULT_MONTHLY_LABELS_PATH = PROCESSED_DATA_DIR / "labels_monthly.parquet"


@dataclass
class MonthlyLabelBuildResult:
    """Result of the monthly-label construction pipeline."""

    rebalance_calendar: pd.DataFrame
    monthly_labels: pd.DataFrame
    rebalance_calendar_path: Path
    monthly_labels_path: Path


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
    *,
    sort_by: list[str],
) -> Path:
    """Write a sorted Parquet dataset atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered_data = data.sort_values(sort_by).reset_index(drop=True)

    temporary_path = path.with_suffix(".tmp.parquet")

    temporary_path.unlink(missing_ok=True)

    ordered_data.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)

    return path


def write_rebalance_calendar(
    rebalance_calendar: pd.DataFrame,
    path: Path = (DEFAULT_REBALANCE_CALENDAR_PATH),
) -> Path:
    """Persist the monthly rebalance calendar."""
    return _write_parquet_atomically(
        rebalance_calendar,
        path,
        sort_by=["as_of_date"],
    )


def write_monthly_labels(
    monthly_labels: pd.DataFrame,
    path: Path = (DEFAULT_MONTHLY_LABELS_PATH),
) -> Path:
    """Persist the monthly forward-return labels."""
    return _write_parquet_atomically(
        monthly_labels,
        path,
        sort_by=[
            "as_of_date",
            "ticker",
        ],
    )


def build_and_store_monthly_labels(
    config: dict[str, Any],
    *,
    market_data_path: Path = (DEFAULT_MARKET_DATA_PATH),
    rebalance_calendar_path: Path = (DEFAULT_REBALANCE_CALENDAR_PATH),
    monthly_labels_path: Path = (DEFAULT_MONTHLY_LABELS_PATH),
) -> MonthlyLabelBuildResult:
    """Build and store the calendar and monthly labels."""
    if not market_data_path.exists():
        raise FileNotFoundError(f"Processed market data was not found: {market_data_path}")

    label_config = config["labels"]

    horizon_sessions = int(label_config["horizon_sessions"])

    relative_to_value = label_config.get("relative_to")

    relative_to = None if relative_to_value is None else str(relative_to_value)

    top_quantile_fraction = float(label_config["top_quantile_fraction"])

    market_data = pd.read_parquet(market_data_path)

    rebalance_calendar = build_rebalance_calendar(
        market_data,
        horizon_sessions=(horizon_sessions),
    )

    monthly_labels = build_forward_return_labels(
        market_data,
        horizon_sessions=(horizon_sessions),
        relative_to=relative_to,
        top_quantile_fraction=(top_quantile_fraction),
        rebalance_calendar=(rebalance_calendar),
    )

    written_calendar_path = write_rebalance_calendar(
        rebalance_calendar,
        rebalance_calendar_path,
    )

    written_labels_path = write_monthly_labels(
        monthly_labels,
        monthly_labels_path,
    )

    return MonthlyLabelBuildResult(
        rebalance_calendar=(rebalance_calendar),
        monthly_labels=monthly_labels,
        rebalance_calendar_path=(written_calendar_path),
        monthly_labels_path=(written_labels_path),
    )
