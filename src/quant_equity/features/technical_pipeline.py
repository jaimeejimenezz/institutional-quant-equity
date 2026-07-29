"""Pipeline for building and storing raw technical features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_equity.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from quant_equity.features.technical import (
    TechnicalFeatureConfig,
    build_raw_technical_features,
)

DEFAULT_MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

DEFAULT_REBALANCE_CALENDAR_PATH = PROCESSED_DATA_DIR / "rebalance_calendar.parquet"

DEFAULT_RAW_TECHNICAL_FEATURES_PATH = INTERIM_DATA_DIR / "features_technical_raw_monthly.parquet"


@dataclass
class RawTechnicalFeatureBuildResult:
    """Result of the raw technical-feature pipeline."""

    features: pd.DataFrame
    features_path: Path


def write_raw_technical_features(
    features: pd.DataFrame,
    path: Path = (DEFAULT_RAW_TECHNICAL_FEATURES_PATH),
) -> Path:
    """Store sorted technical features atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered_features = features.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)

    temporary_path = path.with_suffix(".tmp.parquet")

    temporary_path.unlink(missing_ok=True)

    ordered_features.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)

    return path


def build_and_store_raw_technical_features(
    config: dict[str, Any],
    *,
    market_data_path: Path = (DEFAULT_MARKET_DATA_PATH),
    rebalance_calendar_path: Path = (DEFAULT_REBALANCE_CALENDAR_PATH),
    features_path: Path = (DEFAULT_RAW_TECHNICAL_FEATURES_PATH),
) -> RawTechnicalFeatureBuildResult:
    """Build and persist the raw monthly feature panel."""
    if not market_data_path.exists():
        raise FileNotFoundError(f"Processed market data was not found: {market_data_path}")

    if not rebalance_calendar_path.exists():
        raise FileNotFoundError(f"Rebalance calendar was not found: {rebalance_calendar_path}")

    feature_config = TechnicalFeatureConfig.from_mapping(config["technical_features"])

    market_data = pd.read_parquet(market_data_path)

    rebalance_calendar = pd.read_parquet(rebalance_calendar_path)

    features = build_raw_technical_features(
        market_data,
        rebalance_calendar,
        feature_config=(feature_config),
    )

    written_path = write_raw_technical_features(
        features,
        features_path,
    )

    return RawTechnicalFeatureBuildResult(
        features=features,
        features_path=(written_path),
    )
