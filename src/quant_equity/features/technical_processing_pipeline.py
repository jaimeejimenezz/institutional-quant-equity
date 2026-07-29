"""Pipeline for processing and storing technical features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_equity.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from quant_equity.data import (
    load_universe,
)
from quant_equity.features.technical_processing import (
    TechnicalFeatureProcessingConfig,
    build_processed_technical_features,
)

DEFAULT_RAW_TECHNICAL_FEATURES_PATH = INTERIM_DATA_DIR / "features_technical_raw_monthly.parquet"

DEFAULT_PROCESSED_TECHNICAL_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "features_technical_monthly.parquet"
)


@dataclass
class ProcessedTechnicalFeatureBuildResult:
    """Result of the technical processing pipeline."""

    features: pd.DataFrame
    features_path: Path


def write_processed_technical_features(
    features: pd.DataFrame,
    path: Path = (DEFAULT_PROCESSED_TECHNICAL_FEATURES_PATH),
) -> Path:
    """Store processed features atomically."""
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


def build_and_store_processed_technical_features(
    config: dict[str, Any],
    *,
    raw_features_path: Path = (DEFAULT_RAW_TECHNICAL_FEATURES_PATH),
    processed_features_path: Path = (DEFAULT_PROCESSED_TECHNICAL_FEATURES_PATH),
) -> ProcessedTechnicalFeatureBuildResult:
    """Process and persist the monthly technical features."""
    if not raw_features_path.exists():
        raise FileNotFoundError(f"Raw technical features were not found: {raw_features_path}")

    processing_config = TechnicalFeatureProcessingConfig.from_mapping(
        config["technical_feature_processing"]
    )

    universe = load_universe(str(config["universe"]["version"]))

    raw_features = pd.read_parquet(raw_features_path)

    processed_features = build_processed_technical_features(
        raw_features,
        universe,
        processing_config=(processing_config),
    )

    written_path = write_processed_technical_features(
        processed_features,
        processed_features_path,
    )

    return ProcessedTechnicalFeatureBuildResult(
        features=processed_features,
        features_path=written_path,
    )
