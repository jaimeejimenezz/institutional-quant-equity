"""Cross-sectional processing for monthly technical features."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.features.technical import (
    TECHNICAL_FEATURE_COLUMNS,
    TECHNICAL_PANEL_COLUMNS,
)

PROCESSED_IDENTIFIER_COLUMNS = (
    "as_of_date",
    "ticker",
    "sector",
    "latest_market_date",
    "observations_available",
)

WINSORIZED_FEATURE_COLUMNS = tuple(f"{feature}_winsorized" for feature in TECHNICAL_FEATURE_COLUMNS)

STANDARDIZED_FEATURE_COLUMNS = tuple(f"{feature}_zscore" for feature in TECHNICAL_FEATURE_COLUMNS)

SECTOR_NEUTRAL_FEATURE_COLUMNS = tuple(
    f"{feature}_sector_neutral" for feature in TECHNICAL_FEATURE_COLUMNS
)

TECHNICAL_MODEL_FEATURE_COLUMNS = SECTOR_NEUTRAL_FEATURE_COLUMNS

PROCESSED_TECHNICAL_COLUMNS = (
    *PROCESSED_IDENTIFIER_COLUMNS,
    *TECHNICAL_FEATURE_COLUMNS,
    *WINSORIZED_FEATURE_COLUMNS,
    *STANDARDIZED_FEATURE_COLUMNS,
    *SECTOR_NEUTRAL_FEATURE_COLUMNS,
)


class TechnicalFeatureProcessingError(ValueError):
    """Raised when technical features cannot be processed."""


@dataclass(frozen=True)
class TechnicalFeatureProcessingConfig:
    """Configuration for cross-sectional processing."""

    winsor_lower_quantile: float = 0.01
    winsor_upper_quantile: float = 0.99
    minimum_cross_section_size: int = 10
    minimum_sector_size: int = 2
    sector_neutralization: bool = True

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> TechnicalFeatureProcessingConfig:
        """Build processing settings from configuration."""
        config = cls(
            winsor_lower_quantile=float(
                values.get(
                    "winsor_lower_quantile",
                    0.01,
                )
            ),
            winsor_upper_quantile=float(
                values.get(
                    "winsor_upper_quantile",
                    0.99,
                )
            ),
            minimum_cross_section_size=int(
                values.get(
                    "minimum_cross_section_size",
                    10,
                )
            ),
            minimum_sector_size=int(
                values.get(
                    "minimum_sector_size",
                    2,
                )
            ),
            sector_neutralization=bool(
                values.get(
                    "sector_neutralization",
                    True,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate the processing configuration."""
        if not (0.0 <= self.winsor_lower_quantile < self.winsor_upper_quantile <= 1.0):
            raise TechnicalFeatureProcessingError(
                "Winsor quantiles must satisfy 0 <= lower < upper <= 1."
            )

        if self.minimum_cross_section_size < 2:
            raise TechnicalFeatureProcessingError("minimum_cross_section_size must be at least 2.")

        if self.minimum_sector_size < 2:
            raise TechnicalFeatureProcessingError("minimum_sector_size must be at least 2.")


def _require_columns(
    data: pd.DataFrame,
    required_columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require a dataset to contain expected columns."""
    missing_columns = sorted(set(required_columns).difference(data.columns))

    if missing_columns:
        raise TechnicalFeatureProcessingError(
            f"{dataset_name} is missing columns: " + ", ".join(missing_columns) + "."
        )


def _prepare_raw_features(
    raw_features: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize the raw feature panel."""
    _require_columns(
        raw_features,
        TECHNICAL_PANEL_COLUMNS,
        dataset_name="Raw technical features",
    )

    if raw_features.empty:
        raise TechnicalFeatureProcessingError("Raw technical features are empty.")

    features = raw_features.loc[
        :,
        TECHNICAL_PANEL_COLUMNS,
    ].copy()

    for column in (
        "as_of_date",
        "latest_market_date",
    ):
        features[column] = pd.to_datetime(
            features[column],
            errors="coerce",
        ).dt.normalize()

    features["ticker"] = features["ticker"].astype("string").str.strip().str.upper()

    features["observations_available"] = pd.to_numeric(
        features["observations_available"],
        errors="coerce",
    )

    for feature in TECHNICAL_FEATURE_COLUMNS:
        features[feature] = pd.to_numeric(
            features[feature],
            errors="coerce",
        )

    invalid_identifiers = features[
        [
            "as_of_date",
            "ticker",
            "latest_market_date",
            "observations_available",
        ]
    ].isna().any(axis=1) | features["ticker"].eq("")

    if invalid_identifiers.any():
        raise TechnicalFeatureProcessingError(
            "Raw technical features contain "
            f"{int(invalid_identifiers.sum())} "
            "rows with invalid identifiers."
        )

    duplicated_rows = int(
        features.duplicated(
            subset=[
                "as_of_date",
                "ticker",
            ],
            keep=False,
        ).sum()
    )

    if duplicated_rows:
        raise TechnicalFeatureProcessingError(
            f"Raw technical features contain {duplicated_rows} duplicated date-ticker rows."
        )

    temporal_violations = int(features["latest_market_date"].gt(features["as_of_date"]).sum())

    if temporal_violations:
        raise TechnicalFeatureProcessingError(
            f"{temporal_violations} raw feature rows use future market observations."
        )

    return features.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)


def _prepare_universe(
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare ticker and sector metadata."""
    _require_columns(
        universe,
        (
            "ticker",
            "sector",
        ),
        dataset_name="Universe",
    )

    metadata = universe.loc[
        :,
        [
            "ticker",
            "sector",
        ],
    ].copy()

    metadata["ticker"] = metadata["ticker"].astype("string").str.strip().str.upper()

    metadata["sector"] = metadata["sector"].astype("string").str.strip()

    invalid_rows = (
        metadata[
            [
                "ticker",
                "sector",
            ]
        ]
        .isna()
        .any(axis=1)
        | metadata["ticker"].eq("")
        | metadata["sector"].eq("")
    )

    if invalid_rows.any():
        raise TechnicalFeatureProcessingError(
            f"Universe metadata contains {int(invalid_rows.sum())} invalid rows."
        )

    duplicated_tickers = int(metadata["ticker"].duplicated(keep=False).sum())

    if duplicated_tickers:
        raise TechnicalFeatureProcessingError(
            f"Universe metadata contains {duplicated_tickers} duplicated tickers."
        )

    return metadata


def _winsorize_cross_section(
    values: pd.Series,
    config: TechnicalFeatureProcessingConfig,
) -> pd.Series:
    """Winsorize one feature within one date."""
    result = values.astype(float).copy()

    valid = result.dropna()

    if len(valid) < config.minimum_cross_section_size:
        return result

    lower_bound = float(valid.quantile(config.winsor_lower_quantile))

    upper_bound = float(valid.quantile(config.winsor_upper_quantile))

    result.loc[valid.index] = valid.clip(
        lower=lower_bound,
        upper=upper_bound,
    )

    return result


def _standardize_cross_section(
    values: pd.Series,
    config: TechnicalFeatureProcessingConfig,
) -> pd.Series:
    """Calculate a cross-sectional z-score."""
    result = pd.Series(
        np.nan,
        index=values.index,
        dtype=float,
    )

    valid = values.dropna().astype(float)

    if len(valid) < config.minimum_cross_section_size:
        return result

    mean = float(valid.mean())

    standard_deviation = float(valid.std(ddof=0))

    if not np.isfinite(standard_deviation) or standard_deviation <= np.finfo(float).eps:
        result.loc[valid.index] = 0.0

        return result

    result.loc[valid.index] = (valid - mean) / standard_deviation

    return result


def _sector_neutralize(
    data: pd.DataFrame,
    *,
    score_column: str,
    config: TechnicalFeatureProcessingConfig,
) -> pd.Series:
    """Remove sector means from a standardized score."""
    scores = data[score_column].astype(float)

    if not config.sector_neutralization:
        return scores.copy()

    group_columns = [
        "as_of_date",
        "sector",
    ]

    sector_counts = data.groupby(
        group_columns,
        sort=False,
    )[score_column].transform("count")

    sector_means = data.groupby(
        group_columns,
        sort=False,
    )[score_column].transform("mean")

    neutral_scores = scores.copy()

    eligible = scores.notna() & sector_counts.ge(config.minimum_sector_size)

    neutral_scores.loc[eligible] = scores.loc[eligible] - sector_means.loc[eligible]

    return neutral_scores


def build_processed_technical_features(
    raw_features: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    processing_config: (TechnicalFeatureProcessingConfig | None) = None,
) -> pd.DataFrame:
    """Build winsorized and standardized technical features."""
    config = processing_config or TechnicalFeatureProcessingConfig()

    config.validate()

    features = _prepare_raw_features(raw_features)

    metadata = _prepare_universe(universe)

    processed = features.merge(
        metadata,
        on="ticker",
        how="left",
        validate="many_to_one",
    )

    missing_sector_rows = int(processed["sector"].isna().sum())

    if missing_sector_rows:
        missing_tickers = sorted(
            processed.loc[
                processed["sector"].isna(),
                "ticker",
            ].unique()
        )

        raise TechnicalFeatureProcessingError(
            "Sector metadata is missing for: " + ", ".join(missing_tickers) + "."
        )

    for feature in TECHNICAL_FEATURE_COLUMNS:
        winsorized_column = f"{feature}_winsorized"

        standardized_column = f"{feature}_zscore"

        sector_neutral_column = f"{feature}_sector_neutral"

        processed[winsorized_column] = processed.groupby(
            "as_of_date",
            sort=False,
        )[feature].transform(
            lambda values: _winsorize_cross_section(
                values,
                config,
            )
        )

        processed[standardized_column] = processed.groupby(
            "as_of_date",
            sort=False,
        )[winsorized_column].transform(
            lambda values: _standardize_cross_section(
                values,
                config,
            )
        )

        processed[sector_neutral_column] = _sector_neutralize(
            processed,
            score_column=(standardized_column),
            config=config,
        )

    return (
        processed.loc[
            :,
            PROCESSED_TECHNICAL_COLUMNS,
        ]
        .sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )
