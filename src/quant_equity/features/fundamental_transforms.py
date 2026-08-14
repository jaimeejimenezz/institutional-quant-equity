"""Cross-sectional transformations for fundamental factors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.features.fundamental_factors import (
    RAW_FACTOR_COLUMNS,
)
from quant_equity.features.fundamental_growth import (
    GROWTH_FACTOR_COLUMNS,
)


class FundamentalTransformError(ValueError):
    """Raised when fundamental transformations cannot be built."""


FUNDAMENTAL_FACTOR_COLUMNS = tuple(RAW_FACTOR_COLUMNS) + tuple(GROWTH_FACTOR_COLUMNS)


@dataclass(frozen=True)
class FundamentalTransformConfig:
    """Configuration for cross-sectional transformations."""

    winsor_lower_quantile: float = 0.025
    winsor_upper_quantile: float = 0.975
    min_cross_section_observations: int = 10
    min_sector_observations: int = 3
    zscore_ddof: int = 0
    zero_std_value: float = 0.0

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> FundamentalTransformConfig:
        """Create transformation config from YAML."""
        config = cls(
            winsor_lower_quantile=float(
                values.get(
                    "winsor_lower_quantile",
                    0.025,
                )
            ),
            winsor_upper_quantile=float(
                values.get(
                    "winsor_upper_quantile",
                    0.975,
                )
            ),
            min_cross_section_observations=int(
                values.get(
                    "min_cross_section_observations",
                    10,
                )
            ),
            min_sector_observations=int(
                values.get(
                    "min_sector_observations",
                    3,
                )
            ),
            zscore_ddof=int(
                values.get(
                    "zscore_ddof",
                    0,
                )
            ),
            zero_std_value=float(
                values.get(
                    "zero_std_value",
                    0.0,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate transformation settings."""
        if not (0.0 <= self.winsor_lower_quantile < self.winsor_upper_quantile <= 1.0):
            raise FundamentalTransformError("Invalid winsorization quantiles.")

        if self.min_cross_section_observations < 2:
            raise FundamentalTransformError("min_cross_section_observations must be at least 2.")

        if self.min_sector_observations < 2:
            raise FundamentalTransformError("min_sector_observations must be at least 2.")

        if self.zscore_ddof < 0:
            raise FundamentalTransformError("zscore_ddof cannot be negative.")


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    """Require input columns."""
    missing = [column for column in columns if column not in data.columns]

    if missing:
        raise FundamentalTransformError(f"Fundamental factor data are missing columns: {missing}")


def _winsorize_series(
    values: pd.Series,
    *,
    config: FundamentalTransformConfig,
) -> pd.Series:
    """Winsorize one cross-sectional factor."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    valid = numeric.dropna()

    result = pd.Series(
        np.nan,
        index=values.index,
        dtype="float64",
    )

    if len(valid) < config.min_cross_section_observations:
        return result

    lower = valid.quantile(config.winsor_lower_quantile)

    upper = valid.quantile(config.winsor_upper_quantile)

    result.loc[numeric.notna()] = numeric.loc[numeric.notna()].clip(
        lower=lower,
        upper=upper,
    )

    return result


def _zscore_series(
    values: pd.Series,
    *,
    minimum_observations: int,
    config: FundamentalTransformConfig,
) -> pd.Series:
    """Standardize one cross-sectional factor."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    valid = numeric.dropna()

    result = pd.Series(
        np.nan,
        index=values.index,
        dtype="float64",
    )

    if len(valid) < minimum_observations:
        return result

    mean = float(valid.mean())

    std = float(valid.std(ddof=config.zscore_ddof))

    if not np.isfinite(std):
        return result

    if std <= 1.0e-15:
        result.loc[numeric.notna()] = config.zero_std_value

        return result

    result.loc[numeric.notna()] = (numeric.loc[numeric.notna()] - mean) / std

    return result


def _transform_factor_by_date(
    data: pd.DataFrame,
    factor: str,
    *,
    config: FundamentalTransformConfig,
) -> pd.DataFrame:
    """Create winsorized and global z-score versions."""
    output = pd.DataFrame(index=data.index)

    output[f"{factor}_winsorized"] = np.nan

    output[f"{factor}_zscore"] = np.nan

    for _, group in data.groupby(
        "as_of_date",
        sort=False,
    ):
        winsorized = _winsorize_series(
            group[factor],
            config=config,
        )

        output.loc[
            group.index,
            f"{factor}_winsorized",
        ] = winsorized

        standardized = _zscore_series(
            winsorized,
            minimum_observations=(config.min_cross_section_observations),
            config=config,
        )

        output.loc[
            group.index,
            f"{factor}_zscore",
        ] = standardized

    return output


def _build_sector_zscore(
    data: pd.DataFrame,
    factor: str,
    *,
    config: FundamentalTransformConfig,
) -> pd.Series:
    """Standardize a winsorized factor within sector and date."""
    output = pd.Series(
        np.nan,
        index=data.index,
        dtype="float64",
    )

    winsorized_column = f"{factor}_winsorized"

    for _, group in data.groupby(
        [
            "as_of_date",
            "sector",
        ],
        sort=False,
        dropna=False,
    ):
        standardized = _zscore_series(
            group[winsorized_column],
            minimum_observations=(config.min_sector_observations),
            config=config,
        )

        output.loc[group.index] = standardized

    return output


def build_processed_fundamental_features(
    data: pd.DataFrame,
    *,
    config: FundamentalTransformConfig,
) -> pd.DataFrame:
    """Build model-ready cross-sectional fundamental features."""
    required = (
        "as_of_date",
        "ticker",
        "sector",
        *FUNDAMENTAL_FACTOR_COLUMNS,
    )

    _require_columns(
        data,
        required,
    )

    result = data.copy()

    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()

    result["sector"] = result["sector"].astype("string").str.strip()

    result["as_of_date"] = pd.to_datetime(
        result["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    if result["as_of_date"].isna().any():
        raise FundamentalTransformError("Invalid as_of_date values.")

    duplicates = result.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    )

    if duplicates.any():
        raise FundamentalTransformError("Duplicate date-ticker rows found.")

    if result["sector"].isna().any():
        raise FundamentalTransformError("Sector information is missing.")

    new_columns: dict[
        str,
        pd.Series,
    ] = {}

    for factor in FUNDAMENTAL_FACTOR_COLUMNS:
        numeric = pd.to_numeric(
            result[factor],
            errors="coerce",
        )

        result[factor] = numeric

        new_columns[f"{factor}_missing"] = numeric.isna().astype("int8")

        transformed = _transform_factor_by_date(
            result,
            factor,
            config=config,
        )

        winsorized_column = f"{factor}_winsorized"

        zscore_column = f"{factor}_zscore"

        sector_column = f"{factor}_sector_zscore"

        winsorized = transformed[winsorized_column]

        new_columns[winsorized_column] = winsorized

        new_columns[zscore_column] = transformed[zscore_column]

        sector_input = result.loc[
            :,
            [
                "as_of_date",
                "sector",
            ],
        ].copy()

        sector_input[winsorized_column] = winsorized

        new_columns[sector_column] = _build_sector_zscore(
            sector_input,
            factor,
            config=config,
        )

    additions = pd.DataFrame(
        new_columns,
        index=result.index,
    )

    result = pd.concat(
        [
            result,
            additions,
        ],
        axis=1,
    ).copy()

    transformed_columns = []

    for factor in FUNDAMENTAL_FACTOR_COLUMNS:
        transformed_columns.extend(
            [
                f"{factor}_winsorized",
                f"{factor}_zscore",
                f"{factor}_sector_zscore",
            ]
        )

    transformed_data = result[transformed_columns].to_numpy(dtype=float)

    infinite_values = int(np.isinf(transformed_data).sum())

    if infinite_values:
        raise FundamentalTransformError(
            f"Processed fundamental features contain {infinite_values} infinite values."
        )

    return result.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)
