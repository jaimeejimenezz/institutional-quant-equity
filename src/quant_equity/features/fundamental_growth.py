"""Point-in-time fundamental growth factors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class FundamentalGrowthError(ValueError):
    """Raised when fundamental growth factors cannot be built."""


GROWTH_FACTOR_COLUMNS = (
    "revenue_growth_yoy",
    "net_income_growth_yoy",
    "operating_cash_flow_growth_yoy",
    "asset_growth_yoy",
    "revenue_growth_acceleration",
    "net_income_growth_acceleration",
    "operating_cash_flow_growth_acceleration",
)


@dataclass(frozen=True)
class FundamentalGrowthConfig:
    """Configuration for fundamental growth factors."""

    lag_periods: int = 12
    acceleration_lag_periods: int = 12
    min_lag_days: int = 330
    max_lag_days: int = 400
    min_abs_denominator: float = 1.0e-12

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> FundamentalGrowthConfig:
        """Create configuration from YAML."""
        config = cls(
            lag_periods=int(
                values.get(
                    "lag_periods",
                    12,
                )
            ),
            acceleration_lag_periods=int(
                values.get(
                    "acceleration_lag_periods",
                    12,
                )
            ),
            min_lag_days=int(
                values.get(
                    "min_lag_days",
                    330,
                )
            ),
            max_lag_days=int(
                values.get(
                    "max_lag_days",
                    400,
                )
            ),
            min_abs_denominator=float(
                values.get(
                    "min_abs_denominator",
                    1.0e-12,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate growth configuration."""
        if self.lag_periods < 1:
            raise FundamentalGrowthError("lag_periods must be positive.")

        if self.acceleration_lag_periods < 1:
            raise FundamentalGrowthError("acceleration_lag_periods must be positive.")

        if self.min_lag_days < 1 or self.max_lag_days < self.min_lag_days:
            raise FundamentalGrowthError("Invalid lag-day limits.")

        if self.min_abs_denominator <= 0:
            raise FundamentalGrowthError("min_abs_denominator must be positive.")


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    """Require input columns."""
    missing = [column for column in columns if column not in data.columns]

    if missing:
        raise FundamentalGrowthError(f"Raw fundamental dataset is missing columns: {missing}")


def _positive_base_growth(
    current: pd.Series,
    previous: pd.Series,
    *,
    epsilon: float,
) -> pd.Series:
    """Calculate growth when the prior value must be positive."""
    current_numeric = pd.to_numeric(
        current,
        errors="coerce",
    )

    previous_numeric = pd.to_numeric(
        previous,
        errors="coerce",
    )

    valid = current_numeric.notna() & previous_numeric.notna() & previous_numeric.gt(epsilon)

    result = pd.Series(
        np.nan,
        index=current.index,
        dtype="float64",
    )

    result.loc[valid] = current_numeric.loc[valid] / previous_numeric.loc[valid] - 1.0

    return result


def _signed_growth(
    current: pd.Series,
    previous: pd.Series,
    *,
    epsilon: float,
) -> pd.Series:
    """Calculate growth for metrics that may be negative."""
    current_numeric = pd.to_numeric(
        current,
        errors="coerce",
    )

    previous_numeric = pd.to_numeric(
        previous,
        errors="coerce",
    )

    valid = current_numeric.notna() & previous_numeric.notna() & previous_numeric.abs().gt(epsilon)

    result = pd.Series(
        np.nan,
        index=current.index,
        dtype="float64",
    )

    result.loc[valid] = (
        current_numeric.loc[valid] - previous_numeric.loc[valid]
    ) / previous_numeric.loc[valid].abs()

    return result


def build_fundamental_growth_factors(
    data: pd.DataFrame,
    *,
    config: FundamentalGrowthConfig,
) -> pd.DataFrame:
    """Build point-in-time year-over-year growth factors."""
    required = (
        "as_of_date",
        "ticker",
        "revenue_ttm",
        "net_income_ttm",
        "operating_cash_flow_ttm",
        "assets",
    )

    _require_columns(
        data,
        required,
    )

    result = data.copy()

    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()

    result["as_of_date"] = pd.to_datetime(
        result["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    if result["as_of_date"].isna().any():
        raise FundamentalGrowthError("Invalid as_of_date values.")

    duplicates = result.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    )

    if duplicates.any():
        raise FundamentalGrowthError("Duplicate date-ticker rows found.")

    result = result.sort_values(
        [
            "ticker",
            "as_of_date",
        ]
    ).reset_index(drop=True)

    grouped = result.groupby(
        "ticker",
        sort=False,
    )

    result["growth_reference_date"] = grouped["as_of_date"].shift(config.lag_periods)

    reference_gap_days = (result["as_of_date"] - result["growth_reference_date"]).dt.days

    has_reference = result["growth_reference_date"].notna()

    invalid_gap = has_reference & (
        reference_gap_days.lt(config.min_lag_days) | reference_gap_days.gt(config.max_lag_days)
    )

    if invalid_gap.any():
        raise FundamentalGrowthError("Growth lag does not correspond to approximately one year.")

    if (
        result.loc[
            has_reference,
            "growth_reference_date",
        ]
        >= result.loc[
            has_reference,
            "as_of_date",
        ]
    ).any():
        raise FundamentalGrowthError("Growth reference dates contain future information.")

    lag_columns = (
        "revenue_ttm",
        "net_income_ttm",
        "operating_cash_flow_ttm",
        "assets",
    )

    for column in lag_columns:
        result[f"{column}_lag_12m"] = grouped[column].shift(config.lag_periods)

    epsilon = config.min_abs_denominator

    result["revenue_growth_yoy"] = _positive_base_growth(
        result["revenue_ttm"],
        result["revenue_ttm_lag_12m"],
        epsilon=epsilon,
    )

    result["net_income_growth_yoy"] = _signed_growth(
        result["net_income_ttm"],
        result["net_income_ttm_lag_12m"],
        epsilon=epsilon,
    )

    result["operating_cash_flow_growth_yoy"] = _signed_growth(
        result["operating_cash_flow_ttm"],
        result["operating_cash_flow_ttm_lag_12m"],
        epsilon=epsilon,
    )

    result["asset_growth_yoy"] = _positive_base_growth(
        result["assets"],
        result["assets_lag_12m"],
        epsilon=epsilon,
    )

    acceleration_sources = (
        "revenue_growth_yoy",
        "net_income_growth_yoy",
        "operating_cash_flow_growth_yoy",
    )

    for column in acceleration_sources:
        previous_growth = result.groupby(
            "ticker",
            sort=False,
        )[column].shift(config.acceleration_lag_periods)

        if column == "revenue_growth_yoy":
            output_column = "revenue_growth_acceleration"
        elif column == "net_income_growth_yoy":
            output_column = "net_income_growth_acceleration"
        else:
            output_column = "operating_cash_flow_growth_acceleration"

        result[output_column] = result[column] - previous_growth

    factor_data = result.loc[
        :,
        list(GROWTH_FACTOR_COLUMNS),
    ]

    infinite_count = int(np.isinf(factor_data.to_numpy(dtype=float)).sum())

    if infinite_count:
        raise FundamentalGrowthError(f"Growth factors contain {infinite_count} infinite values.")

    result["growth_factor_count"] = factor_data.notna().sum(axis=1)

    return result.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)
