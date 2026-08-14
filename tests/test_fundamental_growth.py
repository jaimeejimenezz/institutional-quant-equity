"""Tests for fundamental growth factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_equity.features import (
    FundamentalGrowthConfig,
    FundamentalGrowthError,
    build_fundamental_growth_factors,
)


def make_config() -> FundamentalGrowthConfig:
    """Create test configuration."""
    return FundamentalGrowthConfig(
        lag_periods=12,
        acceleration_lag_periods=12,
        min_lag_days=330,
        max_lag_days=400,
        min_abs_denominator=1.0e-12,
    )


def make_history() -> pd.DataFrame:
    """Create 25 monthly observations."""
    dates = pd.date_range(
        start="2022-01-31",
        periods=25,
        freq="ME",
    )

    revenue = [100.0] * 12 + [110.0] * 12 + [121.0]

    net_income = [-10.0] * 12 + [-5.0] * 12 + [-2.5]

    operating_cash_flow = [20.0] * 12 + [30.0] * 12 + [45.0]

    assets = [200.0] * 12 + [220.0] * 12 + [242.0]

    return pd.DataFrame(
        {
            "as_of_date": dates,
            "ticker": ["AAPL"] * len(dates),
            "revenue_ttm": revenue,
            "net_income_ttm": net_income,
            "operating_cash_flow_ttm": (operating_cash_flow),
            "assets": assets,
        }
    )


def test_revenue_growth_uses_twelve_month_lag() -> None:
    """Revenue growth should compare with 12 months earlier."""
    result = build_fundamental_growth_factors(
        make_history(),
        config=make_config(),
    )

    row = result.iloc[12]

    assert row["revenue_growth_yoy"] == pytest.approx(0.10)


def test_negative_income_improvement_is_positive_growth() -> None:
    """A smaller accounting loss should be treated as improvement."""
    result = build_fundamental_growth_factors(
        make_history(),
        config=make_config(),
    )

    row = result.iloc[12]

    assert row["net_income_growth_yoy"] == pytest.approx(0.50)


def test_operating_cash_flow_growth_is_correct() -> None:
    """Cash-flow growth should use signed growth."""
    result = build_fundamental_growth_factors(
        make_history(),
        config=make_config(),
    )

    row = result.iloc[12]

    assert row["operating_cash_flow_growth_yoy"] == pytest.approx(0.50)


def test_asset_growth_is_correct() -> None:
    """Asset growth should compare positive levels."""
    result = build_fundamental_growth_factors(
        make_history(),
        config=make_config(),
    )

    row = result.iloc[12]

    assert row["asset_growth_yoy"] == pytest.approx(0.10)


def test_growth_acceleration_uses_previous_year_growth() -> None:
    """Acceleration compares current YoY growth with prior YoY growth."""
    result = build_fundamental_growth_factors(
        make_history(),
        config=make_config(),
    )

    final = result.iloc[24]

    assert final["revenue_growth_yoy"] == pytest.approx(0.10)

    assert final["revenue_growth_acceleration"] == pytest.approx(0.0)


def test_first_year_has_no_growth_signal() -> None:
    """No future or unavailable history may be invented."""
    result = build_fundamental_growth_factors(
        make_history(),
        config=make_config(),
    )

    first_year = result.iloc[:12]

    assert first_year["revenue_growth_yoy"].isna().all()

    assert first_year["asset_growth_yoy"].isna().all()


def test_growth_reference_is_always_in_the_past() -> None:
    """Growth reference dates must precede the observation."""
    result = build_fundamental_growth_factors(
        make_history(),
        config=make_config(),
    )

    valid = result["growth_reference_date"].notna()

    assert (
        result.loc[
            valid,
            "growth_reference_date",
        ]
        < result.loc[
            valid,
            "as_of_date",
        ]
    ).all()


def test_duplicate_rows_are_rejected() -> None:
    """Date-ticker rows must remain unique."""
    data = make_history()

    data = pd.concat(
        [
            data,
            data.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        FundamentalGrowthError,
        match="Duplicate",
    ):
        build_fundamental_growth_factors(
            data,
            config=make_config(),
        )


def test_growth_contains_no_infinite_values() -> None:
    """Zero prior values should not create infinities."""
    data = make_history()

    data.loc[
        0,
        "net_income_ttm",
    ] = 0.0

    result = build_fundamental_growth_factors(
        data,
        config=make_config(),
    )

    numeric = result.loc[
        :,
        [
            "revenue_growth_yoy",
            "net_income_growth_yoy",
            "operating_cash_flow_growth_yoy",
            "asset_growth_yoy",
        ],
    ].to_numpy(dtype=float)

    assert not np.isinf(numeric).any()
