"""Tests for fundamental cross-sectional transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_equity.features import (
    FundamentalTransformConfig,
    FundamentalTransformError,
    build_processed_fundamental_features,
)
from quant_equity.features.fundamental_transforms import (
    FUNDAMENTAL_FACTOR_COLUMNS,
)


def make_config() -> FundamentalTransformConfig:
    """Create a small transformation config."""
    return FundamentalTransformConfig(
        winsor_lower_quantile=0.10,
        winsor_upper_quantile=0.90,
        min_cross_section_observations=4,
        min_sector_observations=2,
        zscore_ddof=0,
        zero_std_value=0.0,
    )


def make_data() -> pd.DataFrame:
    """Create two dates and two sectors."""
    rows = []

    tickers = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]

    sectors = [
        "Technology",
        "Technology",
        "Technology",
        "Financials",
        "Financials",
        "Financials",
    ]

    for date_index, date in enumerate(
        [
            "2024-01-31",
            "2024-02-29",
        ]
    ):
        for position, (
            ticker,
            sector,
        ) in enumerate(
            zip(
                tickers,
                sectors,
                strict=True,
            )
        ):
            row = {
                "as_of_date": date,
                "ticker": ticker,
                "sector": sector,
            }

            for factor in FUNDAMENTAL_FACTOR_COLUMNS:
                row[factor] = float(position + 1 + date_index)

            rows.append(row)

    return pd.DataFrame(rows)


def test_missingness_is_preserved_and_flagged() -> None:
    """Missing raw values should remain missing and receive a flag."""
    data = make_data()

    data.loc[
        (data["as_of_date"].eq("2024-01-31") & data["ticker"].eq("A")),
        "roe",
    ] = np.nan

    result = build_processed_fundamental_features(
        data,
        config=make_config(),
    )

    row = result.loc[
        result["as_of_date"].eq(pd.Timestamp("2024-01-31")) & result["ticker"].eq("A")
    ].iloc[0]

    assert pd.isna(row["roe"])

    assert row["roe_missing"] == 1

    assert pd.isna(row["roe_winsorized"])


def test_winsorization_limits_extreme_values() -> None:
    """An extreme factor should be clipped cross-sectionally."""
    data = make_data()

    mask = data["as_of_date"].eq("2024-01-31") & data["ticker"].eq("F")

    data.loc[
        mask,
        "roe",
    ] = 1000.0

    result = build_processed_fundamental_features(
        data,
        config=make_config(),
    )

    original = result.loc[
        (result["as_of_date"].eq(pd.Timestamp("2024-01-31")) & result["ticker"].eq("F")),
        "roe",
    ].iloc[0]

    winsorized = result.loc[
        (result["as_of_date"].eq(pd.Timestamp("2024-01-31")) & result["ticker"].eq("F")),
        "roe_winsorized",
    ].iloc[0]

    assert original == 1000.0

    assert winsorized < original


def test_global_zscore_has_zero_mean() -> None:
    """Cross-sectional z-scores should be centered by date."""
    result = build_processed_fundamental_features(
        make_data(),
        config=make_config(),
    )

    first_date = result.loc[result["as_of_date"].eq(pd.Timestamp("2024-01-31"))]

    assert first_date["roe_zscore"].mean() == pytest.approx(
        0.0,
        abs=1.0e-12,
    )


def test_sector_zscore_is_centered_within_sector() -> None:
    """Sector scores should compare companies within their sector."""
    result = build_processed_fundamental_features(
        make_data(),
        config=make_config(),
    )

    first_date = result.loc[result["as_of_date"].eq(pd.Timestamp("2024-01-31"))]

    means = first_date.groupby("sector")["roe_sector_zscore"].mean()

    assert np.allclose(
        means.to_numpy(),
        0.0,
        atol=1.0e-12,
    )


def test_different_dates_are_transformed_independently() -> None:
    """A future cross-section must not affect an earlier date."""
    original = make_data()

    baseline = build_processed_fundamental_features(
        original,
        config=make_config(),
    )

    modified = make_data()

    modified.loc[
        modified["as_of_date"].eq("2024-02-29"),
        "roe",
    ] = 1000000.0

    changed = build_processed_fundamental_features(
        modified,
        config=make_config(),
    )

    baseline_first = baseline.loc[
        baseline["as_of_date"].eq(pd.Timestamp("2024-01-31")),
        "roe_zscore",
    ].reset_index(drop=True)

    changed_first = changed.loc[
        changed["as_of_date"].eq(pd.Timestamp("2024-01-31")),
        "roe_zscore",
    ].reset_index(drop=True)

    pd.testing.assert_series_equal(
        baseline_first,
        changed_first,
    )


def test_zero_cross_sectional_variation_becomes_zero_score() -> None:
    """A factor with no variation should have neutral z-score."""
    data = make_data()

    data["roe"] = 0.25

    result = build_processed_fundamental_features(
        data,
        config=make_config(),
    )

    valid = result["roe_zscore"].dropna()

    assert (valid == 0.0).all()


def test_duplicate_rows_are_rejected() -> None:
    """Date-ticker rows must remain unique."""
    data = make_data()

    data = pd.concat(
        [
            data,
            data.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        FundamentalTransformError,
        match="Duplicate",
    ):
        build_processed_fundamental_features(
            data,
            config=make_config(),
        )


def test_processed_features_contain_no_infinities() -> None:
    """Transformations must never generate infinite values."""
    result = build_processed_fundamental_features(
        make_data(),
        config=make_config(),
    )

    columns = []

    for factor in FUNDAMENTAL_FACTOR_COLUMNS:
        columns.extend(
            [
                f"{factor}_winsorized",
                f"{factor}_zscore",
                f"{factor}_sector_zscore",
            ]
        )

    values = result[columns].to_numpy(dtype=float)

    assert not np.isinf(values).any()
