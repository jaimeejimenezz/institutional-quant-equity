"""Tests for cross-sectional technical-feature processing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_equity.features import (
    TECHNICAL_FEATURE_COLUMNS,
    TechnicalFeatureProcessingConfig,
    TechnicalFeatureProcessingError,
    build_processed_technical_features,
    validate_processed_technical_features,
    write_processed_technical_features,
)


def make_raw_features() -> pd.DataFrame:
    """Create a deterministic monthly technical panel."""
    dates = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
        ]
    )

    tickers = [
        "AAA",
        "BBB",
        "CCC",
        "DDD",
        "EEE",
        "FFF",
    ]

    rows: list[dict[str, object]] = []

    for date_position, date in enumerate(dates):
        for ticker_position, ticker in enumerate(tickers):
            row: dict[
                str,
                object,
            ] = {
                "as_of_date": date,
                "ticker": ticker,
                "latest_market_date": date,
                "observations_available": 300,
            }

            for feature_position, feature in enumerate(TECHNICAL_FEATURE_COLUMNS):
                row[feature] = ticker_position + feature_position * 0.01 + date_position * 0.1

            rows.append(row)

    features = pd.DataFrame(rows)

    extreme_row = features["as_of_date"].eq(pd.Timestamp("2024-01-31")) & features["ticker"].eq(
        "FFF"
    )

    features.loc[
        extreme_row,
        "return_1m",
    ] = 100.0

    return features


def make_universe() -> pd.DataFrame:
    """Create sector metadata for the test universe."""
    return pd.DataFrame(
        {
            "ticker": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",
                "EEE",
                "FFF",
            ],
            "sector": [
                "Sector A",
                "Sector A",
                "Sector A",
                "Sector B",
                "Sector B",
                "Sector B",
            ],
        }
    )


def make_config() -> TechnicalFeatureProcessingConfig:
    """Create a small deterministic processing config."""
    return TechnicalFeatureProcessingConfig(
        winsor_lower_quantile=0.20,
        winsor_upper_quantile=0.80,
        minimum_cross_section_size=4,
        minimum_sector_size=2,
        sector_neutralization=True,
    )


def test_processed_panel_keeps_keys_and_sectors() -> None:
    """Processing should preserve rows and add sectors."""
    raw_features = make_raw_features()

    processed = build_processed_technical_features(
        raw_features,
        make_universe(),
        processing_config=make_config(),
    )

    assert len(processed) == len(raw_features)

    assert processed["sector"].notna().all()

    assert (
        processed[
            [
                "as_of_date",
                "ticker",
            ]
        ]
        .duplicated()
        .sum()
        == 0
    )


def test_winsorization_matches_manual_quantiles() -> None:
    """Winsorization should match manual clipping."""
    raw_features = make_raw_features()

    processed = build_processed_technical_features(
        raw_features,
        make_universe(),
        processing_config=make_config(),
    )

    first_date = pd.Timestamp("2024-01-31")

    observed = processed.loc[
        processed["as_of_date"].eq(first_date),
        "return_1m_winsorized",
    ].reset_index(drop=True)

    raw_values = raw_features.loc[
        raw_features["as_of_date"].eq(first_date),
        "return_1m",
    ].reset_index(drop=True)

    expected = raw_values.clip(
        lower=raw_values.quantile(0.20),
        upper=raw_values.quantile(0.80),
    )

    assert np.allclose(
        observed,
        expected,
    )


def test_standardized_scores_have_zero_mean_and_unit_scale() -> None:
    """Date-level z-scores should be standardized."""
    processed = build_processed_technical_features(
        make_raw_features(),
        make_universe(),
        processing_config=make_config(),
    )

    grouped = processed.groupby("as_of_date")["return_1m_zscore"]

    means = grouped.mean()

    standard_deviations = grouped.std(ddof=0)

    assert np.allclose(
        means,
        0.0,
        atol=1e-12,
    )

    assert np.allclose(
        standard_deviations,
        1.0,
        atol=1e-12,
    )


def test_sector_neutral_scores_are_centered() -> None:
    """Eligible sectors should have zero mean scores."""
    processed = build_processed_technical_features(
        make_raw_features(),
        make_universe(),
        processing_config=make_config(),
    )

    sector_means = processed.groupby(
        [
            "as_of_date",
            "sector",
        ]
    )["return_1m_sector_neutral"].mean()

    assert np.allclose(
        sector_means,
        0.0,
        atol=1e-12,
    )


def test_missing_values_remain_missing() -> None:
    """Processing must not impute missing values."""
    raw_features = make_raw_features()

    missing_row = raw_features["as_of_date"].eq(pd.Timestamp("2024-01-31")) & raw_features[
        "ticker"
    ].eq("AAA")

    raw_features.loc[
        missing_row,
        "return_1m",
    ] = np.nan

    processed = build_processed_technical_features(
        raw_features,
        make_universe(),
        processing_config=make_config(),
    )

    observed = processed.loc[
        missing_row,
        [
            "return_1m",
            "return_1m_winsorized",
            "return_1m_zscore",
            "return_1m_sector_neutral",
        ],
    ]

    assert observed.isna().all(axis=None)


def test_duplicate_raw_rows_are_rejected() -> None:
    """Duplicated date-ticker rows should be rejected."""
    raw_features = make_raw_features()

    duplicated = pd.concat(
        [
            raw_features,
            raw_features.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        TechnicalFeatureProcessingError,
        match="duplicated",
    ):
        build_processed_technical_features(
            duplicated,
            make_universe(),
            processing_config=make_config(),
        )


def test_processed_writer_sorts_rows(
    tmp_path: Path,
) -> None:
    """The processed Parquet file should be sorted."""
    processed = build_processed_technical_features(
        make_raw_features(),
        make_universe(),
        processing_config=make_config(),
    )

    output_path = tmp_path / "processed_features.parquet"

    write_processed_technical_features(
        processed.iloc[::-1],
        output_path,
    )

    stored = pd.read_parquet(output_path)

    expected = stored.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        stored,
        expected,
    )


def test_validation_accepts_valid_panel() -> None:
    """A correctly processed panel should pass validation."""
    raw_features = make_raw_features()

    universe = make_universe()

    config = make_config()

    processed = build_processed_technical_features(
        raw_features,
        universe,
        processing_config=config,
    )

    result = validate_processed_technical_features(
        raw_features,
        processed,
        expected_tickers=(universe["ticker"]),
        processing_config=config,
    )

    assert result.is_valid
    assert not result.issues
