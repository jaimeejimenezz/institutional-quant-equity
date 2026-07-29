"""Tests for point-in-time technical features."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_equity.features import (
    TechnicalFeatureError,
    build_raw_technical_features,
    write_raw_technical_features,
)


def make_market_data(
    days: int = 300,
) -> pd.DataFrame:
    """Create deterministic market data for two stocks."""
    dates = pd.bdate_range(
        "2023-01-02",
        periods=days,
    )

    rows: list[dict[str, object]] = []

    settings = [
        (
            "AAA",
            1.0010,
            100.0,
            100,
        ),
        (
            "BBB",
            1.0004,
            80.0,
            50,
        ),
    ]

    for (
        ticker,
        growth,
        initial_price,
        volume_increment,
    ) in settings:
        prices = initial_price * np.power(
            growth,
            np.arange(days),
        )

        volumes = 1_000_000 + np.arange(days) * volume_increment

        for (
            date,
            price,
            volume,
        ) in zip(
            dates,
            prices,
            volumes,
            strict=True,
        ):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "close": price,
                    "adjusted_close": price,
                    "volume": volume,
                }
            )

    return pd.DataFrame(rows)


def test_manual_momentum_and_returns() -> None:
    """Return features should match manual calculations."""
    market_data = make_market_data()

    as_of_date = market_data["date"].max()

    calendar = pd.DataFrame({"as_of_date": [as_of_date]})

    result = build_raw_technical_features(
        market_data,
        calendar,
    )

    aaa = result.loc[result["ticker"].eq("AAA")].iloc[0]

    prices = market_data.loc[
        market_data["ticker"].eq("AAA"),
        "adjusted_close",
    ].reset_index(drop=True)

    assert aaa["momentum_12_1"] == pytest.approx(prices.iloc[-22] / prices.iloc[-253] - 1.0)

    assert aaa["momentum_6_1"] == pytest.approx(prices.iloc[-22] / prices.iloc[-127] - 1.0)

    assert aaa["return_3m"] == pytest.approx(prices.iloc[-1] / prices.iloc[-64] - 1.0)

    assert aaa["return_1m"] == pytest.approx(prices.iloc[-1] / prices.iloc[-22] - 1.0)

    assert aaa["reversal_1m"] == pytest.approx(-aaa["return_1m"])


def test_future_data_does_not_change_features() -> None:
    """Observations after as-of must not affect features."""
    market_data = make_market_data()

    as_of_date = market_data["date"].max()

    calendar = pd.DataFrame({"as_of_date": [as_of_date]})

    baseline = build_raw_technical_features(
        market_data,
        calendar,
    )

    future_date = as_of_date + pd.offsets.BDay(1)

    future_data = pd.DataFrame(
        {
            "date": [
                future_date,
                future_date,
            ],
            "ticker": [
                "AAA",
                "BBB",
            ],
            "close": [
                9_999.0,
                1.0,
            ],
            "adjusted_close": [
                9_999.0,
                1.0,
            ],
            "volume": [
                9_999_999,
                9_999_999,
            ],
        }
    )

    extended_data = pd.concat(
        [
            market_data,
            future_data,
        ],
        ignore_index=True,
    )

    changed = build_raw_technical_features(
        extended_data,
        calendar,
    )

    pd.testing.assert_frame_equal(
        baseline,
        changed,
    )


def test_short_history_produces_expected_missing_values() -> None:
    """Long-window features should be missing initially."""
    market_data = make_market_data(days=100)

    calendar = pd.DataFrame({"as_of_date": [market_data["date"].max()]})

    result = build_raw_technical_features(
        market_data,
        calendar,
    )

    assert result["return_1m"].notna().all()

    assert result["volatility_60d"].notna().all()

    assert result["momentum_12_1"].isna().all()

    assert result["distance_sma_200d"].isna().all()


def test_beta_matches_manual_calculation() -> None:
    """Market beta should match a manual covariance calculation."""
    market_data = make_market_data()

    calendar = pd.DataFrame({"as_of_date": [market_data["date"].max()]})

    result = build_raw_technical_features(
        market_data,
        calendar,
    )

    observed_beta = result.loc[
        result["ticker"].eq("AAA"),
        "beta_60d_market",
    ].item()

    prices = market_data.pivot(
        index="date",
        columns="ticker",
        values="adjusted_close",
    )

    returns = prices.pct_change(fill_method=None)

    market_returns = returns.mean(axis=1)

    aligned = (
        pd.DataFrame(
            {
                "stock": returns["AAA"],
                "market": (market_returns),
            }
        )
        .dropna()
        .tail(60)
    )

    expected_beta = aligned["stock"].cov(aligned["market"]) / aligned["market"].var(ddof=1)

    assert observed_beta == pytest.approx(expected_beta)


def test_duplicate_market_rows_are_rejected() -> None:
    """Duplicated date-ticker rows should be rejected."""
    market_data = make_market_data()

    duplicated_data = pd.concat(
        [
            market_data,
            market_data.iloc[[0]],
        ],
        ignore_index=True,
    )

    calendar = pd.DataFrame({"as_of_date": [market_data["date"].max()]})

    with pytest.raises(
        TechnicalFeatureError,
        match="duplicated",
    ):
        build_raw_technical_features(
            duplicated_data,
            calendar,
        )


def test_output_is_sorted_and_temporally_valid() -> None:
    """The monthly output should be sorted and point-in-time safe."""
    market_data = make_market_data()

    dates = sorted(market_data["date"].unique())

    calendar = pd.DataFrame(
        {
            "as_of_date": [
                dates[-1],
                dates[-21],
            ]
        }
    )

    result = build_raw_technical_features(
        market_data,
        calendar,
    )

    assert len(result) == 4

    expected = result.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        result,
        expected,
    )

    assert result["latest_market_date"].le(result["as_of_date"]).all()


def test_raw_feature_writer_sorts_data(
    tmp_path: Path,
) -> None:
    """The intermediate Parquet file should be sorted."""
    market_data = make_market_data()

    dates = sorted(market_data["date"].unique())

    calendar = pd.DataFrame(
        {
            "as_of_date": [
                dates[-1],
                dates[-21],
            ]
        }
    )

    features = build_raw_technical_features(
        market_data,
        calendar,
    )

    output_path = tmp_path / "technical_features.parquet"

    write_raw_technical_features(
        features.iloc[::-1],
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


def test_missing_price_rows_are_skipped() -> None:
    """Missing provider observations should be skipped safely."""
    market_data = make_market_data()

    as_of_date = market_data["date"].max()

    missing_row = market_data["ticker"].eq("AAA") & market_data["date"].eq(as_of_date)

    market_data.loc[
        missing_row,
        [
            "close",
            "adjusted_close",
        ],
    ] = np.nan

    market_data.loc[
        missing_row,
        "volume",
    ] = 0

    calendar = pd.DataFrame({"as_of_date": [as_of_date]})

    result = build_raw_technical_features(
        market_data,
        calendar,
    )

    assert len(result) == 2

    aaa = result.loc[result["ticker"].eq("AAA")].iloc[0]

    bbb = result.loc[result["ticker"].eq("BBB")].iloc[0]

    assert aaa["latest_market_date"] < as_of_date

    assert bbb["latest_market_date"] == as_of_date

    assert result["latest_market_date"].le(result["as_of_date"]).all()
