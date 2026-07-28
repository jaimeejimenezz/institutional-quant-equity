"""Tests for market-data providers and normalization."""

from collections.abc import Callable

import pandas as pd
import pytest

from quant_equity.data import (
    MARKET_DATA_COLUMNS,
    MarketDataNormalizationError,
    MarketDataProvider,
    download_with_retries,
    normalize_market_data,
)


def make_provider_frame() -> pd.DataFrame:
    """Create a small yfinance-like table."""
    index = pd.DatetimeIndex(
        [
            "2024-01-02 00:00:00-05:00",
            "2024-01-03 00:00:00-05:00",
        ],
        name="Date",
    )

    return pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [103.0, 104.0],
            "Low": [99.0, 101.0],
            "Close": [102.0, 103.0],
            "Adj Close": [101.5, 102.5],
            "Volume": [1_000_000, 1_200_000],
            "Dividends": [0.0, 0.25],
            "Stock Splits": [0.0, 0.0],
        },
        index=index,
    )


class FlakyProvider(MarketDataProvider):
    """Provider that fails before returning valid data."""

    def __init__(
        self,
        failures_before_success: int,
    ) -> None:
        """Configure the number of transient failures."""
        self.failures_before_success = failures_before_success
        self.calls = 0

    def download_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fail temporarily and then return sample data."""
        del ticker
        del start_date
        del end_date
        del interval

        self.calls += 1

        if self.calls <= self.failures_before_success:
            raise RuntimeError("Temporary provider failure.")

        return make_provider_frame()


def test_provider_data_is_normalized() -> None:
    """Provider columns should map to the canonical schema."""
    result = normalize_market_data(
        make_provider_frame(),
        ticker="aapl",
    )

    assert tuple(result.columns) == MARKET_DATA_COLUMNS
    assert result["ticker"].eq("AAPL").all()
    assert result["date"].iloc[0] == pd.Timestamp("2024-01-02")
    assert result["adjusted_close"].iloc[1] == pytest.approx(102.5)
    assert result["dividends"].iloc[1] == pytest.approx(0.25)


def test_missing_adjusted_close_is_rejected() -> None:
    """An unadjusted response without Adj Close should fail."""
    invalid_data = make_provider_frame().drop(columns="Adj Close")

    with pytest.raises(
        MarketDataNormalizationError,
        match="adjusted_close",
    ):
        normalize_market_data(
            invalid_data,
            ticker="AAPL",
        )


def test_duplicate_dates_are_rejected() -> None:
    """A provider response cannot contain duplicate dates."""
    invalid_data = pd.concat(
        [
            make_provider_frame(),
            make_provider_frame().iloc[[0]],
        ]
    )

    with pytest.raises(
        MarketDataNormalizationError,
        match="Duplicate dates",
    ):
        normalize_market_data(
            invalid_data,
            ticker="AAPL",
        )


def test_download_retries_after_transient_failure() -> None:
    """A temporary provider failure should be retried."""
    provider = FlakyProvider(
        failures_before_success=1,
    )

    recorded_waits: list[float] = []

    def record_wait(seconds: float) -> None:
        recorded_waits.append(seconds)

    result = download_with_retries(
        provider=provider,
        ticker="AAPL",
        start_date="2024-01-01",
        end_date="2024-02-01",
        max_retries=3,
        retry_wait_seconds=2,
        sleeper=record_wait,
    )

    assert not result.empty
    assert provider.calls == 2
    assert recorded_waits == [2]


def test_download_does_not_wait_after_success() -> None:
    """No retry delay should occur after a successful request."""
    provider = FlakyProvider(
        failures_before_success=0,
    )

    unexpected_wait: Callable[[float], None]

    def fail_if_called(seconds: float) -> None:
        raise AssertionError(f"Unexpected retry wait of {seconds} seconds.")

    unexpected_wait = fail_if_called

    result = download_with_retries(
        provider=provider,
        ticker="AAPL",
        start_date="2024-01-01",
        end_date="2024-02-01",
        max_retries=3,
        sleeper=unexpected_wait,
    )

    assert not result.empty
    assert provider.calls == 1
