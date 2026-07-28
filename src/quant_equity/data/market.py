"""Market-data provider abstractions and normalization utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from time import sleep
from typing import Any

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger("quant_equity.market_data")

MARKET_DATA_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "dividends",
    "stock_splits",
)

PROVIDER_COLUMN_MAPPING = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}


class MarketDataError(RuntimeError):
    """Base exception for market-data errors."""


class MarketDataDownloadError(MarketDataError):
    """Raised when market data cannot be downloaded."""


class MarketDataNormalizationError(MarketDataError):
    """Raised when provider data cannot be normalized."""


class MarketDataProvider(ABC):
    """Interface implemented by market-data providers."""

    @abstractmethod
    def download_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download market data for one security."""


class YFinanceMarketDataProvider(MarketDataProvider):
    """Market-data provider backed by yfinance."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        auto_adjust: bool = False,
        actions: bool = True,
        repair: bool = False,
        keep_na: bool = True,
    ) -> None:
        """Initialize the yfinance provider."""
        self.timeout_seconds = timeout_seconds
        self.auto_adjust = auto_adjust
        self.actions = actions
        self.repair = repair
        self.keep_na = keep_na

    def download_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download one ticker from yfinance.

        The end date is exclusive, following the yfinance convention.
        """
        normalized_ticker = ticker.strip().upper()

        LOGGER.info(
            "Downloading %s from %s to %s.",
            normalized_ticker,
            start_date,
            end_date,
        )

        try:
            history = yf.Ticker(normalized_ticker).history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=self.auto_adjust,
                actions=self.actions,
                repair=self.repair,
                keepna=self.keep_na,
                timeout=self.timeout_seconds,
            )
        except Exception as error:
            raise MarketDataDownloadError(
                f"Provider request failed for {normalized_ticker}: {error}"
            ) from error

        if history is None or history.empty:
            raise MarketDataDownloadError(
                f"The provider returned no observations for {normalized_ticker}."
            )

        return history.copy()


def create_market_data_provider(
    config: dict[str, Any],
) -> MarketDataProvider:
    """Create the market-data provider defined in configuration."""
    try:
        market_config = config["market_data"]
        provider_name = str(market_config["provider"]).strip().lower()
    except (KeyError, TypeError) as error:
        raise MarketDataError(
            "The market_data configuration is missing or invalid."
        ) from error

    if provider_name != "yfinance":
        raise MarketDataError(
            f"Unsupported market-data provider: {provider_name}"
        )

    return YFinanceMarketDataProvider(
        timeout_seconds=float(
            market_config.get("timeout_seconds", 30)
        ),
        auto_adjust=bool(
            market_config.get("auto_adjust", False)
        ),
        actions=bool(
            market_config.get("actions", True)
        ),
        repair=bool(
            market_config.get("repair", False)
        ),
        keep_na=bool(
            market_config.get("keep_na", True)
        ),
    )


def resolve_download_end_date(
    configured_end_date: str | None,
) -> str:
    """Resolve the exclusive download end date.

    When no date is configured, today's UTC date is used. Because the
    provider interprets the end date as exclusive, the current session
    will not be included.
    """
    if configured_end_date is None:
        return datetime.now(UTC).date().isoformat()

    normalized_value = str(configured_end_date).strip()

    if not normalized_value:
        return datetime.now(UTC).date().isoformat()

    parsed_date = pd.to_datetime(
        normalized_value,
        errors="raise",
    )

    return parsed_date.date().isoformat()


def download_with_retries(
    provider: MarketDataProvider,
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    interval: str = "1d",
    max_retries: int = 3,
    retry_wait_seconds: float = 5.0,
    sleeper: Callable[[float], None] = sleep,
) -> pd.DataFrame:
    """Download market data with bounded retries."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least one.")

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return provider.download_prices(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
        except Exception as error:
            last_error = error

            LOGGER.warning(
                "Download attempt %s/%s failed for %s: %s",
                attempt,
                max_retries,
                ticker,
                error,
            )

            if attempt < max_retries:
                wait_seconds = retry_wait_seconds * attempt

                LOGGER.info(
                    "Retrying %s in %.1f seconds.",
                    ticker,
                    wait_seconds,
                )

                sleeper(wait_seconds)

    raise MarketDataDownloadError(
        f"Unable to download {ticker} after {max_retries} attempts."
    ) from last_error


def normalize_market_data(
    provider_data: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Convert provider data into the canonical market-data schema."""
    if provider_data.empty:
        raise MarketDataNormalizationError(
            "Cannot normalize an empty market-data table."
        )

    if isinstance(provider_data.columns, pd.MultiIndex):
        raise MarketDataNormalizationError(
            "MultiIndex provider columns are not supported for "
            "single-ticker normalization."
        )

    normalized_ticker = ticker.strip().upper()
    data = provider_data.copy()

    data.index.name = data.index.name or "Date"
    data = data.reset_index()

    date_candidates = [
        column
        for column in data.columns
        if str(column).strip().lower() in {"date", "datetime", "index"}
    ]

    if not date_candidates:
        raise MarketDataNormalizationError(
            "The provider table does not contain a date index or column."
        )

    date_column = date_candidates[0]

    data = data.rename(
        columns={
            date_column: "date",
            **PROVIDER_COLUMN_MAPPING,
        }
    )

    required_provider_columns = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    }

    missing_columns = sorted(
        required_provider_columns.difference(data.columns)
    )

    if missing_columns:
        missing = ", ".join(missing_columns)

        raise MarketDataNormalizationError(
            f"Missing required market-data columns: {missing}"
        )

    if "dividends" not in data.columns:
        data["dividends"] = 0.0

    if "stock_splits" not in data.columns:
        data["stock_splits"] = 0.0

    parsed_dates = pd.to_datetime(
        data["date"],
        errors="coerce",
        utc=True,
    )

    data["date"] = (
        parsed_dates
        .dt.tz_convert(None)
        .dt.normalize()
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dividends",
        "stock_splits",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data.insert(1, "ticker", normalized_ticker)

    data = (
        data.loc[:, MARKET_DATA_COLUMNS]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if data["date"].isna().any():
        raise MarketDataNormalizationError(
            f"Invalid dates were returned for {normalized_ticker}."
        )

    if data["date"].duplicated().any():
        raise MarketDataNormalizationError(
            f"Duplicate dates were returned for {normalized_ticker}."
        )

    return data