"""Batch market-data downloading, caching and processed storage."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_equity.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from quant_equity.data.market import (
    MarketDataDownloadError,
    create_market_data_provider,
    download_with_retries,
    normalize_market_data,
    resolve_download_end_date,
)
from quant_equity.data.universe import load_universe

LOGGER = logging.getLogger("quant_equity.market_pipeline")

DEFAULT_PROCESSED_MARKET_PATH = (
    PROCESSED_DATA_DIR / "market_daily.parquet"
)


class MarketUniverseDownloadError(MarketDataDownloadError):
    """Raised when one or more universe members cannot be downloaded."""


@dataclass
class MarketDownloadResult:
    """Result of a complete universe market-data download."""

    market_data: pd.DataFrame
    downloaded_tickers: tuple[str, ...]
    cached_tickers: tuple[str, ...]
    raw_files: tuple[Path, ...]
    processed_path: Path


def get_raw_market_path(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    provider_name: str = "yfinance",
) -> Path:
    """Return the immutable raw snapshot path for one request."""
    safe_ticker = re.sub(
        r"[^A-Z0-9.-]",
        "_",
        ticker.strip().upper(),
    )

    safe_provider = re.sub(
        r"[^a-z0-9_-]",
        "_",
        provider_name.strip().lower(),
    )

    filename = (
        f"{safe_ticker}"
        f"__{start_date}"
        f"__{end_date}"
        ".parquet"
    )

    return (
        RAW_DATA_DIR
        / "market"
        / safe_provider
        / filename
    )


def save_raw_market_snapshot(
    provider_data: pd.DataFrame,
    path: Path,
) -> None:
    """Persist a provider response without overwriting existing raw data."""
    if path.exists():
        raise FileExistsError(
            f"Raw market snapshot already exists: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_data = provider_data.reset_index()
    serializable_data.columns = [
        str(column)
        for column in serializable_data.columns
    ]

    temporary_path = path.with_suffix(".tmp.parquet")
    temporary_path.unlink(missing_ok=True)

    serializable_data.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)


def load_raw_market_snapshot(
    path: Path,
) -> pd.DataFrame:
    """Load a previously stored provider response."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw market snapshot not found: {path}"
        )

    return pd.read_parquet(path)


def write_processed_market_data(
    market_data: pd.DataFrame,
    path: Path = DEFAULT_PROCESSED_MARKET_PATH,
) -> Path:
    """Write the canonical long-format market dataset atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered_data = (
        market_data
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    temporary_path = path.with_suffix(".tmp.parquet")
    temporary_path.unlink(missing_ok=True)

    ordered_data.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)

    return path


def download_market_universe(
    config: dict[str, Any],
    *,
    universe_version: str | None = None,
) -> MarketDownloadResult:
    """Download, cache and normalize all securities in the universe."""
    market_config = config["market_data"]

    resolved_universe_version = (
        universe_version
        if universe_version is not None
        else str(config["universe"]["version"])
    )

    universe = load_universe(
        resolved_universe_version
    )

    provider_name = str(
        market_config["provider"]
    ).strip().lower()

    start_date = str(
        market_config["start_date"]
    )

    end_date = resolve_download_end_date(
        market_config.get("end_date")
    )

    interval = str(
        market_config["interval"]
    )

    provider = create_market_data_provider(config)

    normalized_frames: list[pd.DataFrame] = []
    downloaded_tickers: list[str] = []
    cached_tickers: list[str] = []
    raw_files: list[Path] = []
    failures: dict[str, str] = {}

    total_tickers = len(universe)

    for position, ticker in enumerate(
        universe["ticker"],
        start=1,
    ):
        LOGGER.info(
            "Processing %s (%s/%s).",
            ticker,
            position,
            total_tickers,
        )

        raw_path = get_raw_market_path(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            provider_name=provider_name,
        )

        try:
            if raw_path.exists():
                LOGGER.info(
                    "Using cached raw snapshot for %s.",
                    ticker,
                )

                provider_data = load_raw_market_snapshot(
                    raw_path
                )

                cached_tickers.append(ticker)
            else:
                provider_data = download_with_retries(
                    provider=provider,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    max_retries=int(
                        market_config["max_retries"]
                    ),
                    retry_wait_seconds=float(
                        market_config[
                            "retry_wait_seconds"
                        ]
                    ),
                )

                save_raw_market_snapshot(
                    provider_data,
                    raw_path,
                )

                downloaded_tickers.append(ticker)

            normalized_data = normalize_market_data(
                provider_data,
                ticker=ticker,
            )

            normalized_frames.append(
                normalized_data
            )

            raw_files.append(raw_path)

            LOGGER.info(
                "%s completed with %s observations.",
                ticker,
                len(normalized_data),
            )
        except Exception as error:
            failures[ticker] = str(error)

            LOGGER.exception(
                "Unable to process %s.",
                ticker,
            )

    if failures:
        failure_details = "; ".join(
            f"{ticker}: {message}"
            for ticker, message in sorted(
                failures.items()
            )
        )

        raise MarketUniverseDownloadError(
            "The complete market universe could not be "
            f"downloaded. Failures: {failure_details}"
        )

    if not normalized_frames:
        raise MarketUniverseDownloadError(
            "No market-data observations were produced."
        )

    market_data = (
        pd.concat(
            normalized_frames,
            ignore_index=True,
        )
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    processed_path = write_processed_market_data(
        market_data
    )

    LOGGER.info(
        "Processed market dataset written to %s.",
        processed_path,
    )

    return MarketDownloadResult(
        market_data=market_data,
        downloaded_tickers=tuple(
            downloaded_tickers
        ),
        cached_tickers=tuple(
            cached_tickers
        ),
        raw_files=tuple(raw_files),
        processed_path=processed_path,
    )