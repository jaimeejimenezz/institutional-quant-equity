"""Download and cache SPY benchmark market data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from quant_equity.config import (
    PROCESSED_DATA_DIR,
)
from quant_equity.logging_config import (
    configure_logging,
)

MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

SPY_PATH = PROCESSED_DATA_DIR / "benchmark_spy_daily.parquet"


def _normalize_yfinance_output(
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize a single-ticker yfinance response."""
    if raw_data.empty:
        raise ValueError("The SPY download returned no observations.")

    data = raw_data.copy()

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    date_column = "Date" if "Date" in data.columns else "Datetime"

    required_columns = (
        date_column,
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    )

    missing = sorted(set(required_columns).difference(data.columns))

    if missing:
        raise ValueError("SPY download is missing columns: " + ", ".join(missing) + ".")

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(
                data[date_column],
                errors="coerce",
            )
            .dt.tz_localize(None)
            .dt.normalize(),
            "ticker": "SPY",
            "open": pd.to_numeric(
                data["Open"],
                errors="coerce",
            ),
            "high": pd.to_numeric(
                data["High"],
                errors="coerce",
            ),
            "low": pd.to_numeric(
                data["Low"],
                errors="coerce",
            ),
            "close": pd.to_numeric(
                data["Close"],
                errors="coerce",
            ),
            "adjusted_close": pd.to_numeric(
                data["Adj Close"],
                errors="coerce",
            ),
            "volume": pd.to_numeric(
                data["Volume"],
                errors="coerce",
            ),
        }
    )

    if output.isna().any().any():
        raise ValueError("Normalized SPY data contain missing values.")

    if output.duplicated(
        [
            "date",
            "ticker",
        ]
    ).any():
        raise ValueError("Normalized SPY data contain duplicates.")

    price_columns = (
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
    )

    if (
        output.loc[
            :,
            price_columns,
        ]
        .le(0.0)
        .any()
        .any()
    ):
        raise ValueError("Normalized SPY data contain non-positive prices.")

    return output.sort_values("date").reset_index(drop=True)


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write a Parquet file atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(".tmp.parquet")

    temporary_path.unlink(missing_ok=True)

    data.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)


def main() -> None:
    """Download SPY over the complete project market period."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    if not MARKET_DATA_PATH.exists():
        raise FileNotFoundError(f"Market data not found: {MARKET_DATA_PATH}")

    market_data = pd.read_parquet(
        MARKET_DATA_PATH,
        columns=["date"],
    )

    market_dates = pd.to_datetime(market_data["date"])

    start_date = market_dates.min()
    final_date = market_dates.max()

    download_end_date = final_date + pd.Timedelta(days=2)

    raw_data = yf.download(
        "SPY",
        start=start_date.date().isoformat(),
        end=download_end_date.date().isoformat(),
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )

    spy_data = _normalize_yfinance_output(raw_data)

    spy_data = spy_data.loc[
        spy_data["date"].between(
            start_date,
            final_date,
        )
    ].copy()

    _write_parquet_atomically(
        spy_data,
        SPY_PATH,
    )

    logger.info("SPY benchmark download completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("SPY benchmark download")
    print("-" * 60)
    print(f"Rows: {len(spy_data)}")
    print(f"Start date: {spy_data['date'].min().date()}")
    print(f"End date: {spy_data['date'].max().date()}")
    print(f"Output: {SPY_PATH}")


if __name__ == "__main__":
    main()
