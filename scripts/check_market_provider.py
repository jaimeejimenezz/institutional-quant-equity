"""Test the configured market-data provider with one ticker."""

from quant_equity.config import PROJECT_ROOT, load_config
from quant_equity.data import (
    create_market_data_provider,
    download_with_retries,
    normalize_market_data,
    resolve_download_end_date,
)
from quant_equity.logging_config import configure_logging


def main() -> None:
    """Download and normalize a single ticker."""
    config = load_config()
    market_config = config["market_data"]

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=PROJECT_ROOT / "logs" / "market_provider.log",
    )

    ticker = "AAPL"
    start_date = str(market_config["start_date"])
    end_date = resolve_download_end_date(market_config.get("end_date"))
    interval = str(market_config["interval"])

    provider = create_market_data_provider(config)

    raw_data = download_with_retries(
        provider=provider,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        max_retries=int(market_config["max_retries"]),
        retry_wait_seconds=float(market_config["retry_wait_seconds"]),
    )

    market_data = normalize_market_data(
        raw_data,
        ticker=ticker,
    )

    logger.info(
        "%s downloaded and normalized successfully.",
        ticker,
    )
    logger.info(
        "Observations: %s",
        len(market_data),
    )
    logger.info(
        "First date: %s",
        market_data["date"].min().date(),
    )
    logger.info(
        "Last date: %s",
        market_data["date"].max().date(),
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Market-data provider check")
    print("------------------------------------------------")
    print(f"Provider: {market_config['provider']}")
    print(f"Ticker: {ticker}")
    print(f"Requested start: {start_date}")
    print(f"Exclusive end: {end_date}")
    print(f"Observations: {len(market_data)}")
    print(
        f"Available range: {market_data['date'].min().date()} to {market_data['date'].max().date()}"
    )
    print()
    print("Canonical columns")
    print("------------------------------------------------")
    print(", ".join(market_data.columns))
    print()
    print("Last five observations")
    print("------------------------------------------------")
    print(
        market_data.tail().to_string(
            index=False,
        )
    )
    print()
    print("Market-data provider check: OK")


if __name__ == "__main__":
    main()
