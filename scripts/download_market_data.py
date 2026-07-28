"""Download and store market data for the complete equity universe."""

from quant_equity.config import PROJECT_ROOT, load_config
from quant_equity.data import download_market_universe
from quant_equity.logging_config import configure_logging


def main() -> None:
    """Run the complete market-data download pipeline."""
    config = load_config()

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "download_market_data.log"),
    )

    result = download_market_universe(config)

    market_data = result.market_data

    logger.info("Complete market-data pipeline finished successfully.")
    logger.info(
        "Processed rows: %s",
        len(market_data),
    )
    logger.info(
        "Processed tickers: %s",
        market_data["ticker"].nunique(),
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Market-data download")
    print("------------------------------------------------")
    print(f"Downloaded tickers: {len(result.downloaded_tickers)}")
    print(f"Cached tickers: {len(result.cached_tickers)}")
    print(f"Total tickers: {market_data['ticker'].nunique()}")
    print(f"Rows: {len(market_data)}")
    print(
        f"Available range: {market_data['date'].min().date()} to {market_data['date'].max().date()}"
    )
    print(f"Raw files: {len(result.raw_files)}")
    print(f"Processed dataset: {result.processed_path}")
    print()
    print("Market-data download: OK")


if __name__ == "__main__":
    main()
