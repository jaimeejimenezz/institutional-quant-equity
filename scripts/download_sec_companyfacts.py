"""Download raw SEC Company Facts for the equity universe."""

from __future__ import annotations

from quant_equity.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    load_config,
)
from quant_equity.data import (
    download_sec_companyfacts_universe,
)
from quant_equity.logging_config import (
    configure_logging,
)


def main() -> None:
    """Run Step 9A SEC Company Facts download."""
    config = load_config()

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "download_sec_companyfacts.log"),
    )

    result = download_sec_companyfacts_universe(config)

    total_concepts = sum(record.concept_count for record in result.records)

    logger.info("SEC Company Facts download completed.")

    logger.info(
        "Companies: %s",
        len(result.records),
    )

    logger.info(
        "Downloaded: %s",
        len(result.downloaded_tickers),
    )

    logger.info(
        "Cached: %s",
        len(result.cached_tickers),
    )

    print()
    print("Institutional Quant Equity Research Platform")

    print("SEC Company Facts - Step 9A")

    print("------------------------------------------------")

    print(f"Companies: {len(result.records)}")

    print(f"Downloaded: {len(result.downloaded_tickers)}")

    print(f"Loaded from cache: {len(result.cached_tickers)}")

    print(f"Total concepts: {total_concepts}")

    print(f"Raw files: {len(result.raw_files)}")

    print(f"Raw directory: {RAW_DATA_DIR / 'fundamentals' / 'sec' / 'companyfacts'}")

    print()

    print("SEC Company Facts download: OK")


if __name__ == "__main__":
    main()
