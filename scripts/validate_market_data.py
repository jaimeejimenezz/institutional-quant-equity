"""Validate the complete processed market dataset."""

import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.data import (
    load_universe,
    resolve_download_end_date,
    validate_market_data,
    write_market_data_report,
)
from quant_equity.logging_config import configure_logging


def main() -> None:
    """Validate market data and generate the quality report."""
    config = load_config()
    market_config = config["market_data"]

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "validate_market_data.log"),
    )

    market_path = PROCESSED_DATA_DIR / "market_daily.parquet"

    if not market_path.exists():
        raise FileNotFoundError(
            "Processed market data not found. Run 'python scripts/download_market_data.py' first."
        )

    universe = load_universe(str(config["universe"]["version"]))

    market_data = pd.read_parquet(market_path)

    requested_end_date = resolve_download_end_date(market_config.get("end_date"))

    result = validate_market_data(
        market_data,
        expected_tickers=universe["ticker"],
        start_date=str(market_config["start_date"]),
        end_date=requested_end_date,
        minimum_observations=int(market_config["minimum_observations"]),
        minimum_coverage_ratio=float(market_config["minimum_coverage_ratio"]),
        extreme_return_threshold=float(market_config["extreme_return_threshold"]),
        maximum_missing_ratio=float(market_config["maximum_missing_ratio"]),
        adjustment_factor_jump_threshold=float(market_config["adjustment_factor_jump_threshold"]),
    )

    report_path = REPORTS_DIR / "data_quality" / "market_data_report.md"

    write_market_data_report(
        result,
        report_path,
        provider_name=str(market_config["provider"]),
        requested_start_date=str(market_config["start_date"]),
        requested_end_date=requested_end_date,
    )

    logger.info(
        "Market-data quality report written to %s.",
        report_path,
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Market-data validation")
    print("------------------------------------------------")
    print(f"Status: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"Rows: {result.summary['rows']}")
    print(f"Tickers: {result.summary['tickers']}")
    print(f"Unique dates: {result.summary['unique_dates']}")
    print(f"Duplicate rows: {result.summary['duplicate_rows']}")
    print(f"Invalid-price rows: {result.summary['invalid_price_rows']}")
    print(f"Missing ratio: {result.summary['missing_ratio']:.6%}")
    print(f"Extreme returns: {result.summary['extreme_return_rows']}")
    print(f"Report: {report_path}")

    if result.warnings:
        print()
        print("Warnings")
        print("------------------------------------------------")

        for warning in result.warnings:
            print(f"- {warning}")

    if result.issues:
        print()
        print("Blocking issues")
        print("------------------------------------------------")

        for issue in result.issues:
            print(f"- {issue}")

        raise SystemExit(1)

    print()
    print("Market-data validation: OK")


if __name__ == "__main__":
    main()
