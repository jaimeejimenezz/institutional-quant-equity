"""Build and store monthly rebalance dates and labels."""

from quant_equity.config import (
    PROJECT_ROOT,
    load_config,
)
from quant_equity.labels import (
    build_and_store_monthly_labels,
)
from quant_equity.logging_config import (
    configure_logging,
)


def main() -> None:
    """Run the monthly-label construction pipeline."""
    config = load_config()

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "build_monthly_labels.log"),
    )

    result = build_and_store_monthly_labels(config)

    calendar = result.rebalance_calendar

    labels = result.monthly_labels

    eligible_dates = int(calendar["has_full_horizon"].sum())

    logger.info("Monthly-label pipeline completed.")

    logger.info(
        "Calendar rows: %s",
        len(calendar),
    )

    logger.info(
        "Eligible dates: %s",
        eligible_dates,
    )

    logger.info(
        "Monthly label rows: %s",
        len(labels),
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Monthly-label construction")
    print("------------------------------------------------")
    print(f"Calendar rows: {len(calendar)}")
    print(f"Eligible dates: {eligible_dates}")
    print(f"Label rows: {len(labels)}")
    print(f"Label dates: {labels['as_of_date'].nunique()}")
    print(f"Tickers: {labels['ticker'].nunique()}")
    print(
        "Available label range: "
        f"{labels['as_of_date'].min().date()} "
        "to "
        f"{labels['as_of_date'].max().date()}"
    )
    print(f"Rebalance calendar: {result.rebalance_calendar_path}")
    print(f"Monthly labels: {result.monthly_labels_path}")
    print()
    print("Monthly-label construction: OK")


if __name__ == "__main__":
    main()
