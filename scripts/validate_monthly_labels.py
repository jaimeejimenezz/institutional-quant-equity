"""Validate stored monthly labels and calendar data."""

import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.data import (
    load_universe,
)
from quant_equity.labels import (
    validate_monthly_labels,
    write_monthly_labels_report,
)
from quant_equity.logging_config import (
    configure_logging,
)


def main() -> None:
    """Validate monthly labels and write a quality report."""
    config = load_config()

    label_config = config["labels"]

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "validate_monthly_labels.log"),
    )

    calendar_path = PROCESSED_DATA_DIR / "rebalance_calendar.parquet"

    labels_path = PROCESSED_DATA_DIR / "labels_monthly.parquet"

    if not calendar_path.exists():
        raise FileNotFoundError(
            "Rebalance calendar not found. Run 'python scripts/build_monthly_labels.py' first."
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            "Monthly labels not found. Run 'python scripts/build_monthly_labels.py' first."
        )

    rebalance_calendar = pd.read_parquet(calendar_path)

    monthly_labels = pd.read_parquet(labels_path)

    universe = load_universe(str(config["universe"]["version"]))

    horizon_sessions = int(label_config["horizon_sessions"])

    top_quantile_fraction = float(label_config["top_quantile_fraction"])

    result = validate_monthly_labels(
        rebalance_calendar,
        monthly_labels,
        expected_tickers=(universe["ticker"]),
        horizon_sessions=(horizon_sessions),
        top_quantile_fraction=(top_quantile_fraction),
    )

    report_path = REPORTS_DIR / "data_quality" / "monthly_labels_report.md"

    write_monthly_labels_report(
        result,
        report_path,
        horizon_sessions=(horizon_sessions),
        top_quantile_fraction=(top_quantile_fraction),
    )

    logger.info(
        "Monthly-label quality report written to %s.",
        report_path,
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Monthly-label validation")
    print("------------------------------------------------")
    print(f"Status: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"Calendar rows: {result.summary['calendar_rows']}")
    print(f"Eligible dates: {result.summary['eligible_calendar_rows']}")
    print(f"Label rows: {result.summary['label_rows']}")
    print(f"Label dates: {result.summary['label_dates']}")
    print(f"Tickers: {result.summary['tickers']}")
    print(
        "Cross-section range: "
        f"{result.summary['minimum_cross_section_size']} "
        "to "
        f"{result.summary['maximum_cross_section_size']}"
    )
    print(f"Duplicate labels: {result.summary['duplicate_label_rows']}")
    print(f"Invalid temporal rows: {result.summary['invalid_temporal_rows']}")
    print(f"Incorrect targets: {result.summary['incorrect_target_rows']}")
    print(f"Maximum median excess: {result.summary['maximum_absolute_median_excess']:.12f}")
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
    print("Monthly-label validation: OK")


if __name__ == "__main__":
    main()
