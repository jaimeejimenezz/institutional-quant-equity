"""Validate processed monthly technical features."""

import pandas as pd

from quant_equity.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.data import (
    load_universe,
)
from quant_equity.features import (
    TechnicalFeatureProcessingConfig,
    validate_processed_technical_features,
    write_technical_features_report,
)
from quant_equity.logging_config import (
    configure_logging,
)


def main() -> None:
    """Validate the processed technical-feature panel."""
    config = load_config()

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "validate_technical_features.log"),
    )

    raw_path = INTERIM_DATA_DIR / "features_technical_raw_monthly.parquet"

    processed_path = PROCESSED_DATA_DIR / "features_technical_monthly.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(
            "Raw technical features were not found. Run build_raw_technical_features.py first."
        )

    if not processed_path.exists():
        raise FileNotFoundError(
            "Processed technical features were not found. Run build_technical_features.py first."
        )

    raw_features = pd.read_parquet(raw_path)

    processed_features = pd.read_parquet(processed_path)

    universe = load_universe(str(config["universe"]["version"]))

    processing_config = TechnicalFeatureProcessingConfig.from_mapping(
        config["technical_feature_processing"]
    )

    result = validate_processed_technical_features(
        raw_features,
        processed_features,
        expected_tickers=(universe["ticker"]),
        processing_config=(processing_config),
    )

    report_path = REPORTS_DIR / "data_quality" / "technical_features_report.md"

    write_technical_features_report(
        result,
        report_path,
        processing_config=(processing_config),
    )

    logger.info(
        "Technical-feature report written to %s.",
        report_path,
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Technical-feature validation")
    print("------------------------------------------------")
    print(f"Status: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"Rows: {result.summary['rows']}")
    print(f"Dates: {result.summary['dates']}")
    print(f"Tickers: {result.summary['tickers']}")
    print(f"Sectors: {result.summary['sectors']}")
    print(f"Duplicate rows: {result.summary['duplicate_rows']}")
    print(f"Temporal violations: {result.summary['temporal_violations']}")
    print(f"Changed raw values: {result.summary['changed_raw_values']}")
    print(f"Non-finite values: {result.summary['non_finite_values']}")
    print(f"Incorrect winsorized values: {result.summary['incorrect_winsorized_values']}")
    print(f"Invalid standardized dates: {result.summary['invalid_standardized_dates']}")
    print(f"Invalid sector groups: {result.summary['invalid_sector_neutral_groups']}")
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
    print("Technical-feature validation: OK")


if __name__ == "__main__":
    main()
