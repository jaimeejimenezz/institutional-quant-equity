"""Build and store raw point-in-time technical features."""

from quant_equity.config import (
    PROJECT_ROOT,
    load_config,
)
from quant_equity.features import (
    TECHNICAL_FEATURE_COLUMNS,
    build_and_store_raw_technical_features,
)
from quant_equity.logging_config import (
    configure_logging,
)


def main() -> None:
    """Run the raw technical-feature pipeline."""
    config = load_config()

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "build_raw_technical_features.log"),
    )

    result = build_and_store_raw_technical_features(config)

    features = result.features

    feature_values = features.loc[
        :,
        TECHNICAL_FEATURE_COLUMNS,
    ]

    missing_cells = int(feature_values.isna().sum().sum())

    total_feature_cells = len(feature_values) * len(TECHNICAL_FEATURE_COLUMNS)

    missing_ratio = missing_cells / total_feature_cells if total_feature_cells else 0.0

    complete_rows = int(feature_values.notna().all(axis=1).sum())

    temporal_violations = int(features["latest_market_date"].gt(features["as_of_date"]).sum())

    logger.info("Raw technical-feature pipeline completed.")

    logger.info(
        "Feature rows: %s",
        len(features),
    )

    logger.info(
        "Feature dates: %s",
        features["as_of_date"].nunique(),
    )

    logger.info(
        "Feature tickers: %s",
        features["ticker"].nunique(),
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Raw technical-feature construction")
    print("------------------------------------------------")
    print(f"Rows: {len(features)}")
    print(f"Dates: {features['as_of_date'].nunique()}")
    print(f"Tickers: {features['ticker'].nunique()}")
    print(f"Technical features: {len(TECHNICAL_FEATURE_COLUMNS)}")
    print(
        "Available range: "
        f"{features['as_of_date'].min().date()} "
        "to "
        f"{features['as_of_date'].max().date()}"
    )
    print(f"Complete rows: {complete_rows}")
    print(f"Missing feature cells: {missing_cells}")
    print(f"Missing ratio: {missing_ratio:.6%}")
    print(f"Point-in-time violations: {temporal_violations}")
    print(f"Dataset: {result.features_path}")
    print()
    print("Raw technical-feature construction: OK")


if __name__ == "__main__":
    main()
