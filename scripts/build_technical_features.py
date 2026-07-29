"""Process and store monthly technical features."""

import numpy as np

from quant_equity.config import (
    PROJECT_ROOT,
    load_config,
)
from quant_equity.features import (
    TECHNICAL_FEATURE_COLUMNS,
    TECHNICAL_MODEL_FEATURE_COLUMNS,
    WINSORIZED_FEATURE_COLUMNS,
    build_and_store_processed_technical_features,
)
from quant_equity.logging_config import (
    configure_logging,
)


def main() -> None:
    """Run technical-feature processing."""
    config = load_config()

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "build_technical_features.log"),
    )

    result = build_and_store_processed_technical_features(config)

    features = result.features

    raw_values = features.loc[
        :,
        TECHNICAL_FEATURE_COLUMNS,
    ]

    model_values = features.loc[
        :,
        TECHNICAL_MODEL_FEATURE_COLUMNS,
    ]

    raw_missing_ratio = float(raw_values.isna().mean().mean())

    model_missing_ratio = float(model_values.isna().mean().mean())

    clipped_observations = 0

    for (
        raw_column,
        winsorized_column,
    ) in zip(
        TECHNICAL_FEATURE_COLUMNS,
        WINSORIZED_FEATURE_COLUMNS,
        strict=True,
    ):
        comparable = features[raw_column].notna() & features[winsorized_column].notna()

        clipped_observations += int(
            (
                comparable
                & ~np.isclose(
                    features[raw_column],
                    features[winsorized_column],
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )
            ).sum()
        )

    temporal_violations = int(features["latest_market_date"].gt(features["as_of_date"]).sum())

    logger.info("Technical-feature processing completed.")

    logger.info(
        "Processed rows: %s",
        len(features),
    )

    logger.info(
        "Processed dates: %s",
        features["as_of_date"].nunique(),
    )

    logger.info(
        "Processed tickers: %s",
        features["ticker"].nunique(),
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Technical-feature processing")
    print("------------------------------------------------")
    print(f"Rows: {len(features)}")
    print(f"Dates: {features['as_of_date'].nunique()}")
    print(f"Tickers: {features['ticker'].nunique()}")
    print(f"Sectors: {features['sector'].nunique()}")
    print(f"Raw technical features: {len(TECHNICAL_FEATURE_COLUMNS)}")
    print(f"Model technical features: {len(TECHNICAL_MODEL_FEATURE_COLUMNS)}")
    print(f"Raw missing ratio: {raw_missing_ratio:.6%}")
    print(f"Model missing ratio: {model_missing_ratio:.6%}")
    print(f"Winsorized observations: {clipped_observations}")
    print(f"Point-in-time violations: {temporal_violations}")
    print(f"Dataset: {result.features_path}")
    print()
    print("Technical-feature processing: OK")


if __name__ == "__main__":
    main()
