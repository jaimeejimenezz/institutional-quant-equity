"""Train and evaluate the cross-sectional LightGBM ranking model."""

from __future__ import annotations

import logging

import pandas as pd

from quant_equity.config import (
    PROJECT_ROOT,
    REPORTS_DIR,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.models import (
    LightGBMRankingConfig,
    evaluate_model_predictions,
    summarize_lightgbm_ranking_importance,
    train_lightgbm_ranking,
)

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_panel.parquet"

FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "walk_forward_folds.parquet"

EXISTING_COMPARISON_PATH = (
    PROJECT_ROOT / "data" / "processed" / "predictions_oos_model_comparison.parquet"
)

RANKING_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "predictions_oos_lightgbm_ranker.parquet"
)

ALL_MODEL_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "predictions_oos_all_models.parquet"
)

HYPERPARAMETER_PATH = REPORTS_DIR / "tables" / "lightgbm_ranker_hyperparameters.csv"

IMPORTANCE_PATH = REPORTS_DIR / "tables" / "lightgbm_ranker_feature_importance.csv"

IMPORTANCE_SUMMARY_PATH = REPORTS_DIR / "tables" / "lightgbm_ranker_feature_importance_summary.csv"

MONTHLY_METRICS_PATH = REPORTS_DIR / "tables" / "model_comparison_monthly_metrics.csv"

SUMMARY_PATH = REPORTS_DIR / "tables" / "model_comparison_summary.csv"


def main() -> None:
    """Train and compare the ranking model."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    panel = pd.read_parquet(PANEL_PATH)

    folds = pd.read_parquet(FOLDS_PATH)

    existing = pd.read_parquet(EXISTING_COMPARISON_PATH)

    existing = existing.loc[~existing["model_name"].eq("lightgbm_ranker")].copy()

    outputs = train_lightgbm_ranking(
        panel,
        folds,
        config=(LightGBMRankingConfig()),
    )

    comparison = pd.concat(
        [
            existing,
            outputs.predictions,
        ],
        ignore_index=True,
        sort=False,
    )

    duplicate_keys = int(
        comparison.duplicated(
            [
                "as_of_date",
                "ticker",
                "model_name",
            ]
        ).sum()
    )

    if duplicate_keys:
        raise ValueError("Combined predictions contain duplicate keys.")

    monthly, summary = evaluate_model_predictions(comparison)

    importance_summary = summarize_lightgbm_ranking_importance(outputs.feature_importance)

    RANKING_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HYPERPARAMETER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs.predictions.to_parquet(
        RANKING_PREDICTIONS_PATH,
        index=False,
    )

    comparison.to_parquet(
        ALL_MODEL_PREDICTIONS_PATH,
        index=False,
    )

    outputs.hyperparameter_search.to_csv(
        HYPERPARAMETER_PATH,
        index=False,
    )

    outputs.feature_importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    importance_summary.to_csv(
        IMPORTANCE_SUMMARY_PATH,
        index=False,
    )

    monthly.to_csv(
        MONTHLY_METRICS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    selected = outputs.hyperparameter_search.loc[outputs.hyperparameter_search["selected"]]

    maturity_violations = int(
        (
            outputs.predictions["latest_fit_target_end_date"] > outputs.predictions["as_of_date"]
        ).sum()
    )

    logger.info("LightGBM ranking training completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("LightGBM cross-sectional ranking")
    print("------------------------------------------------")

    print(f"Features: {len(outputs.feature_columns)}")

    print(f"Folds: {folds['fold_id'].nunique()}")

    print(f"OOS test dates: {outputs.predictions['as_of_date'].nunique()}")

    print(f"Ranking prediction rows: {len(outputs.predictions)}")

    print(f"Models in comparison: {comparison['model_name'].nunique()}")

    print(f"All-model prediction rows: {len(comparison)}")

    print(f"Hyperparameter candidates evaluated: {len(outputs.hyperparameter_search)}")

    print(f"Selected parameter sets: {len(selected)}")

    print(f"Feature importance rows: {len(outputs.feature_importance)}")

    print(f"Duplicate prediction keys: {duplicate_keys}")

    print(f"Fitting-label maturity violations: {maturity_violations}")

    print()
    print(summary.to_string(index=False))

    print()
    print("Selected candidate frequency:")

    print(selected["candidate_name"].value_counts().to_string())

    print()
    print("Top 15 ranking predictors:")

    print(importance_summary.head(15).to_string(index=False))

    print()
    print("LightGBM ranking training: OK")


if __name__ == "__main__":
    main()
