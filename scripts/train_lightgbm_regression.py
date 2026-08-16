"""Train and evaluate the LightGBM equity regression model."""

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
    LightGBMRegressionConfig,
    evaluate_model_predictions,
    summarize_lightgbm_importance,
    train_lightgbm_regression,
)

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_panel.parquet"

FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "walk_forward_folds.parquet"

EXISTING_COMPARISON_PATH = (
    PROJECT_ROOT / "data" / "processed" / "predictions_oos_model_comparison.parquet"
)

LIGHTGBM_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "predictions_oos_lightgbm.parquet"

COMPARISON_PATH = PROJECT_ROOT / "data" / "processed" / "predictions_oos_model_comparison.parquet"

HYPERPARAMETER_PATH = REPORTS_DIR / "tables" / "lightgbm_hyperparameters.csv"

FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "tables" / "lightgbm_feature_importance.csv"

FEATURE_IMPORTANCE_SUMMARY_PATH = REPORTS_DIR / "tables" / "lightgbm_feature_importance_summary.csv"

MONTHLY_METRICS_PATH = REPORTS_DIR / "tables" / "model_comparison_monthly_metrics.csv"

SUMMARY_PATH = REPORTS_DIR / "tables" / "model_comparison_summary.csv"

REPORT_PATH = REPORTS_DIR / "models" / "lightgbm_regression.md"


def write_report(
    *,
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    hyperparameters: pd.DataFrame,
    importance_summary: pd.DataFrame,
    feature_count: int,
) -> None:
    """Write the LightGBM model research report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = hyperparameters.loc[hyperparameters["selected"]].copy()

    candidate_frequency = (
        selected.groupby(
            "candidate_name",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "folds_selected",
            }
        )
        .sort_values(
            "folds_selected",
            ascending=False,
        )
    )

    iteration_summary = selected["best_iteration"].describe().to_string()

    top_features = importance_summary.head(20)

    lines = [
        "# LightGBM Regression Research",
        "",
        "## Scope",
        "",
        (
            "LightGBM is evaluated using the same frozen "
            "walk-forward test dates as all simpler models."
        ),
        "",
        (
            "Hyperparameter candidates and boosting "
            "iterations are selected exclusively from "
            "the validation period of each fold."
        ),
        "",
        "## Dataset",
        "",
        "```text",
        f"features: {feature_count}",
        (f"LightGBM prediction rows: {len(predictions)}"),
        (f"combined prediction rows: {len(comparison)}"),
        (f"out-of-sample dates: {predictions['as_of_date'].nunique()}"),
        (f"companies: {predictions['ticker'].nunique()}"),
        "```",
        "",
        "## Out-of-sample model comparison",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Selected candidate frequency",
        "",
        "```text",
        candidate_frequency.to_string(index=False),
        "```",
        "",
        "## Selected boosting iterations",
        "",
        "```text",
        iteration_summary,
        "```",
        "",
        "## Most important predictors by gain",
        "",
        "```text",
        top_features.to_string(index=False),
        "```",
        "",
        "## Interpretation rule",
        "",
        (
            "The nonlinear model is considered useful "
            "only if its out-of-sample improvement is "
            "meaningful relative to simpler benchmarks "
            "and is not driven by excessive ranking turnover."
        ),
        "",
        "## Status",
        "",
        "**LIGHTGBM REGRESSION TRAINING COMPLETE**",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Train and evaluate LightGBM regression."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        PANEL_PATH,
        FOLDS_PATH,
        EXISTING_COMPARISON_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    panel = pd.read_parquet(PANEL_PATH)

    folds = pd.read_parquet(FOLDS_PATH)

    existing_predictions = pd.read_parquet(EXISTING_COMPARISON_PATH)

    # Make reruns idempotent if a previous
    # LightGBM result already exists.
    existing_predictions = existing_predictions.loc[
        ~existing_predictions["model_name"].eq("lightgbm_regressor")
    ].copy()

    config = LightGBMRegressionConfig()

    outputs = train_lightgbm_regression(
        panel,
        folds,
        config=config,
    )

    lightgbm_predictions = outputs.predictions

    comparison = pd.concat(
        [
            existing_predictions,
            lightgbm_predictions,
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
        raise ValueError(f"Combined model predictions contain {duplicate_keys} duplicate keys.")

    (
        monthly_metrics,
        model_summary,
    ) = evaluate_model_predictions(comparison)

    importance_summary = summarize_lightgbm_importance(outputs.feature_importance)

    LIGHTGBM_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HYPERPARAMETER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lightgbm_predictions.to_parquet(
        LIGHTGBM_PREDICTIONS_PATH,
        index=False,
    )

    comparison.to_parquet(
        COMPARISON_PATH,
        index=False,
    )

    outputs.hyperparameter_search.to_csv(
        HYPERPARAMETER_PATH,
        index=False,
    )

    outputs.feature_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    importance_summary.to_csv(
        FEATURE_IMPORTANCE_SUMMARY_PATH,
        index=False,
    )

    monthly_metrics.to_csv(
        MONTHLY_METRICS_PATH,
        index=False,
    )

    model_summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    write_report(
        predictions=(lightgbm_predictions),
        comparison=comparison,
        summary=model_summary,
        hyperparameters=(outputs.hyperparameter_search),
        importance_summary=(importance_summary),
        feature_count=len(outputs.feature_columns),
    )

    selected = outputs.hyperparameter_search.loc[outputs.hyperparameter_search["selected"]]

    maturity_violations = int(
        (
            lightgbm_predictions["latest_fit_target_end_date"] > lightgbm_predictions["as_of_date"]
        ).sum()
    )

    logger.info("LightGBM regression training completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("LightGBM regression training")
    print("------------------------------------------------")

    print(f"Features: {len(outputs.feature_columns)}")

    print(f"Folds: {folds['fold_id'].nunique()}")

    print(f"OOS test dates: {lightgbm_predictions['as_of_date'].nunique()}")

    print(f"LightGBM prediction rows: {len(lightgbm_predictions)}")

    print(f"Models in comparison: {comparison['model_name'].nunique()}")

    print(f"Comparison prediction rows: {len(comparison)}")

    print(f"Hyperparameter candidates evaluated: {len(outputs.hyperparameter_search)}")

    print(f"Selected parameter sets: {len(selected)}")

    print(f"Feature importance rows: {len(outputs.feature_importance)}")

    print(f"Duplicate prediction keys: {duplicate_keys}")

    print(f"Fitting-label maturity violations: {maturity_violations}")

    print()

    print(model_summary.to_string(index=False))

    print()

    print("Selected candidate frequency:")

    print(selected["candidate_name"].value_counts().to_string())

    print()

    print("Top 15 predictors by mean gain share:")

    print(importance_summary.head(15).to_string(index=False))

    print()

    print(f"LightGBM predictions: {LIGHTGBM_PREDICTIONS_PATH}")

    print(f"Combined predictions: {COMPARISON_PATH}")

    print(f"Hyperparameters: {HYPERPARAMETER_PATH}")

    print(f"Feature importance: {FEATURE_IMPORTANCE_PATH}")

    print(f"Feature importance summary: {FEATURE_IMPORTANCE_SUMMARY_PATH}")

    print(f"Monthly metrics: {MONTHLY_METRICS_PATH}")

    print(f"Model summary: {SUMMARY_PATH}")

    print(f"Report: {REPORT_PATH}")

    print()

    print("LightGBM regression training: OK")


if __name__ == "__main__":
    main()
