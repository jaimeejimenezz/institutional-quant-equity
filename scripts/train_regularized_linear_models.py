"""Train and evaluate regularized linear equity models."""

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
    RegularizedLinearConfig,
    evaluate_model_predictions,
    summarize_linear_coefficients,
    train_regularized_linear_models,
)

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_panel.parquet"

FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "walk_forward_folds.parquet"

BASELINE_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "predictions_oos_baselines.parquet"
)

LINEAR_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "predictions_oos_regularized_linear.parquet"
)

COMPARISON_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "predictions_oos_model_comparison.parquet"
)

HYPERPARAMETER_PATH = REPORTS_DIR / "tables" / "regularized_linear_hyperparameters.csv"

COEFFICIENT_PATH = REPORTS_DIR / "tables" / "regularized_linear_coefficients.csv"

COEFFICIENT_SUMMARY_PATH = REPORTS_DIR / "tables" / "regularized_linear_coefficient_summary.csv"

MONTHLY_METRICS_PATH = REPORTS_DIR / "tables" / "model_comparison_monthly_metrics.csv"

SUMMARY_PATH = REPORTS_DIR / "tables" / "model_comparison_summary.csv"

REPORT_PATH = REPORTS_DIR / "models" / "regularized_linear_models.md"


def write_report(
    *,
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    hyperparameters: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    feature_count: int,
) -> None:
    """Write the regularized linear model research report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = hyperparameters.loc[hyperparameters["selected"]].copy()

    parameter_frequency = (
        selected.groupby(
            [
                "model_name",
                "alpha",
                "l1_ratio",
            ],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "folds_selected",
            }
        )
        .sort_values(
            [
                "model_name",
                "folds_selected",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    top_coefficients = (
        coefficient_summary.groupby(
            "model_name",
            group_keys=False,
        )
        .head(15)
        .reset_index(drop=True)
    )

    lines = [
        "# Regularized Linear Model Research",
        "",
        "## Scope",
        "",
        ("Ridge and Elastic Net are trained using the frozen walk-forward validation framework."),
        "",
        ("Hyperparameters are selected exclusively from each fold's validation period."),
        "",
        (
            "The monthly test cross-section is not used "
            "for preprocessing, hyperparameter selection "
            "or model fitting."
        ),
        "",
        "## Dataset",
        "",
        "```text",
        f"features: {feature_count}",
        (f"linear prediction rows: {len(predictions)}"),
        (f"comparison prediction rows: {len(comparison)}"),
        (f"out-of-sample dates: {predictions['as_of_date'].nunique()}"),
        (f"companies: {predictions['ticker'].nunique()}"),
        "```",
        "",
        "## Out-of-sample comparison",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Selected hyperparameters",
        "",
        "```text",
        parameter_frequency.to_string(index=False),
        "```",
        "",
        "## Largest standardized coefficients",
        "",
        "```text",
        top_coefficients.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
        (
            "The simple technical baselines remain frozen. "
            "Their definitions are not modified after "
            "observing out-of-sample results."
        ),
        "",
        ("The regularized models use the complete technical and fundamental predictor set."),
        "",
        (
            "Model complexity is justified only if the "
            "out-of-sample results improve on the simpler "
            "benchmarks with reasonable stability."
        ),
        "",
        "## Status",
        "",
        "**REGULARIZED LINEAR MODEL TRAINING COMPLETE**",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Train and evaluate Ridge and Elastic Net models."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        PANEL_PATH,
        FOLDS_PATH,
        BASELINE_PREDICTIONS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    panel = pd.read_parquet(PANEL_PATH)

    folds = pd.read_parquet(FOLDS_PATH)

    baselines = pd.read_parquet(BASELINE_PREDICTIONS_PATH)

    config = RegularizedLinearConfig()

    outputs = train_regularized_linear_models(
        panel,
        folds,
        config=config,
    )

    linear_predictions = outputs.predictions

    comparison = pd.concat(
        [
            baselines,
            linear_predictions,
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

    coefficient_summary = summarize_linear_coefficients(outputs.coefficients)

    LINEAR_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HYPERPARAMETER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    linear_predictions.to_parquet(
        LINEAR_PREDICTIONS_PATH,
        index=False,
    )

    comparison.to_parquet(
        COMPARISON_PREDICTIONS_PATH,
        index=False,
    )

    outputs.hyperparameter_search.to_csv(
        HYPERPARAMETER_PATH,
        index=False,
    )

    outputs.coefficients.to_csv(
        COEFFICIENT_PATH,
        index=False,
    )

    coefficient_summary.to_csv(
        COEFFICIENT_SUMMARY_PATH,
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
        predictions=linear_predictions,
        comparison=comparison,
        summary=model_summary,
        hyperparameters=(outputs.hyperparameter_search),
        coefficient_summary=(coefficient_summary),
        feature_count=len(outputs.feature_columns),
    )

    selected_parameters = outputs.hyperparameter_search.loc[
        outputs.hyperparameter_search["selected"]
    ]

    maturity_violations = int(
        (linear_predictions["latest_fit_target_end_date"] > linear_predictions["as_of_date"]).sum()
    )

    logger.info("Regularized linear model training completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Regularized linear model training")
    print("------------------------------------------------")

    print(f"Features: {len(outputs.feature_columns)}")

    print(f"Folds: {folds['fold_id'].nunique()}")

    print(f"OOS test dates: {linear_predictions['as_of_date'].nunique()}")

    print(f"Linear models: {linear_predictions['model_name'].nunique()}")

    print(f"Linear prediction rows: {len(linear_predictions)}")

    print(f"Models in comparison: {comparison['model_name'].nunique()}")

    print(f"Comparison prediction rows: {len(comparison)}")

    print(f"Hyperparameter candidates evaluated: {len(outputs.hyperparameter_search)}")

    print(f"Selected parameter sets: {len(selected_parameters)}")

    print(f"Coefficient rows: {len(outputs.coefficients)}")

    print(f"Duplicate prediction keys: {duplicate_keys}")

    print(f"Fitting-label maturity violations: {maturity_violations}")

    print()

    print(model_summary.to_string(index=False))

    print()

    print(f"Linear predictions: {LINEAR_PREDICTIONS_PATH}")

    print(f"Combined predictions: {COMPARISON_PREDICTIONS_PATH}")

    print(f"Hyperparameters: {HYPERPARAMETER_PATH}")

    print(f"Coefficients: {COEFFICIENT_PATH}")

    print(f"Coefficient summary: {COEFFICIENT_SUMMARY_PATH}")

    print(f"Monthly metrics: {MONTHLY_METRICS_PATH}")

    print(f"Model summary: {SUMMARY_PATH}")

    print(f"Report: {REPORT_PATH}")

    print()

    print("Regularized linear model training: OK")


if __name__ == "__main__":
    main()
