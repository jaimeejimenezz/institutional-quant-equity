"""Run the definitive model baselines."""

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
    evaluate_model_predictions,
    generate_baseline_predictions,
)

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_panel.parquet"

FOLDS_PATH = PROJECT_ROOT / "data" / "processed" / "walk_forward_folds.parquet"

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "predictions_oos_baselines.parquet"

MONTHLY_METRICS_PATH = REPORTS_DIR / "tables" / "model_baseline_monthly_metrics.csv"

SUMMARY_PATH = REPORTS_DIR / "tables" / "model_baseline_summary.csv"

REPORT_PATH = REPORTS_DIR / "models" / "model_baselines_step13a.md"


def write_report(
    predictions: pd.DataFrame,
    monthly: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Write the Step 13A baseline report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "# Model Baselines — Step 13A",
        "",
        "## Purpose",
        "",
        (
            "Establish the definitive simple baselines "
            "and the common out-of-sample evaluation "
            "contract for Step 13."
        ),
        "",
        "## Temporal contract",
        "",
        ("All predictions use the frozen Step 12 walk-forward test dates."),
        "",
        ("No independent train/test split is created inside Step 13A."),
        "",
        "## Baselines",
        "",
        "- `constant`",
        "- `momentum_3m`",
        "- `technical_equal_weight_composite`",
        "",
        "## Prediction coverage",
        "",
        "```text",
        (f"rows: {len(predictions)}"),
        (f"models: {predictions['model_name'].nunique()}"),
        (f"oos dates: {predictions['as_of_date'].nunique()}"),
        (f"tickers: {predictions['ticker'].nunique()}"),
        "```",
        "",
        "## Model summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Interpretation rule",
        "",
        (
            "The constant model has no meaningful "
            "cross-sectional ranking, so rank-based "
            "metrics are intentionally undefined."
        ),
        "",
        (
            "The Step 13A results do not select a final "
            "model. They establish the benchmarks that "
            "Ridge, Elastic Net and LightGBM must beat."
        ),
        "",
        "## Monthly evaluation",
        "",
        (f"Monthly metric rows: {len(monthly)}"),
        "",
        "## Status",
        "",
        "**STEP 13A COMPLETE**",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Execute Step 13A."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        PANEL_PATH,
        FOLDS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    panel = pd.read_parquet(PANEL_PATH)

    folds = pd.read_parquet(FOLDS_PATH)

    predictions = generate_baseline_predictions(
        panel,
        folds,
    )

    (
        monthly,
        summary,
    ) = evaluate_model_predictions(predictions)

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MONTHLY_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        PREDICTIONS_PATH,
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

    write_report(
        predictions,
        monthly,
        summary,
    )

    duplicate_keys = int(
        predictions.duplicated(
            [
                "fold_id",
                "ticker",
                "model_name",
            ]
        ).sum()
    )

    logger.info("Definitive model baselines completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Definitive model baselines - Step 13A")
    print("------------------------------------------------")

    print(f"Folds: {folds['fold_id'].nunique()}")

    print(f"OOS test dates: {predictions['as_of_date'].nunique()}")

    print(f"Models: {predictions['model_name'].nunique()}")

    print(f"Prediction rows: {len(predictions)}")

    print(f"Tickers: {predictions['ticker'].nunique()}")

    print(f"Duplicate prediction keys: {duplicate_keys}")

    print()

    print(summary.to_string(index=False))

    print()

    print(f"Predictions: {PREDICTIONS_PATH}")

    print(f"Monthly metrics: {MONTHLY_METRICS_PATH}")

    print(f"Summary: {SUMMARY_PATH}")

    print(f"Report: {REPORT_PATH}")

    print()


if __name__ == "__main__":
    main()
