"""Analyze stability and robustness across candidate equity models."""

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
    build_model_scorecard,
    compute_feature_concentration,
    compute_pairwise_ic_comparison,
    compute_sector_stability,
    compute_yearly_stability,
)

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "predictions_oos_model_comparison.parquet"

MONTHLY_METRICS_PATH = REPORTS_DIR / "tables" / "model_comparison_monthly_metrics.csv"

MODEL_SUMMARY_PATH = REPORTS_DIR / "tables" / "model_comparison_summary.csv"

LINEAR_IMPORTANCE_PATH = REPORTS_DIR / "tables" / "regularized_linear_coefficient_summary.csv"

LIGHTGBM_IMPORTANCE_PATH = REPORTS_DIR / "tables" / "lightgbm_feature_importance_summary.csv"

YEARLY_PATH = REPORTS_DIR / "tables" / "model_yearly_stability.csv"

SECTOR_PATH = REPORTS_DIR / "tables" / "model_sector_stability.csv"

PAIRWISE_PATH = REPORTS_DIR / "tables" / "model_pairwise_ic_comparison.csv"

CONCENTRATION_PATH = REPORTS_DIR / "tables" / "model_feature_concentration.csv"

SCORECARD_PATH = REPORTS_DIR / "tables" / "model_selection_scorecard.csv"

REPORT_PATH = REPORTS_DIR / "models" / "model_comparison.md"


def write_report(
    *,
    scorecard: pd.DataFrame,
    yearly: pd.DataFrame,
    sectors: pd.DataFrame,
    pairwise: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    """Write the consolidated model comparison report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    yearly_pivot = yearly.pivot(
        index="model_name",
        columns="year",
        values="mean_ic",
    )

    lines = [
        "# Out-of-Sample Model Comparison",
        "",
        "## Objective",
        "",
        (
            "Compare predictive quality, stability, "
            "economic ranking behaviour and feature "
            "dependence across all candidate models."
        ),
        "",
        "## Overall scorecard",
        "",
        "```text",
        scorecard.to_string(index=False),
        "```",
        "",
        "## Mean IC by year",
        "",
        "```text",
        yearly_pivot.to_string(),
        "```",
        "",
        "## Pairwise monthly IC comparison",
        "",
        "```text",
        pairwise.to_string(index=False),
        "```",
        "",
        "## Feature concentration",
        "",
        "```text",
        concentration.to_string(index=False),
        "```",
        "",
        "## Sector diagnostics",
        "",
        (
            "Sector-level IC is calculated only where "
            "at least three companies are available. "
            "These results are diagnostic because "
            "individual sector cross-sections are small."
        ),
        "",
        "```text",
        sectors.to_string(index=False),
        "```",
        "",
        "## Selection principle",
        "",
        (
            "No model should be selected from one metric "
            "alone. Mean IC, IC stability, top-bottom "
            "spread, top-quintile precision, turnover, "
            "temporal stability, sector stability and "
            "feature concentration must be considered "
            "jointly."
        ),
        "",
        "**MODEL COMPARISON DIAGNOSTICS COMPLETE**",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run consolidated model diagnostics."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        PREDICTIONS_PATH,
        MONTHLY_METRICS_PATH,
        MODEL_SUMMARY_PATH,
        LINEAR_IMPORTANCE_PATH,
        LIGHTGBM_IMPORTANCE_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    predictions = pd.read_parquet(PREDICTIONS_PATH)

    monthly = pd.read_csv(
        MONTHLY_METRICS_PATH,
        parse_dates=["as_of_date"],
    )

    summary = pd.read_csv(MODEL_SUMMARY_PATH)

    linear_importance = pd.read_csv(LINEAR_IMPORTANCE_PATH)

    lightgbm_importance = pd.read_csv(LIGHTGBM_IMPORTANCE_PATH)

    yearly = compute_yearly_stability(monthly)

    sectors = compute_sector_stability(
        predictions,
        minimum_companies=3,
    )

    pairwise = compute_pairwise_ic_comparison(
        monthly,
        block_length=3,
        bootstrap_samples=5000,
        random_state=42,
    )

    concentration = compute_feature_concentration(
        linear_importance,
        lightgbm_importance,
    )

    scorecard = build_model_scorecard(
        summary,
        yearly,
        sectors,
    )

    YEARLY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    yearly.to_csv(
        YEARLY_PATH,
        index=False,
    )

    sectors.to_csv(
        SECTOR_PATH,
        index=False,
    )

    pairwise.to_csv(
        PAIRWISE_PATH,
        index=False,
    )

    concentration.to_csv(
        CONCENTRATION_PATH,
        index=False,
    )

    scorecard.to_csv(
        SCORECARD_PATH,
        index=False,
    )

    write_report(
        scorecard=scorecard,
        yearly=yearly,
        sectors=sectors,
        pairwise=pairwise,
        concentration=concentration,
    )

    logger.info("Model comparison diagnostics completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Out-of-sample model comparison diagnostics")
    print("------------------------------------------------")

    print(f"Models: {summary['model_name'].nunique()}")

    print(f"OOS dates: {predictions['as_of_date'].nunique()}")

    print(f"Prediction rows: {len(predictions)}")

    print(f"Years evaluated: {yearly['year'].nunique()}")

    print(f"Sectors evaluated: {sectors['sector'].nunique()}")

    print(f"Pairwise comparisons: {len(pairwise)}")

    print()
    print("Overall scorecard:")
    print(scorecard.to_string(index=False))

    print()
    print("Mean IC by year:")
    print(
        yearly.pivot(
            index="model_name",
            columns="year",
            values="mean_ic",
        ).to_string()
    )

    print()
    print("Feature concentration:")
    print(concentration.to_string(index=False))

    print()
    print("Pairwise IC comparisons involving Elastic Net or technical composite:")

    important_pairs = pairwise.loc[
        pairwise["model_a"].isin(
            [
                "elastic_net",
                ("technical_equal_weight_composite"),
            ]
        )
        | pairwise["model_b"].isin(
            [
                "elastic_net",
                ("technical_equal_weight_composite"),
            ]
        )
    ]

    print(important_pairs.to_string(index=False))

    print()

    print(f"Yearly stability: {YEARLY_PATH}")

    print(f"Sector stability: {SECTOR_PATH}")

    print(f"Pairwise comparison: {PAIRWISE_PATH}")

    print(f"Feature concentration: {CONCENTRATION_PATH}")

    print(f"Selection scorecard: {SCORECARD_PATH}")

    print(f"Report: {REPORT_PATH}")

    print()

    print("Model comparison diagnostics: OK")


if __name__ == "__main__":
    main()
