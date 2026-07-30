"""Evaluate the initial linear models out of sample."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.models import (
    TECHNICAL_COMPOSITE_DIRECTIONS,
)
from quant_equity.research import (
    LinearModelEvaluationConfig,
    evaluate_linear_model_predictions,
)

PREDICTIONS_PATH = PROCESSED_DATA_DIR / "predictions_linear_oos.parquet"

COEFFICIENTS_PATH = REPORTS_DIR / "tables" / "linear_model_coefficients.csv"

TABLES_DIR = REPORTS_DIR / "tables"

FIGURES_DIR = REPORTS_DIR / "figures" / "linear_models"

REPORT_PATH = REPORTS_DIR / "models" / "linear_model_report.md"

MONTHLY_METRICS_PATH = TABLES_DIR / "linear_model_monthly_metrics.csv"

MODEL_SUMMARY_PATH = TABLES_DIR / "linear_model_summary.csv"

MONTHLY_QUINTILES_PATH = TABLES_DIR / "linear_model_quintiles_monthly.csv"

QUINTILE_SUMMARY_PATH = TABLES_DIR / "linear_model_quintiles.csv"

MONTHLY_TURNOVER_PATH = TABLES_DIR / "linear_model_turnover_monthly.csv"

TURNOVER_SUMMARY_PATH = TABLES_DIR / "linear_model_turnover.csv"

YEARLY_SUMMARY_PATH = TABLES_DIR / "linear_model_yearly.csv"

COEFFICIENT_SUMMARY_PATH = TABLES_DIR / "linear_model_coefficient_stability.csv"

RANKED_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "predictions_linear_oos_evaluated.parquet"

CUMULATIVE_IC_FIGURE_PATH = FIGURES_DIR / "linear_model_cumulative_ic.png"

QUINTILE_FIGURE_PATH = FIGURES_DIR / "linear_model_quintile_profiles.png"


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> Path:
    """Write a CSV table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=False,
    )

    return path


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
) -> Path:
    """Write a Parquet dataset atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(".tmp.parquet")

    temporary_path.unlink(missing_ok=True)

    data.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)

    return path


def _format_value(
    value: Any,
) -> str:
    """Format a value for Markdown."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value).replace(
        "|",
        "\\|",
    )


def _to_markdown(
    frame: pd.DataFrame,
) -> str:
    """Convert a dataframe to a Markdown table."""
    if frame.empty:
        return "_No observations._"

    columns = [str(column) for column in frame.columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in frame.itertuples(
        index=False,
        name=None,
    ):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")

    return "\n".join(lines)


def _plot_cumulative_ic(
    monthly_metrics: pd.DataFrame,
    path: Path,
) -> Path:
    """Plot cumulative monthly information coefficients."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = monthly_metrics.pivot(
        index="as_of_date",
        columns="model_name",
        values="information_coefficient",
    ).sort_index()

    figure, axis = plt.subplots(figsize=(12, 7))

    plotted_models = 0

    for model_name in plot_data.columns:
        values = plot_data[model_name]

        if values.notna().sum() == 0:
            continue

        axis.plot(
            values.fillna(0.0).cumsum(),
            label=model_name,
        )

        plotted_models += 1

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_title("Cumulative monthly out-of-sample IC")

    axis.set_xlabel("Prediction date")

    axis.set_ylabel("Cumulative Spearman IC")

    axis.grid(alpha=0.25)

    if plotted_models:
        axis.legend()

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path


def _plot_quintile_profiles(
    quintile_summary: pd.DataFrame,
    path: Path,
) -> Path:
    """Plot mean future excess return by prediction quintile."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(figsize=(11, 7))

    plotted_models = 0

    for model_name, model_data in quintile_summary.groupby(
        "model_name",
        sort=True,
    ):
        model_data = model_data.sort_values("quintile")

        axis.plot(
            model_data["quintile"],
            model_data["mean_target"] * 100.0,
            marker="o",
            label=model_name,
        )

        plotted_models += 1

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_xticks(
        sorted(quintile_summary["quintile"].unique()) if not quintile_summary.empty else []
    )

    axis.set_xlabel("Prediction quintile")

    axis.set_ylabel("Mean future excess return (%)")

    axis.set_title("Out-of-sample future return by prediction quintile")

    axis.grid(alpha=0.25)

    if plotted_models:
        axis.legend()

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path


def _write_report(
    *,
    model_summary: pd.DataFrame,
    quintile_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    predictions: pd.DataFrame,
) -> Path:
    """Write the Step 7C model-evaluation report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_display_columns = [
        "model_name",
        "months",
        "ranking_months",
        "mean_ic",
        "annualized_ic_ir",
        "ic_t_stat",
        "positive_ic_ratio",
        "mean_top_bottom_spread",
        "positive_spread_ratio",
        "mean_top_quintile_precision",
        "mean_top_quintile_turnover",
        "mean_rmse",
    ]

    yearly_display_columns = [
        "model_name",
        "year",
        "months",
        "mean_ic",
        "positive_ic_ratio",
        "mean_top_bottom_spread",
        "mean_top_quintile_precision",
        "mean_top_quintile_turnover",
    ]

    coefficient_display_columns = [
        "model_name",
        "feature",
        "mean_coefficient",
        "mean_absolute_coefficient",
        "nonzero_ratio",
        "sign_consistency_ratio",
        "economic_direction_ratio",
    ]

    quintile_display_columns = [
        "model_name",
        "quintile",
        "months",
        "mean_target",
        "positive_month_ratio",
    ]

    lines = [
        "# Initial Linear Model Report",
        "",
        "## Status",
        "",
        "**PASS**",
        "",
        "## Evaluation scope",
        "",
        ("All metrics in this report use predictions generated outside the model-training sample."),
        "",
        (f"- Prediction rows: `{len(predictions)}`"),
        (f"- Out-of-sample dates: `{predictions['as_of_date'].nunique()}`"),
        (f"- Companies: `{predictions['ticker'].nunique()}`"),
        (f"- Models: `{predictions['model_name'].nunique()}`"),
        "",
        "## Overall model comparison",
        "",
        _to_markdown(
            model_summary.loc[
                :,
                model_display_columns,
            ]
        ),
        "",
        "## Prediction quintiles",
        "",
        ("Quintile 1 contains the lowest model scores and quintile 5 contains the highest scores."),
        "",
        _to_markdown(
            quintile_summary.loc[
                :,
                quintile_display_columns,
            ]
        ),
        "",
        "## Stability by year",
        "",
        _to_markdown(
            yearly_summary.loc[
                :,
                yearly_display_columns,
            ]
        ),
        "",
        "## Coefficient stability",
        "",
        (
            "The economic-direction ratio measures how "
            "frequently a non-zero coefficient agrees with "
            "the direction established during factor research."
        ),
        "",
        _to_markdown(
            coefficient_summary.loc[
                :,
                coefficient_display_columns,
            ]
        ),
        "",
        "## Constant baseline",
        "",
        (
            "The constant model cannot rank companies because "
            "every company receives the same prediction. Its "
            "IC, quintile and ranking-turnover metrics are "
            "therefore intentionally left undefined."
        ),
        "",
        "## Interpretation rule",
        "",
        (
            "No model should be selected from one metric alone. "
            "The preferred candidate should combine positive "
            "out-of-sample IC, positive quintile spread, temporal "
            "stability, economically coherent coefficients and "
            "reasonable turnover."
        ),
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return REPORT_PATH


def main() -> None:
    """Execute Step 7C."""
    project_config = load_config()

    logger = configure_logging(
        level=project_config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "evaluate_linear_models.log"),
    )

    for required_path in (
        PREDICTIONS_PATH,
        COEFFICIENTS_PATH,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 7B output not found: {required_path}")

    evaluation_config = LinearModelEvaluationConfig.from_mapping(
        project_config["linear_model_evaluation"]
    )

    predictions = pd.read_parquet(PREDICTIONS_PATH)

    coefficients = pd.read_csv(
        COEFFICIENTS_PATH,
        parse_dates=["test_date"],
    )

    outputs = evaluate_linear_model_predictions(
        predictions,
        coefficients,
        feature_directions=(TECHNICAL_COMPOSITE_DIRECTIONS),
        config=evaluation_config,
    )

    output_paths = [
        _write_csv(
            outputs.monthly_metrics,
            MONTHLY_METRICS_PATH,
        ),
        _write_csv(
            outputs.model_summary,
            MODEL_SUMMARY_PATH,
        ),
        _write_csv(
            outputs.monthly_quintiles,
            MONTHLY_QUINTILES_PATH,
        ),
        _write_csv(
            outputs.quintile_summary,
            QUINTILE_SUMMARY_PATH,
        ),
        _write_csv(
            outputs.monthly_turnover,
            MONTHLY_TURNOVER_PATH,
        ),
        _write_csv(
            outputs.turnover_summary,
            TURNOVER_SUMMARY_PATH,
        ),
        _write_csv(
            outputs.yearly_summary,
            YEARLY_SUMMARY_PATH,
        ),
        _write_csv(
            outputs.coefficient_summary,
            COEFFICIENT_SUMMARY_PATH,
        ),
        _write_parquet_atomically(
            outputs.ranked_predictions,
            RANKED_PREDICTIONS_PATH,
        ),
    ]

    figure_paths = [
        _plot_cumulative_ic(
            outputs.monthly_metrics,
            CUMULATIVE_IC_FIGURE_PATH,
        ),
        _plot_quintile_profiles(
            outputs.quintile_summary,
            QUINTILE_FIGURE_PATH,
        ),
    ]

    report_path = _write_report(
        model_summary=outputs.model_summary,
        quintile_summary=(outputs.quintile_summary),
        yearly_summary=(outputs.yearly_summary),
        coefficient_summary=(outputs.coefficient_summary),
        predictions=predictions,
    )

    logger.info("Linear model evaluation completed.")

    logger.info(
        "Monthly metric rows: %s",
        len(outputs.monthly_metrics),
    )

    logger.info(
        "Turnover transitions: %s",
        len(outputs.monthly_turnover),
    )

    logger.info(
        "Coefficient summaries: %s",
        len(outputs.coefficient_summary),
    )

    display_columns = [
        "model_name",
        "ranking_months",
        "mean_ic",
        "annualized_ic_ir",
        "positive_ic_ratio",
        "mean_top_bottom_spread",
        "mean_top_quintile_precision",
        "mean_top_quintile_turnover",
    ]

    print()
    print("Institutional Quant Equity Research Platform")
    print("Linear model evaluation - Step 7C")
    print("-" * 48)
    print(f"Prediction rows: {len(predictions)}")
    print(f"Out-of-sample dates: {predictions['as_of_date'].nunique()}")
    print(f"Models evaluated: {predictions['model_name'].nunique()}")
    print(f"Monthly metric rows: {len(outputs.monthly_metrics)}")
    print(f"Turnover transitions: {len(outputs.monthly_turnover)}")
    print(f"Coefficient summaries: {len(outputs.coefficient_summary)}")
    print()
    print("Model summary")
    print("-" * 48)
    print(
        outputs.model_summary.loc[
            :,
            display_columns,
        ].to_string(index=False)
    )
    print()
    print("Tables")
    print("-" * 48)

    for output_path in output_paths:
        print(f"- {output_path}")

    print()
    print("Figures")
    print("-" * 48)

    for figure_path in figure_paths:
        print(f"- {figure_path}")

    print()
    print(f"Report: {report_path}")
    print()
    print("Linear model evaluation Step 7C: OK")


if __name__ == "__main__":
    main()
