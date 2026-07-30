"""Run univariate research for monthly technical factors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.features import (
    TECHNICAL_MODEL_FEATURE_COLUMNS,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.research import (
    TechnicalFactorResearchConfig,
    run_technical_factor_research,
)

FEATURES_PATH = PROCESSED_DATA_DIR / "features_technical_monthly.parquet"

LABELS_PATH = PROCESSED_DATA_DIR / "labels_monthly.parquet"

PANEL_PATH = PROCESSED_DATA_DIR / "technical_factor_research_panel.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

RESEARCH_DIR = REPORTS_DIR / "research"

REPORT_PATH = RESEARCH_DIR / "technical_factor_report.md"


def _format_markdown_value(
    value: Any,
) -> str:
    """Format a scalar for a Markdown table."""
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


def _dataframe_to_markdown(
    frame: pd.DataFrame,
) -> str:
    """Convert a dataframe into a simple Markdown table."""
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
        values = [_format_markdown_value(value) for value in row]

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> Path:
    """Write a research table to CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
    )

    return path


def _write_report(
    *,
    result: Any,
    config: TechnicalFactorResearchConfig,
    path: Path,
) -> Path:
    """Write the initial technical-factor research report."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overview = result.ic_summary.merge(
        result.spread_summary,
        on="signal",
        how="left",
        suffixes=(
            "_ic",
            "_spread",
        ),
    ).merge(
        result.turnover_summary.loc[
            :,
            [
                "signal",
                "mean_turnover",
                "median_turnover",
                "selected_quantile",
            ],
        ],
        on="signal",
        how="left",
    )

    overview = overview.sort_values(
        "abs_mean_ic",
        ascending=False,
    ).loc[
        :,
        [
            "signal",
            "months_ic",
            "mean_ic",
            "annualized_ic_ir",
            "positive_month_ratio",
            "preferred_direction",
            "mean_top_bottom_spread",
            "positive_spread_ratio",
            "mean_quintile_monotonicity",
            "mean_turnover",
            "selected_quantile",
        ],
    ]

    quintile_profiles = (
        result.quintile_summary.pivot(
            index="signal",
            columns="quintile",
            values="mean_target_21d_excess",
        )
        .rename(columns=lambda column: f"Q{int(column)}")
        .reset_index()
    )

    lines = [
        "# Technical Factor Research Report",
        "",
        "## Research design",
        "",
        (f"- Research start date: `{config.research_start_date.date()}`"),
        (f"- Research end date: `{config.research_end_date.date()}`"),
        (f"- Prediction target: `{config.target_column}`"),
        (f"- Quantiles: `{config.number_of_quantiles}`"),
        (f"- Minimum cross-section: `{config.minimum_cross_section_size}`"),
        (f"- Technical signals: `{len(TECHNICAL_MODEL_FEATURE_COLUMNS)}`"),
        (f"- Research rows: `{len(result.panel)}`"),
        (f"- Research dates: `{result.panel['as_of_date'].nunique()}`"),
        "",
        "## Temporal interpretation",
        "",
        ("All technical signals use market information available on or before `as_of_date`."),
        "",
        (
            "The future return begins strictly after "
            "`as_of_date`. Future returns are used only "
            "as evaluation targets and never as model inputs."
        ),
        "",
        (
            "Initial factor selection is restricted to the "
            "configured research period ending in 2019. "
            "Observations from 2020 onward are preserved for "
            "subsequent out-of-sample evaluation."
        ),
        "",
        "## Factor overview",
        "",
        _dataframe_to_markdown(overview),
        "",
        "## Average excess return by quintile",
        "",
        _dataframe_to_markdown(quintile_profiles),
        "",
        "## Metric interpretation",
        "",
        (
            "- `mean_ic`: average monthly Spearman "
            "correlation between the signal ranking and "
            "future relative returns."
        ),
        (
            "- `annualized_ic_ir`: average IC divided by "
            "its volatility and annualized with square root "
            "of 12."
        ),
        (
            "- `preferred_direction`: whether high or low "
            "values of the signal were associated with "
            "better subsequent returns."
        ),
        (
            "- `mean_top_bottom_spread`: average return "
            "difference between Q5 and Q1 before applying "
            "any direction reversal."
        ),
        (
            "- `mean_turnover`: proportion of members "
            "replaced in the economically preferred "
            "quintile between consecutive months."
        ),
        "",
        "## Current status",
        "",
        ("This report is descriptive. No final feature selection is made in Step 6A."),
        "",
        (
            "Step 6B will add correlation and redundancy "
            "analysis, yearly and sector results, figures "
            "and the final selection of technical signals."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path


def main() -> None:
    """Execute technical-factor research."""
    config = load_config()

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "research_technical_factors.log"),
    )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Processed technical features were not found: {FEATURES_PATH}")

    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Monthly labels were not found: {LABELS_PATH}")

    research_config = TechnicalFactorResearchConfig.from_mapping(
        config["technical_factor_research"]
    )

    technical_features = pd.read_parquet(FEATURES_PATH)

    labels = pd.read_parquet(LABELS_PATH)

    result = run_technical_factor_research(
        technical_features,
        labels,
        config=research_config,
    )

    PANEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.panel.to_parquet(
        PANEL_PATH,
        index=False,
    )

    output_paths = [
        _write_csv(
            result.monthly_ic,
            TABLES_DIR / "factor_ic_monthly.csv",
        ),
        _write_csv(
            result.ic_summary,
            TABLES_DIR / "factor_ic.csv",
        ),
        _write_csv(
            result.monthly_quintiles,
            TABLES_DIR / "factor_quintiles_monthly.csv",
        ),
        _write_csv(
            result.quintile_summary,
            TABLES_DIR / "factor_quintiles.csv",
        ),
        _write_csv(
            result.monthly_spreads,
            TABLES_DIR / "factor_spreads_monthly.csv",
        ),
        _write_csv(
            result.spread_summary,
            TABLES_DIR / "factor_spreads.csv",
        ),
        _write_csv(
            result.monthly_turnover,
            TABLES_DIR / "factor_turnover_monthly.csv",
        ),
        _write_csv(
            result.turnover_summary,
            TABLES_DIR / "factor_turnover.csv",
        ),
    ]

    report_path = _write_report(
        result=result,
        config=research_config,
        path=REPORT_PATH,
    )

    logger.info("Technical-factor research completed.")

    logger.info(
        "Research rows: %s",
        len(result.panel),
    )

    logger.info(
        "Research dates: %s",
        result.panel["as_of_date"].nunique(),
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Technical factor research - Step 6A")
    print("------------------------------------------------")
    print(f"Research rows: {len(result.panel)}")
    print(f"Research dates: {result.panel['as_of_date'].nunique()}")
    print(f"Research tickers: {result.panel['ticker'].nunique()}")
    print(f"Signals evaluated: {len(TECHNICAL_MODEL_FEATURE_COLUMNS)}")
    print(
        "Research period: "
        f"{result.panel['as_of_date'].min().date()} "
        "to "
        f"{result.panel['as_of_date'].max().date()}"
    )
    print(f"Panel: {PANEL_PATH}")
    print(f"Report: {report_path}")

    print()
    print("Strongest signals by absolute mean IC")
    print("------------------------------------------------")

    display_columns = [
        "signal",
        "months",
        "mean_ic",
        "annualized_ic_ir",
        "positive_month_ratio",
        "preferred_direction",
    ]

    print(
        result.ic_summary.loc[
            :,
            display_columns,
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("Tables")
    print("------------------------------------------------")

    for output_path in output_paths:
        print(f"- {output_path}")

    print()
    print("Technical factor research Step 6A: OK")


if __name__ == "__main__":
    main()
