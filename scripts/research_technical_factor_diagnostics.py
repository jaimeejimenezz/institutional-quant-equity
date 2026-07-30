"""Run stability, redundancy and sector diagnostics for technical factors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
)
from quant_equity.research.technical_factor_diagnostics import (
    ResearchPeriod,
    TechnicalFactorDiagnosticsConfig,
    build_mean_correlation_matrix,
    build_selection_diagnostics,
    calculate_monthly_sector_ic,
    calculate_monthly_signal_correlations,
    summarize_ic_by_period,
    summarize_sector_ic,
    summarize_signal_correlations,
)

PANEL_PATH = PROCESSED_DATA_DIR / "technical_factor_research_panel.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

FIGURES_DIR = REPORTS_DIR / "figures" / "technical_factors"

REPORT_PATH = REPORTS_DIR / "research" / "technical_factor_report.md"

MONTHLY_IC_PATH = TABLES_DIR / "factor_ic_monthly.csv"

IC_SUMMARY_PATH = TABLES_DIR / "factor_ic.csv"

QUINTILE_SUMMARY_PATH = TABLES_DIR / "factor_quintiles.csv"

SPREAD_SUMMARY_PATH = TABLES_DIR / "factor_spreads.csv"

TURNOVER_SUMMARY_PATH = TABLES_DIR / "factor_turnover.csv"


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
    """Convert a dataframe into a Markdown table."""
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
        values = [_format_value(value) for value in row]

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _short_signal_name(
    signal: str,
) -> str:
    """Remove the common suffix from figure labels."""
    return signal.removesuffix("_sector_neutral")


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> Path:
    """Write a CSV table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
    )

    return path


def _plot_correlation_heatmap(
    matrix: pd.DataFrame,
    path: Path,
) -> Path:
    """Plot the mean signal-correlation matrix."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = [_short_signal_name(signal) for signal in matrix.index]

    figure, axis = plt.subplots(figsize=(14, 12))

    image = axis.imshow(
        matrix.to_numpy(dtype=float),
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        aspect="auto",
    )

    axis.set_xticks(np.arange(len(labels)))

    axis.set_yticks(np.arange(len(labels)))

    axis.set_xticklabels(
        labels,
        rotation=90,
        fontsize=8,
    )

    axis.set_yticklabels(
        labels,
        fontsize=8,
    )

    axis.set_title("Mean monthly cross-sectional Spearman correlation")

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path


def _plot_yearly_ic_heatmap(
    yearly_ic: pd.DataFrame,
    *,
    selected_signals: list[str],
    path: Path,
) -> Path:
    """Plot direction-adjusted yearly IC."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = (
        yearly_ic.loc[yearly_ic["signal"].isin(selected_signals)]
        .pivot(
            index="signal",
            columns="period",
            values="directional_mean_ic",
        )
        .reindex(selected_signals)
    )

    labels = [_short_signal_name(signal) for signal in plot_data.index]

    figure, axis = plt.subplots(figsize=(10, 7))

    image = axis.imshow(
        plot_data.to_numpy(dtype=float),
        cmap="coolwarm",
        aspect="auto",
    )

    axis.set_xticks(np.arange(len(plot_data.columns)))

    axis.set_yticks(np.arange(len(labels)))

    axis.set_xticklabels(
        plot_data.columns,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        labels,
        fontsize=8,
    )

    axis.set_title("Direction-adjusted mean IC by year")

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

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
    *,
    selected_signals: list[str],
    path: Path,
) -> Path:
    """Plot average future excess return by quintile."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(figsize=(11, 7))

    for signal in selected_signals:
        signal_data = quintile_summary.loc[quintile_summary["signal"].eq(signal)].sort_values(
            "quintile"
        )

        if signal_data.empty:
            continue

        axis.plot(
            signal_data["quintile"],
            signal_data["mean_target_21d_excess"] * 100.0,
            marker="o",
            label=_short_signal_name(signal),
        )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_xticks(sorted(quintile_summary["quintile"].unique()))

    axis.set_xlabel("Signal quintile")

    axis.set_ylabel("Mean future excess return (%)")

    axis.set_title("Average future return by signal quintile")

    axis.legend(
        fontsize=8,
        ncol=2,
    )

    axis.grid(alpha=0.25)

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path


def _append_report(
    *,
    report_path: Path,
    diagnostics: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    yearly_ic: pd.DataFrame,
    subperiod_ic: pd.DataFrame,
    sector_ic_summary: pd.DataFrame,
    correlation_threshold: float,
) -> Path:
    """Append idempotent Step 6B results to the factor report."""
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = (
        report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "# Technical Factor Research Report\n"
    )

    marker = "## Step 6B — Stability, redundancy and sector diagnostics"

    base_report = existing.split(marker)[0].rstrip()

    redundant_pairs = correlation_summary.loc[
        correlation_summary["mean_absolute_correlation"].ge(correlation_threshold)
    ].head(20)

    diagnostic_display = diagnostics.loc[
        :,
        [
            "signal",
            "preliminary_status",
            "abs_mean_ic",
            "directional_month_ratio",
            "directional_spread",
            "directional_positive_spread_ratio",
            "mean_turnover",
            "positive_year_ratio",
            "positive_subperiod_ratio",
            "positive_sector_ratio",
            "strongest_correlated_signal",
            "strongest_mean_absolute_correlation",
        ],
    ]

    strongest_signals = diagnostics.dropna(subset=["abs_mean_ic"]).head(10)["signal"].tolist()

    yearly_display = (
        yearly_ic.loc[yearly_ic["signal"].isin(strongest_signals)]
        .pivot(
            index="signal",
            columns="period",
            values="directional_mean_ic",
        )
        .reset_index()
    )

    subperiod_display = subperiod_ic.loc[
        subperiod_ic["signal"].isin(strongest_signals),
        [
            "period",
            "signal",
            "months",
            "directional_mean_ic",
            "directional_month_ratio",
        ],
    ]

    valid_sector_summary = sector_ic_summary.loc[
        sector_ic_summary["signal"].isin(strongest_signals)
    ].sort_values(
        [
            "signal",
            "directional_mean_ic",
        ],
        ascending=[
            True,
            False,
        ],
    )

    lines = [
        base_report,
        "",
        marker,
        "",
        "### Objective",
        "",
        (
            "This section evaluates temporal stability, "
            "sector breadth and redundancy before the "
            "final technical-feature selection."
        ),
        "",
        (
            "All diagnostics remain restricted to the "
            "2014-2019 research window. Data from 2020 "
            "onward are not used for feature selection."
        ),
        "",
        "### Preliminary selection diagnostics",
        "",
        _to_markdown(diagnostic_display),
        "",
        "### Highly correlated signal pairs",
        "",
        (
            f"Pairs shown below have an average absolute "
            f"monthly correlation of at least "
            f"`{correlation_threshold:.2f}`."
        ),
        "",
        _to_markdown(redundant_pairs),
        "",
        "### Direction-adjusted IC by year",
        "",
        ("Positive values mean that the signal worked in its preferred economic direction."),
        "",
        _to_markdown(yearly_display),
        "",
        "### Direction-adjusted IC by subperiod",
        "",
        _to_markdown(subperiod_display),
        "",
        "### Sector results for the strongest signals",
        "",
        _to_markdown(valid_sector_summary),
        "",
        "### Interpretation of preliminary statuses",
        "",
        ("- `candidate`: useful preliminary evidence without an immediate major warning."),
        ("- `candidate_high_turnover`: predictive evidence exists, but trading intensity is high."),
        (
            "- `candidate_unstable`: signal strength or "
            "spread direction is not sufficiently consistent."
        ),
        ("- `review_redundancy`: the signal overlaps strongly with another variable."),
        ("- `weak_candidate`: limited individual predictive evidence."),
        ("- `drop_very_weak`: near-zero mean IC."),
        ("- `drop_no_variation`: the signal cannot meaningfully rank the current universe."),
        "",
        "### Current status",
        "",
        (
            "The statuses are diagnostic labels, not an "
            "automatic final decision. The final list must "
            "also consider economic interpretation and "
            "which member of each redundant family should "
            "be retained."
        ),
        "",
    ]

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_path


def main() -> None:
    """Execute Step 6B diagnostics."""
    config = load_config()

    logger = configure_logging(
        level=config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "research_technical_factor_diagnostics.log"),
    )

    required_paths = [
        PANEL_PATH,
        MONTHLY_IC_PATH,
        IC_SUMMARY_PATH,
        QUINTILE_SUMMARY_PATH,
        SPREAD_SUMMARY_PATH,
        TURNOVER_SUMMARY_PATH,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Step 6A output was not found: {required_path}")

    diagnostics_config = TechnicalFactorDiagnosticsConfig.from_mapping(
        config["technical_factor_diagnostics"]
    )

    research_config = TechnicalFactorResearchConfig.from_mapping(
        config["technical_factor_research"]
    )

    panel = pd.read_parquet(PANEL_PATH)

    panel["as_of_date"] = pd.to_datetime(
        panel["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    monthly_ic = pd.read_csv(
        MONTHLY_IC_PATH,
        parse_dates=["as_of_date"],
    )

    ic_summary = pd.read_csv(IC_SUMMARY_PATH)

    quintile_summary = pd.read_csv(QUINTILE_SUMMARY_PATH)

    spread_summary = pd.read_csv(SPREAD_SUMMARY_PATH)

    turnover_summary = pd.read_csv(TURNOVER_SUMMARY_PATH)

    preferred_direction_by_signal = ic_summary.set_index("signal")["preferred_direction"].to_dict()

    monthly_correlations = calculate_monthly_signal_correlations(
        panel,
        signal_columns=(TECHNICAL_MODEL_FEATURE_COLUMNS),
        minimum_observations=(diagnostics_config.minimum_pair_observations),
    )

    correlation_summary = summarize_signal_correlations(monthly_correlations)

    correlation_matrix = build_mean_correlation_matrix(
        correlation_summary,
        signal_columns=(TECHNICAL_MODEL_FEATURE_COLUMNS),
    )

    research_years = sorted(panel["as_of_date"].dt.year.dropna().unique())

    yearly_periods = tuple(
        ResearchPeriod(
            name=str(int(year)),
            start_date=pd.Timestamp(
                year=int(year),
                month=1,
                day=1,
            ),
            end_date=pd.Timestamp(
                year=int(year),
                month=12,
                day=31,
            ),
        )
        for year in research_years
    )

    yearly_ic = summarize_ic_by_period(
        monthly_ic,
        periods=yearly_periods,
        preferred_direction_by_signal=(preferred_direction_by_signal),
    )

    subperiod_ic = summarize_ic_by_period(
        monthly_ic,
        periods=diagnostics_config.subperiods,
        preferred_direction_by_signal=(preferred_direction_by_signal),
    )

    monthly_sector_ic = calculate_monthly_sector_ic(
        panel,
        target_column=(research_config.target_column),
        signal_columns=(TECHNICAL_MODEL_FEATURE_COLUMNS),
        minimum_sector_size=(diagnostics_config.minimum_sector_cross_section_size),
    )

    sector_ic_summary = summarize_sector_ic(
        monthly_sector_ic,
        preferred_direction_by_signal=(preferred_direction_by_signal),
    )

    diagnostics = build_selection_diagnostics(
        ic_summary,
        spread_summary,
        turnover_summary,
        yearly_ic,
        subperiod_ic,
        sector_ic_summary,
        correlation_summary,
        config=diagnostics_config,
    )

    output_paths = [
        _write_csv(
            monthly_correlations,
            TABLES_DIR / "factor_correlations_monthly.csv",
        ),
        _write_csv(
            correlation_summary,
            TABLES_DIR / "factor_correlations.csv",
        ),
        _write_csv(
            correlation_matrix.reset_index(),
            TABLES_DIR / "factor_correlation_matrix.csv",
        ),
        _write_csv(
            yearly_ic,
            TABLES_DIR / "factor_ic_by_year.csv",
        ),
        _write_csv(
            subperiod_ic,
            TABLES_DIR / "factor_ic_by_subperiod.csv",
        ),
        _write_csv(
            monthly_sector_ic,
            TABLES_DIR / "factor_ic_by_sector_monthly.csv",
        ),
        _write_csv(
            sector_ic_summary,
            TABLES_DIR / "factor_ic_by_sector.csv",
        ),
        _write_csv(
            diagnostics,
            TABLES_DIR / "factor_selection_diagnostics.csv",
        ),
    ]

    strongest_signals = (
        diagnostics.dropna(subset=["abs_mean_ic"])
        .head(diagnostics_config.top_signals_in_figures)["signal"]
        .tolist()
    )

    figure_paths = [
        _plot_correlation_heatmap(
            correlation_matrix,
            FIGURES_DIR / "factor_correlation_heatmap.png",
        ),
        _plot_yearly_ic_heatmap(
            yearly_ic,
            selected_signals=strongest_signals,
            path=(FIGURES_DIR / "factor_ic_by_year.png"),
        ),
        _plot_quintile_profiles(
            quintile_summary,
            selected_signals=strongest_signals,
            path=(FIGURES_DIR / "factor_quintile_profiles.png"),
        ),
    ]

    report_path = _append_report(
        report_path=REPORT_PATH,
        diagnostics=diagnostics,
        correlation_summary=correlation_summary,
        yearly_ic=yearly_ic,
        subperiod_ic=subperiod_ic,
        sector_ic_summary=sector_ic_summary,
        correlation_threshold=(diagnostics_config.correlation_threshold),
    )

    redundant_pairs = correlation_summary.loc[
        correlation_summary["mean_absolute_correlation"].ge(
            diagnostics_config.correlation_threshold
        )
    ]

    status_counts = diagnostics["preliminary_status"].value_counts(dropna=False).sort_index()

    logger.info("Technical-factor diagnostics completed.")

    logger.info(
        "Signal pairs evaluated: %s",
        len(correlation_summary),
    )

    logger.info(
        "Highly correlated pairs: %s",
        len(redundant_pairs),
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Technical factor diagnostics - Step 6B")
    print("------------------------------------------------")
    print(f"Signals evaluated: {len(TECHNICAL_MODEL_FEATURE_COLUMNS)}")
    print(f"Signal pairs evaluated: {len(correlation_summary)}")
    print(f"Highly correlated pairs: {len(redundant_pairs)}")
    print(f"Years evaluated: {len(yearly_periods)}")
    print(f"Subperiods evaluated: {len(diagnostics_config.subperiods)}")
    print(f"Sector-signal summaries: {len(sector_ic_summary)}")

    print()
    print("Preliminary status counts")
    print("------------------------------------------------")
    print(status_counts.to_string())

    print()
    print("Strongest correlated pairs")
    print("------------------------------------------------")

    correlation_display_columns = [
        "first_signal",
        "second_signal",
        "months",
        "mean_correlation",
        "mean_absolute_correlation",
    ]

    print(
        correlation_summary.loc[
            :,
            correlation_display_columns,
        ]
        .head(15)
        .to_string(index=False)
    )

    print()
    print("Selection diagnostics")
    print("------------------------------------------------")

    diagnostic_display_columns = [
        "signal",
        "preliminary_status",
        "abs_mean_ic",
        "directional_spread",
        "mean_turnover",
        "positive_year_ratio",
        "positive_subperiod_ratio",
        "positive_sector_ratio",
    ]

    print(
        diagnostics.loc[
            :,
            diagnostic_display_columns,
        ].to_string(index=False)
    )

    print()
    print("Tables")
    print("------------------------------------------------")

    for output_path in output_paths:
        print(f"- {output_path}")

    print()
    print("Figures")
    print("------------------------------------------------")

    for figure_path in figure_paths:
        print(f"- {figure_path}")

    print()
    print(f"Report: {report_path}")
    print()
    print("Technical factor diagnostics Step 6B: OK")


if __name__ == "__main__":
    main()
