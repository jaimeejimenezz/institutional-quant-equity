"""Evaluate the MVP backtest and perform cost sensitivity."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from quant_equity.backtest import (
    MVPBacktestConfig,
    PerformanceEvaluationConfig,
    build_buy_and_hold_benchmark,
    evaluate_performance,
    run_mvp_backtest,
)
from quant_equity.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_config,
)
from quant_equity.logging_config import (
    configure_logging,
)

DAILY_BACKTEST_PATH = PROCESSED_DATA_DIR / "backtest_mvp_daily.parquet"

TARGET_WEIGHTS_PATH = PROCESSED_DATA_DIR / "mvp_target_weights.parquet"

MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

SPY_DATA_PATH = PROCESSED_DATA_DIR / "benchmark_spy_daily.parquet"

COMBINED_DAILY_PATH = PROCESSED_DATA_DIR / "backtest_mvp_with_benchmarks_daily.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

PERFORMANCE_SUMMARY_PATH = TABLES_DIR / "mvp_performance_summary.csv"

YEARLY_SUMMARY_PATH = TABLES_DIR / "mvp_yearly_performance.csv"

MONTHLY_RETURNS_PATH = TABLES_DIR / "mvp_monthly_returns.csv"

DRAWDOWNS_PATH = TABLES_DIR / "mvp_drawdowns.csv"

COST_SENSITIVITY_PATH = TABLES_DIR / "mvp_cost_sensitivity.csv"

FIGURES_DIR = REPORTS_DIR / "figures" / "backtests"

EQUITY_FIGURE_PATH = FIGURES_DIR / "mvp_equity_curves.png"

DRAWDOWN_FIGURE_PATH = FIGURES_DIR / "mvp_drawdowns.png"

COST_FIGURE_PATH = FIGURES_DIR / "mvp_cost_sensitivity.png"

REPORT_PATH = REPORTS_DIR / "backtests" / "mvp_report.md"


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write a CSV file."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=False,
    )


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write a Parquet file atomically."""
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
    data: pd.DataFrame,
) -> str:
    """Convert a dataframe to Markdown."""
    if data.empty:
        return "_No observations._"

    columns = [str(column) for column in data.columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in data.itertuples(
        index=False,
        name=None,
    ):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")

    return "\n".join(lines)


def _plot_equity_curves(
    combined_daily: pd.DataFrame,
    path: Path,
    *,
    initial_capital: float,
) -> None:
    """Plot normalized equity curves."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pivot = combined_daily.pivot(
        index="date",
        columns="strategy_name",
        values="portfolio_value",
    )

    normalized = pivot / initial_capital

    figure, axis = plt.subplots(figsize=(13, 7))

    normalized.plot(
        ax=axis,
        linewidth=1.5,
    )

    axis.set_title("MVP equity curves after transaction costs")

    axis.set_xlabel("Date")
    axis.set_ylabel("Capital multiple")

    axis.grid(alpha=0.3)

    axis.legend(
        loc="best",
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
    )

    plt.close(figure)


def _plot_drawdowns(
    drawdowns: pd.DataFrame,
    path: Path,
) -> None:
    """Plot strategy drawdowns."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pivot = drawdowns.pivot(
        index="date",
        columns="strategy_name",
        values="drawdown",
    )

    figure, axis = plt.subplots(figsize=(13, 7))

    pivot.plot(
        ax=axis,
        linewidth=1.2,
    )

    axis.set_title("MVP portfolio drawdowns")

    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")

    axis.grid(alpha=0.3)

    axis.legend(
        loc="best",
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
    )

    plt.close(figure)


def _plot_cost_sensitivity(
    cost_sensitivity: pd.DataFrame,
    path: Path,
) -> None:
    """Plot CAGR across cost scenarios."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pivot = cost_sensitivity.pivot(
        index="cost_bps",
        columns="strategy_name",
        values="cagr",
    )

    figure, axis = plt.subplots(figsize=(11, 7))

    pivot.plot(
        ax=axis,
        marker="o",
    )

    axis.set_title("CAGR sensitivity to transaction costs")

    axis.set_xlabel("Transaction cost in basis points")

    axis.set_ylabel("CAGR")

    axis.grid(alpha=0.3)

    axis.legend(
        loc="best",
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
    )

    plt.close(figure)


def _build_report(
    performance_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    *,
    base_cost_bps: float,
) -> str:
    """Build the final MVP report."""
    selected_columns = [
        "strategy_name",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "calmar_ratio",
        "beta_vs_spy",
        "information_ratio_vs_spy",
        "total_transaction_cost",
    ]

    yearly_returns = yearly_summary.pivot(
        index="year",
        columns="strategy_name",
        values="total_return",
    ).reset_index()

    cost_table = cost_sensitivity.loc[
        :,
        [
            "cost_bps",
            "strategy_name",
            "cagr",
            "sharpe_ratio",
            "maximum_drawdown",
            "final_portfolio_value",
            "total_transaction_cost",
        ],
    ].sort_values(
        [
            "cost_bps",
            "strategy_name",
        ]
    )

    return "\n".join(
        [
            "# MVP Backtest Report — Step 8C",
            "",
            "## Evaluation design",
            "",
            ("- All model portfolios use genuinely out-of-sample predictions."),
            ("- Signals are executed at the adjusted opening price of the following session."),
            ("- Positions drift between monthly rebalances."),
            ("- Fractional shares are allowed."),
            (
                "- The primary comparison uses transaction costs "
                f"of `{base_cost_bps:.2f}` basis points."
            ),
            ("- SPY is evaluated as a buy-and-hold benchmark from the first execution date."),
            "",
            "## Main performance comparison",
            "",
            _to_markdown(
                performance_summary.loc[
                    :,
                    selected_columns,
                ]
            ),
            "",
            "## Calendar-year returns",
            "",
            _to_markdown(yearly_returns),
            "",
            "## Cost sensitivity",
            "",
            _to_markdown(cost_table),
            "",
            "## Interpretation rule",
            "",
            (
                "The preferred MVP strategy should not be chosen "
                "only from final capital. It should combine a "
                "competitive CAGR, Sharpe and Sortino ratio, "
                "controlled drawdown, reasonable turnover, "
                "positive performance across several years and "
                "robustness under higher transaction costs."
            ),
            "",
        ]
    )


def main() -> None:
    """Run Step 8C."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    project_config = load_config()

    backtest_config = MVPBacktestConfig.from_mapping(
        project_config.get(
            "mvp_backtest",
            {},
        )
    )

    evaluation_config = PerformanceEvaluationConfig.from_mapping(
        project_config.get(
            "mvp_performance_evaluation",
            {},
        )
    )

    required_paths = (
        DAILY_BACKTEST_PATH,
        TARGET_WEIGHTS_PATH,
        MARKET_DATA_PATH,
        SPY_DATA_PATH,
    )

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Required input not found: {required_path}")

    base_daily = pd.read_parquet(DAILY_BACKTEST_PATH)

    target_weights = pd.read_parquet(TARGET_WEIGHTS_PATH)

    market_data = pd.read_parquet(MARKET_DATA_PATH)

    spy_data = pd.read_parquet(SPY_DATA_PATH)

    start_date = pd.to_datetime(base_daily["date"]).min()

    end_date = pd.to_datetime(base_daily["date"]).max()

    base_spy = build_buy_and_hold_benchmark(
        spy_data,
        strategy_name=(evaluation_config.benchmark_name),
        ticker=(evaluation_config.benchmark_ticker),
        start_date=start_date,
        end_date=end_date,
        initial_capital=(backtest_config.initial_capital),
        transaction_cost_bps=(backtest_config.transaction_cost_bps),
    )

    base_evaluation = evaluate_performance(
        base_daily,
        base_spy,
        initial_capital=(backtest_config.initial_capital),
        config=evaluation_config,
    )

    cost_sensitivity_frames: list[pd.DataFrame] = []

    for cost_bps in evaluation_config.cost_scenarios_bps:
        if cost_bps == backtest_config.transaction_cost_bps:
            scenario_daily = base_daily
        else:
            scenario_config = replace(
                backtest_config,
                transaction_cost_bps=(cost_bps),
            )

            scenario_outputs = run_mvp_backtest(
                target_weights,
                market_data,
                config=(scenario_config),
            )

            scenario_daily = scenario_outputs.daily_performance

        scenario_spy = build_buy_and_hold_benchmark(
            spy_data,
            strategy_name=(evaluation_config.benchmark_name),
            ticker=(evaluation_config.benchmark_ticker),
            start_date=start_date,
            end_date=end_date,
            initial_capital=(backtest_config.initial_capital),
            transaction_cost_bps=(cost_bps),
        )

        scenario_evaluation = evaluate_performance(
            scenario_daily,
            scenario_spy,
            initial_capital=(backtest_config.initial_capital),
            config=evaluation_config,
        )

        scenario_summary = scenario_evaluation.performance_summary.copy()

        scenario_summary.insert(
            0,
            "cost_bps",
            cost_bps,
        )

        cost_sensitivity_frames.append(scenario_summary)

        logger.info(
            "Completed cost scenario: %.2f bps.",
            cost_bps,
        )

    cost_sensitivity = (
        pd.concat(
            cost_sensitivity_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "cost_bps",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )

    _write_parquet_atomically(
        base_evaluation.combined_daily,
        COMBINED_DAILY_PATH,
    )

    _write_csv(
        base_evaluation.performance_summary,
        PERFORMANCE_SUMMARY_PATH,
    )

    _write_csv(
        base_evaluation.yearly_summary,
        YEARLY_SUMMARY_PATH,
    )

    _write_csv(
        base_evaluation.monthly_returns,
        MONTHLY_RETURNS_PATH,
    )

    _write_csv(
        base_evaluation.drawdowns,
        DRAWDOWNS_PATH,
    )

    _write_csv(
        cost_sensitivity,
        COST_SENSITIVITY_PATH,
    )

    _plot_equity_curves(
        base_evaluation.combined_daily,
        EQUITY_FIGURE_PATH,
        initial_capital=(backtest_config.initial_capital),
    )

    _plot_drawdowns(
        base_evaluation.drawdowns,
        DRAWDOWN_FIGURE_PATH,
    )

    _plot_cost_sensitivity(
        cost_sensitivity,
        COST_FIGURE_PATH,
    )

    report = _build_report(
        base_evaluation.performance_summary,
        base_evaluation.yearly_summary,
        cost_sensitivity,
        base_cost_bps=(backtest_config.transaction_cost_bps),
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    logger.info("MVP performance evaluation completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("MVP financial evaluation - Step 8C")
    print("-" * 60)
    print(
        "Strategies and benchmarks: "
        f"{base_evaluation.performance_summary['strategy_name'].nunique()}"
    )
    print(f"Trading days: {base_daily['date'].nunique()}")
    print(f"Cost scenarios: {len(evaluation_config.cost_scenarios_bps)}")
    print()
    print(
        base_evaluation.performance_summary.loc[
            :,
            [
                "strategy_name",
                "total_return",
                "cagr",
                "annualized_volatility",
                "sharpe_ratio",
                "sortino_ratio",
                "maximum_drawdown",
                "calmar_ratio",
                "beta_vs_spy",
                "total_transaction_cost",
            ],
        ].to_string(index=False)
    )
    print()
    print(f"Performance summary: {PERFORMANCE_SUMMARY_PATH}")
    print(f"Yearly results: {YEARLY_SUMMARY_PATH}")
    print(f"Cost sensitivity: {COST_SENSITIVITY_PATH}")
    print(f"Equity figure: {EQUITY_FIGURE_PATH}")
    print(f"Drawdown figure: {DRAWDOWN_FIGURE_PATH}")
    print(f"Final report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
