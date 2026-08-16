"""Evaluate robustness across overlapping rolling evaluation windows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.backtest import (
    ExecutionCostConfig,
    MVPBacktestConfig,
    PerformanceEvaluationConfig,
    build_buy_and_hold_benchmark,
    evaluate_performance,
)
from quant_equity.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_config,
)
from quant_equity.logging_config import configure_logging

NET_DAILY_PATH = PROCESSED_DATA_DIR / "backtest_all_methods_net_daily.parquet"

SPY_DATA_PATH = PROCESSED_DATA_DIR / "benchmark_spy_daily.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

WINDOW_RESULTS_PATH = TABLES_DIR / "robustness_rolling_window_results.csv"

WINDOW_SUMMARY_PATH = TABLES_DIR / "robustness_rolling_window_summary.csv"

CHECKS_PATH = TABLES_DIR / "robustness_rolling_window_checks.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "rolling_window_robustness.md"

WINDOW_LENGTHS_MONTHS = (
    12,
    24,
    36,
)

EXPECTED_METHODS = {
    "alpha_risk_turnover",
    "cvar",
    "median_mad_de",
    "score_weighted",
    "top_n_equal_weight",
}


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write one CSV table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=False,
    )


def _validate_daily(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the stored net daily backtest."""
    required_columns = {
        "date",
        "strategy_name",
        "daily_return",
        "portfolio_value",
    }

    missing = sorted(required_columns.difference(daily.columns))

    if missing:
        raise ValueError("Net daily backtest is missing columns: " + ", ".join(missing) + ".")

    data = daily.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    ).dt.normalize()

    data["strategy_name"] = data["strategy_name"].astype(str)

    data["daily_return"] = pd.to_numeric(
        data["daily_return"],
        errors="coerce",
    )

    if data["date"].isna().any() or data["daily_return"].isna().any():
        raise ValueError("Net daily backtest contains invalid dates or returns.")

    if not np.isfinite(data["daily_return"].to_numpy(dtype=float)).all():
        raise ValueError("Net daily backtest contains non-finite returns.")

    methods = set(data["strategy_name"].unique())

    if methods != EXPECTED_METHODS:
        raise ValueError(
            "Unexpected methods in net daily backtest. "
            f"Expected {sorted(EXPECTED_METHODS)}, "
            f"received {sorted(methods)}."
        )

    if data.duplicated(
        [
            "date",
            "strategy_name",
        ]
    ).any():
        raise ValueError("Net daily backtest contains duplicated date-strategy rows.")

    date_sets = {
        method: tuple(group.sort_values("date")["date"])
        for method, group in data.groupby("strategy_name")
    }

    if len(set(date_sets.values())) != 1:
        raise ValueError("Portfolio methods do not share identical daily dates.")

    return data.sort_values(
        [
            "date",
            "strategy_name",
        ]
    ).reset_index(drop=True)


def _rebase_strategies(
    daily: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    """Rebase strategy return paths for one evaluation window."""
    blocks = []

    for _strategy_name, group in daily.groupby(
        "strategy_name",
        sort=True,
    ):
        group = group.sort_values("date").copy()

        returns = group["daily_return"].to_numpy(dtype=float)

        group["portfolio_value"] = float(initial_capital) * np.cumprod(1.0 + returns)

        blocks.append(group)

    return (
        pd.concat(
            blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "date",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )


def _rebase_benchmark(
    benchmark: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    """Rebase benchmark return path for one evaluation window."""
    data = benchmark.sort_values("date").copy()

    data["portfolio_value"] = float(initial_capital) * np.cumprod(
        1.0 + data["daily_return"].to_numpy(dtype=float)
    )

    return data.reset_index(drop=True)


def _month_end_dates(
    daily: pd.DataFrame,
) -> pd.DatetimeIndex:
    """Return the last observed trading date of each calendar month."""
    dates = (
        daily[
            [
                "date",
            ]
        ]
        .drop_duplicates()
        .sort_values("date")
        .copy()
    )

    dates["month"] = dates["date"].dt.to_period("M")

    return pd.DatetimeIndex(dates.groupby("month")["date"].max().sort_values().to_numpy())


def _window_start_date(
    month_ends: pd.DatetimeIndex,
    *,
    end_index: int,
    window_months: int,
) -> pd.Timestamp:
    """Return the first trading date represented by one rolling window."""
    start_month_end_index = end_index - window_months + 1

    if start_month_end_index < 0:
        raise ValueError("Rolling window does not have enough history.")

    target_month = pd.Timestamp(month_ends[start_month_end_index]).to_period("M")

    return target_month.start_time.normalize()


def _evaluate_window(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    evaluation_config: PerformanceEvaluationConfig,
) -> pd.DataFrame:
    """Evaluate all strategies on one rolling window."""
    strategy_window = daily.loc[
        daily["date"].between(
            start_date,
            end_date,
        )
    ].copy()

    benchmark_window = benchmark.loc[
        benchmark["date"].between(
            start_date,
            end_date,
        )
    ].copy()

    common_dates = sorted(
        set(strategy_window["date"].unique()).intersection(benchmark_window["date"].unique())
    )

    if len(common_dates) < 40:
        raise ValueError("Rolling window contains too few common trading dates.")

    strategy_window = strategy_window.loc[strategy_window["date"].isin(common_dates)]

    benchmark_window = benchmark_window.loc[benchmark_window["date"].isin(common_dates)]

    strategy_window = _rebase_strategies(
        strategy_window,
        initial_capital=initial_capital,
    )

    benchmark_window = _rebase_benchmark(
        benchmark_window,
        initial_capital=initial_capital,
    )

    evaluation = evaluate_performance(
        strategy_window,
        benchmark_window,
        initial_capital=initial_capital,
        config=evaluation_config,
    )

    summary = evaluation.performance_summary.loc[
        ~evaluation.performance_summary["strategy_name"].eq(evaluation_config.benchmark_name)
    ].copy()

    summary["window_start"] = pd.Timestamp(common_dates[0])

    summary["window_end"] = pd.Timestamp(common_dates[-1])

    summary["trading_days"] = int(len(common_dates))

    return summary


def _build_window_results(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    initial_capital: float,
    evaluation_config: PerformanceEvaluationConfig,
) -> pd.DataFrame:
    """Evaluate all requested rolling window lengths."""
    month_ends = _month_end_dates(daily)

    blocks = []

    for window_months in WINDOW_LENGTHS_MONTHS:
        for end_index in range(
            window_months - 1,
            len(month_ends),
        ):
            requested_start = _window_start_date(
                month_ends,
                end_index=end_index,
                window_months=window_months,
            )

            requested_end = pd.Timestamp(month_ends[end_index])

            result = _evaluate_window(
                daily,
                benchmark,
                start_date=requested_start,
                end_date=requested_end,
                initial_capital=initial_capital,
                evaluation_config=evaluation_config,
            )

            result.insert(
                0,
                "window_months",
                int(window_months),
            )

            result.insert(
                1,
                "window_number",
                int(end_index - window_months + 2),
            )

            blocks.append(result)

    return (
        pd.concat(
            blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "window_months",
                "window_end",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )


def _build_summary(
    window_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize performance stability across rolling windows."""
    rows = []

    for (
        window_months,
        strategy_name,
    ), group in window_results.groupby(
        [
            "window_months",
            "strategy_name",
        ],
        sort=True,
    ):
        cagr = group["cagr"].to_numpy(dtype=float)

        sharpe = group["sharpe_ratio"].to_numpy(dtype=float)

        excess_cagr = group["excess_cagr"].to_numpy(dtype=float)

        drawdown = group["maximum_drawdown"].to_numpy(dtype=float)

        rows.append(
            {
                "window_months": int(window_months),
                "strategy_name": str(strategy_name),
                "windows": int(len(group)),
                "mean_cagr": float(np.mean(cagr)),
                "median_cagr": float(np.median(cagr)),
                "cagr_10th_percentile": float(
                    np.quantile(
                        cagr,
                        0.10,
                    )
                ),
                "minimum_cagr": float(np.min(cagr)),
                "positive_cagr_window_ratio": float(np.mean(cagr > 0.0)),
                "mean_sharpe": float(np.mean(sharpe)),
                "median_sharpe": float(np.median(sharpe)),
                "sharpe_10th_percentile": float(
                    np.quantile(
                        sharpe,
                        0.10,
                    )
                ),
                "minimum_sharpe": float(np.min(sharpe)),
                "positive_excess_cagr_window_ratio": float(np.mean(excess_cagr > 0.0)),
                "median_excess_cagr": float(np.median(excess_cagr)),
                "worst_maximum_drawdown": float(np.min(drawdown)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "window_months",
                "median_sharpe",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def _build_checks(
    window_results: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Audit rolling-window robustness outputs."""
    expected_summary_rows = len(WINDOW_LENGTHS_MONTHS) * len(EXPECTED_METHODS)

    methods_per_length = summary.groupby("window_months")["strategy_name"].nunique()

    methods_per_window = window_results.groupby(
        [
            "window_months",
            "window_number",
        ]
    )["strategy_name"].nunique()

    probability_columns = [
        "positive_cagr_window_ratio",
        "positive_excess_cagr_window_ratio",
    ]

    probability_violations = 0

    for column in probability_columns:
        probability_violations += int(
            (
                ~summary[column].between(
                    0.0,
                    1.0,
                )
            ).sum()
        )

    numeric_columns = [
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "excess_cagr",
    ]

    checks = [
        (
            "expected_summary_rows",
            int(len(summary) != expected_summary_rows),
            "Rolling-window summary must contain one row per method and window length.",
        ),
        (
            "five_methods_per_length",
            int(methods_per_length.ne(len(EXPECTED_METHODS)).sum()),
            "Every rolling-window length must contain all five methods.",
        ),
        (
            "five_methods_per_window",
            int(methods_per_window.ne(len(EXPECTED_METHODS)).sum()),
            "Every individual rolling window must contain all five methods.",
        ),
        (
            "finite_window_metrics",
            int((~np.isfinite(window_results[numeric_columns].to_numpy(dtype=float))).sum()),
            "Key rolling-window metrics must remain finite.",
        ),
        (
            "positive_window_lengths",
            int(summary["windows"].le(0).sum()),
            "Every method and window length must contain at least one rolling window.",
        ),
        (
            "valid_window_ratios",
            probability_violations,
            "Rolling-window success ratios must lie between zero and one.",
        ),
        (
            "ordered_windows",
            int((window_results["window_start"] > window_results["window_end"]).sum()),
            "Every rolling window must start before it ends.",
        ),
        (
            "positive_trading_days",
            int(window_results["trading_days"].le(0).sum()),
            "Every rolling window must contain positive trading-day counts.",
        ),
    ]

    return pd.DataFrame(
        [
            {
                "check": name,
                "status": ("PASS" if violations == 0 else "FAIL"),
                "violations": int(violations),
                "description": description,
            }
            for (
                name,
                violations,
                description,
            ) in checks
        ]
    )


def _format_value(
    value: Any,
) -> str:
    """Format one value for Markdown."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.date().isoformat()

    if isinstance(
        value,
        float,
    ):
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


def _build_report(
    summary: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    """Build the rolling-window robustness report."""
    display = summary.loc[
        :,
        [
            "window_months",
            "strategy_name",
            "windows",
            "median_cagr",
            "cagr_10th_percentile",
            "minimum_cagr",
            "positive_cagr_window_ratio",
            "median_sharpe",
            "sharpe_10th_percentile",
            "minimum_sharpe",
            "positive_excess_cagr_window_ratio",
            "median_excess_cagr",
            "worst_maximum_drawdown",
        ],
    ]

    return "\n".join(
        [
            "# Rolling Window Robustness",
            "",
            "## Methodology",
            "",
            (
                "- Uses the frozen net out-of-sample daily return paths "
                "with advanced execution costs already applied."
            ),
            (
                "- Evaluates overlapping 12-, 24- and 36-month windows "
                "ending at each available calendar month."
            ),
            ("- All five methods and SPY are evaluated on identical dates inside each window."),
            (
                "- This test changes only the evaluation window. It does "
                "not re-fit models or tune portfolio parameters."
            ),
            (
                "- A later training-window experiment can separately "
                "compare expanding and rolling model estimation schemes."
            ),
            "",
            "## Rolling-window summary",
            "",
            _to_markdown(display),
            "",
            "## Readiness checks",
            "",
            _to_markdown(checks),
            "",
        ]
    )


def main() -> None:
    """Run rolling-window robustness analysis."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    required_paths = (
        NET_DAILY_PATH,
        SPY_DATA_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    project_config = load_config()

    backtest_config = MVPBacktestConfig.from_mapping(
        project_config.get(
            "mvp_backtest",
            {},
        )
    )

    execution_config = ExecutionCostConfig.from_mapping(
        project_config.get(
            "execution_costs",
            {},
        )
    )

    evaluation_config = PerformanceEvaluationConfig.from_mapping(
        project_config.get(
            "mvp_performance_evaluation",
            {},
        )
    )

    daily = _validate_daily(pd.read_parquet(NET_DAILY_PATH))

    spy_data = pd.read_parquet(SPY_DATA_PATH)

    start_date = daily["date"].min()

    end_date = daily["date"].max()

    benchmark = build_buy_and_hold_benchmark(
        spy_data,
        strategy_name=(evaluation_config.benchmark_name),
        ticker=(evaluation_config.benchmark_ticker),
        start_date=start_date,
        end_date=end_date,
        initial_capital=(backtest_config.initial_capital),
        transaction_cost_bps=(execution_config.linear_cost_bps),
    )

    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()

    logger.info("Evaluating 12-, 24- and 36-month rolling windows.")

    window_results = _build_window_results(
        daily,
        benchmark,
        initial_capital=(backtest_config.initial_capital),
        evaluation_config=(evaluation_config),
    )

    summary = _build_summary(window_results)

    checks = _build_checks(
        window_results,
        summary,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_csv(
        window_results,
        WINDOW_RESULTS_PATH,
    )

    _write_csv(
        summary,
        WINDOW_SUMMARY_PATH,
    )

    _write_csv(
        checks,
        CHECKS_PATH,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        _build_report(
            summary,
            checks,
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(
            f"Rolling-window robustness validation failed with {failed_checks} failed checks."
        )

    logger.info("Rolling-window robustness analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Rolling window robustness")

    print("------------------------------------------------")

    print(f"methods: {summary['strategy_name'].nunique()}")

    print(f"window_lengths: {summary['window_months'].nunique()}")

    print()

    print("Rolling-window summary:")

    print(
        summary.loc[
            :,
            [
                "window_months",
                "strategy_name",
                "windows",
                "median_cagr",
                "cagr_10th_percentile",
                "minimum_cagr",
                "positive_cagr_window_ratio",
                "median_sharpe",
                "sharpe_10th_percentile",
                "minimum_sharpe",
                "positive_excess_cagr_window_ratio",
                "median_excess_cagr",
                "worst_maximum_drawdown",
            ],
        ].to_string(index=False)
    )

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Window results: {WINDOW_RESULTS_PATH}")

    print(f"Summary table: {WINDOW_SUMMARY_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
