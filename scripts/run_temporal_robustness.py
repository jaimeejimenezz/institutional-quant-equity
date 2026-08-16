"""Evaluate temporal robustness of the five portfolio methods."""

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

YEARLY_PATH = TABLES_DIR / "robustness_yearly_performance.csv"

REGIME_PATH = TABLES_DIR / "robustness_regime_performance.csv"

CONDITIONAL_MONTHS_PATH = TABLES_DIR / "robustness_conditional_months.csv"

CHECKS_PATH = TABLES_DIR / "robustness_temporal_checks.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "temporal_robustness.md"

EXPECTED_METHODS = {
    "alpha_risk_turnover",
    "cvar",
    "median_mad_de",
    "score_weighted",
    "top_n_equal_weight",
}

REGIME_WINDOWS = (
    (
        "pre_covid_available",
        "2020-02-03",
        "2020-02-18",
        True,
    ),
    (
        "covid_crash",
        "2020-02-19",
        "2020-03-23",
        False,
    ),
    (
        "covid_recovery_2020_2021",
        "2020-03-24",
        "2021-12-31",
        False,
    ),
    (
        "bear_market_2022",
        "2022-01-03",
        "2022-12-30",
        False,
    ),
    (
        "expansion_2023_2024",
        "2023-01-03",
        "2024-12-31",
        False,
    ),
    (
        "recent_2025_2026",
        "2025-01-02",
        "2026-12-31",
        False,
    ),
)


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write one CSV report table."""
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
        "portfolio_value",
        "daily_return",
    }

    missing = sorted(required_columns.difference(daily.columns))

    if missing:
        raise ValueError("Net daily backtest is missing columns: " + ", ".join(missing) + ".")

    data = daily.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    ).dt.normalize()

    if data["date"].isna().any():
        raise ValueError("Net daily backtest contains invalid dates.")

    data["strategy_name"] = data["strategy_name"].astype(str)

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


def _rebase_daily(
    data: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    """Rebase a daily-return table without changing its return path."""
    blocks = []

    for strategy_name, group in data.groupby(
        "strategy_name",
        sort=True,
    ):
        group = group.sort_values("date").copy()

        returns = pd.to_numeric(
            group["daily_return"],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.isfinite(returns).all():
            raise ValueError("Daily returns contain non-finite observations.")

        group["portfolio_value"] = float(initial_capital) * np.cumprod(1.0 + returns)

        group["strategy_name"] = str(strategy_name)

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
    """Rebase the benchmark return path."""
    data = benchmark.sort_values("date").copy()

    returns = pd.to_numeric(
        data["daily_return"],
        errors="raise",
    ).to_numpy(dtype=float)

    data["portfolio_value"] = float(initial_capital) * np.cumprod(1.0 + returns)

    return data.reset_index(drop=True)


def _evaluate_window(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    evaluation_config: PerformanceEvaluationConfig,
) -> pd.DataFrame:
    """Evaluate all methods on one shared temporal window."""
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

    if strategy_window.empty or benchmark_window.empty:
        return pd.DataFrame()

    strategy_dates = set(strategy_window["date"].unique())

    benchmark_dates = set(benchmark_window["date"].unique())

    common_dates = sorted(strategy_dates.intersection(benchmark_dates))

    if len(common_dates) < 2:
        return pd.DataFrame()

    strategy_window = strategy_window.loc[strategy_window["date"].isin(common_dates)]

    benchmark_window = benchmark_window.loc[benchmark_window["date"].isin(common_dates)]

    strategy_window = _rebase_daily(
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

    summary["trading_days"] = len(common_dates)

    return summary


def _build_yearly_results(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    initial_capital: float,
    evaluation_config: PerformanceEvaluationConfig,
) -> pd.DataFrame:
    """Evaluate each observed calendar year separately."""
    start_date = daily["date"].min()

    end_date = daily["date"].max()

    blocks = []

    for year in range(
        int(start_date.year),
        int(end_date.year) + 1,
    ):
        year_start = max(
            pd.Timestamp(f"{year}-01-01"),
            start_date,
        )

        year_end = min(
            pd.Timestamp(f"{year}-12-31"),
            end_date,
        )

        result = _evaluate_window(
            daily,
            benchmark,
            start_date=year_start,
            end_date=year_end,
            initial_capital=initial_capital,
            evaluation_config=evaluation_config,
        )

        if result.empty:
            continue

        result.insert(
            0,
            "year",
            year,
        )

        result["partial_calendar_year"] = bool(
            year_start > pd.Timestamp(f"{year}-01-01") or year_end < pd.Timestamp(f"{year}-12-31")
        )

        blocks.append(result)

    return (
        pd.concat(
            blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "year",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )


def _build_regime_results(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    initial_capital: float,
    evaluation_config: PerformanceEvaluationConfig,
) -> pd.DataFrame:
    """Evaluate pre-defined economically interpretable windows."""
    available_start = daily["date"].min()

    available_end = daily["date"].max()

    blocks = []

    for (
        regime,
        raw_start,
        raw_end,
        short_sample_warning,
    ) in REGIME_WINDOWS:
        start_date = max(
            pd.Timestamp(raw_start),
            available_start,
        )

        end_date = min(
            pd.Timestamp(raw_end),
            available_end,
        )

        if start_date > end_date:
            continue

        result = _evaluate_window(
            daily,
            benchmark,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            evaluation_config=evaluation_config,
        )

        if result.empty:
            continue

        result.insert(
            0,
            "regime",
            regime,
        )

        result["short_sample_warning"] = bool(
            short_sample_warning or int(result["trading_days"].iloc[0]) < 40
        )

        blocks.append(result)

    return (
        pd.concat(
            blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "regime",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )


def _monthly_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Compound daily returns into calendar-month returns."""
    working = data.loc[
        :,
        [
            "date",
            "strategy_name",
            "daily_return",
        ],
    ].copy()

    working["month"] = working["date"].dt.to_period("M").dt.to_timestamp("M")

    return (
        working.groupby(
            [
                "month",
                "strategy_name",
            ],
            as_index=False,
        )["daily_return"]
        .agg(lambda values: float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0))
        .rename(
            columns={
                "daily_return": "monthly_return",
            }
        )
    )


def _benchmark_monthly_state(
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Build market-direction and volatility states from SPY."""
    data = benchmark.loc[
        :,
        [
            "date",
            "daily_return",
        ],
    ].copy()

    data["month"] = data["date"].dt.to_period("M").dt.to_timestamp("M")

    monthly = data.groupby(
        "month",
        as_index=False,
    ).agg(
        spy_monthly_return=(
            "daily_return",
            lambda values: float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0),
        ),
        spy_realized_volatility=(
            "daily_return",
            lambda values: (
                float(
                    np.std(
                        values.to_numpy(dtype=float),
                        ddof=1,
                    )
                    * np.sqrt(252.0)
                )
                if len(values) > 1
                else 0.0
            ),
        ),
    )

    volatility_threshold = float(monthly["spy_realized_volatility"].quantile(0.75))

    monthly["market_direction"] = np.where(
        monthly["spy_monthly_return"].ge(0.0),
        "up_market_month",
        "down_market_month",
    )

    monthly["volatility_regime"] = np.where(
        monthly["spy_realized_volatility"].ge(volatility_threshold),
        "high_volatility_month",
        "normal_volatility_month",
    )

    return monthly


def _summarize_condition(
    merged: pd.DataFrame,
    *,
    dimension: str,
    condition_column: str,
) -> pd.DataFrame:
    """Summarize strategy behavior inside one monthly market condition."""
    rows = []

    for (
        strategy_name,
        condition,
    ), group in merged.groupby(
        [
            "strategy_name",
            condition_column,
        ],
        sort=True,
    ):
        excess = group["monthly_return"] - group["spy_monthly_return"]

        rows.append(
            {
                "dimension": dimension,
                "condition": str(condition),
                "strategy_name": str(strategy_name),
                "months": int(len(group)),
                "mean_monthly_return": float(group["monthly_return"].mean()),
                "median_monthly_return": float(group["monthly_return"].median()),
                "mean_spy_monthly_return": float(group["spy_monthly_return"].mean()),
                "mean_excess_monthly_return": float(excess.mean()),
                "positive_month_ratio": float(group["monthly_return"].gt(0.0).mean()),
                "outperform_spy_ratio": float(excess.gt(0.0).mean()),
            }
        )

    return pd.DataFrame(rows)


def _build_conditional_months(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate up/down and high/normal-volatility market months."""
    strategy_monthly = _monthly_returns(daily)

    market_state = _benchmark_monthly_state(benchmark)

    merged = strategy_monthly.merge(
        market_state,
        on="month",
        how="inner",
        validate="many_to_one",
    )

    market_direction = _summarize_condition(
        merged,
        dimension="market_direction",
        condition_column="market_direction",
    )

    volatility = _summarize_condition(
        merged,
        dimension="volatility_regime",
        condition_column="volatility_regime",
    )

    return (
        pd.concat(
            [
                market_direction,
                volatility,
            ],
            ignore_index=True,
        )
        .sort_values(
            [
                "dimension",
                "condition",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )


def _build_checks(
    yearly: pd.DataFrame,
    regimes: pd.DataFrame,
    conditional: pd.DataFrame,
) -> pd.DataFrame:
    """Audit temporal robustness outputs."""
    key_yearly = yearly[
        [
            "total_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
        ]
    ].to_numpy(dtype=float)

    key_regimes = regimes[
        [
            "total_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
        ]
    ].to_numpy(dtype=float)

    yearly_method_counts = yearly.groupby("year")["strategy_name"].nunique()

    regime_method_counts = regimes.groupby("regime")["strategy_name"].nunique()

    conditional_method_counts = conditional.groupby(
        [
            "dimension",
            "condition",
        ]
    )["strategy_name"].nunique()

    checks = [
        (
            "five_methods_per_year",
            int((yearly_method_counts != len(EXPECTED_METHODS)).sum()),
            "Every observed calendar year must contain all five methods.",
        ),
        (
            "five_methods_per_regime",
            int((regime_method_counts != len(EXPECTED_METHODS)).sum()),
            "Every temporal regime must contain all five methods.",
        ),
        (
            "five_methods_per_condition",
            int((conditional_method_counts != len(EXPECTED_METHODS)).sum()),
            "Every monthly market condition must contain all five methods.",
        ),
        (
            "finite_yearly_metrics",
            int((~np.isfinite(key_yearly)).sum()),
            "Key yearly performance metrics must be finite.",
        ),
        (
            "finite_regime_metrics",
            int((~np.isfinite(key_regimes)).sum()),
            "Key regime performance metrics must be finite.",
        ),
        (
            "positive_year_month_counts",
            int(conditional["months"].le(0).sum()),
            "Conditional analyses must contain at least one month.",
        ),
        (
            "valid_positive_month_ratios",
            int(
                (
                    ~conditional["positive_month_ratio"].between(
                        0.0,
                        1.0,
                    )
                ).sum()
            ),
            "Positive-month ratios must lie between zero and one.",
        ),
        (
            "valid_outperformance_ratios",
            int(
                (
                    ~conditional["outperform_spy_ratio"].between(
                        0.0,
                        1.0,
                    )
                ).sum()
            ),
            "SPY outperformance ratios must lie between zero and one.",
        ),
    ]

    return pd.DataFrame(
        [
            {
                "check": name,
                "status": ("PASS" if violations == 0 else "FAIL"),
                "violations": violations,
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
    yearly: pd.DataFrame,
    regimes: pd.DataFrame,
    conditional: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    """Build temporal robustness report."""
    yearly_display = yearly.loc[
        :,
        [
            "year",
            "strategy_name",
            "trading_days",
            "partial_calendar_year",
            "total_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "excess_total_return",
        ],
    ]

    regime_display = regimes.loc[
        :,
        [
            "regime",
            "strategy_name",
            "trading_days",
            "short_sample_warning",
            "total_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "excess_total_return",
        ],
    ]

    return "\n".join(
        [
            "# Temporal Robustness Analysis",
            "",
            "## Interpretation",
            "",
            (
                "This analysis does not change model parameters. "
                "It evaluates the same net out-of-sample portfolio "
                "returns across calendar years and market regimes."
            ),
            (
                "The available out-of-sample backtest begins in "
                "February 2020, so the pre-COVID sample is too short "
                "for a strong conclusion and is explicitly flagged."
            ),
            (
                "Up/down market conditions are defined from SPY monthly "
                "returns. High-volatility months are the top quartile of "
                "SPY monthly realized volatility within the available "
                "out-of-sample sample."
            ),
            "",
            "## Calendar-year results",
            "",
            _to_markdown(yearly_display),
            "",
            "## Regime results",
            "",
            _to_markdown(regime_display),
            "",
            "## Conditional monthly behavior",
            "",
            _to_markdown(conditional),
            "",
            "## Readiness checks",
            "",
            _to_markdown(checks),
            "",
        ]
    )


def main() -> None:
    """Run temporal robustness diagnostics."""
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

    logger.info("Evaluating calendar-year robustness.")

    yearly = _build_yearly_results(
        daily,
        benchmark,
        initial_capital=(backtest_config.initial_capital),
        evaluation_config=(evaluation_config),
    )

    logger.info("Evaluating temporal market regimes.")

    regimes = _build_regime_results(
        daily,
        benchmark,
        initial_capital=(backtest_config.initial_capital),
        evaluation_config=(evaluation_config),
    )

    logger.info("Evaluating conditional monthly robustness.")

    conditional = _build_conditional_months(
        daily,
        benchmark,
    )

    checks = _build_checks(
        yearly,
        regimes,
        conditional,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_csv(
        yearly,
        YEARLY_PATH,
    )

    _write_csv(
        regimes,
        REGIME_PATH,
    )

    _write_csv(
        conditional,
        CONDITIONAL_MONTHS_PATH,
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
            yearly,
            regimes,
            conditional,
            checks,
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(
            f"Temporal robustness validation failed with {failed_checks} failed checks."
        )

    logger.info("Temporal robustness analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Temporal robustness analysis")

    print("------------------------------------------------")

    print(f"methods: {daily['strategy_name'].nunique()}")

    print(f"years: {yearly['year'].nunique()}")

    print(f"regimes: {regimes['regime'].nunique()}")

    print()

    print("Yearly performance:")

    print(
        yearly.loc[
            :,
            [
                "year",
                "strategy_name",
                "total_return",
                "sharpe_ratio",
                "maximum_drawdown",
                "excess_total_return",
            ],
        ].to_string(index=False)
    )

    print()

    print("Regime performance:")

    print(
        regimes.loc[
            :,
            [
                "regime",
                "strategy_name",
                "trading_days",
                "total_return",
                "sharpe_ratio",
                "maximum_drawdown",
                "excess_total_return",
                "short_sample_warning",
            ],
        ].to_string(index=False)
    )

    print()

    print("Conditional monthly behavior:")

    print(conditional.to_string(index=False))

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Yearly table: {YEARLY_PATH}")

    print(f"Regime table: {REGIME_PATH}")

    print(f"Conditional table: {CONDITIONAL_MONTHS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
