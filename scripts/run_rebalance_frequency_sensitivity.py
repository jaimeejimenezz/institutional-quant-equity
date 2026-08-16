"""Compare monthly and quarterly portfolio rebalancing."""

from __future__ import annotations

import logging
from dataclasses import replace
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
    run_mvp_backtest,
)
from quant_equity.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_config,
)
from quant_equity.logging_config import configure_logging

ALL_WEIGHTS_PATH = PROCESSED_DATA_DIR / "target_weights_all_methods.parquet"

MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

RISK_ESTIMATES_PATH = PROCESSED_DATA_DIR / "risk_estimates.parquet"

SPY_DATA_PATH = PROCESSED_DATA_DIR / "benchmark_spy_daily.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

RESULTS_PATH = TABLES_DIR / "robustness_rebalance_frequency.csv"

CHECKS_PATH = TABLES_DIR / "robustness_rebalance_frequency_checks.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "rebalance_frequency_sensitivity.md"

EXPECTED_METHODS = {
    "alpha_risk_turnover",
    "cvar",
    "median_mad_de",
    "score_weighted",
    "top_n_equal_weight",
}

QUARTER_END_MONTHS = {
    3,
    6,
    9,
    12,
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


def _prepare_targets(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Convert combined construction weights to backtest targets."""
    required_columns = {
        "as_of_date",
        "ticker",
        "sector",
        "method",
        "weight",
    }

    missing = sorted(required_columns.difference(weights.columns))

    if missing:
        raise ValueError("Combined target weights are missing columns: " + ", ".join(missing) + ".")

    targets = (
        weights.loc[
            :,
            [
                "as_of_date",
                "method",
                "ticker",
                "sector",
                "weight",
            ],
        ]
        .rename(
            columns={
                "method": "strategy_name",
                "weight": "target_weight",
            }
        )
        .copy()
    )

    targets["as_of_date"] = pd.to_datetime(
        targets["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    for column in (
        "strategy_name",
        "ticker",
        "sector",
    ):
        targets[column] = targets[column].astype("string").str.strip()

    targets["target_weight"] = pd.to_numeric(
        targets["target_weight"],
        errors="coerce",
    )

    if (
        targets["as_of_date"].isna().any()
        or targets[
            [
                "strategy_name",
                "ticker",
                "sector",
            ]
        ]
        .isna()
        .any()
        .any()
        or targets["target_weight"].isna().any()
    ):
        raise ValueError("Combined target weights contain invalid values.")

    methods = set(targets["strategy_name"].unique())

    if methods != EXPECTED_METHODS:
        raise ValueError(
            "Unexpected portfolio methods. "
            f"Expected {sorted(EXPECTED_METHODS)}, "
            f"received {sorted(methods)}."
        )

    return targets.sort_values(
        [
            "as_of_date",
            "strategy_name",
            "ticker",
        ]
    ).reset_index(drop=True)


def _quarterly_targets(
    monthly_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Keep calendar quarter-end signal months."""
    quarterly = monthly_targets.loc[
        monthly_targets["as_of_date"].dt.month.isin(QUARTER_END_MONTHS)
    ].copy()

    if quarterly.empty:
        raise ValueError("No calendar quarter-end target portfolios were found.")

    return quarterly.sort_values(
        [
            "as_of_date",
            "strategy_name",
            "ticker",
        ]
    ).reset_index(drop=True)


def _rebase_daily(
    daily: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
) -> pd.DataFrame:
    """Slice and rebase daily portfolio returns to a shared window."""
    blocks = []

    for strategy_name, group in daily.groupby(
        "strategy_name",
        sort=True,
    ):
        group = (
            group.loc[
                group["date"].between(
                    start_date,
                    end_date,
                )
            ]
            .sort_values("date")
            .copy()
        )

        if group.empty:
            raise ValueError(
                f"No daily observations for {strategy_name} inside the common comparison window."
            )

        returns = pd.to_numeric(
            group["daily_return"],
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(returns).all():
            raise ValueError("Daily returns contain non-finite values.")

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
    """Rebase benchmark values to the comparison capital."""
    data = benchmark.sort_values("date").copy()

    returns = pd.to_numeric(
        data["daily_return"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(returns).all():
        raise ValueError("Benchmark returns contain non-finite values.")

    data["portfolio_value"] = float(initial_capital) * np.cumprod(1.0 + returns)

    return data.reset_index(drop=True)


def _strategy_performance(
    summary: pd.DataFrame,
    *,
    benchmark_name: str,
) -> pd.DataFrame:
    """Return only portfolio-strategy rows."""
    return (
        summary.loc[~summary["strategy_name"].eq(benchmark_name)]
        .sort_values("strategy_name")
        .reset_index(drop=True)
    )


def _execution_metrics(
    rebalance_summary: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Summarize execution inside the shared comparison window."""
    data = rebalance_summary.loc[
        rebalance_summary["execution_date"].between(
            start_date,
            end_date,
        )
    ].copy()

    rows = []

    for strategy_name, group in data.groupby(
        "strategy_name",
        sort=True,
    ):
        total_cost = float(group["transaction_cost"].sum())

        total_notional = float(group["traded_notional"].sum())

        rows.append(
            {
                "strategy_name": str(strategy_name),
                "rebalances": int(len(group)),
                "total_transaction_cost": (total_cost),
                "total_traded_notional": (total_notional),
                "mean_one_way_turnover": float(group["one_way_turnover"].mean()),
                "mean_two_way_turnover": float(group["two_way_turnover"].mean()),
                "effective_cost_bps": (
                    total_cost / total_notional * 10_000.0 if total_notional > 0.0 else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def _scenario_results(
    performance: pd.DataFrame,
    execution: pd.DataFrame,
    *,
    frequency: str,
) -> pd.DataFrame:
    """Combine performance and execution for one frequency."""
    columns = [
        "strategy_name",
        "final_portfolio_value",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "beta_vs_spy",
        "annualized_alpha_vs_spy",
        "excess_cagr",
    ]

    result = performance.loc[
        :,
        columns,
    ].merge(
        execution,
        on="strategy_name",
        how="inner",
        validate="one_to_one",
    )

    result.insert(
        0,
        "rebalance_frequency",
        frequency,
    )

    return result


def _build_comparison(
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
) -> pd.DataFrame:
    """Build monthly-versus-quarterly differences by method."""
    monthly_view = monthly.drop(
        columns=[
            "rebalance_frequency",
        ]
    ).rename(
        columns={
            column: (column if column == "strategy_name" else f"monthly_{column}")
            for column in monthly.columns
            if column != "rebalance_frequency"
        }
    )

    quarterly_view = quarterly.drop(
        columns=[
            "rebalance_frequency",
        ]
    ).rename(
        columns={
            column: (column if column == "strategy_name" else f"quarterly_{column}")
            for column in quarterly.columns
            if column != "rebalance_frequency"
        }
    )

    comparison = monthly_view.merge(
        quarterly_view,
        on="strategy_name",
        how="inner",
        validate="one_to_one",
    )

    comparison["quarterly_minus_monthly_cagr"] = (
        comparison["quarterly_cagr"] - comparison["monthly_cagr"]
    )

    comparison["quarterly_minus_monthly_sharpe"] = (
        comparison["quarterly_sharpe_ratio"] - comparison["monthly_sharpe_ratio"]
    )

    comparison["quarterly_minus_monthly_drawdown"] = (
        comparison["quarterly_maximum_drawdown"] - comparison["monthly_maximum_drawdown"]
    )

    comparison["transaction_cost_reduction_fraction"] = np.where(
        comparison["monthly_total_transaction_cost"].gt(0.0),
        (
            1.0
            - comparison["quarterly_total_transaction_cost"]
            / comparison["monthly_total_transaction_cost"]
        ),
        0.0,
    )

    return comparison.sort_values("strategy_name").reset_index(drop=True)


def _build_checks(
    monthly_targets: pd.DataFrame,
    quarterly_targets: pd.DataFrame,
    monthly_daily: pd.DataFrame,
    quarterly_daily: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Audit the rebalance-frequency sensitivity analysis."""
    monthly_methods = set(monthly_targets["strategy_name"].unique())

    quarterly_methods = set(quarterly_targets["strategy_name"].unique())

    monthly_dates = monthly_daily.groupby("strategy_name")["date"].agg(
        [
            "min",
            "max",
            "count",
        ]
    )

    quarterly_dates = quarterly_daily.groupby("strategy_name")["date"].agg(
        [
            "min",
            "max",
            "count",
        ]
    )

    same_window = monthly_dates[
        [
            "min",
            "max",
            "count",
        ]
    ].equals(
        quarterly_dates[
            [
                "min",
                "max",
                "count",
            ]
        ]
    )

    weight_sums_monthly = monthly_targets.groupby(
        [
            "strategy_name",
            "as_of_date",
        ]
    )["target_weight"].sum()

    weight_sums_quarterly = quarterly_targets.groupby(
        [
            "strategy_name",
            "as_of_date",
        ]
    )["target_weight"].sum()

    quarterly_signal_months = set(quarterly_targets["as_of_date"].dt.month.unique())

    finite_columns = [
        "monthly_cagr",
        "quarterly_cagr",
        "monthly_sharpe_ratio",
        "quarterly_sharpe_ratio",
        "monthly_maximum_drawdown",
        "quarterly_maximum_drawdown",
        "monthly_effective_cost_bps",
        "quarterly_effective_cost_bps",
    ]

    checks = [
        (
            "expected_methods",
            int(monthly_methods != EXPECTED_METHODS or quarterly_methods != EXPECTED_METHODS),
            "Monthly and quarterly scenarios must contain all five portfolio methods.",
        ),
        (
            "calendar_quarter_signals",
            int(not quarterly_signal_months.issubset(QUARTER_END_MONTHS)),
            "Quarterly portfolios must use March, June, September and December signals only.",
        ),
        (
            "quarterly_has_fewer_signals",
            int(
                quarterly_targets["as_of_date"].nunique() >= monthly_targets["as_of_date"].nunique()
            ),
            "Quarterly rebalancing must use fewer signal dates than monthly rebalancing.",
        ),
        (
            "fully_invested_monthly",
            int(weight_sums_monthly.sub(1.0).abs().gt(1.0e-6).sum()),
            "Monthly target portfolios must sum to one.",
        ),
        (
            "fully_invested_quarterly",
            int(weight_sums_quarterly.sub(1.0).abs().gt(1.0e-6).sum()),
            "Quarterly target portfolios must sum to one.",
        ),
        (
            "identical_comparison_window",
            int(not same_window),
            "Monthly and quarterly performance must be compared on identical daily dates.",
        ),
        (
            "comparison_rows",
            int(len(comparison) != len(EXPECTED_METHODS)),
            "The final comparison must contain one row per portfolio method.",
        ),
        (
            "finite_comparison_metrics",
            int((~np.isfinite(comparison[finite_columns].to_numpy(dtype=float))).sum()),
            "Key rebalance-frequency metrics must remain finite.",
        ),
        (
            "quarterly_fewer_rebalances",
            int(comparison["quarterly_rebalances"].ge(comparison["monthly_rebalances"]).sum()),
            "Quarterly scenarios must execute fewer rebalances than monthly scenarios.",
        ),
        (
            "positive_final_values",
            int(
                comparison[
                    [
                        "monthly_final_portfolio_value",
                        "quarterly_final_portfolio_value",
                    ]
                ]
                .le(0.0)
                .sum()
                .sum()
            ),
            "Monthly and quarterly strategies must retain positive final value.",
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
    comparison: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    monthly_signal_dates: int,
    quarterly_signal_dates: int,
    common_start: pd.Timestamp,
    common_end: pd.Timestamp,
) -> str:
    """Build the rebalance-frequency sensitivity report."""
    display = comparison.loc[
        :,
        [
            "strategy_name",
            "monthly_cagr",
            "quarterly_cagr",
            "quarterly_minus_monthly_cagr",
            "monthly_sharpe_ratio",
            "quarterly_sharpe_ratio",
            "quarterly_minus_monthly_sharpe",
            "monthly_maximum_drawdown",
            "quarterly_maximum_drawdown",
            "monthly_rebalances",
            "quarterly_rebalances",
            "monthly_total_transaction_cost",
            "quarterly_total_transaction_cost",
            "transaction_cost_reduction_fraction",
        ],
    ]

    return "\n".join(
        [
            "# Rebalance Frequency Sensitivity",
            "",
            "## Methodology",
            "",
            (f"- Monthly signal dates available: `{monthly_signal_dates}`."),
            (f"- Calendar quarter-end signal dates used: `{quarterly_signal_dates}`."),
            (
                "- Quarterly signals are the March, June, September and "
                "December month-end portfolios from the frozen OOS weights."
            ),
            (
                f"- Both frequencies are evaluated on the identical daily "
                f"window `{common_start.date()}` to `{common_end.date()}`."
            ),
            (
                "- Portfolio construction, final alpha signal, risk inputs "
                "and advanced execution-cost assumptions are unchanged."
            ),
            (
                "- This is a sensitivity analysis only; the frozen production "
                "baseline is not changed based on these OOS results."
            ),
            "",
            "## Results",
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
    """Run monthly-versus-quarterly rebalancing sensitivity."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    required_paths = (
        ALL_WEIGHTS_PATH,
        MARKET_DATA_PATH,
        RISK_ESTIMATES_PATH,
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

    all_weights = pd.read_parquet(ALL_WEIGHTS_PATH)

    market_data = pd.read_parquet(MARKET_DATA_PATH)

    risk_estimates = pd.read_parquet(RISK_ESTIMATES_PATH)

    spy_data = pd.read_parquet(SPY_DATA_PATH)

    monthly_targets = _prepare_targets(all_weights)

    quarterly_targets = _quarterly_targets(monthly_targets)

    net_config = replace(
        backtest_config,
        transaction_cost_bps=0.0,
    )

    logger.info("Running monthly rebalance reference backtest.")

    monthly_outputs = run_mvp_backtest(
        monthly_targets,
        market_data,
        config=net_config,
        risk_estimates=risk_estimates,
        execution_cost_config=execution_config,
    )

    logger.info("Running calendar-quarter rebalance backtest.")

    quarterly_outputs = run_mvp_backtest(
        quarterly_targets,
        market_data,
        config=net_config,
        risk_estimates=risk_estimates,
        execution_cost_config=execution_config,
    )

    common_start = max(
        pd.to_datetime(monthly_outputs.daily_performance["date"]).min(),
        pd.to_datetime(quarterly_outputs.daily_performance["date"]).min(),
    )

    common_end = min(
        pd.to_datetime(monthly_outputs.daily_performance["date"]).max(),
        pd.to_datetime(quarterly_outputs.daily_performance["date"]).max(),
    )

    if common_start >= common_end:
        raise ValueError("Monthly and quarterly backtests do not share a valid comparison window.")

    monthly_daily = _rebase_daily(
        monthly_outputs.daily_performance,
        start_date=common_start,
        end_date=common_end,
        initial_capital=(net_config.initial_capital),
    )

    quarterly_daily = _rebase_daily(
        quarterly_outputs.daily_performance,
        start_date=common_start,
        end_date=common_end,
        initial_capital=(net_config.initial_capital),
    )

    benchmark = build_buy_and_hold_benchmark(
        spy_data,
        strategy_name=(evaluation_config.benchmark_name),
        ticker=(evaluation_config.benchmark_ticker),
        start_date=common_start,
        end_date=common_end,
        initial_capital=(net_config.initial_capital),
        transaction_cost_bps=(execution_config.linear_cost_bps),
    )

    benchmark = _rebase_benchmark(
        benchmark,
        initial_capital=(net_config.initial_capital),
    )

    monthly_evaluation = evaluate_performance(
        monthly_daily,
        benchmark,
        initial_capital=(net_config.initial_capital),
        config=evaluation_config,
    )

    quarterly_evaluation = evaluate_performance(
        quarterly_daily,
        benchmark,
        initial_capital=(net_config.initial_capital),
        config=evaluation_config,
    )

    monthly_performance = _strategy_performance(
        monthly_evaluation.performance_summary,
        benchmark_name=(evaluation_config.benchmark_name),
    )

    quarterly_performance = _strategy_performance(
        quarterly_evaluation.performance_summary,
        benchmark_name=(evaluation_config.benchmark_name),
    )

    monthly_execution = _execution_metrics(
        monthly_outputs.rebalance_summary,
        start_date=common_start,
        end_date=common_end,
    )

    quarterly_execution = _execution_metrics(
        quarterly_outputs.rebalance_summary,
        start_date=common_start,
        end_date=common_end,
    )

    monthly_results = _scenario_results(
        monthly_performance,
        monthly_execution,
        frequency="monthly",
    )

    quarterly_results = _scenario_results(
        quarterly_performance,
        quarterly_execution,
        frequency="quarterly",
    )

    comparison = _build_comparison(
        monthly_results,
        quarterly_results,
    )

    checks = _build_checks(
        monthly_targets,
        quarterly_targets,
        monthly_daily,
        quarterly_daily,
        comparison,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_csv(
        comparison,
        RESULTS_PATH,
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
            comparison,
            checks,
            monthly_signal_dates=(monthly_targets["as_of_date"].nunique()),
            quarterly_signal_dates=(quarterly_targets["as_of_date"].nunique()),
            common_start=common_start,
            common_end=common_end,
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(
            f"Rebalance-frequency validation failed with {failed_checks} failed checks."
        )

    logger.info("Rebalance-frequency sensitivity analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Rebalance frequency sensitivity")

    print("------------------------------------------------")

    print(f"methods: {len(EXPECTED_METHODS)}")

    print(f"monthly_signal_dates: {monthly_targets['as_of_date'].nunique()}")

    print(f"quarterly_signal_dates: {quarterly_targets['as_of_date'].nunique()}")

    print(f"common_start: {common_start.date()}")

    print(f"common_end: {common_end.date()}")

    print()

    print("Monthly versus quarterly:")

    print(
        comparison.loc[
            :,
            [
                "strategy_name",
                "monthly_cagr",
                "quarterly_cagr",
                "quarterly_minus_monthly_cagr",
                "monthly_sharpe_ratio",
                "quarterly_sharpe_ratio",
                "quarterly_minus_monthly_sharpe",
                "monthly_maximum_drawdown",
                "quarterly_maximum_drawdown",
                "monthly_rebalances",
                "quarterly_rebalances",
                "monthly_total_transaction_cost",
                "quarterly_total_transaction_cost",
                "transaction_cost_reduction_fraction",
            ],
        ].to_string(index=False)
    )

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Results table: {RESULTS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
