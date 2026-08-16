"""Evaluate portfolio capacity across increasing capital levels."""

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
    estimate_trade_execution_costs,
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

CAPACITY_ANALYSIS_PATH = TABLES_DIR / "capacity_analysis.csv"

CAPACITY_COST_COMPONENTS_PATH = TABLES_DIR / "capacity_cost_components.csv"

CAPACITY_CHECKS_PATH = TABLES_DIR / "capacity_checks.csv"

REPORT_PATH = REPORTS_DIR / "backtests" / "capacity_analysis.md"

CAPITAL_LEVELS = (
    100_000.0,
    1_000_000.0,
    10_000_000.0,
    100_000_000.0,
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
    """Write a CSV table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=False,
    )


def _prepare_backtest_targets(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Convert portfolio-construction output to backtest targets."""
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

    targets["as_of_date"] = pd.to_datetime(targets["as_of_date"]).dt.normalize()

    for column in (
        "strategy_name",
        "ticker",
        "sector",
    ):
        targets[column] = targets[column].astype(str)

    targets["target_weight"] = pd.to_numeric(
        targets["target_weight"],
        errors="raise",
    )

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


def _strategy_performance(
    summary: pd.DataFrame,
    *,
    benchmark_name: str,
) -> pd.DataFrame:
    """Return strategy rows without the benchmark."""
    return (
        summary.loc[~summary["strategy_name"].eq(benchmark_name)]
        .sort_values("strategy_name")
        .reset_index(drop=True)
    )


def _cost_summary(
    enriched_trades: pd.DataFrame,
    *,
    capital: float,
) -> pd.DataFrame:
    """Summarize execution costs for one capital level."""
    summary = enriched_trades.groupby(
        "strategy_name",
        as_index=False,
    ).agg(
        trades=(
            "ticker",
            "size",
        ),
        traded_notional=(
            "absolute_trade_notional",
            "sum",
        ),
        commission_cost=(
            "commission_cost",
            "sum",
        ),
        spread_cost=(
            "spread_cost",
            "sum",
        ),
        slippage_cost=(
            "slippage_cost",
            "sum",
        ),
        market_impact_cost=(
            "market_impact_cost",
            "sum",
        ),
        total_execution_cost=(
            "total_execution_cost",
            "sum",
        ),
        maximum_order_adv_fraction=(
            "order_adv_fraction",
            "max",
        ),
        mean_order_adv_fraction=(
            "order_adv_fraction",
            "mean",
        ),
    )

    summary["capital"] = float(capital)

    summary["effective_cost_bps"] = np.where(
        summary["traded_notional"].gt(0.0),
        (summary["total_execution_cost"] / summary["traded_notional"] * 10_000.0),
        0.0,
    )

    summary["transaction_cost_fraction_initial_capital"] = summary["total_execution_cost"] / float(
        capital
    )

    return summary


def _build_capacity_rows(
    gross_performance: pd.DataFrame,
    net_performance: pd.DataFrame,
    execution_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    *,
    capital: float,
) -> pd.DataFrame:
    """Build the capacity table for one capital level."""
    gross = gross_performance.loc[
        :,
        [
            "strategy_name",
            "cagr",
            "sharpe_ratio",
            "sortino_ratio",
            "maximum_drawdown",
        ],
    ].rename(
        columns={
            "cagr": "gross_cagr",
            "sharpe_ratio": "gross_sharpe_ratio",
            "sortino_ratio": "gross_sortino_ratio",
            "maximum_drawdown": "gross_maximum_drawdown",
        }
    )

    net = net_performance.loc[
        :,
        [
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
        ],
    ].rename(
        columns={
            "final_portfolio_value": "net_final_portfolio_value",
            "total_return": "net_total_return",
            "cagr": "net_cagr",
            "annualized_volatility": "net_annualized_volatility",
            "sharpe_ratio": "net_sharpe_ratio",
            "sortino_ratio": "net_sortino_ratio",
            "maximum_drawdown": "net_maximum_drawdown",
            "beta_vs_spy": "net_beta_vs_spy",
            "annualized_alpha_vs_spy": "net_annualized_alpha_vs_spy",
            "excess_cagr": "net_excess_cagr",
        }
    )

    execution = execution_summary.loc[
        :,
        [
            "strategy_name",
            "total_transaction_cost",
            "total_traded_notional",
            "mean_two_way_turnover",
            "mean_one_way_turnover",
            "maximum_absolute_cash",
        ],
    ]

    costs = cost_summary.loc[
        :,
        [
            "strategy_name",
            "market_impact_cost",
            "effective_cost_bps",
            "maximum_order_adv_fraction",
            "mean_order_adv_fraction",
            "transaction_cost_fraction_initial_capital",
        ],
    ]

    result = (
        gross.merge(
            net,
            on="strategy_name",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            execution,
            on="strategy_name",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            costs,
            on="strategy_name",
            how="inner",
            validate="one_to_one",
        )
    )

    result.insert(
        0,
        "capital",
        float(capital),
    )

    result["cagr_cost_drag"] = result["gross_cagr"] - result["net_cagr"]

    result["sharpe_cost_drag"] = result["gross_sharpe_ratio"] - result["net_sharpe_ratio"]

    return result


def _build_checks(
    capacity: pd.DataFrame,
    *,
    cash_tolerance: float,
    weight_tolerance: float,
) -> pd.DataFrame:
    """Audit capacity-analysis outputs."""
    expected_rows = len(CAPITAL_LEVELS) * len(EXPECTED_METHODS)

    rows_per_capital = capacity.groupby("capital")["strategy_name"].nunique()

    ordered_costs = (
        capacity.sort_values(
            [
                "strategy_name",
                "capital",
            ]
        )
        .groupby("strategy_name")["effective_cost_bps"]
        .apply(lambda values: bool(np.all(np.diff(values.to_numpy(dtype=float)) >= -1.0e-10)))
    )

    ordered_participation = (
        capacity.sort_values(
            [
                "strategy_name",
                "capital",
            ]
        )
        .groupby("strategy_name")["maximum_order_adv_fraction"]
        .apply(lambda values: bool(np.all(np.diff(values.to_numpy(dtype=float)) >= -1.0e-10)))
    )

    checks = [
        (
            "expected_rows",
            int(len(capacity) != expected_rows),
            "Capacity analysis must contain one row per method and capital level.",
        ),
        (
            "five_methods_per_capital",
            int((rows_per_capital != len(EXPECTED_METHODS)).sum()),
            "Every capital level must contain all five portfolio methods.",
        ),
        (
            "positive_final_values",
            int(capacity["net_final_portfolio_value"].le(0.0).sum()),
            "All capacity scenarios must retain positive portfolio value.",
        ),
        (
            "finite_capacity_metrics",
            int(
                (
                    ~np.isfinite(
                        capacity[
                            [
                                "net_cagr",
                                "net_sharpe_ratio",
                                "effective_cost_bps",
                                "maximum_order_adv_fraction",
                            ]
                        ].to_numpy(dtype=float)
                    )
                ).sum()
            ),
            "Key capacity metrics must remain finite.",
        ),
        (
            "nonnegative_costs",
            int(capacity["total_transaction_cost"].lt(0.0).sum()),
            "Transaction costs cannot be negative.",
        ),
        (
            "cash_accounting",
            int(
                capacity["maximum_absolute_cash"]
                .gt(
                    np.maximum(
                        cash_tolerance,
                        capacity["capital"].to_numpy(dtype=float) * weight_tolerance,
                    )
                    + 1.0e-6
                )
                .sum()
            ),
            (
                "Residual cash must remain within the same absolute-or-relative "
                "accounting tolerance used by the backtest engine."
            ),
        ),
        (
            "cost_bps_non_decreasing",
            int((~ordered_costs).sum()),
            "Effective execution cost should not decline as capital increases.",
        ),
        (
            "adv_participation_non_decreasing",
            int((~ordered_participation).sum()),
            "Maximum order-to-ADV participation should not decline as capital increases.",
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
    """Convert a dataframe to Markdown without optional dependencies."""
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
    capacity: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    execution_config: ExecutionCostConfig,
) -> str:
    """Build the capacity-analysis Markdown report."""
    display = capacity.loc[
        :,
        [
            "capital",
            "strategy_name",
            "gross_cagr",
            "net_cagr",
            "cagr_cost_drag",
            "net_sharpe_ratio",
            "net_maximum_drawdown",
            "net_excess_cagr",
            "effective_cost_bps",
            "market_impact_cost",
            "maximum_order_adv_fraction",
        ],
    ].copy()

    return "\n".join(
        [
            "# Capacity Analysis",
            "",
            "## Execution assumptions",
            "",
            (f"- Linear execution cost: `{execution_config.linear_cost_bps:.2f}` bps."),
            (
                "- Market impact: coefficient × daily volatility × "
                "square root of order notional divided by ADV."
            ),
            ("- Capital levels: $100k, $1M, $10M and $100M."),
            (
                "- Portfolio weights, rebalance dates and signals are held "
                "constant across capital scenarios."
            ),
            "",
            "## Capacity results",
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
    """Run capacity analysis for all portfolio methods."""
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

    base_backtest_config = MVPBacktestConfig.from_mapping(
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

    combined_weights = pd.read_parquet(ALL_WEIGHTS_PATH)

    market_data = pd.read_parquet(MARKET_DATA_PATH)

    risk_estimates = pd.read_parquet(RISK_ESTIMATES_PATH)

    spy_data = pd.read_parquet(SPY_DATA_PATH)

    targets = _prepare_backtest_targets(combined_weights)

    gross_config = replace(
        base_backtest_config,
        initial_capital=1_000_000.0,
        transaction_cost_bps=0.0,
    )

    logger.info("Running gross reference backtest.")

    gross_outputs = run_mvp_backtest(
        targets,
        market_data,
        config=gross_config,
    )

    start_date = pd.to_datetime(gross_outputs.daily_performance["date"]).min()

    end_date = pd.to_datetime(gross_outputs.daily_performance["date"]).max()

    gross_spy = build_buy_and_hold_benchmark(
        spy_data,
        strategy_name=(evaluation_config.benchmark_name),
        ticker=(evaluation_config.benchmark_ticker),
        start_date=start_date,
        end_date=end_date,
        initial_capital=(gross_config.initial_capital),
        transaction_cost_bps=0.0,
    )

    gross_evaluation = evaluate_performance(
        gross_outputs.daily_performance,
        gross_spy,
        initial_capital=(gross_config.initial_capital),
        config=evaluation_config,
    )

    gross_performance = _strategy_performance(
        gross_evaluation.performance_summary,
        benchmark_name=(evaluation_config.benchmark_name),
    )

    capacity_blocks = []
    cost_blocks = []

    for capital in CAPITAL_LEVELS:
        logger.info(
            "Running capacity backtest for $%s.",
            f"{capital:,.0f}",
        )

        net_config = replace(
            base_backtest_config,
            initial_capital=float(capital),
            transaction_cost_bps=0.0,
        )

        net_outputs = run_mvp_backtest(
            targets,
            market_data,
            config=net_config,
            risk_estimates=risk_estimates,
            execution_cost_config=(execution_config),
        )

        net_spy = build_buy_and_hold_benchmark(
            spy_data,
            strategy_name=(evaluation_config.benchmark_name),
            ticker=(evaluation_config.benchmark_ticker),
            start_date=start_date,
            end_date=end_date,
            initial_capital=float(capital),
            transaction_cost_bps=(execution_config.linear_cost_bps),
        )

        net_evaluation = evaluate_performance(
            net_outputs.daily_performance,
            net_spy,
            initial_capital=float(capital),
            config=evaluation_config,
        )

        net_performance = _strategy_performance(
            net_evaluation.performance_summary,
            benchmark_name=(evaluation_config.benchmark_name),
        )

        enriched_trades = estimate_trade_execution_costs(
            net_outputs.trades,
            risk_estimates,
            config=execution_config,
        )

        costs = _cost_summary(
            enriched_trades,
            capital=float(capital),
        )

        capacity = _build_capacity_rows(
            gross_performance,
            net_performance,
            net_outputs.execution_summary,
            costs,
            capital=float(capital),
        )

        cost_blocks.append(costs)

        capacity_blocks.append(capacity)

    capacity_analysis = (
        pd.concat(
            capacity_blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "capital",
                "net_sharpe_ratio",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    cost_components = (
        pd.concat(
            cost_blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "capital",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )

    checks = _build_checks(
        capacity_analysis,
        cash_tolerance=(base_backtest_config.cash_tolerance),
        weight_tolerance=(base_backtest_config.weight_tolerance),
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_csv(
        capacity_analysis,
        CAPACITY_ANALYSIS_PATH,
    )

    _write_csv(
        cost_components,
        CAPACITY_COST_COMPONENTS_PATH,
    )

    _write_csv(
        checks,
        CAPACITY_CHECKS_PATH,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        _build_report(
            capacity_analysis,
            checks,
            execution_config=execution_config,
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(f"Capacity validation failed with {failed_checks} failed checks.")

    logger.info("Capacity analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Portfolio capacity analysis")

    print("------------------------------------------------")

    print(f"methods: {capacity_analysis['strategy_name'].nunique()}")

    print(f"capital_levels: {capacity_analysis['capital'].nunique()}")

    print(f"minimum_capital: ${capacity_analysis['capital'].min():,.0f}")

    print(f"maximum_capital: ${capacity_analysis['capital'].max():,.0f}")

    print()

    print("Capacity comparison:")

    print(
        capacity_analysis.loc[
            :,
            [
                "capital",
                "strategy_name",
                "gross_cagr",
                "net_cagr",
                "cagr_cost_drag",
                "net_sharpe_ratio",
                "net_maximum_drawdown",
                "net_excess_cagr",
                "effective_cost_bps",
                "maximum_order_adv_fraction",
            ],
        ].to_string(index=False)
    )

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Capacity table: {CAPACITY_ANALYSIS_PATH}")

    print(f"Cost components: {CAPACITY_COST_COMPONENTS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
