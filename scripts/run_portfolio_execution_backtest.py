"""Compare all portfolio-construction methods with realistic execution costs."""

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

GROSS_DAILY_PATH = PROCESSED_DATA_DIR / "backtest_all_methods_gross_daily.parquet"

NET_DAILY_PATH = PROCESSED_DATA_DIR / "backtest_all_methods_net_daily.parquet"

NET_POSITIONS_PATH = PROCESSED_DATA_DIR / "positions_all_methods_net.parquet"

NET_TRADES_PATH = PROCESSED_DATA_DIR / "trades_all_methods_net.parquet"

NET_REBALANCES_PATH = PROCESSED_DATA_DIR / "rebalances_all_methods_net.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

GROSS_PERFORMANCE_PATH = TABLES_DIR / "all_methods_gross_performance.csv"

NET_PERFORMANCE_PATH = TABLES_DIR / "all_methods_net_performance.csv"

EXECUTION_SUMMARY_PATH = TABLES_DIR / "all_methods_execution_summary.csv"

COST_COMPONENTS_PATH = TABLES_DIR / "all_methods_execution_cost_components.csv"

COMPARISON_PATH = TABLES_DIR / "all_methods_gross_net_comparison.csv"

CHECKS_PATH = TABLES_DIR / "all_methods_execution_checks.csv"

REPORT_PATH = REPORTS_DIR / "backtests" / "portfolio_execution_comparison.md"

EXPECTED_METHODS = {
    "alpha_risk_turnover",
    "cvar",
    "median_mad_de",
    "score_weighted",
    "top_n_equal_weight",
}


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
) -> None:
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
    """Convert portfolio-construction output to backtest target weights."""
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

    targets["strategy_name"] = targets["strategy_name"].astype(str)

    targets["ticker"] = targets["ticker"].astype(str)

    targets["sector"] = targets["sector"].astype(str)

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

    date_counts = targets.groupby("strategy_name")["as_of_date"].nunique()

    if date_counts.nunique() != 1:
        raise ValueError("Portfolio methods do not contain the same number of signal dates.")

    weight_sums = targets.groupby(
        [
            "strategy_name",
            "as_of_date",
        ]
    )["target_weight"].sum()

    invalid_sums = weight_sums.sub(1.0).abs().gt(1.0e-6)

    if invalid_sums.any():
        invalid_key = invalid_sums.loc[invalid_sums].index[0]

        raise ValueError(f"At least one portfolio does not sum to one: {invalid_key}.")

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
    """Return performance rows for portfolio strategies only."""
    return (
        summary.loc[~summary["strategy_name"].eq(benchmark_name)]
        .sort_values("strategy_name")
        .reset_index(drop=True)
    )


def _build_cost_component_summary(
    enriched_trades: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize execution-cost components by strategy."""
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
    )

    summary["effective_cost_bps"] = np.where(
        summary["traded_notional"].gt(0.0),
        (summary["total_execution_cost"] / summary["traded_notional"] * 10_000.0),
        0.0,
    )

    return summary.sort_values("strategy_name").reset_index(drop=True)


def _build_comparison(
    gross_performance: pd.DataFrame,
    net_performance: pd.DataFrame,
    execution_summary: pd.DataFrame,
    cost_components: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    """Build the main gross-versus-net comparison."""
    gross = gross_performance.loc[
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
        ],
    ].rename(
        columns={
            column: (column if column == "strategy_name" else f"gross_{column}")
            for column in [
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
            ]
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
            column: (column if column == "strategy_name" else f"net_{column}")
            for column in [
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
        }
    )

    execution = execution_summary.loc[
        :,
        [
            "strategy_name",
            "rebalances",
            "total_transaction_cost",
            "total_traded_notional",
            "mean_two_way_turnover",
            "mean_one_way_turnover",
            "maximum_absolute_cash",
        ],
    ]

    components = cost_components.loc[
        :,
        [
            "strategy_name",
            "commission_cost",
            "spread_cost",
            "slippage_cost",
            "market_impact_cost",
            "effective_cost_bps",
            "maximum_order_adv_fraction",
        ],
    ]

    comparison = (
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
            components,
            on="strategy_name",
            how="inner",
            validate="one_to_one",
        )
    )

    comparison["total_return_cost_drag"] = (
        comparison["gross_total_return"] - comparison["net_total_return"]
    )

    comparison["cagr_cost_drag"] = comparison["gross_cagr"] - comparison["net_cagr"]

    comparison["transaction_cost_fraction_initial_capital"] = (
        comparison["total_transaction_cost"] / initial_capital
    )

    return comparison.sort_values(
        [
            "net_sharpe_ratio",
            "net_cagr",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)


def _build_checks(
    targets: pd.DataFrame,
    gross_daily: pd.DataFrame,
    net_daily: pd.DataFrame,
    net_trades: pd.DataFrame,
    execution_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    cash_tolerance: float,
) -> pd.DataFrame:
    """Audit the combined execution backtest."""
    expected_methods = len(EXPECTED_METHODS)

    expected_dates = int(targets["as_of_date"].nunique())

    expected_rebalances = expected_methods * expected_dates

    gross_methods = set(gross_daily["strategy_name"].unique())

    net_methods = set(net_daily["strategy_name"].unique())

    gross_date_sets = {
        method: set(group["date"]) for method, group in gross_daily.groupby("strategy_name")
    }

    net_date_sets = {
        method: set(group["date"]) for method, group in net_daily.groupby("strategy_name")
    }

    aligned_dates = (
        len({tuple(sorted(dates)) for dates in gross_date_sets.values()}) == 1
        and len({tuple(sorted(dates)) for dates in net_date_sets.values()}) == 1
    )

    checks = [
        (
            "expected_methods",
            int(gross_methods != EXPECTED_METHODS or net_methods != EXPECTED_METHODS),
            "Gross and net backtests must contain all five portfolio methods.",
        ),
        (
            "aligned_backtest_dates",
            int(not aligned_dates),
            "All methods must use identical trading dates.",
        ),
        (
            "expected_rebalances",
            int(execution_summary["rebalances"].sum() != expected_rebalances),
            "Every method must execute every scheduled rebalance.",
        ),
        (
            "positive_portfolio_values",
            int(
                gross_daily["portfolio_value"].le(0.0).sum()
                + net_daily["portfolio_value"].le(0.0).sum()
            ),
            "Gross and net portfolio values must remain positive.",
        ),
        (
            "nonnegative_transaction_costs",
            int(net_trades["transaction_cost"].lt(0.0).sum()),
            "Execution costs cannot be negative.",
        ),
        (
            "finite_transaction_costs",
            int((~np.isfinite(net_trades["transaction_cost"].to_numpy(dtype=float))).sum()),
            "Execution costs must be finite.",
        ),
        (
            "cash_accounting",
            int(execution_summary["maximum_absolute_cash"].gt(cash_tolerance + 1.0e-6).sum()),
            "Residual cash must remain within the configured accounting tolerance.",
        ),
        (
            "comparison_rows",
            int(len(comparison) != expected_methods),
            "The final comparison must contain one row per portfolio method.",
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
    """Format values for the Markdown report."""
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
    """Convert a dataframe to a compact Markdown table."""
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
    cost_components: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    execution_config: ExecutionCostConfig,
    initial_capital: float,
) -> str:
    """Build the execution comparison report."""
    main_columns = [
        "strategy_name",
        "gross_cagr",
        "net_cagr",
        "cagr_cost_drag",
        "net_sharpe_ratio",
        "net_sortino_ratio",
        "net_maximum_drawdown",
        "net_beta_vs_spy",
        "net_excess_cagr",
        "mean_one_way_turnover",
        "total_transaction_cost",
        "effective_cost_bps",
    ]

    return "\n".join(
        [
            "# Portfolio Execution Comparison",
            "",
            "## Assumptions",
            "",
            f"- Initial capital: `${initial_capital:,.0f}`.",
            (
                "- Linear execution costs: "
                f"`{execution_config.linear_cost_bps:.2f}` bps "
                "per dollar traded."
            ),
            (f"- Market impact coefficient: `{execution_config.market_impact_coefficient:.4f}`."),
            (
                "- Market impact uses daily volatility and the "
                "square root of order size divided by ADV."
            ),
            (
                "- Gross portfolios use zero transaction costs. "
                "Net portfolios use the advanced execution model."
            ),
            (
                "- SPY net performance includes the same linear "
                "execution cost on its initial buy-and-hold trade; "
                "no market-impact term is applied to SPY."
            ),
            "",
            "## Gross versus net performance",
            "",
            _to_markdown(
                comparison.loc[
                    :,
                    main_columns,
                ]
            ),
            "",
            "## Execution-cost decomposition",
            "",
            _to_markdown(cost_components),
            "",
            "## Readiness checks",
            "",
            _to_markdown(checks),
            "",
        ]
    )


def main() -> None:
    """Run the five-method gross and net execution comparison."""
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

    combined_weights = pd.read_parquet(ALL_WEIGHTS_PATH)

    market_data = pd.read_parquet(MARKET_DATA_PATH)

    risk_estimates = pd.read_parquet(RISK_ESTIMATES_PATH)

    spy_data = pd.read_parquet(SPY_DATA_PATH)

    targets = _prepare_backtest_targets(combined_weights)

    gross_config = replace(
        backtest_config,
        transaction_cost_bps=0.0,
    )

    net_config = replace(
        backtest_config,
        transaction_cost_bps=0.0,
    )

    logger.info("Running gross five-method backtest.")

    gross_outputs = run_mvp_backtest(
        targets,
        market_data,
        config=gross_config,
    )

    logger.info("Running transaction-cost-aware five-method backtest.")

    net_outputs = run_mvp_backtest(
        targets,
        market_data,
        config=net_config,
        risk_estimates=risk_estimates,
        execution_cost_config=execution_config,
    )

    start_date = pd.to_datetime(net_outputs.daily_performance["date"]).min()

    end_date = pd.to_datetime(net_outputs.daily_performance["date"]).max()

    gross_spy = build_buy_and_hold_benchmark(
        spy_data,
        strategy_name=(evaluation_config.benchmark_name),
        ticker=(evaluation_config.benchmark_ticker),
        start_date=start_date,
        end_date=end_date,
        initial_capital=(gross_config.initial_capital),
        transaction_cost_bps=0.0,
    )

    net_spy = build_buy_and_hold_benchmark(
        spy_data,
        strategy_name=(evaluation_config.benchmark_name),
        ticker=(evaluation_config.benchmark_ticker),
        start_date=start_date,
        end_date=end_date,
        initial_capital=(net_config.initial_capital),
        transaction_cost_bps=(execution_config.linear_cost_bps),
    )

    gross_evaluation = evaluate_performance(
        gross_outputs.daily_performance,
        gross_spy,
        initial_capital=(gross_config.initial_capital),
        config=evaluation_config,
    )

    net_evaluation = evaluate_performance(
        net_outputs.daily_performance,
        net_spy,
        initial_capital=(net_config.initial_capital),
        config=evaluation_config,
    )

    gross_performance = _strategy_performance(
        gross_evaluation.performance_summary,
        benchmark_name=(evaluation_config.benchmark_name),
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

    cost_components = _build_cost_component_summary(enriched_trades)

    comparison = _build_comparison(
        gross_performance,
        net_performance,
        net_outputs.execution_summary,
        cost_components,
        initial_capital=(net_config.initial_capital),
    )

    checks = _build_checks(
        targets,
        gross_outputs.daily_performance,
        net_outputs.daily_performance,
        net_outputs.trades,
        net_outputs.execution_summary,
        comparison,
        cash_tolerance=(net_config.cash_tolerance),
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_parquet_atomically(
        gross_outputs.daily_performance,
        GROSS_DAILY_PATH,
    )

    _write_parquet_atomically(
        net_outputs.daily_performance,
        NET_DAILY_PATH,
    )

    _write_parquet_atomically(
        net_outputs.daily_positions,
        NET_POSITIONS_PATH,
    )

    _write_parquet_atomically(
        enriched_trades,
        NET_TRADES_PATH,
    )

    _write_parquet_atomically(
        net_outputs.rebalance_summary,
        NET_REBALANCES_PATH,
    )

    _write_csv(
        gross_performance,
        GROSS_PERFORMANCE_PATH,
    )

    _write_csv(
        net_performance,
        NET_PERFORMANCE_PATH,
    )

    _write_csv(
        net_outputs.execution_summary,
        EXECUTION_SUMMARY_PATH,
    )

    _write_csv(
        cost_components,
        COST_COMPONENTS_PATH,
    )

    _write_csv(
        comparison,
        COMPARISON_PATH,
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
            cost_components,
            checks,
            execution_config=execution_config,
            initial_capital=(net_config.initial_capital),
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(
            f"Execution backtest validation failed with {failed_checks} failed checks."
        )

    logger.info("Five-method execution comparison completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Five-method transaction-cost-aware backtest")

    print("------------------------------------------------")

    print(f"methods: {targets['strategy_name'].nunique()}")

    print(f"signal_dates: {targets['as_of_date'].nunique()}")

    print(f"trading_days: {net_outputs.daily_performance['date'].nunique()}")

    print(f"initial_capital: ${net_config.initial_capital:,.0f}")

    print(f"linear_cost_bps: {execution_config.linear_cost_bps:.2f}")

    print(f"market_impact_coefficient: {execution_config.market_impact_coefficient:.4f}")

    print()

    print("Gross versus net comparison:")

    print(
        comparison.loc[
            :,
            [
                "strategy_name",
                "gross_cagr",
                "net_cagr",
                "cagr_cost_drag",
                "net_sharpe_ratio",
                "net_sortino_ratio",
                "net_maximum_drawdown",
                "net_excess_cagr",
                "mean_one_way_turnover",
                "total_transaction_cost",
                "effective_cost_bps",
            ],
        ].to_string(index=False)
    )

    print()

    print("Execution cost decomposition:")

    print(
        cost_components.loc[
            :,
            [
                "strategy_name",
                "commission_cost",
                "spread_cost",
                "slippage_cost",
                "market_impact_cost",
                "total_execution_cost",
                "effective_cost_bps",
                "maximum_order_adv_fraction",
            ],
        ].to_string(index=False)
    )

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Comparison table: {COMPARISON_PATH}")

    print(f"Cost components: {COST_COMPONENTS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
