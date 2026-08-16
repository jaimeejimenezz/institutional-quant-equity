"""Run transaction-cost sensitivity analysis for all portfolio methods."""

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

CAPACITY_ANALYSIS_PATH = REPORTS_DIR / "tables" / "capacity_analysis.csv"

SENSITIVITY_PATH = REPORTS_DIR / "tables" / "transaction_cost_sensitivity.csv"

CHECKS_PATH = REPORTS_DIR / "tables" / "transaction_cost_sensitivity_checks.csv"

REPORT_PATH = REPORTS_DIR / "execution" / "transaction_cost_analysis.md"

LINEAR_COST_SCENARIOS_BPS = (
    0.0,
    5.0,
    10.0,
    20.0,
    50.0,
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
    """Return portfolio strategies without the benchmark row."""
    return (
        summary.loc[~summary["strategy_name"].eq(benchmark_name)]
        .sort_values("strategy_name")
        .reset_index(drop=True)
    )


def _build_scenario_rows(
    performance: pd.DataFrame,
    execution_summary: pd.DataFrame,
    *,
    scenario_name: str,
    linear_cost_bps: float,
    liquidity_model: bool,
) -> pd.DataFrame:
    """Combine performance and execution metrics for one cost scenario."""
    metrics = performance.loc[
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
    ].copy()

    execution = execution_summary.loc[
        :,
        [
            "strategy_name",
            "total_transaction_cost",
            "total_traded_notional",
            "mean_two_way_turnover",
            "mean_one_way_turnover",
        ],
    ].copy()

    result = metrics.merge(
        execution,
        on="strategy_name",
        how="inner",
        validate="one_to_one",
    )

    result.insert(
        0,
        "scenario",
        scenario_name,
    )

    result.insert(
        1,
        "linear_cost_bps",
        float(linear_cost_bps),
    )

    result.insert(
        2,
        "liquidity_model",
        bool(liquidity_model),
    )

    result["realized_effective_cost_bps"] = np.where(
        result["total_traded_notional"].gt(0.0),
        (result["total_transaction_cost"] / result["total_traded_notional"] * 10_000.0),
        0.0,
    )

    return result


def _build_checks(
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Audit transaction-cost sensitivity results."""
    expected_scenarios = len(LINEAR_COST_SCENARIOS_BPS) + 1

    expected_rows = expected_scenarios * len(EXPECTED_METHODS)

    methods_per_scenario = sensitivity.groupby("scenario")["strategy_name"].nunique()

    linear = sensitivity.loc[~sensitivity["liquidity_model"]].sort_values(
        [
            "strategy_name",
            "linear_cost_bps",
        ]
    )

    monotonic_final_value = linear.groupby("strategy_name")["final_portfolio_value"].apply(
        lambda values: bool(np.all(np.diff(values.to_numpy(dtype=float)) <= 1.0e-6))
    )

    monotonic_realized_cost = linear.groupby("strategy_name")["realized_effective_cost_bps"].apply(
        lambda values: bool(np.all(np.diff(values.to_numpy(dtype=float)) >= -1.0e-8))
    )

    key_values = sensitivity[
        [
            "final_portfolio_value",
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "total_transaction_cost",
            "realized_effective_cost_bps",
        ]
    ].to_numpy(dtype=float)

    checks = [
        (
            "expected_rows",
            int(len(sensitivity) != expected_rows),
            "Sensitivity analysis must contain one row per method and cost scenario.",
        ),
        (
            "five_methods_per_scenario",
            int((methods_per_scenario != len(EXPECTED_METHODS)).sum()),
            "Every cost scenario must contain all five portfolio methods.",
        ),
        (
            "finite_metrics",
            int((~np.isfinite(key_values)).sum()),
            "Key transaction-cost sensitivity metrics must remain finite.",
        ),
        (
            "positive_final_values",
            int(sensitivity["final_portfolio_value"].le(0.0).sum()),
            "All strategies must retain positive final portfolio value.",
        ),
        (
            "nonnegative_transaction_costs",
            int(sensitivity["total_transaction_cost"].lt(0.0).sum()),
            "Transaction costs cannot be negative.",
        ),
        (
            "linear_cost_reduces_value",
            int((~monotonic_final_value).sum()),
            "Final value must not increase when the linear cost assumption rises.",
        ),
        (
            "realized_cost_non_decreasing",
            int((~monotonic_realized_cost).sum()),
            "Realized effective cost must not decline as the linear cost assumption rises.",
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
    """Format values for Markdown."""
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
    sensitivity: pd.DataFrame,
    capacity: pd.DataFrame | None,
    checks: pd.DataFrame,
    *,
    execution_config: ExecutionCostConfig,
    initial_capital: float,
) -> str:
    """Build the final transaction-cost analysis report."""
    sensitivity_display = sensitivity.loc[
        :,
        [
            "scenario",
            "strategy_name",
            "cagr",
            "sharpe_ratio",
            "maximum_drawdown",
            "excess_cagr",
            "mean_one_way_turnover",
            "total_transaction_cost",
            "realized_effective_cost_bps",
        ],
    ]

    lines = [
        "# Transaction Cost Analysis",
        "",
        "## Scope",
        "",
        (f"- Reference capital for sensitivity analysis: `${initial_capital:,.0f}`."),
        ("- Linear scenarios: `0`, `5`, `10`, `20` and `50` bps per dollar traded."),
        (
            "- Liquidity model: commission + half spread + slippage + "
            "volatility/ADV-dependent market impact."
        ),
        (f"- Liquidity-model linear component: `{execution_config.linear_cost_bps:.2f}` bps."),
        (f"- Market-impact coefficient: `{execution_config.market_impact_coefficient:.4f}`."),
        "",
        "## Cost sensitivity",
        "",
        _to_markdown(sensitivity_display),
        "",
    ]

    if capacity is not None:
        capacity_display = capacity.loc[
            :,
            [
                "capital",
                "strategy_name",
                "net_cagr",
                "cagr_cost_drag",
                "net_sharpe_ratio",
                "effective_cost_bps",
                "maximum_order_adv_fraction",
            ],
        ]

        lines.extend(
            [
                "## Capacity",
                "",
                _to_markdown(capacity_display),
                "",
            ]
        )

    lines.extend(
        [
            "## Readiness checks",
            "",
            _to_markdown(checks),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Run cost sensitivity and generate the final execution report."""
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

    initial_capital = float(base_backtest_config.initial_capital)

    scenario_blocks = []

    for cost_bps in LINEAR_COST_SCENARIOS_BPS:
        logger.info(
            "Running linear transaction-cost scenario: %.0f bps.",
            cost_bps,
        )

        scenario_config = replace(
            base_backtest_config,
            transaction_cost_bps=float(cost_bps),
        )

        outputs = run_mvp_backtest(
            targets,
            market_data,
            config=scenario_config,
        )

        start_date = pd.to_datetime(outputs.daily_performance["date"]).min()

        end_date = pd.to_datetime(outputs.daily_performance["date"]).max()

        benchmark = build_buy_and_hold_benchmark(
            spy_data,
            strategy_name=(evaluation_config.benchmark_name),
            ticker=(evaluation_config.benchmark_ticker),
            start_date=start_date,
            end_date=end_date,
            initial_capital=(initial_capital),
            transaction_cost_bps=float(cost_bps),
        )

        evaluation = evaluate_performance(
            outputs.daily_performance,
            benchmark,
            initial_capital=(initial_capital),
            config=evaluation_config,
        )

        performance = _strategy_performance(
            evaluation.performance_summary,
            benchmark_name=(evaluation_config.benchmark_name),
        )

        scenario_blocks.append(
            _build_scenario_rows(
                performance,
                outputs.execution_summary,
                scenario_name=(f"linear_{int(cost_bps)}bps"),
                linear_cost_bps=float(cost_bps),
                liquidity_model=False,
            )
        )

    logger.info("Running liquidity-dependent execution-cost scenario.")

    advanced_config = replace(
        base_backtest_config,
        transaction_cost_bps=0.0,
    )

    advanced_outputs = run_mvp_backtest(
        targets,
        market_data,
        config=advanced_config,
        risk_estimates=risk_estimates,
        execution_cost_config=(execution_config),
    )

    advanced_start_date = pd.to_datetime(advanced_outputs.daily_performance["date"]).min()

    advanced_end_date = pd.to_datetime(advanced_outputs.daily_performance["date"]).max()

    advanced_benchmark = build_buy_and_hold_benchmark(
        spy_data,
        strategy_name=(evaluation_config.benchmark_name),
        ticker=(evaluation_config.benchmark_ticker),
        start_date=advanced_start_date,
        end_date=advanced_end_date,
        initial_capital=(initial_capital),
        transaction_cost_bps=(execution_config.linear_cost_bps),
    )

    advanced_evaluation = evaluate_performance(
        advanced_outputs.daily_performance,
        advanced_benchmark,
        initial_capital=(initial_capital),
        config=evaluation_config,
    )

    advanced_performance = _strategy_performance(
        advanced_evaluation.performance_summary,
        benchmark_name=(evaluation_config.benchmark_name),
    )

    scenario_blocks.append(
        _build_scenario_rows(
            advanced_performance,
            advanced_outputs.execution_summary,
            scenario_name=("liquidity_model"),
            linear_cost_bps=(execution_config.linear_cost_bps),
            liquidity_model=True,
        )
    )

    sensitivity = (
        pd.concat(
            scenario_blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "strategy_name",
                "liquidity_model",
                "linear_cost_bps",
            ]
        )
        .reset_index(drop=True)
    )

    checks = _build_checks(sensitivity)

    failed_checks = int(checks["status"].eq("FAIL").sum())

    capacity = None

    if CAPACITY_ANALYSIS_PATH.exists():
        capacity = pd.read_csv(CAPACITY_ANALYSIS_PATH)

    _write_csv(
        sensitivity,
        SENSITIVITY_PATH,
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
            sensitivity,
            capacity,
            checks,
            execution_config=(execution_config),
            initial_capital=(initial_capital),
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(
            f"Transaction-cost sensitivity validation failed with {failed_checks} failed checks."
        )

    logger.info("Transaction-cost sensitivity analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Transaction cost sensitivity analysis")

    print("------------------------------------------------")

    print(f"methods: {sensitivity['strategy_name'].nunique()}")

    print(f"scenarios: {sensitivity['scenario'].nunique()}")

    print(f"reference_capital: ${initial_capital:,.0f}")

    print()

    print("Cost sensitivity:")

    print(
        sensitivity.loc[
            :,
            [
                "scenario",
                "strategy_name",
                "cagr",
                "sharpe_ratio",
                "maximum_drawdown",
                "excess_cagr",
                "mean_one_way_turnover",
                "total_transaction_cost",
                "realized_effective_cost_bps",
            ],
        ].to_string(index=False)
    )

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Sensitivity table: {SENSITIVITY_PATH}")

    print(f"Final execution report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
