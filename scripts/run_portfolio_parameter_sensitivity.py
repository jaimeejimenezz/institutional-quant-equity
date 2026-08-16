"""Evaluate Top-N and security-weight parameter robustness."""

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
from quant_equity.portfolio import (
    BaselinePortfolioConfig,
    build_equal_weight_portfolios,
    build_score_weighted_portfolios,
    compute_portfolio_diagnostics,
)

FINAL_SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

RISK_ESTIMATES_PATH = PROCESSED_DATA_DIR / "risk_estimates.parquet"

SPY_DATA_PATH = PROCESSED_DATA_DIR / "benchmark_spy_daily.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

WEIGHTS_PATH = PROCESSED_DATA_DIR / "robustness_portfolio_parameter_weights.parquet"

RESULTS_PATH = TABLES_DIR / "robustness_portfolio_parameter_sensitivity.csv"

DIAGNOSTICS_PATH = TABLES_DIR / "robustness_portfolio_parameter_diagnostics.csv"

CHECKS_PATH = TABLES_DIR / "robustness_portfolio_parameter_checks.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "portfolio_parameter_sensitivity.md"


TOP_N_VALUES = (
    10,
    20,
    25,
    30,
    40,
)

SECURITY_CAP_VALUES = (
    0.04,
    0.05,
    0.075,
    0.10,
)

BASELINE_TOP_N = 25
BASELINE_SECURITY_CAP = 0.05
MAX_SECTOR_WEIGHT = 0.25
WEIGHT_TOLERANCE = 1.0e-8


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


def _write_parquet(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write one Parquet dataset."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        path,
        index=False,
    )


def _top_n_config(
    top_n: int,
) -> BaselinePortfolioConfig:
    """Create a feasible equal-weight Top-N configuration."""
    minimum_required_cap = 1.0 / float(top_n)

    security_cap = max(
        BASELINE_SECURITY_CAP,
        minimum_required_cap,
    )

    config = BaselinePortfolioConfig(
        candidate_count=top_n,
        equal_weight_positions=top_n,
        max_security_weight=security_cap,
        max_sector_weight=MAX_SECTOR_WEIGHT,
        minimum_positions=top_n,
        weight_tolerance=WEIGHT_TOLERANCE,
    )

    config.validate()

    return config


def _score_cap_config(
    security_cap: float,
) -> BaselinePortfolioConfig:
    """Create a score-weighted configuration for one security cap."""
    config = BaselinePortfolioConfig(
        candidate_count=BASELINE_TOP_N,
        equal_weight_positions=BASELINE_TOP_N,
        max_security_weight=security_cap,
        max_sector_weight=MAX_SECTOR_WEIGHT,
        minimum_positions=20,
        weight_tolerance=WEIGHT_TOLERANCE,
    )

    config.validate()

    return config


def _build_sensitivity_weights(
    final_signal: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build all Top-N and security-cap sensitivity portfolios."""
    blocks = []
    metadata_rows = []

    for top_n in TOP_N_VALUES:
        config = _top_n_config(top_n)

        weights = build_equal_weight_portfolios(
            final_signal,
            config=config,
        ).copy()

        scenario = f"top_n_{top_n}"

        weights["method"] = scenario

        blocks.append(weights)

        metadata_rows.append(
            {
                "scenario": scenario,
                "experiment": "top_n",
                "top_n": int(top_n),
                "configured_security_cap": float(config.max_security_weight),
                "sector_cap": float(config.max_sector_weight),
                "is_baseline": bool(top_n == BASELINE_TOP_N),
            }
        )

    for security_cap in SECURITY_CAP_VALUES:
        config = _score_cap_config(security_cap)

        weights = build_score_weighted_portfolios(
            final_signal,
            config=config,
        ).copy()

        cap_bps = int(round(security_cap * 10_000.0))

        scenario = f"score_cap_{cap_bps}bps"

        weights["method"] = scenario

        blocks.append(weights)

        metadata_rows.append(
            {
                "scenario": scenario,
                "experiment": "security_cap",
                "top_n": int(BASELINE_TOP_N),
                "configured_security_cap": float(security_cap),
                "sector_cap": float(MAX_SECTOR_WEIGHT),
                "is_baseline": bool(
                    np.isclose(
                        security_cap,
                        BASELINE_SECURITY_CAP,
                    )
                ),
            }
        )

    combined = (
        pd.concat(
            blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "method",
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    metadata = (
        pd.DataFrame(metadata_rows)
        .sort_values(
            [
                "experiment",
                "top_n",
                "configured_security_cap",
            ]
        )
        .reset_index(drop=True)
    )

    return (
        combined,
        metadata,
    )


def _prepare_backtest_targets(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Convert construction weights to backtest targets."""
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
    """Return only strategy rows from performance evaluation."""
    return (
        summary.loc[~summary["strategy_name"].eq(benchmark_name)]
        .sort_values("strategy_name")
        .reset_index(drop=True)
    )


def _build_results(
    performance: pd.DataFrame,
    execution_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Combine performance, execution and construction diagnostics."""
    construction = (
        diagnostics.groupby(
            "method",
            as_index=False,
        )
        .agg(
            mean_positions=(
                "positions",
                "mean",
            ),
            maximum_weight=(
                "maximum_weight",
                "max",
            ),
            maximum_sector_weight=(
                "maximum_sector_weight",
                "max",
            ),
            mean_effective_positions=(
                "effective_positions",
                "mean",
            ),
            construction_one_way_turnover=(
                "one_way_turnover",
                "mean",
            ),
        )
        .rename(
            columns={
                "method": "strategy_name",
            }
        )
    )

    performance_columns = [
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

    execution_columns = [
        "strategy_name",
        "total_transaction_cost",
        "total_traded_notional",
        "mean_two_way_turnover",
        "mean_one_way_turnover",
    ]

    result = (
        metadata.rename(
            columns={
                "scenario": "strategy_name",
            }
        )
        .merge(
            performance.loc[
                :,
                performance_columns,
            ],
            on="strategy_name",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            execution_summary.loc[
                :,
                execution_columns,
            ],
            on="strategy_name",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            construction,
            on="strategy_name",
            how="inner",
            validate="one_to_one",
        )
    )

    result["effective_cost_bps"] = np.where(
        result["total_traded_notional"].gt(0.0),
        (result["total_transaction_cost"] / result["total_traded_notional"] * 10_000.0),
        0.0,
    )

    result["baseline_cagr"] = np.nan

    result["baseline_sharpe"] = np.nan

    for experiment, group in result.groupby("experiment"):
        baseline = group.loc[group["is_baseline"]]

        if len(baseline) != 1:
            raise ValueError(f"Experiment {experiment} does not have exactly one baseline.")

        baseline_cagr = float(baseline["cagr"].iloc[0])

        baseline_sharpe = float(baseline["sharpe_ratio"].iloc[0])

        mask = result["experiment"].eq(experiment)

        result.loc[
            mask,
            "baseline_cagr",
        ] = baseline_cagr

        result.loc[
            mask,
            "baseline_sharpe",
        ] = baseline_sharpe

    result["cagr_difference_vs_baseline"] = result["cagr"] - result["baseline_cagr"]

    result["sharpe_difference_vs_baseline"] = result["sharpe_ratio"] - result["baseline_sharpe"]

    return result.sort_values(
        [
            "experiment",
            "top_n",
            "configured_security_cap",
        ]
    ).reset_index(drop=True)


def _build_checks(
    weights: pd.DataFrame,
    results: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Audit parameter-sensitivity construction and backtest outputs."""
    expected_scenarios = len(TOP_N_VALUES) + len(SECURITY_CAP_VALUES)

    dates_per_method = weights.groupby("method")["as_of_date"].nunique()

    weight_sums = weights.groupby(
        [
            "method",
            "as_of_date",
        ]
    )["weight"].sum()

    sector_weights = weights.groupby(
        [
            "method",
            "as_of_date",
            "sector",
        ],
        as_index=False,
    )["weight"].sum()

    maximum_weights = (
        weights.groupby(
            "method",
            as_index=False,
        )["weight"]
        .max()
        .rename(
            columns={
                "method": "scenario",
                "weight": "observed_maximum_weight",
            }
        )
        .merge(
            metadata.loc[
                :,
                [
                    "scenario",
                    "configured_security_cap",
                ],
            ],
            on="scenario",
            how="left",
            validate="one_to_one",
        )
    )

    checks = [
        (
            "expected_scenarios",
            int(results["strategy_name"].nunique() != expected_scenarios),
            "All Top-N and security-cap scenarios must be present.",
        ),
        (
            "expected_signal_dates",
            int(dates_per_method.ne(77).sum()),
            "Every scenario must contain all 77 OOS signal dates.",
        ),
        (
            "fully_invested",
            int(weight_sums.sub(1.0).abs().gt(1.0e-6).sum()),
            "Every sensitivity portfolio must sum to one.",
        ),
        (
            "long_only",
            int(weights["weight"].lt(-WEIGHT_TOLERANCE).sum()),
            "Sensitivity portfolios must remain long-only.",
        ),
        (
            "security_caps",
            int(
                maximum_weights["observed_maximum_weight"]
                .gt(maximum_weights["configured_security_cap"] + 1.0e-6)
                .sum()
            ),
            "Observed weights must respect each scenario security cap.",
        ),
        (
            "sector_cap",
            int(sector_weights["weight"].gt(MAX_SECTOR_WEIGHT + 1.0e-6).sum()),
            "Every sensitivity portfolio must respect the 25% sector cap.",
        ),
        (
            "finite_performance",
            int(
                (
                    ~np.isfinite(
                        results[
                            [
                                "cagr",
                                "annualized_volatility",
                                "sharpe_ratio",
                                "maximum_drawdown",
                                "excess_cagr",
                                "effective_cost_bps",
                            ]
                        ].to_numpy(dtype=float)
                    )
                ).sum()
            ),
            "Key sensitivity performance metrics must be finite.",
        ),
        (
            "positive_final_value",
            int(results["final_portfolio_value"].le(0.0).sum()),
            "Every sensitivity backtest must retain positive portfolio value.",
        ),
        (
            "one_baseline_per_experiment",
            int(metadata.groupby("experiment")["is_baseline"].sum().ne(1).sum()),
            "Each parameter experiment must contain exactly one frozen baseline.",
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
    """Convert a dataframe to a Markdown table."""
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
    results: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    """Build the portfolio-parameter sensitivity report."""
    display = results.loc[
        :,
        [
            "experiment",
            "strategy_name",
            "top_n",
            "configured_security_cap",
            "is_baseline",
            "cagr",
            "cagr_difference_vs_baseline",
            "sharpe_ratio",
            "sharpe_difference_vs_baseline",
            "maximum_drawdown",
            "excess_cagr",
            "mean_one_way_turnover",
            "maximum_weight",
            "maximum_sector_weight",
            "mean_effective_positions",
            "effective_cost_bps",
        ],
    ]

    return "\n".join(
        [
            "# Portfolio Parameter Sensitivity",
            "",
            "## Methodology",
            "",
            (
                "- Top-N sensitivity uses sector-controlled equal-weight "
                "portfolios for N = 10, 20, 25, 30 and 40."
            ),
            (
                "- A fixed 5% security cap is mathematically infeasible for "
                "Top-10, so each equal-weight scenario uses the larger of "
                "5% and 1/N as its configured security cap."
            ),
            (
                "- Security-cap sensitivity keeps the score-weighted "
                "candidate count fixed at 25 and tests 4%, 5%, 7.5% and 10%."
            ),
            ("- Sector cap remains fixed at 25% in every scenario."),
            (
                "- All scenarios use the same frozen final alpha signal, "
                "same OOS dates, same market data and same advanced execution-cost model."
            ),
            (
                "- Baselines are Top-25 for the Top-N experiment and a "
                "5% security cap for the score-weighted cap experiment."
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
    """Run Top-N and security-cap robustness analysis."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    required_paths = (
        FINAL_SIGNAL_PATH,
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

    final_signal = pd.read_parquet(FINAL_SIGNAL_PATH)

    market_data = pd.read_parquet(MARKET_DATA_PATH)

    risk_estimates = pd.read_parquet(RISK_ESTIMATES_PATH)

    spy_data = pd.read_parquet(SPY_DATA_PATH)

    logger.info("Building Top-N and security-cap sensitivity portfolios.")

    weights, metadata = _build_sensitivity_weights(final_signal)

    diagnostics = compute_portfolio_diagnostics(
        weights,
        weight_tolerance=WEIGHT_TOLERANCE,
    )

    targets = _prepare_backtest_targets(weights)

    net_config = replace(
        backtest_config,
        transaction_cost_bps=0.0,
    )

    logger.info("Running parameter-sensitivity backtests.")

    outputs = run_mvp_backtest(
        targets,
        market_data,
        config=net_config,
        risk_estimates=risk_estimates,
        execution_cost_config=execution_config,
    )

    start_date = pd.to_datetime(outputs.daily_performance["date"]).min()

    end_date = pd.to_datetime(outputs.daily_performance["date"]).max()

    benchmark = build_buy_and_hold_benchmark(
        spy_data,
        strategy_name=(evaluation_config.benchmark_name),
        ticker=(evaluation_config.benchmark_ticker),
        start_date=start_date,
        end_date=end_date,
        initial_capital=(net_config.initial_capital),
        transaction_cost_bps=(execution_config.linear_cost_bps),
    )

    evaluation = evaluate_performance(
        outputs.daily_performance,
        benchmark,
        initial_capital=(net_config.initial_capital),
        config=evaluation_config,
    )

    performance = _strategy_performance(
        evaluation.performance_summary,
        benchmark_name=(evaluation_config.benchmark_name),
    )

    results = _build_results(
        performance,
        outputs.execution_summary,
        diagnostics,
        metadata,
    )

    checks = _build_checks(
        weights,
        results,
        metadata,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_parquet(
        weights,
        WEIGHTS_PATH,
    )

    _write_csv(
        results,
        RESULTS_PATH,
    )

    _write_csv(
        diagnostics,
        DIAGNOSTICS_PATH,
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
            results,
            checks,
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(
            f"Portfolio parameter sensitivity validation failed with {failed_checks} failed checks."
        )

    logger.info("Portfolio parameter sensitivity analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Portfolio parameter sensitivity")

    print("------------------------------------------------")

    print(f"scenarios: {results['strategy_name'].nunique()}")

    print(f"top_n_scenarios: {results['experiment'].eq('top_n').sum()}")

    print(f"security_cap_scenarios: {results['experiment'].eq('security_cap').sum()}")

    print()

    print("Sensitivity results:")

    print(
        results.loc[
            :,
            [
                "experiment",
                "strategy_name",
                "top_n",
                "configured_security_cap",
                "is_baseline",
                "cagr",
                "cagr_difference_vs_baseline",
                "sharpe_ratio",
                "sharpe_difference_vs_baseline",
                "maximum_drawdown",
                "excess_cagr",
                "mean_one_way_turnover",
                "maximum_weight",
                "maximum_sector_weight",
                "mean_effective_positions",
                "effective_cost_bps",
            ],
        ].to_string(index=False)
    )

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Results table: {RESULTS_PATH}")

    print(f"Diagnostics table: {DIAGNOSTICS_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
