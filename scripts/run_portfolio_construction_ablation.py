"""Evaluate portfolio-construction ablations with the frozen alpha signal."""

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
    PortfolioOptimizerConfig,
    build_alpha_risk_turnover_portfolios,
    build_score_weighted_portfolios,
    compute_portfolio_diagnostics,
)

SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

COVARIANCE_PATH = PROCESSED_DATA_DIR / "covariance_matrices.parquet"

RISK_ESTIMATES_PATH = PROCESSED_DATA_DIR / "risk_estimates.parquet"

MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

SPY_DATA_PATH = PROCESSED_DATA_DIR / "benchmark_spy_daily.parquet"

WEIGHTS_PATH = PROCESSED_DATA_DIR / "robustness_portfolio_construction_ablation_weights.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

RESULTS_PATH = TABLES_DIR / "robustness_portfolio_construction_ablation.csv"

DIAGNOSTICS_PATH = TABLES_DIR / "robustness_portfolio_construction_ablation_diagnostics.csv"

OPTIMIZER_PATH = TABLES_DIR / "robustness_portfolio_construction_optimizer_diagnostics.csv"

CHECKS_PATH = TABLES_DIR / "robustness_portfolio_construction_ablation_checks.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "portfolio_construction_ablation.md"

SCORE_CONTROLLED = "score_weighted_sector_controlled"
SCORE_NO_SECTOR = "score_weighted_no_sector_control"
OPTIMIZER_PENALIZED = "alpha_risk_turnover_penalized"
OPTIMIZER_NO_PENALTY = "alpha_risk_turnover_no_penalty"

EXPECTED_SCENARIOS = {
    SCORE_CONTROLLED,
    SCORE_NO_SECTOR,
    OPTIMIZER_PENALIZED,
    OPTIMIZER_NO_PENALTY,
}

EXPECTED_OOS_DATES = 77


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
    """Write one Parquet table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    data.to_parquet(
        path,
        index=False,
    )


def _rename_method(
    weights: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Assign a research scenario name without altering weights."""
    result = weights.copy()
    result["method"] = method
    return result


def _build_construction_scenarios(
    final_signal: pd.DataFrame,
    covariance: pd.DataFrame,
    risk_estimates: pd.DataFrame,
    *,
    construction_config: BaselinePortfolioConfig,
    optimizer_config: PortfolioOptimizerConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build sector-control and turnover-penalty ablation portfolios."""
    logger = logging.getLogger("quant_equity")

    logger.info("Building score-weighted sector-control baseline.")

    score_controlled = _rename_method(
        build_score_weighted_portfolios(
            final_signal,
            config=construction_config,
        ),
        SCORE_CONTROLLED,
    )

    no_sector_config = replace(
        construction_config,
        max_sector_weight=1.0,
    )
    no_sector_config.validate()

    logger.info("Building score-weighted portfolio without sector control.")

    score_no_sector = _rename_method(
        build_score_weighted_portfolios(
            final_signal,
            config=no_sector_config,
        ),
        SCORE_NO_SECTOR,
    )

    logger.info("Building alpha-risk-turnover optimizer baseline.")

    (
        optimizer_penalized,
        penalized_diagnostics,
    ) = build_alpha_risk_turnover_portfolios(
        final_signal,
        covariance,
        risk_estimates,
        config=optimizer_config,
    )

    optimizer_penalized = _rename_method(
        optimizer_penalized,
        OPTIMIZER_PENALIZED,
    )

    penalized_diagnostics = penalized_diagnostics.copy()
    penalized_diagnostics["scenario"] = OPTIMIZER_PENALIZED

    no_penalty_config = replace(
        optimizer_config,
        turnover_penalty=0.0,
    )
    no_penalty_config.validate()

    logger.info("Building alpha-risk optimizer without turnover penalty.")

    (
        optimizer_no_penalty,
        no_penalty_diagnostics,
    ) = build_alpha_risk_turnover_portfolios(
        final_signal,
        covariance,
        risk_estimates,
        config=no_penalty_config,
    )

    optimizer_no_penalty = _rename_method(
        optimizer_no_penalty,
        OPTIMIZER_NO_PENALTY,
    )

    no_penalty_diagnostics = no_penalty_diagnostics.copy()
    no_penalty_diagnostics["scenario"] = OPTIMIZER_NO_PENALTY

    weights = (
        pd.concat(
            [
                score_controlled,
                score_no_sector,
                optimizer_penalized,
                optimizer_no_penalty,
            ],
            ignore_index=True,
            sort=False,
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

    optimizer_diagnostics = (
        pd.concat(
            [
                penalized_diagnostics,
                no_penalty_diagnostics,
            ],
            ignore_index=True,
            sort=False,
        )
        .sort_values(
            [
                "scenario",
                "as_of_date",
            ]
        )
        .reset_index(drop=True)
    )

    return (
        weights,
        optimizer_diagnostics,
    )


def _prepare_targets(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Convert construction weights to backtest targets."""
    return (
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
        .sort_values(
            [
                "as_of_date",
                "strategy_name",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )


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


def _optimizer_summary(
    optimizer_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize optimizer objective diagnostics by scenario."""
    required = {
        "scenario",
        "predicted_alpha_proxy",
        "predicted_volatility",
        "one_way_turnover",
        "maximum_weight",
        "maximum_sector_weight",
    }

    missing = sorted(required.difference(optimizer_diagnostics.columns))

    if missing:
        raise ValueError("Optimizer diagnostics are missing columns: " + ", ".join(missing) + ".")

    return (
        optimizer_diagnostics.groupby(
            "scenario",
            as_index=False,
        )
        .agg(
            mean_predicted_alpha_proxy=(
                "predicted_alpha_proxy",
                "mean",
            ),
            mean_predicted_volatility=(
                "predicted_volatility",
                "mean",
            ),
            optimizer_mean_one_way_turnover=(
                "one_way_turnover",
                "mean",
            ),
            optimizer_maximum_weight=(
                "maximum_weight",
                "max",
            ),
            optimizer_maximum_sector_weight=(
                "maximum_sector_weight",
                "max",
            ),
        )
        .rename(
            columns={
                "scenario": "strategy_name",
            }
        )
    )


def _build_results(
    performance: pd.DataFrame,
    execution_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    optimizer_diagnostics: pd.DataFrame,
    *,
    construction_config: BaselinePortfolioConfig,
    optimizer_config: PortfolioOptimizerConfig,
) -> pd.DataFrame:
    """Combine performance and construction-ablation diagnostics."""
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

    result = (
        performance.loc[
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
        ]
        .merge(
            execution_summary.loc[
                :,
                [
                    "strategy_name",
                    "total_transaction_cost",
                    "total_traded_notional",
                    "mean_two_way_turnover",
                    "mean_one_way_turnover",
                ],
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

    optimizer = _optimizer_summary(optimizer_diagnostics)

    result = result.merge(
        optimizer,
        on="strategy_name",
        how="left",
        validate="one_to_one",
    )

    result["effective_cost_bps"] = np.where(
        result["total_traded_notional"].gt(0.0),
        (result["total_transaction_cost"] / result["total_traded_notional"] * 10_000.0),
        0.0,
    )

    experiment_map = {
        SCORE_CONTROLLED: (
            "sector_control",
            True,
        ),
        SCORE_NO_SECTOR: (
            "sector_control",
            False,
        ),
        OPTIMIZER_PENALIZED: (
            "turnover_penalty",
            True,
        ),
        OPTIMIZER_NO_PENALTY: (
            "turnover_penalty",
            False,
        ),
    }

    result["experiment"] = result["strategy_name"].map(lambda value: experiment_map[value][0])

    result["is_controlled_baseline"] = result["strategy_name"].map(
        lambda value: experiment_map[value][1]
    )

    result["configured_sector_cap"] = result["strategy_name"].map(
        {
            SCORE_CONTROLLED: (construction_config.max_sector_weight),
            SCORE_NO_SECTOR: 1.0,
            OPTIMIZER_PENALIZED: (optimizer_config.max_sector_weight),
            OPTIMIZER_NO_PENALTY: (optimizer_config.max_sector_weight),
        }
    )

    result["configured_turnover_penalty"] = result["strategy_name"].map(
        {
            SCORE_CONTROLLED: np.nan,
            SCORE_NO_SECTOR: np.nan,
            OPTIMIZER_PENALIZED: (optimizer_config.turnover_penalty),
            OPTIMIZER_NO_PENALTY: 0.0,
        }
    )

    result["baseline_cagr"] = np.nan

    result["baseline_sharpe"] = np.nan

    result["baseline_turnover"] = np.nan

    for experiment, group in result.groupby("experiment"):
        baseline = group.loc[group["is_controlled_baseline"]]

        if len(baseline) != 1:
            raise ValueError(f"Experiment {experiment} requires exactly one baseline.")

        mask = result["experiment"].eq(experiment)

        result.loc[
            mask,
            "baseline_cagr",
        ] = float(baseline["cagr"].iloc[0])

        result.loc[
            mask,
            "baseline_sharpe",
        ] = float(baseline["sharpe_ratio"].iloc[0])

        result.loc[
            mask,
            "baseline_turnover",
        ] = float(baseline["mean_one_way_turnover"].iloc[0])

    result["cagr_difference_vs_controlled"] = result["cagr"] - result["baseline_cagr"]

    result["sharpe_difference_vs_controlled"] = result["sharpe_ratio"] - result["baseline_sharpe"]

    result["turnover_difference_vs_controlled"] = (
        result["mean_one_way_turnover"] - result["baseline_turnover"]
    )

    return result.sort_values(
        [
            "experiment",
            "is_controlled_baseline",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


def _build_checks(
    weights: pd.DataFrame,
    results: pd.DataFrame,
    *,
    construction_config: BaselinePortfolioConfig,
    optimizer_config: PortfolioOptimizerConfig,
) -> pd.DataFrame:
    """Audit construction-ablation outputs."""
    scenarios = set(weights["method"].unique())

    dates_per_method = weights.groupby("method")["as_of_date"].nunique()

    weight_sums = weights.groupby(
        [
            "method",
            "as_of_date",
        ]
    )["weight"].sum()

    maximum_weights = weights.groupby("method")["weight"].max()

    sector_weights = weights.groupby(
        [
            "method",
            "as_of_date",
            "sector",
        ],
        as_index=False,
    )["weight"].sum()

    controlled_methods = {
        SCORE_CONTROLLED: (construction_config.max_sector_weight),
        OPTIMIZER_PENALIZED: (optimizer_config.max_sector_weight),
        OPTIMIZER_NO_PENALTY: (optimizer_config.max_sector_weight),
    }

    sector_violations = 0

    for method, cap in controlled_methods.items():
        sector_violations += int(
            sector_weights.loc[
                sector_weights["method"].eq(method),
                "weight",
            ]
            .gt(cap + 1.0e-6)
            .sum()
        )

    security_cap = max(
        construction_config.max_security_weight,
        optimizer_config.max_security_weight,
    )

    key_metrics = results[
        [
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "excess_cagr",
            "mean_one_way_turnover",
            "effective_cost_bps",
        ]
    ].to_numpy(dtype=float)

    checks = [
        (
            "expected_scenarios",
            int(scenarios != EXPECTED_SCENARIOS),
            "The analysis must contain all four construction-ablation scenarios.",
        ),
        (
            "expected_oos_dates",
            int(dates_per_method.ne(EXPECTED_OOS_DATES).sum()),
            "Every construction scenario must contain all frozen OOS dates.",
        ),
        (
            "fully_invested",
            int(weight_sums.sub(1.0).abs().gt(1.0e-6).sum()),
            "Every construction-ablation portfolio must sum to one.",
        ),
        (
            "long_only",
            int(weights["weight"].lt(-1.0e-8).sum()),
            "Construction-ablation portfolios must remain long-only.",
        ),
        (
            "security_caps",
            int(maximum_weights.gt(security_cap + 1.0e-6).sum()),
            "Every scenario must respect the configured security cap.",
        ),
        (
            "controlled_sector_caps",
            sector_violations,
            "Sector-controlled scenarios must respect the configured sector cap.",
        ),
        (
            "finite_performance",
            int((~np.isfinite(key_metrics)).sum()),
            "Key construction-ablation metrics must remain finite.",
        ),
        (
            "positive_final_values",
            int(results["final_portfolio_value"].le(0.0).sum()),
            "Every construction-ablation backtest must retain positive final value.",
        ),
        (
            "one_baseline_per_experiment",
            int(results.groupby("experiment")["is_controlled_baseline"].sum().ne(1).sum()),
            "Each construction experiment must contain exactly one controlled baseline.",
        ),
        (
            "turnover_penalty_removed",
            int(
                not np.isclose(
                    float(
                        results.loc[
                            results["strategy_name"].eq(OPTIMIZER_NO_PENALTY),
                            "configured_turnover_penalty",
                        ].iloc[0]
                    ),
                    0.0,
                    atol=0.0,
                    rtol=0.0,
                )
            ),
            "The turnover-penalty ablation must set the penalty exactly to zero.",
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
    results: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    """Build the construction-ablation report."""
    display = results.loc[
        :,
        [
            "experiment",
            "strategy_name",
            "is_controlled_baseline",
            "configured_sector_cap",
            "configured_turnover_penalty",
            "cagr",
            "cagr_difference_vs_controlled",
            "sharpe_ratio",
            "sharpe_difference_vs_controlled",
            "maximum_drawdown",
            "excess_cagr",
            "mean_one_way_turnover",
            "turnover_difference_vs_controlled",
            "maximum_sector_weight",
            "total_transaction_cost",
            "effective_cost_bps",
            "mean_predicted_alpha_proxy",
            "mean_predicted_volatility",
        ],
    ]

    return "\n".join(
        [
            "# Portfolio Construction Ablation",
            "",
            "## Methodology",
            "",
            (
                "- The final alpha signal, covariance estimates, risk estimates, "
                "market data and execution-cost assumptions remain frozen."
            ),
            (
                "- Sector-control ablation compares the score-weighted portfolio "
                "with the original 25% sector cap against the same construction "
                "with a 100% sector cap."
            ),
            (
                "- Turnover-penalty ablation compares the alpha-risk-turnover "
                "optimizer with its original penalty against an otherwise identical "
                "optimizer with turnover_penalty = 0."
            ),
            ("- No out-of-sample result is used to retune the remaining construction parameters."),
            (
                "- The score-weighted portfolio also serves as the project's "
                "non-optimized construction reference when interpreting whether "
                "portfolio optimization adds economic value."
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
    """Run portfolio-construction ablation analysis."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    required_paths = (
        SIGNAL_PATH,
        COVARIANCE_PATH,
        RISK_ESTIMATES_PATH,
        MARKET_DATA_PATH,
        SPY_DATA_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    project_config = load_config()

    construction_config = BaselinePortfolioConfig.from_mapping(
        project_config.get(
            "portfolio_construction",
            {},
        )
    )

    optimizer_config = PortfolioOptimizerConfig.from_mapping(
        project_config.get(
            "portfolio_optimizer",
            {},
        )
    )

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

    final_signal = pd.read_parquet(SIGNAL_PATH)

    covariance = pd.read_parquet(COVARIANCE_PATH)

    risk_estimates = pd.read_parquet(RISK_ESTIMATES_PATH)

    market_data = pd.read_parquet(MARKET_DATA_PATH)

    spy_data = pd.read_parquet(SPY_DATA_PATH)

    weights, optimizer_diagnostics = _build_construction_scenarios(
        final_signal,
        covariance,
        risk_estimates,
        construction_config=(construction_config),
        optimizer_config=(optimizer_config),
    )

    diagnostics = compute_portfolio_diagnostics(
        weights,
        weight_tolerance=(construction_config.weight_tolerance),
    )

    targets = _prepare_targets(weights)

    net_config = replace(
        backtest_config,
        transaction_cost_bps=0.0,
    )

    logger.info("Running construction-ablation backtests.")

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
        optimizer_diagnostics,
        construction_config=(construction_config),
        optimizer_config=(optimizer_config),
    )

    checks = _build_checks(
        weights,
        results,
        construction_config=(construction_config),
        optimizer_config=(optimizer_config),
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
        optimizer_diagnostics,
        OPTIMIZER_PATH,
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
            f"Portfolio-construction ablation validation failed with {failed_checks} failed checks."
        )

    logger.info("Portfolio-construction ablation analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Portfolio construction ablation")

    print("------------------------------------------------")

    print(f"scenarios: {results['strategy_name'].nunique()}")

    print(f"experiments: {results['experiment'].nunique()}")

    print()

    print("Construction ablation results:")

    print(
        results.loc[
            :,
            [
                "experiment",
                "strategy_name",
                "is_controlled_baseline",
                "configured_sector_cap",
                "configured_turnover_penalty",
                "cagr",
                "cagr_difference_vs_controlled",
                "sharpe_ratio",
                "sharpe_difference_vs_controlled",
                "maximum_drawdown",
                "excess_cagr",
                "mean_one_way_turnover",
                "turnover_difference_vs_controlled",
                "maximum_sector_weight",
                "total_transaction_cost",
                "effective_cost_bps",
                "mean_predicted_alpha_proxy",
                "mean_predicted_volatility",
            ],
        ].to_string(index=False)
    )

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Weights: {WEIGHTS_PATH}")

    print(f"Results table: {RESULTS_PATH}")

    print(f"Optimizer diagnostics: {OPTIMIZER_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
