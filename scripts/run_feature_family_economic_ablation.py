"""Run economic feature-family ablations through the frozen portfolio backtest."""

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
    build_score_weighted_portfolios,
    compute_portfolio_diagnostics,
)

FULL_SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

ABLATION_DIR = PROCESSED_DATA_DIR / "robustness" / "feature_family_ablation"

NO_FUNDAMENTALS_SIGNAL_PATH = ABLATION_DIR / "no_fundamentals_final_alpha_signal.parquet"

NO_MOMENTUM_SIGNAL_PATH = ABLATION_DIR / "no_momentum_final_alpha_signal.parquet"

MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

RISK_ESTIMATES_PATH = PROCESSED_DATA_DIR / "risk_estimates.parquet"

SPY_DATA_PATH = PROCESSED_DATA_DIR / "benchmark_spy_daily.parquet"

SIGNALS_PATH = ABLATION_DIR / "feature_family_economic_signals.parquet"

WEIGHTS_PATH = ABLATION_DIR / "feature_family_economic_weights.parquet"

DAILY_PERFORMANCE_PATH = ABLATION_DIR / "feature_family_economic_daily_performance.parquet"

COMBINED_DAILY_PATH = ABLATION_DIR / "feature_family_economic_combined_daily.parquet"

TABLES_DIR = REPORTS_DIR / "tables" / "feature_family_ablation"

RESULTS_PATH = TABLES_DIR / "economic_comparison.csv"

DIAGNOSTICS_PATH = TABLES_DIR / "economic_portfolio_diagnostics.csv"

EXECUTION_SUMMARY_PATH = TABLES_DIR / "economic_execution_summary.csv"

YEARLY_SUMMARY_PATH = TABLES_DIR / "economic_yearly_summary.csv"

MONTHLY_RETURNS_PATH = TABLES_DIR / "economic_monthly_returns.csv"

DRAWDOWNS_PATH = ABLATION_DIR / "feature_family_economic_drawdowns.parquet"

CHECKS_PATH = TABLES_DIR / "economic_checks.csv"

REFERENCE_RESULTS_PATH = REPORTS_DIR / "tables" / "robustness_ensemble_component_ablation.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "feature_family_ablation" / "economic_comparison.md"

BASELINE_SCENARIO = "full_ensemble"

SCENARIO_PATHS = {
    BASELINE_SCENARIO: FULL_SIGNAL_PATH,
    "no_fundamentals": NO_FUNDAMENTALS_SIGNAL_PATH,
    "no_momentum": NO_MOMENTUM_SIGNAL_PATH,
}

EXPECTED_SCENARIOS = set(SCENARIO_PATHS)

EXPECTED_OOS_DATES = 77
EXPECTED_CROSS_SECTION_SIZE = 50
EXPECTED_SIGNAL_ROWS_PER_SCENARIO = EXPECTED_OOS_DATES * EXPECTED_CROSS_SECTION_SIZE

CANDIDATE_COUNT = 25
MAX_SECURITY_WEIGHT = 0.05
MAX_SECTOR_WEIGHT = 0.25
MINIMUM_POSITIONS = 20
WEIGHT_TOLERANCE = 1.0e-8
REFERENCE_TOLERANCE = 1.0e-10


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


def _portfolio_config() -> BaselinePortfolioConfig:
    """Return the frozen score-weighted portfolio configuration."""
    config = BaselinePortfolioConfig(
        candidate_count=CANDIDATE_COUNT,
        equal_weight_positions=CANDIDATE_COUNT,
        max_security_weight=MAX_SECURITY_WEIGHT,
        max_sector_weight=MAX_SECTOR_WEIGHT,
        minimum_positions=MINIMUM_POSITIONS,
        weight_tolerance=WEIGHT_TOLERANCE,
    )

    config.validate()

    return config


def _load_signals() -> pd.DataFrame:
    """Load all final alpha signals and attach economic scenario names."""
    required_columns = (
        "fold_id",
        "as_of_date",
        "ticker",
        "sector",
        "raw_prediction",
        "percentile_score",
        "rank",
    )

    blocks = []

    for scenario, path in SCENARIO_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Required final signal not found: {path}")

        signal = pd.read_parquet(path).copy()

        missing = sorted(set(required_columns).difference(signal.columns))

        if missing:
            raise ValueError(
                f"{scenario} final signal is missing columns: " + ", ".join(missing) + "."
            )

        signal = signal.loc[
            :,
            required_columns,
        ].copy()

        signal["as_of_date"] = pd.to_datetime(signal["as_of_date"]).dt.normalize()

        if len(signal) != EXPECTED_SIGNAL_ROWS_PER_SCENARIO:
            raise ValueError(
                f"{scenario} should contain "
                f"{EXPECTED_SIGNAL_ROWS_PER_SCENARIO} rows; "
                f"found {len(signal)}."
            )

        duplicate_keys = int(
            signal.duplicated(
                [
                    "as_of_date",
                    "ticker",
                ]
            ).sum()
        )

        if duplicate_keys:
            raise ValueError(f"{scenario} contains {duplicate_keys} duplicated signal keys.")

        dates = int(signal["as_of_date"].nunique())

        if dates != EXPECTED_OOS_DATES:
            raise ValueError(
                f"{scenario} should contain {EXPECTED_OOS_DATES} OOS dates; found {dates}."
            )

        cross_sections = signal.groupby("as_of_date")["ticker"].nunique()

        invalid_cross_sections = int(cross_sections.ne(EXPECTED_CROSS_SECTION_SIZE).sum())

        if invalid_cross_sections:
            raise ValueError(f"{scenario} has {invalid_cross_sections} invalid cross-sections.")

        signal["scenario"] = scenario

        blocks.append(signal)

    combined = (
        pd.concat(
            blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "scenario",
                "as_of_date",
                "rank",
            ]
        )
        .reset_index(drop=True)
    )

    return combined


def _build_portfolios(
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Construct one score-weighted portfolio for every feature scenario."""
    config = _portfolio_config()

    blocks = []

    for scenario, signal in signals.groupby(
        "scenario",
        sort=True,
    ):
        weights = build_score_weighted_portfolios(
            signal,
            config=config,
        ).copy()

        weights["method"] = str(scenario)

        blocks.append(weights)

    return (
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


def _prepare_targets(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Convert portfolio weights to backtest targets."""
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
    """Return only portfolio-strategy rows."""
    return (
        summary.loc[~summary["strategy_name"].eq(benchmark_name)]
        .sort_values("strategy_name")
        .reset_index(drop=True)
    )


def _build_results(
    performance: pd.DataFrame,
    execution_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine economic, execution and construction diagnostics."""
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

    result["effective_cost_bps"] = np.where(
        result["total_traded_notional"].gt(0.0),
        (result["total_transaction_cost"] / result["total_traded_notional"] * 10_000.0),
        0.0,
    )

    baseline = result.loc[result["strategy_name"].eq(BASELINE_SCENARIO)]

    if len(baseline) != 1:
        raise ValueError(
            "Economic feature-family results require exactly one full-ensemble baseline."
        )

    baseline_cagr = float(baseline["cagr"].iloc[0])

    baseline_sharpe = float(baseline["sharpe_ratio"].iloc[0])

    baseline_excess = float(baseline["excess_cagr"].iloc[0])

    result["cagr_difference_vs_full"] = result["cagr"] - baseline_cagr

    result["sharpe_difference_vs_full"] = result["sharpe_ratio"] - baseline_sharpe

    result["excess_cagr_difference_vs_full"] = result["excess_cagr"] - baseline_excess

    result["is_baseline"] = result["strategy_name"].eq(BASELINE_SCENARIO)

    return result.sort_values(
        [
            "is_baseline",
            "strategy_name",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)


def _reference_full_checks(
    results: pd.DataFrame,
) -> list[tuple[str, int, str]]:
    """Cross-check FULL against the prior component-ablation baseline when available."""
    if not REFERENCE_RESULTS_PATH.exists():
        return []

    reference = pd.read_csv(REFERENCE_RESULTS_PATH)

    required = {
        "strategy_name",
        "cagr",
        "sharpe_ratio",
        "maximum_drawdown",
        "excess_cagr",
        "mean_one_way_turnover",
        "effective_cost_bps",
    }

    missing = sorted(required.difference(reference.columns))

    if missing:
        return [
            (
                "reference_full_schema",
                len(missing),
                (
                    "Existing ensemble-component ablation reference "
                    "must expose the baseline comparison metrics."
                ),
            )
        ]

    reference_full = reference.loc[reference["strategy_name"].eq(BASELINE_SCENARIO)]

    current_full = results.loc[results["strategy_name"].eq(BASELINE_SCENARIO)]

    if len(reference_full) != 1 or len(current_full) != 1:
        return [
            (
                "reference_full_row",
                1,
                (
                    "Exactly one FULL row must exist in both the "
                    "reference and current economic comparison."
                ),
            )
        ]

    metrics = (
        "cagr",
        "sharpe_ratio",
        "maximum_drawdown",
        "excess_cagr",
        "mean_one_way_turnover",
        "effective_cost_bps",
    )

    violations = 0

    for metric in metrics:
        observed = float(current_full[metric].iloc[0])

        expected = float(reference_full[metric].iloc[0])

        if not np.isclose(
            observed,
            expected,
            atol=REFERENCE_TOLERANCE,
            rtol=REFERENCE_TOLERANCE,
        ):
            violations += 1

    return [
        (
            "reference_full_reproduction",
            violations,
            (
                "FULL must reproduce the previously stored "
                "ensemble-component-ablation economic baseline."
            ),
        )
    ]


def _build_checks(
    signals: pd.DataFrame,
    weights: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Audit feature-family economic ablation outputs."""
    scenarios = set(signals["scenario"].unique())

    dates_per_scenario = signals.groupby("scenario")["as_of_date"].nunique()

    cross_sections = signals.groupby(
        [
            "scenario",
            "as_of_date",
        ]
    )["ticker"].nunique()

    full_keys = set(
        zip(
            signals.loc[
                signals["scenario"].eq(BASELINE_SCENARIO),
                "as_of_date",
            ],
            signals.loc[
                signals["scenario"].eq(BASELINE_SCENARIO),
                "ticker",
            ].astype(str),
            strict=True,
        )
    )

    key_differences = 0

    for scenario in EXPECTED_SCENARIOS:
        scenario_data = signals.loc[signals["scenario"].eq(scenario)]

        scenario_keys = set(
            zip(
                scenario_data["as_of_date"],
                scenario_data["ticker"].astype(str),
                strict=True,
            )
        )

        key_differences += len(full_keys.symmetric_difference(scenario_keys))

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

    max_weights = weights.groupby("method")["weight"].max()

    key_metrics = results[
        [
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "excess_cagr",
            "effective_cost_bps",
        ]
    ].to_numpy(dtype=float)

    checks = [
        (
            "expected_scenarios",
            int(scenarios != EXPECTED_SCENARIOS),
            ("The analysis must contain FULL, no fundamentals and no momentum."),
        ),
        (
            "expected_oos_dates",
            int(dates_per_scenario.ne(EXPECTED_OOS_DATES).sum()),
            (f"Every feature scenario must contain all {EXPECTED_OOS_DATES} frozen OOS dates."),
        ),
        (
            "complete_cross_sections",
            int(cross_sections.ne(EXPECTED_CROSS_SECTION_SIZE).sum()),
            (
                "Every feature scenario must retain the full "
                f"{EXPECTED_CROSS_SECTION_SIZE}-stock cross-section."
            ),
        ),
        (
            "identical_oos_keys",
            int(key_differences),
            ("All scenarios must use exactly the same date-ticker OOS keys as FULL."),
        ),
        (
            "fully_invested",
            int(weight_sums.sub(1.0).abs().gt(1.0e-6).sum()),
            ("Every feature-family portfolio must sum to one."),
        ),
        (
            "long_only",
            int(weights["weight"].lt(-WEIGHT_TOLERANCE).sum()),
            ("Feature-family portfolios must remain long-only."),
        ),
        (
            "security_cap",
            int(max_weights.gt(MAX_SECURITY_WEIGHT + 1.0e-6).sum()),
            ("Feature-family portfolios must respect the 5% security cap."),
        ),
        (
            "sector_cap",
            int(sector_weights["weight"].gt(MAX_SECTOR_WEIGHT + 1.0e-6).sum()),
            ("Feature-family portfolios must respect the 25% sector cap."),
        ),
        (
            "finite_performance",
            int((~np.isfinite(key_metrics)).sum()),
            ("Key economic performance metrics must remain finite."),
        ),
        (
            "positive_final_values",
            int(results["final_portfolio_value"].le(0.0).sum()),
            ("Every feature-family backtest must retain positive final value."),
        ),
    ]

    checks.extend(_reference_full_checks(results))

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
    """Build the feature-family economic-ablation report."""
    display = results.loc[
        :,
        [
            "strategy_name",
            "is_baseline",
            "cagr",
            "cagr_difference_vs_full",
            "sharpe_ratio",
            "sharpe_difference_vs_full",
            "maximum_drawdown",
            "excess_cagr",
            "excess_cagr_difference_vs_full",
            "mean_one_way_turnover",
            "construction_one_way_turnover",
            "maximum_sector_weight",
            "effective_cost_bps",
        ],
    ]

    return "\n".join(
        [
            "# Feature-Family Economic Ablation",
            "",
            "## Methodology",
            "",
            (
                "- Uses the frozen FULL final alpha signal and the two "
                "walk-forward-retrained feature-family final alpha signals."
            ),
            (
                "- All three signals are passed through the same frozen "
                "score-weighted portfolio construction."
            ),
            (
                "- Portfolio construction uses 25 candidates, a 5% security "
                "cap, a 25% sector cap and at least 20 positions."
            ),
            (
                "- Backtests use the same project configuration, advanced "
                "execution-cost model, risk estimates and SPY benchmark as the "
                "existing ensemble-component ablation."
            ),
            (
                "- The fixed transaction-cost field of the MVP engine is set "
                "to zero so costs are charged only through the advanced "
                "execution-cost model, matching the existing robustness template."
            ),
            (
                "- No test-period result is used to retune signals, portfolio "
                "limits, costs or execution assumptions."
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
    """Run portfolio-level feature-family economic ablation analysis."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    required_paths = (
        *SCENARIO_PATHS.values(),
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

    signals = _load_signals()

    market_data = pd.read_parquet(MARKET_DATA_PATH)

    risk_estimates = pd.read_parquet(RISK_ESTIMATES_PATH)

    spy_data = pd.read_parquet(SPY_DATA_PATH)

    logger.info("Building score-weighted feature-family portfolios.")

    weights = _build_portfolios(signals)

    diagnostics = compute_portfolio_diagnostics(
        weights,
        weight_tolerance=WEIGHT_TOLERANCE,
    )

    targets = _prepare_targets(weights)

    net_config = replace(
        backtest_config,
        transaction_cost_bps=0.0,
    )

    logger.info("Running feature-family economic backtests.")

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
    )

    checks = _build_checks(
        signals,
        weights,
        results,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_parquet(
        signals,
        SIGNALS_PATH,
    )

    _write_parquet(
        weights,
        WEIGHTS_PATH,
    )

    _write_parquet(
        outputs.daily_performance,
        DAILY_PERFORMANCE_PATH,
    )

    _write_parquet(
        evaluation.combined_daily,
        COMBINED_DAILY_PATH,
    )

    _write_parquet(
        evaluation.drawdowns,
        DRAWDOWNS_PATH,
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
        outputs.execution_summary,
        EXECUTION_SUMMARY_PATH,
    )

    _write_csv(
        evaluation.yearly_summary,
        YEARLY_SUMMARY_PATH,
    )

    _write_csv(
        evaluation.monthly_returns,
        MONTHLY_RETURNS_PATH,
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
        print()
        print(checks.to_string(index=False))

        raise ValueError(
            "Feature-family economic ablation validation failed with "
            f"{failed_checks} failed checks."
        )

    logger.info("Feature-family economic ablation completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Feature-family economic ablation")
    print("------------------------------------------------")
    print(f"scenarios: {results['strategy_name'].nunique()}")

    print()
    print("Economic results:")
    print(
        results.loc[
            :,
            [
                "strategy_name",
                "is_baseline",
                "cagr",
                "cagr_difference_vs_full",
                "sharpe_ratio",
                "sharpe_difference_vs_full",
                "maximum_drawdown",
                "excess_cagr",
                "excess_cagr_difference_vs_full",
                "mean_one_way_turnover",
                "construction_one_way_turnover",
                "effective_cost_bps",
            ],
        ].to_string(index=False)
    )

    print()
    print(f"readiness_checks: {len(checks)}")
    print(f"failed_readiness_checks: {failed_checks}")

    print()
    print(f"Signals: {SIGNALS_PATH}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Results table: {RESULTS_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
