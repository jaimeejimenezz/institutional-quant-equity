"""Evaluate portfolio robustness to exclusions within the frozen 50-stock universe."""

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

FINAL_SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

MARKET_DATA_PATH = PROCESSED_DATA_DIR / "market_daily.parquet"

RISK_ESTIMATES_PATH = PROCESSED_DATA_DIR / "risk_estimates.parquet"

SPY_DATA_PATH = PROCESSED_DATA_DIR / "benchmark_spy_daily.parquet"

WEIGHTS_PATH = PROCESSED_DATA_DIR / "robustness_universe_exclusion_weights.parquet"

TABLES_DIR = REPORTS_DIR / "tables"

RESULTS_PATH = TABLES_DIR / "robustness_universe_exclusion_results.csv"

DIAGNOSTICS_PATH = TABLES_DIR / "robustness_universe_exclusion_diagnostics.csv"

CHECKS_PATH = TABLES_DIR / "robustness_universe_exclusion_checks.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "universe_exclusion_robustness.md"

BASELINE_SCENARIO = "full_universe"
TECHNOLOGY_SECTOR = "Information Technology"

CANDIDATE_COUNT = 25
MAX_SECURITY_WEIGHT = 0.05
MAX_SECTOR_WEIGHT = 0.25
MINIMUM_POSITIONS = 20
WEIGHT_TOLERANCE = 1.0e-8
TOP_ADV_EXCLUSION_COUNT = 5


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


def _prepare_signal(
    signal: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the frozen final alpha signal."""
    required_columns = {
        "as_of_date",
        "ticker",
        "sector",
        "percentile_score",
        "rank",
    }

    missing = sorted(required_columns.difference(signal.columns))

    if missing:
        raise ValueError("Final alpha signal is missing columns: " + ", ".join(missing) + ".")

    data = signal.copy()

    data["as_of_date"] = pd.to_datetime(
        data["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    for column in (
        "ticker",
        "sector",
    ):
        data[column] = data[column].astype("string").str.strip()

    if (
        data["as_of_date"].isna().any()
        or data[
            [
                "ticker",
                "sector",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Final alpha signal contains invalid keys.")

    if data.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    ).any():
        raise ValueError("Final alpha signal contains duplicate date-ticker rows.")

    return data.sort_values(
        [
            "as_of_date",
            "rank",
            "ticker",
        ]
    ).reset_index(drop=True)


def _prepare_risk(
    risk: pd.DataFrame,
) -> pd.DataFrame:
    """Validate point-in-time ADV used for the liquidity exclusion stress test."""
    required_columns = {
        "as_of_date",
        "ticker",
        "average_dollar_volume",
    }

    missing = sorted(required_columns.difference(risk.columns))

    if missing:
        raise ValueError("Risk estimates are missing columns: " + ", ".join(missing) + ".")

    data = risk.copy()

    data["as_of_date"] = pd.to_datetime(
        data["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    data["ticker"] = data["ticker"].astype("string").str.strip()

    data["average_dollar_volume"] = pd.to_numeric(
        data["average_dollar_volume"],
        errors="coerce",
    )

    if (
        data["as_of_date"].isna().any()
        or data["ticker"].isna().any()
        or data["average_dollar_volume"].isna().any()
    ):
        raise ValueError("Risk estimates contain invalid ADV observations.")

    return data


def _exclude_top_adv(
    signal: pd.DataFrame,
    risk: pd.DataFrame,
) -> pd.DataFrame:
    """Exclude the five highest-ADV stocks independently on each signal date."""
    adv = risk.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "average_dollar_volume",
        ],
    ].copy()

    merged = signal.merge(
        adv,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    if merged["average_dollar_volume"].isna().any():
        raise ValueError("ADV exclusion stress test is missing point-in-time liquidity data.")

    merged["adv_rank"] = merged.groupby("as_of_date")["average_dollar_volume"].rank(
        method="first",
        ascending=False,
    )

    filtered = merged.loc[merged["adv_rank"].gt(TOP_ADV_EXCLUSION_COUNT)].drop(
        columns=[
            "average_dollar_volume",
            "adv_rank",
        ]
    )

    return filtered.reset_index(drop=True)


def _scenario_signals(
    signal: pd.DataFrame,
    risk: pd.DataFrame,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """Build frozen-universe exclusion scenarios."""
    sectors = sorted(signal["sector"].unique())

    scenarios = {
        BASELINE_SCENARIO: signal.copy(),
    }

    metadata_rows = [
        {
            "scenario": BASELINE_SCENARIO,
            "scenario_type": "baseline",
            "excluded_group": "",
            "is_baseline": True,
        }
    ]

    if TECHNOLOGY_SECTOR not in sectors:
        raise ValueError(f"Expected sector not found: {TECHNOLOGY_SECTOR}.")

    scenarios["exclude_information_technology"] = signal.loc[
        signal["sector"].ne(TECHNOLOGY_SECTOR)
    ].copy()

    metadata_rows.append(
        {
            "scenario": "exclude_information_technology",
            "scenario_type": "named_sector_exclusion",
            "excluded_group": TECHNOLOGY_SECTOR,
            "is_baseline": False,
        }
    )

    for sector in sectors:
        scenario = "leave_out_sector_" + sector.lower().replace(
            " ",
            "_",
        ).replace(
            "&",
            "and",
        )

        if sector == TECHNOLOGY_SECTOR:
            continue

        scenarios[scenario] = signal.loc[signal["sector"].ne(sector)].copy()

        metadata_rows.append(
            {
                "scenario": scenario,
                "scenario_type": "leave_one_sector_out",
                "excluded_group": sector,
                "is_baseline": False,
            }
        )

    scenarios["exclude_top5_adv"] = _exclude_top_adv(
        signal,
        risk,
    )

    metadata_rows.append(
        {
            "scenario": "exclude_top5_adv",
            "scenario_type": "liquidity_concentration_proxy",
            "excluded_group": (f"Top {TOP_ADV_EXCLUSION_COUNT} stocks by point-in-time ADV"),
            "is_baseline": False,
        }
    )

    metadata = pd.DataFrame(metadata_rows)

    return (
        scenarios,
        metadata,
    )


def _build_weights(
    scenarios: dict[
        str,
        pd.DataFrame,
    ],
    *,
    config: BaselinePortfolioConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Construct score-weighted portfolios for every exclusion scenario."""
    weight_blocks = []
    universe_rows = []

    for scenario, signal in scenarios.items():
        counts = signal.groupby("as_of_date")["ticker"].nunique()

        if counts.min() < CANDIDATE_COUNT:
            raise ValueError(
                f"Scenario {scenario} has fewer than "
                f"{CANDIDATE_COUNT} eligible stocks on at least one date."
            )

        weights = build_score_weighted_portfolios(
            signal,
            config=config,
        ).copy()

        weights["method"] = scenario

        weight_blocks.append(weights)

        universe_rows.append(
            {
                "scenario": scenario,
                "minimum_eligible_stocks": int(counts.min()),
                "mean_eligible_stocks": float(counts.mean()),
                "maximum_eligible_stocks": int(counts.max()),
                "eligible_sectors": int(signal["sector"].nunique()),
            }
        )

    combined = (
        pd.concat(
            weight_blocks,
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

    universe_summary = pd.DataFrame(universe_rows)

    return (
        combined,
        universe_summary,
    )


def _prepare_targets(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Convert construction weights to backtest target weights."""
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

    return targets


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


def _build_results(
    performance: pd.DataFrame,
    execution_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    metadata: pd.DataFrame,
    universe_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine universe, performance, execution and construction metrics."""
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
        metadata.rename(
            columns={
                "scenario": "strategy_name",
            }
        )
        .merge(
            universe_summary.rename(
                columns={
                    "scenario": "strategy_name",
                }
            ),
            on="strategy_name",
            how="inner",
            validate="one_to_one",
        )
        .merge(
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
            ],
            on="strategy_name",
            how="inner",
            validate="one_to_one",
        )
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

    baseline = result.loc[result["is_baseline"]]

    if len(baseline) != 1:
        raise ValueError("Universe robustness must contain exactly one baseline.")

    baseline_cagr = float(baseline["cagr"].iloc[0])

    baseline_sharpe = float(baseline["sharpe_ratio"].iloc[0])

    baseline_drawdown = float(baseline["maximum_drawdown"].iloc[0])

    result["cagr_difference_vs_full"] = result["cagr"] - baseline_cagr

    result["sharpe_difference_vs_full"] = result["sharpe_ratio"] - baseline_sharpe

    result["drawdown_difference_vs_full"] = result["maximum_drawdown"] - baseline_drawdown

    return result.sort_values(
        [
            "is_baseline",
            "scenario_type",
            "strategy_name",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)


def _build_checks(
    weights: pd.DataFrame,
    results: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Audit frozen-universe exclusion robustness outputs."""
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

    maximum_weights = weights.groupby("method")["weight"].max()

    numeric_columns = [
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "excess_cagr",
        "effective_cost_bps",
    ]

    checks = [
        (
            "expected_scenarios",
            int(results["strategy_name"].nunique() != len(metadata)),
            "Every defined universe-exclusion scenario must be backtested.",
        ),
        (
            "expected_oos_dates",
            int(dates_per_method.ne(77).sum()),
            "Every exclusion scenario must contain all 77 frozen OOS dates.",
        ),
        (
            "fully_invested",
            int(weight_sums.sub(1.0).abs().gt(1.0e-6).sum()),
            "Every universe-exclusion portfolio must sum to one.",
        ),
        (
            "long_only",
            int(weights["weight"].lt(-WEIGHT_TOLERANCE).sum()),
            "Universe-exclusion portfolios must remain long-only.",
        ),
        (
            "security_cap",
            int(maximum_weights.gt(MAX_SECURITY_WEIGHT + 1.0e-6).sum()),
            "Universe-exclusion portfolios must respect the 5% security cap.",
        ),
        (
            "sector_cap",
            int(sector_weights["weight"].gt(MAX_SECTOR_WEIGHT + 1.0e-6).sum()),
            "Universe-exclusion portfolios must respect the 25% sector cap.",
        ),
        (
            "minimum_eligible_universe",
            int(results["minimum_eligible_stocks"].lt(CANDIDATE_COUNT).sum()),
            "Every scenario must retain at least 25 eligible stocks on every date.",
        ),
        (
            "finite_performance",
            int((~np.isfinite(results[numeric_columns].to_numpy(dtype=float))).sum()),
            "Key universe-robustness metrics must remain finite.",
        ),
        (
            "positive_final_values",
            int(results["final_portfolio_value"].le(0.0).sum()),
            "Every universe-exclusion strategy must retain positive final value.",
        ),
        (
            "one_baseline",
            int(results["is_baseline"].sum() != 1),
            "Exactly one full-universe baseline must be present.",
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
    """Build the frozen-universe exclusion robustness report."""
    display = results.loc[
        :,
        [
            "strategy_name",
            "scenario_type",
            "excluded_group",
            "minimum_eligible_stocks",
            "eligible_sectors",
            "cagr",
            "cagr_difference_vs_full",
            "sharpe_ratio",
            "sharpe_difference_vs_full",
            "maximum_drawdown",
            "drawdown_difference_vs_full",
            "excess_cagr",
            "mean_one_way_turnover",
            "effective_cost_bps",
        ],
    ]

    return "\n".join(
        [
            "# Frozen-Universe Exclusion Robustness",
            "",
            "## Methodology",
            "",
            (
                "- The production final alpha signal remains frozen. "
                "No model is retrained or tuned for these exclusions."
            ),
            (
                "- The score-weighted portfolio is reconstructed after "
                "removing selected groups from the existing 50-stock universe."
            ),
            (
                "- Tests include full universe, exclusion of Information "
                "Technology, leave-one-sector-out scenarios, and exclusion "
                "of the five highest-ADV stocks on each date."
            ),
            (
                "- The ADV test is a liquidity-concentration proxy only. "
                "It is not presented as a market-cap test."
            ),
            (
                "- A genuine expanded-universe test still requires adding "
                "new securities and rebuilding the upstream point-in-time data pipeline."
            ),
            ("- Portfolio constraints and advanced transaction-cost assumptions remain unchanged."),
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
    """Run frozen-universe exclusion robustness analysis."""
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

    signal = _prepare_signal(pd.read_parquet(FINAL_SIGNAL_PATH))

    risk_estimates = _prepare_risk(pd.read_parquet(RISK_ESTIMATES_PATH))

    market_data = pd.read_parquet(MARKET_DATA_PATH)

    spy_data = pd.read_parquet(SPY_DATA_PATH)

    scenarios, metadata = _scenario_signals(
        signal,
        risk_estimates,
    )

    logger.info("Building frozen-universe exclusion portfolios.")

    weights, universe_summary = _build_weights(
        scenarios,
        config=_portfolio_config(),
    )

    diagnostics = compute_portfolio_diagnostics(
        weights,
        weight_tolerance=WEIGHT_TOLERANCE,
    )

    targets = _prepare_targets(weights)

    net_config = replace(
        backtest_config,
        transaction_cost_bps=0.0,
    )

    logger.info("Running frozen-universe exclusion backtests.")

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
        universe_summary,
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
            f"Frozen-universe exclusion validation failed with {failed_checks} failed checks."
        )

    logger.info("Frozen-universe exclusion robustness analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Frozen-universe exclusion robustness")

    print("------------------------------------------------")

    print(f"scenarios: {results['strategy_name'].nunique()}")

    print(f"sectors_in_baseline: {signal['sector'].nunique()}")

    print()

    print("Universe exclusion results:")

    print(
        results.loc[
            :,
            [
                "strategy_name",
                "scenario_type",
                "excluded_group",
                "minimum_eligible_stocks",
                "eligible_sectors",
                "cagr",
                "cagr_difference_vs_full",
                "sharpe_ratio",
                "sharpe_difference_vs_full",
                "maximum_drawdown",
                "excess_cagr",
                "mean_one_way_turnover",
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
