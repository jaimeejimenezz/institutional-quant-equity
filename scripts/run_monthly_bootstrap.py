"""Run paired monthly bootstrap robustness analysis."""

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

BOOTSTRAP_SUMMARY_PATH = TABLES_DIR / "bootstrap_strategy_summary.csv"

PAIRWISE_PATH = TABLES_DIR / "bootstrap_pairwise_comparison.csv"

RANK_STABILITY_PATH = TABLES_DIR / "bootstrap_rank_stability.csv"

MONTHLY_RETURNS_PATH = TABLES_DIR / "bootstrap_monthly_returns.csv"

CHECKS_PATH = TABLES_DIR / "bootstrap_robustness_checks.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "monthly_bootstrap.md"

EXPECTED_METHODS = (
    "alpha_risk_turnover",
    "cvar",
    "median_mad_de",
    "score_weighted",
    "top_n_equal_weight",
)

BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 42
CONFIDENCE_LEVEL = 0.95


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
    """Validate the net daily portfolio return panel."""
    required_columns = {
        "date",
        "strategy_name",
        "daily_return",
    }

    missing = sorted(required_columns.difference(daily.columns))

    if missing:
        raise ValueError("Net daily backtest is missing columns: " + ", ".join(missing) + ".")

    data = daily.loc[
        :,
        [
            "date",
            "strategy_name",
            "daily_return",
        ],
    ].copy()

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

    methods = tuple(sorted(data["strategy_name"].unique()))

    if methods != tuple(sorted(EXPECTED_METHODS)):
        raise ValueError(f"Unexpected methods in net daily backtest. Received {methods}.")

    if data.duplicated(
        [
            "date",
            "strategy_name",
        ]
    ).any():
        raise ValueError("Net daily backtest contains duplicated date-strategy rows.")

    return data.sort_values(
        [
            "date",
            "strategy_name",
        ]
    ).reset_index(drop=True)


def _compound_monthly(
    data: pd.DataFrame,
    *,
    name_column: str,
) -> pd.DataFrame:
    """Compound daily returns into calendar-month returns."""
    working = data.copy()

    working["month"] = working["date"].dt.to_period("M").dt.to_timestamp("M")

    return (
        working.groupby(
            [
                "month",
                name_column,
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


def _build_monthly_panel(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    benchmark_name: str,
) -> pd.DataFrame:
    """Build one aligned monthly-return panel for all methods and SPY."""
    strategy_monthly = _compound_monthly(
        daily,
        name_column="strategy_name",
    )

    benchmark_data = benchmark.loc[
        :,
        [
            "date",
            "strategy_name",
            "daily_return",
        ],
    ].copy()

    benchmark_data["date"] = pd.to_datetime(benchmark_data["date"]).dt.normalize()

    benchmark_monthly = _compound_monthly(
        benchmark_data,
        name_column="strategy_name",
    )

    benchmark_monthly = benchmark_monthly.loc[benchmark_monthly["strategy_name"].eq(benchmark_name)]

    combined = pd.concat(
        [
            strategy_monthly,
            benchmark_monthly,
        ],
        ignore_index=True,
    )

    panel = combined.pivot(
        index="month",
        columns="strategy_name",
        values="monthly_return",
    ).sort_index()

    required_columns = [
        *EXPECTED_METHODS,
        benchmark_name,
    ]

    missing_columns = sorted(set(required_columns).difference(panel.columns))

    if missing_columns:
        raise ValueError("Monthly panel is missing methods: " + ", ".join(missing_columns) + ".")

    panel = panel.loc[
        :,
        required_columns,
    ]

    if panel.isna().any().any():
        missing_month = panel.index[panel.isna().any(axis=1)][0]

        raise ValueError(
            "Monthly bootstrap panel is not aligned. "
            f"First incomplete month: {missing_month.date()}."
        )

    if len(panel) < 24:
        raise ValueError("Monthly bootstrap requires at least 24 aligned months.")

    return panel


def _annualized_geometric_return(
    monthly_returns: np.ndarray,
) -> np.ndarray:
    """Annualize compounded monthly returns row by row."""
    months = monthly_returns.shape[1]

    gross = np.prod(
        1.0 + monthly_returns,
        axis=1,
    )

    return (
        np.power(
            gross,
            12.0 / float(months),
        )
        - 1.0
    )


def _annualized_sharpe(
    monthly_returns: np.ndarray,
) -> np.ndarray:
    """Calculate annualized zero-risk-free Sharpe from monthly returns."""
    means = np.mean(
        monthly_returns,
        axis=1,
    )

    volatility = np.std(
        monthly_returns,
        axis=1,
        ddof=1,
    )

    sharpe = np.full(
        len(means),
        np.nan,
        dtype=float,
    )

    valid = volatility > 0.0

    sharpe[valid] = means[valid] / volatility[valid] * np.sqrt(12.0)

    return sharpe


def _confidence_interval(
    values: np.ndarray,
) -> tuple[
    float,
    float,
]:
    """Return an equal-tailed confidence interval."""
    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = float(
        np.quantile(
            values,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            values,
            1.0 - alpha / 2.0,
        )
    )

    return (
        lower,
        upper,
    )


def _observed_metrics(
    values: np.ndarray,
    spy_values: np.ndarray,
) -> dict[str, float]:
    """Calculate observed monthly performance statistics."""
    annualized_return = float(
        _annualized_geometric_return(
            values.reshape(
                1,
                -1,
            )
        )[0]
    )

    spy_annualized_return = float(
        _annualized_geometric_return(
            spy_values.reshape(
                1,
                -1,
            )
        )[0]
    )

    sharpe = float(
        _annualized_sharpe(
            values.reshape(
                1,
                -1,
            )
        )[0]
    )

    excess = values - spy_values

    return {
        "observed_annualized_return": annualized_return,
        "observed_excess_annualized_return": (annualized_return - spy_annualized_return),
        "observed_sharpe": sharpe,
        "observed_positive_month_ratio": float(np.mean(values > 0.0)),
        "observed_outperform_spy_ratio": float(np.mean(excess > 0.0)),
        "observed_mean_excess_monthly_return": float(np.mean(excess)),
    }


def _bootstrap_indices(
    *,
    months: int,
    replications: int,
    seed: int,
) -> np.ndarray:
    """Generate paired monthly bootstrap indices."""
    rng = np.random.default_rng(seed)

    return rng.integers(
        0,
        months,
        size=(
            replications,
            months,
        ),
    )


def _bootstrap_method(
    strategy_values: np.ndarray,
    spy_values: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
]:
    """Bootstrap one strategy using the same sampled months as SPY."""
    sampled_strategy = strategy_values[bootstrap_indices]

    sampled_spy = spy_values[bootstrap_indices]

    strategy_cagr = _annualized_geometric_return(sampled_strategy)

    spy_cagr = _annualized_geometric_return(sampled_spy)

    excess_cagr = strategy_cagr - spy_cagr

    sharpe = _annualized_sharpe(sampled_strategy)

    positive_month_ratio = np.mean(
        sampled_strategy > 0.0,
        axis=1,
    )

    excess_months = sampled_strategy - sampled_spy

    outperform_spy_ratio = np.mean(
        excess_months > 0.0,
        axis=1,
    )

    mean_excess_monthly = np.mean(
        excess_months,
        axis=1,
    )

    finite_sharpe = sharpe[np.isfinite(sharpe)]

    if len(finite_sharpe) != len(sharpe):
        raise ValueError("Bootstrap generated non-finite Sharpe observations.")

    cagr_ci = _confidence_interval(strategy_cagr)

    excess_cagr_ci = _confidence_interval(excess_cagr)

    sharpe_ci = _confidence_interval(sharpe)

    mean_excess_ci = _confidence_interval(mean_excess_monthly)

    summary = {
        "bootstrap_replications": int(len(bootstrap_indices)),
        "bootstrap_months": int(bootstrap_indices.shape[1]),
        "annualized_return_bootstrap_mean": float(np.mean(strategy_cagr)),
        "annualized_return_ci_lower": cagr_ci[0],
        "annualized_return_ci_upper": cagr_ci[1],
        "excess_annualized_return_bootstrap_mean": float(np.mean(excess_cagr)),
        "excess_annualized_return_ci_lower": excess_cagr_ci[0],
        "excess_annualized_return_ci_upper": excess_cagr_ci[1],
        "probability_excess_annualized_return_positive": float(np.mean(excess_cagr > 0.0)),
        "sharpe_bootstrap_mean": float(np.mean(sharpe)),
        "sharpe_ci_lower": sharpe_ci[0],
        "sharpe_ci_upper": sharpe_ci[1],
        "positive_month_ratio_bootstrap_mean": float(np.mean(positive_month_ratio)),
        "outperform_spy_ratio_bootstrap_mean": float(np.mean(outperform_spy_ratio)),
        "mean_excess_monthly_return_bootstrap_mean": float(np.mean(mean_excess_monthly)),
        "mean_excess_monthly_return_ci_lower": mean_excess_ci[0],
        "mean_excess_monthly_return_ci_upper": mean_excess_ci[1],
        "probability_mean_excess_monthly_return_positive": float(
            np.mean(mean_excess_monthly > 0.0)
        ),
    }

    draws = {
        "annualized_return": strategy_cagr,
        "excess_annualized_return": excess_cagr,
        "sharpe": sharpe,
    }

    return (
        summary,
        draws,
    )


def _build_pairwise(
    method_draws: dict[
        str,
        dict[
            str,
            np.ndarray,
        ],
    ],
) -> pd.DataFrame:
    """Compare methods using paired bootstrap draws."""
    rows = []

    methods = sorted(method_draws)

    for left_index, left_method in enumerate(methods):
        for right_method in methods[left_index + 1 :]:
            cagr_difference = (
                method_draws[left_method]["annualized_return"]
                - method_draws[right_method]["annualized_return"]
            )

            sharpe_difference = (
                method_draws[left_method]["sharpe"] - method_draws[right_method]["sharpe"]
            )

            cagr_ci = _confidence_interval(cagr_difference)

            sharpe_ci = _confidence_interval(sharpe_difference)

            rows.append(
                {
                    "method_a": left_method,
                    "method_b": right_method,
                    "mean_cagr_difference_a_minus_b": float(np.mean(cagr_difference)),
                    "cagr_difference_ci_lower": cagr_ci[0],
                    "cagr_difference_ci_upper": cagr_ci[1],
                    "probability_cagr_a_greater_b": float(np.mean(cagr_difference > 0.0)),
                    "mean_sharpe_difference_a_minus_b": float(np.mean(sharpe_difference)),
                    "sharpe_difference_ci_lower": sharpe_ci[0],
                    "sharpe_difference_ci_upper": sharpe_ci[1],
                    "probability_sharpe_a_greater_b": float(np.mean(sharpe_difference > 0.0)),
                }
            )

    return pd.DataFrame(rows)


def _build_rank_stability(
    method_draws: dict[
        str,
        dict[
            str,
            np.ndarray,
        ],
    ],
) -> pd.DataFrame:
    """Estimate how often each method ranks first in bootstrap samples."""
    methods = sorted(method_draws)

    cagr_matrix = np.column_stack([method_draws[method]["annualized_return"] for method in methods])

    sharpe_matrix = np.column_stack([method_draws[method]["sharpe"] for method in methods])

    cagr_winner = np.argmax(
        cagr_matrix,
        axis=1,
    )

    sharpe_winner = np.argmax(
        sharpe_matrix,
        axis=1,
    )

    rows = []

    for index, method in enumerate(methods):
        rows.append(
            {
                "strategy_name": method,
                "probability_rank_1_by_annualized_return": float(np.mean(cagr_winner == index)),
                "probability_rank_1_by_sharpe": float(np.mean(sharpe_winner == index)),
            }
        )

    return pd.DataFrame(rows)


def _build_checks(
    monthly_panel: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    rank_stability: pd.DataFrame,
) -> pd.DataFrame:
    """Audit bootstrap robustness outputs."""
    probability_columns = [
        "probability_excess_annualized_return_positive",
        "probability_mean_excess_monthly_return_positive",
    ]

    summary_probabilities_valid = all(
        summary[column]
        .between(
            0.0,
            1.0,
        )
        .all()
        for column in probability_columns
    )

    pairwise_probabilities_valid = (
        pairwise[
            [
                "probability_cagr_a_greater_b",
                "probability_sharpe_a_greater_b",
            ]
        ]
        .apply(
            lambda values: values.between(
                0.0,
                1.0,
            ).all()
        )
        .all()
    )

    rank_return_sum = float(rank_stability["probability_rank_1_by_annualized_return"].sum())

    rank_sharpe_sum = float(rank_stability["probability_rank_1_by_sharpe"].sum())

    interval_order_violations = int(
        summary["annualized_return_ci_lower"].gt(summary["annualized_return_ci_upper"]).sum()
        + summary["excess_annualized_return_ci_lower"]
        .gt(summary["excess_annualized_return_ci_upper"])
        .sum()
        + summary["sharpe_ci_lower"].gt(summary["sharpe_ci_upper"]).sum()
    )

    checks = [
        (
            "minimum_months",
            int(len(monthly_panel) < 60),
            "Bootstrap analysis should contain at least 60 aligned months.",
        ),
        (
            "expected_methods",
            int(len(summary) != len(EXPECTED_METHODS)),
            "Bootstrap summary must contain all five portfolio methods.",
        ),
        (
            "expected_replications",
            int(summary["bootstrap_replications"].ne(BOOTSTRAP_REPLICATIONS).sum()),
            "Every method must use the configured number of bootstrap replications.",
        ),
        (
            "finite_summary_metrics",
            int(
                (
                    ~np.isfinite(
                        summary.select_dtypes(
                            include=[
                                np.number,
                            ]
                        ).to_numpy(dtype=float)
                    )
                ).sum()
            ),
            "Bootstrap summary metrics must be finite.",
        ),
        (
            "confidence_interval_order",
            interval_order_violations,
            "Bootstrap confidence-interval lower bounds must not exceed upper bounds.",
        ),
        (
            "valid_summary_probabilities",
            int(not summary_probabilities_valid),
            "Bootstrap strategy probabilities must lie between zero and one.",
        ),
        (
            "valid_pairwise_probabilities",
            int(not pairwise_probabilities_valid),
            "Pairwise bootstrap probabilities must lie between zero and one.",
        ),
        (
            "rank_probabilities_sum_to_one",
            int(
                not (
                    np.isclose(
                        rank_return_sum,
                        1.0,
                        atol=1.0e-10,
                    )
                    and np.isclose(
                        rank_sharpe_sum,
                        1.0,
                        atol=1.0e-10,
                    )
                )
            ),
            "Bootstrap rank-one probabilities must sum to one for each metric.",
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
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    rank_stability: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    months: int,
) -> str:
    """Build the paired monthly bootstrap report."""
    summary_display = summary.loc[
        :,
        [
            "strategy_name",
            "observed_annualized_return",
            "annualized_return_ci_lower",
            "annualized_return_ci_upper",
            "observed_excess_annualized_return",
            "excess_annualized_return_ci_lower",
            "excess_annualized_return_ci_upper",
            "probability_excess_annualized_return_positive",
            "observed_sharpe",
            "sharpe_ci_lower",
            "sharpe_ci_upper",
            "observed_positive_month_ratio",
            "observed_outperform_spy_ratio",
        ],
    ]

    return "\n".join(
        [
            "# Monthly Bootstrap Robustness",
            "",
            "## Methodology",
            "",
            (
                f"- `{BOOTSTRAP_REPLICATIONS:,}` paired bootstrap replications "
                f"over `{months}` aligned calendar months."
            ),
            (
                "- The same sampled month indices are used for every strategy "
                "and SPY, preserving cross-strategy comparability."
            ),
            (f"- Equal-tailed `{CONFIDENCE_LEVEL:.0%}` confidence intervals."),
            (
                "- Bootstrap Sharpe is calculated from monthly returns, "
                "annualized by multiplying mean/std by sqrt(12), with "
                "zero risk-free rate for this robustness diagnostic."
            ),
            (
                "- These intervals quantify sampling uncertainty inside the "
                "observed out-of-sample period; they are not guarantees of "
                "future performance."
            ),
            "",
            "## Strategy bootstrap summary",
            "",
            _to_markdown(summary_display),
            "",
            "## Pairwise comparison",
            "",
            _to_markdown(pairwise),
            "",
            "## Rank stability",
            "",
            _to_markdown(rank_stability),
            "",
            "## Readiness checks",
            "",
            _to_markdown(checks),
            "",
        ]
    )


def main() -> None:
    """Run paired monthly bootstrap analysis."""
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

    monthly_panel = _build_monthly_panel(
        daily,
        benchmark,
        benchmark_name=(evaluation_config.benchmark_name),
    )

    logger.info(
        "Running paired monthly bootstrap with %s replications.",
        f"{BOOTSTRAP_REPLICATIONS:,}",
    )

    indices = _bootstrap_indices(
        months=len(monthly_panel),
        replications=(BOOTSTRAP_REPLICATIONS),
        seed=BOOTSTRAP_SEED,
    )

    spy_values = monthly_panel[evaluation_config.benchmark_name].to_numpy(dtype=float)

    summary_rows = []
    method_draws = {}

    for method in EXPECTED_METHODS:
        strategy_values = monthly_panel[method].to_numpy(dtype=float)

        observed = _observed_metrics(
            strategy_values,
            spy_values,
        )

        bootstrap_summary, draws = _bootstrap_method(
            strategy_values,
            spy_values,
            indices,
        )

        summary_rows.append(
            {
                "strategy_name": method,
                **observed,
                **bootstrap_summary,
            }
        )

        method_draws[method] = draws

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            "observed_annualized_return",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    pairwise = _build_pairwise(method_draws)

    rank_stability = (
        _build_rank_stability(method_draws)
        .sort_values(
            "probability_rank_1_by_annualized_return",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    monthly_output = monthly_panel.reset_index().rename_axis(columns=None)

    checks = _build_checks(
        monthly_panel,
        summary,
        pairwise,
        rank_stability,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_csv(
        summary,
        BOOTSTRAP_SUMMARY_PATH,
    )

    _write_csv(
        pairwise,
        PAIRWISE_PATH,
    )

    _write_csv(
        rank_stability,
        RANK_STABILITY_PATH,
    )

    _write_csv(
        monthly_output,
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
            summary,
            pairwise,
            rank_stability,
            checks,
            months=len(monthly_panel),
        ),
        encoding="utf-8",
    )

    if failed_checks:
        raise ValueError(f"Monthly bootstrap validation failed with {failed_checks} failed checks.")

    logger.info("Monthly bootstrap robustness analysis completed.")

    print()

    print("Institutional Quant Equity Research Platform")

    print("Paired monthly bootstrap robustness")

    print("------------------------------------------------")

    print(f"methods: {len(EXPECTED_METHODS)}")

    print(f"months: {len(monthly_panel)}")

    print(f"bootstrap_replications: {BOOTSTRAP_REPLICATIONS}")

    print()

    print("Bootstrap strategy summary:")

    print(
        summary.loc[
            :,
            [
                "strategy_name",
                "observed_annualized_return",
                "annualized_return_ci_lower",
                "annualized_return_ci_upper",
                "observed_excess_annualized_return",
                "excess_annualized_return_ci_lower",
                "excess_annualized_return_ci_upper",
                "probability_excess_annualized_return_positive",
                "observed_sharpe",
                "sharpe_ci_lower",
                "sharpe_ci_upper",
                "observed_positive_month_ratio",
                "observed_outperform_spy_ratio",
            ],
        ].to_string(index=False)
    )

    print()

    print("Bootstrap rank stability:")

    print(rank_stability.to_string(index=False))

    print()

    print(f"readiness_checks: {len(checks)}")

    print(f"failed_readiness_checks: {failed_checks}")

    print()

    print(f"Summary table: {BOOTSTRAP_SUMMARY_PATH}")

    print(f"Pairwise table: {PAIRWISE_PATH}")

    print(f"Rank stability: {RANK_STABILITY_PATH}")

    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
