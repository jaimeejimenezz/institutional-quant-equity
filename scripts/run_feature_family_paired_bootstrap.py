"""Run paired monthly bootstrap tests for feature-family economic ablations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.config import REPORTS_DIR
from quant_equity.logging_config import configure_logging

INPUT_DIR = REPORTS_DIR / "tables" / "feature_family_ablation"

MONTHLY_RETURNS_PATH = INPUT_DIR / "economic_monthly_returns.csv"

ECONOMIC_RESULTS_PATH = INPUT_DIR / "economic_comparison.csv"

BOOTSTRAP_SUMMARY_PATH = INPUT_DIR / "economic_paired_bootstrap.csv"

YEARLY_STABILITY_PATH = INPUT_DIR / "economic_yearly_stability.csv"

COST_DELTAS_PATH = INPUT_DIR / "economic_cost_deltas_vs_full.csv"

CHECKS_PATH = INPUT_DIR / "economic_bootstrap_checks.csv"

REPORT_PATH = (
    REPORTS_DIR / "robustness" / "feature_family_ablation" / "economic_paired_bootstrap.md"
)

BASELINE = "full_ensemble"

COMPARISONS = (
    "no_fundamentals",
    "no_momentum",
)

BENCHMARK = "spy_buy_and_hold"

EXPECTED_STRATEGIES = {
    BASELINE,
    *COMPARISONS,
    BENCHMARK,
}

EXPECTED_MONTHS = 77
EXPECTED_ROWS = EXPECTED_MONTHS * len(EXPECTED_STRATEGIES)

BOOTSTRAP_REPLICATIONS = 10_000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write one CSV output table."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=False,
    )


def _annualized_geometric_return(
    returns: np.ndarray,
) -> np.ndarray:
    """Annualize monthly compounded returns along the last axis."""
    if np.any(returns <= -1.0):
        raise ValueError("Monthly returns must be greater than -100%.")

    months = returns.shape[-1]

    return np.exp(np.log1p(returns).sum(axis=-1) * (12.0 / months)) - 1.0


def _annualized_monthly_sharpe(
    returns: np.ndarray,
) -> np.ndarray:
    """Compute annualized Sharpe from monthly returns."""
    mean_return = returns.mean(axis=-1)

    std_return = returns.std(
        axis=-1,
        ddof=1,
    )

    return np.divide(
        mean_return * np.sqrt(12.0),
        std_return,
        out=np.full_like(
            mean_return,
            np.nan,
            dtype=float,
        ),
        where=(std_return > 0.0),
    )


def _percentile_interval(
    values: np.ndarray,
) -> tuple[
    float,
    float,
]:
    """Return a two-sided percentile confidence interval."""
    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = float(
        np.nanquantile(
            values,
            alpha / 2.0,
        )
    )

    upper = float(
        np.nanquantile(
            values,
            1.0 - alpha / 2.0,
        )
    )

    return (
        lower,
        upper,
    )


def _load_monthly_returns() -> pd.DataFrame:
    """Load and validate the common monthly return panel."""
    if not MONTHLY_RETURNS_PATH.exists():
        raise FileNotFoundError(f"Monthly economic return table not found: {MONTHLY_RETURNS_PATH}")

    data = pd.read_csv(MONTHLY_RETURNS_PATH)

    required = {
        "month",
        "strategy_name",
        "monthly_return",
    }

    missing = sorted(required.difference(data.columns))

    if missing:
        raise ValueError("Monthly return table is missing columns: " + ", ".join(missing) + ".")

    data = data.loc[
        :,
        [
            "month",
            "strategy_name",
            "monthly_return",
        ],
    ].copy()

    data["month"] = pd.to_datetime(data["month"]).dt.normalize()

    data["strategy_name"] = data["strategy_name"].astype(str)

    data["monthly_return"] = pd.to_numeric(
        data["monthly_return"],
        errors="coerce",
    )

    if data["monthly_return"].isna().any():
        raise ValueError("Monthly return table contains missing or invalid returns.")

    if not np.isfinite(data["monthly_return"].to_numpy(dtype=float)).all():
        raise ValueError("Monthly return table contains non-finite returns.")

    if data["monthly_return"].le(-1.0).any():
        raise ValueError("Monthly return table contains a return at or below -100%.")

    duplicate_keys = int(
        data.duplicated(
            [
                "month",
                "strategy_name",
            ]
        ).sum()
    )

    if duplicate_keys:
        raise ValueError(
            f"Monthly return table contains {duplicate_keys} duplicate month-strategy keys."
        )

    observed_strategies = set(data["strategy_name"].unique())

    if observed_strategies != EXPECTED_STRATEGIES:
        raise ValueError(
            f"Unexpected strategy set in monthly returns: {sorted(observed_strategies)}."
        )

    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} monthly rows; found {len(data)}.")

    counts = data.groupby("strategy_name")["month"].nunique()

    invalid_counts = int(counts.ne(EXPECTED_MONTHS).sum())

    if invalid_counts:
        raise ValueError(f"Every strategy must contain exactly {EXPECTED_MONTHS} months.")

    return data.sort_values(
        [
            "month",
            "strategy_name",
        ]
    ).reset_index(drop=True)


def _pivot_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create the paired month-by-strategy return matrix."""
    pivot = data.pivot(
        index="month",
        columns="strategy_name",
        values="monthly_return",
    ).sort_index()

    if pivot.isna().any().any():
        raise ValueError("Monthly return panel is not fully paired across strategies.")

    if len(pivot) != EXPECTED_MONTHS:
        raise ValueError(
            f"Paired panel should contain {EXPECTED_MONTHS} months; found {len(pivot)}."
        )

    return pivot


def _paired_bootstrap(
    pivot: pd.DataFrame,
    scenario: str,
) -> dict[
    str,
    Any,
]:
    """Bootstrap paired months for one ablation versus FULL."""
    baseline = pivot[BASELINE].to_numpy(dtype=float)

    candidate = pivot[scenario].to_numpy(dtype=float)

    paired_difference = candidate - baseline

    observed_mean_difference = float(paired_difference.mean())

    observed_cagr_difference = float(
        _annualized_geometric_return(
            candidate[
                np.newaxis,
                :,
            ]
        )[0]
        - _annualized_geometric_return(
            baseline[
                np.newaxis,
                :,
            ]
        )[0]
    )

    observed_sharpe_difference = float(
        _annualized_monthly_sharpe(
            candidate[
                np.newaxis,
                :,
            ]
        )[0]
        - _annualized_monthly_sharpe(
            baseline[
                np.newaxis,
                :,
            ]
        )[0]
    )

    rng = np.random.default_rng(RANDOM_SEED)

    sample_indices = rng.integers(
        0,
        EXPECTED_MONTHS,
        size=(
            BOOTSTRAP_REPLICATIONS,
            EXPECTED_MONTHS,
        ),
    )

    sampled_baseline = baseline[sample_indices]

    sampled_candidate = candidate[sample_indices]

    sampled_difference = sampled_candidate - sampled_baseline

    mean_difference_bootstrap = sampled_difference.mean(axis=1)

    cagr_difference_bootstrap = _annualized_geometric_return(
        sampled_candidate
    ) - _annualized_geometric_return(sampled_baseline)

    sharpe_difference_bootstrap = _annualized_monthly_sharpe(
        sampled_candidate
    ) - _annualized_monthly_sharpe(sampled_baseline)

    (
        mean_ci_low,
        mean_ci_high,
    ) = _percentile_interval(mean_difference_bootstrap)

    (
        cagr_ci_low,
        cagr_ci_high,
    ) = _percentile_interval(cagr_difference_bootstrap)

    finite_sharpe = sharpe_difference_bootstrap[np.isfinite(sharpe_difference_bootstrap)]

    if finite_sharpe.size == 0:
        raise ValueError(f"No finite bootstrap Sharpe differences for {scenario}.")

    (
        sharpe_ci_low,
        sharpe_ci_high,
    ) = _percentile_interval(finite_sharpe)

    return {
        "scenario": scenario,
        "months": EXPECTED_MONTHS,
        "bootstrap_replications": (BOOTSTRAP_REPLICATIONS),
        "observed_mean_monthly_return_difference": (observed_mean_difference),
        "mean_monthly_difference_ci_low": (mean_ci_low),
        "mean_monthly_difference_ci_high": (mean_ci_high),
        "probability_mean_monthly_difference_gt_zero": float(
            np.mean(mean_difference_bootstrap > 0.0)
        ),
        "observed_annualized_geometric_return_difference": (observed_cagr_difference),
        "annualized_geometric_difference_ci_low": (cagr_ci_low),
        "annualized_geometric_difference_ci_high": (cagr_ci_high),
        "probability_annualized_geometric_difference_gt_zero": float(
            np.mean(cagr_difference_bootstrap > 0.0)
        ),
        "observed_annualized_monthly_sharpe_difference": (observed_sharpe_difference),
        "annualized_monthly_sharpe_difference_ci_low": (sharpe_ci_low),
        "annualized_monthly_sharpe_difference_ci_high": (sharpe_ci_high),
        "probability_annualized_monthly_sharpe_difference_gt_zero": float(
            np.mean(finite_sharpe > 0.0)
        ),
        "candidate_beats_full_month_frequency": float(np.mean(paired_difference > 0.0)),
        "candidate_ties_full_month_frequency": float(
            np.mean(
                np.isclose(
                    paired_difference,
                    0.0,
                    atol=1.0e-15,
                    rtol=0.0,
                )
            )
        ),
    }


def _yearly_stability(
    pivot: pd.DataFrame,
) -> pd.DataFrame:
    """Measure actual calendar-year performance stability versus FULL."""
    monthly = pivot.copy()

    monthly["year"] = monthly.index.year

    rows = []

    for scenario in COMPARISONS:
        for year, group in monthly.groupby(
            "year",
            sort=True,
        ):
            baseline = group[BASELINE].to_numpy(dtype=float)

            candidate = group[scenario].to_numpy(dtype=float)

            full_return = float(np.prod(1.0 + baseline) - 1.0)

            candidate_return = float(np.prod(1.0 + candidate) - 1.0)

            difference = candidate - baseline

            rows.append(
                {
                    "scenario": scenario,
                    "year": int(year),
                    "months": int(len(group)),
                    "full_compounded_return": (full_return),
                    "candidate_compounded_return": (candidate_return),
                    "compounded_return_difference_vs_full": float(candidate_return - full_return),
                    "mean_monthly_return_difference_vs_full": float(difference.mean()),
                    "candidate_beats_full_month_frequency": float(np.mean(difference > 0.0)),
                    "candidate_beats_full_year": bool(candidate_return > full_return),
                }
            )

    return pd.DataFrame(rows)


def _cost_deltas() -> pd.DataFrame:
    """Summarize total cost and trading differences versus FULL."""
    if not ECONOMIC_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Economic comparison table not found: {ECONOMIC_RESULTS_PATH}")

    data = pd.read_csv(ECONOMIC_RESULTS_PATH)

    required = {
        "strategy_name",
        "total_transaction_cost",
        "total_traded_notional",
        "mean_one_way_turnover",
        "effective_cost_bps",
    }

    missing = sorted(required.difference(data.columns))

    if missing:
        raise ValueError(
            "Economic comparison table is missing columns: " + ", ".join(missing) + "."
        )

    indexed = data.set_index("strategy_name")

    if BASELINE not in indexed.index:
        raise ValueError("FULL economic baseline is missing.")

    full = indexed.loc[BASELINE]

    rows = []

    for scenario in COMPARISONS:
        if scenario not in indexed.index:
            raise ValueError(f"Economic result missing scenario: {scenario}.")

        current = indexed.loc[scenario]

        rows.append(
            {
                "scenario": scenario,
                "total_transaction_cost": float(current["total_transaction_cost"]),
                "transaction_cost_difference_vs_full": float(
                    current["total_transaction_cost"] - full["total_transaction_cost"]
                ),
                "total_traded_notional": float(current["total_traded_notional"]),
                "traded_notional_difference_vs_full": float(
                    current["total_traded_notional"] - full["total_traded_notional"]
                ),
                "mean_one_way_turnover": float(current["mean_one_way_turnover"]),
                "turnover_difference_vs_full": float(
                    current["mean_one_way_turnover"] - full["mean_one_way_turnover"]
                ),
                "effective_cost_bps": float(current["effective_cost_bps"]),
                "effective_cost_bps_difference_vs_full": float(
                    current["effective_cost_bps"] - full["effective_cost_bps"]
                ),
            }
        )

    return pd.DataFrame(rows)


def _build_checks(
    monthly_data: pd.DataFrame,
    pivot: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    """Audit paired bootstrap inputs and outputs."""
    checks = [
        (
            "expected_monthly_rows",
            int(len(monthly_data) != EXPECTED_ROWS),
            (f"Monthly return table must contain {EXPECTED_ROWS} rows."),
        ),
        (
            "expected_months",
            int(len(pivot) != EXPECTED_MONTHS),
            (f"Paired return panel must contain {EXPECTED_MONTHS} common months."),
        ),
        (
            "expected_strategies",
            int(set(pivot.columns) != EXPECTED_STRATEGIES),
            ("Paired panel must contain FULL, both feature ablations and SPY."),
        ),
        (
            "complete_pairs",
            int(pivot.isna().sum().sum()),
            ("Every strategy must have a return for every paired calendar month."),
        ),
        (
            "bootstrap_scenarios",
            int(set(bootstrap["scenario"]) != set(COMPARISONS)),
            ("Bootstrap output must contain both feature ablations."),
        ),
        (
            "finite_bootstrap_statistics",
            int(
                (
                    ~np.isfinite(bootstrap.select_dtypes(include=[np.number]).to_numpy(dtype=float))
                ).sum()
            ),
            ("All stored bootstrap statistics must be finite."),
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
    """Format one Markdown table value."""
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


def _write_report(
    bootstrap: pd.DataFrame,
    yearly: pd.DataFrame,
    cost_deltas: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    """Write the paired monthly bootstrap report."""
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Feature-Family Economic Paired Bootstrap",
                "",
                "## Methodology",
                "",
                ("- Uses the 77 common calendar months from the net economic backtests."),
                (
                    "- Each bootstrap draw resamples months with replacement "
                    "and keeps FULL and both feature ablations paired on the "
                    "same sampled month."
                ),
                (
                    f"- Uses {BOOTSTRAP_REPLICATIONS:,} paired bootstrap "
                    f"replications with random seed {RANDOM_SEED}."
                ),
                ("- Confidence intervals are two-sided 95% percentile bootstrap intervals."),
                (
                    "- Annualized geometric return is calculated from monthly "
                    "compounding; annualized monthly Sharpe uses sqrt(12) and "
                    "a zero monthly risk-free rate."
                ),
                (
                    "- These bootstrap Sharpe statistics are monthly-return "
                    "statistics and are intentionally distinct from the daily "
                    "Sharpe reported by the main performance engine."
                ),
                "",
                "## Paired bootstrap summary",
                "",
                _to_markdown(bootstrap),
                "",
                "## Calendar-year stability",
                "",
                _to_markdown(yearly),
                "",
                "## Trading-cost deltas versus FULL",
                "",
                _to_markdown(cost_deltas),
                "",
                "## Checks",
                "",
                _to_markdown(checks),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run paired monthly bootstrap robustness analysis."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    monthly_data = _load_monthly_returns()

    pivot = _pivot_returns(monthly_data)

    bootstrap = pd.DataFrame(
        [
            _paired_bootstrap(
                pivot,
                scenario,
            )
            for scenario in COMPARISONS
        ]
    )

    yearly = _yearly_stability(pivot)

    cost_deltas = _cost_deltas()

    checks = _build_checks(
        monthly_data,
        pivot,
        bootstrap,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    _write_csv(
        bootstrap,
        BOOTSTRAP_SUMMARY_PATH,
    )

    _write_csv(
        yearly,
        YEARLY_STABILITY_PATH,
    )

    _write_csv(
        cost_deltas,
        COST_DELTAS_PATH,
    )

    _write_csv(
        checks,
        CHECKS_PATH,
    )

    _write_report(
        bootstrap,
        yearly,
        cost_deltas,
        checks,
    )

    if failed_checks:
        print()
        print(checks.to_string(index=False))

        raise ValueError(
            f"Feature-family paired bootstrap validation failed with {failed_checks} failed checks."
        )

    logger.info("Feature-family paired monthly bootstrap completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Feature-family paired monthly bootstrap")
    print("------------------------------------------------")
    print(f"months: {EXPECTED_MONTHS}")
    print(f"bootstrap_replications: {BOOTSTRAP_REPLICATIONS}")
    print(f"confidence_level: {CONFIDENCE_LEVEL:.2f}")

    print()
    print("Paired bootstrap summary:")
    print(bootstrap.to_string(index=False))

    print()
    print("Calendar-year stability:")
    print(yearly.to_string(index=False))

    print()
    print("Trading-cost deltas versus FULL:")
    print(cost_deltas.to_string(index=False))

    print()
    print(f"readiness_checks: {len(checks)}")
    print(f"failed_readiness_checks: {failed_checks}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
