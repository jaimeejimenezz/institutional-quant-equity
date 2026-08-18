from __future__ import annotations

import pandas as pd

DISPLAY_LABELS: dict[str, str] = {
    "score_weighted": "Score Weighted",
    "top_n_equal_weight": "Top-N Equal Weight",
    "median_mad_de": "Median-MAD DE",
    "alpha_risk_turnover": "Alpha-Risk-Turnover",
    "cvar": "CVaR",
    "full_ensemble": "Full Ensemble",
    "no_fundamentals": "No Fundamentals",
    "no_momentum": "No Momentum",
}

CHART_LABELS: dict[str, str] = {
    "score_weighted": "Score Weighted",
    "top_n_equal_weight": "Top-N EW",
    "median_mad_de": "Median-MAD",
    "alpha_risk_turnover": "Alpha-Risk",
    "cvar": "CVaR",
}


def humanize_identifier(value: object) -> str:
    """Convert persisted identifiers into concise dashboard labels."""
    raw = str(value).strip()
    if raw in DISPLAY_LABELS:
        return DISPLAY_LABELS[raw]

    label = raw.replace("_", " ").strip().title()
    replacements = {
        "Lightgbm": "LightGBM",
        "Top N": "Top-N",
        "Adv": "ADV",
        "Ic": "IC",
        "Oos": "OOS",
        "Cvar": "CVaR",
    }
    for source, target in replacements.items():
        label = label.replace(source, target)
    return label


def strategy_display_label(value: object) -> str:
    """Return the canonical dashboard label for a strategy identifier."""
    return DISPLAY_LABELS.get(str(value), humanize_identifier(value))


def strategy_chart_label(value: object) -> str:
    """Return a shortened label suitable for dense charts."""
    return CHART_LABELS.get(str(value), strategy_display_label(value))


def robustness_headline(
    inventory: pd.DataFrame,
    coverage: pd.DataFrame,
    signal_bootstrap: pd.DataFrame,
) -> dict[str, object]:
    """Build headline robustness metrics from canonical persisted artifacts."""
    suite_status = inventory["suite_status"].astype(str).str.upper()
    passed_suites = int(suite_status.eq("PASS").sum())
    total_suites = int(len(inventory))

    coverage_status = coverage["status"].astype(str).str.upper()
    complete_dimensions = int(coverage_status.eq("COMPLETE").sum())
    deferred_dimensions = int(coverage_status.eq("DEFERRED_LIMITATION").sum())

    bootstrap_replications = 0
    oos_months = 0
    if not signal_bootstrap.empty:
        row = signal_bootstrap.iloc[0]
        bootstrap_replications = int(row["bootstrap_replications"])
        oos_months = int(row["months"])

    return {
        "passed_suites": passed_suites,
        "total_suites": total_suites,
        "complete_dimensions": complete_dimensions,
        "deferred_dimensions": deferred_dimensions,
        "bootstrap_replications": bootstrap_replications,
        "oos_months": oos_months,
    }


def coverage_table(coverage: pd.DataFrame) -> pd.DataFrame:
    """Return a compact robustness-coverage map."""
    result = coverage.loc[:, ["dimension", "category", "status"]].copy()
    result["dimension"] = result["dimension"].map(humanize_identifier)
    result["category"] = result["category"].map(humanize_identifier)
    result["status"] = result["status"].replace(
        {
            "COMPLETE": "Complete",
            "DEFERRED_LIMITATION": "Documented limitation",
        }
    )
    return result.rename(
        columns={
            "dimension": "Dimension",
            "category": "Category",
            "status": "Status",
        }
    )


def inventory_table(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return a compact validation-suite inventory."""
    result = inventory.loc[
        :,
        [
            "suite",
            "category",
            "checks",
            "passed_checks",
            "failed_checks",
            "suite_status",
        ],
    ].copy()
    result["suite"] = result["suite"].map(humanize_identifier)
    result["category"] = result["category"].map(humanize_identifier)
    return result.rename(
        columns={
            "suite": "Suite",
            "category": "Category",
            "checks": "Checks",
            "passed_checks": "Passed",
            "failed_checks": "Failed",
            "suite_status": "Status",
        }
    )


def strategy_bootstrap_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return portfolio bootstrap evidence with compact confidence intervals."""
    result = data.loc[
        :,
        [
            "strategy_name",
            "observed_annualized_return",
            "annualized_return_ci_lower",
            "annualized_return_ci_upper",
            "observed_sharpe",
            "sharpe_ci_lower",
            "sharpe_ci_upper",
            "probability_excess_annualized_return_positive",
        ],
    ].copy()
    result["Strategy"] = result["strategy_name"].map(strategy_display_label)
    result["Return"] = result["observed_annualized_return"]
    result["Return 95% CI"] = result.apply(
        lambda row: (
            f"{float(row['annualized_return_ci_lower']):.1%} to "
            f"{float(row['annualized_return_ci_upper']):.1%}"
        ),
        axis=1,
    )
    result["Sharpe"] = result["observed_sharpe"]
    result["Sharpe 95% CI"] = result.apply(
        lambda row: (
            f"{float(row['sharpe_ci_lower']):.2f} to "
            f"{float(row['sharpe_ci_upper']):.2f}"
        ),
        axis=1,
    )
    result["P(excess > 0)"] = result[
        "probability_excess_annualized_return_positive"
    ]
    return result[
        [
            "Strategy",
            "Return",
            "Return 95% CI",
            "Sharpe",
            "Sharpe 95% CI",
            "P(excess > 0)",
        ]
    ]


def feature_ablation_table(
    predictive: pd.DataFrame,
    economic: pd.DataFrame,
) -> pd.DataFrame:
    """Join predictive and economic feature-family ablation evidence."""
    left = predictive.loc[
        :,
        [
            "scenario",
            "mean_ic",
            "annualized_ic_ir",
            "mean_top_bottom_spread",
            "mean_top_quintile_precision",
        ],
    ].copy()

    right = economic.loc[
        :,
        [
            "strategy_name",
            "cagr",
            "sharpe_ratio",
            "cagr_difference_vs_full",
        ],
    ].copy()
    right["scenario"] = right["strategy_name"].replace({"full_ensemble": "full"})
    right = right.drop(columns="strategy_name")

    result = left.merge(right, on="scenario", how="left", validate="one_to_one")
    result["scenario"] = result["scenario"].map(humanize_identifier)

    return result.rename(
        columns={
            "scenario": "Scenario",
            "mean_ic": "Mean IC",
            "annualized_ic_ir": "IC IR",
            "mean_top_bottom_spread": "T-B spread",
            "mean_top_quintile_precision": "Top-Q hit",
            "cagr": "CAGR",
            "sharpe_ratio": "Sharpe",
            "cagr_difference_vs_full": "CAGR vs full",
        }
    )


def construction_ablation_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return compact portfolio-construction ablation evidence."""
    result = data.loc[
        :,
        [
            "strategy_name",
            "experiment",
            "is_controlled_baseline",
            "cagr",
            "sharpe_ratio",
            "mean_one_way_turnover",
            "maximum_sector_weight",
        ],
    ].copy()
    result["strategy_name"] = result["strategy_name"].map(humanize_identifier)
    result["experiment"] = result["experiment"].map(humanize_identifier)
    return result.rename(
        columns={
            "strategy_name": "Configuration",
            "experiment": "Experiment",
            "is_controlled_baseline": "Controlled",
            "cagr": "CAGR",
            "sharpe_ratio": "Sharpe",
            "mean_one_way_turnover": "Turnover",
            "maximum_sector_weight": "Max sector",
        }
    )


def universe_exclusion_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return compact frozen-universe exclusion diagnostics."""
    result = data.loc[
        :,
        [
            "strategy_name",
            "excluded_group",
            "mean_eligible_stocks",
            "cagr",
            "sharpe_ratio",
            "cagr_difference_vs_full",
        ],
    ].copy()
    result["strategy_name"] = result["strategy_name"].map(humanize_identifier)
    result["excluded_group"] = result["excluded_group"].fillna("None")
    return result.rename(
        columns={
            "strategy_name": "Scenario",
            "excluded_group": "Excluded group",
            "mean_eligible_stocks": "Eligible",
            "cagr": "CAGR",
            "sharpe_ratio": "Sharpe",
            "cagr_difference_vs_full": "CAGR vs full",
        }
    )
