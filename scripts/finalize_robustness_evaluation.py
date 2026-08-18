"""Audit robustness artifacts and write the final research summary."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.config import REPORTS_DIR
from quant_equity.logging_config import configure_logging

TABLES_DIR = REPORTS_DIR / "tables"
FEATURE_FAMILY_DIR = TABLES_DIR / "feature_family_ablation"

CHECK_INVENTORY_PATH = TABLES_DIR / "robustness_evaluation_check_inventory.csv"

COVERAGE_PATH = TABLES_DIR / "robustness_evaluation_coverage.csv"

REPORT_PATH = REPORTS_DIR / "robustness" / "robustness_evaluation_summary.md"

FEATURE_FAMILY_ECONOMIC_PATH = FEATURE_FAMILY_DIR / "economic_comparison.csv"

FEATURE_FAMILY_BOOTSTRAP_PATH = FEATURE_FAMILY_DIR / "economic_paired_bootstrap.csv"

FEATURE_FAMILY_YEARLY_PATH = FEATURE_FAMILY_DIR / "economic_yearly_stability.csv"

FEATURE_FAMILY_COST_PATH = FEATURE_FAMILY_DIR / "economic_cost_deltas_vs_full.csv"

ENSEMBLE_ABLATION_PATH = TABLES_DIR / "robustness_ensemble_component_ablation.csv"

CONSTRUCTION_ABLATION_PATH = TABLES_DIR / "robustness_portfolio_construction_ablation.csv"


@dataclass(frozen=True)
class CheckSuite:
    """Describe one required robustness validation artifact."""

    suite: str
    category: str
    path: Path


CHECK_SUITES = (
    CheckSuite(
        "temporal_robustness",
        "temporal",
        TABLES_DIR / "robustness_temporal_checks.csv",
    ),
    CheckSuite(
        "transaction_cost_sensitivity",
        "parameters",
        TABLES_DIR / "transaction_cost_sensitivity_checks.csv",
    ),
    CheckSuite(
        "monthly_portfolio_bootstrap",
        "statistics",
        TABLES_DIR / "bootstrap_robustness_checks.csv",
    ),
    CheckSuite(
        "final_signal_statistics",
        "statistics",
        TABLES_DIR / "robustness_final_signal_checks.csv",
    ),
    CheckSuite(
        "portfolio_parameter_sensitivity",
        "parameters",
        TABLES_DIR / "robustness_portfolio_parameter_checks.csv",
    ),
    CheckSuite(
        "rebalance_frequency_sensitivity",
        "parameters",
        TABLES_DIR / "robustness_rebalance_frequency_checks.csv",
    ),
    CheckSuite(
        "rolling_evaluation_windows",
        "parameters",
        TABLES_DIR / "robustness_rolling_window_checks.csv",
    ),
    CheckSuite(
        "prediction_horizon_robustness",
        "parameters",
        TABLES_DIR / "robustness_prediction_horizon_checks.csv",
    ),
    CheckSuite(
        "universe_exclusion_robustness",
        "universe",
        TABLES_DIR / "robustness_universe_exclusion_checks.csv",
    ),
    CheckSuite(
        "ensemble_component_ablation",
        "ablation",
        TABLES_DIR / "robustness_ensemble_component_ablation_checks.csv",
    ),
    CheckSuite(
        "portfolio_construction_ablation",
        "ablation",
        TABLES_DIR / "robustness_portfolio_construction_ablation_checks.csv",
    ),
    CheckSuite(
        "feature_family_contract",
        "ablation",
        TABLES_DIR / "robustness_feature_family_contract_checks.csv",
    ),
    CheckSuite(
        "feature_family_predictive_keys",
        "ablation",
        FEATURE_FAMILY_DIR / "official_predictive_key_checks.csv",
    ),
    CheckSuite(
        "feature_family_predictive_full_reference",
        "ablation",
        FEATURE_FAMILY_DIR / "official_predictive_frozen_full_checks.csv",
    ),
    CheckSuite(
        "feature_family_predictive_candidate_match",
        "ablation",
        FEATURE_FAMILY_DIR / "official_predictive_full_candidate_match.csv",
    ),
    CheckSuite(
        "feature_family_economic_backtest",
        "ablation",
        FEATURE_FAMILY_DIR / "economic_checks.csv",
    ),
    CheckSuite(
        "feature_family_economic_bootstrap",
        "ablation",
        FEATURE_FAMILY_DIR / "economic_bootstrap_checks.csv",
    ),
)


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


def _audit_check_suite(
    suite: CheckSuite,
) -> dict[str, object]:
    """Summarize one persisted readiness-check table."""
    if not suite.path.exists():
        return {
            "suite": suite.suite,
            "category": suite.category,
            "artifact": str(suite.path),
            "artifact_exists": False,
            "checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "suite_status": "MISSING",
        }

    data = pd.read_csv(suite.path)

    if data.empty:
        return {
            "suite": suite.suite,
            "category": suite.category,
            "artifact": str(suite.path),
            "artifact_exists": True,
            "checks": 0,
            "passed_checks": 0,
            "failed_checks": 1,
            "suite_status": "INVALID",
        }

    if suite.suite == "feature_family_predictive_candidate_match":
        required_columns = {
            "model_name",
            "matches_frozen_full",
            "maximum_absolute_difference",
        }

        missing = required_columns.difference(data.columns)

        if missing:
            return {
                "suite": suite.suite,
                "category": suite.category,
                "artifact": str(suite.path),
                "artifact_exists": True,
                "checks": 1,
                "passed_checks": 0,
                "failed_checks": 1,
                "suite_status": "INVALID",
            }

        matches = (
            data["matches_frozen_full"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

        finite_differences = np.isfinite(
            pd.to_numeric(
                data["maximum_absolute_difference"],
                errors="coerce",
            ).to_numpy(dtype=float)
        ).all()

        passed = bool(matches.sum() == 1 and finite_differences)

        return {
            "suite": suite.suite,
            "category": suite.category,
            "artifact": str(suite.path),
            "artifact_exists": True,
            "checks": 1,
            "passed_checks": int(passed),
            "failed_checks": int(not passed),
            "suite_status": ("PASS" if passed else "FAIL"),
        }

    if "status" not in data.columns:
        return {
            "suite": suite.suite,
            "category": suite.category,
            "artifact": str(suite.path),
            "artifact_exists": True,
            "checks": int(len(data)),
            "passed_checks": 0,
            "failed_checks": int(
                max(
                    len(data),
                    1,
                )
            ),
            "suite_status": "INVALID",
        }

    status = data["status"].astype("string").str.strip().str.upper()

    passed = int(status.eq("PASS").sum())

    failed = int((~status.eq("PASS")).sum())

    return {
        "suite": suite.suite,
        "category": suite.category,
        "artifact": str(suite.path),
        "artifact_exists": True,
        "checks": int(len(data)),
        "passed_checks": passed,
        "failed_checks": failed,
        "suite_status": ("PASS" if failed == 0 else "FAIL"),
    }


def _status_for_suites(
    inventory: pd.DataFrame,
    suite_names: tuple[
        str,
        ...,
    ],
) -> str:
    """Return aggregate status for a set of validation suites."""
    selected = inventory.loc[inventory["suite"].isin(suite_names)]

    if len(selected) != len(suite_names):
        return "MISSING"

    statuses = set(selected["suite_status"])

    if statuses == {
        "PASS",
    }:
        return "COMPLETE"

    if "MISSING" in statuses or "INVALID" in statuses:
        return "MISSING"

    return "FAILED"


def _build_coverage(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Map persisted validations to the planned robustness dimensions."""
    rows = [
        {
            "dimension": "calendar_years_and_market_regimes",
            "category": "temporal",
            "status": _status_for_suites(
                inventory,
                ("temporal_robustness",),
            ),
            "note": (
                "Calendar years, COVID-related periods, "
                "up/down markets and high-volatility conditions."
            ),
        },
        {
            "dimension": "transaction_cost_assumptions",
            "category": "parameters",
            "status": _status_for_suites(
                inventory,
                ("transaction_cost_sensitivity",),
            ),
            "note": ("Linear cost scenarios plus the liquidity-dependent execution model."),
        },
        {
            "dimension": "top_n_and_security_caps",
            "category": "parameters",
            "status": _status_for_suites(
                inventory,
                ("portfolio_parameter_sensitivity",),
            ),
            "note": ("Top-N and maximum-security-weight sensitivity."),
        },
        {
            "dimension": "rebalance_frequency",
            "category": "parameters",
            "status": _status_for_suites(
                inventory,
                ("rebalance_frequency_sensitivity",),
            ),
            "note": ("Monthly versus calendar-quarter rebalancing."),
        },
        {
            "dimension": "evaluation_windows",
            "category": "parameters",
            "status": _status_for_suites(
                inventory,
                ("rolling_evaluation_windows",),
            ),
            "note": ("Overlapping 12-, 24- and 36-month evaluation windows."),
        },
        {
            "dimension": "return_horizons",
            "category": "parameters",
            "status": _status_for_suites(
                inventory,
                ("prediction_horizon_robustness",),
            ),
            "note": ("Frozen signal evaluated at 10-, 21- and 42-session realized horizons."),
        },
        {
            "dimension": "frozen_universe_exclusions",
            "category": "universe",
            "status": _status_for_suites(
                inventory,
                ("universe_exclusion_robustness",),
            ),
            "note": ("Technology, leave-one-sector-out and largest-liquidity exclusions."),
        },
        {
            "dimension": "expanded_universe",
            "category": "universe",
            "status": "DEFERRED_LIMITATION",
            "note": (
                "A genuine expanded-universe experiment "
                "requires adding securities and rebuilding "
                "the upstream point-in-time pipeline."
            ),
        },
        {
            "dimension": "portfolio_return_bootstrap",
            "category": "statistics",
            "status": _status_for_suites(
                inventory,
                ("monthly_portfolio_bootstrap",),
            ),
            "note": ("Paired monthly bootstrap for the portfolio methods."),
        },
        {
            "dimension": "signal_ic_spread_sector_statistics",
            "category": "statistics",
            "status": _status_for_suites(
                inventory,
                ("final_signal_statistics",),
            ),
            "note": ("IC, spread, yearly and sector stability diagnostics."),
        },
        {
            "dimension": "no_fundamentals_and_no_momentum",
            "category": "ablation",
            "status": _status_for_suites(
                inventory,
                (
                    "feature_family_contract",
                    "feature_family_predictive_keys",
                    "feature_family_predictive_full_reference",
                    "feature_family_predictive_candidate_match",
                    "feature_family_economic_backtest",
                    "feature_family_economic_bootstrap",
                ),
            ),
            "note": (
                "Walk-forward feature-family ablations with predictive and economic comparison."
            ),
        },
        {
            "dimension": "no_lightgbm",
            "category": "ablation",
            "status": _status_for_suites(
                inventory,
                ("ensemble_component_ablation",),
            ),
            "note": ("LightGBM Ranker removed from the frozen ensemble."),
        },
        {
            "dimension": "no_optimization_no_sector_control_no_turnover_penalty",
            "category": "ablation",
            "status": _status_for_suites(
                inventory,
                ("portfolio_construction_ablation",),
            ),
            "note": ("Non-optimized reference, sector-cap ablation and turnover-penalty ablation."),
        },
    ]

    return pd.DataFrame(rows)


def _safe_read(
    path: Path,
) -> pd.DataFrame:
    """Read an optional CSV artifact."""
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def _feature_family_conclusions(
    economic: pd.DataFrame,
    bootstrap: pd.DataFrame,
    yearly: pd.DataFrame,
    cost_deltas: pd.DataFrame,
) -> list[str]:
    """Build cautious data-driven conclusions for feature-family ablations."""
    lines: list[str] = []

    required_economic = {
        "strategy_name",
        "cagr",
        "cagr_difference_vs_full",
        "sharpe_ratio",
        "sharpe_difference_vs_full",
        "mean_one_way_turnover",
    }

    required_bootstrap = {
        "scenario",
        "observed_annualized_geometric_return_difference",
        "annualized_geometric_difference_ci_low",
        "annualized_geometric_difference_ci_high",
        "probability_annualized_geometric_difference_gt_zero",
        "observed_annualized_monthly_sharpe_difference",
        "annualized_monthly_sharpe_difference_ci_low",
        "annualized_monthly_sharpe_difference_ci_high",
        "probability_annualized_monthly_sharpe_difference_gt_zero",
        "candidate_beats_full_month_frequency",
    }

    if not required_economic.issubset(economic.columns) or not required_bootstrap.issubset(
        bootstrap.columns
    ):
        return [
            (
                "Detailed feature-family interpretation "
                "was not generated because one or more "
                "expected result columns are missing."
            )
        ]

    for scenario in (
        "no_fundamentals",
        "no_momentum",
    ):
        economic_row = economic.loc[economic["strategy_name"].eq(scenario)]

        bootstrap_row = bootstrap.loc[bootstrap["scenario"].eq(scenario)]

        if len(economic_row) != 1 or len(bootstrap_row) != 1:
            lines.append(
                
                    f"`{scenario}` could not be summarized "
                    "because its result row is missing or duplicated."
                
            )
            continue

        econ = economic_row.iloc[0]

        boot = bootstrap_row.iloc[0]

        annualized_delta = float(boot["observed_annualized_geometric_return_difference"])

        ci_low = float(boot["annualized_geometric_difference_ci_low"])

        ci_high = float(boot["annualized_geometric_difference_ci_high"])

        probability = float(boot["probability_annualized_geometric_difference_gt_zero"])

        monthly_win_rate = float(boot["candidate_beats_full_month_frequency"])

        turnover_delta = np.nan

        if not cost_deltas.empty and {
            "scenario",
            "turnover_difference_vs_full",
        }.issubset(cost_deltas.columns):
            turnover_row = cost_deltas.loc[cost_deltas["scenario"].eq(scenario)]

            if len(turnover_row) == 1:
                turnover_delta = float(turnover_row["turnover_difference_vs_full"].iloc[0])

        yearly_wins = None
        yearly_total = None

        if not yearly.empty and {
            "scenario",
            "candidate_beats_full_year",
        }.issubset(yearly.columns):
            scenario_yearly = yearly.loc[yearly["scenario"].eq(scenario)]

            if not scenario_yearly.empty:
                yearly_total = int(len(scenario_yearly))

                yearly_wins = int(scenario_yearly["candidate_beats_full_year"].astype(bool).sum())

        interval_includes_zero = ci_low <= 0.0 <= ci_high

        significance_text = (
            "the two-sided 95% interval includes zero"
            if interval_includes_zero
            else "the two-sided 95% interval excludes zero"
        )

        yearly_text = (
            (f"; calendar-year blocks won: {yearly_wins}/{yearly_total}")
            if (yearly_wins is not None and yearly_total is not None)
            else ""
        )

        turnover_text = (
            (f"; turnover delta: {turnover_delta:+.4f}") if np.isfinite(turnover_delta) else ""
        )

        lines.append(
            
                f"`{scenario}`: observed economic CAGR delta "
                f"{float(econ['cagr_difference_vs_full']):+.4f}; "
                f"paired annualized geometric-return delta "
                f"{annualized_delta:+.4f}, 95% CI "
                f"[{ci_low:+.4f}, {ci_high:+.4f}], "
                f"P(delta > 0)={probability:.4f}; "
                f"monthly win frequency={monthly_win_rate:.4f}; "
                f"{significance_text}"
                f"{yearly_text}"
                f"{turnover_text}."
            
        )

    lines.append(
        
            "Ablation outcomes are diagnostics, not permission "
            "to retune the frozen production specification using "
            "the same out-of-sample period. Any specification "
            "change should be validated on new untouched data."
        
    )

    return lines


def _build_report(
    inventory: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    economic: pd.DataFrame,
    bootstrap: pd.DataFrame,
    yearly: pd.DataFrame,
    cost_deltas: pd.DataFrame,
    ensemble_ablation: pd.DataFrame,
    construction_ablation: pd.DataFrame,
) -> str:
    """Build the final robustness-evaluation summary."""
    blocking = coverage.loc[
        coverage["status"].isin(
            [
                "MISSING",
                "FAILED",
            ]
        )
    ]

    if blocking.empty:
        overall_status = "COMPLETE WITH DOCUMENTED LIMITATION"
    else:
        overall_status = "INCOMPLETE"

    lines = [
        "# Robustness Evaluation Summary",
        "",
        "## Status",
        "",
        f"**{overall_status}**",
        "",
        "## Validation inventory",
        "",
        _to_markdown(
            inventory.loc[
                :,
                [
                    "category",
                    "suite",
                    "suite_status",
                    "checks",
                    "passed_checks",
                    "failed_checks",
                    "artifact",
                ],
            ]
        ),
        "",
        "## Coverage",
        "",
        _to_markdown(coverage),
        "",
        "## Feature-family interpretation",
        "",
    ]

    lines.extend(
        "- " + value
        for value in _feature_family_conclusions(
            economic,
            bootstrap,
            yearly,
            cost_deltas,
        )
    )

    if not ensemble_ablation.empty:
        lines.extend(
            [
                "",
                "## Ensemble-component ablation",
                "",
                _to_markdown(ensemble_ablation),
            ]
        )

    if not construction_ablation.empty:
        lines.extend(
            [
                "",
                "## Portfolio-construction ablation",
                "",
                _to_markdown(construction_ablation),
            ]
        )

    lines.extend(
        [
            "",
            "## Documented limitation",
            "",
            (
                "A genuine expanded-universe experiment is not "
                "claimed as completed. It requires additional "
                "securities and a rebuilt point-in-time upstream "
                "data pipeline so that the comparison remains "
                "methodologically valid."
            ),
            "",
            "## Research interpretation rule",
            "",
            (
                "The objective of robustness analysis is not to "
                "search for the best-looking historical variant. "
                "The frozen specification remains the reference, "
                "and alternative specifications are interpreted "
                "as evidence about stability, dependence and "
                "possible simplification."
            ),
            "",
        ]
    )

    if not blocking.empty:
        lines.extend(
            [
                "## Blocking coverage items",
                "",
                _to_markdown(blocking),
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    """Audit all persisted robustness validations and finalize the summary."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    inventory = pd.DataFrame([_audit_check_suite(suite) for suite in CHECK_SUITES])

    coverage = _build_coverage(inventory)

    economic = _safe_read(FEATURE_FAMILY_ECONOMIC_PATH)

    bootstrap = _safe_read(FEATURE_FAMILY_BOOTSTRAP_PATH)

    yearly = _safe_read(FEATURE_FAMILY_YEARLY_PATH)

    cost_deltas = _safe_read(FEATURE_FAMILY_COST_PATH)

    ensemble_ablation = _safe_read(ENSEMBLE_ABLATION_PATH)

    construction_ablation = _safe_read(CONSTRUCTION_ABLATION_PATH)

    _write_csv(
        inventory,
        CHECK_INVENTORY_PATH,
    )

    _write_csv(
        coverage,
        COVERAGE_PATH,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        _build_report(
            inventory,
            coverage,
            economic=economic,
            bootstrap=bootstrap,
            yearly=yearly,
            cost_deltas=cost_deltas,
            ensemble_ablation=ensemble_ablation,
            construction_ablation=construction_ablation,
        ),
        encoding="utf-8",
    )

    blocking = coverage.loc[
        coverage["status"].isin(
            [
                "MISSING",
                "FAILED",
            ]
        )
    ]

    logger.info("Robustness evaluation audit completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Robustness evaluation final audit")
    print("------------------------------------------------")
    print(f"check_suites: {len(inventory)}")
    print(f"passed_check_suites: {int(inventory['suite_status'].eq('PASS').sum())}")
    print(f"missing_or_failed_check_suites: {int((~inventory['suite_status'].eq('PASS')).sum())}")
    print()

    print("Coverage:")
    print(
        coverage.loc[
            :,
            [
                "category",
                "dimension",
                "status",
            ],
        ].to_string(index=False)
    )
    print()

    print(f"Check inventory: {CHECK_INVENTORY_PATH}")
    print(f"Coverage table: {COVERAGE_PATH}")
    print(f"Final report: {REPORT_PATH}")
    print()

    if not blocking.empty:
        print("Robustness evaluation: INCOMPLETE")
        print()
        print("Blocking items:")
        print(
            blocking.loc[
                :,
                [
                    "category",
                    "dimension",
                    "status",
                ],
            ].to_string(index=False)
        )

        raise ValueError(
            "Robustness evaluation cannot be finalized "
            "while required coverage items are missing or failed."
        )

    print("Robustness evaluation: COMPLETE WITH DOCUMENTED LIMITATION")
    print("Expanded-universe testing remains explicitly deferred.")


if __name__ == "__main__":
    main()
