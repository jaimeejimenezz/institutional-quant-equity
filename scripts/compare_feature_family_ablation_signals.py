"""Compare predictive performance of frozen feature-family ablations."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from quant_equity.config import PROCESSED_DATA_DIR, REPORTS_DIR
from quant_equity.logging_config import configure_logging

FULL_SIGNAL_PATH = PROCESSED_DATA_DIR / "final_alpha_signal.parquet"

ABLATION_DIR = PROCESSED_DATA_DIR / "robustness" / "feature_family_ablation"

MODELING_PANEL_PATH = PROCESSED_DATA_DIR / "modeling_panel.parquet"

OUTPUT_TABLE_DIR = REPORTS_DIR / "tables" / "feature_family_ablation"

OUTPUT_REPORT_PATH = (
    REPORTS_DIR / "robustness" / "feature_family_ablation" / "predictive_comparison.md"
)

EXPECTED_DATES = 77
EXPECTED_CROSS_SECTION_SIZE = 50
EXPECTED_ROWS = EXPECTED_DATES * EXPECTED_CROSS_SECTION_SIZE
TOP_N = 10

SCENARIOS = {
    "full": FULL_SIGNAL_PATH,
    "no_fundamentals": (ABLATION_DIR / "no_fundamentals_final_alpha_signal.parquet"),
    "no_momentum": (ABLATION_DIR / "no_momentum_final_alpha_signal.parquet"),
}

FROZEN_FULL_METRICS = {
    "mean_ic": 0.046380,
    "annualized_ic_ir": 0.948316,
    "mean_top_bottom_spread": 0.013566,
    "mean_top_quintile_precision": 0.254545,
    "mean_top_quintile_turnover": 0.409211,
}

FULL_METRIC_TOLERANCE = 5e-6


def _load_targets() -> pd.DataFrame:
    """Load the OOS target columns required for comparison."""
    if not MODELING_PANEL_PATH.exists():
        raise FileNotFoundError(f"Modeling panel not found: {MODELING_PANEL_PATH}")

    panel = pd.read_parquet(
        MODELING_PANEL_PATH,
        columns=[
            "as_of_date",
            "ticker",
            "target_21d_excess",
            "label_top_quintile",
        ],
    )

    panel["as_of_date"] = pd.to_datetime(panel["as_of_date"]).dt.normalize()

    duplicate_keys = int(
        panel.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    if duplicate_keys:
        raise ValueError(f"Modeling panel target keys are not unique: {duplicate_keys} duplicates.")

    return panel


def _load_signal(
    scenario: str,
    path: Path,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Load one final signal and attach frozen realized targets."""
    if not path.exists():
        raise FileNotFoundError(f"{scenario} final signal not found: {path}")

    signal = pd.read_parquet(path).copy()

    required = {
        "as_of_date",
        "ticker",
        "percentile_score",
    }

    missing = sorted(required.difference(signal.columns))

    if missing:
        raise ValueError(f"{scenario} final signal is missing columns: " + ", ".join(missing) + ".")

    signal["as_of_date"] = pd.to_datetime(signal["as_of_date"]).dt.normalize()

    duplicate_keys = int(
        signal.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    if duplicate_keys:
        raise ValueError(f"{scenario} contains {duplicate_keys} duplicate signal keys.")

    if len(signal) != EXPECTED_ROWS:
        raise ValueError(f"{scenario} should contain {EXPECTED_ROWS} rows; found {len(signal)}.")

    cross_section_sizes = signal.groupby("as_of_date")["ticker"].nunique()

    wrong_cross_sections = int(cross_section_sizes.ne(EXPECTED_CROSS_SECTION_SIZE).sum())

    if wrong_cross_sections:
        raise ValueError(f"{scenario} has {wrong_cross_sections} invalid cross-sections.")

    merged = signal.merge(
        targets,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    target_missing = int(merged["target_21d_excess"].isna().sum())

    label_missing = int(merged["label_top_quintile"].isna().sum())

    if target_missing or label_missing:
        raise ValueError(
            f"{scenario} target join produced "
            f"{target_missing} missing targets and "
            f"{label_missing} missing labels."
        )

    merged["percentile_score"] = pd.to_numeric(
        merged["percentile_score"],
        errors="coerce",
    )

    if not np.isfinite(merged["percentile_score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{scenario} contains non-finite percentile scores.")

    return merged.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)


def _monthly_metrics(
    data: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """Compute date-level IC, spread, precision and top-quintile turnover."""
    rows = []
    previous_top: set[str] | None = None

    for as_of_date, cross_section in data.groupby(
        "as_of_date",
        sort=True,
    ):
        ordered = cross_section.sort_values(
            [
                "percentile_score",
                "ticker",
            ],
            ascending=[
                False,
                True,
            ],
        )

        top = ordered.head(TOP_N)
        bottom = ordered.tail(TOP_N)

        ic = float(
            cross_section["percentile_score"].corr(
                cross_section["target_21d_excess"],
                method="spearman",
            )
        )

        top_bottom_spread = float(
            top["target_21d_excess"].mean() - bottom["target_21d_excess"].mean()
        )

        top_precision = float(
            pd.to_numeric(
                top["label_top_quintile"],
                errors="coerce",
            ).mean()
        )

        current_top = set(top["ticker"].astype(str))

        if previous_top is None:
            turnover = np.nan
        else:
            overlap = len(current_top.intersection(previous_top))

            turnover = float(1.0 - overlap / TOP_N)

        rows.append(
            {
                "scenario": scenario,
                "as_of_date": pd.Timestamp(as_of_date),
                "ic": ic,
                "top_bottom_spread": (top_bottom_spread),
                "top_quintile_precision": (top_precision),
                "top_quintile_turnover": (turnover),
            }
        )

        previous_top = current_top

    result = pd.DataFrame(rows)

    if len(result) != EXPECTED_DATES:
        raise ValueError(
            f"{scenario} should produce {EXPECTED_DATES} monthly metric rows; found {len(result)}."
        )

    return result


def _aggregate_metrics(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate predictive metrics by scenario."""
    rows = []

    for scenario, group in monthly.groupby(
        "scenario",
        sort=False,
    ):
        ic = group["ic"].astype(float)

        ic_std = float(ic.std(ddof=1))

        mean_ic = float(ic.mean())

        annualized_ic_ir = float(mean_ic / ic_std * np.sqrt(12.0)) if ic_std > 0.0 else np.nan

        rows.append(
            {
                "scenario": scenario,
                "oos_dates": int(group["as_of_date"].nunique()),
                "mean_ic": mean_ic,
                "ic_std": ic_std,
                "annualized_ic_ir": (annualized_ic_ir),
                "positive_ic_frequency": float(ic.gt(0.0).mean()),
                "mean_top_bottom_spread": float(group["top_bottom_spread"].mean()),
                "positive_spread_frequency": float(group["top_bottom_spread"].gt(0.0).mean()),
                "mean_top_quintile_precision": float(group["top_quintile_precision"].mean()),
                "mean_top_quintile_turnover": float(group["top_quintile_turnover"].mean()),
            }
        )

    return pd.DataFrame(rows)


def _yearly_metrics(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize predictive stability by calendar year."""
    data = monthly.copy()
    data["year"] = data["as_of_date"].dt.year

    rows = []

    for (
        scenario,
        year,
    ), group in data.groupby(
        [
            "scenario",
            "year",
        ],
        sort=True,
    ):
        ic = group["ic"].astype(float)

        ic_std = float(ic.std(ddof=1))

        mean_ic = float(ic.mean())

        annualized_ic_ir = (
            float(mean_ic / ic_std * np.sqrt(12.0))
            if np.isfinite(ic_std) and ic_std > 0.0
            else np.nan
        )

        rows.append(
            {
                "scenario": scenario,
                "year": int(year),
                "months": int(len(group)),
                "mean_ic": mean_ic,
                "annualized_ic_ir": (annualized_ic_ir),
                "mean_top_bottom_spread": float(group["top_bottom_spread"].mean()),
                "mean_top_quintile_precision": float(group["top_quintile_precision"].mean()),
                "mean_top_quintile_turnover": float(group["top_quintile_turnover"].mean()),
            }
        )

    return pd.DataFrame(rows)


def _build_deltas(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Express ablation changes relative to the frozen full ensemble."""
    indexed = comparison.set_index("scenario")

    if "full" not in indexed.index:
        raise ValueError("Full scenario is missing from comparison.")

    full = indexed.loc["full"]

    rows = []

    for scenario in (
        "no_fundamentals",
        "no_momentum",
    ):
        current = indexed.loc[scenario]

        rows.append(
            {
                "scenario": scenario,
                "delta_mean_ic_vs_full": float(current["mean_ic"] - full["mean_ic"]),
                "delta_annualized_ic_ir_vs_full": float(
                    current["annualized_ic_ir"] - full["annualized_ic_ir"]
                ),
                "delta_top_bottom_spread_vs_full": float(
                    current["mean_top_bottom_spread"] - full["mean_top_bottom_spread"]
                ),
                "delta_top_quintile_precision_vs_full": float(
                    current["mean_top_quintile_precision"] - full["mean_top_quintile_precision"]
                ),
                "delta_top_quintile_turnover_vs_full": float(
                    current["mean_top_quintile_turnover"] - full["mean_top_quintile_turnover"]
                ),
                "ic_retention_ratio": (
                    float(current["mean_ic"] / full["mean_ic"])
                    if abs(full["mean_ic"]) > 1e-12
                    else np.nan
                ),
                "spread_retention_ratio": (
                    float(current["mean_top_bottom_spread"] / full["mean_top_bottom_spread"])
                    if abs(full["mean_top_bottom_spread"]) > 1e-12
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def _validate_full_metrics(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-check formulas against the previously frozen FULL results."""
    full = comparison.loc[comparison["scenario"].eq("full")].iloc[0]

    rows = []

    for metric, expected in FROZEN_FULL_METRICS.items():
        observed = float(full[metric])

        absolute_difference = float(abs(observed - expected))

        passed = absolute_difference <= FULL_METRIC_TOLERANCE

        rows.append(
            {
                "check": (f"full_{metric}"),
                "status": ("PASS" if passed else "FAIL"),
                "observed": observed,
                "expected": float(expected),
                "absolute_difference": (absolute_difference),
                "tolerance": (FULL_METRIC_TOLERANCE),
            }
        )

    return pd.DataFrame(rows)


def _validate_common_keys(
    datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """Confirm all scenarios are evaluated on identical OOS keys."""
    full_keys = set(
        zip(
            datasets["full"]["as_of_date"],
            datasets["full"]["ticker"].astype(str),
            strict=True,
        )
    )

    rows = []

    for scenario, data in datasets.items():
        keys = set(
            zip(
                data["as_of_date"],
                data["ticker"].astype(str),
                strict=True,
            )
        )

        symmetric_difference = len(full_keys.symmetric_difference(keys))

        rows.append(
            {
                "scenario": scenario,
                "status": ("PASS" if symmetric_difference == 0 else "FAIL"),
                "key_differences_vs_full": int(symmetric_difference),
            }
        )

    return pd.DataFrame(rows)


def _write_outputs(
    comparison: pd.DataFrame,
    deltas: pd.DataFrame,
    monthly: pd.DataFrame,
    yearly: pd.DataFrame,
    formula_checks: pd.DataFrame,
    key_checks: pd.DataFrame,
) -> None:
    """Persist comparison tables and report."""
    OUTPUT_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = OUTPUT_TABLE_DIR / "predictive_comparison.csv"

    deltas_path = OUTPUT_TABLE_DIR / "predictive_deltas_vs_full.csv"

    monthly_path = OUTPUT_TABLE_DIR / "predictive_monthly_metrics.csv"

    yearly_path = OUTPUT_TABLE_DIR / "predictive_yearly_metrics.csv"

    formula_checks_path = OUTPUT_TABLE_DIR / "predictive_formula_checks.csv"

    key_checks_path = OUTPUT_TABLE_DIR / "predictive_key_checks.csv"

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    deltas.to_csv(
        deltas_path,
        index=False,
    )

    monthly.to_csv(
        monthly_path,
        index=False,
    )

    yearly.to_csv(
        yearly_path,
        index=False,
    )

    formula_checks.to_csv(
        formula_checks_path,
        index=False,
    )

    key_checks.to_csv(
        key_checks_path,
        index=False,
    )

    OUTPUT_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = "\n".join(
        [
            "# Feature-Family Ablation Predictive Comparison",
            "",
            "## Aggregate comparison",
            "",
            comparison.to_string(index=False),
            "",
            "## Deltas versus FULL",
            "",
            deltas.to_string(index=False),
            "",
            "## Frozen FULL metric cross-check",
            "",
            formula_checks.to_string(index=False),
            "",
            "## OOS key checks",
            "",
            key_checks.to_string(index=False),
            "",
            "## Methodology",
            "",
            (
                "- Monthly IC is the cross-sectional Spearman correlation "
                "between `percentile_score` and `target_21d_excess`."
            ),
            (
                "- Annualized IC IR is mean monthly IC divided by the "
                "sample standard deviation of monthly IC, multiplied by sqrt(12)."
            ),
            (
                "- Top-bottom spread is mean realized excess return of the "
                "top predicted quintile minus the bottom predicted quintile."
            ),
            (
                "- Top-quintile precision is the share of the predicted top "
                "10 names that belong to the realized top quintile."
            ),
            (
                "- Top-quintile turnover is 1 minus the overlap between "
                "consecutive top-10 sets divided by 10."
            ),
            (
                "- All three scenarios are evaluated on exactly the same "
                "77 OOS dates and 50 companies per date."
            ),
            "",
        ]
    )

    OUTPUT_REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    """Run predictive comparison for FULL and both frozen ablations."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    targets = _load_targets()

    datasets = {
        scenario: _load_signal(
            scenario,
            path,
            targets,
        )
        for scenario, path in SCENARIOS.items()
    }

    key_checks = _validate_common_keys(datasets)

    if key_checks["status"].eq("FAIL").any():
        raise ValueError("Ablation scenarios are not evaluated on identical OOS keys.")

    monthly_frames = [
        _monthly_metrics(
            data,
            scenario,
        )
        for scenario, data in datasets.items()
    ]

    monthly = pd.concat(
        monthly_frames,
        ignore_index=True,
    )

    comparison = _aggregate_metrics(monthly)

    scenario_order = [
        "full",
        "no_fundamentals",
        "no_momentum",
    ]

    comparison["scenario"] = pd.Categorical(
        comparison["scenario"],
        categories=(scenario_order),
        ordered=True,
    )

    comparison = (
        comparison.sort_values("scenario")
        .assign(scenario=lambda frame: frame["scenario"].astype("string"))
        .reset_index(drop=True)
    )

    deltas = _build_deltas(comparison)

    yearly = _yearly_metrics(monthly)

    formula_checks = _validate_full_metrics(comparison)

    failed_formula_checks = int(formula_checks["status"].eq("FAIL").sum())

    _write_outputs(
        comparison,
        deltas,
        monthly,
        yearly,
        formula_checks,
        key_checks,
    )

    logger.info("Feature-family ablation predictive comparison completed.")

    print()
    print("Institutional Quant Equity Research Platform")
    print("Feature-family ablation predictive comparison")
    print("------------------------------------------------")

    print()
    print("Aggregate comparison:")
    print(comparison.to_string(index=False))

    print()
    print("Deltas versus FULL:")
    print(deltas.to_string(index=False))

    print()
    print("Frozen FULL metric cross-check:")
    print(formula_checks.to_string(index=False))

    print()
    print("OOS key checks:")
    print(key_checks.to_string(index=False))

    print()
    print(f"formula_checks: {len(formula_checks)}")
    print(f"failed_formula_checks: {failed_formula_checks}")
    print(f"report: {OUTPUT_REPORT_PATH}")

    if failed_formula_checks:
        raise ValueError(
            "Predictive metric formulas do not reproduce the frozen FULL ensemble results."
        )


if __name__ == "__main__":
    main()
