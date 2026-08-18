"""Compare feature-family ablations with the frozen official evaluation contract."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from quant_equity.config import PROCESSED_DATA_DIR, PROJECT_ROOT, REPORTS_DIR
from quant_equity.logging_config import configure_logging
from quant_equity.models import (
    EnsembleConfig,
    build_component_scores,
    build_ensemble_candidates,
    build_validation_weights,
    evaluate_model_predictions,
)

FULL_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_final_alpha_signal.py"

ABLATION_DIR = PROCESSED_DATA_DIR / "robustness" / "feature_family_ablation"

TABLE_DIR = REPORTS_DIR / "tables" / "feature_family_ablation"

REPORT_PATH = (
    REPORTS_DIR / "robustness" / "feature_family_ablation" / "official_predictive_comparison.md"
)

SCENARIO_CANDIDATE_PATHS = {
    "no_fundamentals": (ABLATION_DIR / "no_fundamentals_ensemble_candidates.parquet"),
    "no_momentum": (ABLATION_DIR / "no_momentum_ensemble_candidates.parquet"),
}

FROZEN_FULL_METRICS = {
    "mean_ic": 0.046380,
    "annualized_ic_ir": 0.948316,
    "mean_top_bottom_spread": 0.013566,
    "mean_top_quintile_precision": 0.254545,
    "mean_top_quintile_turnover": 0.409211,
}

MATCH_TOLERANCE = 5e-6
EXPECTED_DATES = 77
EXPECTED_CROSS_SECTION_SIZE = 50
EXPECTED_ROWS = EXPECTED_DATES * EXPECTED_CROSS_SECTION_SIZE


def _load_full_script_module() -> ModuleType:
    """Import the frozen final-alpha script without executing main()."""
    if not FULL_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Final alpha script not found: {FULL_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location(
        "frozen_final_alpha_script",
        FULL_SCRIPT_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create an import specification for build_final_alpha_signal.py."
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def _required_full_paths(
    module: ModuleType,
) -> dict[str, Path]:
    """Read the exact input paths used by the frozen full ensemble script."""
    names = {
        "panel": "PANEL_PATH",
        "folds": "FOLDS_PATH",
        "predictions": "ALL_MODEL_PREDICTIONS_PATH",
        "elastic": "ELASTIC_HYPERPARAMETER_PATH",
        "ranking": "RANKER_HYPERPARAMETER_PATH",
    }

    paths: dict[str, Path] = {}

    for key, attribute in names.items():
        value = getattr(
            module,
            attribute,
            None,
        )

        if value is None:
            raise AttributeError(f"Frozen final-alpha script does not define {attribute}.")

        path = Path(value)

        if not path.exists():
            raise FileNotFoundError(f"Frozen FULL input not found: {path}")

        paths[key] = path

    return paths


def _rebuild_full_candidates(
    module: ModuleType,
) -> pd.DataFrame:
    """Rebuild FULL candidate predictions through the original public pipeline."""
    paths = _required_full_paths(module)

    panel = pd.read_parquet(paths["panel"])

    folds = pd.read_parquet(paths["folds"])

    predictions = pd.read_parquet(paths["predictions"])

    elastic_hyperparameters = pd.read_csv(paths["elastic"])

    ranking_hyperparameters = pd.read_csv(paths["ranking"])

    config = EnsembleConfig()

    validation_weights = build_validation_weights(
        panel,
        folds,
        elastic_hyperparameters,
        ranking_hyperparameters,
        config=config,
    )

    component_scores = build_component_scores(
        predictions,
        config=config,
    )

    candidates = build_ensemble_candidates(
        component_scores,
        validation_weights,
    )

    return candidates


def _evaluate_candidates(
    scenario: str,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one candidate table with the frozen official evaluator."""
    monthly, summary = evaluate_model_predictions(predictions)

    monthly = monthly.copy()
    summary = summary.copy()

    monthly.insert(
        0,
        "scenario",
        scenario,
    )

    summary.insert(
        0,
        "scenario",
        scenario,
    )

    return monthly, summary


def _find_frozen_full_model(
    full_summary: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Identify the exact candidate whose metrics match frozen FULL results."""
    required_columns = {
        "model_name",
        *FROZEN_FULL_METRICS,
    }

    missing = sorted(required_columns.difference(full_summary.columns))

    if missing:
        raise ValueError("Official FULL summary is missing columns: " + ", ".join(missing) + ".")

    checks = []

    for row in full_summary.itertuples(index=False):
        model_name = str(row.model_name)

        metric_differences = {
            metric: abs(
                float(
                    getattr(
                        row,
                        metric,
                    )
                )
                - expected
            )
            for metric, expected in FROZEN_FULL_METRICS.items()
        }

        maximum_difference = max(metric_differences.values())

        checks.append(
            {
                "model_name": model_name,
                **{
                    f"{metric}_absolute_difference": value
                    for metric, value in metric_differences.items()
                },
                "maximum_absolute_difference": (maximum_difference),
                "matches_frozen_full": (maximum_difference <= MATCH_TOLERANCE),
            }
        )

    match_checks = (
        pd.DataFrame(checks)
        .sort_values(
            [
                "matches_frozen_full",
                "maximum_absolute_difference",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    matches = match_checks.loc[match_checks["matches_frozen_full"]]

    if len(matches) != 1:
        print()
        print("Official FULL candidate summary:")
        print(full_summary.to_string(index=False))

        print()
        print("Frozen FULL candidate match diagnostics:")
        print(match_checks.to_string(index=False))

        raise ValueError(
            "Expected exactly one FULL candidate to reproduce "
            "the frozen metrics; found "
            f"{len(matches)}."
        )

    return (
        str(matches.iloc[0]["model_name"]),
        match_checks,
    )


def _selected_summary(
    summary: pd.DataFrame,
    scenario: str,
    model_name: str,
) -> pd.Series:
    """Return one official summary row for the selected ensemble model."""
    selected = summary.loc[summary["model_name"].astype(str).eq(model_name)]

    if len(selected) != 1:
        raise ValueError(
            f"{scenario} should contain exactly one "
            f"summary row for {model_name}; "
            f"found {len(selected)}."
        )

    return selected.iloc[0]


def _selected_monthly(
    monthly: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Return monthly rows for the selected official ensemble candidate."""
    selected = monthly.loc[monthly["model_name"].astype(str).eq(model_name)].copy()

    if len(selected) != EXPECTED_DATES:
        raise ValueError(
            f"{model_name} should have {EXPECTED_DATES} monthly rows; found {len(selected)}."
        )

    return selected


def _prediction_keys(
    predictions: pd.DataFrame,
    model_name: str,
) -> set[tuple[pd.Timestamp, str]]:
    """Return OOS keys for one candidate model."""
    selected = predictions.loc[predictions["model_name"].astype(str).eq(model_name)].copy()

    if len(selected) != EXPECTED_ROWS:
        raise ValueError(
            f"{model_name} should have {EXPECTED_ROWS} OOS prediction rows; found {len(selected)}."
        )

    selected["as_of_date"] = pd.to_datetime(selected["as_of_date"]).dt.normalize()

    duplicate_keys = int(
        selected.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    if duplicate_keys:
        raise ValueError(f"{model_name} has {duplicate_keys} duplicate OOS keys.")

    cross_sections = selected.groupby("as_of_date")["ticker"].nunique()

    wrong_sizes = int(cross_sections.ne(EXPECTED_CROSS_SECTION_SIZE).sum())

    if wrong_sizes:
        raise ValueError(f"{model_name} has {wrong_sizes} invalid OOS cross-sections.")

    return set(
        zip(
            selected["as_of_date"],
            selected["ticker"].astype(str),
            strict=True,
        )
    )


def _build_comparison(
    summaries: dict[str, pd.DataFrame],
    monthlies: dict[str, pd.DataFrame],
    model_name: str,
) -> pd.DataFrame:
    """Build one official comparison row per scenario."""
    rows = []

    for scenario in (
        "full",
        "no_fundamentals",
        "no_momentum",
    ):
        summary_row = _selected_summary(
            summaries[scenario],
            scenario,
            model_name,
        )

        monthly = _selected_monthly(
            monthlies[scenario],
            model_name,
        )

        spread = pd.to_numeric(
            monthly["top_bottom_spread"],
            errors="coerce",
        )

        valid_spread = spread.dropna()

        rows.append(
            {
                "scenario": scenario,
                "model_name": model_name,
                "months": int(summary_row["months"]),
                "valid_ic_months": int(summary_row["valid_ic_months"]),
                "mean_ic": float(summary_row["mean_ic"]),
                "median_ic": float(summary_row["median_ic"]),
                "std_ic": float(summary_row["std_ic"]),
                "annualized_ic_ir": float(summary_row["annualized_ic_ir"]),
                "positive_ic_ratio": float(summary_row["positive_ic_ratio"]),
                "mean_top_bottom_spread": float(summary_row["mean_top_bottom_spread"]),
                "positive_spread_ratio": (
                    float(valid_spread.gt(0.0).mean()) if not valid_spread.empty else np.nan
                ),
                "mean_top_quintile_precision": float(summary_row["mean_top_quintile_precision"]),
                "mean_top_quintile_turnover": float(summary_row["mean_top_quintile_turnover"]),
            }
        )

    return pd.DataFrame(rows)


def _build_deltas(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Compute ablation deltas and retention ratios versus FULL."""
    indexed = comparison.set_index("scenario")

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


def _build_key_checks(
    candidates: dict[str, pd.DataFrame],
    model_name: str,
) -> pd.DataFrame:
    """Confirm identical selected-candidate OOS keys across scenarios."""
    full_keys = _prediction_keys(
        candidates["full"],
        model_name,
    )

    rows = []

    for scenario, predictions in candidates.items():
        keys = _prediction_keys(
            predictions,
            model_name,
        )

        differences = len(full_keys.symmetric_difference(keys))

        rows.append(
            {
                "scenario": scenario,
                "status": ("PASS" if differences == 0 else "FAIL"),
                "key_differences_vs_full": (differences),
            }
        )

    return pd.DataFrame(rows)


def _build_frozen_checks(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Verify the selected FULL row reproduces all frozen metrics."""
    full = comparison.loc[comparison["scenario"].eq("full")].iloc[0]

    rows = []

    for metric, expected in FROZEN_FULL_METRICS.items():
        observed = float(full[metric])

        difference = abs(observed - expected)

        rows.append(
            {
                "check": (f"full_{metric}"),
                "status": ("PASS" if difference <= MATCH_TOLERANCE else "FAIL"),
                "observed": observed,
                "expected": expected,
                "absolute_difference": (difference),
                "tolerance": (MATCH_TOLERANCE),
            }
        )

    return pd.DataFrame(rows)


def _build_yearly(
    selected_monthlies: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build yearly stability metrics from official monthly outputs."""
    rows = []

    for scenario, monthly in selected_monthlies.items():
        data = monthly.copy()

        data["year"] = pd.to_datetime(data["as_of_date"]).dt.year

        for year, group in data.groupby(
            "year",
            sort=True,
        ):
            valid_ic = pd.to_numeric(
                group["ic"],
                errors="coerce",
            ).dropna()

            mean_ic = float(valid_ic.mean())

            std_ic = float(valid_ic.std(ddof=1)) if len(valid_ic) > 1 else np.nan

            ic_ir = (
                float(mean_ic / std_ic * np.sqrt(12.0))
                if np.isfinite(std_ic) and std_ic > 0.0
                else np.nan
            )

            rows.append(
                {
                    "scenario": scenario,
                    "year": int(year),
                    "months": int(len(group)),
                    "mean_ic": mean_ic,
                    "annualized_ic_ir": (ic_ir),
                    "mean_top_bottom_spread": float(
                        pd.to_numeric(
                            group["top_bottom_spread"],
                            errors="coerce",
                        ).mean()
                    ),
                    "mean_top_quintile_precision": float(
                        pd.to_numeric(
                            group["top_quintile_precision"],
                            errors="coerce",
                        ).mean()
                    ),
                    "mean_top_quintile_turnover": float(
                        pd.to_numeric(
                            group["top_quintile_turnover"],
                            errors="coerce",
                        ).mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def _write_outputs(
    comparison: pd.DataFrame,
    deltas: pd.DataFrame,
    key_checks: pd.DataFrame,
    frozen_checks: pd.DataFrame,
    match_checks: pd.DataFrame,
    official_monthly: pd.DataFrame,
    yearly: pd.DataFrame,
    model_name: str,
) -> None:
    """Persist official-contract comparison outputs."""
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        TABLE_DIR / "official_predictive_comparison.csv",
        index=False,
    )

    deltas.to_csv(
        TABLE_DIR / "official_predictive_deltas_vs_full.csv",
        index=False,
    )

    key_checks.to_csv(
        TABLE_DIR / "official_predictive_key_checks.csv",
        index=False,
    )

    frozen_checks.to_csv(
        TABLE_DIR / "official_predictive_frozen_full_checks.csv",
        index=False,
    )

    match_checks.to_csv(
        TABLE_DIR / "official_predictive_full_candidate_match.csv",
        index=False,
    )

    official_monthly.to_csv(
        TABLE_DIR / "official_predictive_monthly_metrics.csv",
        index=False,
    )

    yearly.to_csv(
        TABLE_DIR / "official_predictive_yearly_metrics.csv",
        index=False,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Official Feature-Family Ablation Predictive Comparison",
                "",
                (f"- Frozen ensemble candidate: `{model_name}`"),
                ("- Evaluation function: `evaluate_model_predictions`"),
                ("- All scenarios use the same 77 OOS dates and 50 names per date."),
                "",
                "## Aggregate comparison",
                "",
                comparison.to_string(index=False),
                "",
                "## Deltas versus FULL",
                "",
                deltas.to_string(index=False),
                "",
                "## Frozen FULL checks",
                "",
                frozen_checks.to_string(index=False),
                "",
                "## OOS key checks",
                "",
                key_checks.to_string(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run the official-contract predictive comparison."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    full_script = _load_full_script_module()

    print()
    print("Institutional Quant Equity Research Platform")
    print("Official feature-family ablation predictive comparison")
    print("------------------------------------------------")
    print("Rebuilding FULL candidate predictions with frozen pipeline...")

    candidates: dict[str, pd.DataFrame] = {"full": _rebuild_full_candidates(full_script)}

    for scenario, path in SCENARIO_CANDIDATE_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"{scenario} candidate predictions not found: {path}")

        candidates[scenario] = pd.read_parquet(path)

    monthlies: dict[str, pd.DataFrame] = {}
    summaries: dict[str, pd.DataFrame] = {}

    for scenario, predictions in candidates.items():
        (
            monthlies[scenario],
            summaries[scenario],
        ) = _evaluate_candidates(
            scenario,
            predictions,
        )

    (
        frozen_model_name,
        match_checks,
    ) = _find_frozen_full_model(summaries["full"])

    print(f"Frozen FULL candidate matched: {frozen_model_name}")

    comparison = _build_comparison(
        summaries,
        monthlies,
        frozen_model_name,
    )

    deltas = _build_deltas(comparison)

    key_checks = _build_key_checks(
        candidates,
        frozen_model_name,
    )

    if key_checks["status"].eq("FAIL").any():
        raise ValueError(
            "Selected candidate is not evaluated on identical OOS keys across scenarios."
        )

    frozen_checks = _build_frozen_checks(comparison)

    failed_frozen_checks = int(frozen_checks["status"].eq("FAIL").sum())

    selected_monthlies = {
        scenario: _selected_monthly(
            monthly,
            frozen_model_name,
        )
        for scenario, monthly in monthlies.items()
    }

    official_monthly = pd.concat(
        selected_monthlies.values(),
        ignore_index=True,
    )

    yearly = _build_yearly(selected_monthlies)

    _write_outputs(
        comparison,
        deltas,
        key_checks,
        frozen_checks,
        match_checks,
        official_monthly,
        yearly,
        frozen_model_name,
    )

    logger.info("Official feature-family ablation predictive comparison completed.")

    print()
    print("Aggregate comparison:")
    print(comparison.to_string(index=False))

    print()
    print("Deltas versus FULL:")
    print(deltas.to_string(index=False))

    print()
    print("Frozen FULL checks:")
    print(frozen_checks.to_string(index=False))

    print()
    print("OOS key checks:")
    print(key_checks.to_string(index=False))

    print()
    print(f"frozen_full_checks: {len(frozen_checks)}")
    print(f"failed_frozen_full_checks: {failed_frozen_checks}")
    print(f"report: {REPORT_PATH}")

    if failed_frozen_checks:
        raise ValueError("Official evaluation still does not reproduce the frozen FULL metrics.")


if __name__ == "__main__":
    main()
