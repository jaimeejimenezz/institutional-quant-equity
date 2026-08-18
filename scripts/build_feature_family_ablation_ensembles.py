"""Build final alpha ensembles for frozen feature-family ablations."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from quant_equity.config import PROCESSED_DATA_DIR, REPORTS_DIR
from quant_equity.logging_config import configure_logging
from quant_equity.models import (
    COMPOSITE_FEATURE_DIRECTIONS,
    EnsembleConfig,
    build_ablation_candidates,
    build_component_scores,
    build_ensemble_candidates,
    build_final_alpha_signal,
    build_validation_weights,
    score_technical_composite,
)

MODELING_PANEL_PATH = PROCESSED_DATA_DIR / "modeling_panel.parquet"
FOLDS_PATH = PROCESSED_DATA_DIR / "walk_forward_folds.parquet"
CONTRACT_PATH = REPORTS_DIR / "tables" / "robustness_feature_family_contract.csv"

INPUT_DIR = PROCESSED_DATA_DIR / "robustness" / "feature_family_ablation"
OUTPUT_DIR = INPUT_DIR

TABLES_DIR = REPORTS_DIR / "tables" / "feature_family_ablation"
REPORT_DIR = REPORTS_DIR / "robustness" / "feature_family_ablation"

EXPECTED_FOLDS = 77
EXPECTED_CROSS_SECTION_SIZE = 50
EXPECTED_SIGNAL_ROWS = EXPECTED_FOLDS * EXPECTED_CROSS_SECTION_SIZE

SCENARIO_FLAGS = {
    "no_fundamentals": "included_no_fundamentals",
    "no_momentum": "included_no_momentum",
}

SCENARIO_EXPECTED_FEATURES = {
    "no_fundamentals": 19,
    "no_momentum": 85,
}


def _as_bool(values: pd.Series) -> pd.Series:
    """Normalize CSV boolean values."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)

    normalized = values.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    result = normalized.map(mapping)

    if result.isna().any():
        invalid = sorted(normalized.loc[result.isna()].dropna().unique().tolist())
        raise ValueError(f"Cannot interpret contract boolean values: {invalid}.")

    return result.astype(bool)


def _load_contract() -> pd.DataFrame:
    """Load the frozen feature-family contract."""
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Feature-family contract not found: {CONTRACT_PATH}")

    contract = pd.read_csv(CONTRACT_PATH)

    required = {
        "feature",
        "is_momentum",
        "included_no_fundamentals",
        "included_no_momentum",
    }
    missing = sorted(required.difference(contract.columns))

    if missing:
        raise ValueError("Feature-family contract is missing columns: " + ", ".join(missing) + ".")

    for column in (
        "is_momentum",
        "included_no_fundamentals",
        "included_no_momentum",
    ):
        contract[column] = _as_bool(contract[column])

    if len(contract) != 91:
        raise ValueError(
            f"Feature-family contract must contain 91 predictors; found {len(contract)}."
        )

    if contract["feature"].duplicated().any():
        raise ValueError("Feature-family contract contains duplicated predictors.")

    return contract


def _scenario_panel(
    panel: pd.DataFrame,
    contract: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """Return the modeling panel with only scenario predictors retained."""
    flag = SCENARIO_FLAGS[scenario]
    selected = tuple(contract.loc[contract[flag], "feature"].astype(str))
    expected = SCENARIO_EXPECTED_FEATURES[scenario]

    if len(selected) != expected:
        raise ValueError(f"{scenario} should contain {expected} predictors; found {len(selected)}.")

    all_features = tuple(contract["feature"].astype(str))
    missing = sorted(set(all_features).difference(panel.columns))

    if missing:
        raise ValueError(
            "Modeling panel is missing frozen predictors: "
            + ", ".join(missing[:10])
            + ("..." if len(missing) > 10 else "")
        )

    selected_set = set(selected)
    removed = [feature for feature in all_features if feature not in selected_set]

    return panel.drop(columns=removed).copy()


def _prepare_validation_panel(
    scenario_panel: pd.DataFrame,
    contract: pd.DataFrame,
    scenario: str,
) -> tuple[pd.DataFrame, int]:
    """Prepare a panel compatible with the original validation-weight builder."""
    if scenario != "no_momentum":
        return (
            scenario_panel.copy(),
            len(COMPOSITE_FEATURE_DIRECTIONS),
        )

    momentum_features = set(
        contract.loc[
            contract["is_momentum"],
            "feature",
        ].astype(str)
    )

    composite_momentum = tuple(
        feature for feature in COMPOSITE_FEATURE_DIRECTIONS if feature in momentum_features
    )
    surviving_composite = tuple(
        feature for feature in COMPOSITE_FEATURE_DIRECTIONS if feature not in momentum_features
    )

    if len(composite_momentum) != 2:
        raise ValueError(
            "No-momentum validation should remove exactly two "
            "technical-composite inputs; found "
            f"{len(composite_momentum)}."
        )

    if len(surviving_composite) != 6:
        raise ValueError(
            "No-momentum validation should retain exactly six "
            "technical-composite inputs; found "
            f"{len(surviving_composite)}."
        )

    proxy = scenario_panel.copy()

    for feature in composite_momentum:
        proxy[feature] = np.nan

    original_style_score = score_technical_composite(proxy)

    directions = pd.Series(
        {feature: float(COMPOSITE_FEATURE_DIRECTIONS[feature]) for feature in surviving_composite},
        dtype=float,
    )

    surviving_values = (
        proxy.loc[:, surviving_composite].apply(pd.to_numeric, errors="coerce").astype(float)
    )

    signed = surviving_values.mul(
        directions,
        axis="columns",
    )
    direct_reduced_score = signed.mean(
        axis=1,
        skipna=True,
    ).where(signed.notna().sum(axis=1) >= 6)

    original_values = original_style_score.to_numpy(dtype=float)
    reduced_values = direct_reduced_score.to_numpy(dtype=float)

    if not np.allclose(
        original_values,
        reduced_values,
        equal_nan=True,
    ):
        raise ValueError(
            "The validation proxy does not reproduce the exact six-component no-momentum composite."
        )

    return proxy, len(surviving_composite)


def _load_scenario_inputs(
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load OOS predictions and selected-validation research outputs."""
    predictions_path = INPUT_DIR / f"{scenario}_ensemble_predictions.parquet"
    elastic_path = TABLES_DIR / f"{scenario}_regularized_linear_hyperparameters.csv"
    ranking_path = TABLES_DIR / f"{scenario}_lightgbm_ranker_hyperparameters.csv"

    for path in (
        predictions_path,
        elastic_path,
        ranking_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required scenario input not found: {path}")

    return (
        pd.read_parquet(predictions_path),
        pd.read_csv(elastic_path),
        pd.read_csv(ranking_path),
    )


def _validate_signal_shape(
    data: pd.DataFrame,
    *,
    dataset_name: str,
) -> list[dict[str, object]]:
    """Return basic OOS shape checks for a final-signal-like table."""
    required = {
        "as_of_date",
        "ticker",
    }
    missing = sorted(required.difference(data.columns))

    checks: list[dict[str, object]] = []

    checks.append(
        {
            "check": f"{dataset_name}_required_keys",
            "status": "PASS" if not missing else "FAIL",
            "violations": len(missing),
            "description": ("Output must contain as_of_date and ticker."),
        }
    )

    if missing:
        return checks

    dates = pd.to_datetime(data["as_of_date"]).dt.normalize()

    duplicate_keys = int(data.assign(_date=dates).duplicated(["_date", "ticker"]).sum())

    cross_section_sizes = data.assign(_date=dates).groupby("_date")["ticker"].nunique()

    wrong_cross_sections = int(cross_section_sizes.ne(EXPECTED_CROSS_SECTION_SIZE).sum())

    checks.extend(
        [
            {
                "check": f"{dataset_name}_rows",
                "status": ("PASS" if len(data) == EXPECTED_SIGNAL_ROWS else "FAIL"),
                "violations": int(len(data) != EXPECTED_SIGNAL_ROWS),
                "description": (f"Output must contain {EXPECTED_SIGNAL_ROWS} OOS rows."),
            },
            {
                "check": f"{dataset_name}_dates",
                "status": ("PASS" if dates.nunique() == EXPECTED_FOLDS else "FAIL"),
                "violations": int(dates.nunique() != EXPECTED_FOLDS),
                "description": (f"Output must contain {EXPECTED_FOLDS} OOS dates."),
            },
            {
                "check": f"{dataset_name}_cross_sections",
                "status": ("PASS" if wrong_cross_sections == 0 else "FAIL"),
                "violations": wrong_cross_sections,
                "description": (
                    f"Every OOS date must contain exactly {EXPECTED_CROSS_SECTION_SIZE} tickers."
                ),
            },
            {
                "check": f"{dataset_name}_unique_keys",
                "status": ("PASS" if duplicate_keys == 0 else "FAIL"),
                "violations": duplicate_keys,
                "description": ("Output must be unique by as_of_date and ticker."),
            },
        ]
    )

    return checks


def _write_parquet(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write Parquet atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)
    data.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write CSV output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def _write_report(
    *,
    scenario: str,
    predictor_count: int,
    composite_components: int,
    validation_weights: pd.DataFrame,
    component_scores: pd.DataFrame,
    final_signal: pd.DataFrame,
    checks: pd.DataFrame,
    path: Path,
) -> None:
    """Write a concise methodology and readiness report."""
    weight_columns = [
        "composite_weight",
        "elastic_net_weight",
        "lightgbm_ranker_weight",
    ]

    weight_summary = (
        validation_weights.loc[:, weight_columns]
        .agg(["mean", "std", "min", "max"])
        .T.reset_index()
        .rename(columns={"index": "component"})
    )

    lines = [
        "# Feature-Family Ablation Ensemble",
        "",
        f"- Scenario: `{scenario}`",
        f"- Predictor count: `{predictor_count}`",
        (f"- Technical composite components used in validation: `{composite_components}`"),
        (f"- Validation-weight rows: `{len(validation_weights)}`"),
        (f"- Component-score rows: `{len(component_scores)}`"),
        (f"- Final alpha rows: `{len(final_signal)}`"),
        "",
        "## Methodology",
        "",
        (
            "- Uses the same `EnsembleConfig` and public ensemble "
            "functions as the frozen full model."
        ),
        (
            "- Elastic Net and LightGBM validation IC values come "
            "from each ablation's own hyperparameter-selection run."
        ),
        (
            "- For `no_momentum`, the two removed composite inputs "
            "are represented as missing only inside validation-weight "
            "construction. Because the frozen composite requires at "
            "least six available components and averages with "
            "`skipna=True`, this exactly reproduces the six surviving "
            "signed components without changing ensemble code."
        ),
        (
            "- Final OOS scores are built from ablation-specific OOS "
            "component predictions and fold-specific validation weights."
        ),
        "",
        "## Validation-weight summary",
        "",
        weight_summary.to_string(index=False),
        "",
        "## Readiness checks",
        "",
        checks.to_string(index=False),
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _run_scenario(
    panel: pd.DataFrame,
    folds: pd.DataFrame,
    contract: pd.DataFrame,
    scenario: str,
) -> None:
    """Build one ablation-specific final ensemble."""
    logger = logging.getLogger("quant_equity")

    scenario_panel = _scenario_panel(
        panel,
        contract,
        scenario,
    )
    predictor_count = SCENARIO_EXPECTED_FEATURES[scenario]

    (
        validation_panel,
        composite_components,
    ) = _prepare_validation_panel(
        scenario_panel,
        contract,
        scenario,
    )

    (
        predictions,
        elastic_hyperparameters,
        ranking_hyperparameters,
    ) = _load_scenario_inputs(scenario)

    config = EnsembleConfig(
        expected_cross_section_size=(EXPECTED_CROSS_SECTION_SIZE),
        equal_weight_prior=0.5,
        minimum_validation_dates=12,
    )

    print()
    print(f"[{scenario}] predictors: {predictor_count}")
    print(f"[{scenario}] validation composite components: {composite_components}")
    print(f"[{scenario}] building fold-specific validation weights...")

    validation_weights = build_validation_weights(
        validation_panel,
        folds,
        elastic_hyperparameters,
        ranking_hyperparameters,
        config=config,
    )

    print(f"[{scenario}] building OOS component percentile scores...")

    component_scores = build_component_scores(
        predictions,
        config=config,
    )

    print(f"[{scenario}] building ensemble diagnostics and final alpha...")

    ensemble_candidates = build_ensemble_candidates(
        component_scores,
        validation_weights,
    )

    final_signal = build_final_alpha_signal(
        component_scores,
        validation_weights,
    )

    ablation_candidates = build_ablation_candidates(
        component_scores,
        validation_weights,
    )

    weight_columns = (
        "composite_weight",
        "elastic_net_weight",
        "lightgbm_ranker_weight",
    )

    weights_numeric = validation_weights.loc[
        :,
        weight_columns,
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    weight_sums = weights_numeric.sum(axis=1)
    weight_nonfinite = int((~np.isfinite(weights_numeric.to_numpy(dtype=float))).sum())
    invalid_weight_sums = int(
        (
            ~np.isclose(
                weight_sums.to_numpy(dtype=float),
                1.0,
            )
        ).sum()
    )
    negative_weights = int((weights_numeric < -1e-12).sum().sum())

    checks = [
        {
            "check": "validation_weight_rows",
            "status": ("PASS" if len(validation_weights) == EXPECTED_FOLDS else "FAIL"),
            "violations": int(len(validation_weights) != EXPECTED_FOLDS),
            "description": ("There must be exactly one validation-weight row per frozen fold."),
        },
        {
            "check": "validation_weights_finite",
            "status": ("PASS" if weight_nonfinite == 0 else "FAIL"),
            "violations": weight_nonfinite,
            "description": "All ensemble weights must be finite.",
        },
        {
            "check": "validation_weights_sum_to_one",
            "status": ("PASS" if invalid_weight_sums == 0 else "FAIL"),
            "violations": invalid_weight_sums,
            "description": ("Fold-specific ensemble weights must sum to one."),
        },
        {
            "check": "validation_weights_nonnegative",
            "status": ("PASS" if negative_weights == 0 else "FAIL"),
            "violations": negative_weights,
            "description": ("Fold-specific ensemble weights must be non-negative."),
        },
    ]

    checks.extend(
        _validate_signal_shape(
            component_scores,
            dataset_name="component_scores",
        )
    )
    checks.extend(
        _validate_signal_shape(
            final_signal,
            dataset_name="final_signal",
        )
    )

    checks_df = pd.DataFrame(checks)
    failed = int(checks_df["status"].eq("FAIL").sum())

    if failed:
        print()
        print(checks_df.to_string(index=False))
        raise ValueError(f"{scenario} ensemble validation failed with {failed} failed checks.")

    final_signal_path = OUTPUT_DIR / f"{scenario}_final_alpha_signal.parquet"
    component_scores_path = OUTPUT_DIR / f"{scenario}_component_scores.parquet"
    ensemble_candidates_path = OUTPUT_DIR / f"{scenario}_ensemble_candidates.parquet"
    ablation_candidates_path = OUTPUT_DIR / f"{scenario}_ensemble_ablation_candidates.parquet"
    validation_weights_path = TABLES_DIR / f"{scenario}_validation_weights.csv"
    checks_path = TABLES_DIR / f"{scenario}_ensemble_checks.csv"
    report_path = REPORT_DIR / f"{scenario}_ensemble.md"

    _write_parquet(
        final_signal,
        final_signal_path,
    )
    _write_parquet(
        component_scores,
        component_scores_path,
    )
    _write_parquet(
        ensemble_candidates,
        ensemble_candidates_path,
    )
    _write_parquet(
        ablation_candidates,
        ablation_candidates_path,
    )
    _write_csv(
        validation_weights,
        validation_weights_path,
    )
    _write_csv(
        checks_df,
        checks_path,
    )

    _write_report(
        scenario=scenario,
        predictor_count=predictor_count,
        composite_components=composite_components,
        validation_weights=validation_weights,
        component_scores=component_scores,
        final_signal=final_signal,
        checks=checks_df,
        path=report_path,
    )

    logger.info(
        "%s ablation ensemble completed.",
        scenario,
    )

    print()
    print(f"[{scenario}] completed.")
    print(f"[{scenario}] validation weight rows: {len(validation_weights)}")
    print(f"[{scenario}] component score rows: {len(component_scores)}")
    print(f"[{scenario}] final alpha rows: {len(final_signal)}")
    print(f"[{scenario}] readiness checks: {len(checks_df)}")
    print(f"[{scenario}] failed readiness checks: {failed}")
    print(f"[{scenario}] final alpha: {final_signal_path}")
    print(f"[{scenario}] report: {report_path}")


def _parse_args() -> argparse.Namespace:
    """Parse requested scenario."""
    parser = argparse.ArgumentParser(
        description=("Build final alpha signals for frozen feature-family ablations.")
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "no_fundamentals",
            "no_momentum",
            "all",
        ),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    """Build the requested ablation ensembles."""
    configure_logging()
    args = _parse_args()

    for path in (
        MODELING_PANEL_PATH,
        FOLDS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    panel = pd.read_parquet(MODELING_PANEL_PATH)
    folds = pd.read_parquet(FOLDS_PATH)
    contract = _load_contract()

    if len(folds) != EXPECTED_FOLDS:
        raise ValueError(
            "Walk-forward metadata must contain exactly "
            f"{EXPECTED_FOLDS} folds; found {len(folds)}."
        )

    scenarios = tuple(SCENARIO_FLAGS) if args.scenario == "all" else (args.scenario,)

    print()
    print("Institutional Quant Equity Research Platform")
    print("Feature-family ablation ensembles")
    print("------------------------------------------------")
    print(f"scenarios: {', '.join(scenarios)}")
    print(f"walk_forward_folds: {len(folds)}")
    print(f"expected_final_signal_rows: {EXPECTED_SIGNAL_ROWS}")

    for scenario in scenarios:
        _run_scenario(
            panel,
            folds,
            contract,
            scenario,
        )


if __name__ == "__main__":
    main()
