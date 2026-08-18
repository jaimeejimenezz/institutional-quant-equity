"""Retrain feature-family ablations on the frozen walk-forward folds."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_equity.config import PROCESSED_DATA_DIR, REPORTS_DIR
from quant_equity.logging_config import configure_logging
from quant_equity.models import (
    COMPOSITE_FEATURE_DIRECTIONS,
    LightGBMRankingConfig,
    RegularizedLinearConfig,
    score_technical_composite,
    train_lightgbm_ranking,
    train_regularized_linear_models,
)
from quant_equity.models.regularized_linear import detect_model_features

MODELING_PANEL_PATH = PROCESSED_DATA_DIR / "modeling_panel.parquet"
FOLDS_PATH = PROCESSED_DATA_DIR / "walk_forward_folds.parquet"
CONTRACT_PATH = REPORTS_DIR / "tables" / "robustness_feature_family_contract.csv"

OUTPUT_DIR = PROCESSED_DATA_DIR / "robustness" / "feature_family_ablation"
TABLES_DIR = REPORTS_DIR / "tables" / "feature_family_ablation"
REPORT_DIR = REPORTS_DIR / "robustness" / "feature_family_ablation"

EXPECTED_FOLDS = 77
EXPECTED_CROSS_SECTION_SIZE = 50
EXPECTED_PREDICTION_ROWS = EXPECTED_FOLDS * EXPECTED_CROSS_SECTION_SIZE

BASELINE_PREDICTION_COLUMNS = (
    "fold_id",
    "as_of_date",
    "ticker",
    "sector",
    "model_name",
    "prediction",
    "target_21d_excess",
    "label_top_quintile",
)

SCENARIO_FLAGS = {
    "no_fundamentals": "included_no_fundamentals",
    "no_momentum": "included_no_momentum",
}
SCENARIO_EXPECTED_FEATURES = {
    "no_fundamentals": 19,
    "no_momentum": 85,
}


@dataclass(frozen=True)
class ScenarioOutputs:
    """Persisted outputs for one feature-family ablation."""

    scenario: str
    feature_count: int
    composite_components: int
    elapsed_seconds: float
    predictions_path: Path
    checks_path: Path
    report_path: Path


def _write_parquet(data: pd.DataFrame, path: Path) -> None:
    """Write a Parquet file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)
    data.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_csv(data: pd.DataFrame, path: Path) -> None:
    """Write a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


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


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and minimally validate the frozen ablation inputs."""
    for path in (
        MODELING_PANEL_PATH,
        FOLDS_PATH,
        CONTRACT_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    panel = pd.read_parquet(MODELING_PANEL_PATH)
    folds = pd.read_parquet(FOLDS_PATH)
    contract = pd.read_csv(CONTRACT_PATH)

    required_contract_columns = {
        "feature",
        "is_momentum",
        "included_no_fundamentals",
        "included_no_momentum",
    }
    missing_contract_columns = sorted(required_contract_columns.difference(contract.columns))
    if missing_contract_columns:
        raise ValueError(
            "Feature-family contract is missing columns: "
            + ", ".join(missing_contract_columns)
            + "."
        )

    for column in (
        "is_momentum",
        "included_no_fundamentals",
        "included_no_momentum",
    ):
        contract[column] = _as_bool(contract[column])

    if len(contract) != 91:
        raise ValueError(
            f"Feature-family contract must contain exactly 91 predictors; found {len(contract)}."
        )

    if contract["feature"].duplicated().any():
        raise ValueError("Feature-family contract contains duplicated predictors.")

    if len(folds) != EXPECTED_FOLDS:
        raise ValueError(
            "Walk-forward metadata must contain exactly "
            f"{EXPECTED_FOLDS} folds; found {len(folds)}."
        )

    return panel, folds, contract


def _scenario_features(
    contract: pd.DataFrame,
    scenario: str,
) -> tuple[str, ...]:
    """Return the frozen predictor set for one ablation scenario."""
    flag = SCENARIO_FLAGS[scenario]
    features = tuple(contract.loc[contract[flag], "feature"].astype(str))
    expected_count = SCENARIO_EXPECTED_FEATURES[scenario]

    if len(features) != expected_count:
        raise ValueError(
            f"{scenario} should contain {expected_count} predictors; found {len(features)}."
        )

    return features


def _reduced_panel(
    panel: pd.DataFrame,
    contract: pd.DataFrame,
    scenario: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Create a modeling panel containing only the frozen scenario predictors."""
    selected_features = _scenario_features(contract, scenario)
    selected_set = set(selected_features)
    all_model_features = tuple(contract["feature"].astype(str))

    missing = sorted(set(all_model_features).difference(panel.columns))
    if missing:
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            "Modeling panel is missing frozen predictors: " + ", ".join(missing[:10]) + suffix
        )

    removed_features = [feature for feature in all_model_features if feature not in selected_set]
    reduced = panel.drop(columns=removed_features).copy()

    detected = detect_model_features(
        reduced,
        expected_count=len(selected_features),
    )

    if set(detected) != selected_set:
        raise ValueError(
            f"{scenario} feature detection does not match the frozen ablation contract."
        )

    return reduced, detected


def _reduced_composite_directions(
    contract: pd.DataFrame,
) -> dict[str, float]:
    """Remove frozen momentum components from the technical composite."""
    momentum_features = set(
        contract.loc[
            contract["is_momentum"],
            "feature",
        ].astype(str)
    )

    directions = {
        str(feature): float(direction)
        for feature, direction in COMPOSITE_FEATURE_DIRECTIONS.items()
        if str(feature) not in momentum_features
    }

    if len(directions) != 6:
        raise ValueError(
            "The no-momentum technical composite should retain "
            f"exactly 6 components; found {len(directions)}."
        )

    return directions


def _score_reduced_composite(
    test: pd.DataFrame,
    directions: dict[str, float],
) -> pd.Series:
    """Score the no-momentum technical composite using six surviving inputs."""
    columns = tuple(directions)
    missing = sorted(set(columns).difference(test.columns))
    if missing:
        raise ValueError(
            "Reduced technical composite is missing columns: " + ", ".join(missing) + "."
        )

    values = test.loc[:, columns].apply(pd.to_numeric, errors="coerce").astype(float)
    signed = values.mul(
        pd.Series(directions, dtype=float),
        axis="columns",
    )
    available_components = signed.notna().sum(axis=1)
    score = signed.mean(axis=1, skipna=True)

    return score.where(available_components >= 6)


def _build_composite_predictions(
    panel: pd.DataFrame,
    folds: pd.DataFrame,
    contract: pd.DataFrame,
    scenario: str,
) -> tuple[pd.DataFrame, int]:
    """Build technical-composite OOS predictions for one scenario."""
    data = panel.copy()
    metadata = folds.copy()

    data["as_of_date"] = pd.to_datetime(data["as_of_date"]).dt.normalize()
    metadata["test_date"] = pd.to_datetime(metadata["test_date"]).dt.normalize()

    if scenario == "no_momentum":
        directions = _reduced_composite_directions(contract)
        composite_components = len(directions)
    else:
        directions = None
        composite_components = len(COMPOSITE_FEATURE_DIRECTIONS)

    blocks = []

    for fold in metadata.sort_values("test_date").itertuples(index=False):
        test_date = pd.Timestamp(fold.test_date)
        test = data.loc[data["as_of_date"].eq(test_date)].sort_values("ticker").copy()

        expected_rows = int(fold.test_rows)
        if len(test) != expected_rows:
            raise ValueError(
                f"{fold.fold_id} expected {expected_rows} test rows but found {len(test)}."
            )

        if scenario == "no_momentum":
            prediction = _score_reduced_composite(
                test,
                directions,
            )
        else:
            prediction = score_technical_composite(
                test,
                minimum_components=6,
            )

        if prediction.isna().any():
            raise ValueError(
                f"{scenario} technical composite produced missing values for {fold.fold_id}."
            )

        block = test.loc[
            :,
            [
                "as_of_date",
                "ticker",
                "sector",
                "target_21d_excess",
                "label_top_quintile",
            ],
        ].copy()
        block.insert(0, "fold_id", str(fold.fold_id))
        block["model_name"] = "technical_equal_weight_composite"
        block["prediction"] = prediction.astype(float)
        blocks.append(block.loc[:, BASELINE_PREDICTION_COLUMNS])

    return (
        pd.concat(blocks, ignore_index=True),
        composite_components,
    )


def _check_rows(
    predictions: pd.DataFrame,
    model_name: str,
) -> int:
    """Return violations in expected OOS row count for one model."""
    rows = predictions.loc[predictions["model_name"].eq(model_name)]
    return int(len(rows) != EXPECTED_PREDICTION_ROWS)


def _build_checks(
    *,
    scenario: str,
    feature_columns: tuple[str, ...],
    composite_components: int,
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    """Audit one completed ablation retraining run."""
    expected_feature_count = SCENARIO_EXPECTED_FEATURES[scenario]
    expected_composite_components = 6 if scenario == "no_momentum" else 8
    expected_models = {
        "technical_equal_weight_composite",
        "elastic_net",
        "lightgbm_ranker",
    }
    observed_models = set(predictions["model_name"].astype(str))

    prediction_dates = set(pd.to_datetime(predictions["as_of_date"]).dt.normalize())
    fold_dates = set(pd.to_datetime(folds["test_date"]).dt.normalize())

    duplicate_predictions = int(
        predictions.duplicated(
            [
                "fold_id",
                "ticker",
                "model_name",
            ]
        ).sum()
    )
    numeric_predictions = pd.to_numeric(
        predictions["prediction"],
        errors="coerce",
    ).to_numpy(dtype=float)
    nonfinite_predictions = int((~np.isfinite(numeric_predictions)).sum())

    checks = [
        (
            "expected_feature_count",
            int(len(feature_columns) != expected_feature_count),
            (f"{scenario} must use exactly {expected_feature_count} predictors."),
        ),
        (
            "expected_composite_components",
            int(composite_components != expected_composite_components),
            ("The technical composite must use the frozen surviving component set."),
        ),
        (
            "expected_models",
            int(observed_models != expected_models),
            (
                "Ablation ensemble predictions must contain "
                "technical composite, Elastic Net and LightGBM Ranker."
            ),
        ),
        (
            "composite_oos_rows",
            _check_rows(
                predictions,
                "technical_equal_weight_composite",
            ),
            (f"Technical composite must contain {EXPECTED_PREDICTION_ROWS} OOS predictions."),
        ),
        (
            "elastic_net_oos_rows",
            _check_rows(predictions, "elastic_net"),
            (f"Elastic Net must contain {EXPECTED_PREDICTION_ROWS} OOS predictions."),
        ),
        (
            "lightgbm_ranker_oos_rows",
            _check_rows(
                predictions,
                "lightgbm_ranker",
            ),
            (f"LightGBM Ranker must contain {EXPECTED_PREDICTION_ROWS} OOS predictions."),
        ),
        (
            "oos_dates_match_folds",
            int(prediction_dates != fold_dates),
            ("Prediction dates must match the frozen walk-forward test dates exactly."),
        ),
        (
            "unique_prediction_keys",
            duplicate_predictions,
            ("Predictions must be unique by fold, ticker and model."),
        ),
        (
            "finite_predictions",
            nonfinite_predictions,
            "All stored OOS predictions must be finite.",
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
            for name, violations, description in checks
        ]
    )


def _format_seconds(seconds: float) -> str:
    """Format elapsed seconds."""
    minutes = seconds / 60.0
    return f"{seconds:.1f} seconds ({minutes:.1f} minutes)"


def _build_report(
    *,
    scenario: str,
    feature_columns: tuple[str, ...],
    composite_components: int,
    predictions: pd.DataFrame,
    checks: pd.DataFrame,
    elapsed_seconds: float,
) -> str:
    """Build a concise retraining report."""
    model_counts = predictions.groupby("model_name").size().rename("prediction_rows").reset_index()

    return "\n".join(
        [
            "# Feature-Family Ablation Retraining",
            "",
            f"- Scenario: `{scenario}`",
            (f"- Predictor count: `{len(feature_columns)}`"),
            (f"- Technical composite components: `{composite_components}`"),
            (f"- OOS dates: `{predictions['as_of_date'].nunique()}`"),
            (f"- OOS companies: `{predictions['ticker'].nunique()}`"),
            (f"- Elapsed time: `{_format_seconds(elapsed_seconds)}`"),
            "",
            "## Methodology",
            "",
            ("- Uses the frozen feature-family contract and the original 77 walk-forward folds."),
            (
                "- Elastic Net and LightGBM Ranker retain their "
                "original candidate hyperparameter grids."
            ),
            (
                "- Hyperparameters continue to be selected only "
                "inside the validation interval of each fold."
            ),
            (
                "- The no-momentum technical composite removes "
                "the frozen momentum inputs and retains the signed "
                "equal-weight construction over the six surviving components."
            ),
            "",
            "## OOS prediction rows",
            "",
            model_counts.to_string(index=False),
            "",
            "## Readiness checks",
            "",
            checks.to_string(index=False),
            "",
        ]
    )


def _scenario_paths(
    scenario: str,
) -> dict[str, Path]:
    """Return all output paths for one scenario."""
    return {
        "predictions": (OUTPUT_DIR / f"{scenario}_ensemble_predictions.parquet"),
        "linear_predictions": (OUTPUT_DIR / f"{scenario}_regularized_linear_predictions.parquet"),
        "linear_hyperparameters": (
            TABLES_DIR / f"{scenario}_regularized_linear_hyperparameters.csv"
        ),
        "linear_coefficients": (TABLES_DIR / f"{scenario}_regularized_linear_coefficients.csv"),
        "ranker_predictions": (OUTPUT_DIR / f"{scenario}_lightgbm_ranker_predictions.parquet"),
        "ranker_hyperparameters": (TABLES_DIR / f"{scenario}_lightgbm_ranker_hyperparameters.csv"),
        "ranker_importance": (TABLES_DIR / f"{scenario}_lightgbm_ranker_feature_importance.csv"),
        "feature_columns": (TABLES_DIR / f"{scenario}_feature_columns.csv"),
        "checks": (TABLES_DIR / f"{scenario}_retraining_checks.csv"),
        "report": (REPORT_DIR / f"{scenario}_retraining.md"),
    }


def _run_scenario(
    panel: pd.DataFrame,
    folds: pd.DataFrame,
    contract: pd.DataFrame,
    scenario: str,
) -> ScenarioOutputs:
    """Retrain one frozen feature-family ablation."""
    logger = logging.getLogger("quant_equity")
    started = time.perf_counter()

    reduced_panel, feature_columns = _reduced_panel(
        panel,
        contract,
        scenario,
    )
    feature_count = len(feature_columns)

    logger.info(
        "%s: prepared reduced modeling panel with %d predictors.",
        scenario,
        feature_count,
    )

    print()
    print(f"[{scenario}] predictors: {feature_count}")
    print(f"[{scenario}] training regularized linear models across {len(folds)} folds...")
    print("This stage can be quiet for several minutes.")

    linear_started = time.perf_counter()
    linear_outputs = train_regularized_linear_models(
        reduced_panel,
        folds,
        config=RegularizedLinearConfig(
            expected_feature_count=feature_count,
        ),
    )
    linear_elapsed = time.perf_counter() - linear_started

    print(
        f"[{scenario}] regularized linear training completed in {_format_seconds(linear_elapsed)}."
    )
    print(f"[{scenario}] training LightGBM Ranker across {len(folds)} folds...")
    print("This is normally the longest stage and may also be quiet.")

    ranker_started = time.perf_counter()
    ranker_outputs = train_lightgbm_ranking(
        reduced_panel,
        folds,
        config=LightGBMRankingConfig(
            expected_feature_count=feature_count,
        ),
    )
    ranker_elapsed = time.perf_counter() - ranker_started

    print(f"[{scenario}] LightGBM Ranker training completed in {_format_seconds(ranker_elapsed)}.")
    print(f"[{scenario}] building technical composite OOS predictions...")

    (
        composite_predictions,
        composite_components,
    ) = _build_composite_predictions(
        reduced_panel,
        folds,
        contract,
        scenario,
    )

    elastic_predictions = linear_outputs.predictions.loc[
        linear_outputs.predictions["model_name"].eq("elastic_net")
    ].copy()
    ranker_predictions = ranker_outputs.predictions.copy()

    ensemble_predictions = (
        pd.concat(
            [
                composite_predictions,
                elastic_predictions.loc[
                    :,
                    BASELINE_PREDICTION_COLUMNS,
                ],
                ranker_predictions.loc[
                    :,
                    BASELINE_PREDICTION_COLUMNS,
                ],
            ],
            ignore_index=True,
        )
        .sort_values(
            [
                "as_of_date",
                "model_name",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    checks = _build_checks(
        scenario=scenario,
        feature_columns=feature_columns,
        composite_components=composite_components,
        predictions=ensemble_predictions,
        folds=folds,
    )

    failed_checks = int(checks["status"].eq("FAIL").sum())

    if failed_checks:
        print()
        print(checks.to_string(index=False))
        raise ValueError(
            f"{scenario} retraining validation failed with {failed_checks} failed checks."
        )

    paths = _scenario_paths(scenario)

    _write_parquet(
        ensemble_predictions,
        paths["predictions"],
    )
    _write_parquet(
        linear_outputs.predictions,
        paths["linear_predictions"],
    )
    _write_csv(
        linear_outputs.hyperparameter_search,
        paths["linear_hyperparameters"],
    )
    _write_csv(
        linear_outputs.coefficients,
        paths["linear_coefficients"],
    )
    _write_parquet(
        ranker_outputs.predictions,
        paths["ranker_predictions"],
    )
    _write_csv(
        ranker_outputs.hyperparameter_search,
        paths["ranker_hyperparameters"],
    )
    _write_csv(
        ranker_outputs.feature_importance,
        paths["ranker_importance"],
    )
    _write_csv(
        pd.DataFrame({"feature": feature_columns}),
        paths["feature_columns"],
    )
    _write_csv(
        checks,
        paths["checks"],
    )

    elapsed_seconds = time.perf_counter() - started

    paths["report"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths["report"].write_text(
        _build_report(
            scenario=scenario,
            feature_columns=feature_columns,
            composite_components=composite_components,
            predictions=ensemble_predictions,
            checks=checks,
            elapsed_seconds=elapsed_seconds,
        ),
        encoding="utf-8",
    )

    logger.info(
        "%s retraining completed.",
        scenario,
    )

    print()
    print(f"[{scenario}] completed.")
    print(f"[{scenario}] total elapsed: {_format_seconds(elapsed_seconds)}")
    print(f"[{scenario}] OOS ensemble rows: {len(ensemble_predictions)}")
    print(f"[{scenario}] composite components: {composite_components}")
    print(f"[{scenario}] readiness checks: {len(checks)}")
    print(f"[{scenario}] failed readiness checks: {failed_checks}")
    print(f"[{scenario}] predictions: {paths['predictions']}")
    print(f"[{scenario}] report: {paths['report']}")

    return ScenarioOutputs(
        scenario=scenario,
        feature_count=feature_count,
        composite_components=composite_components,
        elapsed_seconds=elapsed_seconds,
        predictions_path=paths["predictions"],
        checks_path=paths["checks"],
        report_path=paths["report"],
    )


def _parse_args() -> argparse.Namespace:
    """Parse the requested ablation scenario."""
    parser = argparse.ArgumentParser(
        description=("Retrain frozen feature-family ablations on the original walk-forward folds.")
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "no_fundamentals",
            "no_momentum",
            "all",
        ),
        default="all",
        help=("Ablation scenario to retrain. Use one scenario first for a controlled run."),
    )
    return parser.parse_args()


def main() -> None:
    """Run feature-family ablation retraining."""
    configure_logging()
    args = _parse_args()

    panel, folds, contract = _load_inputs()

    scenarios = tuple(SCENARIO_FLAGS) if args.scenario == "all" else (args.scenario,)

    print()
    print("Institutional Quant Equity Research Platform")
    print("Feature-family ablation retraining")
    print("------------------------------------------------")
    print(f"scenarios: {', '.join(scenarios)}")
    print(f"walk_forward_folds: {len(folds)}")
    print(f"expected_cross_section_size: {EXPECTED_CROSS_SECTION_SIZE}")

    outputs = []
    for scenario in scenarios:
        outputs.append(
            _run_scenario(
                panel,
                folds,
                contract,
                scenario,
            )
        )

    print()
    print("Completed scenarios")
    print("------------------------------------------------")
    for output in outputs:
        print(
            f"{output.scenario}: "
            f"{output.feature_count} predictors, "
            f"{output.composite_components} composite components, "
            f"{_format_seconds(output.elapsed_seconds)}"
        )


if __name__ == "__main__":
    main()
