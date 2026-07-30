"""Train initial baselines and regularized linear models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_equity.config import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.features import (
    SELECTED_TECHNICAL_FEATURE_COLUMNS,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.models import (
    TECHNICAL_COMPOSITE_DIRECTIONS,
    LinearModelsConfig,
    run_linear_models_walk_forward,
)

PANEL_PATH = PROCESSED_DATA_DIR / "technical_modeling_panel.parquet"

FOLDS_PATH = PROCESSED_DATA_DIR / "linear_walk_forward_folds.parquet"

PREDICTIONS_PATH = PROCESSED_DATA_DIR / "predictions_linear_oos.parquet"

VALIDATION_GRID_PATH = REPORTS_DIR / "tables" / "linear_model_validation_grid.csv"

COEFFICIENTS_PATH = REPORTS_DIR / "tables" / "linear_model_coefficients.csv"


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
) -> Path:
    """Write a Parquet file atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(".tmp.parquet")

    temporary_path.unlink(missing_ok=True)

    data.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)

    return path


def _write_csv(
    data: pd.DataFrame,
    path: Path,
) -> Path:
    """Write a CSV file."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        path,
        index=False,
    )

    return path


def main() -> None:
    """Execute Step 7B model training."""
    project_config = load_config()

    logger = configure_logging(
        level=project_config["runtime"]["log_level"],
        log_file=(PROJECT_ROOT / "logs" / "train_linear_models.log"),
    )

    for required_path in (
        PANEL_PATH,
        FOLDS_PATH,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required dataset not found: {required_path}")

    model_config = LinearModelsConfig.from_mapping(project_config["linear_models"])

    validation_months = int(project_config["linear_modeling"]["validation_months"])

    panel = pd.read_parquet(PANEL_PATH)

    folds = pd.read_parquet(FOLDS_PATH)

    (
        predictions,
        validation_grid,
        coefficients,
    ) = run_linear_models_walk_forward(
        panel,
        folds,
        feature_columns=(SELECTED_TECHNICAL_FEATURE_COLUMNS),
        feature_directions=(TECHNICAL_COMPOSITE_DIRECTIONS),
        validation_months=validation_months,
        config=model_config,
    )

    predictions_path = _write_parquet_atomically(
        predictions,
        PREDICTIONS_PATH,
    )

    validation_path = _write_csv(
        validation_grid,
        VALIDATION_GRID_PATH,
    )

    coefficients_path = _write_csv(
        coefficients,
        COEFFICIENTS_PATH,
    )

    selected_parameters = validation_grid.loc[validation_grid["selected"]]

    logger.info("Linear model training completed.")

    logger.info(
        "Prediction rows: %s",
        len(predictions),
    )

    logger.info(
        "Validation candidates: %s",
        len(validation_grid),
    )

    logger.info(
        "Coefficient rows: %s",
        len(coefficients),
    )

    print()
    print("Institutional Quant Equity Research Platform")
    print("Initial linear models - Step 7B")
    print("-" * 48)
    print(f"Walk-forward folds: {folds['fold_id'].nunique()}")
    print(f"Out-of-sample dates: {predictions['as_of_date'].nunique()}")
    print(f"Models: {predictions['model_name'].nunique()}")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Validation candidates: {len(validation_grid)}")
    print(f"Selected hyperparameter rows: {len(selected_parameters)}")
    print(f"Coefficient rows: {len(coefficients)}")
    print(
        "Temporal violations: "
        f"{int((predictions['latest_fit_target_end_date'] > predictions['as_of_date']).sum())}"
    )
    print()
    print("Model names:")

    for model_name in sorted(predictions["model_name"].unique()):
        print(f"- {model_name}")

    print()
    print(f"Predictions: {predictions_path}")
    print(f"Validation grid: {validation_path}")
    print(f"Coefficients: {coefficients_path}")
    print()
    print("Initial linear model training: OK")


if __name__ == "__main__":
    main()
