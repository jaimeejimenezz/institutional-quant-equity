"""Tests for initial linear models."""

import numpy as np
import pandas as pd
import pytest

from quant_equity.models import (
    LinearModelsConfig,
    LinearModelsError,
    run_linear_models_walk_forward,
)
from quant_equity.validation import (
    LinearModelingConfig,
    build_expanding_walk_forward_folds,
)

FEATURE_COLUMNS = (
    "feature_a",
    "feature_b",
)

FEATURE_DIRECTIONS = {
    "feature_a": 1.0,
    "feature_b": -1.0,
}


def make_model_config() -> LinearModelsConfig:
    """Create compact model configuration."""
    return LinearModelsConfig(
        target_column="target_21d_excess",
        top_label_column="label_top_quintile",
        momentum_signal="feature_a",
        ridge_alphas=(
            0.1,
            1.0,
        ),
        elastic_net_alphas=(
            0.001,
            0.01,
        ),
        elastic_net_l1_ratios=(
            0.2,
            0.8,
        ),
        max_iterations=10_000,
        tolerance=0.000001,
    )


def make_validation_config() -> LinearModelingConfig:
    """Create compact walk-forward configuration."""
    return LinearModelingConfig(
        target_column="target_21d_excess",
        top_label_column="label_top_quintile",
        out_of_sample_start_date=pd.Timestamp("2020-05-01"),
        validation_months=1,
        minimum_training_months=2,
        minimum_cross_section_size=2,
    )


def make_panel() -> pd.DataFrame:
    """Create a small predictive panel."""
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-28",
            "2020-03-31",
            "2020-04-30",
            "2020-05-29",
            "2020-06-30",
        ]
    )

    target_end_dates = pd.to_datetime(
        [
            "2020-02-28",
            "2020-03-31",
            "2020-04-30",
            "2020-05-29",
            "2020-06-30",
            "2020-07-31",
        ]
    )

    tickers = (
        "AAA",
        "BBB",
        "CCC",
        "DDD",
    )

    feature_values = (
        -1.5,
        -0.5,
        0.5,
        1.5,
    )

    rows = []

    for date_number, (
        as_of_date,
        target_end_date,
    ) in enumerate(
        zip(
            dates,
            target_end_dates,
            strict=True,
        ),
        start=1,
    ):
        for ticker_number, (
            ticker,
            feature_a,
        ) in enumerate(
            zip(
                tickers,
                feature_values,
                strict=True,
            ),
            start=1,
        ):
            feature_b = -feature_a + date_number * 0.01

            target = 0.03 * feature_a - 0.01 * feature_b

            rows.append(
                {
                    "as_of_date": as_of_date,
                    "ticker": ticker,
                    "sector": ("Technology" if ticker_number <= 2 else "Financials"),
                    "target_end_date": (target_end_date),
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "target_21d_excess": target,
                    "label_top_quintile": int(ticker == "DDD"),
                }
            )

    panel = pd.DataFrame(rows)

    panel.loc[
        panel.index[0],
        "feature_b",
    ] = np.nan

    return panel


def make_folds(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Build folds from the compact panel."""
    return build_expanding_walk_forward_folds(
        panel,
        config=make_validation_config(),
    )


def run_models() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run models on the compact fixture."""
    panel = make_panel()
    folds = make_folds(panel)

    return run_linear_models_walk_forward(
        panel,
        folds,
        feature_columns=FEATURE_COLUMNS,
        feature_directions=FEATURE_DIRECTIONS,
        validation_months=1,
        config=make_model_config(),
    )


def test_invalid_feature_directions_are_rejected() -> None:
    """Every selected feature needs a valid direction."""
    panel = make_panel()
    folds = make_folds(panel)

    with pytest.raises(
        LinearModelsError,
        match="exactly match",
    ):
        run_linear_models_walk_forward(
            panel,
            folds,
            feature_columns=FEATURE_COLUMNS,
            feature_directions={
                "feature_a": 1.0,
            },
            validation_months=1,
            config=make_model_config(),
        )


def test_all_expected_models_generate_predictions() -> None:
    """Five initial models must generate predictions."""
    predictions, _, _ = run_models()

    assert set(predictions["model_name"]) == {
        "constant",
        "momentum_3m",
        "equal_weight_composite",
        "ridge",
        "elastic_net",
    }


def test_predictions_are_out_of_sample_only() -> None:
    """Predictions must exist only for fold test dates."""
    panel = make_panel()
    folds = make_folds(panel)

    predictions, _, _ = run_linear_models_walk_forward(
        panel,
        folds,
        feature_columns=FEATURE_COLUMNS,
        feature_directions=(FEATURE_DIRECTIONS),
        validation_months=1,
        config=make_model_config(),
    )

    assert set(predictions["as_of_date"]) == set(folds["test_date"])


def test_prediction_row_count_is_complete() -> None:
    """Every model must predict every test company."""
    panel = make_panel()
    folds = make_folds(panel)

    predictions, _, _ = run_linear_models_walk_forward(
        panel,
        folds,
        feature_columns=FEATURE_COLUMNS,
        feature_directions=(FEATURE_DIRECTIONS),
        validation_months=1,
        config=make_model_config(),
    )

    expected_rows = len(folds) * 4 * 5

    assert len(predictions) == expected_rows


def test_missing_features_are_imputed() -> None:
    """Missing historical values must not create missing predictions."""
    predictions, _, _ = run_models()

    assert predictions["prediction"].notna().all()

    assert np.isfinite(predictions["prediction"].to_numpy(dtype=float)).all()


def test_hyperparameters_are_selected_per_fold() -> None:
    """Each fold selects one Ridge and one Elastic Net configuration."""
    _, validation_grid, _ = run_models()

    selected = validation_grid.loc[validation_grid["selected"]]

    selected_counts = selected.groupby(
        [
            "fold_id",
            "model_name",
        ]
    ).size()

    assert selected_counts.eq(1).all()
    assert set(selected["model_name"]) == {
        "ridge",
        "elastic_net",
    }


def test_coefficients_are_saved_for_regularized_models() -> None:
    """Every regularized model must expose every feature coefficient."""
    _, _, coefficients = run_models()

    assert set(coefficients["model_name"]) == {
        "ridge",
        "elastic_net",
    }

    counts = coefficients.groupby(
        [
            "fold_id",
            "model_name",
        ]
    )["feature"].nunique()

    assert counts.eq(len(FEATURE_COLUMNS)).all()


def test_no_future_targets_enter_model_fit() -> None:
    """Every fitted target must be known on the prediction date."""
    predictions, _, _ = run_models()

    assert (predictions["latest_fit_target_end_date"] <= predictions["as_of_date"]).all()
