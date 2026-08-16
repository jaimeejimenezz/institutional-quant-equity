"""Tests for walk-forward regularized linear models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_equity.models import (
    RegularizedLinearConfig,
    RegularizedLinearError,
    detect_model_features,
    fit_feature_preprocessor,
    train_regularized_linear_models,
)

FEATURE_COLUMNS = (
    "tech__signal_a",
    "tech__signal_b",
    "fund__quality",
    "fund__quality_missing",
)


def make_panel() -> pd.DataFrame:
    """Create a compact synthetic modeling panel."""
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

    rows = []

    for date_number, date in enumerate(dates):
        for ticker_number in range(5):
            signal_a = float(ticker_number - 2)

            signal_b = float(2 - ticker_number)

            quality_missing = float(ticker_number == 0)

            quality = np.nan if quality_missing else float(ticker_number + 0.1 * date_number)

            target = 0.02 * signal_a - 0.01 * signal_b + 0.002 * date_number

            rows.append(
                {
                    "as_of_date": date,
                    "target_end_date": (date + pd.Timedelta(days=1)),
                    "ticker": (f"T{ticker_number}"),
                    "sector": ("Technology"),
                    "target_21d_excess": (target),
                    "label_top_quintile": (int(ticker_number == 4)),
                    "tech__signal_a": (signal_a),
                    "tech__signal_b": (signal_b),
                    "fund__quality": (quality),
                    "fund__quality_missing": (quality_missing),
                }
            )

    return pd.DataFrame(rows)


def make_folds() -> pd.DataFrame:
    """Create one synthetic walk-forward fold."""
    return pd.DataFrame(
        {
            "fold_id": ["fold_0001"],
            "test_date": pd.to_datetime(["2020-06-30"]),
            "train_start_date": (pd.to_datetime(["2020-01-31"])),
            "train_end_date": (pd.to_datetime(["2020-03-31"])),
            "validation_start_date": (pd.to_datetime(["2020-04-30"])),
            "validation_end_date": (pd.to_datetime(["2020-05-29"])),
            "test_rows": [5],
        }
    )


def make_config() -> RegularizedLinearConfig:
    """Create a compact configuration for tests."""
    return RegularizedLinearConfig(
        expected_feature_count=4,
        ridge_alphas=(
            0.1,
            1.0,
        ),
        elastic_net_alphas=(
            0.0001,
            0.001,
        ),
        elastic_net_l1_ratios=(
            0.1,
            0.9,
        ),
        minimum_validation_dates=2,
        elastic_net_max_iter=10_000,
    )


def test_model_feature_contract_is_detected() -> None:
    """Technical and fundamental predictors must be detected."""
    features = detect_model_features(
        make_panel(),
        expected_count=4,
    )

    assert features == FEATURE_COLUMNS


def test_preprocessor_uses_training_data_only() -> None:
    """Validation or test values must not affect fitted parameters."""
    panel = make_panel()

    train = panel.loc[panel["as_of_date"].le(pd.Timestamp("2020-03-31"))]

    first = fit_feature_preprocessor(
        train,
        FEATURE_COLUMNS,
    )

    modified = panel.copy()

    modified.loc[
        modified["as_of_date"].gt(pd.Timestamp("2020-03-31")),
        "tech__signal_a",
    ] = 1_000_000.0

    second_train = modified.loc[modified["as_of_date"].le(pd.Timestamp("2020-03-31"))]

    second = fit_feature_preprocessor(
        second_train,
        FEATURE_COLUMNS,
    )

    assert np.allclose(
        first.means.to_numpy(),
        second.means.to_numpy(),
    )

    assert np.allclose(
        first.scales.to_numpy(),
        second.scales.to_numpy(),
    )


def test_missing_indicators_remain_binary() -> None:
    """Missing flags must retain their 0/1 meaning."""
    panel = make_panel()

    train = panel.loc[panel["as_of_date"].le(pd.Timestamp("2020-03-31"))]

    preprocessor = fit_feature_preprocessor(
        train,
        FEATURE_COLUMNS,
    )

    transformed = preprocessor.transform(panel)

    assert set(transformed["fund__quality_missing"].unique()).issubset(
        {
            0.0,
            1.0,
        }
    )


def test_training_generates_two_models_per_test_security() -> None:
    """Every test security must receive Ridge and Elastic Net predictions."""
    outputs = train_regularized_linear_models(
        make_panel(),
        make_folds(),
        config=make_config(),
    )

    assert len(outputs.predictions) == 10

    assert set(outputs.predictions["model_name"].unique()) == {
        "ridge",
        "elastic_net",
    }

    assert (
        outputs.predictions.duplicated(
            [
                "fold_id",
                "ticker",
                "model_name",
            ]
        ).sum()
        == 0
    )


def test_test_targets_do_not_influence_model_predictions() -> None:
    """Changing test outcomes must not alter fitted models or predictions."""
    original = make_panel()

    changed = original.copy()

    test_mask = changed["as_of_date"].eq(pd.Timestamp("2020-06-30"))

    changed.loc[
        test_mask,
        "target_21d_excess",
    ] = np.linspace(
        100.0,
        500.0,
        test_mask.sum(),
    )

    first = train_regularized_linear_models(
        original,
        make_folds(),
        config=make_config(),
    )

    second = train_regularized_linear_models(
        changed,
        make_folds(),
        config=make_config(),
    )

    first_predictions = first.predictions.sort_values(
        [
            "model_name",
            "ticker",
        ]
    )["prediction"].to_numpy()

    second_predictions = second.predictions.sort_values(
        [
            "model_name",
            "ticker",
        ]
    )["prediction"].to_numpy()

    assert np.allclose(
        first_predictions,
        second_predictions,
    )

    first_selected = (
        first.hyperparameter_search.loc[first.hyperparameter_search["selected"]]
        .sort_values("model_name")[
            [
                "model_name",
                "alpha",
                "l1_ratio",
            ]
        ]
        .reset_index(drop=True)
    )

    second_selected = (
        second.hyperparameter_search.loc[second.hyperparameter_search["selected"]]
        .sort_values("model_name")[
            [
                "model_name",
                "alpha",
                "l1_ratio",
            ]
        ]
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        first_selected,
        second_selected,
    )


def test_immature_fitting_target_is_rejected() -> None:
    """Training must reject labels that were not mature at prediction time."""
    panel = make_panel()

    panel.loc[
        panel["as_of_date"].eq(pd.Timestamp("2020-05-29")),
        "target_end_date",
    ] = pd.Timestamp("2020-07-15")

    try:
        train_regularized_linear_models(
            panel,
            make_folds(),
            config=make_config(),
        )
    except RegularizedLinearError as exc:
        assert "not mature" in str(exc)
    else:
        raise AssertionError("Immature fitting labels were not rejected.")
