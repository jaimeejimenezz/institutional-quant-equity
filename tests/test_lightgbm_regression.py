"""Tests for walk-forward LightGBM regression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_equity.models import (
    LightGBMCandidate,
    LightGBMRegressionConfig,
    LightGBMRegressionError,
    summarize_lightgbm_importance,
    train_lightgbm_regression,
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
        for ticker_number in range(10):
            signal_a = float(ticker_number - 4.5)

            signal_b = float(4.5 - ticker_number)

            quality_missing = float(ticker_number == 0)

            quality = np.nan if quality_missing else float(ticker_number + 0.1 * date_number)

            target = (
                0.015 * signal_a
                - 0.007 * signal_b
                + 0.003 * np.square(signal_a)
                + 0.001 * date_number
            )

            rows.append(
                {
                    "as_of_date": date,
                    "target_end_date": (date + pd.Timedelta(days=1)),
                    "ticker": (f"T{ticker_number:02d}"),
                    "sector": "Technology",
                    "target_21d_excess": (target),
                    "label_top_quintile": int(ticker_number >= 8),
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
            "train_start_date": pd.to_datetime(["2020-01-31"]),
            "train_end_date": pd.to_datetime(["2020-03-31"]),
            "validation_start_date": pd.to_datetime(["2020-04-30"]),
            "validation_end_date": pd.to_datetime(["2020-05-29"]),
            "test_rows": [10],
        }
    )


def make_config() -> LightGBMRegressionConfig:
    """Create a fast deterministic configuration for tests."""
    candidate = LightGBMCandidate(
        candidate_name="test_candidate",
        num_leaves=4,
        max_depth=2,
        min_child_samples=5,
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        colsample_bytree=1.0,
        subsample=1.0,
    )

    return LightGBMRegressionConfig(
        expected_feature_count=4,
        max_estimators=50,
        early_stopping_rounds=5,
        minimum_validation_dates=2,
        random_state=42,
        n_jobs=1,
        candidates=(candidate,),
    )


def test_training_generates_one_prediction_per_security() -> None:
    """Every test security must receive one LightGBM prediction."""
    outputs = train_lightgbm_regression(
        make_panel(),
        make_folds(),
        config=make_config(),
    )

    assert len(outputs.predictions) == 10

    assert set(outputs.predictions["model_name"].unique()) == {"lightgbm_regressor"}

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


def test_test_targets_do_not_influence_predictions() -> None:
    """Changing OOS outcomes must not change fitted predictions."""
    original = make_panel()

    changed = original.copy()

    test_mask = changed["as_of_date"].eq(pd.Timestamp("2020-06-30"))

    changed.loc[
        test_mask,
        "target_21d_excess",
    ] = np.linspace(
        100.0,
        1000.0,
        test_mask.sum(),
    )

    first = train_lightgbm_regression(
        original,
        make_folds(),
        config=make_config(),
    )

    second = train_lightgbm_regression(
        changed,
        make_folds(),
        config=make_config(),
    )

    assert np.allclose(
        first.predictions["prediction"].to_numpy(),
        second.predictions["prediction"].to_numpy(),
    )

    first_selected = first.hyperparameter_search.loc[first.hyperparameter_search["selected"]][
        [
            "candidate_name",
            "best_iteration",
        ]
    ].reset_index(drop=True)

    second_selected = second.hyperparameter_search.loc[second.hyperparameter_search["selected"]][
        [
            "candidate_name",
            "best_iteration",
        ]
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        first_selected,
        second_selected,
    )


def test_immature_fitting_target_is_rejected() -> None:
    """Labels unavailable at prediction time must be rejected."""
    panel = make_panel()

    panel.loc[
        panel["as_of_date"].eq(pd.Timestamp("2020-05-29")),
        "target_end_date",
    ] = pd.Timestamp("2020-07-15")

    try:
        train_lightgbm_regression(
            panel,
            make_folds(),
            config=make_config(),
        )
    except LightGBMRegressionError as exc:
        assert "not mature" in str(exc)
    else:
        raise AssertionError("Immature fitting labels were not rejected.")


def test_feature_importance_covers_every_feature() -> None:
    """Every predictor must receive an importance record."""
    outputs = train_lightgbm_regression(
        make_panel(),
        make_folds(),
        config=make_config(),
    )

    assert len(outputs.feature_importance) == len(FEATURE_COLUMNS)

    assert set(outputs.feature_importance["feature"]) == set(FEATURE_COLUMNS)


def test_importance_summary_is_sorted_by_gain_share() -> None:
    """Importance summary must rank features by average gain."""
    outputs = train_lightgbm_regression(
        make_panel(),
        make_folds(),
        config=make_config(),
    )

    summary = summarize_lightgbm_importance(outputs.feature_importance)

    gain_share = summary["mean_gain_share"].to_numpy()

    assert np.all(gain_share[:-1] >= gain_share[1:])
