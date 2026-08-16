"""Tests for leakage-safe fold preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_equity.validation import (
    FoldPreprocessingError,
    fit_fold_preprocessor,
)

FEATURES = (
    "signal_a",
    "signal_b",
    "fund__metric_missing",
)


def make_train() -> pd.DataFrame:
    """Create synthetic training features."""
    return pd.DataFrame(
        {
            "signal_a": [
                1.0,
                2.0,
                np.nan,
                4.0,
            ],
            "signal_b": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "fund__metric_missing": [
                0,
                0,
                1,
                0,
            ],
        }
    )


def test_training_median_is_used_for_imputation() -> None:
    """Missing values must use a median learned from train only."""
    train = make_train()

    preprocessor = fit_fold_preprocessor(
        train,
        feature_columns=FEATURES,
    )

    assert preprocessor.imputation_values["signal_a"] == 2.0

    transformed = preprocessor.transform(train)

    assert not transformed.isna().any().any()


def test_validation_values_do_not_change_fitted_parameters() -> None:
    """Future validation values must not affect train fitting."""
    train = make_train()

    first = fit_fold_preprocessor(
        train,
        feature_columns=FEATURES,
    )

    future = pd.DataFrame(
        {
            "signal_a": [
                1_000_000.0,
            ],
            "signal_b": [
                -1_000_000.0,
            ],
            "fund__metric_missing": [
                0,
            ],
        }
    )

    _ = first.transform(future)

    second = fit_fold_preprocessor(
        train,
        feature_columns=FEATURES,
    )

    assert first.imputation_values == second.imputation_values

    assert first.means == second.means

    assert first.scales == second.scales


def test_transformed_train_is_scaled() -> None:
    """Continuous training features should be centered and scaled."""
    train = make_train()

    preprocessor = fit_fold_preprocessor(
        train,
        feature_columns=FEATURES,
    )

    transformed = preprocessor.transform(train)

    for column in (
        "signal_a",
        "signal_b",
    ):
        assert np.isclose(
            transformed[column].mean(),
            0.0,
        )

        assert np.isclose(
            transformed[column].std(ddof=0),
            1.0,
        )


def test_missing_indicator_is_not_scaled() -> None:
    """Binary missing indicators must retain 0/1 meaning."""
    train = make_train()

    preprocessor = fit_fold_preprocessor(
        train,
        feature_columns=FEATURES,
    )

    transformed = preprocessor.transform(train)

    assert transformed["fund__metric_missing"].tolist() == [
        0.0,
        0.0,
        1.0,
        0.0,
    ]


def test_all_missing_train_feature_is_excluded() -> None:
    """A feature unavailable in training must not appear later."""
    train = make_train()

    train["signal_a"] = np.nan

    preprocessor = fit_fold_preprocessor(
        train,
        feature_columns=FEATURES,
    )

    assert "signal_a" in preprocessor.unavailable_features

    assert "signal_a" not in preprocessor.active_features

    test = pd.DataFrame(
        {
            "signal_a": [
                999.0,
            ],
            "signal_b": [
                25.0,
            ],
            "fund__metric_missing": [
                0,
            ],
        }
    )

    transformed = preprocessor.transform(test)

    assert "signal_a" not in transformed.columns


def test_infinite_value_fails() -> None:
    """Infinite predictors must block preprocessing."""
    train = make_train()

    train.loc[
        train.index[0],
        "signal_a",
    ] = np.inf

    with pytest.raises(FoldPreprocessingError):
        fit_fold_preprocessor(
            train,
            feature_columns=FEATURES,
        )


def test_missing_indicator_nan_fails() -> None:
    """Missing flags themselves must always be observed."""
    train = make_train()

    train.loc[
        train.index[0],
        "fund__metric_missing",
    ] = np.nan

    with pytest.raises(FoldPreprocessingError):
        fit_fold_preprocessor(
            train,
            feature_columns=FEATURES,
        )
