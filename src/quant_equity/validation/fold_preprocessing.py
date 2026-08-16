"""Leakage-safe preprocessing fitted inside walk-forward folds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_equity.models.modeling_panel import (
    MODEL_FEATURE_COLUMNS,
)
from quant_equity.validation.walk_forward import (
    WalkForwardFold,
    split_panel_by_fold,
)


class FoldPreprocessingError(ValueError):
    """Raised when fold-local preprocessing cannot be performed."""


@dataclass(frozen=True)
class FittedFoldPreprocessor:
    """Preprocessing parameters learned exclusively from training data."""

    input_features: tuple[str, ...]
    active_features: tuple[str, ...]
    continuous_features: tuple[str, ...]
    indicator_features: tuple[str, ...]
    unavailable_features: tuple[str, ...]
    constant_continuous_features: tuple[str, ...]
    imputation_values: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform data using training-only fitted parameters."""
        missing_columns = sorted(set(self.input_features).difference(data.columns))

        if missing_columns:
            raise FoldPreprocessingError(
                "Data is missing required feature columns: " + ", ".join(missing_columns) + "."
            )

        raw = data.loc[
            :,
            self.input_features,
        ]

        numeric = raw.apply(
            pd.to_numeric,
            errors="coerce",
        )

        non_numeric = raw.notna() & numeric.isna()

        if non_numeric.any().any():
            raise FoldPreprocessingError("Non-numeric predictor values were found.")

        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise FoldPreprocessingError("Infinite predictor values were found.")

        if self.indicator_features:
            indicators = numeric.loc[
                :,
                self.indicator_features,
            ]

            if indicators.isna().any().any():
                raise FoldPreprocessingError("Missing-indicator features must not contain NaN.")

            invalid_indicators = ~indicators.isin(
                [
                    0,
                    1,
                ]
            )

            if invalid_indicators.any().any():
                raise FoldPreprocessingError("Missing-indicator features must contain only 0 or 1.")

        transformed = pd.DataFrame(index=data.index)

        for column in self.active_features:
            values = numeric[column].astype(float)

            if column in (self.indicator_features):
                transformed[column] = values
                continue

            filled = values.fillna(self.imputation_values[column])

            transformed[column] = (filled - self.means[column]) / self.scales[column]

        if transformed.isna().any().any():
            raise FoldPreprocessingError("NaN remained after preprocessing.")

        if np.isinf(transformed.to_numpy(dtype=float)).any():
            raise FoldPreprocessingError("Non-finite values remained after preprocessing.")

        return transformed


def fit_fold_preprocessor(
    train: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
    indicator_suffix: str = "_missing",
) -> FittedFoldPreprocessor:
    """Fit imputation and scaling exclusively on training rows."""
    missing_columns = sorted(set(feature_columns).difference(train.columns))

    if missing_columns:
        raise FoldPreprocessingError(
            "Training data is missing feature columns: " + ", ".join(missing_columns) + "."
        )

    if train.empty:
        raise FoldPreprocessingError("Training data must not be empty.")

    raw = train.loc[
        :,
        feature_columns,
    ]

    numeric = raw.apply(
        pd.to_numeric,
        errors="coerce",
    )

    non_numeric = raw.notna() & numeric.isna()

    if non_numeric.any().any():
        raise FoldPreprocessingError("Training predictors must be numeric.")

    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise FoldPreprocessingError("Training predictors must not contain infinity.")

    indicator_features = tuple(
        column for column in feature_columns if column.endswith(indicator_suffix)
    )

    continuous_candidates = tuple(
        column for column in feature_columns if column not in indicator_features
    )

    if indicator_features:
        indicators = numeric.loc[
            :,
            indicator_features,
        ]

        if indicators.isna().any().any():
            raise FoldPreprocessingError("Training missing indicators must be complete.")

        invalid_indicators = ~indicators.isin(
            [
                0,
                1,
            ]
        )

        if invalid_indicators.any().any():
            raise FoldPreprocessingError("Training missing indicators must contain only 0 or 1.")

    unavailable_features = tuple(
        column for column in continuous_candidates if numeric[column].isna().all()
    )

    continuous_features = tuple(
        column for column in continuous_candidates if column not in unavailable_features
    )

    active_set = set(continuous_features).union(indicator_features)

    active_features = tuple(column for column in feature_columns if column in active_set)

    imputation_values: dict[
        str,
        float,
    ] = {}

    means: dict[
        str,
        float,
    ] = {}

    scales: dict[
        str,
        float,
    ] = {}

    constant_features: list[str] = []

    for column in continuous_features:
        values = numeric[column].astype(float)

        median = float(values.median())

        if not np.isfinite(median):
            raise FoldPreprocessingError(f"Could not compute training median for {column}.")

        filled = values.fillna(median)

        mean = float(filled.mean())

        scale = float(filled.std(ddof=0))

        if not np.isfinite(scale) or scale <= 1.0e-12:
            scale = 1.0
            constant_features.append(column)

        imputation_values[column] = median

        means[column] = mean

        scales[column] = scale

    return FittedFoldPreprocessor(
        input_features=tuple(feature_columns),
        active_features=active_features,
        continuous_features=continuous_features,
        indicator_features=indicator_features,
        unavailable_features=unavailable_features,
        constant_continuous_features=tuple(constant_features),
        imputation_values=imputation_values,
        means=means,
        scales=scales,
    )


def audit_fold_preprocessing(
    panel: pd.DataFrame,
    folds: tuple[
        WalkForwardFold,
        ...,
    ],
    *,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Audit fold-local preprocessing across all walk-forward folds."""
    rows: list[dict[str, object]] = []

    for fold in folds:
        train, validation, test = split_panel_by_fold(
            panel,
            fold,
        )

        preprocessor = fit_fold_preprocessor(
            train,
            feature_columns=feature_columns,
        )

        train_x = preprocessor.transform(train)

        validation_x = preprocessor.transform(validation)

        test_x = preprocessor.transform(test)

        continuous = list(preprocessor.continuous_features)

        non_constant = [
            column
            for column in continuous
            if column not in (preprocessor.constant_continuous_features)
        ]

        if continuous:
            max_abs_train_mean = float(train_x[continuous].mean().abs().max())
        else:
            max_abs_train_mean = 0.0

        if non_constant:
            train_std = train_x[non_constant].std(ddof=0)

            max_abs_train_std_error = float((train_std - 1.0).abs().max())
        else:
            max_abs_train_std_error = 0.0

        raw_train = train.loc[
            :,
            feature_columns,
        ]

        raw_validation = validation.loc[
            :,
            feature_columns,
        ]

        raw_test = test.loc[
            :,
            feature_columns,
        ]

        rows.append(
            {
                "fold_id": fold.fold_id,
                "test_date": fold.test_date,
                "candidate_features": len(feature_columns),
                "active_features": len(preprocessor.active_features),
                "continuous_features": len(preprocessor.continuous_features),
                "indicator_features": len(preprocessor.indicator_features),
                "unavailable_features": len(preprocessor.unavailable_features),
                "constant_continuous_features": len(preprocessor.constant_continuous_features),
                "train_missing_before": int(raw_train.isna().sum().sum()),
                "validation_missing_before": int(raw_validation.isna().sum().sum()),
                "test_missing_before": int(raw_test.isna().sum().sum()),
                "train_missing_after": int(train_x.isna().sum().sum()),
                "validation_missing_after": int(validation_x.isna().sum().sum()),
                "test_missing_after": int(test_x.isna().sum().sum()),
                "train_non_finite_after": int(np.isinf(train_x.to_numpy(dtype=float)).sum()),
                "validation_non_finite_after": int(
                    np.isinf(validation_x.to_numpy(dtype=float)).sum()
                ),
                "test_non_finite_after": int(np.isinf(test_x.to_numpy(dtype=float)).sum()),
                "max_abs_train_scaled_mean": (max_abs_train_mean),
                "max_abs_train_scaled_std_error": (max_abs_train_std_error),
            }
        )

    return pd.DataFrame(rows)
