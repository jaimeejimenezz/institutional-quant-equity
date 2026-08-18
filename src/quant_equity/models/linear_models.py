"""Initial baselines and regularized linear models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LinearModelsError(ValueError):
    """Raised when linear-model training cannot be completed."""


TECHNICAL_COMPOSITE_DIRECTIONS: dict[str, float] = {
    "amihud_illiquidity_20d_sector_neutral": 1.0,
    "reversal_1m_sector_neutral": 1.0,
    "volatility_60d_sector_neutral": 1.0,
    "distance_sma_50d_sector_neutral": -1.0,
    "beta_60d_market_sector_neutral": 1.0,
    "return_3m_sector_neutral": 1.0,
    "max_drawdown_126d_sector_neutral": -1.0,
    "average_dollar_volume_20d_sector_neutral": -1.0,
}


@dataclass(frozen=True)
class LinearModelsConfig:
    """Configuration for initial linear models."""

    target_column: str = "target_21d_excess"
    top_label_column: str = "label_top_quintile"
    momentum_signal: str = "return_3m_sector_neutral"
    ridge_alphas: tuple[float, ...] = (
        0.01,
        0.10,
        1.00,
        10.00,
        100.00,
    )
    elastic_net_alphas: tuple[float, ...] = (
        0.0001,
        0.001,
        0.010,
        0.100,
    )
    elastic_net_l1_ratios: tuple[float, ...] = (
        0.10,
        0.50,
        0.90,
    )
    max_iterations: int = 50_000
    tolerance: float = 0.000001

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> LinearModelsConfig:
        """Create configuration from YAML values."""
        config = cls(
            target_column=str(
                values.get(
                    "target_column",
                    "target_21d_excess",
                )
            ),
            top_label_column=str(
                values.get(
                    "top_label_column",
                    "label_top_quintile",
                )
            ),
            momentum_signal=str(
                values.get(
                    "momentum_signal",
                    "return_3m_sector_neutral",
                )
            ),
            ridge_alphas=tuple(
                float(value)
                for value in values.get(
                    "ridge_alphas",
                    (
                        0.01,
                        0.10,
                        1.00,
                        10.00,
                        100.00,
                    ),
                )
            ),
            elastic_net_alphas=tuple(
                float(value)
                for value in values.get(
                    "elastic_net_alphas",
                    (
                        0.0001,
                        0.001,
                        0.010,
                        0.100,
                    ),
                )
            ),
            elastic_net_l1_ratios=tuple(
                float(value)
                for value in values.get(
                    "elastic_net_l1_ratios",
                    (
                        0.10,
                        0.50,
                        0.90,
                    ),
                )
            ),
            max_iterations=int(
                values.get(
                    "max_iterations",
                    50_000,
                )
            ),
            tolerance=float(
                values.get(
                    "tolerance",
                    0.000001,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate model configuration."""
        if not self.target_column:
            raise LinearModelsError(
                "target_column cannot be empty."
            )

        if not self.top_label_column:
            raise LinearModelsError(
                "top_label_column cannot be empty."
            )

        if not self.momentum_signal:
            raise LinearModelsError(
                "momentum_signal cannot be empty."
            )

        if not self.ridge_alphas:
            raise LinearModelsError(
                "At least one Ridge alpha is required."
            )

        if not self.elastic_net_alphas:
            raise LinearModelsError(
                "At least one Elastic Net alpha is required."
            )

        if not self.elastic_net_l1_ratios:
            raise LinearModelsError(
                "At least one Elastic Net l1_ratio is required."
            )

        if any(alpha <= 0.0 for alpha in self.ridge_alphas):
            raise LinearModelsError(
                "Every Ridge alpha must be positive."
            )

        if any(
            alpha <= 0.0
            for alpha in self.elastic_net_alphas
        ):
            raise LinearModelsError(
                "Every Elastic Net alpha must be positive."
            )

        if any(
            not 0.0 <= ratio <= 1.0
            for ratio in self.elastic_net_l1_ratios
        ):
            raise LinearModelsError(
                "Elastic Net l1_ratio values must be in [0, 1]."
            )

        if self.max_iterations < 1:
            raise LinearModelsError(
                "max_iterations must be positive."
            )

        if self.tolerance <= 0.0:
            raise LinearModelsError(
                "tolerance must be positive."
            )


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Require columns in a dataframe."""
    missing = sorted(
        set(columns).difference(data.columns)
    )

    if missing:
        raise LinearModelsError(
            f"{dataset_name} is missing columns: "
            + ", ".join(missing)
            + "."
        )


def _normalize_dates(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Normalize date columns in place."""
    for column in columns:
        data[column] = pd.to_datetime(
            data[column],
            errors="coerce",
        ).dt.normalize()

        if data[column].isna().any():
            raise LinearModelsError(
                f"{dataset_name} contains invalid "
                f"dates in {column}."
            )


def _safe_spearman(
    first: pd.Series,
    second: pd.Series,
) -> float:
    """Calculate Spearman correlation safely."""
    comparable = pd.DataFrame(
        {
            "first": pd.to_numeric(
                first,
                errors="coerce",
            ),
            "second": pd.to_numeric(
                second,
                errors="coerce",
            ),
        }
    ).dropna()

    if (
        len(comparable) < 2
        or comparable["first"].nunique() < 2
        or comparable["second"].nunique() < 2
    ):
        return float("nan")

    first_ranks = comparable["first"].rank(
        method="average"
    )

    second_ranks = comparable["second"].rank(
        method="average"
    )

    return float(
        first_ranks.corr(second_ranks)
    )


def _evaluate_validation_predictions(
    validation_data: pd.DataFrame,
    predictions: np.ndarray,
    *,
    target_column: str,
) -> tuple[float, float, int]:
    """Evaluate validation predictions by monthly IC and RMSE."""
    evaluation = validation_data.loc[
        :,
        [
            "as_of_date",
            target_column,
        ],
    ].copy()

    evaluation["prediction"] = np.asarray(
        predictions,
        dtype=float,
    )

    monthly_ic = (
        evaluation.groupby(
            "as_of_date",
            sort=True,
        )
        .apply(
            lambda month: _safe_spearman(
                month["prediction"],
                month[target_column],
            ),
            include_groups=False,
        )
        .dropna()
    )

    mean_ic = (
        float(monthly_ic.mean())
        if not monthly_ic.empty
        else float("nan")
    )

    errors = (
        evaluation["prediction"]
        - evaluation[target_column]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(errors.to_numpy(dtype=float))
            )
        )
    )

    return (
        mean_ic,
        rmse,
        len(monthly_ic),
    )


def _build_pipeline(
    estimator: Ridge | ElasticNet,
) -> Pipeline:
    """Build a leakage-safe preprocessing and model pipeline."""
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


def _build_ridge(
    alpha: float,
) -> Ridge:
    """Build a Ridge estimator."""
    return Ridge(
        alpha=alpha,
        fit_intercept=True,
    )


def _build_elastic_net(
    *,
    alpha: float,
    l1_ratio: float,
    config: LinearModelsConfig,
) -> ElasticNet:
    """Build an Elastic Net estimator."""
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=config.max_iterations,
        tol=config.tolerance,
        selection="cyclic",
    )


def _score_candidate(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    estimator: Ridge | ElasticNet,
) -> tuple[float, float, int]:
    """Fit one candidate and evaluate it on validation dates."""
    pipeline = _build_pipeline(
        estimator
    )

    pipeline.fit(
        train_data.loc[
            :,
            list(feature_columns),
        ],
        train_data[target_column],
    )

    predictions = pipeline.predict(
        validation_data.loc[
            :,
            list(feature_columns),
        ]
    )

    return _evaluate_validation_predictions(
        validation_data,
        predictions,
        target_column=target_column,
    )


def _choose_candidate(
    results: pd.DataFrame,
) -> int:
    """Choose a candidate using IC and RMSE as fallback."""
    valid_ic = results[
        "validation_mean_ic"
    ].notna()

    if valid_ic.any():
        ordered = results.loc[
            valid_ic
        ].sort_values(
            [
                "validation_mean_ic",
                "validation_rmse",
                "alpha",
                "l1_ratio",
            ],
            ascending=[
                False,
                True,
                False,
                False,
            ],
            na_position="last",
        )
    else:
        ordered = results.sort_values(
            [
                "validation_rmse",
                "alpha",
                "l1_ratio",
            ],
            ascending=[
                True,
                False,
                False,
            ],
            na_position="last",
        )

    return int(ordered.index[0])


def _select_ridge_hyperparameters(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    *,
    fold_id: str,
    test_date: pd.Timestamp,
    feature_columns: Sequence[str],
    config: LinearModelsConfig,
) -> tuple[float, pd.DataFrame]:
    """Select Ridge alpha using validation data."""
    rows: list[dict[str, Any]] = []

    for alpha in config.ridge_alphas:
        (
            mean_ic,
            rmse,
            valid_months,
        ) = _score_candidate(
            train_data,
            validation_data,
            feature_columns=feature_columns,
            target_column=config.target_column,
            estimator=_build_ridge(alpha),
        )

        rows.append(
            {
                "fold_id": fold_id,
                "test_date": test_date,
                "model_name": "ridge",
                "alpha": alpha,
                "l1_ratio": np.nan,
                "validation_mean_ic": mean_ic,
                "validation_rmse": rmse,
                "validation_valid_ic_months": (
                    valid_months
                ),
            }
        )

    results = pd.DataFrame(rows)

    chosen_index = _choose_candidate(
        results
    )

    results["selected"] = False
    results.loc[
        chosen_index,
        "selected",
    ] = True

    selected_alpha = float(
        results.loc[
            chosen_index,
            "alpha",
        ]
    )

    return (
        selected_alpha,
        results,
    )


def _select_elastic_net_hyperparameters(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    *,
    fold_id: str,
    test_date: pd.Timestamp,
    feature_columns: Sequence[str],
    config: LinearModelsConfig,
) -> tuple[float, float, pd.DataFrame]:
    """Select Elastic Net parameters using validation data."""
    rows: list[dict[str, Any]] = []

    candidates = product(
        config.elastic_net_alphas,
        config.elastic_net_l1_ratios,
    )

    for alpha, l1_ratio in candidates:
        (
            mean_ic,
            rmse,
            valid_months,
        ) = _score_candidate(
            train_data,
            validation_data,
            feature_columns=feature_columns,
            target_column=config.target_column,
            estimator=_build_elastic_net(
                alpha=alpha,
                l1_ratio=l1_ratio,
                config=config,
            ),
        )

        rows.append(
            {
                "fold_id": fold_id,
                "test_date": test_date,
                "model_name": "elastic_net",
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "validation_mean_ic": mean_ic,
                "validation_rmse": rmse,
                "validation_valid_ic_months": (
                    valid_months
                ),
            }
        )

    results = pd.DataFrame(rows)

    chosen_index = _choose_candidate(
        results
    )

    results["selected"] = False
    results.loc[
        chosen_index,
        "selected",
    ] = True

    selected_alpha = float(
        results.loc[
            chosen_index,
            "alpha",
        ]
    )

    selected_l1_ratio = float(
        results.loc[
            chosen_index,
            "l1_ratio",
        ]
    )

    return (
        selected_alpha,
        selected_l1_ratio,
        results,
    )


def _resolve_fold_partitions(
    panel: pd.DataFrame,
    fold: pd.Series,
    *,
    validation_months: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Resolve exact training, validation and test observations."""
    test_date = pd.Timestamp(
        fold["test_date"]
    ).normalize()

    date_information = (
        panel.groupby(
            "as_of_date",
            as_index=False,
            sort=True,
        )
        .agg(
            target_end_date=(
                "target_end_date",
                "max",
            ),
        )
    )

    known_dates = (
        date_information.loc[
            date_information[
                "as_of_date"
            ].lt(test_date)
            & date_information[
                "target_end_date"
            ].le(test_date)
        ]
        .sort_values("as_of_date")
        .reset_index(drop=True)
    )

    if len(known_dates) <= validation_months:
        raise LinearModelsError(
            f"Insufficient history for fold "
            f"{fold['fold_id']}."
        )

    validation_date_values = set(
        known_dates.tail(
            validation_months
        )["as_of_date"]
    )

    training_date_values = set(
        known_dates.iloc[
            :-validation_months
        ]["as_of_date"]
    )

    train_data = panel.loc[
        panel["as_of_date"].isin(
            training_date_values
        )
    ].copy()

    validation_data = panel.loc[
        panel["as_of_date"].isin(
            validation_date_values
        )
    ].copy()

    test_data = panel.loc[
        panel["as_of_date"].eq(
            test_date
        )
    ].copy()

    if train_data.empty:
        raise LinearModelsError(
            f"Training data is empty for {fold['fold_id']}."
        )

    if validation_data.empty:
        raise LinearModelsError(
            f"Validation data is empty for {fold['fold_id']}."
        )

    if test_data.empty:
        raise LinearModelsError(
            f"Test data is empty for {fold['fold_id']}."
        )

    if (
        train_data["as_of_date"].nunique()
        != int(fold["training_dates"])
    ):
        raise LinearModelsError(
            f"Training-date count differs from "
            f"fold definition for {fold['fold_id']}."
        )

    if (
        validation_data["as_of_date"].nunique()
        != int(fold["validation_dates"])
    ):
        raise LinearModelsError(
            f"Validation-date count differs from "
            f"fold definition for {fold['fold_id']}."
        )

    if len(test_data) != int(fold["test_rows"]):
        raise LinearModelsError(
            f"Test-row count differs from fold "
            f"definition for {fold['fold_id']}."
        )

    return (
        train_data,
        validation_data,
        test_data,
    )


def _extract_coefficients(
    pipeline: Pipeline,
    *,
    fold_id: str,
    test_date: pd.Timestamp,
    model_name: str,
    feature_columns: Sequence[str],
    alpha: float,
    l1_ratio: float | None,
) -> pd.DataFrame:
    """Extract standardized and original-scale coefficients."""
    imputer = pipeline.named_steps[
        "imputer"
    ]

    scaler = pipeline.named_steps[
        "scaler"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    standardized_coefficients = np.asarray(
        model.coef_,
        dtype=float,
    )

    original_coefficients = (
        standardized_coefficients
        / np.asarray(
            scaler.scale_,
            dtype=float,
        )
    )

    original_intercept = float(
        model.intercept_
        - np.dot(
            standardized_coefficients,
            np.asarray(
                scaler.mean_,
                dtype=float,
            )
            / np.asarray(
                scaler.scale_,
                dtype=float,
            ),
        )
    )

    rows: list[dict[str, Any]] = []

    for position, feature in enumerate(
        feature_columns
    ):
        rows.append(
            {
                "fold_id": fold_id,
                "test_date": test_date,
                "model_name": model_name,
                "feature": feature,
                "alpha": alpha,
                "l1_ratio": (
                    l1_ratio
                    if l1_ratio is not None
                    else np.nan
                ),
                "coefficient_standardized": float(
                    standardized_coefficients[
                        position
                    ]
                ),
                "coefficient_original_scale": float(
                    original_coefficients[
                        position
                    ]
                ),
                "imputation_median": float(
                    imputer.statistics_[
                        position
                    ]
                ),
                "scaler_mean": float(
                    scaler.mean_[
                        position
                    ]
                ),
                "scaler_scale": float(
                    scaler.scale_[
                        position
                    ]
                ),
                "intercept_standardized": float(
                    model.intercept_
                ),
                "intercept_original_scale": (
                    original_intercept
                ),
                "nonzero_coefficient": bool(
                    abs(
                        standardized_coefficients[
                            position
                        ]
                    )
                    > 1e-12
                ),
            }
        )

    return pd.DataFrame(rows)


def _build_prediction_rows(
    test_data: pd.DataFrame,
    predictions: np.ndarray,
    *,
    fold: pd.Series,
    model_name: str,
    config: LinearModelsConfig,
    alpha: float | None = None,
    l1_ratio: float | None = None,
) -> pd.DataFrame:
    """Build standardized prediction rows."""
    output = test_data.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "sector",
            config.target_column,
            config.top_label_column,
        ],
    ].copy()

    output.insert(
        0,
        "fold_id",
        str(fold["fold_id"]),
    )

    output.insert(
        4,
        "model_name",
        model_name,
    )

    output["prediction"] = np.asarray(
        predictions,
        dtype=float,
    )

    output["alpha"] = (
        alpha
        if alpha is not None
        else np.nan
    )

    output["l1_ratio"] = (
        l1_ratio
        if l1_ratio is not None
        else np.nan
    )

    output["fit_start_date"] = pd.Timestamp(
        fold["train_start_date"]
    ).normalize()

    output["fit_end_date"] = pd.Timestamp(
        fold["validation_end_date"]
    ).normalize()

    output[
        "latest_fit_target_end_date"
    ] = pd.Timestamp(
        fold[
            "latest_known_target_end_date"
        ]
    ).normalize()

    output["training_dates"] = int(
        fold["training_dates"]
    )

    output["validation_dates"] = int(
        fold["validation_dates"]
    )

    return output


def run_linear_models_walk_forward(
    modeling_panel: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    feature_directions: Mapping[str, float],
    validation_months: int,
    config: LinearModelsConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Train baselines, Ridge and Elastic Net walk-forward."""
    feature_columns = tuple(
        feature_columns
    )

    if not feature_columns:
        raise LinearModelsError(
            "At least one feature is required."
        )

    if len(feature_columns) != len(
        set(feature_columns)
    ):
        raise LinearModelsError(
            "feature_columns contains duplicates."
        )

    if set(feature_columns) != set(
        feature_directions
    ):
        raise LinearModelsError(
            "Feature-direction keys must exactly match "
            "feature_columns."
        )

    invalid_directions = {
        feature: direction
        for feature, direction in (
            feature_directions.items()
        )
        if direction not in {
            -1.0,
            1.0,
        }
    }

    if invalid_directions:
        raise LinearModelsError(
            "Feature directions must be -1.0 or 1.0."
        )

    if config.momentum_signal not in feature_columns:
        raise LinearModelsError(
            "The configured momentum signal is not "
            "part of the selected features."
        )

    required_panel_columns = (
        "as_of_date",
        "ticker",
        "sector",
        "target_end_date",
        config.target_column,
        config.top_label_column,
        *feature_columns,
    )

    _require_columns(
        modeling_panel,
        required_panel_columns,
        dataset_name="Technical modeling panel",
    )

    _require_columns(
        folds,
        (
            "fold_id",
            "test_date",
            "train_start_date",
            "validation_end_date",
            "latest_known_target_end_date",
            "training_dates",
            "validation_dates",
            "test_rows",
        ),
        dataset_name="Walk-forward folds",
    )

    panel = modeling_panel.copy()
    fold_definitions = folds.copy()

    _normalize_dates(
        panel,
        (
            "as_of_date",
            "target_end_date",
        ),
        dataset_name="Technical modeling panel",
    )

    _normalize_dates(
        fold_definitions,
        (
            "test_date",
            "train_start_date",
            "validation_end_date",
            "latest_known_target_end_date",
        ),
        dataset_name="Walk-forward folds",
    )

    duplicated_panel_rows = int(
        panel.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
    )

    if duplicated_panel_rows:
        raise LinearModelsError(
            "Technical modeling panel contains "
            "duplicated date-ticker rows."
        )

    prediction_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []

    direction_vector = np.asarray(
        [
            feature_directions[feature]
            for feature in feature_columns
        ],
        dtype=float,
    )

    momentum_direction = float(
        feature_directions[
            config.momentum_signal
        ]
    )

    for fold in fold_definitions.itertuples(
        index=False
    ):
        fold_series = pd.Series(
            fold._asdict()
        )

        test_date = pd.Timestamp(
            fold_series["test_date"]
        ).normalize()

        (
            train_data,
            validation_data,
            test_data,
        ) = _resolve_fold_partitions(
            panel,
            fold_series,
            validation_months=(
                validation_months
            ),
        )

        known_data = pd.concat(
            [
                train_data,
                validation_data,
            ],
            ignore_index=True,
        )

        constant_prediction = float(
            known_data[
                config.target_column
            ].mean()
        )

        constant_predictions = np.full(
            len(test_data),
            constant_prediction,
            dtype=float,
        )

        prediction_frames.append(
            _build_prediction_rows(
                test_data,
                constant_predictions,
                fold=fold_series,
                model_name="constant",
                config=config,
            )
        )

        momentum_imputer = SimpleImputer(
            strategy="median"
        )

        momentum_imputer.fit(
            known_data.loc[
                :,
                [config.momentum_signal],
            ]
        )

        momentum_values = (
            momentum_imputer.transform(
                test_data.loc[
                    :,
                    [config.momentum_signal],
                ]
            )[:, 0]
        )

        momentum_predictions = (
            momentum_values
            * momentum_direction
        )

        prediction_frames.append(
            _build_prediction_rows(
                test_data,
                momentum_predictions,
                fold=fold_series,
                model_name="momentum_3m",
                config=config,
            )
        )

        composite_imputer = SimpleImputer(
            strategy="median"
        )

        composite_imputer.fit(
            known_data.loc[
                :,
                list(feature_columns),
            ]
        )

        composite_values = (
            composite_imputer.transform(
                test_data.loc[
                    :,
                    list(feature_columns),
                ]
            )
        )

        composite_predictions = np.mean(
            composite_values
            * direction_vector,
            axis=1,
        )

        prediction_frames.append(
            _build_prediction_rows(
                test_data,
                composite_predictions,
                fold=fold_series,
                model_name=(
                    "equal_weight_composite"
                ),
                config=config,
            )
        )

        (
            ridge_alpha,
            ridge_validation,
        ) = _select_ridge_hyperparameters(
            train_data,
            validation_data,
            fold_id=str(
                fold_series["fold_id"]
            ),
            test_date=test_date,
            feature_columns=feature_columns,
            config=config,
        )

        validation_frames.append(
            ridge_validation
        )

        ridge_pipeline = _build_pipeline(
            _build_ridge(
                ridge_alpha
            )
        )

        ridge_pipeline.fit(
            known_data.loc[
                :,
                list(feature_columns),
            ],
            known_data[
                config.target_column
            ],
        )

        ridge_predictions = (
            ridge_pipeline.predict(
                test_data.loc[
                    :,
                    list(feature_columns),
                ]
            )
        )

        prediction_frames.append(
            _build_prediction_rows(
                test_data,
                ridge_predictions,
                fold=fold_series,
                model_name="ridge",
                config=config,
                alpha=ridge_alpha,
            )
        )

        coefficient_frames.append(
            _extract_coefficients(
                ridge_pipeline,
                fold_id=str(
                    fold_series["fold_id"]
                ),
                test_date=test_date,
                model_name="ridge",
                feature_columns=feature_columns,
                alpha=ridge_alpha,
                l1_ratio=None,
            )
        )

        (
            elastic_alpha,
            elastic_l1_ratio,
            elastic_validation,
        ) = (
            _select_elastic_net_hyperparameters(
                train_data,
                validation_data,
                fold_id=str(
                    fold_series["fold_id"]
                ),
                test_date=test_date,
                feature_columns=feature_columns,
                config=config,
            )
        )

        validation_frames.append(
            elastic_validation
        )

        elastic_pipeline = _build_pipeline(
            _build_elastic_net(
                alpha=elastic_alpha,
                l1_ratio=elastic_l1_ratio,
                config=config,
            )
        )

        elastic_pipeline.fit(
            known_data.loc[
                :,
                list(feature_columns),
            ],
            known_data[
                config.target_column
            ],
        )

        elastic_predictions = (
            elastic_pipeline.predict(
                test_data.loc[
                    :,
                    list(feature_columns),
                ]
            )
        )

        prediction_frames.append(
            _build_prediction_rows(
                test_data,
                elastic_predictions,
                fold=fold_series,
                model_name="elastic_net",
                config=config,
                alpha=elastic_alpha,
                l1_ratio=elastic_l1_ratio,
            )
        )

        coefficient_frames.append(
            _extract_coefficients(
                elastic_pipeline,
                fold_id=str(
                    fold_series["fold_id"]
                ),
                test_date=test_date,
                model_name="elastic_net",
                feature_columns=feature_columns,
                alpha=elastic_alpha,
                l1_ratio=elastic_l1_ratio,
            )
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    validation_grid = pd.concat(
        validation_frames,
        ignore_index=True,
    )

    coefficients = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )

    if predictions["prediction"].isna().any():
        raise LinearModelsError(
            "Generated predictions contain missing values."
        )

    if np.isinf(
        predictions[
            "prediction"
        ].to_numpy(dtype=float)
    ).any():
        raise LinearModelsError(
            "Generated predictions contain infinite values."
        )

    duplicate_predictions = int(
        predictions.duplicated(
            [
                "as_of_date",
                "ticker",
                "model_name",
            ]
        ).sum()
    )

    if duplicate_predictions:
        raise LinearModelsError(
            "Generated predictions contain duplicate "
            "date-ticker-model rows."
        )

    if (
        predictions[
            "latest_fit_target_end_date"
        ]
        > predictions["as_of_date"]
    ).any():
        raise LinearModelsError(
            "At least one prediction uses a target "
            "that was not known on the test date."
        )

    predictions[
        "prediction_rank"
    ] = (
        predictions.groupby(
            [
                "model_name",
                "as_of_date",
            ]
        )["prediction"]
        .rank(
            method="average",
            ascending=False,
        )
    )

    predictions[
        "prediction_percentile"
    ] = (
        predictions.groupby(
            [
                "model_name",
                "as_of_date",
            ]
        )["prediction"]
        .rank(
            method="average",
            ascending=True,
            pct=True,
        )
    )

    prediction_columns = [
        "fold_id",
        "as_of_date",
        "ticker",
        "sector",
        "model_name",
        "prediction",
        "prediction_rank",
        "prediction_percentile",
        config.target_column,
        config.top_label_column,
        "alpha",
        "l1_ratio",
        "fit_start_date",
        "fit_end_date",
        "latest_fit_target_end_date",
        "training_dates",
        "validation_dates",
    ]

    predictions = (
        predictions.loc[
            :,
            prediction_columns,
        ]
        .sort_values(
            [
                "as_of_date",
                "model_name",
                "prediction_rank",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    validation_grid = (
        validation_grid.sort_values(
            [
                "test_date",
                "model_name",
                "alpha",
                "l1_ratio",
            ]
        )
        .reset_index(drop=True)
    )

    coefficients = (
        coefficients.sort_values(
            [
                "test_date",
                "model_name",
                "feature",
            ]
        )
        .reset_index(drop=True)
    )

    return (
        predictions,
        validation_grid,
        coefficients,
    )