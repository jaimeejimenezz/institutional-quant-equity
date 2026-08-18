"""Walk-forward training for regularized linear equity models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge


class RegularizedLinearError(ValueError):
    """Raised when regularized linear training cannot be completed."""


MODEL_FEATURE_PREFIXES = (
    "tech__",
    "fund__",
)

MISSING_INDICATOR_SUFFIX = "_missing"

TARGET_COLUMN = "target_21d_excess"
TOP_LABEL_COLUMN = "label_top_quintile"


@dataclass(frozen=True)
class RegularizedLinearConfig:
    """Configuration for regularized linear model training."""

    expected_feature_count: int = 91

    ridge_alphas: tuple[float, ...] = (
        0.1,
        1.0,
        10.0,
        100.0,
    )

    elastic_net_alphas: tuple[float, ...] = (
        0.0001,
        0.001,
        0.01,
    )

    elastic_net_l1_ratios: tuple[float, ...] = (
        0.1,
        0.5,
        0.9,
    )

    minimum_validation_dates: int = 12
    elastic_net_max_iter: int = 50_000

    def validate(self) -> None:
        """Validate model-training configuration."""
        if self.expected_feature_count < 1:
            raise RegularizedLinearError(
                "expected_feature_count must be positive."
            )

        if not self.ridge_alphas:
            raise RegularizedLinearError(
                "At least one Ridge alpha is required."
            )

        if not self.elastic_net_alphas:
            raise RegularizedLinearError(
                "At least one Elastic Net alpha is required."
            )

        if not self.elastic_net_l1_ratios:
            raise RegularizedLinearError(
                "At least one Elastic Net l1_ratio is required."
            )

        if any(
            value <= 0.0
            for value in self.ridge_alphas
        ):
            raise RegularizedLinearError(
                "All Ridge alphas must be positive."
            )

        if any(
            value <= 0.0
            for value in self.elastic_net_alphas
        ):
            raise RegularizedLinearError(
                "All Elastic Net alphas must be positive."
            )

        if any(
            not 0.0 < value <= 1.0
            for value in self.elastic_net_l1_ratios
        ):
            raise RegularizedLinearError(
                "Elastic Net l1_ratio values must be in (0, 1]."
            )


@dataclass(frozen=True)
class FittedFeaturePreprocessor:
    """Training-only feature preprocessing parameters."""

    feature_columns: tuple[str, ...]
    continuous_columns: tuple[str, ...]
    indicator_columns: tuple[str, ...]
    medians: pd.Series
    means: pd.Series
    scales: pd.Series

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform data without refitting preprocessing parameters."""
        _require_columns(
            data,
            self.feature_columns,
            dataset_name="feature data",
        )

        transformed = pd.DataFrame(
            index=data.index
        )

        if self.continuous_columns:
            continuous = (
                data.loc[
                    :,
                    self.continuous_columns,
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .astype(float)
            )

            continuous = continuous.fillna(
                self.medians
            )

            continuous = (
                continuous
                - self.means
            ).divide(
                self.scales,
                axis="columns",
            )

            transformed.loc[
                :,
                self.continuous_columns,
            ] = continuous

        if self.indicator_columns:
            indicators = (
                data.loc[
                    :,
                    self.indicator_columns,
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .astype(float)
            )

            if indicators.isna().any().any():
                raise RegularizedLinearError(
                    "Missing-indicator features contain missing "
                    "or non-numeric values."
                )

            valid_binary = np.isin(
                indicators.to_numpy(
                    dtype=float
                ),
                [
                    0.0,
                    1.0,
                ],
            )

            if not valid_binary.all():
                raise RegularizedLinearError(
                    "Missing-indicator features must remain binary."
                )

            transformed.loc[
                :,
                self.indicator_columns,
            ] = indicators

        transformed = transformed.loc[
            :,
            self.feature_columns,
        ].astype(float)

        if not np.isfinite(
            transformed.to_numpy(
                dtype=float
            )
        ).all():
            raise RegularizedLinearError(
                "Preprocessed features contain non-finite values."
            )

        return transformed


@dataclass(frozen=True)
class RegularizedLinearOutputs:
    """Artifacts produced by walk-forward model training."""

    predictions: pd.DataFrame
    hyperparameter_search: pd.DataFrame
    coefficients: pd.DataFrame
    feature_columns: tuple[str, ...]


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Require a collection of columns."""
    missing = sorted(
        set(columns).difference(
            data.columns
        )
    )

    if missing:
        raise RegularizedLinearError(
            f"{dataset_name} is missing columns: "
            + ", ".join(missing)
            + "."
        )


def detect_model_features(
    panel: pd.DataFrame,
    *,
    expected_count: int,
) -> tuple[str, ...]:
    """Detect the frozen technical and fundamental predictors."""
    columns = tuple(
        column
        for column in panel.columns
        if column.startswith(
            MODEL_FEATURE_PREFIXES
        )
    )

    if len(columns) != expected_count:
        raise RegularizedLinearError(
            "Unexpected model feature count: "
            f"expected {expected_count}, found {len(columns)}."
        )

    if len(columns) != len(
        set(columns)
    ):
        raise RegularizedLinearError(
            "Model feature names are not unique."
        )

    return columns


def fit_feature_preprocessor(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
) -> FittedFeaturePreprocessor:
    """Fit imputation and scaling parameters using training data only."""
    feature_columns = tuple(
        feature_columns
    )

    _require_columns(
        train,
        feature_columns,
        dataset_name="training data",
    )

    indicator_columns = tuple(
        column
        for column in feature_columns
        if column.endswith(
            MISSING_INDICATOR_SUFFIX
        )
    )

    continuous_columns = tuple(
        column
        for column in feature_columns
        if column not in indicator_columns
    )

    if continuous_columns:
        continuous = (
            train.loc[
                :,
                continuous_columns,
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .astype(float)
        )

        medians = continuous.median(
            axis=0
        )

        if medians.isna().any():
            unavailable = medians[
                medians.isna()
            ].index.tolist()

            raise RegularizedLinearError(
                "Continuous features without usable training "
                "observations: "
                + ", ".join(
                    unavailable
                )
                + "."
            )

        imputed = continuous.fillna(
            medians
        )

        means = imputed.mean(
            axis=0
        )

        scales = imputed.std(
            axis=0,
            ddof=0,
        )

        scales = scales.mask(
            scales.abs().le(
                1e-12
            ),
            1.0,
        )
    else:
        medians = pd.Series(
            dtype=float
        )
        means = pd.Series(
            dtype=float
        )
        scales = pd.Series(
            dtype=float
        )

    if indicator_columns:
        indicators = (
            train.loc[
                :,
                indicator_columns,
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .astype(float)
        )

        if indicators.isna().any().any():
            raise RegularizedLinearError(
                "Training missing-indicator features "
                "contain invalid values."
            )

        valid_binary = np.isin(
            indicators.to_numpy(
                dtype=float
            ),
            [
                0.0,
                1.0,
            ],
        )

        if not valid_binary.all():
            raise RegularizedLinearError(
                "Training missing-indicator features "
                "must contain only 0 and 1."
            )

    return FittedFeaturePreprocessor(
        feature_columns=feature_columns,
        continuous_columns=continuous_columns,
        indicator_columns=indicator_columns,
        medians=medians,
        means=means,
        scales=scales,
    )


def _numeric_target(
    data: pd.DataFrame,
) -> pd.Series:
    """Return a validated numeric regression target."""
    target = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    ).astype(float)

    if target.isna().any():
        raise RegularizedLinearError(
            "Regression target contains missing "
            "or non-numeric values."
        )

    if not np.isfinite(
        target.to_numpy(
            dtype=float
        )
    ).all():
        raise RegularizedLinearError(
            "Regression target contains non-finite values."
        )

    return target


def _mean_monthly_spearman(
    prediction: np.ndarray,
    target: pd.Series,
    dates: pd.Series,
) -> tuple[float, int]:
    """Calculate mean cross-sectional validation IC."""
    evaluation = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(
                dates
            ).to_numpy(),
            "prediction": np.asarray(
                prediction,
                dtype=float,
            ),
            "target": target.to_numpy(
                dtype=float
            ),
        }
    )

    monthly_ic: list[float] = []

    for _, month in evaluation.groupby(
        "as_of_date",
        sort=True,
    ):
        if (
            len(month) < 2
            or month[
                "prediction"
            ].nunique()
            < 2
            or month[
                "target"
            ].nunique()
            < 2
        ):
            continue

        ic = month[
            "prediction"
        ].corr(
            month[
                "target"
            ],
            method="spearman",
        )

        if pd.notna(
            ic
        ):
            monthly_ic.append(
                float(
                    ic
                )
            )

    if not monthly_ic:
        return (
            float("nan"),
            0,
        )

    return (
        float(
            np.mean(
                monthly_ic
            )
        ),
        len(
            monthly_ic
        ),
    )


def _fit_ridge(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    alpha: float,
) -> Ridge:
    """Fit a Ridge regression model."""
    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
    )

    model.fit(
        x,
        y,
    )

    return model


def _fit_elastic_net(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
) -> ElasticNet:
    """Fit an Elastic Net regression model."""
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=max_iter,
        tol=1e-6,
        selection="cyclic",
    )

    model.fit(
        x,
        y,
    )

    return model


def _select_ridge_alpha(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_dates: pd.Series,
    *,
    config: RegularizedLinearConfig,
    fold_id: str,
    test_date: pd.Timestamp,
) -> tuple[
    float,
    list[dict[str, Any]],
]:
    """Select Ridge regularization using validation data only."""
    rows: list[
        dict[str, Any]
    ] = []

    for alpha in config.ridge_alphas:
        model = _fit_ridge(
            x_train,
            y_train,
            alpha=alpha,
        )

        prediction = model.predict(
            x_validation
        )

        mean_ic, valid_months = (
            _mean_monthly_spearman(
                prediction,
                y_validation,
                validation_dates,
            )
        )

        rows.append(
            {
                "fold_id": fold_id,
                "test_date": test_date,
                "model_name": "ridge",
                "alpha": alpha,
                "l1_ratio": np.nan,
                "validation_mean_ic": mean_ic,
                "validation_valid_months": valid_months,
                "selected": False,
            }
        )

    valid_rows = [
        row
        for row in rows
        if np.isfinite(
            row[
                "validation_mean_ic"
            ]
        )
    ]

    if not valid_rows:
        raise RegularizedLinearError(
            f"No Ridge candidate produced a valid "
            f"validation IC for {fold_id}."
        )

    best = max(
        valid_rows,
        key=lambda row: (
            row[
                "validation_mean_ic"
            ],
            row[
                "alpha"
            ],
        ),
    )

    best[
        "selected"
    ] = True

    return (
        float(
            best[
                "alpha"
            ]
        ),
        rows,
    )


def _select_elastic_net_parameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_dates: pd.Series,
    *,
    config: RegularizedLinearConfig,
    fold_id: str,
    test_date: pd.Timestamp,
) -> tuple[
    float,
    float,
    list[dict[str, Any]],
]:
    """Select Elastic Net hyperparameters using validation data only."""
    rows: list[
        dict[str, Any]
    ] = []

    for alpha in (
        config.elastic_net_alphas
    ):
        for l1_ratio in (
            config.elastic_net_l1_ratios
        ):
            model = _fit_elastic_net(
                x_train,
                y_train,
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=(
                    config.elastic_net_max_iter
                ),
            )

            prediction = model.predict(
                x_validation
            )

            mean_ic, valid_months = (
                _mean_monthly_spearman(
                    prediction,
                    y_validation,
                    validation_dates,
                )
            )

            rows.append(
                {
                    "fold_id": fold_id,
                    "test_date": test_date,
                    "model_name": (
                        "elastic_net"
                    ),
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "validation_mean_ic": (
                        mean_ic
                    ),
                    "validation_valid_months": (
                        valid_months
                    ),
                    "selected": False,
                }
            )

    valid_rows = [
        row
        for row in rows
        if np.isfinite(
            row[
                "validation_mean_ic"
            ]
        )
    ]

    if not valid_rows:
        raise RegularizedLinearError(
            f"No Elastic Net candidate produced a valid "
            f"validation IC for {fold_id}."
        )

    best = max(
        valid_rows,
        key=lambda row: (
            row[
                "validation_mean_ic"
            ],
            row[
                "alpha"
            ],
            row[
                "l1_ratio"
            ],
        ),
    )

    best[
        "selected"
    ] = True

    return (
        float(
            best[
                "alpha"
            ]
        ),
        float(
            best[
                "l1_ratio"
            ]
        ),
        rows,
    )


def _prepare_fold_data(
    panel: pd.DataFrame,
    fold: Any,
    *,
    config: RegularizedLinearConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Extract train, validation and test partitions from frozen metadata."""
    train_start = pd.Timestamp(
        fold.train_start_date
    )

    train_end = pd.Timestamp(
        fold.train_end_date
    )

    validation_start = pd.Timestamp(
        fold.validation_start_date
    )

    validation_end = pd.Timestamp(
        fold.validation_end_date
    )

    test_date = pd.Timestamp(
        fold.test_date
    )

    train = panel.loc[
        panel[
            "as_of_date"
        ].between(
            train_start,
            train_end,
            inclusive="both",
        )
    ].copy()

    validation = panel.loc[
        panel[
            "as_of_date"
        ].between(
            validation_start,
            validation_end,
            inclusive="both",
        )
    ].copy()

    test = panel.loc[
        panel[
            "as_of_date"
        ].eq(
            test_date
        )
    ].copy()

    if train.empty:
        raise RegularizedLinearError(
            f"{fold.fold_id} has no training rows."
        )

    if validation.empty:
        raise RegularizedLinearError(
            f"{fold.fold_id} has no validation rows."
        )

    validation_dates = (
        validation[
            "as_of_date"
        ].nunique()
    )

    if (
        validation_dates
        < config.minimum_validation_dates
    ):
        raise RegularizedLinearError(
            f"{fold.fold_id} has only "
            f"{validation_dates} validation dates."
        )

    expected_test_rows = int(
        fold.test_rows
    )

    if len(
        test
    ) != expected_test_rows:
        raise RegularizedLinearError(
            f"{fold.fold_id} expected "
            f"{expected_test_rows} test rows but found "
            f"{len(test)}."
        )

    fitting_data = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    fitting_target_end = pd.to_datetime(
        fitting_data[
            "target_end_date"
        ],
        errors="coerce",
    ).dt.normalize()

    if fitting_target_end.isna().any():
        raise RegularizedLinearError(
            f"{fold.fold_id} contains fitting rows "
            "without a valid target_end_date."
        )

    maturity_violations = int(
        fitting_target_end.gt(
            test_date
        ).sum()
    )

    if maturity_violations:
        raise RegularizedLinearError(
            f"{fold.fold_id} contains "
            f"{maturity_violations} fitting rows "
            "whose targets were not mature on "
            "the test date."
        )

    latest_fit_target_end = (
        fitting_target_end.max()
    )

    return (
        train,
        validation,
        test,
        latest_fit_target_end,
    )


def train_regularized_linear_models(
    panel: pd.DataFrame,
    fold_metadata: pd.DataFrame,
    *,
    config: RegularizedLinearConfig | None = None,
) -> RegularizedLinearOutputs:
    """Train Ridge and Elastic Net across frozen walk-forward folds."""
    if config is None:
        config = (
            RegularizedLinearConfig()
        )

    config.validate()

    required_panel_columns = (
        "as_of_date",
        "ticker",
        "sector",
        "target_end_date",
        TARGET_COLUMN,
        TOP_LABEL_COLUMN,
    )

    required_fold_columns = (
        "fold_id",
        "test_date",
        "train_start_date",
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
        "test_rows",
    )

    _require_columns(
        panel,
        required_panel_columns,
        dataset_name="modeling panel",
    )

    _require_columns(
        fold_metadata,
        required_fold_columns,
        dataset_name="walk-forward metadata",
    )

    panel = panel.copy()
    folds = fold_metadata.copy()

    panel[
        "target_end_date"
    ] = pd.to_datetime(
        panel[
            "target_end_date"
        ],
        errors="coerce",
    ).dt.normalize()

    for column in (
        "test_date",
        "train_start_date",
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
    ):
        folds[
            column
        ] = pd.to_datetime(
            folds[
                column
            ]
        ).dt.normalize()

    feature_columns = (
        detect_model_features(
            panel,
            expected_count=(
                config.expected_feature_count
            ),
        )
    )

    prediction_frames: list[
        pd.DataFrame
    ] = []

    search_rows: list[
        dict[str, Any]
    ] = []

    coefficient_rows: list[
        dict[str, Any]
    ] = []

    for fold in (
        folds.sort_values(
            "test_date"
        )
        .itertuples(
            index=False
        )
    ):
        (
            train,
            validation,
            test,
            latest_fit_target_end,
        ) = _prepare_fold_data(
            panel,
            fold,
            config=config,
        )

        preprocessor = (
            fit_feature_preprocessor(
                train,
                feature_columns,
            )
        )

        x_train = (
            preprocessor.transform(
                train
            )
        )

        x_validation = (
            preprocessor.transform(
                validation
            )
        )

        x_test = (
            preprocessor.transform(
                test
            )
        )

        y_train = _numeric_target(
            train
        )

        y_validation = _numeric_target(
            validation
        )

        ridge_alpha, ridge_search = (
            _select_ridge_alpha(
                x_train,
                y_train,
                x_validation,
                y_validation,
                validation[
                    "as_of_date"
                ],
                config=config,
                fold_id=str(
                    fold.fold_id
                ),
                test_date=pd.Timestamp(
                    fold.test_date
                ),
            )
        )

        (
            elastic_alpha,
            elastic_l1_ratio,
            elastic_search,
        ) = (
            _select_elastic_net_parameters(
                x_train,
                y_train,
                x_validation,
                y_validation,
                validation[
                    "as_of_date"
                ],
                config=config,
                fold_id=str(
                    fold.fold_id
                ),
                test_date=pd.Timestamp(
                    fold.test_date
                ),
            )
        )

        search_rows.extend(
            ridge_search
        )

        search_rows.extend(
            elastic_search
        )

        x_fit = pd.concat(
            [
                x_train,
                x_validation,
            ],
            ignore_index=True,
        )

        y_fit = pd.concat(
            [
                y_train,
                y_validation,
            ],
            ignore_index=True,
        )

        fitted_models = {
            "ridge": (
                _fit_ridge(
                    x_fit,
                    y_fit,
                    alpha=ridge_alpha,
                ),
                ridge_alpha,
                np.nan,
            ),
            "elastic_net": (
                _fit_elastic_net(
                    x_fit,
                    y_fit,
                    alpha=elastic_alpha,
                    l1_ratio=(
                        elastic_l1_ratio
                    ),
                    max_iter=(
                        config.elastic_net_max_iter
                    ),
                ),
                elastic_alpha,
                elastic_l1_ratio,
            ),
        }

        for (
            model_name,
            (
                model,
                selected_alpha,
                selected_l1_ratio,
            ),
        ) in fitted_models.items():
            prediction = model.predict(
                x_test
            )

            block = test.loc[
                :,
                [
                    "as_of_date",
                    "ticker",
                    "sector",
                    TARGET_COLUMN,
                    TOP_LABEL_COLUMN,
                ],
            ].copy()

            block.insert(
                0,
                "fold_id",
                str(
                    fold.fold_id
                ),
            )

            block[
                "model_name"
            ] = model_name

            block[
                "prediction"
            ] = prediction.astype(
                float
            )

            block[
                "latest_fit_target_end_date"
            ] = latest_fit_target_end

            block[
                "selected_alpha"
            ] = selected_alpha

            block[
                "selected_l1_ratio"
            ] = selected_l1_ratio

            prediction_frames.append(
                block
            )

            coefficients = np.asarray(
                model.coef_,
                dtype=float,
            )

            for (
                feature,
                coefficient,
            ) in zip(
                feature_columns,
                coefficients,
                strict=True,
            ):
                coefficient_rows.append(
                    {
                        "fold_id": str(
                            fold.fold_id
                        ),
                        "test_date": pd.Timestamp(
                            fold.test_date
                        ),
                        "model_name": (
                            model_name
                        ),
                        "feature": feature,
                        "coefficient_standardized": (
                            float(
                                coefficient
                            )
                        ),
                        "absolute_coefficient": (
                            float(
                                abs(
                                    coefficient
                                )
                            )
                        ),
                        "nonzero_coefficient": (
                            bool(
                                abs(
                                    coefficient
                                )
                                > 1e-12
                            )
                        ),
                        "selected_alpha": (
                            selected_alpha
                        ),
                        "selected_l1_ratio": (
                            selected_l1_ratio
                        ),
                    }
                )

    if not prediction_frames:
        raise RegularizedLinearError(
            "No linear-model predictions were generated."
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    duplicate_predictions = int(
        predictions.duplicated(
            [
                "fold_id",
                "ticker",
                "model_name",
            ]
        ).sum()
    )

    if duplicate_predictions:
        raise RegularizedLinearError(
            "Generated predictions contain duplicate keys."
        )

    predictions = predictions.sort_values(
        [
            "as_of_date",
            "model_name",
            "ticker",
        ]
    ).reset_index(
        drop=True
    )

    hyperparameter_search = (
        pd.DataFrame(
            search_rows
        )
        .sort_values(
            [
                "test_date",
                "model_name",
                "alpha",
                "l1_ratio",
            ],
            na_position="first",
        )
        .reset_index(
            drop=True
        )
    )

    coefficients = (
        pd.DataFrame(
            coefficient_rows
        )
        .sort_values(
            [
                "test_date",
                "model_name",
                "feature",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return RegularizedLinearOutputs(
        predictions=predictions,
        hyperparameter_search=(
            hyperparameter_search
        ),
        coefficients=coefficients,
        feature_columns=feature_columns,
    )


def summarize_linear_coefficients(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize coefficient magnitude and sparsity across folds."""
    _require_columns(
        coefficients,
        (
            "model_name",
            "feature",
            "coefficient_standardized",
            "absolute_coefficient",
            "nonzero_coefficient",
        ),
        dataset_name="coefficient data",
    )

    summary = (
        coefficients.groupby(
            [
                "model_name",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            mean_coefficient=(
                "coefficient_standardized",
                "mean",
            ),
            mean_absolute_coefficient=(
                "absolute_coefficient",
                "mean",
            ),
            nonzero_ratio=(
                "nonzero_coefficient",
                "mean",
            ),
        )
    )

    return summary.sort_values(
        [
            "model_name",
            "mean_absolute_coefficient",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )