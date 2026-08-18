"""Walk-forward training for LightGBM equity regression models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import (
    LGBMRegressor,
    early_stopping,
    log_evaluation,
)

from quant_equity.models.regularized_linear import (
    detect_model_features,
    fit_feature_preprocessor,
)


class LightGBMRegressionError(ValueError):
    """Raised when LightGBM regression training cannot be completed."""


TARGET_COLUMN = "target_21d_excess"
TOP_LABEL_COLUMN = "label_top_quintile"


@dataclass(frozen=True)
class LightGBMCandidate:
    """One controlled LightGBM hyperparameter configuration."""

    candidate_name: str
    num_leaves: int
    max_depth: int
    min_child_samples: int
    learning_rate: float
    reg_alpha: float
    reg_lambda: float
    colsample_bytree: float
    subsample: float


DEFAULT_CANDIDATES = (
    LightGBMCandidate(
        candidate_name="shallow",
        num_leaves=7,
        max_depth=3,
        min_child_samples=80,
        learning_rate=0.03,
        reg_alpha=0.0,
        reg_lambda=1.0,
        colsample_bytree=0.8,
        subsample=0.9,
    ),
    LightGBMCandidate(
        candidate_name="balanced",
        num_leaves=15,
        max_depth=4,
        min_child_samples=60,
        learning_rate=0.03,
        reg_alpha=0.0,
        reg_lambda=1.0,
        colsample_bytree=0.8,
        subsample=0.9,
    ),
    LightGBMCandidate(
        candidate_name="flexible",
        num_leaves=31,
        max_depth=5,
        min_child_samples=40,
        learning_rate=0.03,
        reg_alpha=0.0,
        reg_lambda=1.0,
        colsample_bytree=0.8,
        subsample=0.9,
    ),
    LightGBMCandidate(
        candidate_name="regularized",
        num_leaves=15,
        max_depth=4,
        min_child_samples=40,
        learning_rate=0.03,
        reg_alpha=0.1,
        reg_lambda=5.0,
        colsample_bytree=0.7,
        subsample=0.9,
    ),
)


@dataclass(frozen=True)
class LightGBMRegressionConfig:
    """Configuration for walk-forward LightGBM regression."""

    expected_feature_count: int = 91
    max_estimators: int = 1000
    early_stopping_rounds: int = 50
    minimum_validation_dates: int = 12
    random_state: int = 42
    n_jobs: int = -1
    candidates: tuple[
        LightGBMCandidate,
        ...,
    ] = DEFAULT_CANDIDATES

    def validate(self) -> None:
        """Validate training configuration."""
        if self.expected_feature_count < 1:
            raise LightGBMRegressionError(
                "expected_feature_count must be positive."
            )

        if self.max_estimators < 1:
            raise LightGBMRegressionError(
                "max_estimators must be positive."
            )

        if self.early_stopping_rounds < 1:
            raise LightGBMRegressionError(
                "early_stopping_rounds must be positive."
            )

        if not self.candidates:
            raise LightGBMRegressionError(
                "At least one LightGBM candidate is required."
            )

        names = [
            candidate.candidate_name
            for candidate in self.candidates
        ]

        if len(names) != len(
            set(names)
        ):
            raise LightGBMRegressionError(
                "LightGBM candidate names must be unique."
            )


@dataclass(frozen=True)
class LightGBMRegressionOutputs:
    """Artifacts produced by LightGBM walk-forward training."""

    predictions: pd.DataFrame
    hyperparameter_search: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_columns: tuple[str, ...]


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Require columns to exist."""
    missing = sorted(
        set(columns).difference(
            data.columns
        )
    )

    if missing:
        raise LightGBMRegressionError(
            f"{dataset_name} is missing columns: "
            + ", ".join(missing)
            + "."
        )


def _numeric_target(
    data: pd.DataFrame,
) -> pd.Series:
    """Return a finite numeric regression target."""
    target = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    ).astype(float)

    if target.isna().any():
        raise LightGBMRegressionError(
            "Regression target contains missing values."
        )

    if not np.isfinite(
        target.to_numpy(
            dtype=float
        )
    ).all():
        raise LightGBMRegressionError(
            "Regression target contains non-finite values."
        )

    return target


def _mean_monthly_spearman(
    prediction: np.ndarray,
    target: pd.Series,
    dates: pd.Series,
) -> tuple[float, int]:
    """Calculate mean cross-sectional Spearman IC."""
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
            or month["prediction"].nunique() < 2
            or month["target"].nunique() < 2
        ):
            continue

        ic = month["prediction"].corr(
            month["target"],
            method="spearman",
        )

        if pd.notna(ic):
            monthly_ic.append(
                float(ic)
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
        len(monthly_ic),
    )


def _prepare_fold_data(
    panel: pd.DataFrame,
    fold: Any,
    *,
    minimum_validation_dates: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Timestamp,
]:
    """Extract and validate one frozen temporal partition."""
    test_date = pd.Timestamp(
        fold.test_date
    )

    train = panel.loc[
        panel["as_of_date"].between(
            pd.Timestamp(
                fold.train_start_date
            ),
            pd.Timestamp(
                fold.train_end_date
            ),
            inclusive="both",
        )
    ].copy()

    validation = panel.loc[
        panel["as_of_date"].between(
            pd.Timestamp(
                fold.validation_start_date
            ),
            pd.Timestamp(
                fold.validation_end_date
            ),
            inclusive="both",
        )
    ].copy()

    test = panel.loc[
        panel["as_of_date"].eq(
            test_date
        )
    ].copy()

    if train.empty:
        raise LightGBMRegressionError(
            f"{fold.fold_id} has no training rows."
        )

    if validation.empty:
        raise LightGBMRegressionError(
            f"{fold.fold_id} has no validation rows."
        )

    validation_date_count = int(
        validation[
            "as_of_date"
        ].nunique()
    )

    if (
        validation_date_count
        < minimum_validation_dates
    ):
        raise LightGBMRegressionError(
            f"{fold.fold_id} has only "
            f"{validation_date_count} validation dates."
        )

    expected_test_rows = int(
        fold.test_rows
    )

    if len(test) != expected_test_rows:
        raise LightGBMRegressionError(
            f"{fold.fold_id} expected "
            f"{expected_test_rows} test rows "
            f"but found {len(test)}."
        )

    fitting_data = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    target_end = pd.to_datetime(
        fitting_data[
            "target_end_date"
        ],
        errors="coerce",
    ).dt.normalize()

    if target_end.isna().any():
        raise LightGBMRegressionError(
            f"{fold.fold_id} contains fitting rows "
            "without a valid target_end_date."
        )

    maturity_violations = int(
        target_end.gt(
            test_date
        ).sum()
    )

    if maturity_violations:
        raise LightGBMRegressionError(
            f"{fold.fold_id} contains "
            f"{maturity_violations} fitting rows "
            "whose targets were not mature on "
            "the test date."
        )

    return (
        train,
        validation,
        test,
        target_end.max(),
    )


def _build_model(
    candidate: LightGBMCandidate,
    *,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
) -> LGBMRegressor:
    """Create one deterministic LightGBM regressor."""
    return LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=(
            candidate.learning_rate
        ),
        num_leaves=(
            candidate.num_leaves
        ),
        max_depth=(
            candidate.max_depth
        ),
        min_child_samples=(
            candidate.min_child_samples
        ),
        reg_alpha=(
            candidate.reg_alpha
        ),
        reg_lambda=(
            candidate.reg_lambda
        ),
        colsample_bytree=(
            candidate.colsample_bytree
        ),
        subsample=(
            candidate.subsample
        ),
        subsample_freq=1,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,
        force_col_wise=True,
        importance_type="gain",
    )


def _evaluate_candidate(
    candidate: LightGBMCandidate,
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_dates: pd.Series,
    config: LightGBMRegressionConfig,
) -> tuple[
    LGBMRegressor,
    int,
    float,
    int,
]:
    """Fit one candidate and evaluate its validation ranking."""
    model = _build_model(
        candidate,
        n_estimators=(
            config.max_estimators
        ),
        random_state=(
            config.random_state
        ),
        n_jobs=config.n_jobs,
    )

    model.fit(
        x_train,
        y_train,
        eval_X=x_validation,
        eval_y=y_validation,
        eval_metric="l2",
        callbacks=[
            early_stopping(
                stopping_rounds=(
                    config.early_stopping_rounds
                ),
                first_metric_only=True,
                verbose=False,
            ),
            log_evaluation(
                period=0
            ),
        ],
    )

    best_iteration = int(
        model.best_iteration_
    )

    if best_iteration < 1:
        raise LightGBMRegressionError(
            "LightGBM did not produce a valid best iteration."
        )

    validation_prediction = (
        model.predict(
            x_validation,
            num_iteration=(
                best_iteration
            ),
        )
    )

    mean_ic, valid_months = (
        _mean_monthly_spearman(
            validation_prediction,
            y_validation,
            validation_dates,
        )
    )

    return (
        model,
        best_iteration,
        mean_ic,
        valid_months,
    )


def train_lightgbm_regression(
    panel: pd.DataFrame,
    fold_metadata: pd.DataFrame,
    *,
    config: LightGBMRegressionConfig | None = None,
) -> LightGBMRegressionOutputs:
    """Train one selected LightGBM regressor per OOS fold."""
    if config is None:
        config = (
            LightGBMRegressionConfig()
        )

    config.validate()

    _require_columns(
        panel,
        (
            "as_of_date",
            "ticker",
            "sector",
            "target_end_date",
            TARGET_COLUMN,
            TOP_LABEL_COLUMN,
        ),
        dataset_name="modeling panel",
    )

    _require_columns(
        fold_metadata,
        (
            "fold_id",
            "test_date",
            "train_start_date",
            "train_end_date",
            "validation_start_date",
            "validation_end_date",
            "test_rows",
        ),
        dataset_name="walk-forward metadata",
    )

    panel = panel.copy()
    folds = fold_metadata.copy()

    panel[
        "as_of_date"
    ] = pd.to_datetime(
        panel["as_of_date"]
    ).dt.normalize()

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
        folds[column] = pd.to_datetime(
            folds[column]
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

    importance_rows: list[
        dict[str, Any]
    ] = []

    ordered_folds = (
        folds.sort_values(
            "test_date"
        )
        .reset_index(
            drop=True
        )
    )

    total_folds = len(
        ordered_folds
    )

    for fold_number, fold in enumerate(
        ordered_folds.itertuples(
            index=False
        ),
        start=1,
    ):
        fold_started_at = (
            perf_counter()
        )

        print(
            f"Training {fold_number}/{total_folds} "
            f"| test date "
            f"{pd.Timestamp(fold.test_date).date()}",
            flush=True,
        )

        fold_elapsed = (
            perf_counter()
            - fold_started_at
        )

        print(
            f"Completed {fold_number}/{total_folds} "
            f"in {fold_elapsed:.2f}s",
            flush=True,
        )

        (
            train,
            validation,
            test,
            latest_fit_target_end,
        ) = _prepare_fold_data(
            panel,
            fold,
            minimum_validation_dates=(
                config.minimum_validation_dates
            ),
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

        candidate_results: list[
            dict[str, Any]
        ] = []

        for candidate in (
            config.candidates
        ):
            (
                _,
                best_iteration,
                validation_mean_ic,
                validation_valid_months,
            ) = _evaluate_candidate(
                candidate,
                x_train=x_train,
                y_train=y_train,
                x_validation=(
                    x_validation
                ),
                y_validation=(
                    y_validation
                ),
                validation_dates=(
                    validation[
                        "as_of_date"
                    ]
                ),
                config=config,
            )

            result = {
                "fold_id": str(
                    fold.fold_id
                ),
                "test_date": pd.Timestamp(
                    fold.test_date
                ),
                "candidate_name": (
                    candidate.candidate_name
                ),
                "num_leaves": (
                    candidate.num_leaves
                ),
                "max_depth": (
                    candidate.max_depth
                ),
                "min_child_samples": (
                    candidate.min_child_samples
                ),
                "learning_rate": (
                    candidate.learning_rate
                ),
                "reg_alpha": (
                    candidate.reg_alpha
                ),
                "reg_lambda": (
                    candidate.reg_lambda
                ),
                "colsample_bytree": (
                    candidate.colsample_bytree
                ),
                "subsample": (
                    candidate.subsample
                ),
                "best_iteration": (
                    best_iteration
                ),
                "validation_mean_ic": (
                    validation_mean_ic
                ),
                "validation_valid_months": (
                    validation_valid_months
                ),
                "selected": False,
                "_candidate": candidate,
            }

            candidate_results.append(
                result
            )

        valid_results = [
            result
            for result in candidate_results
            if np.isfinite(
                result[
                    "validation_mean_ic"
                ]
            )
        ]

        if not valid_results:
            raise LightGBMRegressionError(
                f"No LightGBM candidate produced "
                f"a valid validation IC for "
                f"{fold.fold_id}."
            )

        best = max(
            valid_results,
            key=lambda result: (
                result[
                    "validation_mean_ic"
                ],
                -result[
                    "num_leaves"
                ],
                -result[
                    "best_iteration"
                ],
            ),
        )

        best[
            "selected"
        ] = True

        selected_candidate = best[
            "_candidate"
        ]

        selected_estimators = int(
            best[
                "best_iteration"
            ]
        )

        for result in candidate_results:
            stored = {
                key: value
                for key, value in result.items()
                if key != "_candidate"
            }

            search_rows.append(
                stored
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

        final_model = _build_model(
            selected_candidate,
            n_estimators=(
                selected_estimators
            ),
            random_state=(
                config.random_state
            ),
            n_jobs=config.n_jobs,
        )

        final_model.fit(
            x_fit,
            y_fit,
        )

        prediction = final_model.predict(
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
        ] = "lightgbm_regressor"

        block[
            "prediction"
        ] = prediction.astype(
            float
        )

        block[
            "latest_fit_target_end_date"
        ] = latest_fit_target_end

        block[
            "selected_candidate"
        ] = (
            selected_candidate.candidate_name
        )

        block[
            "selected_estimators"
        ] = selected_estimators

        prediction_frames.append(
            block
        )

        booster = final_model.booster_

        gain_importance = (
            booster.feature_importance(
                importance_type="gain"
            )
        )

        split_importance = (
            booster.feature_importance(
                importance_type="split"
            )
        )

        total_gain = float(
            np.sum(
                gain_importance
            )
        )

        for (
            feature,
            gain,
            split,
        ) in zip(
            feature_columns,
            gain_importance,
            split_importance,
            strict=True,
        ):
            importance_rows.append(
                {
                    "fold_id": str(
                        fold.fold_id
                    ),
                    "test_date": pd.Timestamp(
                        fold.test_date
                    ),
                    "feature": feature,
                    "gain": float(
                        gain
                    ),
                    "gain_share": (
                        float(
                            gain
                            / total_gain
                        )
                        if total_gain > 0.0
                        else 0.0
                    ),
                    "split_count": int(
                        split
                    ),
                    "selected_candidate": (
                        selected_candidate.candidate_name
                    ),
                    "selected_estimators": (
                        selected_estimators
                    ),
                }
            )

    if not prediction_frames:
        raise LightGBMRegressionError(
            "No LightGBM predictions were generated."
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    ).sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(
        drop=True
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
        raise LightGBMRegressionError(
            "Generated LightGBM predictions "
            "contain duplicate keys."
        )

    hyperparameter_search = (
        pd.DataFrame(
            search_rows
        )
        .sort_values(
            [
                "test_date",
                "candidate_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    feature_importance = (
        pd.DataFrame(
            importance_rows
        )
        .sort_values(
            [
                "test_date",
                "feature",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return LightGBMRegressionOutputs(
        predictions=predictions,
        hyperparameter_search=(
            hyperparameter_search
        ),
        feature_importance=(
            feature_importance
        ),
        feature_columns=(
            feature_columns
        ),
    )


def summarize_lightgbm_importance(
    feature_importance: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize feature importance across OOS folds."""
    _require_columns(
        feature_importance,
        (
            "feature",
            "gain",
            "gain_share",
            "split_count",
        ),
        dataset_name="feature importance",
    )

    summary = (
        feature_importance.groupby(
            "feature",
            as_index=False,
        )
        .agg(
            mean_gain=(
                "gain",
                "mean",
            ),
            mean_gain_share=(
                "gain_share",
                "mean",
            ),
            median_gain_share=(
                "gain_share",
                "median",
            ),
            mean_split_count=(
                "split_count",
                "mean",
            ),
            folds_used=(
                "split_count",
                lambda values: int(
                    np.sum(
                        np.asarray(
                            values
                        )
                        > 0
                    )
                ),
            ),
        )
    )

    return summary.sort_values(
        "mean_gain_share",
        ascending=False,
    ).reset_index(
        drop=True
    )