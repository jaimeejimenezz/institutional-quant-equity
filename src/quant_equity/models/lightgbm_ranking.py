"""Walk-forward learning-to-rank models for cross-sectional equities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import (
    LGBMRanker,
    early_stopping,
    log_evaluation,
)

from quant_equity.models.lightgbm_regression import (
    DEFAULT_CANDIDATES,
    LightGBMCandidate,
)
from quant_equity.models.regularized_linear import (
    detect_model_features,
    fit_feature_preprocessor,
)


class LightGBMRankingError(ValueError):
    """Raised when learning-to-rank training cannot be completed."""


TARGET_COLUMN = "target_21d_excess"
TOP_LABEL_COLUMN = "label_top_quintile"


@dataclass(frozen=True)
class LightGBMRankingConfig:
    """Configuration for cross-sectional learning to rank."""

    expected_feature_count: int = 91
    relevance_levels: int = 5
    ndcg_cutoff: int = 10
    lambdarank_truncation_level: int = 13
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
        """Validate ranking configuration."""
        if self.expected_feature_count < 1:
            raise LightGBMRankingError(
                "expected_feature_count must be positive."
            )

        if self.relevance_levels < 2:
            raise LightGBMRankingError(
                "At least two relevance levels are required."
            )

        if self.ndcg_cutoff < 1:
            raise LightGBMRankingError(
                "ndcg_cutoff must be positive."
            )

        if (
            self.lambdarank_truncation_level
            <= self.ndcg_cutoff
        ):
            raise LightGBMRankingError(
                "lambdarank_truncation_level should exceed "
                "the NDCG cutoff."
            )

        if not self.candidates:
            raise LightGBMRankingError(
                "At least one ranking candidate is required."
            )


@dataclass(frozen=True)
class LightGBMRankingOutputs:
    """Artifacts produced by walk-forward ranking."""

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
        raise LightGBMRankingError(
            f"{dataset_name} is missing columns: "
            + ", ".join(missing)
            + "."
        )


def build_relevance_labels(
    data: pd.DataFrame,
    *,
    levels: int = 5,
) -> pd.Series:
    """Convert each monthly target cross-section into relevance grades."""
    _require_columns(
        data,
        (
            "as_of_date",
            "ticker",
            TARGET_COLUMN,
        ),
        dataset_name="ranking data",
    )

    if levels < 2:
        raise LightGBMRankingError(
            "At least two relevance levels are required."
        )

    result = pd.Series(
        index=data.index,
        dtype="int64",
    )

    for _, month in data.groupby(
        "as_of_date",
        sort=True,
    ):
        target = pd.to_numeric(
            month[TARGET_COLUMN],
            errors="coerce",
        )

        if (
            target.isna().any()
            or not np.isfinite(
                target.to_numpy(
                    dtype=float
                )
            ).all()
        ):
            raise LightGBMRankingError(
                "Ranking targets must be finite."
            )

        ordered = (
            month.assign(
                _target=target
            )
            .sort_values(
                [
                    "_target",
                    "ticker",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
        )

        count = len(
            ordered
        )

        if count < levels:
            raise LightGBMRankingError(
                "Cross-section is too small for "
                f"{levels} relevance levels."
            )

        positions = np.arange(
            count
        )

        grades = np.minimum(
            (
                positions
                * levels
                // count
            ),
            levels - 1,
        )

        result.loc[
            ordered.index
        ] = grades.astype(
            int
        )

    return result.astype(
        int
    )


def build_group_sizes(
    data: pd.DataFrame,
) -> np.ndarray:
    """Build contiguous query sizes for LightGBM."""
    if data.empty:
        raise LightGBMRankingError(
            "Ranking data cannot be empty."
        )

    dates = pd.to_datetime(
        data[
            "as_of_date"
        ]
    )

    if not dates.is_monotonic_increasing:
        raise LightGBMRankingError(
            "Ranking observations must be sorted by date."
        )

    group_sizes = (
        data.groupby(
            "as_of_date",
            sort=False,
        )
        .size()
        .to_numpy(
            dtype=int
        )
    )

    if int(
        group_sizes.sum()
    ) != len(
        data
    ):
        raise LightGBMRankingError(
            "Ranking group sizes do not match row count."
        )

    return group_sizes


def _mean_monthly_spearman(
    prediction: np.ndarray,
    target: pd.Series,
    dates: pd.Series,
) -> tuple[
    float,
    int,
]:
    """Calculate validation mean cross-sectional Spearman IC."""
    frame = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(
                dates
            ).to_numpy(),
            "prediction": np.asarray(
                prediction,
                dtype=float,
            ),
            "target": pd.to_numeric(
                target,
                errors="coerce",
            ).to_numpy(
                dtype=float
            ),
        }
    )

    values = []

    for _, month in frame.groupby(
        "as_of_date",
        sort=True,
    ):
        if (
            month[
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
            values.append(
                float(
                    ic
                )
            )

    if not values:
        return (
            np.nan,
            0,
        )

    return (
        float(
            np.mean(
                values
            )
        ),
        len(
            values
        ),
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
    """Extract and validate one temporal ranking fold."""
    test_date = pd.Timestamp(
        fold.test_date
    )

    train = panel.loc[
        panel[
            "as_of_date"
        ].between(
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
        panel[
            "as_of_date"
        ].between(
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
        panel[
            "as_of_date"
        ].eq(
            test_date
        )
    ].copy()

    if train.empty or validation.empty:
        raise LightGBMRankingError(
            f"{fold.fold_id} has an empty fitting partition."
        )

    if (
        validation[
            "as_of_date"
        ].nunique()
        < minimum_validation_dates
    ):
        raise LightGBMRankingError(
            f"{fold.fold_id} has insufficient "
            "validation dates."
        )

    if len(
        test
    ) != int(
        fold.test_rows
    ):
        raise LightGBMRankingError(
            f"{fold.fold_id} has an invalid "
            "test cross-section size."
        )

    fitting = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    target_end = pd.to_datetime(
        fitting[
            "target_end_date"
        ],
        errors="coerce",
    ).dt.normalize()

    if target_end.isna().any():
        raise LightGBMRankingError(
            f"{fold.fold_id} contains invalid "
            "target maturity dates."
        )

    violations = int(
        target_end.gt(
            test_date
        ).sum()
    )

    if violations:
        raise LightGBMRankingError(
            f"{fold.fold_id} contains "
            f"{violations} fitting labels "
            "that were not mature."
        )

    train = train.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    )

    validation = (
        validation.sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
    )

    test = test.sort_values(
        "ticker"
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
    config: LightGBMRankingConfig,
    n_estimators: int,
) -> LGBMRanker:
    """Create one controlled LambdaRank model."""
    label_gain = [
        (2**level) - 1
        for level in range(
            config.relevance_levels
        )
    ]

    return LGBMRanker(
        objective="lambdarank",
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
        random_state=(
            config.random_state
        ),
        n_jobs=(
            config.n_jobs
        ),
        importance_type="gain",
        label_gain=label_gain,
        lambdarank_truncation_level=(
            config.lambdarank_truncation_level
        ),
        verbosity=-1,
        force_col_wise=True,
    )


def _evaluate_candidate(
    candidate: LightGBMCandidate,
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_group: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_group: np.ndarray,
    validation_target: pd.Series,
    validation_dates: pd.Series,
    config: LightGBMRankingConfig,
) -> tuple[
    int,
    float,
    int,
    float,
]:
    """Train one candidate and measure validation ranking quality."""
    model = _build_model(
        candidate,
        config=config,
        n_estimators=(
            config.max_estimators
        ),
    )

    model.fit(
        x_train,
        y_train,
        group=train_group,
        eval_X=x_validation,
        eval_y=y_validation,
        eval_group=[
            validation_group
        ],
        eval_metric="ndcg",
        eval_at=(
            config.ndcg_cutoff,
        ),
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

    prediction = model.predict(
        x_validation,
        num_iteration=(
            best_iteration
        ),
    )

    mean_ic, valid_months = (
        _mean_monthly_spearman(
            prediction,
            validation_target,
            validation_dates,
        )
    )

    best_score = float(
        next(
            iter(
                model.best_score_[
                    "valid_0"
                ].values()
            )
        )
    )

    return (
        best_iteration,
        mean_ic,
        valid_months,
        best_score,
    )


def train_lightgbm_ranking(
    panel: pd.DataFrame,
    fold_metadata: pd.DataFrame,
    *,
    config: LightGBMRankingConfig | None = None,
) -> LightGBMRankingOutputs:
    """Train a selected learning-to-rank model for every OOS month."""
    if config is None:
        config = (
            LightGBMRankingConfig()
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
        panel[
            "as_of_date"
        ]
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

    prediction_frames = []
    search_rows = []
    importance_rows = []

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

        y_train = (
            build_relevance_labels(
                train,
                levels=(
                    config.relevance_levels
                ),
            )
        )

        y_validation = (
            build_relevance_labels(
                validation,
                levels=(
                    config.relevance_levels
                ),
            )
        )

        train_group = (
            build_group_sizes(
                train
            )
        )

        validation_group = (
            build_group_sizes(
                validation
            )
        )

        candidate_results = []

        for candidate in (
            config.candidates
        ):
            (
                best_iteration,
                validation_mean_ic,
                validation_valid_months,
                validation_ndcg,
            ) = _evaluate_candidate(
                candidate,
                x_train=x_train,
                y_train=y_train,
                train_group=train_group,
                x_validation=(
                    x_validation
                ),
                y_validation=(
                    y_validation
                ),
                validation_group=(
                    validation_group
                ),
                validation_target=(
                    validation[
                        TARGET_COLUMN
                    ]
                ),
                validation_dates=(
                    validation[
                        "as_of_date"
                    ]
                ),
                config=config,
            )

            candidate_results.append(
                {
                    "fold_id": str(
                        fold.fold_id
                    ),
                    "test_date": (
                        pd.Timestamp(
                            fold.test_date
                        )
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
                    "best_iteration": (
                        best_iteration
                    ),
                    "validation_mean_ic": (
                        validation_mean_ic
                    ),
                    "validation_valid_months": (
                        validation_valid_months
                    ),
                    "validation_ndcg_at_10": (
                        validation_ndcg
                    ),
                    "selected": False,
                    "_candidate": candidate,
                }
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
            raise LightGBMRankingError(
                f"No ranking candidate produced "
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

        selected_candidate = (
            best[
                "_candidate"
            ]
        )

        selected_estimators = int(
            best[
                "best_iteration"
            ]
        )

        for result in candidate_results:
            search_rows.append(
                {
                    key: value
                    for key, value
                    in result.items()
                    if key != "_candidate"
                }
            )

        fitting = pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        ).sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )

        x_fit = (
            preprocessor.transform(
                fitting
            )
        )

        y_fit = (
            build_relevance_labels(
                fitting,
                levels=(
                    config.relevance_levels
                ),
            )
        )

        fitting_group = (
            build_group_sizes(
                fitting
            )
        )

        final_model = _build_model(
            selected_candidate,
            config=config,
            n_estimators=(
                selected_estimators
            ),
        )

        final_model.fit(
            x_fit,
            y_fit,
            group=fitting_group,
        )

        prediction = (
            final_model.predict(
                x_test
            )
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
        ] = "lightgbm_ranker"

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

        booster = (
            final_model.booster_
        )

        gain = (
            booster.feature_importance(
                importance_type="gain"
            )
        )

        split = (
            booster.feature_importance(
                importance_type="split"
            )
        )

        total_gain = float(
            gain.sum()
        )

        for (
            feature,
            gain_value,
            split_value,
        ) in zip(
            feature_columns,
            gain,
            split,
            strict=True,
        ):
            importance_rows.append(
                {
                    "fold_id": str(
                        fold.fold_id
                    ),
                    "test_date": (
                        pd.Timestamp(
                            fold.test_date
                        )
                    ),
                    "feature": feature,
                    "gain": float(
                        gain_value
                    ),
                    "gain_share": (
                        float(
                            gain_value
                            / total_gain
                        )
                        if total_gain
                        > 0.0
                        else 0.0
                    ),
                    "split_count": int(
                        split_value
                    ),
                }
            )

    predictions = (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    duplicate_keys = int(
        predictions.duplicated(
            [
                "fold_id",
                "ticker",
                "model_name",
            ]
        ).sum()
    )

    if duplicate_keys:
        raise LightGBMRankingError(
            "Ranking predictions contain duplicate keys."
        )

    return LightGBMRankingOutputs(
        predictions=predictions,
        hyperparameter_search=(
            pd.DataFrame(
                search_rows
            )
        ),
        feature_importance=(
            pd.DataFrame(
                importance_rows
            )
        ),
        feature_columns=(
            feature_columns
        ),
    )


def summarize_lightgbm_ranking_importance(
    feature_importance: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize ranking feature importance across folds."""
    return (
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
        .sort_values(
            "mean_gain_share",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )