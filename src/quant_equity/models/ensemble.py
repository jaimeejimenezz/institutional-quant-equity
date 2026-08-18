"""Stable ensemble construction for cross-sectional equity signals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from quant_equity.models.model_baselines import (
    score_technical_composite,
)


class EnsembleError(ValueError):
    """Raised when the alpha ensemble cannot be constructed."""


COMPONENT_MODELS = (
    "technical_equal_weight_composite",
    "elastic_net",
    "lightgbm_ranker",
)

COMPONENT_ALIASES = {
    "technical_equal_weight_composite": "composite",
    "elastic_net": "elastic_net",
    "lightgbm_ranker": "lightgbm_ranker",
}


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for final alpha ensemble construction."""

    expected_cross_section_size: int = 50
    equal_weight_prior: float = 0.50
    minimum_validation_dates: int = 12

    def validate(self) -> None:
        """Validate ensemble configuration."""
        if self.expected_cross_section_size < 2:
            raise EnsembleError(
                "expected_cross_section_size must exceed one."
            )

        if not 0.0 <= self.equal_weight_prior <= 1.0:
            raise EnsembleError(
                "equal_weight_prior must be between 0 and 1."
            )

        if self.minimum_validation_dates < 1:
            raise EnsembleError(
                "minimum_validation_dates must be positive."
            )


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
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
        raise EnsembleError(
            f"{dataset_name} is missing columns: "
            + ", ".join(missing)
            + "."
        )


def _selected_mask(
    data: pd.DataFrame,
) -> pd.Series:
    """Return a robust boolean mask for selected hyperparameters."""
    selected = data[
        "selected"
    ]

    if pd.api.types.is_bool_dtype(
        selected
    ):
        return selected.fillna(
            False
        )

    return (
        selected.astype(
            str
        )
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )


def _cross_section_percentile(
    values: pd.Series,
) -> pd.Series:
    """Map one cross-section to a stable score between zero and one."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    if numeric.isna().any():
        raise EnsembleError(
            "Component predictions contain missing values."
        )

    if not np.isfinite(
        numeric.to_numpy()
    ).all():
        raise EnsembleError(
            "Component predictions contain non-finite values."
        )

    if numeric.nunique() < 2:
        return pd.Series(
            0.5,
            index=numeric.index,
            dtype=float,
        )

    rank = numeric.rank(
        method="average",
        ascending=True,
    )

    return (
        rank
        - 1.0
    ) / (
        len(
            numeric
        )
        - 1.0
    )


def _mean_monthly_spearman(
    prediction: pd.Series,
    target: pd.Series,
    dates: pd.Series,
) -> tuple[
    float,
    int,
]:
    """Calculate mean monthly cross-sectional Spearman IC."""
    frame = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(
                dates
            ),
            "prediction": pd.to_numeric(
                prediction,
                errors="coerce",
            ),
            "target": pd.to_numeric(
                target,
                errors="coerce",
            ),
        }
    )

    monthly_ic = []

    for _, month in frame.groupby(
        "as_of_date",
        sort=True,
    ):
        valid = month.dropna(
            subset=[
                "prediction",
                "target",
            ]
        )

        if (
            len(valid) < 2
            or valid[
                "prediction"
            ].nunique()
            < 2
            or valid[
                "target"
            ].nunique()
            < 2
        ):
            continue

        ic = valid[
            "prediction"
        ].corr(
            valid[
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
            np.nan,
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


def _validation_weights_from_ic(
    ic_values: np.ndarray,
    *,
    equal_weight_prior: float,
) -> np.ndarray:
    """Convert validation IC values into conservative positive weights."""
    values = np.asarray(
        ic_values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    positive = np.clip(
        values,
        a_min=0.0,
        a_max=None,
    )

    component_count = len(
        positive
    )

    equal_weights = np.full(
        component_count,
        1.0
        / component_count,
        dtype=float,
    )

    if positive.sum() > 0.0:
        evidence_weights = (
            positive
            / positive.sum()
        )
    else:
        evidence_weights = (
            equal_weights.copy()
        )

    weights = (
        equal_weight_prior
        * equal_weights
        + (
            1.0
            - equal_weight_prior
        )
        * evidence_weights
    )

    return (
        weights
        / weights.sum()
    )


def build_validation_weights(
    panel: pd.DataFrame,
    fold_metadata: pd.DataFrame,
    elastic_hyperparameters: pd.DataFrame,
    ranking_hyperparameters: pd.DataFrame,
    *,
    config: EnsembleConfig | None = None,
) -> pd.DataFrame:
    """Build fold-specific weights using validation information only."""
    if config is None:
        config = EnsembleConfig()

    config.validate()

    _require_columns(
        panel,
        (
            "as_of_date",
            "ticker",
            "target_21d_excess",
        ),
        dataset_name="modeling panel",
    )

    _require_columns(
        fold_metadata,
        (
            "fold_id",
            "test_date",
            "validation_start_date",
            "validation_end_date",
        ),
        dataset_name="walk-forward metadata",
    )

    _require_columns(
        elastic_hyperparameters,
        (
            "fold_id",
            "model_name",
            "validation_mean_ic",
            "selected",
        ),
        dataset_name="linear hyperparameter research",
    )

    _require_columns(
        ranking_hyperparameters,
        (
            "fold_id",
            "validation_mean_ic",
            "selected",
        ),
        dataset_name="ranking hyperparameter research",
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

    for column in (
        "test_date",
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

    elastic_selected = (
        elastic_hyperparameters.loc[
            _selected_mask(
                elastic_hyperparameters
            )
            & elastic_hyperparameters[
                "model_name"
            ].eq(
                "elastic_net"
            ),
            [
                "fold_id",
                "validation_mean_ic",
            ],
        ]
        .rename(
            columns={
                "validation_mean_ic": (
                    "elastic_net_validation_ic"
                )
            }
        )
    )

    ranking_selected = (
        ranking_hyperparameters.loc[
            _selected_mask(
                ranking_hyperparameters
            ),
            [
                "fold_id",
                "validation_mean_ic",
            ],
        ]
        .rename(
            columns={
                "validation_mean_ic": (
                    "lightgbm_ranker_validation_ic"
                )
            }
        )
    )

    if elastic_selected[
        "fold_id"
    ].duplicated().any():
        raise EnsembleError(
            "Elastic Net has multiple selected configurations "
            "for the same fold."
        )

    if ranking_selected[
        "fold_id"
    ].duplicated().any():
        raise EnsembleError(
            "LightGBM Ranker has multiple selected configurations "
            "for the same fold."
        )

    rows = []

    for fold in folds.sort_values(
        "test_date"
    ).itertuples(
        index=False
    ):
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

        validation_dates = int(
            validation[
                "as_of_date"
            ].nunique()
        )

        if (
            validation_dates
            < config.minimum_validation_dates
        ):
            raise EnsembleError(
                f"{fold.fold_id} has only "
                f"{validation_dates} validation dates."
            )

        composite_prediction = (
            score_technical_composite(
                validation
            )
        )

        (
            composite_ic,
            composite_valid_months,
        ) = _mean_monthly_spearman(
            composite_prediction,
            validation[
                "target_21d_excess"
            ],
            validation[
                "as_of_date"
            ],
        )

        elastic_row = (
            elastic_selected.loc[
                elastic_selected[
                    "fold_id"
                ].astype(
                    str
                ).eq(
                    str(
                        fold.fold_id
                    )
                )
            ]
        )

        ranking_row = (
            ranking_selected.loc[
                ranking_selected[
                    "fold_id"
                ].astype(
                    str
                ).eq(
                    str(
                        fold.fold_id
                    )
                )
            ]
        )

        if len(
            elastic_row
        ) != 1:
            raise EnsembleError(
                f"Missing selected Elastic Net configuration "
                f"for {fold.fold_id}."
            )

        if len(
            ranking_row
        ) != 1:
            raise EnsembleError(
                f"Missing selected ranking configuration "
                f"for {fold.fold_id}."
            )

        elastic_ic = float(
            elastic_row.iloc[
                0
            ][
                "elastic_net_validation_ic"
            ]
        )

        ranking_ic = float(
            ranking_row.iloc[
                0
            ][
                "lightgbm_ranker_validation_ic"
            ]
        )

        weights = (
            _validation_weights_from_ic(
                np.array(
                    [
                        composite_ic,
                        elastic_ic,
                        ranking_ic,
                    ]
                ),
                equal_weight_prior=(
                    config.equal_weight_prior
                ),
            )
        )

        rows.append(
            {
                "fold_id": str(
                    fold.fold_id
                ),
                "test_date": pd.Timestamp(
                    fold.test_date
                ),
                "composite_validation_ic": (
                    composite_ic
                ),
                "composite_validation_months": (
                    composite_valid_months
                ),
                "elastic_net_validation_ic": (
                    elastic_ic
                ),
                "lightgbm_ranker_validation_ic": (
                    ranking_ic
                ),
                "composite_weight": (
                    float(
                        weights[0]
                    )
                ),
                "elastic_net_weight": (
                    float(
                        weights[1]
                    )
                ),
                "lightgbm_ranker_weight": (
                    float(
                        weights[2]
                    )
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    weight_sum = (
        result[
            [
                "composite_weight",
                "elastic_net_weight",
                "lightgbm_ranker_weight",
            ]
        ]
        .sum(
            axis=1
        )
    )

    if not np.allclose(
        weight_sum,
        1.0,
    ):
        raise EnsembleError(
            "Validation weights do not sum to one."
        )

    return result


def build_component_scores(
    predictions: pd.DataFrame,
    *,
    config: EnsembleConfig | None = None,
) -> pd.DataFrame:
    """Align component predictions and convert them to monthly percentiles."""
    if config is None:
        config = EnsembleConfig()

    config.validate()

    _require_columns(
        predictions,
        (
            "fold_id",
            "as_of_date",
            "ticker",
            "sector",
            "model_name",
            "prediction",
            "target_21d_excess",
            "label_top_quintile",
        ),
        dataset_name="out-of-sample predictions",
    )

    selected = predictions.loc[
        predictions[
            "model_name"
        ].isin(
            COMPONENT_MODELS
        )
    ].copy()

    observed_models = set(
        selected[
            "model_name"
        ].unique()
    )

    if observed_models != set(
        COMPONENT_MODELS
    ):
        raise EnsembleError(
            "The ensemble requires exactly the frozen "
            "composite, Elastic Net and LightGBM Ranker signals."
        )

    selected[
        "as_of_date"
    ] = pd.to_datetime(
        selected[
            "as_of_date"
        ]
    ).dt.normalize()

    keys = [
        "fold_id",
        "as_of_date",
        "ticker",
    ]

    counts = (
        selected.groupby(
            keys
        )
        .size()
    )

    if not counts.eq(
        len(
            COMPONENT_MODELS
        )
    ).all():
        raise EnsembleError(
            "Component predictions are not perfectly aligned."
        )

    for column in (
        "sector",
        "target_21d_excess",
        "label_top_quintile",
    ):
        consistency = (
            selected.groupby(
                keys
            )[
                column
            ]
            .nunique(
                dropna=False
            )
        )

        if not consistency.eq(
            1
        ).all():
            raise EnsembleError(
                f"{column} differs across component models."
            )

    metadata = (
        selected.groupby(
            keys,
            as_index=False,
        )
        .agg(
            sector=(
                "sector",
                "first",
            ),
            target_21d_excess=(
                "target_21d_excess",
                "first",
            ),
            label_top_quintile=(
                "label_top_quintile",
                "first",
            ),
        )
    )

    wide = (
        selected.pivot(
            index=keys,
            columns="model_name",
            values="prediction",
        )
        .reset_index()
    )

    wide.columns.name = None

    result = metadata.merge(
        wide,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    cross_section_sizes = (
        result.groupby(
            "as_of_date"
        )[
            "ticker"
        ]
        .nunique()
    )

    if not cross_section_sizes.eq(
        config.expected_cross_section_size
    ).all():
        raise EnsembleError(
            "Unexpected ensemble cross-section size."
        )

    for model_name in COMPONENT_MODELS:
        alias = COMPONENT_ALIASES[
            model_name
        ]

        raw_column = (
            f"{alias}_raw"
        )

        percentile_column = (
            f"{alias}_percentile"
        )

        result[
            raw_column
        ] = pd.to_numeric(
            result[
                model_name
            ],
            errors="coerce",
        )

        result[
            percentile_column
        ] = (
            result.groupby(
                "as_of_date"
            )[
                raw_column
            ]
            .transform(
                _cross_section_percentile
            )
        )

        result = result.drop(
            columns=[
                model_name
            ]
        )

    return result.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(
        drop=True
    )


def _evaluation_block(
    data: pd.DataFrame,
    *,
    model_name: str,
    prediction: pd.Series,
) -> pd.DataFrame:
    """Build one long-format evaluation block."""
    block = data.loc[
        :,
        [
            "fold_id",
            "as_of_date",
            "ticker",
            "sector",
            "target_21d_excess",
            "label_top_quintile",
        ],
    ].copy()

    block[
        "model_name"
    ] = model_name

    block[
        "prediction"
    ] = prediction.astype(
        float
    )

    return block


def build_ensemble_candidates(
    component_scores: pd.DataFrame,
    validation_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Create a small pre-defined ensemble comparison set."""
    data = component_scores.merge(
        validation_weights[
            [
                "fold_id",
                "composite_weight",
                "elastic_net_weight",
                "lightgbm_ranker_weight",
            ]
        ],
        on="fold_id",
        how="left",
        validate="many_to_one",
    )

    weight_columns = [
        "composite_weight",
        "elastic_net_weight",
        "lightgbm_ranker_weight",
    ]

    if data[
        weight_columns
    ].isna().any().any():
        raise EnsembleError(
            "Some OOS rows do not have validation weights."
        )

    equal_three = (
        data[
            [
                "composite_percentile",
                "elastic_net_percentile",
                "lightgbm_ranker_percentile",
            ]
        ]
        .mean(
            axis=1
        )
    )

    core_two = (
        data[
            [
                "composite_percentile",
                "elastic_net_percentile",
            ]
        ]
        .mean(
            axis=1
        )
    )

    validation_weighted = (
        data[
            "composite_percentile"
        ]
        * data[
            "composite_weight"
        ]
        + data[
            "elastic_net_percentile"
        ]
        * data[
            "elastic_net_weight"
        ]
        + data[
            "lightgbm_ranker_percentile"
        ]
        * data[
            "lightgbm_ranker_weight"
        ]
    )

    blocks = [
        _evaluation_block(
            data,
            model_name=(
                "equal_percentile_ensemble"
            ),
            prediction=(
                equal_three
            ),
        ),
        _evaluation_block(
            data,
            model_name=(
                "core_percentile_ensemble"
            ),
            prediction=(
                core_two
            ),
        ),
        _evaluation_block(
            data,
            model_name=(
                "validation_weighted_ensemble"
            ),
            prediction=(
                validation_weighted
            ),
        ),
    ]

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def build_final_alpha_signal(
    component_scores: pd.DataFrame,
    validation_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Build the production-facing final alpha signal."""
    data = component_scores.merge(
        validation_weights[
            [
                "fold_id",
                "composite_weight",
                "elastic_net_weight",
                "lightgbm_ranker_weight",
            ]
        ],
        on="fold_id",
        how="left",
        validate="many_to_one",
    )

    data[
        "composite_contribution"
    ] = (
        data[
            "composite_percentile"
        ]
        * data[
            "composite_weight"
        ]
    )

    data[
        "elastic_net_contribution"
    ] = (
        data[
            "elastic_net_percentile"
        ]
        * data[
            "elastic_net_weight"
        ]
    )

    data[
        "lightgbm_ranker_contribution"
    ] = (
        data[
            "lightgbm_ranker_percentile"
        ]
        * data[
            "lightgbm_ranker_weight"
        ]
    )

    data[
        "raw_prediction"
    ] = data[
        [
            "composite_contribution",
            "elastic_net_contribution",
            "lightgbm_ranker_contribution",
        ]
    ].sum(
        axis=1
    )

    data[
        "percentile_score"
    ] = (
        data.groupby(
            "as_of_date"
        )[
            "raw_prediction"
        ]
        .transform(
            _cross_section_percentile
        )
    )

    ranked_parts = []

    for _, month in data.groupby(
        "as_of_date",
        sort=True,
    ):
        month = month.sort_values(
            [
                "raw_prediction",
                "ticker",
            ],
            ascending=[
                False,
                True,
            ],
        ).copy()

        month[
            "rank"
        ] = np.arange(
            1,
            len(month) + 1,
            dtype=int,
        )

        ranked_parts.append(
            month
        )

    data = pd.concat(
        ranked_parts,
        ignore_index=True,
    )

    data[
        "model_contributions"
    ] = data.apply(
        lambda row: json.dumps(
            {
                "technical_composite": {
                    "weight": float(
                        row[
                            "composite_weight"
                        ]
                    ),
                    "percentile": float(
                        row[
                            "composite_percentile"
                        ]
                    ),
                    "contribution": float(
                        row[
                            "composite_contribution"
                        ]
                    ),
                },
                "elastic_net": {
                    "weight": float(
                        row[
                            "elastic_net_weight"
                        ]
                    ),
                    "percentile": float(
                        row[
                            "elastic_net_percentile"
                        ]
                    ),
                    "contribution": float(
                        row[
                            "elastic_net_contribution"
                        ]
                    ),
                },
                "lightgbm_ranker": {
                    "weight": float(
                        row[
                            "lightgbm_ranker_weight"
                        ]
                    ),
                    "percentile": float(
                        row[
                            "lightgbm_ranker_percentile"
                        ]
                    ),
                    "contribution": float(
                        row[
                            "lightgbm_ranker_contribution"
                        ]
                    ),
                },
            },
            sort_keys=True,
        ),
        axis=1,
    )

    expected_raw = data[
        [
            "composite_contribution",
            "elastic_net_contribution",
            "lightgbm_ranker_contribution",
        ]
    ].sum(
        axis=1
    )

    if not np.allclose(
        expected_raw,
        data[
            "raw_prediction"
        ],
    ):
        raise EnsembleError(
            "Model contributions do not sum to the final score."
        )

    output_columns = [
        "fold_id",
        "as_of_date",
        "ticker",
        "sector",
        "raw_prediction",
        "percentile_score",
        "rank",
        "composite_weight",
        "elastic_net_weight",
        "lightgbm_ranker_weight",
        "composite_contribution",
        "elastic_net_contribution",
        "lightgbm_ranker_contribution",
        "model_contributions",
    ]

    return data.loc[
        :,
        output_columns,
    ].sort_values(
        [
            "as_of_date",
            "rank",
        ]
    ).reset_index(
        drop=True
    )


def compute_component_correlations(
    component_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize monthly rank correlation between ensemble components."""
    columns = {
        "composite": (
            "composite_percentile"
        ),
        "elastic_net": (
            "elastic_net_percentile"
        ),
        "lightgbm_ranker": (
            "lightgbm_ranker_percentile"
        ),
    }

    rows = []

    for (
        name_a,
        name_b,
    ) in combinations(
        columns,
        2,
    ):
        monthly_values = []

        for _, month in component_scores.groupby(
            "as_of_date",
            sort=True,
        ):
            correlation = month[
                columns[
                    name_a
                ]
            ].corr(
                month[
                    columns[
                        name_b
                    ]
                ],
                method="spearman",
            )

            if pd.notna(
                correlation
            ):
                monthly_values.append(
                    float(
                        correlation
                    )
                )

        rows.append(
            {
                "signal_a": name_a,
                "signal_b": name_b,
                "months": len(
                    monthly_values
                ),
                "mean_spearman": float(
                    np.mean(
                        monthly_values
                    )
                ),
                "median_spearman": float(
                    np.median(
                        monthly_values
                    )
                ),
                "minimum_spearman": float(
                    np.min(
                        monthly_values
                    )
                ),
                "maximum_spearman": float(
                    np.max(
                        monthly_values
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_ablation_candidates(
    component_scores: pd.DataFrame,
    validation_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Remove one component at a time and renormalize remaining weights."""
    data = component_scores.merge(
        validation_weights[
            [
                "fold_id",
                "composite_weight",
                "elastic_net_weight",
                "lightgbm_ranker_weight",
            ]
        ],
        on="fold_id",
        how="left",
        validate="many_to_one",
    )

    components = {
        "composite": (
            "composite_percentile",
            "composite_weight",
        ),
        "elastic_net": (
            "elastic_net_percentile",
            "elastic_net_weight",
        ),
        "lightgbm_ranker": (
            "lightgbm_ranker_percentile",
            "lightgbm_ranker_weight",
        ),
    }

    blocks = []

    for omitted in components:
        remaining = [
            name
            for name in components
            if name != omitted
        ]

        denominator = sum(
            data[
                components[
                    name
                ][1]
            ]
            for name in remaining
        )

        prediction = sum(
            data[
                components[
                    name
                ][0]
            ]
            * data[
                components[
                    name
                ][1]
            ]
            for name in remaining
        ) / denominator

        blocks.append(
            _evaluation_block(
                data,
                model_name=(
                    f"without_{omitted}"
                ),
                prediction=(
                    prediction
                ),
            )
        )

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def compute_sector_signal_diagnostics(
    final_signal: pd.DataFrame,
    *,
    top_fraction: float = 0.20,
) -> pd.DataFrame:
    """Measure sector tilts in the final alpha ranking."""
    rows = []

    for (
        as_of_date,
        month,
    ) in final_signal.groupby(
        "as_of_date",
        sort=True,
    ):
        top_count = max(
            1,
            int(
                round(
                    len(month)
                    * top_fraction
                )
            ),
        )

        top_tickers = set(
            month.nsmallest(
                top_count,
                "rank",
            )[
                "ticker"
            ]
        )

        for (
            sector,
            group,
        ) in month.groupby(
            "sector",
            sort=True,
        ):
            sector_top_count = int(
                group[
                    "ticker"
                ]
                .isin(
                    top_tickers
                )
                .sum()
            )

            rows.append(
                {
                    "as_of_date": (
                        as_of_date
                    ),
                    "sector": sector,
                    "companies": len(
                        group
                    ),
                    "mean_percentile_score": (
                        float(
                            group[
                                "percentile_score"
                            ].mean()
                        )
                    ),
                    "top_group_companies": (
                        sector_top_count
                    ),
                    "top_group_share": (
                        float(
                            sector_top_count
                            / top_count
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )