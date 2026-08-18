"""Diagnostics for out-of-sample equity model comparison."""

from __future__ import annotations

from itertools import combinations
from math import ceil

import numpy as np
import pandas as pd


class ModelDiagnosticsError(ValueError):
    """Raised when model diagnostics cannot be calculated."""


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
        raise ModelDiagnosticsError(
            f"{dataset_name} is missing columns: "
            + ", ".join(missing)
            + "."
        )


def compute_yearly_stability(
    monthly_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize model behaviour by calendar year."""
    _require_columns(
        monthly_metrics,
        (
            "model_name",
            "as_of_date",
            "ic",
            "top_bottom_spread",
            "top_quintile_precision",
            "top_quintile_turnover",
        ),
        dataset_name="monthly model metrics",
    )

    data = monthly_metrics.copy()

    data[
        "as_of_date"
    ] = pd.to_datetime(
        data[
            "as_of_date"
        ]
    )

    data[
        "year"
    ] = data[
        "as_of_date"
    ].dt.year

    rows = []

    for (
        model_name,
        year,
    ), group in data.groupby(
        [
            "model_name",
            "year",
        ],
        sort=True,
    ):
        valid_ic = (
            pd.to_numeric(
                group["ic"],
                errors="coerce",
            )
            .dropna()
        )

        rows.append(
            {
                "model_name": model_name,
                "year": int(year),
                "months": len(group),
                "valid_ic_months": len(
                    valid_ic
                ),
                "mean_ic": (
                    float(
                        valid_ic.mean()
                    )
                    if not valid_ic.empty
                    else np.nan
                ),
                "median_ic": (
                    float(
                        valid_ic.median()
                    )
                    if not valid_ic.empty
                    else np.nan
                ),
                "positive_ic_ratio": (
                    float(
                        valid_ic.gt(
                            0.0
                        ).mean()
                    )
                    if not valid_ic.empty
                    else np.nan
                ),
                "mean_top_bottom_spread": (
                    float(
                        group[
                            "top_bottom_spread"
                        ].mean()
                    )
                ),
                "mean_top_quintile_precision": (
                    float(
                        group[
                            "top_quintile_precision"
                        ].mean()
                    )
                ),
                "mean_top_quintile_turnover": (
                    float(
                        group[
                            "top_quintile_turnover"
                        ].mean()
                    )
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "model_name",
                "year",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _spearman(
    prediction: pd.Series,
    target: pd.Series,
) -> float:
    """Calculate Spearman correlation when defined."""
    valid = (
        prediction.notna()
        & target.notna()
    )

    x = prediction.loc[
        valid
    ]

    y = target.loc[
        valid
    ]

    if (
        len(x) < 3
        or x.nunique() < 2
        or y.nunique() < 2
    ):
        return np.nan

    return float(
        x.corr(
            y,
            method="spearman",
        )
    )


def compute_sector_stability(
    predictions: pd.DataFrame,
    *,
    minimum_companies: int = 3,
) -> pd.DataFrame:
    """Measure ranking quality within sectors."""
    _require_columns(
        predictions,
        (
            "model_name",
            "as_of_date",
            "ticker",
            "sector",
            "prediction",
            "target_21d_excess",
        ),
        dataset_name="model predictions",
    )

    data = predictions.copy()

    data[
        "as_of_date"
    ] = pd.to_datetime(
        data[
            "as_of_date"
        ]
    )

    monthly_rows = []

    for (
        model_name,
        as_of_date,
        sector,
    ), group in data.groupby(
        [
            "model_name",
            "as_of_date",
            "sector",
        ],
        sort=True,
    ):
        if (
            len(group)
            < minimum_companies
        ):
            continue

        ic = _spearman(
            pd.to_numeric(
                group[
                    "prediction"
                ],
                errors="coerce",
            ),
            pd.to_numeric(
                group[
                    "target_21d_excess"
                ],
                errors="coerce",
            ),
        )

        monthly_rows.append(
            {
                "model_name": (
                    model_name
                ),
                "as_of_date": (
                    as_of_date
                ),
                "sector": sector,
                "companies": len(
                    group
                ),
                "sector_ic": ic,
            }
        )

    monthly = pd.DataFrame(
        monthly_rows
    )

    if monthly.empty:
        raise ModelDiagnosticsError(
            "No sector-level observations "
            "were available."
        )

    rows = []

    for (
        model_name,
        sector,
    ), group in monthly.groupby(
        [
            "model_name",
            "sector",
        ],
        sort=True,
    ):
        valid_ic = (
            group[
                "sector_ic"
            ]
            .dropna()
            .astype(float)
        )

        rows.append(
            {
                "model_name": (
                    model_name
                ),
                "sector": sector,
                "sector_months": len(
                    group
                ),
                "valid_ic_months": len(
                    valid_ic
                ),
                "mean_sector_ic": (
                    float(
                        valid_ic.mean()
                    )
                    if not valid_ic.empty
                    else np.nan
                ),
                "median_sector_ic": (
                    float(
                        valid_ic.median()
                    )
                    if not valid_ic.empty
                    else np.nan
                ),
                "positive_ic_ratio": (
                    float(
                        valid_ic.gt(
                            0.0
                        ).mean()
                    )
                    if not valid_ic.empty
                    else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "model_name",
                "sector",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _moving_block_bootstrap_ci(
    values: np.ndarray,
    *,
    block_length: int,
    samples: int,
    random_state: int,
) -> tuple[
    float,
    float,
]:
    """Bootstrap the mean while preserving short temporal blocks."""
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) < 2:
        return (
            np.nan,
            np.nan,
        )

    block_length = min(
        block_length,
        len(values),
    )

    starts = np.arange(
        len(values)
        - block_length
        + 1
    )

    blocks_needed = ceil(
        len(values)
        / block_length
    )

    rng = np.random.default_rng(
        random_state
    )

    boot_means = np.empty(
        samples,
        dtype=float,
    )

    for sample_number in range(
        samples
    ):
        selected_starts = (
            rng.choice(
                starts,
                size=blocks_needed,
                replace=True,
            )
        )

        sampled = np.concatenate(
            [
                values[
                    start:
                    start
                    + block_length
                ]
                for start in (
                    selected_starts
                )
            ]
        )[
            : len(values)
        ]

        boot_means[
            sample_number
        ] = float(
            sampled.mean()
        )

    lower, upper = (
        np.quantile(
            boot_means,
            [
                0.025,
                0.975,
            ],
        )
    )

    return (
        float(lower),
        float(upper),
    )


def compute_pairwise_ic_comparison(
    monthly_metrics: pd.DataFrame,
    *,
    block_length: int = 3,
    bootstrap_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare every pair of models on the same OOS months."""
    _require_columns(
        monthly_metrics,
        (
            "model_name",
            "as_of_date",
            "ic",
        ),
        dataset_name="monthly model metrics",
    )

    data = monthly_metrics.copy()

    data[
        "as_of_date"
    ] = pd.to_datetime(
        data[
            "as_of_date"
        ]
    )

    pivot = data.pivot(
        index="as_of_date",
        columns="model_name",
        values="ic",
    )

    models = [
        model
        for model in pivot.columns
        if pivot[
            model
        ].notna().any()
    ]

    rows = []

    for (
        model_a,
        model_b,
    ) in combinations(
        models,
        2,
    ):
        pair = pivot[
            [
                model_a,
                model_b,
            ]
        ].dropna()

        if pair.empty:
            continue

        difference = (
            pair[
                model_a
            ]
            - pair[
                model_b
            ]
        ).to_numpy(
            dtype=float
        )

        lower, upper = (
            _moving_block_bootstrap_ci(
                difference,
                block_length=(
                    block_length
                ),
                samples=(
                    bootstrap_samples
                ),
                random_state=(
                    random_state
                ),
            )
        )

        rows.append(
            {
                "model_a": (
                    model_a
                ),
                "model_b": (
                    model_b
                ),
                "paired_months": (
                    len(
                        difference
                    )
                ),
                "mean_ic_difference": (
                    float(
                        difference.mean()
                    )
                ),
                "median_ic_difference": (
                    float(
                        np.median(
                            difference
                        )
                    )
                ),
                "model_a_win_ratio": (
                    float(
                        np.mean(
                            difference
                            > 0.0
                        )
                    )
                ),
                "model_b_win_ratio": (
                    float(
                        np.mean(
                            difference
                            < 0.0
                        )
                    )
                ),
                "tie_ratio": (
                    float(
                        np.mean(
                            difference
                            == 0.0
                        )
                    )
                ),
                "bootstrap_95_ci_lower": (
                    lower
                ),
                "bootstrap_95_ci_upper": (
                    upper
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "model_a",
                "model_b",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def compute_feature_concentration(
    linear_coefficients: pd.DataFrame,
    lightgbm_importance: pd.DataFrame,
) -> pd.DataFrame:
    """Measure dependence on a small number of predictors."""
    _require_columns(
        linear_coefficients,
        (
            "model_name",
            "feature",
            "mean_absolute_coefficient",
            "nonzero_ratio",
        ),
        dataset_name=(
            "linear coefficient summary"
        ),
    )

    _require_columns(
        lightgbm_importance,
        (
            "feature",
            "mean_gain_share",
        ),
        dataset_name=(
            "LightGBM importance summary"
        ),
    )

    rows = []

    for (
        model_name,
        group,
    ) in linear_coefficients.groupby(
        "model_name",
        sort=True,
    ):
        values = (
            pd.to_numeric(
                group[
                    "mean_absolute_coefficient"
                ],
                errors="coerce",
            )
            .fillna(
                0.0
            )
            .clip(
                lower=0.0
            )
            .sort_values(
                ascending=False
            )
            .to_numpy()
        )

        total = float(
            values.sum()
        )

        shares = (
            values
            / total
            if total > 0.0
            else np.zeros_like(
                values
            )
        )

        rows.append(
            {
                "model_name": (
                    model_name
                ),
                "features": len(
                    group
                ),
                "top1_share": float(
                    shares[
                        :1
                    ].sum()
                ),
                "top5_share": float(
                    shares[
                        :5
                    ].sum()
                ),
                "top10_share": float(
                    shares[
                        :10
                    ].sum()
                ),
                "mean_nonzero_ratio": (
                    float(
                        group[
                            "nonzero_ratio"
                        ].mean()
                    )
                ),
            }
        )

    lightgbm_values = (
        pd.to_numeric(
            lightgbm_importance[
                "mean_gain_share"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
        .sort_values(
            ascending=False
        )
        .to_numpy()
    )

    total_gain_share = float(
        lightgbm_values.sum()
    )

    lightgbm_shares = (
        lightgbm_values
        / total_gain_share
        if total_gain_share > 0.0
        else np.zeros_like(
            lightgbm_values
        )
    )

    rows.append(
        {
            "model_name": (
                "lightgbm_regressor"
            ),
            "features": len(
                lightgbm_importance
            ),
            "top1_share": float(
                lightgbm_shares[
                    :1
                ].sum()
            ),
            "top5_share": float(
                lightgbm_shares[
                    :5
                ].sum()
            ),
            "top10_share": float(
                lightgbm_shares[
                    :10
                ].sum()
            ),
            "mean_nonzero_ratio": (
                np.nan
            ),
        }
    )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "model_name"
        )
        .reset_index(
            drop=True
        )
    )


def build_model_scorecard(
    model_summary: pd.DataFrame,
    yearly_stability: pd.DataFrame,
    sector_stability: pd.DataFrame,
) -> pd.DataFrame:
    """Combine overall, temporal and sector-level diagnostics."""
    _require_columns(
        model_summary,
        (
            "model_name",
            "mean_ic",
            "annualized_ic_ir",
            "positive_ic_ratio",
            "mean_top_bottom_spread",
            "mean_top_quintile_precision",
            "mean_top_quintile_turnover",
        ),
        dataset_name="model summary",
    )

    yearly_rows = []

    for (
        model_name,
        group,
    ) in yearly_stability.groupby(
        "model_name",
        sort=True,
    ):
        valid = group.dropna(
            subset=[
                "mean_ic"
            ]
        )

        yearly_rows.append(
            {
                "model_name": (
                    model_name
                ),
                "valid_years": len(
                    valid
                ),
                "positive_mean_ic_years": (
                    int(
                        valid[
                            "mean_ic"
                        ].gt(
                            0.0
                        ).sum()
                    )
                ),
                "worst_year_mean_ic": (
                    float(
                        valid[
                            "mean_ic"
                        ].min()
                    )
                    if not valid.empty
                    else np.nan
                ),
                "best_year_mean_ic": (
                    float(
                        valid[
                            "mean_ic"
                        ].max()
                    )
                    if not valid.empty
                    else np.nan
                ),
            }
        )

    sector_rows = []

    for (
        model_name,
        group,
    ) in sector_stability.groupby(
        "model_name",
        sort=True,
    ):
        valid = group.dropna(
            subset=[
                "mean_sector_ic"
            ]
        )

        sector_rows.append(
            {
                "model_name": (
                    model_name
                ),
                "valid_sectors": len(
                    valid
                ),
                "positive_mean_ic_sectors": (
                    int(
                        valid[
                            "mean_sector_ic"
                        ].gt(
                            0.0
                        ).sum()
                    )
                ),
                "worst_sector_mean_ic": (
                    float(
                        valid[
                            "mean_sector_ic"
                        ].min()
                    )
                    if not valid.empty
                    else np.nan
                ),
                "best_sector_mean_ic": (
                    float(
                        valid[
                            "mean_sector_ic"
                        ].max()
                    )
                    if not valid.empty
                    else np.nan
                ),
            }
        )

    scorecard = (
        model_summary.merge(
            pd.DataFrame(
                yearly_rows
            ),
            on="model_name",
            how="left",
        )
        .merge(
            pd.DataFrame(
                sector_rows
            ),
            on="model_name",
            how="left",
        )
    )

    return scorecard.sort_values(
        "mean_ic",
        ascending=False,
        na_position="last",
    ).reset_index(
        drop=True
    )