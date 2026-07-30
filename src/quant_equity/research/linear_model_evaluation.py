"""Out-of-sample evaluation for the initial linear models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class LinearModelEvaluationError(ValueError):
    """Raised when linear-model evaluation cannot be completed."""


@dataclass(frozen=True)
class LinearModelEvaluationConfig:
    """Configuration for out-of-sample model evaluation."""

    target_column: str = "target_21d_excess"
    top_label_column: str = "label_top_quintile"
    quintiles: int = 5
    top_fraction: float = 0.20
    annualization_periods: int = 12
    minimum_cross_section_size: int = 30

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> LinearModelEvaluationConfig:
        """Create evaluation configuration from YAML values."""
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
            quintiles=int(
                values.get(
                    "quintiles",
                    5,
                )
            ),
            top_fraction=float(
                values.get(
                    "top_fraction",
                    0.20,
                )
            ),
            annualization_periods=int(
                values.get(
                    "annualization_periods",
                    12,
                )
            ),
            minimum_cross_section_size=int(
                values.get(
                    "minimum_cross_section_size",
                    30,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate evaluation settings."""
        if not self.target_column:
            raise LinearModelEvaluationError("target_column cannot be empty.")

        if not self.top_label_column:
            raise LinearModelEvaluationError("top_label_column cannot be empty.")

        if self.quintiles < 2:
            raise LinearModelEvaluationError("quintiles must be at least 2.")

        if not 0.0 < self.top_fraction < 1.0:
            raise LinearModelEvaluationError("top_fraction must be in (0, 1).")

        if self.annualization_periods < 1:
            raise LinearModelEvaluationError("annualization_periods must be positive.")

        if self.minimum_cross_section_size < self.quintiles:
            raise LinearModelEvaluationError(
                "minimum_cross_section_size cannot be smaller than quintiles."
            )


@dataclass(frozen=True)
class LinearModelEvaluationOutputs:
    """Tables produced by the model evaluation."""

    ranked_predictions: pd.DataFrame
    monthly_metrics: pd.DataFrame
    model_summary: pd.DataFrame
    monthly_quintiles: pd.DataFrame
    quintile_summary: pd.DataFrame
    monthly_turnover: pd.DataFrame
    turnover_summary: pd.DataFrame
    yearly_summary: pd.DataFrame
    coefficient_summary: pd.DataFrame


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Require columns to exist in a dataframe."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise LinearModelEvaluationError(
            f"{dataset_name} is missing columns: " + ", ".join(missing) + "."
        )


def _normalize_date_columns(
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
            raise LinearModelEvaluationError(f"{dataset_name} contains invalid dates in {column}.")


def _safe_spearman(
    first: pd.Series,
    second: pd.Series,
) -> float:
    """Calculate Spearman correlation after removing invalid rows."""
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

    return float(
        comparable["first"].rank(method="average").corr(comparable["second"].rank(method="average"))
    )


def _mean_or_nan(
    values: pd.Series,
) -> float:
    """Return a mean or NaN for an empty series."""
    valid = values.dropna().astype(float)

    if valid.empty:
        return float("nan")

    return float(valid.mean())


def _median_or_nan(
    values: pd.Series,
) -> float:
    """Return a median or NaN for an empty series."""
    valid = values.dropna().astype(float)

    if valid.empty:
        return float("nan")

    return float(valid.median())


def _standard_deviation_or_nan(
    values: pd.Series,
) -> float:
    """Return a sample standard deviation or NaN."""
    valid = values.dropna().astype(float)

    if len(valid) < 2:
        return float("nan")

    return float(valid.std(ddof=1))


def _positive_ratio(
    values: pd.Series,
) -> float:
    """Return the proportion of valid values greater than zero."""
    valid = values.dropna().astype(float)

    if valid.empty:
        return float("nan")

    return float(valid.gt(0.0).mean())


def _validate_predictions(
    predictions: pd.DataFrame,
    *,
    config: LinearModelEvaluationConfig,
) -> pd.DataFrame:
    """Validate and normalize out-of-sample predictions."""
    required_columns = (
        "fold_id",
        "as_of_date",
        "ticker",
        "sector",
        "model_name",
        "prediction",
        config.target_column,
        config.top_label_column,
        "latest_fit_target_end_date",
    )

    _require_columns(
        predictions,
        required_columns,
        dataset_name="Linear model predictions",
    )

    data = predictions.copy()

    _normalize_date_columns(
        data,
        (
            "as_of_date",
            "latest_fit_target_end_date",
        ),
        dataset_name="Linear model predictions",
    )

    for column in (
        "prediction",
        config.target_column,
        config.top_label_column,
    ):
        original = data[column]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        invalid_values = original.notna() & converted.isna()

        if invalid_values.any():
            raise LinearModelEvaluationError(
                f"Linear model predictions contain non-numeric values in {column}."
            )

        if converted.isna().any():
            raise LinearModelEvaluationError(
                f"Linear model predictions contain missing values in {column}."
            )

        if np.isinf(converted.to_numpy(dtype=float)).any():
            raise LinearModelEvaluationError(
                f"Linear model predictions contain infinite values in {column}."
            )

        data[column] = converted.astype(float)

    duplicated_rows = int(
        data.duplicated(
            [
                "as_of_date",
                "ticker",
                "model_name",
            ]
        ).sum()
    )

    if duplicated_rows:
        raise LinearModelEvaluationError(
            "Linear model predictions contain duplicated date-ticker-model rows."
        )

    if (data["latest_fit_target_end_date"] > data["as_of_date"]).any():
        raise LinearModelEvaluationError(
            "At least one prediction uses a target that was not known on its prediction date."
        )

    invalid_labels = ~data[config.top_label_column].isin(
        [
            0.0,
            1.0,
        ]
    )

    if invalid_labels.any():
        raise LinearModelEvaluationError("The top-quintile label must contain only 0 and 1.")

    cross_section_sizes = data.groupby(
        [
            "model_name",
            "as_of_date",
        ]
    )["ticker"].nunique()

    undersized = cross_section_sizes[cross_section_sizes.lt(config.minimum_cross_section_size)]

    if not undersized.empty:
        raise LinearModelEvaluationError(
            "At least one model-date cross-section is smaller than minimum_cross_section_size."
        )

    return data


def prepare_prediction_rankings(
    predictions: pd.DataFrame,
    *,
    config: LinearModelEvaluationConfig,
) -> pd.DataFrame:
    """Create deterministic monthly ranks, quintiles and top selections."""
    data = _validate_predictions(
        predictions,
        config=config,
    )

    frames: list[pd.DataFrame] = []

    grouped = data.groupby(
        [
            "model_name",
            "as_of_date",
        ],
        sort=True,
    )

    for (
        _model_name,
        _as_of_date,
    ), month_data in grouped:
        month = month_data.sort_values("ticker").copy()

        observations = len(month)

        valid_ranking = month["prediction"].nunique() >= 2

        month["valid_ranking"] = valid_ranking

        if not valid_ranking:
            month["evaluation_rank"] = np.nan
            month["evaluation_percentile"] = np.nan
            month["evaluation_quintile"] = pd.Series(
                pd.NA,
                index=month.index,
                dtype="Int64",
            )
            month["predicted_top_quintile"] = pd.Series(
                False,
                index=month.index,
                dtype="boolean",
            )

            frames.append(month)

            continue

        ascending_rank = month["prediction"].rank(
            method="first",
            ascending=True,
        )

        descending_rank = month["prediction"].rank(
            method="first",
            ascending=False,
        )

        percentile = ascending_rank / observations

        quintile = np.ceil(ascending_rank * config.quintiles / observations).astype(int)

        quintile = quintile.clip(
            1,
            config.quintiles,
        )

        top_count = max(
            1,
            int(np.ceil(observations * config.top_fraction)),
        )

        month["evaluation_rank"] = descending_rank.astype(float)

        month["evaluation_percentile"] = percentile.astype(float)

        month["evaluation_quintile"] = pd.Series(
            quintile,
            index=month.index,
            dtype="Int64",
        )

        month["predicted_top_quintile"] = pd.Series(
            descending_rank.le(top_count),
            index=month.index,
            dtype="boolean",
        )

        frames.append(month)

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "as_of_date",
                "model_name",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )


def calculate_monthly_model_metrics(
    ranked_predictions: pd.DataFrame,
    *,
    config: LinearModelEvaluationConfig,
) -> pd.DataFrame:
    """Calculate predictive metrics separately for every model and month."""
    rows: list[dict[str, Any]] = []

    grouped = ranked_predictions.groupby(
        [
            "model_name",
            "as_of_date",
        ],
        sort=True,
    )

    for (
        model_name,
        as_of_date,
    ), month in grouped:
        target = month[config.target_column].astype(float)

        prediction = month["prediction"].astype(float)

        errors = prediction - target

        rmse = float(np.sqrt(np.mean(np.square(errors.to_numpy(dtype=float)))))

        mae = float(np.mean(np.abs(errors.to_numpy(dtype=float))))

        valid_ranking = bool(month["valid_ranking"].iloc[0])

        information_coefficient = float("nan")

        top_mean_target = float("nan")
        bottom_mean_target = float("nan")
        top_bottom_spread = float("nan")
        top_quintile_precision = float("nan")
        quintile_monotonicity = float("nan")
        predicted_top_count = 0

        if valid_ranking:
            information_coefficient = _safe_spearman(
                prediction,
                target,
            )

            top_mask = month["predicted_top_quintile"].astype(bool)

            bottom_mask = month["evaluation_quintile"].eq(1)

            predicted_top_count = int(top_mask.sum())

            top_mean_target = float(target.loc[top_mask].mean())

            bottom_mean_target = float(target.loc[bottom_mask].mean())

            top_bottom_spread = top_mean_target - bottom_mean_target

            top_quintile_precision = float(
                month.loc[
                    top_mask,
                    config.top_label_column,
                ].mean()
            )

            quintile_means = (
                month.groupby(
                    "evaluation_quintile",
                    sort=True,
                )[config.target_column]
                .mean()
                .dropna()
            )

            if len(quintile_means) == config.quintiles:
                quintile_monotonicity = _safe_spearman(
                    pd.Series(
                        quintile_means.index,
                        dtype=float,
                    ),
                    quintile_means.reset_index(drop=True),
                )

        rows.append(
            {
                "model_name": model_name,
                "as_of_date": as_of_date,
                "observations": len(month),
                "valid_ranking": valid_ranking,
                "unique_predictions": int(prediction.nunique()),
                "prediction_mean": float(prediction.mean()),
                "prediction_std": float(prediction.std(ddof=0)),
                "target_mean": float(target.mean()),
                "rmse": rmse,
                "mae": mae,
                "information_coefficient": (information_coefficient),
                "top_mean_target": (top_mean_target),
                "bottom_mean_target": (bottom_mean_target),
                "top_bottom_spread": (top_bottom_spread),
                "top_quintile_precision": (top_quintile_precision),
                "quintile_monotonicity": (quintile_monotonicity),
                "predicted_top_count": (predicted_top_count),
                "actual_top_count": int(month[config.top_label_column].sum()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "as_of_date",
                "model_name",
            ]
        )
        .reset_index(drop=True)
    )


def calculate_monthly_quintile_returns(
    ranked_predictions: pd.DataFrame,
    *,
    config: LinearModelEvaluationConfig,
) -> pd.DataFrame:
    """Calculate future excess returns for prediction quintiles."""
    valid = ranked_predictions.loc[ranked_predictions["valid_ranking"]].copy()

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "as_of_date",
                "quintile",
                "observations",
                "mean_target",
                "median_target",
                "positive_target_ratio",
            ]
        )

    rows: list[dict[str, Any]] = []

    grouped = valid.groupby(
        [
            "model_name",
            "as_of_date",
            "evaluation_quintile",
        ],
        sort=True,
        observed=True,
    )

    for (
        model_name,
        as_of_date,
        quintile,
    ), group in grouped:
        target = group[config.target_column].astype(float)

        rows.append(
            {
                "model_name": model_name,
                "as_of_date": as_of_date,
                "quintile": int(quintile),
                "observations": len(group),
                "mean_target": float(target.mean()),
                "median_target": float(target.median()),
                "positive_target_ratio": float(target.gt(0.0).mean()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "model_name",
                "as_of_date",
                "quintile",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_quintile_returns(
    monthly_quintiles: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize prediction-quintile returns through time."""
    if monthly_quintiles.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "quintile",
                "months",
                "mean_target",
                "median_month_target",
                "std_month_target",
                "positive_month_ratio",
            ]
        )

    rows: list[dict[str, Any]] = []

    grouped = monthly_quintiles.groupby(
        [
            "model_name",
            "quintile",
        ],
        sort=True,
    )

    for (
        model_name,
        quintile,
    ), group in grouped:
        values = group["mean_target"]

        rows.append(
            {
                "model_name": model_name,
                "quintile": int(quintile),
                "months": len(group),
                "mean_target": _mean_or_nan(values),
                "median_month_target": (_median_or_nan(values)),
                "std_month_target": (_standard_deviation_or_nan(values)),
                "positive_month_ratio": (_positive_ratio(values)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "model_name",
                "quintile",
            ]
        )
        .reset_index(drop=True)
    )


def calculate_monthly_ranking_turnover(
    ranked_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate top-selection and full-ranking turnover."""
    rows: list[dict[str, Any]] = []

    for model_name, model_data in ranked_predictions.groupby(
        "model_name",
        sort=True,
    ):
        dates = sorted(model_data["as_of_date"].unique())

        date_groups = {
            date: model_data.loc[model_data["as_of_date"].eq(date)].copy() for date in dates
        }

        for previous_date, current_date in zip(
            dates[:-1],
            dates[1:],
            strict=True,
        ):
            previous = date_groups[previous_date]

            current = date_groups[current_date]

            valid_transition = bool(
                previous["valid_ranking"].iloc[0] and current["valid_ranking"].iloc[0]
            )

            top_turnover = float("nan")
            top_overlap_ratio = float("nan")
            mean_absolute_percentile_change = float("nan")

            if valid_transition:
                previous_top = set(
                    previous.loc[
                        previous["predicted_top_quintile"].astype(bool),
                        "ticker",
                    ]
                )

                current_top = set(
                    current.loc[
                        current["predicted_top_quintile"].astype(bool),
                        "ticker",
                    ]
                )

                all_top_tickers = previous_top.union(current_top)

                previous_weight = 1.0 / len(previous_top)

                current_weight = 1.0 / len(current_top)

                top_turnover = 0.5 * sum(
                    abs(
                        (previous_weight if ticker in previous_top else 0.0)
                        - (current_weight if ticker in current_top else 0.0)
                    )
                    for ticker in all_top_tickers
                )

                top_overlap_ratio = float(
                    len(previous_top.intersection(current_top))
                    / max(
                        len(current_top),
                        1,
                    )
                )

                rank_comparison = (
                    previous.loc[
                        :,
                        [
                            "ticker",
                            "evaluation_percentile",
                        ],
                    ]
                    .rename(columns={"evaluation_percentile": ("previous_percentile")})
                    .merge(
                        current.loc[
                            :,
                            [
                                "ticker",
                                "evaluation_percentile",
                            ],
                        ].rename(columns={"evaluation_percentile": ("current_percentile")}),
                        on="ticker",
                        how="inner",
                        validate="one_to_one",
                    )
                )

                mean_absolute_percentile_change = float(
                    (rank_comparison["current_percentile"] - rank_comparison["previous_percentile"])
                    .abs()
                    .mean()
                )

            rows.append(
                {
                    "model_name": model_name,
                    "from_date": previous_date,
                    "to_date": current_date,
                    "valid_transition": (valid_transition),
                    "top_quintile_turnover": (top_turnover),
                    "top_quintile_overlap_ratio": (top_overlap_ratio),
                    "mean_absolute_percentile_change": (mean_absolute_percentile_change),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "to_date",
                "model_name",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_ranking_turnover(
    monthly_turnover: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize turnover by model."""
    rows: list[dict[str, Any]] = []

    for model_name, group in monthly_turnover.groupby(
        "model_name",
        sort=True,
    ):
        valid = group.loc[group["valid_transition"]]

        rows.append(
            {
                "model_name": model_name,
                "transitions": len(group),
                "valid_turnover_transitions": (len(valid)),
                "mean_top_quintile_turnover": (_mean_or_nan(valid["top_quintile_turnover"])),
                "median_top_quintile_turnover": (_median_or_nan(valid["top_quintile_turnover"])),
                "maximum_top_quintile_turnover": (
                    float(valid["top_quintile_turnover"].max()) if not valid.empty else float("nan")
                ),
                "mean_top_quintile_overlap": (_mean_or_nan(valid["top_quintile_overlap_ratio"])),
                "mean_absolute_percentile_change": (
                    _mean_or_nan(valid["mean_absolute_percentile_change"])
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_model_metrics(
    monthly_metrics: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    *,
    config: LinearModelEvaluationConfig,
) -> pd.DataFrame:
    """Summarize monthly predictive performance by model."""
    rows: list[dict[str, Any]] = []

    for model_name, group in monthly_metrics.groupby(
        "model_name",
        sort=True,
    ):
        ic_values = group["information_coefficient"].dropna()

        ic_std = float(ic_values.std(ddof=1)) if len(ic_values) > 1 else float("nan")

        mean_ic = float(ic_values.mean()) if not ic_values.empty else float("nan")

        annualized_ic_ir = (
            mean_ic / ic_std * np.sqrt(config.annualization_periods)
            if (np.isfinite(mean_ic) and np.isfinite(ic_std) and ic_std > 0.0)
            else float("nan")
        )

        ic_t_stat = (
            mean_ic / (ic_std / np.sqrt(len(ic_values)))
            if (len(ic_values) > 1 and np.isfinite(ic_std) and ic_std > 0.0)
            else float("nan")
        )

        rows.append(
            {
                "model_name": model_name,
                "months": len(group),
                "ranking_months": int(group["valid_ranking"].sum()),
                "valid_ic_months": len(ic_values),
                "mean_ic": mean_ic,
                "median_ic": _median_or_nan(ic_values),
                "std_ic": ic_std,
                "annualized_ic_ir": (annualized_ic_ir),
                "ic_t_stat": ic_t_stat,
                "positive_ic_ratio": (_positive_ratio(ic_values)),
                "mean_top_target": (_mean_or_nan(group["top_mean_target"])),
                "mean_bottom_target": (_mean_or_nan(group["bottom_mean_target"])),
                "mean_top_bottom_spread": (_mean_or_nan(group["top_bottom_spread"])),
                "positive_spread_ratio": (_positive_ratio(group["top_bottom_spread"])),
                "mean_top_quintile_precision": (_mean_or_nan(group["top_quintile_precision"])),
                "mean_quintile_monotonicity": (_mean_or_nan(group["quintile_monotonicity"])),
                "positive_monotonicity_ratio": (_positive_ratio(group["quintile_monotonicity"])),
                "mean_rmse": float(group["rmse"].mean()),
                "mean_mae": float(group["mae"].mean()),
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.merge(
        turnover_summary,
        on="model_name",
        how="left",
        validate="one_to_one",
    )

    return summary.sort_values(
        [
            "mean_ic",
            "mean_top_bottom_spread",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)


def summarize_metrics_by_year(
    monthly_metrics: pd.DataFrame,
    monthly_turnover: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize predictive metrics by calendar year."""
    metrics = monthly_metrics.copy()

    metrics["year"] = metrics["as_of_date"].dt.year

    turnover = monthly_turnover.copy()

    turnover["year"] = turnover["to_date"].dt.year

    turnover_yearly = (
        turnover.loc[turnover["valid_transition"]]
        .groupby(
            [
                "model_name",
                "year",
            ],
            as_index=False,
        )
        .agg(
            mean_top_quintile_turnover=(
                "top_quintile_turnover",
                "mean",
            ),
            mean_absolute_percentile_change=(
                "mean_absolute_percentile_change",
                "mean",
            ),
        )
    )

    rows: list[dict[str, Any]] = []

    grouped = metrics.groupby(
        [
            "model_name",
            "year",
        ],
        sort=True,
    )

    for (
        model_name,
        year,
    ), group in grouped:
        rows.append(
            {
                "model_name": model_name,
                "year": int(year),
                "months": len(group),
                "ranking_months": int(group["valid_ranking"].sum()),
                "mean_ic": _mean_or_nan(group["information_coefficient"]),
                "positive_ic_ratio": (_positive_ratio(group["information_coefficient"])),
                "mean_top_bottom_spread": (_mean_or_nan(group["top_bottom_spread"])),
                "positive_spread_ratio": (_positive_ratio(group["top_bottom_spread"])),
                "mean_top_quintile_precision": (_mean_or_nan(group["top_quintile_precision"])),
                "mean_rmse": float(group["rmse"].mean()),
            }
        )

    yearly = pd.DataFrame(rows)

    return (
        yearly.merge(
            turnover_yearly,
            on=[
                "model_name",
                "year",
            ],
            how="left",
        )
        .sort_values(
            [
                "year",
                "model_name",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_coefficient_stability(
    coefficients: pd.DataFrame,
    *,
    feature_directions: Mapping[str, float],
) -> pd.DataFrame:
    """Summarize coefficient magnitude, signs and economic direction."""
    required_columns = (
        "fold_id",
        "test_date",
        "model_name",
        "feature",
        "coefficient_standardized",
        "nonzero_coefficient",
    )

    _require_columns(
        coefficients,
        required_columns,
        dataset_name="Linear model coefficients",
    )

    data = coefficients.copy()

    _normalize_date_columns(
        data,
        ("test_date",),
        dataset_name="Linear model coefficients",
    )

    data["coefficient_standardized"] = pd.to_numeric(
        data["coefficient_standardized"],
        errors="coerce",
    )

    if data["coefficient_standardized"].isna().any():
        raise LinearModelEvaluationError(
            "Coefficient table contains invalid standardized coefficients."
        )

    if not pd.api.types.is_bool_dtype(data["nonzero_coefficient"]):
        mapped_nonzero = (
            data["nonzero_coefficient"]
            .astype("string")
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
        )

        if mapped_nonzero.isna().any():
            raise LinearModelEvaluationError(
                "Coefficient table contains invalid nonzero_coefficient values."
            )

        data["nonzero_coefficient"] = mapped_nonzero.astype(bool)

    unknown_features = sorted(set(data["feature"]).difference(feature_directions))

    if unknown_features:
        raise LinearModelEvaluationError(
            "Coefficient table contains features "
            "without economic directions: " + ", ".join(unknown_features)
        )

    invalid_directions = {
        feature: direction
        for feature, direction in (feature_directions.items())
        if direction
        not in {
            -1.0,
            1.0,
        }
    }

    if invalid_directions:
        raise LinearModelEvaluationError("Feature directions must be -1.0 or 1.0.")

    data["expected_direction"] = data["feature"].map(feature_directions)

    data["directional_coefficient"] = data["coefficient_standardized"] * data["expected_direction"]

    rows: list[dict[str, Any]] = []

    grouped = data.groupby(
        [
            "model_name",
            "feature",
        ],
        sort=True,
    )

    for (
        model_name,
        feature,
    ), group in grouped:
        coefficients_values = group["coefficient_standardized"].astype(float)

        nonzero_group = group.loc[group["nonzero_coefficient"]]

        nonzero_coefficients = nonzero_group["coefficient_standardized"].astype(float)

        positive_ratio = (
            float(nonzero_coefficients.gt(0.0).mean()) if not nonzero_group.empty else float("nan")
        )

        negative_ratio = (
            float(nonzero_coefficients.lt(0.0).mean()) if not nonzero_group.empty else float("nan")
        )

        sign_consistency = (
            max(
                positive_ratio,
                negative_ratio,
            )
            if (np.isfinite(positive_ratio) and np.isfinite(negative_ratio))
            else float("nan")
        )

        economic_direction_ratio = (
            float(nonzero_group["directional_coefficient"].gt(0.0).mean())
            if not nonzero_group.empty
            else float("nan")
        )

        rows.append(
            {
                "model_name": model_name,
                "feature": feature,
                "folds": len(group),
                "expected_direction": float(group["expected_direction"].iloc[0]),
                "mean_coefficient": float(coefficients_values.mean()),
                "median_coefficient": float(coefficients_values.median()),
                "std_coefficient": (
                    float(coefficients_values.std(ddof=1)) if len(group) > 1 else float("nan")
                ),
                "mean_absolute_coefficient": float(coefficients_values.abs().mean()),
                "nonzero_ratio": float(group["nonzero_coefficient"].mean()),
                "positive_sign_ratio_nonzero": (positive_ratio),
                "negative_sign_ratio_nonzero": (negative_ratio),
                "sign_consistency_ratio": (sign_consistency),
                "economic_direction_ratio": (economic_direction_ratio),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "model_name",
                "mean_absolute_coefficient",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def evaluate_linear_model_predictions(
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    feature_directions: Mapping[str, float],
    config: LinearModelEvaluationConfig,
) -> LinearModelEvaluationOutputs:
    """Run the complete Step 7C out-of-sample evaluation."""
    ranked_predictions = prepare_prediction_rankings(
        predictions,
        config=config,
    )

    monthly_metrics = calculate_monthly_model_metrics(
        ranked_predictions,
        config=config,
    )

    monthly_quintiles = calculate_monthly_quintile_returns(
        ranked_predictions,
        config=config,
    )

    quintile_summary = summarize_quintile_returns(monthly_quintiles)

    monthly_turnover = calculate_monthly_ranking_turnover(ranked_predictions)

    turnover_summary = summarize_ranking_turnover(monthly_turnover)

    model_summary = summarize_model_metrics(
        monthly_metrics,
        turnover_summary,
        config=config,
    )

    yearly_summary = summarize_metrics_by_year(
        monthly_metrics,
        monthly_turnover,
    )

    coefficient_summary = summarize_coefficient_stability(
        coefficients,
        feature_directions=(feature_directions),
    )

    return LinearModelEvaluationOutputs(
        ranked_predictions=(ranked_predictions),
        monthly_metrics=monthly_metrics,
        model_summary=model_summary,
        monthly_quintiles=(monthly_quintiles),
        quintile_summary=(quintile_summary),
        monthly_turnover=(monthly_turnover),
        turnover_summary=(turnover_summary),
        yearly_summary=yearly_summary,
        coefficient_summary=(coefficient_summary),
    )
