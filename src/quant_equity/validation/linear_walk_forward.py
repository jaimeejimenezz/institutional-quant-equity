"""Technical modeling panel and initial walk-forward validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class LinearModelingError(ValueError):
    """Raised when the linear-modeling dataset is invalid."""


@dataclass(frozen=True)
class LinearModelingConfig:
    """Configuration for the initial linear-modeling stage."""

    target_column: str = "target_21d_excess"
    top_label_column: str = "label_top_quintile"
    out_of_sample_start_date: pd.Timestamp = pd.Timestamp("2020-01-01")
    validation_months: int = 12
    minimum_training_months: int = 60
    minimum_cross_section_size: int = 30

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> LinearModelingConfig:
        """Build configuration from YAML values."""
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
            out_of_sample_start_date=pd.Timestamp(
                values.get(
                    "out_of_sample_start_date",
                    "2020-01-01",
                )
            ).normalize(),
            validation_months=int(
                values.get(
                    "validation_months",
                    12,
                )
            ),
            minimum_training_months=int(
                values.get(
                    "minimum_training_months",
                    60,
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
        """Validate configuration values."""
        if not self.target_column:
            raise LinearModelingError("target_column cannot be empty.")

        if not self.top_label_column:
            raise LinearModelingError("top_label_column cannot be empty.")

        if pd.isna(self.out_of_sample_start_date):
            raise LinearModelingError("out_of_sample_start_date is invalid.")

        if self.validation_months < 1:
            raise LinearModelingError("validation_months must be positive.")

        if self.minimum_training_months < 1:
            raise LinearModelingError("minimum_training_months must be positive.")

        if self.minimum_cross_section_size < 2:
            raise LinearModelingError("minimum_cross_section_size must be at least 2.")


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Require a dataframe to contain specified columns."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise LinearModelingError(f"{dataset_name} is missing columns: " + ", ".join(missing) + ".")


def _normalize_date_column(
    data: pd.DataFrame,
    column: str,
    *,
    dataset_name: str,
) -> None:
    """Normalize and validate a date column in place."""
    data[column] = pd.to_datetime(
        data[column],
        errors="coerce",
    ).dt.normalize()

    if data[column].isna().any():
        raise LinearModelingError(f"{dataset_name} contains invalid values in {column}.")


def _require_unique_keys(
    data: pd.DataFrame,
    *,
    dataset_name: str,
) -> None:
    """Require unique as-of-date and ticker keys."""
    duplicates = int(
        data.duplicated(
            subset=[
                "as_of_date",
                "ticker",
            ],
            keep=False,
        ).sum()
    )

    if duplicates:
        raise LinearModelingError(
            f"{dataset_name} contains {duplicates} duplicated date-ticker rows."
        )


def _convert_numeric_column(
    data: pd.DataFrame,
    column: str,
    *,
    allow_missing: bool,
    dataset_name: str,
) -> None:
    """Convert a column to numeric without hiding invalid values."""
    original = data[column]

    converted = pd.to_numeric(
        original,
        errors="coerce",
    )

    invalid_non_missing = original.notna() & converted.isna()

    if invalid_non_missing.any():
        raise LinearModelingError(f"{dataset_name} contains non-numeric values in {column}.")

    if not allow_missing and converted.isna().any():
        raise LinearModelingError(f"{dataset_name} contains missing values in {column}.")

    finite_values = converted.dropna().to_numpy(dtype=float)

    if np.isinf(finite_values).any():
        raise LinearModelingError(f"{dataset_name} contains infinite values in {column}.")

    data[column] = converted.astype(float)


def build_linear_modeling_panel(
    technical_features: pd.DataFrame,
    monthly_labels: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    config: LinearModelingConfig,
) -> pd.DataFrame:
    """Join selected technical features and monthly labels."""
    if not feature_columns:
        raise LinearModelingError("At least one feature column is required.")

    if len(feature_columns) != len(set(feature_columns)):
        raise LinearModelingError("feature_columns contains duplicates.")

    feature_columns = tuple(feature_columns)

    _require_columns(
        technical_features,
        (
            "as_of_date",
            "ticker",
            "sector",
            *feature_columns,
        ),
        dataset_name="Technical features",
    )

    label_columns = (
        "as_of_date",
        "ticker",
        "first_future_date",
        "target_end_date",
        "horizon_sessions",
        "target_21d",
        "target_21d_excess",
        "target_rank",
        "target_percentile",
        "label_top_quintile",
    )

    _require_columns(
        monthly_labels,
        label_columns,
        dataset_name="Monthly labels",
    )

    if config.target_column not in label_columns:
        raise LinearModelingError(
            "Configured target_column is not part of the monthly-label schema."
        )

    if config.top_label_column not in label_columns:
        raise LinearModelingError(
            "Configured top_label_column is not part of the monthly-label schema."
        )

    features = technical_features.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "sector",
            *feature_columns,
        ],
    ].copy()

    labels = monthly_labels.loc[
        :,
        list(label_columns),
    ].copy()

    _normalize_date_column(
        features,
        "as_of_date",
        dataset_name="Technical features",
    )

    for column in (
        "as_of_date",
        "first_future_date",
        "target_end_date",
    ):
        _normalize_date_column(
            labels,
            column,
            dataset_name="Monthly labels",
        )

    features["ticker"] = features["ticker"].astype("string").str.strip().str.upper()

    labels["ticker"] = labels["ticker"].astype("string").str.strip().str.upper()

    features["sector"] = features["sector"].astype("string").str.strip()

    if features["ticker"].isna().any():
        raise LinearModelingError("Technical features contain missing tickers.")

    if labels["ticker"].isna().any():
        raise LinearModelingError("Monthly labels contain missing tickers.")

    if features["sector"].isna().any():
        raise LinearModelingError("Technical features contain missing sectors.")

    _require_unique_keys(
        features,
        dataset_name="Technical features",
    )

    _require_unique_keys(
        labels,
        dataset_name="Monthly labels",
    )

    for feature in feature_columns:
        _convert_numeric_column(
            features,
            feature,
            allow_missing=True,
            dataset_name="Technical features",
        )

    for target_column in (
        "target_21d",
        "target_21d_excess",
        "target_rank",
        "target_percentile",
        "label_top_quintile",
        "horizon_sessions",
    ):
        _convert_numeric_column(
            labels,
            target_column,
            allow_missing=False,
            dataset_name="Monthly labels",
        )

    invalid_top_labels = ~labels[config.top_label_column].isin(
        [
            0.0,
            1.0,
        ]
    )

    if invalid_top_labels.any():
        raise LinearModelingError("The top-quintile label must contain only 0 and 1.")

    invalid_temporal_rows = labels["first_future_date"].le(labels["as_of_date"]) | labels[
        "target_end_date"
    ].lt(labels["first_future_date"])

    if invalid_temporal_rows.any():
        raise LinearModelingError("Monthly labels contain invalid temporal ordering.")

    panel = features.merge(
        labels,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="inner",
        validate="one_to_one",
    )

    if panel.empty:
        raise LinearModelingError("The technical modeling panel is empty.")

    _require_unique_keys(
        panel,
        dataset_name="Technical modeling panel",
    )

    feature_dates = set(features["as_of_date"])
    label_dates = set(labels["as_of_date"])
    panel_dates = set(panel["as_of_date"])

    expected_dates = feature_dates.intersection(label_dates)

    if panel_dates != expected_dates:
        raise LinearModelingError("The modeling panel lost one or more expected monthly dates.")

    target_end_counts = panel.groupby(
        "as_of_date",
        sort=True,
    )["target_end_date"].nunique()

    if target_end_counts.gt(1).any():
        raise LinearModelingError(
            "Companies from the same as-of date have different target end dates."
        )

    cross_section_sizes = panel.groupby(
        "as_of_date",
        sort=True,
    )["ticker"].nunique()

    undersized_dates = cross_section_sizes[
        cross_section_sizes.lt(config.minimum_cross_section_size)
    ]

    if not undersized_dates.empty:
        raise LinearModelingError(
            "One or more dates contain fewer companies than minimum_cross_section_size."
        )

    panel["feature_missing_count"] = (
        panel.loc[
            :,
            list(feature_columns),
        ]
        .isna()
        .sum(axis=1)
        .astype("int16")
    )

    panel["is_complete_feature_row"] = panel["feature_missing_count"].eq(0)

    panel["sample_period"] = np.where(
        panel["as_of_date"].lt(config.out_of_sample_start_date),
        "research",
        "out_of_sample",
    )

    output_columns = [
        "as_of_date",
        "ticker",
        "sector",
        "sample_period",
        "first_future_date",
        "target_end_date",
        "horizon_sessions",
        *feature_columns,
        "target_21d",
        "target_21d_excess",
        "target_rank",
        "target_percentile",
        "label_top_quintile",
        "feature_missing_count",
        "is_complete_feature_row",
    ]

    return (
        panel.loc[
            :,
            output_columns,
        ]
        .sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )


def build_expanding_walk_forward_folds(
    modeling_panel: pd.DataFrame,
    *,
    config: LinearModelingConfig,
) -> pd.DataFrame:
    """Build monthly expanding walk-forward fold definitions."""
    _require_columns(
        modeling_panel,
        (
            "as_of_date",
            "ticker",
            "target_end_date",
        ),
        dataset_name="Technical modeling panel",
    )

    panel = modeling_panel.copy()

    for column in (
        "as_of_date",
        "target_end_date",
    ):
        _normalize_date_column(
            panel,
            column,
            dataset_name="Technical modeling panel",
        )

    _require_unique_keys(
        panel,
        dataset_name="Technical modeling panel",
    )

    date_information = panel.groupby(
        "as_of_date",
        as_index=False,
        sort=True,
    ).agg(
        target_end_date_min=(
            "target_end_date",
            "min",
        ),
        target_end_date_max=(
            "target_end_date",
            "max",
        ),
        cross_section_size=(
            "ticker",
            "nunique",
        ),
    )

    inconsistent_target_dates = date_information["target_end_date_min"].ne(
        date_information["target_end_date_max"]
    )

    if inconsistent_target_dates.any():
        raise LinearModelingError("A monthly cross-section contains multiple target end dates.")

    test_dates = (
        date_information.loc[
            date_information["as_of_date"].ge(config.out_of_sample_start_date),
            "as_of_date",
        ]
        .sort_values()
        .tolist()
    )

    if not test_dates:
        raise LinearModelingError("No out-of-sample dates are available.")

    rows: list[dict[str, Any]] = []

    required_history = config.minimum_training_months + config.validation_months

    for fold_number, test_date in enumerate(
        test_dates,
        start=1,
    ):
        known_dates = (
            date_information.loc[
                date_information["as_of_date"].lt(test_date)
                & date_information["target_end_date_max"].le(test_date)
            ]
            .sort_values("as_of_date")
            .reset_index(drop=True)
        )

        if len(known_dates) < required_history:
            raise LinearModelingError(
                "Insufficient known historical labels "
                f"before test date {test_date.date()}. "
                f"Required {required_history} dates, "
                f"found {len(known_dates)}."
            )

        validation_dates = known_dates.tail(config.validation_months)

        training_dates = known_dates.iloc[: -config.validation_months]

        if len(training_dates) < config.minimum_training_months:
            raise LinearModelingError(
                "The training period contains fewer dates than minimum_training_months."
            )

        training_date_values = set(training_dates["as_of_date"])

        validation_date_values = set(validation_dates["as_of_date"])

        if training_date_values.intersection(validation_date_values):
            raise LinearModelingError("Training and validation dates overlap.")

        if test_date in training_date_values:
            raise LinearModelingError("The test date appears in training.")

        if test_date in validation_date_values:
            raise LinearModelingError("The test date appears in validation.")

        train_mask = panel["as_of_date"].isin(training_date_values)

        validation_mask = panel["as_of_date"].isin(validation_date_values)

        test_mask = panel["as_of_date"].eq(test_date)

        if not test_mask.any():
            raise LinearModelingError(f"No rows exist for test date {test_date.date()}.")

        latest_known_target_end = known_dates["target_end_date_max"].max()

        if latest_known_target_end > test_date:
            raise LinearModelingError(
                "A fold uses a label that was not yet known on its test date."
            )

        test_information = date_information.loc[date_information["as_of_date"].eq(test_date)].iloc[
            0
        ]

        rows.append(
            {
                "fold_id": f"fold_{fold_number:04d}",
                "test_date": test_date,
                "train_start_date": training_dates["as_of_date"].min(),
                "train_end_date": training_dates["as_of_date"].max(),
                "validation_start_date": validation_dates["as_of_date"].min(),
                "validation_end_date": validation_dates["as_of_date"].max(),
                "latest_known_target_end_date": (latest_known_target_end),
                "test_target_end_date": test_information["target_end_date_max"],
                "training_dates": len(training_dates),
                "validation_dates": len(validation_dates),
                "test_dates": 1,
                "training_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "test_cross_section_size": int(test_information["cross_section_size"]),
                "split_type": "expanding",
            }
        )

    return pd.DataFrame(rows).sort_values("test_date").reset_index(drop=True)
