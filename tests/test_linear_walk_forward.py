"""Tests for the initial linear walk-forward validation."""

import pandas as pd
import pytest

from quant_equity.validation import (
    LinearModelingConfig,
    LinearModelingError,
    build_expanding_walk_forward_folds,
    build_linear_modeling_panel,
)

FEATURE_COLUMNS = (
    "feature_a",
    "feature_b",
)


def make_config() -> LinearModelingConfig:
    """Create a compact test configuration."""
    return LinearModelingConfig(
        target_column="target_21d_excess",
        top_label_column="label_top_quintile",
        out_of_sample_start_date=pd.Timestamp("2020-04-01"),
        validation_months=1,
        minimum_training_months=2,
        minimum_cross_section_size=2,
    )


def make_features() -> pd.DataFrame:
    """Create monthly technical features for two companies."""
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-28",
            "2020-03-31",
            "2020-04-30",
            "2020-05-29",
        ]
    )

    rows = []

    for date_number, as_of_date in enumerate(
        dates,
        start=1,
    ):
        rows.extend(
            [
                {
                    "as_of_date": as_of_date,
                    "ticker": "AAA",
                    "sector": "Technology",
                    "feature_a": float(date_number),
                    "feature_b": float(date_number * 2),
                },
                {
                    "as_of_date": as_of_date,
                    "ticker": "BBB",
                    "sector": "Financials",
                    "feature_a": float(-date_number),
                    "feature_b": float(-date_number * 2),
                },
            ]
        )

    return pd.DataFrame(rows)


def make_labels() -> pd.DataFrame:
    """Create temporally ordered monthly labels."""
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-28",
            "2020-03-31",
            "2020-04-30",
            "2020-05-29",
        ]
    )

    first_future_dates = pd.to_datetime(
        [
            "2020-02-03",
            "2020-03-02",
            "2020-04-01",
            "2020-05-01",
            "2020-06-01",
        ]
    )

    target_end_dates = pd.to_datetime(
        [
            "2020-02-28",
            "2020-03-31",
            "2020-04-30",
            "2020-05-29",
            "2020-06-30",
        ]
    )

    rows = []

    for (
        as_of_date,
        first_future_date,
        target_end_date,
    ) in zip(
        dates,
        first_future_dates,
        target_end_dates,
        strict=True,
    ):
        rows.extend(
            [
                {
                    "as_of_date": as_of_date,
                    "ticker": "AAA",
                    "first_future_date": (first_future_date),
                    "target_end_date": (target_end_date),
                    "horizon_sessions": 21,
                    "target_21d": 0.05,
                    "target_21d_excess": 0.03,
                    "target_rank": 1,
                    "target_percentile": 1.0,
                    "label_top_quintile": 1,
                },
                {
                    "as_of_date": as_of_date,
                    "ticker": "BBB",
                    "first_future_date": (first_future_date),
                    "target_end_date": (target_end_date),
                    "horizon_sessions": 21,
                    "target_21d": -0.01,
                    "target_21d_excess": -0.03,
                    "target_rank": 2,
                    "target_percentile": 0.5,
                    "label_top_quintile": 0,
                },
            ]
        )

    return pd.DataFrame(rows)


def build_test_panel() -> pd.DataFrame:
    """Build a valid compact modeling panel."""
    return build_linear_modeling_panel(
        make_features(),
        make_labels(),
        feature_columns=FEATURE_COLUMNS,
        config=make_config(),
    )


def test_modeling_panel_contains_only_selected_features() -> None:
    """The panel should include the explicitly selected features."""
    panel = build_test_panel()

    assert "feature_a" in panel.columns
    assert "feature_b" in panel.columns
    assert len(panel) == 10
    assert panel["as_of_date"].nunique() == 5
    assert panel["ticker"].nunique() == 2


def test_modeling_panel_marks_research_and_oos_rows() -> None:
    """Dates must be labelled according to the OOS boundary."""
    panel = build_test_panel()

    research_dates = panel.loc[
        panel["sample_period"].eq("research"),
        "as_of_date",
    ].nunique()

    out_of_sample_dates = panel.loc[
        panel["sample_period"].eq("out_of_sample"),
        "as_of_date",
    ].nunique()

    assert research_dates == 3
    assert out_of_sample_dates == 2


def test_duplicate_feature_keys_are_rejected() -> None:
    """A company cannot appear twice on the same date."""
    features = make_features()

    duplicated = pd.concat(
        [
            features,
            features.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        LinearModelingError,
        match="duplicated date-ticker",
    ):
        build_linear_modeling_panel(
            duplicated,
            make_labels(),
            feature_columns=FEATURE_COLUMNS,
            config=make_config(),
        )


def test_invalid_target_dates_are_rejected() -> None:
    """Targets must begin strictly after the feature date."""
    labels = make_labels()

    labels.loc[
        labels.index[0],
        "first_future_date",
    ] = labels.loc[
        labels.index[0],
        "as_of_date",
    ]

    with pytest.raises(
        LinearModelingError,
        match="invalid temporal ordering",
    ):
        build_linear_modeling_panel(
            make_features(),
            labels,
            feature_columns=FEATURE_COLUMNS,
            config=make_config(),
        )


def test_walk_forward_folds_keep_months_together() -> None:
    """Every test fold must contain the complete monthly cross-section."""
    panel = build_test_panel()

    folds = build_expanding_walk_forward_folds(
        panel,
        config=make_config(),
    )

    assert len(folds) == 2
    assert folds["test_rows"].eq(2).all()
    assert folds["test_cross_section_size"].eq(2).all()
    assert folds["test_dates"].eq(1).all()


def test_walk_forward_uses_only_known_labels() -> None:
    """Historical targets must be realized by the test date."""
    panel = build_test_panel()

    folds = build_expanding_walk_forward_folds(
        panel,
        config=make_config(),
    )

    assert (folds["latest_known_target_end_date"] <= folds["test_date"]).all()


def test_first_fold_has_expanding_train_validation_test_order() -> None:
    """The first fold must respect chronological ordering."""
    panel = build_test_panel()

    folds = build_expanding_walk_forward_folds(
        panel,
        config=make_config(),
    )

    first_fold = folds.iloc[0]

    assert first_fold["training_dates"] == 2
    assert first_fold["validation_dates"] == 1
    assert first_fold["train_end_date"] < first_fold["validation_start_date"]
    assert first_fold["validation_end_date"] < first_fold["test_date"]
