"""Tests for cross-sectional LightGBM ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_equity.models import (
    LightGBMCandidate,
    LightGBMRankingConfig,
    LightGBMRankingError,
    build_group_sizes,
    build_relevance_labels,
    train_lightgbm_ranking,
)


def make_panel() -> pd.DataFrame:
    """Create a synthetic monthly ranking panel."""
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-28",
            "2020-03-31",
            "2020-04-30",
            "2020-05-29",
            "2020-06-30",
        ]
    )

    rows = []

    for date_number, date in enumerate(dates):
        for ticker_number in range(10):
            signal = float(ticker_number)

            rows.append(
                {
                    "as_of_date": date,
                    "target_end_date": (date + pd.Timedelta(days=1)),
                    "ticker": (f"T{ticker_number:02d}"),
                    "sector": "Technology",
                    "target_21d_excess": (0.01 * signal + 0.001 * date_number),
                    "label_top_quintile": int(ticker_number >= 8),
                    "tech__signal_a": (signal),
                    "tech__signal_b": (-signal),
                    "fund__quality": (signal / 10.0),
                    "fund__quality_missing": (0.0),
                }
            )

    return pd.DataFrame(rows)


def make_folds() -> pd.DataFrame:
    """Create one walk-forward fold."""
    return pd.DataFrame(
        {
            "fold_id": ["fold_0001"],
            "test_date": pd.to_datetime(["2020-06-30"]),
            "train_start_date": pd.to_datetime(["2020-01-31"]),
            "train_end_date": pd.to_datetime(["2020-03-31"]),
            "validation_start_date": pd.to_datetime(["2020-04-30"]),
            "validation_end_date": pd.to_datetime(["2020-05-29"]),
            "test_rows": [10],
        }
    )


def make_config() -> LightGBMRankingConfig:
    """Create a fast ranking configuration."""
    candidate = LightGBMCandidate(
        candidate_name="test",
        num_leaves=4,
        max_depth=2,
        min_child_samples=5,
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        colsample_bytree=1.0,
        subsample=1.0,
    )

    return LightGBMRankingConfig(
        expected_feature_count=4,
        relevance_levels=5,
        ndcg_cutoff=2,
        lambdarank_truncation_level=3,
        max_estimators=50,
        early_stopping_rounds=5,
        minimum_validation_dates=2,
        random_state=42,
        n_jobs=1,
        candidates=(candidate,),
    )


def test_relevance_labels_create_five_ordered_levels() -> None:
    """Higher future returns must receive higher relevance."""
    month = make_panel().loc[lambda frame: frame["as_of_date"].eq(pd.Timestamp("2020-01-31"))]

    labels = build_relevance_labels(
        month,
        levels=5,
    )

    ordered = month.assign(relevance=labels).sort_values("target_21d_excess")

    assert set(labels.unique()) == {
        0,
        1,
        2,
        3,
        4,
    }

    assert ordered["relevance"].is_monotonic_increasing


def test_group_sizes_match_monthly_cross_sections() -> None:
    """Every monthly cross-section must become one query."""
    panel = make_panel().sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    )

    groups = build_group_sizes(panel)

    assert np.array_equal(
        groups,
        np.array(
            [
                10,
                10,
                10,
                10,
                10,
                10,
            ]
        ),
    )


def test_ranking_generates_one_prediction_per_test_security() -> None:
    """Every test company must receive one ranking score."""
    outputs = train_lightgbm_ranking(
        make_panel(),
        make_folds(),
        config=make_config(),
    )

    assert len(outputs.predictions) == 10

    assert set(outputs.predictions["model_name"]) == {"lightgbm_ranker"}


def test_test_targets_do_not_influence_ranking_predictions() -> None:
    """Changing test outcomes must not change OOS scores."""
    original = make_panel()

    changed = original.copy()

    mask = changed["as_of_date"].eq(pd.Timestamp("2020-06-30"))

    changed.loc[
        mask,
        "target_21d_excess",
    ] = np.linspace(
        100.0,
        1000.0,
        mask.sum(),
    )

    first = train_lightgbm_ranking(
        original,
        make_folds(),
        config=make_config(),
    )

    second = train_lightgbm_ranking(
        changed,
        make_folds(),
        config=make_config(),
    )

    assert np.allclose(
        first.predictions["prediction"],
        second.predictions["prediction"],
    )


def test_immature_training_label_is_rejected() -> None:
    """Unavailable future labels must never enter fitting."""
    panel = make_panel()

    panel.loc[
        panel["as_of_date"].eq(pd.Timestamp("2020-05-29")),
        "target_end_date",
    ] = pd.Timestamp("2020-07-15")

    try:
        train_lightgbm_ranking(
            panel,
            make_folds(),
            config=make_config(),
        )
    except LightGBMRankingError as exc:
        assert "not mature" in str(exc)
    else:
        raise AssertionError("Immature training labels were not rejected.")
