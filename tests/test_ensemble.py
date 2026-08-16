"""Tests for stable cross-sectional model ensembles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_equity.models.ensemble import (
    EnsembleConfig,
    _validation_weights_from_ic,
    build_component_scores,
    build_final_alpha_signal,
)


def make_predictions() -> pd.DataFrame:
    """Create aligned synthetic OOS model predictions."""
    rows = []

    models = (
        "technical_equal_weight_composite",
        "elastic_net",
        "lightgbm_ranker",
    )

    for date_number, date in enumerate(
        pd.to_datetime(
            [
                "2020-01-31",
                "2020-02-28",
            ]
        )
    ):
        for ticker_number in range(5):
            for model_number, model in enumerate(models):
                rows.append(
                    {
                        "fold_id": (f"fold_{date_number + 1:04d}"),
                        "as_of_date": date,
                        "ticker": (f"T{ticker_number}"),
                        "sector": ("Technology" if ticker_number < 3 else "Financials"),
                        "model_name": model,
                        "prediction": (ticker_number + 0.1 * model_number),
                        "target_21d_excess": (0.01 * ticker_number),
                        "label_top_quintile": int(ticker_number == 4),
                    }
                )

    return pd.DataFrame(rows)


def make_weights() -> pd.DataFrame:
    """Create synthetic fold-specific ensemble weights."""
    return pd.DataFrame(
        {
            "fold_id": [
                "fold_0001",
                "fold_0002",
            ],
            "composite_weight": [
                0.4,
                0.4,
            ],
            "elastic_net_weight": [
                0.4,
                0.4,
            ],
            "lightgbm_ranker_weight": [
                0.2,
                0.2,
            ],
        }
    )


def test_validation_weights_are_positive_and_sum_to_one() -> None:
    """Validation weights must remain conservative and normalized."""
    weights = _validation_weights_from_ic(
        np.array(
            [
                0.10,
                0.05,
                -0.10,
            ]
        ),
        equal_weight_prior=0.5,
    )

    assert np.all(weights > 0.0)

    assert np.isclose(
        weights.sum(),
        1.0,
    )


def test_all_negative_validation_ic_falls_back_to_equal_weights() -> None:
    """No positive validation evidence must imply equal weights."""
    weights = _validation_weights_from_ic(
        np.array(
            [
                -0.10,
                -0.05,
                -0.01,
            ]
        ),
        equal_weight_prior=0.5,
    )

    assert np.allclose(
        weights,
        np.array(
            [
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ]
        ),
    )


def test_component_scores_are_bounded_between_zero_and_one() -> None:
    """Every monthly component score must become a percentile."""
    scores = build_component_scores(
        make_predictions(),
        config=EnsembleConfig(
            expected_cross_section_size=5,
            minimum_validation_dates=1,
        ),
    )

    for column in (
        "composite_percentile",
        "elastic_net_percentile",
        "lightgbm_ranker_percentile",
    ):
        assert (
            scores[column]
            .between(
                0.0,
                1.0,
            )
            .all()
        )


def test_final_signal_has_one_row_per_security_and_date() -> None:
    """The final signal must preserve the complete cross-section."""
    scores = build_component_scores(
        make_predictions(),
        config=EnsembleConfig(
            expected_cross_section_size=5,
            minimum_validation_dates=1,
        ),
    )

    final_signal = build_final_alpha_signal(
        scores,
        make_weights(),
    )

    assert len(final_signal) == 10

    assert (
        final_signal.duplicated(
            [
                "as_of_date",
                "ticker",
            ]
        ).sum()
        == 0
    )


def test_model_contributions_sum_to_final_raw_score() -> None:
    """Stored model contributions must exactly reconstruct the ensemble."""
    scores = build_component_scores(
        make_predictions(),
        config=EnsembleConfig(
            expected_cross_section_size=5,
            minimum_validation_dates=1,
        ),
    )

    final_signal = build_final_alpha_signal(
        scores,
        make_weights(),
    )

    reconstructed = final_signal[
        [
            "composite_contribution",
            "elastic_net_contribution",
            "lightgbm_ranker_contribution",
        ]
    ].sum(axis=1)

    assert np.allclose(
        reconstructed,
        final_signal["raw_prediction"],
    )


def test_future_targets_do_not_change_final_alpha_signal() -> None:
    """Changing OOS outcomes must not alter ensemble scores."""
    first_data = make_predictions()

    second_data = first_data.copy()

    security_dates = (
        second_data.loc[
            :,
            [
                "as_of_date",
                "ticker",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    security_dates["modified_target"] = np.linspace(
        100.0,
        1000.0,
        len(security_dates),
    )

    second_data = (
        second_data.drop(columns=["target_21d_excess"])
        .merge(
            security_dates,
            on=[
                "as_of_date",
                "ticker",
            ],
            how="left",
            validate="many_to_one",
        )
        .rename(
            columns={
                "modified_target": ("target_21d_excess"),
            }
        )
    )

    config = EnsembleConfig(
        expected_cross_section_size=5,
        minimum_validation_dates=1,
    )

    first_scores = build_component_scores(
        first_data,
        config=config,
    )

    second_scores = build_component_scores(
        second_data,
        config=config,
    )

    first_signal = build_final_alpha_signal(
        first_scores,
        make_weights(),
    )

    second_signal = build_final_alpha_signal(
        second_scores,
        make_weights(),
    )

    first_signal = first_signal.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)

    second_signal = second_signal.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)

    assert np.allclose(
        first_signal["raw_prediction"],
        second_signal["raw_prediction"],
    )

    assert np.allclose(
        first_signal["percentile_score"],
        second_signal["percentile_score"],
    )

    assert np.array_equal(
        first_signal["rank"].to_numpy(),
        second_signal["rank"].to_numpy(),
    )
