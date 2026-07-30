"""Tests for out-of-sample linear-model evaluation."""

import numpy as np
import pandas as pd
import pytest

from quant_equity.research import (
    LinearModelEvaluationConfig,
    LinearModelEvaluationError,
    evaluate_linear_model_predictions,
)

FEATURE_DIRECTIONS = {
    "feature_a": 1.0,
    "feature_b": -1.0,
}


def make_config() -> LinearModelEvaluationConfig:
    """Create compact evaluation configuration."""
    return LinearModelEvaluationConfig(
        target_column="target_21d_excess",
        top_label_column="label_top_quintile",
        quintiles=5,
        top_fraction=0.20,
        annualization_periods=12,
        minimum_cross_section_size=5,
    )


def make_predictions() -> pd.DataFrame:
    """Create three months of synthetic predictions."""
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-28",
            "2020-03-31",
        ]
    )

    tickers = tuple(f"T{position:02d}" for position in range(10))

    rows = []

    for date_number, as_of_date in enumerate(
        dates,
        start=1,
    ):
        target_end_date = as_of_date - pd.Timedelta(days=1)

        for position, ticker in enumerate(tickers):
            target = (position - 4.5) / 100.0

            actual_top = int(position >= 8)

            model_predictions = {
                "constant": 0.0,
                "perfect": target,
                "reverse": -target,
            }

            for (
                model_name,
                prediction,
            ) in model_predictions.items():
                rows.append(
                    {
                        "fold_id": (f"fold_{date_number:04d}"),
                        "as_of_date": as_of_date,
                        "ticker": ticker,
                        "sector": "Test",
                        "model_name": model_name,
                        "prediction": prediction,
                        "target_21d_excess": target,
                        "label_top_quintile": (actual_top),
                        "latest_fit_target_end_date": (target_end_date),
                    }
                )

    return pd.DataFrame(rows)


def make_coefficients() -> pd.DataFrame:
    """Create stable synthetic coefficients."""
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-28",
            "2020-03-31",
        ]
    )

    rows = []

    for date_number, test_date in enumerate(
        dates,
        start=1,
    ):
        for model_name in (
            "ridge",
            "elastic_net",
        ):
            rows.extend(
                [
                    {
                        "fold_id": (f"fold_{date_number:04d}"),
                        "test_date": test_date,
                        "model_name": model_name,
                        "feature": "feature_a",
                        "coefficient_standardized": 0.20,
                        "nonzero_coefficient": True,
                    },
                    {
                        "fold_id": (f"fold_{date_number:04d}"),
                        "test_date": test_date,
                        "model_name": model_name,
                        "feature": "feature_b",
                        "coefficient_standardized": -0.10,
                        "nonzero_coefficient": True,
                    },
                ]
            )

    return pd.DataFrame(rows)


def evaluate():
    """Run the compact evaluation fixture."""
    return evaluate_linear_model_predictions(
        make_predictions(),
        make_coefficients(),
        feature_directions=FEATURE_DIRECTIONS,
        config=make_config(),
    )


def test_constant_model_has_no_valid_ranking_metrics() -> None:
    """A constant prediction cannot rank companies."""
    outputs = evaluate()

    constant = outputs.model_summary.loc[outputs.model_summary["model_name"].eq("constant")].iloc[0]

    assert constant["ranking_months"] == 0
    assert pd.isna(constant["mean_ic"])
    assert pd.isna(constant["mean_top_bottom_spread"])


def test_perfect_model_has_perfect_ic_and_precision() -> None:
    """A perfectly ordered model should score one."""
    outputs = evaluate()

    perfect = outputs.model_summary.loc[outputs.model_summary["model_name"].eq("perfect")].iloc[0]

    assert perfect["mean_ic"] == pytest.approx(1.0)

    assert perfect["mean_top_quintile_precision"] == pytest.approx(1.0)

    assert perfect["mean_top_bottom_spread"] > 0.0


def test_reverse_model_has_negative_ic() -> None:
    """An inverse ranking should produce an IC of minus one."""
    outputs = evaluate()

    reverse = outputs.model_summary.loc[outputs.model_summary["model_name"].eq("reverse")].iloc[0]

    assert reverse["mean_ic"] == pytest.approx(-1.0)


def test_valid_models_generate_five_quintiles() -> None:
    """A valid ranking should populate every quintile."""
    outputs = evaluate()

    perfect = outputs.monthly_quintiles.loc[outputs.monthly_quintiles["model_name"].eq("perfect")]

    counts = perfect.groupby("as_of_date")["quintile"].nunique()

    assert counts.eq(5).all()


def test_stable_top_selection_has_zero_turnover() -> None:
    """An unchanged top set should have no turnover."""
    outputs = evaluate()

    perfect_turnover = outputs.monthly_turnover.loc[
        outputs.monthly_turnover["model_name"].eq("perfect")
    ]

    assert perfect_turnover["top_quintile_turnover"].eq(0.0).all()

    assert perfect_turnover["top_quintile_overlap_ratio"].eq(1.0).all()


def test_duplicate_prediction_rows_are_rejected() -> None:
    """A model cannot predict the same company twice in one month."""
    predictions = make_predictions()

    duplicated = pd.concat(
        [
            predictions,
            predictions.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        LinearModelEvaluationError,
        match="duplicated",
    ):
        evaluate_linear_model_predictions(
            duplicated,
            make_coefficients(),
            feature_directions=(FEATURE_DIRECTIONS),
            config=make_config(),
        )


def test_coefficient_summary_agrees_with_economic_direction() -> None:
    """Stable coefficient signs should match expected directions."""
    outputs = evaluate()

    assert outputs.coefficient_summary["economic_direction_ratio"].eq(1.0).all()

    assert outputs.coefficient_summary["sign_consistency_ratio"].eq(1.0).all()


def test_monthly_metrics_cover_every_model_and_date() -> None:
    """Every model-month pair must receive one metric row."""
    outputs = evaluate()

    assert len(outputs.monthly_metrics) == 9

    assert (
        outputs.monthly_metrics[
            [
                "model_name",
                "as_of_date",
            ]
        ]
        .duplicated()
        .sum()
        == 0
    )

    assert np.isfinite(outputs.monthly_metrics["rmse"].to_numpy(dtype=float)).all()
