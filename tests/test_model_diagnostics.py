"""Tests for out-of-sample model diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_equity.models import (
    build_model_scorecard,
    compute_feature_concentration,
    compute_pairwise_ic_comparison,
    compute_sector_stability,
    compute_yearly_stability,
)


def make_monthly_metrics() -> pd.DataFrame:
    """Create synthetic monthly model results."""
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-28",
            "2021-01-29",
            "2021-02-26",
        ]
    )

    rows = []

    for model, offset in (
        (
            "strong",
            0.10,
        ),
        (
            "weak",
            -0.05,
        ),
    ):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "model_name": model,
                    "as_of_date": date,
                    "ic": (offset + 0.01 * index),
                    "top_bottom_spread": (offset),
                    "top_quintile_precision": (0.30 if model == "strong" else 0.20),
                    "top_quintile_turnover": (0.25 if model == "strong" else 0.50),
                }
            )

    return pd.DataFrame(rows)


def test_yearly_stability_preserves_models_and_years() -> None:
    """Every model-year combination must be retained."""
    result = compute_yearly_stability(make_monthly_metrics())

    assert len(result) == 4

    assert set(result["year"]) == {
        2020,
        2021,
    }


def test_pairwise_comparison_detects_stronger_model() -> None:
    """A consistently stronger model must have positive IC difference."""
    result = compute_pairwise_ic_comparison(
        make_monthly_metrics(),
        block_length=2,
        bootstrap_samples=500,
        random_state=42,
    )

    row = result.iloc[0]

    if row["model_a"] == "strong":
        assert row["mean_ic_difference"] > 0.0
    else:
        assert row["mean_ic_difference"] < 0.0


def test_sector_stability_detects_perfect_ranking() -> None:
    """Perfect within-sector ranking must produce IC equal to one."""
    data = pd.DataFrame(
        {
            "model_name": ["model"] * 4,
            "as_of_date": pd.to_datetime(["2020-01-31"] * 4),
            "ticker": [
                "A",
                "B",
                "C",
                "D",
            ],
            "sector": ["Technology"] * 4,
            "prediction": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "target_21d_excess": [
                0.01,
                0.02,
                0.03,
                0.04,
            ],
        }
    )

    result = compute_sector_stability(data)

    assert np.isclose(
        result.iloc[0]["mean_sector_ic"],
        1.0,
    )


def test_feature_concentration_reports_top_shares() -> None:
    """Concentration metrics must identify dominant predictors."""
    linear = pd.DataFrame(
        {
            "model_name": ["elastic_net"] * 3,
            "feature": [
                "a",
                "b",
                "c",
            ],
            "mean_absolute_coefficient": [
                8.0,
                1.0,
                1.0,
            ],
            "nonzero_ratio": [
                1.0,
                0.5,
                0.5,
            ],
        }
    )

    lightgbm = pd.DataFrame(
        {
            "feature": [
                "a",
                "b",
                "c",
            ],
            "mean_gain_share": [
                0.6,
                0.3,
                0.1,
            ],
        }
    )

    result = compute_feature_concentration(
        linear,
        lightgbm,
    )

    elastic = result.loc[result["model_name"].eq("elastic_net")].iloc[0]

    assert np.isclose(
        elastic["top1_share"],
        0.8,
    )


def test_scorecard_combines_stability_information() -> None:
    """Scorecard must combine overall and stability diagnostics."""
    monthly = make_monthly_metrics()

    yearly = compute_yearly_stability(monthly)

    predictions = []

    for model in (
        "strong",
        "weak",
    ):
        for date in (
            "2020-01-31",
            "2020-02-28",
        ):
            for index in range(3):
                predictions.append(
                    {
                        "model_name": (model),
                        "as_of_date": (date),
                        "ticker": (f"T{index}"),
                        "sector": ("Technology"),
                        "prediction": (float(index)),
                        "target_21d_excess": (float(index)),
                    }
                )

    sectors = compute_sector_stability(pd.DataFrame(predictions))

    summary = pd.DataFrame(
        {
            "model_name": [
                "strong",
                "weak",
            ],
            "mean_ic": [
                0.10,
                -0.05,
            ],
            "annualized_ic_ir": [
                1.0,
                -0.5,
            ],
            "positive_ic_ratio": [
                0.75,
                0.25,
            ],
            "mean_top_bottom_spread": [
                0.02,
                -0.01,
            ],
            "mean_top_quintile_precision": [
                0.30,
                0.20,
            ],
            "mean_top_quintile_turnover": [
                0.25,
                0.50,
            ],
        }
    )

    scorecard = build_model_scorecard(
        summary,
        yearly,
        sectors,
    )

    assert "positive_mean_ic_years" in scorecard.columns

    assert "positive_mean_ic_sectors" in scorecard.columns

    assert len(scorecard) == 2
