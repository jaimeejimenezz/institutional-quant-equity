"""Tests for the common OOS evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_equity.models import (
    evaluate_model_predictions,
)


def test_perfect_ranking_has_ic_one() -> None:
    """A perfect ranking must produce Spearman IC equal to one."""
    data = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2020-01-31"] * 10),
            "ticker": [f"T{i:02d}" for i in range(10)],
            "model_name": ["perfect"] * 10,
            "prediction": np.arange(
                10,
                dtype=float,
            ),
            "target_21d_excess": np.arange(
                10,
                dtype=float,
            ),
            "label_top_quintile": ([0] * 8 + [1] * 2),
        }
    )

    monthly, summary = evaluate_model_predictions(data)

    assert np.isclose(
        monthly.iloc[0]["ic"],
        1.0,
    )

    assert np.isclose(
        monthly.iloc[0]["top_quintile_precision"],
        1.0,
    )

    assert np.isclose(
        summary.iloc[0]["mean_ic"],
        1.0,
    )


def test_constant_predictions_have_no_ranking_metrics() -> None:
    """A constant prediction must not create artificial rankings."""
    data = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2020-01-31"] * 10),
            "ticker": [f"T{i:02d}" for i in range(10)],
            "model_name": ["constant"] * 10,
            "prediction": [0.0] * 10,
            "target_21d_excess": np.arange(
                10,
                dtype=float,
            ),
            "label_top_quintile": ([0] * 8 + [1] * 2),
        }
    )

    monthly, summary = evaluate_model_predictions(data)

    assert np.isnan(monthly.iloc[0]["ic"])

    assert np.isnan(monthly.iloc[0]["top_bottom_spread"])

    assert np.isnan(monthly.iloc[0]["top_quintile_precision"])

    assert np.isnan(summary.iloc[0]["mean_ic"])


def test_top_quintile_turnover_uses_membership_overlap() -> None:
    """Turnover must measure changes in the predicted top group."""
    rows = []

    first_scores = {
        "A": 10.0,
        "B": 9.0,
        "C": 8.0,
        "D": 7.0,
        "E": 6.0,
        "F": 5.0,
        "G": 4.0,
        "H": 3.0,
        "I": 2.0,
        "J": 1.0,
    }

    second_scores = {
        "A": 10.0,
        "C": 9.0,
        "B": 8.0,
        "D": 7.0,
        "E": 6.0,
        "F": 5.0,
        "G": 4.0,
        "H": 3.0,
        "I": 2.0,
        "J": 1.0,
    }

    for date, scores in (
        (
            "2020-01-31",
            first_scores,
        ),
        (
            "2020-02-28",
            second_scores,
        ),
    ):
        for ticker, score in scores.items():
            rows.append(
                {
                    "as_of_date": date,
                    "ticker": ticker,
                    "model_name": "test",
                    "prediction": score,
                    "target_21d_excess": score,
                    "label_top_quintile": (
                        1
                        if ticker
                        in {
                            "A",
                            "B",
                        }
                        else 0
                    ),
                }
            )

    data = pd.DataFrame(rows)

    data["as_of_date"] = pd.to_datetime(data["as_of_date"])

    monthly, _ = evaluate_model_predictions(data)

    second_month = monthly.iloc[1]

    # Top 20% contains two securities.
    # January: A, B
    # February: A, C
    # One of two remains -> turnover = 50%.
    assert np.isclose(
        second_month["top_quintile_turnover"],
        0.5,
    )
