"""Tests for definitive model baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_equity.models import (
    COMPOSITE_FEATURE_DIRECTIONS,
    generate_baseline_predictions,
    score_technical_composite,
)


def make_panel() -> pd.DataFrame:
    """Create a synthetic one-date modeling panel."""
    tickers = [f"T{i:02d}" for i in range(10)]

    data: dict[
        str,
        object,
    ] = {
        "as_of_date": pd.to_datetime(["2020-01-31"] * 10),
        "ticker": tickers,
        "sector": ["Technology"] * 10,
        "target_21d_excess": np.linspace(
            -0.05,
            0.05,
            10,
        ),
        "label_top_quintile": ([0] * 8 + [1] * 2),
    }

    for index, column in enumerate(COMPOSITE_FEATURE_DIRECTIONS):
        data[column] = np.linspace(
            -1.0 + index * 0.01,
            1.0 + index * 0.01,
            10,
        )

    return pd.DataFrame(data)


def make_folds() -> pd.DataFrame:
    """Create synthetic frozen fold metadata."""
    return pd.DataFrame(
        {
            "fold_id": ["fold_0001"],
            "test_date": pd.to_datetime(["2020-01-31"]),
            "test_rows": [10],
        }
    )


def test_baselines_generate_three_predictions_per_security() -> None:
    """Every security must receive all three baseline scores."""
    predictions = generate_baseline_predictions(
        make_panel(),
        make_folds(),
    )

    assert len(predictions) == 30

    assert predictions["model_name"].nunique() == 3

    assert (
        predictions.duplicated(
            [
                "fold_id",
                "ticker",
                "model_name",
            ]
        ).sum()
        == 0
    )


def test_constant_baseline_is_zero() -> None:
    """The null baseline must contain no information."""
    predictions = generate_baseline_predictions(
        make_panel(),
        make_folds(),
    )

    constant = predictions.loc[predictions["model_name"].eq("constant")]

    assert np.allclose(
        constant["prediction"],
        0.0,
    )


def test_momentum_baseline_matches_frozen_feature() -> None:
    """Momentum prediction must equal the stored test feature."""
    panel = make_panel()

    predictions = generate_baseline_predictions(
        panel,
        make_folds(),
    )

    momentum = predictions.loc[predictions["model_name"].eq("momentum_3m")].sort_values("ticker")

    expected = panel.sort_values("ticker")[("tech__return_3m_sector_neutral")].to_numpy()

    assert np.allclose(
        momentum["prediction"].to_numpy(),
        expected,
    )


def test_composite_respects_frozen_directions() -> None:
    """Composite signs must remain frozen."""
    panel = make_panel()

    score = score_technical_composite(panel)

    manual = pd.DataFrame(
        {
            column: (panel[column] * direction)
            for (
                column,
                direction,
            ) in (COMPOSITE_FEATURE_DIRECTIONS.items())
        }
    ).mean(axis=1)

    assert np.allclose(
        score,
        manual,
    )


def test_baseline_scores_do_not_depend_on_target() -> None:
    """Changing future returns must not alter predictions."""
    original = make_panel()

    changed = original.copy()

    changed["target_21d_excess"] = np.linspace(
        100.0,
        200.0,
        len(changed),
    )

    first = generate_baseline_predictions(
        original,
        make_folds(),
    ).sort_values(
        [
            "model_name",
            "ticker",
        ]
    )

    second = generate_baseline_predictions(
        changed,
        make_folds(),
    ).sort_values(
        [
            "model_name",
            "ticker",
        ]
    )

    assert np.allclose(
        first["prediction"],
        second["prediction"],
        equal_nan=True,
    )
