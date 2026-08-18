from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant_equity.reporting.dashboard_metrics import (
    build_alpha_snapshot,
    ensemble_weights,
    parse_model_contributions,
    signal_dates,
)


def _signal_frame() -> pd.DataFrame:
    payload = json.dumps(
        {
            "technical_composite": {
                "contribution": 0.20,
                "percentile": 0.80,
                "weight": 0.25,
            },
            "elastic_net": {
                "contribution": 0.15,
                "percentile": 0.75,
                "weight": 0.20,
            },
            "lightgbm_ranker": {
                "contribution": 0.50,
                "percentile": 0.90,
                "weight": 0.55,
            },
        }
    )
    return pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2026-05-29", "2026-05-29"]),
            "ticker": ["AAA", "BBB"],
            "sector": ["Technology", "Financials"],
            "raw_prediction": [0.85, 0.40],
            "percentile_score": [1.0, 0.0],
            "rank": [1, 2],
            "composite_weight": [0.25, 0.25],
            "elastic_net_weight": [0.20, 0.20],
            "lightgbm_ranker_weight": [0.55, 0.55],
            "composite_contribution": [0.20, 0.10],
            "elastic_net_contribution": [0.15, 0.05],
            "lightgbm_ranker_contribution": [0.50, 0.25],
            "model_contributions": [payload, payload],
        }
    )


def _risk_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2026-05-29", "2026-05-29"]),
            "ticker": ["AAA", "BBB"],
            "annualized_volatility": [0.20, 0.25],
            "annualized_downside_volatility": [0.15, 0.18],
            "beta_vs_spy": [1.10, 0.90],
            "correlation_vs_spy": [0.70, 0.65],
            "average_dollar_volume": [1_000_000.0, 2_000_000.0],
        }
    )


def _weights_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2026-05-29"]),
            "ticker": ["AAA"],
            "method": ["score_weighted"],
            "weight": [1.0],
        }
    )


def test_signal_dates_are_sorted() -> None:
    frame = _signal_frame()
    dates = signal_dates(frame)
    assert dates == (pd.Timestamp("2026-05-29"),)


def test_alpha_snapshot_joins_risk_and_selected_weights() -> None:
    snapshot = build_alpha_snapshot(
        _signal_frame(),
        _risk_frame(),
        _weights_frame(),
        "score_weighted",
        pd.Timestamp("2026-05-29"),
    )
    assert snapshot["ticker"].tolist() == ["AAA", "BBB"]
    assert snapshot["selected_weight"].tolist() == [1.0, 0.0]
    assert np.isclose(float(snapshot.iloc[0]["beta_vs_spy"]), 1.10)


def test_ensemble_weights_sum_to_one() -> None:
    weights = ensemble_weights(_signal_frame())
    assert np.isclose(sum(weights.values()), 1.0)
    assert np.isclose(weights["LightGBM ranker"], 0.55)


def test_model_contribution_payload_is_parsed() -> None:
    parsed = parse_model_contributions(_signal_frame().iloc[0]["model_contributions"])
    assert parsed["component"].tolist() == [
        "Technical composite",
        "Elastic Net",
        "LightGBM ranker",
    ]
    assert np.isclose(parsed["contribution"].sum(), 0.85)
