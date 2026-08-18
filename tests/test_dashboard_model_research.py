from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.reporting.dashboard_research import (
    ENSEMBLE_COMPONENTS,
    ensemble_component_monthly,
    ensemble_correlation_matrix,
    feature_family_summary,
    feature_importance_table,
    model_comparison_table,
    sector_stability_matrix,
    yearly_stability_matrix,
)


def test_model_comparison_preserves_canonical_order_and_components() -> None:
    summary = pd.DataFrame(
        {
            "model_name": ["elastic_net", "constant", "lightgbm_ranker"],
            "months": [77, 77, 77],
            "mean_ic": [0.03, float("nan"), 0.02],
            "annualized_ic_ir": [0.6, float("nan"), 0.4],
        }
    )
    result = model_comparison_table(summary)
    assert result["model_name"].tolist() == ["constant", "elastic_net", "lightgbm_ranker"]
    assert result.loc[result["model_name"] == "elastic_net", "ensemble_component"].item()
    assert result.loc[result["model_name"] == "lightgbm_ranker", "ensemble_component"].item()


def test_monthly_component_filter_keeps_only_frozen_component_models() -> None:
    monthly = pd.DataFrame(
        {
            "model_name": [
                "technical_equal_weight_composite",
                "elastic_net",
                "lightgbm_ranker",
                "ridge",
            ],
            "as_of_date": ["2026-01-30"] * 4,
            "ic": [0.01, 0.02, 0.03, 0.04],
        }
    )
    result = ensemble_component_monthly(monthly)
    assert set(result["model_name"]) == set(ENSEMBLE_COMPONENTS)
    assert pd.api.types.is_datetime64_any_dtype(result["as_of_date"])


def test_stability_matrices_do_not_invent_ranker_rows() -> None:
    yearly = pd.DataFrame(
        {
            "model_name": ["elastic_net", "lightgbm_regressor"],
            "year": [2025, 2025],
            "mean_ic": [0.03, 0.02],
        }
    )
    sector = pd.DataFrame(
        {
            "model_name": ["elastic_net", "lightgbm_regressor"],
            "sector": ["Technology", "Technology"],
            "mean_sector_ic": [0.04, 0.01],
        }
    )
    yearly_matrix = yearly_stability_matrix(yearly)
    sector_matrix = sector_stability_matrix(sector)
    assert "LightGBM Ranker" not in yearly_matrix.index
    assert "LightGBM Ranker" not in sector_matrix.index


def test_feature_importance_labels_and_family_shares() -> None:
    features = pd.DataFrame(
        {
            "feature": [
                "tech__momentum_6_1_sector_neutral",
                "fund__book_to_market_zscore",
            ],
            "mean_gain_share": [0.6, 0.4],
            "median_gain_share": [0.5, 0.3],
            "mean_gain": [10.0, 8.0],
            "mean_split_count": [5.0, 4.0],
            "folds_used": [70, 70],
        }
    )
    top = feature_importance_table(features, top_n=2)
    families = feature_family_summary(features).set_index("family")["gain_share"]
    assert top["family"].tolist() == ["Technical", "Fundamental"]
    assert "Momentum 6 1" in top.iloc[0]["feature_label"]
    assert float(families["Technical"]) == pytest.approx(0.6)
    assert float(families["Fundamental"]) == pytest.approx(0.4)


def test_ensemble_correlation_matrix_is_symmetric() -> None:
    correlations = pd.DataFrame(
        {
            "signal_a": ["composite", "composite", "elastic_net"],
            "signal_b": ["elastic_net", "lightgbm_ranker", "lightgbm_ranker"],
            "mean_spearman": [0.50, 0.38, 0.42],
        }
    )
    matrix = ensemble_correlation_matrix(correlations)
    assert matrix.loc["Technical Composite", "Elastic Net"] == pytest.approx(0.50)
    assert matrix.loc["Elastic Net", "Technical Composite"] == pytest.approx(0.50)
    assert matrix.loc["LightGBM Ranker", "LightGBM Ranker"] == pytest.approx(1.0)
