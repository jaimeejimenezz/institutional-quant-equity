from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.reporting.dashboard_metrics import (
    covariance_history,
    covariance_snapshot,
    current_security_risk,
    reference_risk_contribution_snapshot,
    risk_dates,
    risk_history,
    risk_method_comparison,
    risk_summary_row,
)


def _portfolio_risk() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in ("2026-04-30", "2026-05-29"):
        rows.extend(
            [
                {
                    "as_of_date": date,
                    "method": "selected",
                    "positions": 2,
                    "portfolio_value": 1_000_000.0,
                    "predicted_volatility": 0.12,
                    "predicted_variance": 0.0144,
                    "portfolio_beta_vs_spy": 0.95,
                    "maximum_weight": 0.60,
                    "minimum_weight": 0.40,
                    "concentration_hhi": 0.52,
                    "effective_positions": 1.92,
                    "maximum_sector_weight": 0.60,
                    "maximum_active_sector_weight": 0.10,
                    "maximum_position_adv_fraction": 0.002,
                    "weighted_position_adv_fraction": 0.001,
                    "maximum_liquidation_days": 0.03,
                    "weighted_liquidation_days": 0.02,
                    "risk_contribution_sum": 1.0,
                },
                {
                    "as_of_date": date,
                    "method": "top_n_equal_weight",
                    "positions": 2,
                    "portfolio_value": 1_000_000.0,
                    "predicted_volatility": 0.13,
                    "predicted_variance": 0.0169,
                    "portfolio_beta_vs_spy": 1.02,
                    "maximum_weight": 0.50,
                    "minimum_weight": 0.50,
                    "concentration_hhi": 0.50,
                    "effective_positions": 2.0,
                    "maximum_sector_weight": 0.50,
                    "maximum_active_sector_weight": 0.08,
                    "maximum_position_adv_fraction": 0.003,
                    "weighted_position_adv_fraction": 0.002,
                    "maximum_liquidation_days": 0.04,
                    "weighted_liquidation_days": 0.03,
                    "risk_contribution_sum": 1.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": ["2026-05-29", "2026-05-29", "2026-05-29"],
            "ticker": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Health", "Energy"],
            "method": ["selected", "selected", "selected"],
            "weight": [0.60, 0.40, 0.0],
        }
    )


def _security_risk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": ["2026-05-29", "2026-05-29", "2026-05-29"],
            "ticker": ["AAA", "BBB", "CCC"],
            "annualized_volatility": [0.30, 0.20, 0.25],
            "annualized_downside_volatility": [0.22, 0.15, 0.18],
            "beta_vs_spy": [1.20, 0.80, 1.00],
            "correlation_vs_spy": [0.70, 0.50, 0.60],
            "average_dollar_volume": [100_000_000.0, 200_000_000.0, 80_000_000.0],
        }
    )


def _covariance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": ["2026-04-30", "2026-05-29"],
            "shrinkage": [0.15, 0.20],
            "shrinkage_condition_number": [12.0, 10.0],
            "mean_pairwise_correlation": [0.30, 0.35],
            "maximum_pairwise_correlation": [0.80, 0.85],
        }
    )


def _reference_contributions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": ["2026-05-29", "2026-05-29"],
            "ticker": ["AAA", "BBB"],
            "sector": ["Tech", "Health"],
            "weight": [0.50, 0.50],
            "annualized_volatility": [0.30, 0.20],
            "beta_vs_spy": [1.20, 0.80],
            "marginal_risk": [0.10, 0.08],
            "component_risk": [0.06, 0.04],
            "risk_contribution_share": [0.60, 0.40],
            "average_dollar_volume": [100_000_000.0, 200_000_000.0],
            "position_adv_fraction": [0.005, 0.0025],
            "liquidation_days": [0.05, 0.03],
        }
    )


def test_risk_dates_and_selected_snapshot() -> None:
    dates = risk_dates(_portfolio_risk(), "selected")
    assert dates == (
        pd.Timestamp("2026-04-30"),
        pd.Timestamp("2026-05-29"),
    )

    row = risk_summary_row(
        _portfolio_risk(),
        "selected",
        pd.Timestamp("2026-05-29"),
    )
    assert float(row["predicted_volatility"]) == pytest.approx(0.12)


def test_risk_history_keeps_selected_and_baseline_roles() -> None:
    history = risk_history(_portfolio_risk(), "selected")
    assert set(history["role"]) == {"selected", "baseline"}
    assert set(history["method"]) == {"selected", "top_n_equal_weight"}


def test_current_security_risk_uses_positive_holdings_only() -> None:
    snapshot = current_security_risk(
        _weights(),
        _security_risk(),
        "selected",
        pd.Timestamp("2026-05-29"),
    )
    assert snapshot["ticker"].tolist() == ["AAA", "BBB"]
    assert snapshot["weight"].sum() == pytest.approx(1.0)
    assert snapshot["annualized_volatility"].notna().all()


def test_covariance_snapshot_and_history() -> None:
    row = covariance_snapshot(_covariance(), pd.Timestamp("2026-05-29"))
    assert float(row["shrinkage"]) == pytest.approx(0.20)

    history = covariance_history(_covariance())
    assert history["as_of_date"].is_monotonic_increasing


def test_reference_contributions_are_date_scoped_without_method_assignment() -> None:
    snapshot = reference_risk_contribution_snapshot(
        _reference_contributions(),
        pd.Timestamp("2026-05-29"),
    )
    assert "method" not in snapshot.columns
    assert snapshot["risk_contribution_share"].sum() == pytest.approx(1.0)


def test_risk_method_comparison_marks_selected_method() -> None:
    comparison = risk_method_comparison(
        _portfolio_risk(),
        "selected",
        pd.Timestamp("2026-05-29"),
    )
    selected = comparison.loc[comparison["role"] == "selected"]
    assert selected["method"].tolist() == ["selected"]
