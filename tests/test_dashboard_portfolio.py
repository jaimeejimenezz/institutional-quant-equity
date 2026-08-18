from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.reporting.dashboard_metrics import (
    enrich_portfolio_snapshot,
    portfolio_dates,
    portfolio_method_comparison,
    portfolio_sector_changes,
    portfolio_snapshot,
    realized_positions_for_signal,
    turnover_history,
)


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": [
                "2026-04-30",
                "2026-04-30",
                "2026-05-29",
                "2026-05-29",
                "2026-04-30",
                "2026-04-30",
                "2026-05-29",
                "2026-05-29",
            ],
            "ticker": ["AAA", "CCC", "AAA", "BBB", "AAA", "CCC", "AAA", "BBB"],
            "sector": [
                "Tech",
                "Energy",
                "Tech",
                "Health",
                "Tech",
                "Energy",
                "Tech",
                "Health",
            ],
            "method": [
                "selected",
                "selected",
                "selected",
                "selected",
                "baseline",
                "baseline",
                "baseline",
                "baseline",
            ],
            "weight": [0.50, 0.50, 0.60, 0.40, 0.50, 0.50, 0.50, 0.50],
            "previous_weight": [0.0] * 8,
            "beta_vs_spy": [1.1, 0.7, float("nan"), float("nan"), 1.1, 0.7, 1.1, 0.8],
            "average_dollar_volume": [
                100.0,
                150.0,
                float("nan"),
                float("nan"),
                100.0,
                150.0,
                100.0,
                200.0,
            ],
        }
    )


def _diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": ["2026-05-29", "2026-05-29", "2026-04-30"],
            "method": ["selected", "baseline", "selected"],
            "positions": [2, 2, 2],
            "maximum_weight": [0.60, 0.50, 0.55],
            "maximum_sector_weight": [0.60, 0.50, 0.55],
            "effective_positions": [1.9, 2.0, 1.95],
            "one_way_turnover": [0.20, 0.15, 0.18],
        }
    )


def _risk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": ["2026-05-29", "2026-05-29"],
            "method": ["selected", "baseline"],
            "predicted_volatility": [0.14, 0.15],
            "portfolio_beta_vs_spy": [0.95, 1.00],
            "maximum_liquidation_days": [0.1, 0.1],
        }
    )


def test_portfolio_snapshot_and_sector_changes() -> None:
    snapshot = portfolio_snapshot(_weights(), "selected", pd.Timestamp("2026-05-29"))
    assert snapshot["weight"].sum() == pytest.approx(1.0)
    assert snapshot["previous_weight"].sum() == pytest.approx(1.0)
    assert snapshot["weight_delta"].sum() == pytest.approx(0.0)

    exited = snapshot.loc[snapshot["ticker"] == "CCC"].iloc[0]
    assert exited["weight"] == pytest.approx(0.0)
    assert exited["previous_weight"] == pytest.approx(0.5)

    sectors = portfolio_sector_changes(snapshot)
    assert sectors["current_weight"].sum() == pytest.approx(1.0)
    assert sectors["previous_weight"].sum() == pytest.approx(1.0)


def test_portfolio_snapshot_enriches_missing_security_risk() -> None:
    snapshot = portfolio_snapshot(_weights(), "selected", pd.Timestamp("2026-05-29"))
    risk = pd.DataFrame(
        {
            "as_of_date": ["2026-05-29", "2026-05-29", "2026-05-29"],
            "ticker": ["AAA", "BBB", "CCC"],
            "beta_vs_spy": [1.05, 0.85, 0.65],
            "average_dollar_volume": [110.0, 210.0, 160.0],
        }
    )

    enriched = enrich_portfolio_snapshot(snapshot, risk)
    current = enriched.loc[enriched["weight"] > 0.0]

    assert current["beta_vs_spy"].notna().all()
    assert current["average_dollar_volume"].notna().all()


def test_portfolio_dates_and_turnover_history() -> None:
    dates = portfolio_dates(_weights(), "selected")
    assert dates == (
        pd.Timestamp("2026-04-30"),
        pd.Timestamp("2026-05-29"),
    )

    history = turnover_history(
        _diagnostics(),
        "selected",
        baseline_strategy="baseline",
    )
    assert set(history["role"]) == {"selected", "baseline"}
    assert len(history) == 3


def test_realized_positions_select_final_day_for_signal() -> None:
    positions = pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-30", "2026-06-30"],
            "strategy_name": ["selected", "selected", "selected"],
            "active_signal_date": ["2026-05-29"] * 3,
            "ticker": ["AAA", "AAA", "BBB"],
            "actual_weight": [0.55, 0.58, 0.42],
            "target_weight": [0.60, 0.60, 0.40],
            "weight_drift": [-0.05, -0.02, 0.02],
        }
    )
    snapshot = realized_positions_for_signal(
        positions,
        "selected",
        pd.Timestamp("2026-05-29"),
    )
    assert snapshot["date"].nunique() == 1
    assert pd.Timestamp(snapshot["date"].iloc[0]) == pd.Timestamp("2026-06-30")
    assert snapshot["actual_weight"].sum() == pytest.approx(1.0)


def test_method_comparison_merges_diagnostics_and_risk() -> None:
    comparison = portfolio_method_comparison(
        _diagnostics(),
        _risk(),
        "selected",
        pd.Timestamp("2026-05-29"),
    )
    assert len(comparison) == 2
    assert comparison.loc[comparison["method"] == "selected", "role"].iloc[0] == "selected"
    assert comparison["predicted_volatility"].notna().all()
