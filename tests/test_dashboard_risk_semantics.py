from __future__ import annotations

import pandas as pd

from quant_equity.reporting.dashboard_metrics import current_security_risk


def test_security_risk_respects_canonical_active_position_count() -> None:
    weights = pd.DataFrame(
        {
            "as_of_date": ["2026-05-29"] * 3,
            "ticker": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Health", "Energy"],
            "method": ["selected"] * 3,
            "weight": [0.60, 0.399999, 0.000001],
        }
    )
    risk = pd.DataFrame(
        {
            "as_of_date": ["2026-05-29"] * 3,
            "ticker": ["AAA", "BBB", "CCC"],
            "annualized_volatility": [0.30, 0.20, 0.50],
            "annualized_downside_volatility": [0.22, 0.15, 0.40],
            "beta_vs_spy": [1.20, 0.80, 2.00],
            "correlation_vs_spy": [0.70, 0.50, 0.90],
            "average_dollar_volume": [100.0, 200.0, 50.0],
        }
    )

    snapshot = current_security_risk(
        weights,
        risk,
        "selected",
        pd.Timestamp("2026-05-29"),
        active_positions=2,
    )

    assert set(snapshot["ticker"]) == {"AAA", "BBB"}
    assert "CCC" not in snapshot["ticker"].tolist()
