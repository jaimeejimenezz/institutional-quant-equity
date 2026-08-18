from __future__ import annotations

from quant_equity.reporting.dashboard_validation import run_dashboard_validation


def test_dashboard_contract_validation_passes() -> None:
    validation = run_dashboard_validation()
    failures = validation[validation["status"] != "PASS"]

    assert failures.empty, failures.to_dict(orient="records")
