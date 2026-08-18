from __future__ import annotations

from quant_equity.reporting.dashboard_catalog import (
    DASHBOARD_SOURCES,
    STRATEGY_ORDER,
    source_path,
)
from quant_equity.reporting.dashboard_data import read_dashboard_source


def test_canonical_dashboard_sources_exist() -> None:
    missing = [source_id for source_id in DASHBOARD_SOURCES if not source_path(source_id).exists()]
    assert not missing, f"Missing dashboard sources: {missing}"


def test_canonical_dashboard_source_schemas() -> None:
    violations: dict[str, list[str]] = {}

    for source_id, source in DASHBOARD_SOURCES.items():
        frame = read_dashboard_source(source_id)
        missing_columns = sorted(set(source.required_columns) - set(frame.columns))
        if missing_columns:
            violations[source_id] = missing_columns

    assert not violations, f"Dashboard schema violations: {violations}"


def test_strategy_contract_has_five_methods() -> None:
    assert len(STRATEGY_ORDER) == 5
    assert len(set(STRATEGY_ORDER)) == 5
