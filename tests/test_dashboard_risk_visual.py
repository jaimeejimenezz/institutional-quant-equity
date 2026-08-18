from __future__ import annotations

import ast
from pathlib import Path


def test_risk_view_contains_small_value_formatters() -> None:
    path = Path("app/views/risk.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "_format_small_percent" in function_names
    assert "_format_weight" in function_names


def test_risk_map_limits_persistent_security_labels() -> None:
    path = Path("app/components/charts.py")
    source = path.read_text(encoding="utf-8")
    assert "label_count = min(6, len(ordered))" in source
    assert 'ordered["display_label"] = ""' in source
