"""Tests for the definitive walk-forward report."""

from __future__ import annotations

import pandas as pd

from quant_equity.validation import (
    WalkForwardConfig,
    WalkForwardReadinessResult,
    write_walk_forward_readiness_report,
)


def make_result(
    *,
    ready: bool,
) -> WalkForwardReadinessResult:
    """Create a synthetic readiness result."""
    checks = pd.DataFrame(
        {
            "check": [
                "temporal_order",
            ],
            "status": [("PASS" if ready else "FAIL")],
            "violations": [(0 if ready else 1)],
            "description": ["Synthetic temporal check."],
        }
    )

    return WalkForwardReadinessResult(
        is_ready=ready,
        summary={
            "folds": 77,
            "candidate_features": 91,
        },
        checks=checks,
        issues=(() if ready else ("temporal_order: 1 violation(s).",)),
    )


def test_ready_report_is_written(
    tmp_path,
) -> None:
    """A successful audit must produce an approval report."""
    path = tmp_path / "walk_forward_report.md"

    result = make_result(ready=True)

    config = WalkForwardConfig(
        min_train_dates=60,
        validation_dates=12,
        mode="expanding",
    )

    written = write_walk_forward_readiness_report(
        result,
        config,
        path,
    )

    content = written.read_text(encoding="utf-8")

    assert written == path

    assert "READY FOR MODEL TRAINING" in content

    assert "training data only" in content

    assert "77" in content


def test_failed_report_is_not_approved(
    tmp_path,
) -> None:
    """A failed audit must never be reported as ready."""
    path = tmp_path / "walk_forward_report.md"

    result = make_result(ready=False)

    config = WalkForwardConfig(
        min_train_dates=60,
        validation_dates=12,
        mode="expanding",
    )

    write_walk_forward_readiness_report(
        result,
        config,
        path,
    )

    content = path.read_text(encoding="utf-8")

    assert "NOT READY" in content

    assert "temporal_order: 1 violation(s)." in content
