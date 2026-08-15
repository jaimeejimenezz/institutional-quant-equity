"""Tests for final modeling-panel readiness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_equity.models.modeling_panel import (
    MODEL_FEATURE_COLUMNS,
)
from quant_equity.reporting import (
    build_modeling_panel_data_dictionary,
)
from quant_equity.validation import (
    audit_modeling_panel_readiness,
)
from tests.test_modeling_panel_dictionary import (
    make_panel,
)


def make_leakage_checks() -> pd.DataFrame:
    """Create successful synthetic Step 11B checks."""
    return pd.DataFrame(
        {
            "check": [
                "technical_point_in_time",
                "ttm_point_in_time",
                "target_starts_after_as_of",
            ],
            "status": [
                "PASS",
                "PASS",
                "PASS",
            ],
            "violations": [
                0,
                0,
                0,
            ],
        }
    )


def make_ready_panel() -> pd.DataFrame:
    """Create a valid synthetic panel with correct targets."""
    panel = make_panel()

    modeling = panel["has_target"].eq(1)

    first_rows = panel.loc[modeling].index.tolist()

    panel.loc[
        first_rows[0],
        "target_21d",
    ] = 0.02

    panel.loc[
        first_rows[1],
        "target_21d",
    ] = 0.08

    median = 0.05

    panel.loc[
        first_rows[0],
        "target_21d_excess",
    ] = 0.02 - median

    panel.loc[
        first_rows[1],
        "target_21d_excess",
    ] = 0.08 - median

    panel.loc[
        first_rows[0],
        "label_top_quintile",
    ] = 0

    panel.loc[
        first_rows[1],
        "label_top_quintile",
    ] = 1

    return panel


def run_audit(
    panel: pd.DataFrame | None = None,
    leakage_checks: pd.DataFrame | None = None,
):
    """Run the final synthetic readiness audit."""
    working_panel = make_ready_panel() if panel is None else panel

    dictionary = build_modeling_panel_data_dictionary(working_panel)

    return audit_modeling_panel_readiness(
        working_panel,
        dictionary,
        (make_leakage_checks() if leakage_checks is None else leakage_checks),
    )


def test_valid_panel_is_ready() -> None:
    """A valid master panel should be ready."""
    result = run_audit()

    assert result.is_ready
    assert not result.issues

    assert (result.checks["status"] == "PASS").all()


def test_infinite_predictor_fails() -> None:
    """Infinite predictor values must block readiness."""
    panel = make_ready_panel()

    panel.loc[
        panel.index[0],
        MODEL_FEATURE_COLUMNS[0],
    ] = np.inf

    result = run_audit(panel=panel)

    check = result.checks.set_index("check").loc["finite_predictors"]

    assert check["violations"] == 1


def test_fully_missing_predictor_fails() -> None:
    """Completely unavailable predictors must be detected."""
    panel = make_ready_panel()

    panel[MODEL_FEATURE_COLUMNS[0]] = np.nan

    result = run_audit(panel=panel)

    check = result.checks.set_index("check").loc["no_fully_missing_predictors"]

    assert check["violations"] == 1


def test_wrong_excess_target_fails() -> None:
    """Relative target must equal return minus date median."""
    panel = make_ready_panel()

    modeling_index = panel.loc[panel["has_target"].eq(1)].index[0]

    panel.loc[
        modeling_index,
        "target_21d_excess",
    ] = 99.0

    result = run_audit(panel=panel)

    check = result.checks.set_index("check").loc["target_excess_reconstruction"]

    assert check["violations"] == 1


def test_wrong_top_quintile_ordering_fails() -> None:
    """High-return observations must own the positive label."""
    panel = make_ready_panel()

    modeling_rows = panel.loc[panel["has_target"].eq(1)].index.tolist()

    panel.loc[
        modeling_rows[0],
        "label_top_quintile",
    ] = 1

    panel.loc[
        modeling_rows[1],
        "label_top_quintile",
    ] = 0

    result = run_audit(panel=panel)

    check = result.checks.set_index("check").loc["top_quintile_ordering"]

    assert check["violations"] == 1


def test_inference_rows_inside_history_fail() -> None:
    """Inference-only rows may exist only after modeling history."""
    panel = make_ready_panel()

    last_modeling_date = panel.loc[
        panel["has_target"].eq(1),
        "as_of_date",
    ].max()

    inference_index = panel.loc[panel["has_target"].eq(0)].index[0]

    panel.loc[
        inference_index,
        "as_of_date",
    ] = last_modeling_date

    result = run_audit(panel=panel)

    check = result.checks.set_index("check").loc["inference_only_tail"]

    assert check["violations"] > 0


def test_failed_prior_leakage_check_blocks_readiness() -> None:
    """Step 11B failures must propagate into final readiness."""
    leakage = make_leakage_checks()

    leakage.loc[
        leakage.index[0],
        "status",
    ] = "FAIL"

    leakage.loc[
        leakage.index[0],
        "violations",
    ] = 1

    result = run_audit(leakage_checks=leakage)

    check = result.checks.set_index("check").loc["prior_leakage_audit"]

    assert check["violations"] == 1
