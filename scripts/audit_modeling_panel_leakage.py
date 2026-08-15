"""Audit the master modeling panel for temporal leakage."""

from __future__ import annotations

import logging

import pandas as pd

from quant_equity.config import (
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.validation import (
    ModelingPanelAuditError,
    audit_modeling_panel,
    write_modeling_panel_audit_report,
)

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_panel.parquet"

CALENDAR_PATH = PROJECT_ROOT / "data" / "processed" / "rebalance_calendar.parquet"

TTM_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_ttm_pit.parquet"

CHECKS_PATH = REPORTS_DIR / "tables" / "modeling_panel_leakage_checks.csv"

REPORT_PATH = REPORTS_DIR / "data_quality" / "modeling_panel_report.md"


def main() -> None:
    """Execute Step 11B."""
    config = load_config()

    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        PANEL_PATH,
        CALENDAR_PATH,
        TTM_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    panel = pd.read_parquet(PANEL_PATH)

    calendar = pd.read_parquet(CALENDAR_PATH)

    ttm = pd.read_parquet(TTM_PATH)

    horizon = int(config["labels"]["horizon_sessions"])

    result = audit_modeling_panel(
        panel,
        calendar,
        ttm,
        expected_horizon_sessions=(horizon),
    )

    CHECKS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.checks.to_csv(
        CHECKS_PATH,
        index=False,
    )

    write_modeling_panel_audit_report(
        result,
        REPORT_PATH,
    )

    logger.info("Modeling panel leakage audit completed.")

    print()
    print("Institutional Quant Equity Research Platform")

    print("Modeling panel leakage audit - Step 11B")

    print("------------------------------------------------")

    for key, value in result.summary.items():
        print(f"{key}: {value}")

    print()

    print("Leakage and point-in-time checks:")

    print(result.checks.to_string(index=False))

    print()

    print(f"Checks table: {CHECKS_PATH}")

    print(f"Quality report: {REPORT_PATH}")

    print()

    if not result.is_valid:
        print("Modeling panel leakage audit: FAILED")

        print()

        for issue in result.issues:
            print(f"- {issue}")

        raise ModelingPanelAuditError("Master modeling panel failed the leakage audit.")

    print("Modeling panel leakage audit: OK")


if __name__ == "__main__":
    main()
