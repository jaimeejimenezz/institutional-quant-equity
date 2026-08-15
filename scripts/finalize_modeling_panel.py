"""Finalize and approve the Step 11 master modeling panel."""

from __future__ import annotations

import logging

import pandas as pd

from quant_equity.config import (
    PROJECT_ROOT,
    REPORTS_DIR,
)
from quant_equity.logging_config import (
    configure_logging,
)
from quant_equity.validation import (
    ModelingPanelReadinessError,
    audit_modeling_panel_readiness,
    write_modeling_panel_readiness_report,
)

PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_panel.parquet"

DICTIONARY_PATH = REPORTS_DIR / "tables" / "modeling_panel_data_dictionary.csv"

LEAKAGE_CHECKS_PATH = REPORTS_DIR / "tables" / "modeling_panel_leakage_checks.csv"

FINAL_CHECKS_PATH = REPORTS_DIR / "tables" / "modeling_panel_readiness_checks.csv"

FINAL_REPORT_PATH = REPORTS_DIR / "data_quality" / "modeling_panel_final_audit.md"


def main() -> None:
    """Execute Step 11D."""
    configure_logging()

    logger = logging.getLogger("quant_equity")

    for path in (
        PANEL_PATH,
        DICTIONARY_PATH,
        LEAKAGE_CHECKS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    panel = pd.read_parquet(PANEL_PATH)

    dictionary = pd.read_csv(DICTIONARY_PATH)

    leakage_checks = pd.read_csv(LEAKAGE_CHECKS_PATH)

    result = audit_modeling_panel_readiness(
        panel,
        dictionary,
        leakage_checks,
    )

    FINAL_CHECKS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.checks.to_csv(
        FINAL_CHECKS_PATH,
        index=False,
    )

    write_modeling_panel_readiness_report(
        result,
        FINAL_REPORT_PATH,
    )

    logger.info("Modeling panel final audit completed.")

    print()
    print("Institutional Quant Equity Research Platform")

    print("Modeling panel final audit - Step 11D")

    print("------------------------------------------------")

    for key, value in result.summary.items():
        print(f"{key}: {value}")

    print()

    print("Final readiness checks:")

    print(result.checks.to_string(index=False))

    print()

    print(f"Checks table: {FINAL_CHECKS_PATH}")

    print(f"Final report: {FINAL_REPORT_PATH}")

    print()

    if not result.is_ready:
        print("Modeling panel readiness: FAILED")

        for issue in result.issues:
            print(f"- {issue}")

        raise ModelingPanelReadinessError("Step 11 dataset contract failed.")

    print("Modeling panel readiness: OK")

    print()


if __name__ == "__main__":
    main()
