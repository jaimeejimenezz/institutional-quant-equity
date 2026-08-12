"""Build canonical SEC fundamental events from normalized Company Facts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from quant_equity.config import (
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
)
from quant_equity.data import (
    DurationBands,
    build_canonical_coverage,
    canonicalize_sec_facts,
    load_concept_mapping,
)
from quant_equity.logging_config import (
    configure_logging,
)

NORMALIZED_PATH = PROJECT_ROOT / "data" / "interim" / "sec_companyfacts_long.parquet"

OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_events_canonical.parquet"

COVERAGE_PATH = REPORTS_DIR / "tables" / "fundamental_canonical_coverage.csv"

CONCEPT_USAGE_PATH = REPORTS_DIR / "tables" / "fundamental_concept_usage.csv"


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write a Parquet file atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(".tmp.parquet")

    temporary_path.unlink(missing_ok=True)

    data.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)


def main() -> None:
    """Run Step 9C SEC canonicalization."""
    config = load_config()

    configure_logging()

    logger = logging.getLogger("quant_equity")

    if not NORMALIZED_PATH.exists():
        raise FileNotFoundError(f"Normalized SEC data not found: {NORMALIZED_PATH}")

    canonical_config = config["sec_fundamentals"]["canonicalization"]

    mapping_path = PROJECT_ROOT / canonical_config["mapping_path"]

    mapping = load_concept_mapping(mapping_path)

    duration_bands = DurationBands.from_mapping(canonical_config)

    normalized = pd.read_parquet(NORMALIZED_PATH)

    logger.info(
        "Loaded normalized SEC observations: %s",
        len(normalized),
    )

    canonical = canonicalize_sec_facts(
        normalized,
        mapping=mapping,
        duration_bands=(duration_bands),
    )

    coverage = build_canonical_coverage(canonical)

    concept_usage = (
        canonical.groupby(
            [
                "canonical_metric",
                "concept",
                "concept_priority",
            ],
            as_index=False,
        )
        .agg(
            observations=(
                "value",
                "size",
            ),
            companies=(
                "ticker",
                "nunique",
            ),
        )
        .sort_values(
            [
                "canonical_metric",
                "concept_priority",
            ]
        )
        .reset_index(drop=True)
    )

    _write_parquet_atomically(
        canonical,
        OUTPUT_PATH,
    )

    COVERAGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage.to_csv(
        COVERAGE_PATH,
        index=False,
    )

    concept_usage.to_csv(
        CONCEPT_USAGE_PATH,
        index=False,
    )

    mismatch_count = int((~canonical["statement_type_match"]).sum())

    amendment_count = int(canonical["is_amendment"].sum())

    print()
    print("Institutional Quant Equity Research Platform")
    print("Canonical SEC fundamentals - Step 9C")
    print("------------------------------------------------")

    print(f"Input rows: {len(normalized)}")

    print(f"Canonical rows: {len(canonical)}")

    print(f"Canonical metrics: {canonical['canonical_metric'].nunique()}")

    print(f"Companies represented: {canonical['ticker'].nunique()}")

    print(f"Underlying XBRL concepts used: {canonical['concept'].nunique()}")

    print(f"Statement-type mismatches: {mismatch_count}")

    print(f"Amendment observations: {amendment_count}")

    print()

    print("Duration classes:")

    print(canonical["duration_class"].value_counts().to_string())

    print()

    print("Metric coverage:")

    display_columns = [
        "canonical_metric",
        "observations",
        "companies",
        "concepts_used",
        "statement_type_match_rate",
    ]

    print(coverage[display_columns].to_string(index=False))

    print()

    print(f"Output: {OUTPUT_PATH}")

    print(f"Coverage: {COVERAGE_PATH}")

    print(f"Concept usage: {CONCEPT_USAGE_PATH}")

    print()

    print("Canonical SEC fundamentals: OK")


if __name__ == "__main__":
    main()
