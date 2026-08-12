"""Normalize cached SEC Company Facts into a long dataset."""

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
    build_companyfacts_quality_summary,
    get_raw_companyfacts_path,
    load_cached_companyfacts,
    load_universe,
    normalize_companyfacts_payload,
    validate_normalized_companyfacts,
)
from quant_equity.logging_config import (
    configure_logging,
)

INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"

NORMALIZED_PATH = INTERIM_DATA_DIR / "sec_companyfacts_long.parquet"

SUMMARY_PATH = REPORTS_DIR / "tables" / "sec_companyfacts_normalization_summary.csv"

REPORT_PATH = REPORTS_DIR / "data_quality" / "sec_companyfacts_normalization_report.md"


def _write_parquet_atomically(
    data: pd.DataFrame,
    path: Path,
) -> None:
    """Write Parquet atomically."""
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


def _write_report(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """Write a compact normalization report."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    form_counts = (
        data["form"]
        .fillna("MISSING")
        .value_counts()
        .rename_axis("form")
        .reset_index(name="observations")
    )

    taxonomy_counts = (
        data["taxonomy"].value_counts().rename_axis("taxonomy").reset_index(name="observations")
    )

    lines = [
        "# SEC Company Facts normalization report",
        "",
        "## Overview",
        "",
        f"- Companies: {data['ticker'].nunique()}",
        f"- Rows: {len(data)}",
        (
            "- Unique taxonomy/concept pairs: "
            f"{data[['taxonomy', 'concept']].drop_duplicates().shape[0]}"
        ),
        f"- Taxonomies: {data['taxonomy'].nunique()}",
        f"- Units: {data['unit'].nunique()}",
        (f"- First filed date: {data['filed_date'].min().date()}"),
        (f"- Last filed date: {data['filed_date'].max().date()}"),
        "",
        "## Taxonomies",
        "",
        taxonomy_counts.to_markdown(index=False),
        "",
        "## Filing forms",
        "",
        form_counts.to_markdown(index=False),
        "",
        "## Company summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Important note",
        "",
        (
            "This dataset preserves SEC observations as reported. "
            "Economic concept mapping, duplicate resolution, "
            "quarter/YTD interpretation and point-in-time selection "
            "are intentionally deferred to later Step 9 stages."
        ),
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run Step 9B Company Facts normalization."""
    config = load_config()

    configure_logging()

    logger = logging.getLogger("quant_equity")

    universe_version = str(config["universe"]["version"])

    universe = load_universe(universe_version)

    frames: list[pd.DataFrame] = []

    for row in universe.itertuples(index=False):
        ticker = str(row.ticker).strip().upper()

        cik = row.cik

        raw_path = get_raw_companyfacts_path(
            ticker,
            cik,
        )

        payload = load_cached_companyfacts(
            raw_path,
            expected_cik=cik,
        )

        normalized = normalize_companyfacts_payload(
            payload,
            ticker=ticker,
            source_file=raw_path.name,
        )

        frames.append(normalized)

        logger.info(
            "Normalized SEC Company Facts: %s (%s rows).",
            ticker,
            len(normalized),
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "ticker",
            "taxonomy",
            "concept",
            "unit",
            "end_date",
            "filed_date",
            "accession_number",
        ],
        na_position="last",
    ).reset_index(drop=True)

    validate_normalized_companyfacts(combined)

    summary = build_companyfacts_quality_summary(combined)

    _write_parquet_atomically(
        combined,
        NORMALIZED_PATH,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    _write_report(
        combined,
        summary,
        REPORT_PATH,
    )

    exact_duplicates = int(combined.duplicated().sum())

    print()
    print("Institutional Quant Equity Research Platform")
    print("SEC Company Facts normalization - Step 9B")
    print("------------------------------------------------")

    print(f"Companies: {combined['ticker'].nunique()}")

    print(f"Rows: {len(combined)}")

    print(f"Unique concepts: {combined[['taxonomy', 'concept']].drop_duplicates().shape[0]}")

    print(f"Taxonomies: {combined['taxonomy'].nunique()}")

    print(f"Units: {combined['unit'].nunique()}")

    print(f"Exact duplicate rows: {exact_duplicates}")

    print(f"First filed date: {combined['filed_date'].min().date()}")

    print(f"Last filed date: {combined['filed_date'].max().date()}")

    print(f"Output: {NORMALIZED_PATH}")

    print(f"Summary: {SUMMARY_PATH}")

    print(f"Report: {REPORT_PATH}")

    print()

    print("SEC Company Facts normalization: OK")


if __name__ == "__main__":
    main()
