from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    ROOT / "data" / "processed",
    ROOT / "reports" / "tables",
)

OUTPUT_DIR = ROOT / "reports" / "tables"
CATALOG_CSV = OUTPUT_DIR / "dashboard_artifact_catalog.csv"
CANDIDATES_CSV = OUTPUT_DIR / "dashboard_source_candidates.csv"
SUMMARY_MD = ROOT / "reports" / "dashboard" / "dashboard_source_audit.md"

TABULAR_SUFFIXES = {".parquet", ".csv"}

DATE_HINTS = (
    "date",
    "as_of",
    "rebalance",
    "timestamp",
    "period_end",
    "filed",
)

METHOD_HINTS = (
    "method",
    "strategy",
    "model",
    "scenario",
    "portfolio",
)

CANONICAL_KEYWORDS = {
    "overview": (
        "backtest_all_methods",
        "benchmark_spy",
        "performance_summary",
        "gross_net",
    ),
    "stock_ranking": (
        "final_alpha_signal",
        "risk_estimates",
        "modeling_panel",
    ),
    "portfolio": (
        "target_weights_all",
        "positions_all",
        "portfolio_diagnostic",
        "portfolio_risk",
    ),
    "model_research": (
        "model_comparison",
        "model_yearly",
        "model_sector",
        "feature_importance",
        "ensemble",
        "factor_ic",
        "factor_quintiles",
    ),
    "risk": (
        "risk_estimates",
        "risk_summary",
        "covariance_diagnostic",
        "portfolio_risk",
    ),
    "trades_costs": (
        "trades_all",
        "execution",
        "transaction_cost",
        "capacity",
        "gross_net",
    ),
    "robustness": (
        "robustness",
        "bootstrap",
        "ablation",
    ),
    "data_quality": (
        "check",
        "readiness",
        "data_quality",
        "leakage",
        "coverage",
    ),
}


@dataclass
class ArtifactRecord:
    path: str
    relative_path: str
    suffix: str
    size_bytes: int
    sha256_12: str
    rows: int | None
    columns: int | None
    column_names: str
    date_columns: str
    min_date: str
    max_date: str
    categorical_hints: str
    candidate_pages: str
    error: str


def sha256_12(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_timestamp(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def detect_date_summary(df: pd.DataFrame) -> tuple[str, str, str]:
    detected: list[str] = []
    minima: list[pd.Timestamp] = []
    maxima: list[pd.Timestamp] = []

    for column in df.columns:
        column_name = str(column)
        lower = column_name.lower()
        if not any(hint in lower for hint in DATE_HINTS):
            continue

        parsed = pd.to_datetime(df[column], errors="coerce")
        valid = parsed.dropna()
        if valid.empty:
            continue

        detected.append(column_name)
        minima.append(valid.min())
        maxima.append(valid.max())

    if not detected:
        return "", "", ""

    return (
        "|".join(detected),
        normalize_timestamp(min(minima)),
        normalize_timestamp(max(maxima)),
    )


def categorical_summary(df: pd.DataFrame) -> str:
    summary: dict[str, list[str]] = {}
    for column in df.columns:
        column_name = str(column)
        lower = column_name.lower()
        if not any(hint in lower for hint in METHOD_HINTS):
            continue

        values = df[column].dropna().astype(str).drop_duplicates().sort_values().head(30).tolist()
        if values:
            summary[column_name] = values

    return json.dumps(summary, ensure_ascii=False, sort_keys=True)


def candidate_pages(relative_path: str) -> str:
    lower = relative_path.lower()
    pages = []
    for page, keywords in CANONICAL_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            pages.append(page)
    return "|".join(pages)


def inspect_artifact(path: Path) -> ArtifactRecord:
    relative = path.relative_to(ROOT).as_posix()
    record = ArtifactRecord(
        path=str(path),
        relative_path=relative,
        suffix=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        sha256_12=sha256_12(path),
        rows=None,
        columns=None,
        column_names="",
        date_columns="",
        min_date="",
        max_date="",
        categorical_hints="",
        candidate_pages=candidate_pages(relative),
        error="",
    )

    if path.suffix.lower() not in TABULAR_SUFFIXES:
        return record

    try:
        df = read_tabular(path)
        record.rows = int(len(df))
        record.columns = int(len(df.columns))
        record.column_names = "|".join(map(str, df.columns))
        (
            record.date_columns,
            record.min_date,
            record.max_date,
        ) = detect_date_summary(df)
        record.categorical_hints = categorical_summary(df)
    except Exception as exc:
        record.error = f"{type(exc).__name__}: {exc}"

    return record


def build_catalog() -> pd.DataFrame:
    paths: list[Path] = []
    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue
        paths.extend(
            path
            for path in scan_root.rglob("*")
            if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES
        )

    records = [asdict(inspect_artifact(path)) for path in sorted(paths)]
    return pd.DataFrame(records)


def write_summary(catalog: pd.DataFrame) -> None:
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)

    errors = catalog[catalog["error"].astype(str).str.len() > 0]
    candidates = catalog[catalog["candidate_pages"].astype(str).str.len() > 0]

    lines = [
        "# Dashboard source audit",
        "",
        "Read-only inventory of tabular artifacts available to the presentation layer.",
        "",
        f"- Tabular artifacts inspected: **{len(catalog)}**",
        f"- Artifacts with read/schema errors: **{len(errors)}**",
        f"- Candidate dashboard sources: **{len(candidates)}**",
        "",
        "## Candidate sources by dashboard area",
        "",
    ]

    for page in CANONICAL_KEYWORDS:
        subset = candidates[
            candidates["candidate_pages"]
            .astype(str)
            .str.split("|")
            .apply(lambda values, current_page=page: current_page in values)
        ]
        lines.append(f"### {page}")
        lines.append("")
        if subset.empty:
            lines.append("_No candidate source detected automatically._")
            lines.append("")
            continue

        for row in subset.itertuples(index=False):
            schema = (
                f"{row.rows} rows × {row.columns} cols"
                if pd.notna(row.rows) and pd.notna(row.columns)
                else "schema unavailable"
            )
            date_range = ""
            if row.min_date or row.max_date:
                date_range = f"; dates {row.min_date or '?'} → {row.max_date or '?'}"
            lines.append(f"- `{row.relative_path}` — {schema}{date_range}")
        lines.append("")

    if not errors.empty:
        lines.extend(
            [
                "## Read/schema errors",
                "",
            ]
        )
        for row in errors.itertuples(index=False):
            lines.append(f"- `{row.relative_path}` — {row.error}")
        lines.append("")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = build_catalog()
    catalog.to_csv(CATALOG_CSV, index=False)

    candidates = catalog[catalog["candidate_pages"].astype(str).str.len() > 0].copy()
    candidates.to_csv(CANDIDATES_CSV, index=False)

    write_summary(catalog)

    print("Institutional Quant Equity Research Platform")
    print("Dashboard source audit")
    print("-" * 48)
    print(f"tabular_artifacts: {len(catalog)}")
    print(f"read_or_schema_errors: {int((catalog['error'].astype(str).str.len() > 0).sum())}")
    print(f"candidate_sources: {len(candidates)}")
    print()
    print(f"Catalog: {CATALOG_CSV}")
    print(f"Candidates: {CANDIDATES_CSV}")
    print(f"Summary: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
