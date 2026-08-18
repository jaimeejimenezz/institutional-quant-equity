from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype, is_string_dtype

ROOT = Path(__file__).resolve().parents[1]

SOURCES: dict[str, tuple[str, ...]] = {
    "overview": (
        "data/processed/backtest_all_methods_net_daily.parquet",
        "data/processed/backtest_all_methods_gross_daily.parquet",
        "data/processed/benchmark_spy_daily.parquet",
        "reports/tables/all_methods_gross_net_comparison.csv",
    ),
    "stock_ranking": (
        "data/processed/final_alpha_signal.parquet",
        "data/processed/risk_estimates.parquet",
        "data/processed/modeling_panel.parquet",
    ),
    "portfolio": (
        "data/processed/target_weights_all_methods.parquet",
        "data/processed/positions_all_methods_net.parquet",
        "reports/tables/all_method_portfolio_diagnostics.csv",
        "reports/tables/all_method_risk_summary.csv",
    ),
    "model_research": (
        "data/processed/predictions_oos_model_comparison.parquet",
        "reports/tables/model_comparison_summary.csv",
        "reports/tables/model_comparison_monthly_metrics.csv",
        "reports/tables/model_yearly_stability.csv",
        "reports/tables/model_sector_stability.csv",
        "reports/tables/lightgbm_ranker_feature_importance_summary.csv",
        "reports/tables/ensemble_candidate_summary.csv",
        "reports/tables/ensemble_signal_correlations.csv",
    ),
    "risk": (
        "data/processed/risk_estimates.parquet",
        "reports/tables/all_method_risk_summary.csv",
        "reports/tables/covariance_diagnostics.csv",
        "reports/tables/reference_portfolio_risk_contributions.csv",
    ),
    "execution": (
        "data/processed/trades_all_methods_net.parquet",
        "reports/tables/all_methods_execution_summary.csv",
        "reports/tables/all_methods_execution_cost_components.csv",
        "reports/tables/transaction_cost_sensitivity.csv",
        "reports/tables/capacity_analysis.csv",
    ),
    "robustness": (
        "reports/tables/robustness_evaluation_check_inventory.csv",
        "reports/tables/robustness_evaluation_coverage.csv",
        "reports/tables/bootstrap_strategy_summary.csv",
        "reports/tables/bootstrap_pairwise_comparison.csv",
        "reports/tables/robustness_final_signal_bootstrap.csv",
        "reports/tables/robustness_final_signal_yearly.csv",
        "reports/tables/robustness_final_signal_sector.csv",
        "reports/tables/feature_family_ablation/official_predictive_comparison.csv",
        "reports/tables/feature_family_ablation/economic_comparison.csv",
        "reports/tables/robustness_portfolio_construction_ablation.csv",
        "reports/tables/robustness_prediction_horizon_summary.csv",
        "reports/tables/robustness_rebalance_frequency.csv",
        "reports/tables/robustness_rolling_window_summary.csv",
        "reports/tables/robustness_universe_exclusion_results.csv",
        "reports/tables/robustness_regime_performance.csv",
    ),
    "data_quality": (
        "reports/tables/modeling_panel_leakage_checks.csv",
        "reports/tables/modeling_panel_readiness_checks.csv",
        "reports/tables/walk_forward_readiness_checks.csv",
        "reports/tables/risk_estimate_checks.csv",
        "reports/tables/covariance_checks.csv",
        "reports/tables/all_method_portfolio_checks.csv",
        "reports/tables/all_methods_execution_checks.csv",
        "reports/tables/robustness_evaluation_check_inventory.csv",
    ),
}

OUTPUT_TABLES = ROOT / "reports" / "tables"
OUTPUT_REPORT = ROOT / "reports" / "dashboard" / "dashboard_source_contract_review.md"
DATASET_PROFILE = OUTPUT_TABLES / "dashboard_source_profile.csv"
COLUMN_PROFILE = OUTPUT_TABLES / "dashboard_source_columns.csv"

DATE_NAME_RE = re.compile(
    r"(^|_)(date|as_of_date|rebalance_date|trade_date|timestamp|datetime|period_end|filed_date)($|_)",
    re.IGNORECASE,
)
ISO_DATE_RE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}")

CATEGORY_NAME_HINTS = {
    "method",
    "strategy",
    "model",
    "scenario",
    "portfolio",
    "sector",
    "ticker",
    "status",
    "category",
    "dimension",
    "regime",
}


@dataclass
class DatasetProfile:
    area: str
    relative_path: str
    exists: bool
    rows: int | None
    columns: int | None
    min_date: str
    max_date: str
    date_columns: str
    categorical_values: str
    error: str


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported tabular file: {path}")


def parse_date_series(series: pd.Series, column_name: str) -> pd.Series | None:
    if is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed if parsed.notna().any() else None

    if not DATE_NAME_RE.search(column_name):
        return None

    if not (is_object_dtype(series) or is_string_dtype(series)):
        return None

    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return None

    iso_like_share = non_null.str.match(ISO_DATE_RE).mean()
    if iso_like_share < 0.80:
        return None

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() < 0.80:
        return None

    return parsed


def summarize_dates(df: pd.DataFrame) -> tuple[str, str, str]:
    date_columns: list[str] = []
    minima: list[pd.Timestamp] = []
    maxima: list[pd.Timestamp] = []

    for column in df.columns:
        name = str(column)
        parsed = parse_date_series(df[column], name)
        if parsed is None:
            continue

        valid = parsed.dropna()
        if valid.empty:
            continue

        date_columns.append(name)
        minima.append(valid.min())
        maxima.append(valid.max())

    if not date_columns:
        return "", "", ""

    return (
        "|".join(date_columns),
        min(minima).isoformat(),
        max(maxima).isoformat(),
    )


def summarize_categories(df: pd.DataFrame) -> str:
    summary: dict[str, list[str]] = {}

    for column in df.columns:
        name = str(column)
        lower = name.lower()
        tokens = set(lower.split("_"))
        if not (tokens & CATEGORY_NAME_HINTS or lower in CATEGORY_NAME_HINTS):
            continue

        values = df[column].dropna().astype(str).drop_duplicates().head(25).tolist()
        if values:
            summary[name] = values

    return json.dumps(summary, ensure_ascii=False, sort_keys=True)


def sample_values(series: pd.Series, limit: int = 5) -> str:
    values = series.dropna().astype(str).drop_duplicates().head(limit).tolist()
    return json.dumps(values, ensure_ascii=False)


def inspect_dataset(
    area: str, relative_path: str
) -> tuple[DatasetProfile, list[dict[str, object]]]:
    path = ROOT / relative_path

    if not path.exists():
        return (
            DatasetProfile(
                area=area,
                relative_path=relative_path,
                exists=False,
                rows=None,
                columns=None,
                min_date="",
                max_date="",
                date_columns="",
                categorical_values="",
                error="MISSING",
            ),
            [],
        )

    try:
        df = read_table(path)
        date_columns, min_date, max_date = summarize_dates(df)

        profile = DatasetProfile(
            area=area,
            relative_path=relative_path,
            exists=True,
            rows=int(len(df)),
            columns=int(len(df.columns)),
            min_date=min_date,
            max_date=max_date,
            date_columns=date_columns,
            categorical_values=summarize_categories(df),
            error="",
        )

        column_rows: list[dict[str, object]] = []
        for column in df.columns:
            series = df[column]
            column_rows.append(
                {
                    "area": area,
                    "relative_path": relative_path,
                    "column": str(column),
                    "dtype": str(series.dtype),
                    "non_null": int(series.notna().sum()),
                    "null_pct": float(series.isna().mean()),
                    "n_unique": int(series.nunique(dropna=True)),
                    "sample_values": sample_values(series),
                }
            )

        return profile, column_rows

    except Exception as exc:
        return (
            DatasetProfile(
                area=area,
                relative_path=relative_path,
                exists=True,
                rows=None,
                columns=None,
                min_date="",
                max_date="",
                date_columns="",
                categorical_values="",
                error=f"{type(exc).__name__}: {exc}",
            ),
            [],
        )


def build_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows: list[dict[str, object]] = []
    column_rows: list[dict[str, object]] = []

    seen: set[tuple[str, str]] = set()
    for area, paths in SOURCES.items():
        for relative_path in paths:
            key = (area, relative_path)
            if key in seen:
                continue
            seen.add(key)

            profile, columns = inspect_dataset(area, relative_path)
            dataset_rows.append(asdict(profile))
            column_rows.extend(columns)

    return pd.DataFrame(dataset_rows), pd.DataFrame(column_rows)


def write_report(dataset_profile: pd.DataFrame, column_profile: pd.DataFrame) -> None:
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Dashboard source contract review",
        "",
        (
            "Focused schema review for the candidate sources that may become "
            "canonical dashboard inputs."
        ),
        "",
        f"- Candidate source usages profiled: **{len(dataset_profile)}**",
        f"- Missing sources: **{int((~dataset_profile['exists']).sum())}**",
        (
            "- Read/schema errors: **"
            f"{int(dataset_profile['error'].astype(str).str.len().gt(0).sum())}**"
        ),
        "",
    ]

    for area in SOURCES:
        lines.extend([f"## {area}", ""])
        subset = dataset_profile[dataset_profile["area"] == area]

        for row in subset.itertuples(index=False):
            if not row.exists:
                lines.append(f"- `{row.relative_path}` — **MISSING**")
                continue
            if row.error:
                lines.append(f"- `{row.relative_path}` — **ERROR:** {row.error}")
                continue

            date_text = ""
            if row.min_date or row.max_date:
                date_text = f"; dates {row.min_date or '?'} → {row.max_date or '?'}"

            lines.append(
                f"- `{row.relative_path}` — {row.rows} rows × {row.columns} cols{date_text}"
            )

            cols = column_profile[
                (column_profile["area"] == area)
                & (column_profile["relative_path"] == row.relative_path)
            ]
            if not cols.empty:
                rendered = ", ".join(
                    f"`{c.column}` ({c.dtype})" for c in cols.itertuples(index=False)
                )
                lines.append(f"  - Columns: {rendered}")

            if row.categorical_values and row.categorical_values != "{}":
                lines.append(f"  - Key values: `{row.categorical_values}`")

        lines.append("")

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    dataset_profile, column_profile = build_profiles()

    dataset_profile.to_csv(DATASET_PROFILE, index=False)
    column_profile.to_csv(COLUMN_PROFILE, index=False)
    write_report(dataset_profile, column_profile)

    missing = int((~dataset_profile["exists"]).sum())
    errors = int(dataset_profile["error"].astype(str).str.len().gt(0).sum())

    print("Institutional Quant Equity Research Platform")
    print("Dashboard source contract profiling")
    print("-" * 48)
    print(f"source_usages_profiled: {len(dataset_profile)}")
    print(f"missing_sources: {missing}")
    print(f"read_or_schema_errors: {errors}")
    print()
    print(f"Dataset profile: {DATASET_PROFILE}")
    print(f"Column profile: {COLUMN_PROFILE}")
    print(f"Review: {OUTPUT_REPORT}")

    if missing or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
