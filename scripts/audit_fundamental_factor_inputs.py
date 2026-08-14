"""Audit fundamental inputs before building raw fundamental factors."""

from __future__ import annotations

import pandas as pd

from quant_equity.config import PROJECT_ROOT

PIT_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_snapshots_pit.parquet"

TTM_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_ttm_pit.parquet"

BASE_PATH = PROJECT_ROOT / "data" / "interim" / "fundamental_base_monthly.parquet"


def _normalize_dates(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize common date columns when present."""
    result = data.copy()

    for column in (
        "as_of_date",
        "available_date",
        "filed_date",
        "end_date",
    ):
        if column in result.columns:
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce",
            ).dt.normalize()

    return result


def _print_share_audit(
    pit: pd.DataFrame,
    base: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> None:
    """Audit available share-count measures."""
    print()
    print("SHARE COUNT AUDIT")
    print("-" * 48)

    latest_base = base.loc[base["as_of_date"].eq(latest_date)].copy()

    outstanding_available = int(latest_base["shares_outstanding"].notna().sum())

    total_companies = int(latest_base["ticker"].nunique())

    print(f"shares_outstanding in current 10A base: {outstanding_available}/{total_companies}")

    diluted = pit.loc[
        pit["canonical_metric"].eq("diluted_shares") & pit["as_of_date"].eq(latest_date)
    ].copy()

    if diluted.empty:
        print("diluted_shares at latest date: NOT FOUND")
        return

    diluted["value"] = pd.to_numeric(
        diluted["value"],
        errors="coerce",
    )

    positive = diluted.loc[diluted["value"].gt(0.0)].copy()

    company_count = int(positive["ticker"].nunique())

    print(f"diluted_shares positive coverage: {company_count}/{total_companies}")

    print(f"diluted_shares rows at latest date: {len(diluted)}")

    if "statement_type" in diluted.columns:
        print()
        print("Statement types:")
        print(diluted["statement_type"].value_counts(dropna=False).to_string())

    if "duration_class" in diluted.columns:
        print()
        print("Duration classes:")
        print(diluted["duration_class"].value_counts(dropna=False).to_string())

    if "unit" in diluted.columns:
        print()
        print("Units:")
        print(diluted["unit"].value_counts(dropna=False).to_string())

    if positive.empty:
        return

    best = positive.sort_values(
        [
            "ticker",
            "end_date",
            "available_date",
        ],
        na_position="first",
    ).drop_duplicates(
        "ticker",
        keep="last",
    )

    print()
    print("Example latest diluted share values:")

    example_columns = [
        "ticker",
        "value",
    ]

    for column in (
        "end_date",
        "available_date",
        "unit",
    ):
        if column in best.columns:
            example_columns.append(column)

    print(best[example_columns].head(15).to_string(index=False))


def _print_capex_audit(
    ttm: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> None:
    """Audit the sign convention of CAPEX."""
    print()
    print("CAPEX SIGN AUDIT")
    print("-" * 48)

    capex = ttm.loc[ttm["as_of_date"].eq(latest_date) & ttm["canonical_metric"].eq("capex")].copy()

    if capex.empty:
        print("No CAPEX TTM observations found.")
        return

    capex["ttm_value"] = pd.to_numeric(
        capex["ttm_value"],
        errors="coerce",
    )

    values = capex["ttm_value"].dropna()

    positive_count = int(values.gt(0.0).sum())

    negative_count = int(values.lt(0.0).sum())

    zero_count = int(values.eq(0.0).sum())

    print(f"Companies with CAPEX: {capex['ticker'].nunique()}")

    print(f"Positive CAPEX values: {positive_count}")

    print(f"Negative CAPEX values: {negative_count}")

    print(f"Zero CAPEX values: {zero_count}")

    if values.empty:
        return

    print()
    print("CAPEX value distribution:")

    print(
        values.describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        ).to_string()
    )

    print()
    print("Example CAPEX values:")

    print(
        capex.loc[
            :,
            [
                "ticker",
                "ttm_value",
            ],
        ]
        .sort_values("ticker")
        .head(20)
        .to_string(index=False)
    )


def _print_joint_coverage(
    base: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> None:
    """Audit inputs needed by the first factor families."""
    print()
    print("LATEST JOINT INPUT COVERAGE")
    print("-" * 48)

    latest = base.loc[base["as_of_date"].eq(latest_date)].copy()

    groups = {
        "ROE": [
            "net_income_ttm",
            "equity",
        ],
        "ROA": [
            "net_income_ttm",
            "assets",
        ],
        "Net margin": [
            "net_income_ttm",
            "revenue_ttm",
        ],
        "Operating margin": [
            "operating_income_ttm",
            "revenue_ttm",
        ],
        "Gross margin": [
            "gross_profit_ttm",
            "revenue_ttm",
        ],
        "Cash conversion": [
            "operating_cash_flow_ttm",
            "net_income_ttm",
        ],
        "Debt / Assets": [
            "assets",
        ],
        "Current ratio": [
            "current_assets",
            "current_liabilities",
        ],
        "FCF input pair": [
            "operating_cash_flow_ttm",
            "capex_ttm",
        ],
    }

    total = len(latest)

    for name, columns in groups.items():
        missing_columns = [column for column in columns if column not in latest.columns]

        if missing_columns:
            print(f"{name:<24} missing columns: {missing_columns}")
            continue

        available = int(latest[columns].notna().all(axis=1).sum())

        print(f"{name:<24} {available:>2}/{total}")


def main() -> None:
    """Run the pre-factor input audit."""
    for path in (
        PIT_PATH,
        TTM_PATH,
        BASE_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    pit = _normalize_dates(pd.read_parquet(PIT_PATH))

    ttm = _normalize_dates(pd.read_parquet(TTM_PATH))

    base = _normalize_dates(pd.read_parquet(BASE_PATH))

    latest_date = pd.Timestamp(base["as_of_date"].max())

    print()
    print("Institutional Quant Equity Research Platform")
    print("Fundamental factor input audit - Step 10B-0")
    print("------------------------------------------------")

    print(f"Latest rebalance date: {latest_date.date()}")

    print(f"Companies: {base['ticker'].nunique()}")

    _print_share_audit(
        pit,
        base,
        latest_date,
    )

    _print_capex_audit(
        ttm,
        latest_date,
    )

    _print_joint_coverage(
        base,
        latest_date,
    )

    print()
    print("Fundamental factor input audit: OK")


if __name__ == "__main__":
    main()
