"""Diagnostics and documentation for fundamental factors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.features.fundamental_transforms import (
    FUNDAMENTAL_FACTOR_COLUMNS,
)


class FundamentalAuditError(ValueError):
    """Raised when fundamental diagnostics cannot be completed."""


@dataclass(frozen=True)
class FundamentalAuditConfig:
    """Configuration for fundamental-factor diagnostics."""

    high_correlation_threshold: float = 0.80
    min_pair_dates: int = 24
    min_pair_observations: int = 10
    zscore_mean_tolerance: float = 1.0e-10

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> FundamentalAuditConfig:
        """Create audit configuration from project YAML."""
        config = cls(
            high_correlation_threshold=float(
                values.get(
                    "high_correlation_threshold",
                    0.80,
                )
            ),
            min_pair_dates=int(
                values.get(
                    "min_pair_dates",
                    24,
                )
            ),
            min_pair_observations=int(
                values.get(
                    "min_pair_observations",
                    10,
                )
            ),
            zscore_mean_tolerance=float(
                values.get(
                    "zscore_mean_tolerance",
                    1.0e-10,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate audit settings."""
        if not (0.0 < self.high_correlation_threshold <= 1.0):
            raise FundamentalAuditError("high_correlation_threshold must be in (0, 1].")

        if self.min_pair_dates < 1:
            raise FundamentalAuditError("min_pair_dates must be positive.")

        if self.min_pair_observations < 3:
            raise FundamentalAuditError("min_pair_observations must be at least 3.")

        if self.zscore_mean_tolerance <= 0:
            raise FundamentalAuditError("zscore_mean_tolerance must be positive.")


FUNDAMENTAL_FEATURE_METADATA: dict[
    str,
    dict[str, str],
] = {
    "earnings_yield": {
        "family": "Value",
        "formula": ("Net Income TTM / Market Cap Proxy"),
        "interpretation": ("Net accounting profit relative to the estimated equity market value."),
    },
    "sales_yield": {
        "family": "Value",
        "formula": ("Revenue TTM / Market Cap Proxy"),
        "interpretation": ("Sales generated relative to the estimated equity market value."),
    },
    "book_to_market": {
        "family": "Value",
        "formula": ("Equity / Market Cap Proxy"),
        "interpretation": (
            "Accounting book equity relative to the estimated market value of equity."
        ),
    },
    "fcf_yield": {
        "family": "Value",
        "formula": ("(Operating Cash Flow TTM - CAPEX TTM) / Market Cap Proxy"),
        "interpretation": (
            "Free cash flow generated relative to the estimated equity market value."
        ),
    },
    "roe": {
        "family": "Quality",
        "formula": ("Net Income TTM / Equity"),
        "interpretation": ("Accounting profitability generated on positive shareholder equity."),
    },
    "roa": {
        "family": "Quality",
        "formula": ("Net Income TTM / Assets"),
        "interpretation": ("Accounting profitability generated relative to total assets."),
    },
    "gross_profitability": {
        "family": "Quality",
        "formula": ("Gross Profit TTM / Assets"),
        "interpretation": ("Gross profit generated relative to the company's asset base."),
    },
    "gross_margin": {
        "family": "Quality",
        "formula": ("Gross Profit TTM / Revenue TTM"),
        "interpretation": ("Share of revenue remaining after direct cost of goods or services."),
    },
    "operating_margin": {
        "family": "Quality",
        "formula": ("Operating Income TTM / Revenue TTM"),
        "interpretation": ("Operating profit generated for each unit of revenue."),
    },
    "net_margin": {
        "family": "Quality",
        "formula": ("Net Income TTM / Revenue TTM"),
        "interpretation": ("Net accounting profit generated for each unit of revenue."),
    },
    "cash_conversion": {
        "family": "Quality",
        "formula": ("Operating Cash Flow TTM / Net Income TTM"),
        "interpretation": (
            "Relationship between operating cash generation and accounting earnings."
        ),
    },
    "debt_to_assets": {
        "family": "Leverage",
        "formula": ("(Current Debt + Non-current Debt) / Assets"),
        "interpretation": ("Financial debt relative to the total asset base."),
    },
    "net_debt_to_assets": {
        "family": "Leverage",
        "formula": ("(Total Debt - Cash) / Assets"),
        "interpretation": ("Financial debt net of cash relative to total assets."),
    },
    "current_ratio": {
        "family": "Solvency",
        "formula": ("Current Assets / Current Liabilities"),
        "interpretation": ("Short-term assets available relative to short-term obligations."),
    },
    "interest_coverage": {
        "family": "Solvency",
        "formula": ("Operating Income TTM / Interest Expense TTM"),
        "interpretation": ("Operating earnings available relative to interest expense."),
    },
    "capex_to_assets": {
        "family": "Investment",
        "formula": ("CAPEX TTM / Assets"),
        "interpretation": ("Capital expenditure intensity relative to the company's asset base."),
    },
    "accruals": {
        "family": "Accruals",
        "formula": ("(Net Income TTM - Operating Cash Flow TTM) / Assets"),
        "interpretation": (
            "Difference between accounting earnings "
            "and operating cash generation relative "
            "to assets."
        ),
    },
    "revenue_growth_yoy": {
        "family": "Growth",
        "formula": ("Revenue TTM / Revenue TTM 12M Ago - 1"),
        "interpretation": ("Year-over-year growth in trailing twelve-month revenue."),
    },
    "net_income_growth_yoy": {
        "family": "Growth",
        "formula": ("(Net Income TTM - Net Income TTM 12M Ago) / abs(Net Income TTM 12M Ago)"),
        "interpretation": (
            "Year-over-year improvement or deterioration in net income, allowing negative values."
        ),
    },
    "operating_cash_flow_growth_yoy": {
        "family": "Growth",
        "formula": (
            "(Operating Cash Flow TTM - Operating Cash "
            "Flow TTM 12M Ago) / abs(Operating Cash Flow "
            "TTM 12M Ago)"
        ),
        "interpretation": (
            "Year-over-year change in operating cash flow, allowing negative historical values."
        ),
    },
    "asset_growth_yoy": {
        "family": "Investment",
        "formula": ("Assets / Assets 12M Ago - 1"),
        "interpretation": ("Year-over-year expansion or contraction of the total asset base."),
    },
    "revenue_growth_acceleration": {
        "family": "Growth",
        "formula": ("Current Revenue Growth YoY - Revenue Growth YoY 12M Ago"),
        "interpretation": ("Change in the company's year-over-year revenue growth rate."),
    },
    "net_income_growth_acceleration": {
        "family": "Growth",
        "formula": ("Current Net Income Growth YoY - Net Income Growth YoY 12M Ago"),
        "interpretation": ("Change in the company's year-over-year net-income growth rate."),
    },
    "operating_cash_flow_growth_acceleration": {
        "family": "Growth",
        "formula": (
            "Current Operating Cash Flow Growth YoY - Operating Cash Flow Growth YoY 12M Ago"
        ),
        "interpretation": (
            "Change in the company's year-over-year operating cash-flow growth rate."
        ),
    },
}


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    """Require input columns."""
    missing = [column for column in columns if column not in data.columns]

    if missing:
        raise FundamentalAuditError(f"Fundamental feature data are missing columns: {missing}")


def validate_feature_metadata() -> None:
    """Ensure every fundamental factor has documentation."""
    expected = set(FUNDAMENTAL_FACTOR_COLUMNS)

    documented = set(FUNDAMENTAL_FEATURE_METADATA)

    missing = expected - documented
    unexpected = documented - expected

    if missing or unexpected:
        raise FundamentalAuditError(
            "Fundamental feature metadata mismatch. "
            f"Missing: {sorted(missing)}. "
            f"Unexpected: {sorted(unexpected)}."
        )


def build_fundamental_feature_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize historical and latest factor coverage."""
    required = (
        "as_of_date",
        "ticker",
        *FUNDAMENTAL_FACTOR_COLUMNS,
    )

    _require_columns(
        data,
        required,
    )

    frame = data.copy()

    frame["as_of_date"] = pd.to_datetime(
        frame["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    latest_date = frame["as_of_date"].max()

    latest = frame.loc[frame["as_of_date"].eq(latest_date)]

    rows = []

    for factor in FUNDAMENTAL_FACTOR_COLUMNS:
        raw = pd.to_numeric(
            frame[factor],
            errors="coerce",
        )

        global_column = f"{factor}_zscore"

        sector_column = f"{factor}_sector_zscore"

        _require_columns(
            frame,
            (
                global_column,
                sector_column,
            ),
        )

        overall_coverage = float(raw.notna().mean())

        latest_coverage = float(
            pd.to_numeric(
                latest[factor],
                errors="coerce",
            )
            .notna()
            .mean()
        )

        coverage_by_date = (
            frame.assign(_available=raw.notna()).groupby("as_of_date")["_available"].mean()
        )

        valid_dates = frame.loc[
            raw.notna(),
            "as_of_date",
        ]

        global_coverage = float(frame[global_column].notna().mean())

        sector_coverage = float(frame[sector_column].notna().mean())

        rows.append(
            {
                "factor": factor,
                "family": (FUNDAMENTAL_FEATURE_METADATA[factor]["family"]),
                "overall_raw_coverage": (overall_coverage),
                "mean_date_raw_coverage": float(coverage_by_date.mean()),
                "minimum_date_raw_coverage": float(coverage_by_date.min()),
                "latest_raw_coverage": (latest_coverage),
                "overall_global_zscore_coverage": (global_coverage),
                "overall_sector_zscore_coverage": (sector_coverage),
                "first_valid_date": (valid_dates.min() if not valid_dates.empty else pd.NaT),
                "last_valid_date": (valid_dates.max() if not valid_dates.empty else pd.NaT),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "family",
                "factor",
            ]
        )
        .reset_index(drop=True)
    )


def build_cross_sectional_correlations(
    data: pd.DataFrame,
    *,
    config: FundamentalAuditConfig,
) -> pd.DataFrame:
    """Average same-date Spearman correlations across history."""
    required = (
        "as_of_date",
        "ticker",
        *(f"{factor}_zscore" for factor in FUNDAMENTAL_FACTOR_COLUMNS),
    )

    _require_columns(
        data,
        required,
    )

    factor_pairs = list(
        combinations(
            FUNDAMENTAL_FACTOR_COLUMNS,
            2,
        )
    )

    observations: dict[
        tuple[str, str],
        list[float],
    ] = {pair: [] for pair in factor_pairs}

    for _, group in data.groupby(
        "as_of_date",
        sort=True,
    ):
        columns = {
            factor: pd.to_numeric(
                group[f"{factor}_zscore"],
                errors="coerce",
            )
            for factor in (FUNDAMENTAL_FACTOR_COLUMNS)
        }

        factor_frame = pd.DataFrame(
            columns,
            index=group.index,
        )

        correlations = factor_frame.corr(
            method="spearman",
            min_periods=(config.min_pair_observations),
        )

        for left, right in factor_pairs:
            value = correlations.loc[
                left,
                right,
            ]

            if pd.notna(value) and np.isfinite(value):
                observations[
                    (
                        left,
                        right,
                    )
                ].append(float(value))

    rows = []

    for (
        left,
        right,
    ), values in observations.items():
        if not values:
            continue

        array = np.asarray(
            values,
            dtype=float,
        )

        rows.append(
            {
                "factor_1": left,
                "factor_2": right,
                "dates_evaluated": (len(array)),
                "mean_spearman": float(array.mean()),
                "median_spearman": float(np.median(array)),
                "mean_abs_spearman": float(np.abs(array).mean()),
                "max_abs_spearman": float(np.abs(array).max()),
                "same_sign_ratio": float(
                    max(
                        np.mean(array >= 0.0),
                        np.mean(array <= 0.0),
                    )
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "factor_1",
                "factor_2",
                "dates_evaluated",
                "mean_spearman",
                "median_spearman",
                "mean_abs_spearman",
                "max_abs_spearman",
                "same_sign_ratio",
            ]
        )

    result = pd.DataFrame(rows)

    result["abs_mean_spearman"] = result["mean_spearman"].abs()

    return (
        result.sort_values(
            [
                "abs_mean_spearman",
                "dates_evaluated",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .drop(
            columns=[
                "abs_mean_spearman",
            ]
        )
        .reset_index(drop=True)
    )


def select_high_correlations(
    correlations: pd.DataFrame,
    *,
    config: FundamentalAuditConfig,
) -> pd.DataFrame:
    """Select potentially redundant factor pairs."""
    if correlations.empty:
        return correlations.copy()

    selected = correlations.loc[
        correlations["dates_evaluated"].ge(config.min_pair_dates)
        & correlations["mean_spearman"].abs().ge(config.high_correlation_threshold)
    ].copy()

    return (
        selected.assign(abs_mean_spearman=selected["mean_spearman"].abs())
        .sort_values(
            "abs_mean_spearman",
            ascending=False,
        )
        .drop(
            columns=[
                "abs_mean_spearman",
            ]
        )
        .reset_index(drop=True)
    )


def build_zscore_audit(
    data: pd.DataFrame,
    *,
    config: FundamentalAuditConfig,
) -> pd.DataFrame:
    """Audit centering and scaling of z-score features."""
    rows = []

    for factor in FUNDAMENTAL_FACTOR_COLUMNS:
        for version in (
            "zscore",
            "sector_zscore",
        ):
            column = f"{factor}_{version}"

            _require_columns(
                data,
                (column,),
            )

            if version == "zscore":
                group_columns = [
                    "as_of_date",
                ]
            else:
                group_columns = [
                    "as_of_date",
                    "sector",
                ]

            group_means = []
            group_stds = []

            for _, group in data.groupby(
                group_columns,
                sort=False,
                dropna=False,
            ):
                values = pd.to_numeric(
                    group[column],
                    errors="coerce",
                ).dropna()

                if len(values) < 2:
                    continue

                group_means.append(float(values.mean()))

                group_stds.append(float(values.std(ddof=0)))

            if group_means:
                means = np.asarray(
                    group_means,
                    dtype=float,
                )

                stds = np.asarray(
                    group_stds,
                    dtype=float,
                )

                violations = int((np.abs(means) > config.zscore_mean_tolerance).sum())

                max_abs_mean = float(np.abs(means).max())

                mean_std = float(stds.mean())
            else:
                violations = 0
                max_abs_mean = np.nan
                mean_std = np.nan

            rows.append(
                {
                    "factor": factor,
                    "version": version,
                    "groups_evaluated": (len(group_means)),
                    "max_abs_group_mean": (max_abs_mean),
                    "mean_group_std": (mean_std),
                    "centering_violations": (violations),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "version",
                "factor",
            ]
        )
        .reset_index(drop=True)
    )


def render_fundamental_dictionary() -> str:
    """Render the managed fundamental section of FEATURE_DICTIONARY."""
    validate_feature_metadata()

    lines = [
        "<!-- FUNDAMENTAL_FEATURES_START -->",
        "",
        "# Fundamental features",
        "",
        (
            "The following fundamental features are constructed "
            "point-in-time using only accounting information that "
            "was available on or before each rebalance date."
        ),
        "",
        "## Transformation suffixes",
        "",
        "- `_winsorized`: cross-sectionally winsorized by rebalance date.",
        (
            "- `_zscore`: standardized relative to all available "
            "companies on the same rebalance date."
        ),
        (
            "- `_sector_zscore`: standardized relative to available "
            "companies in the same sector and rebalance date."
        ),
        (
            "- `_missing`: binary indicator equal to 1 when the "
            "underlying raw factor is unavailable."
        ),
        "",
        "Missing accounting values are not automatically imputed.",
        "",
    ]

    for factor in FUNDAMENTAL_FACTOR_COLUMNS:
        metadata = FUNDAMENTAL_FEATURE_METADATA[factor]

        lines.extend(
            [
                f"## `{factor}`",
                "",
                (f"**Family:** {metadata['family']}"),
                "",
                (f"**Formula:** {metadata['formula']}"),
                "",
                (f"**Interpretation:** {metadata['interpretation']}"),
                "",
            ]
        )

    lines.extend(
        [
            "## Important construction conventions",
            "",
            (
                "- Market Cap Proxy uses historical close price "
                "multiplied by point-in-time shares outstanding, "
                "with quarterly diluted shares used only as a fallback."
            ),
            (
                "- CAPEX is represented as a positive cash outflow; "
                "therefore Free Cash Flow equals Operating Cash Flow "
                "minus CAPEX."
            ),
            ("- Missing debt components are not automatically interpreted as zero."),
            (
                "- Growth signals compare each company only with "
                "its own historical accounting information."
            ),
            (
                "- Cross-sectional transformations are calculated "
                "independently for each rebalance date."
            ),
            "",
            "<!-- FUNDAMENTAL_FEATURES_END -->",
            "",
        ]
    )

    return "\n".join(lines)


def update_feature_dictionary(
    path: Path,
) -> None:
    """Insert or replace the managed fundamental dictionary section."""
    section = render_fundamental_dictionary()

    start_marker = "<!-- FUNDAMENTAL_FEATURES_START -->"

    end_marker = "<!-- FUNDAMENTAL_FEATURES_END -->"

    if path.exists():
        existing = path.read_text(
            encoding="utf-8",
        )
    else:
        existing = "# Feature Dictionary\n\n"

    if start_marker in existing and end_marker in existing:
        start = existing.index(start_marker)

        end = existing.index(end_marker) + len(end_marker)

        updated = (
            existing[:start].rstrip() + "\n\n" + section.strip() + "\n" + existing[end:].lstrip()
        )
    else:
        updated = existing.rstrip() + "\n\n" + section

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        updated,
        encoding="utf-8",
    )
