"""Raw point-in-time fundamental factors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class FundamentalFactorError(ValueError):
    """Raised when raw fundamental factors cannot be built."""


RAW_FACTOR_COLUMNS = (
    "earnings_yield",
    "sales_yield",
    "book_to_market",
    "fcf_yield",
    "roe",
    "roa",
    "gross_profitability",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "cash_conversion",
    "debt_to_assets",
    "net_debt_to_assets",
    "current_ratio",
    "interest_coverage",
    "capex_to_assets",
    "accruals",
)


@dataclass(frozen=True)
class FundamentalFactorConfig:
    """Configuration for raw fundamental factors."""

    diluted_share_metric: str = "diluted_shares"
    diluted_share_duration_class: str = "quarter"
    capex_positive_outflow: bool = True
    min_abs_denominator: float = 1.0e-12

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> FundamentalFactorConfig:
        """Create configuration from project YAML."""
        config = cls(
            diluted_share_metric=str(
                values.get(
                    "diluted_share_metric",
                    "diluted_shares",
                )
            ),
            diluted_share_duration_class=str(
                values.get(
                    "diluted_share_duration_class",
                    "quarter",
                )
            ),
            capex_positive_outflow=bool(
                values.get(
                    "capex_positive_outflow",
                    True,
                )
            ),
            min_abs_denominator=float(
                values.get(
                    "min_abs_denominator",
                    1.0e-12,
                )
            ),
        )

        if config.min_abs_denominator <= 0:
            raise FundamentalFactorError("min_abs_denominator must be positive.")

        return config


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require columns in one dataset."""
    missing = [column for column in columns if column not in data.columns]

    if missing:
        raise FundamentalFactorError(f"{dataset_name} is missing columns: {missing}")


def _normalize_tickers(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize ticker symbols."""
    result = data.copy()

    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()

    return result


def _normalize_date(
    data: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Normalize one date column."""
    result = data.copy()

    result[column] = pd.to_datetime(
        result[column],
        errors="coerce",
    ).dt.normalize()

    return result


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    epsilon: float,
    require_positive_denominator: bool = False,
) -> pd.Series:
    """Safely divide two numeric series."""
    numerator_numeric = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator_numeric = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    valid = (
        numerator_numeric.notna()
        & denominator_numeric.notna()
        & denominator_numeric.abs().gt(epsilon)
    )

    if require_positive_denominator:
        valid &= denominator_numeric.gt(epsilon)

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype="float64",
    )

    result.loc[valid] = numerator_numeric.loc[valid] / denominator_numeric.loc[valid]

    return result


def _prepare_diluted_share_fallback(
    pit_snapshots: pd.DataFrame,
    *,
    config: FundamentalFactorConfig,
) -> pd.DataFrame:
    """Extract latest quarterly diluted shares point-in-time."""
    _require_columns(
        pit_snapshots,
        (
            "as_of_date",
            "ticker",
            "canonical_metric",
            "statement_type",
            "duration_class",
            "value",
            "end_date",
            "available_date",
        ),
        dataset_name="Point-in-time fundamentals",
    )

    data = _normalize_tickers(pit_snapshots)

    for column in (
        "as_of_date",
        "end_date",
        "available_date",
    ):
        data = _normalize_date(
            data,
            column,
        )

    data = data.loc[
        data["canonical_metric"].eq(config.diluted_share_metric)
        & data["statement_type"].eq("duration")
        & data["duration_class"].eq(config.diluted_share_duration_class)
    ].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "as_of_date",
                "ticker",
                "diluted_shares_quarter",
                "diluted_shares_period_end",
                "diluted_shares_available_date",
            ]
        )

    data["value"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    data = data.loc[
        data["value"].gt(0.0) & data["end_date"].notna() & data["available_date"].notna()
    ].copy()

    future_information = data["available_date"] > data["as_of_date"]

    if future_information.any():
        raise FundamentalFactorError("Diluted-share fallback contains future information.")

    future_period = data["end_date"] > data["as_of_date"]

    if future_period.any():
        raise FundamentalFactorError("Diluted-share fallback contains future reporting periods.")

    data = data.sort_values(
        [
            "as_of_date",
            "ticker",
            "end_date",
            "available_date",
        ]
    ).drop_duplicates(
        [
            "as_of_date",
            "ticker",
        ],
        keep="last",
    )

    result = data.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "value",
            "end_date",
            "available_date",
        ],
    ].rename(
        columns={
            "value": ("diluted_shares_quarter"),
            "end_date": ("diluted_shares_period_end"),
            "available_date": ("diluted_shares_available_date"),
        }
    )

    return result.reset_index(drop=True)


def build_raw_fundamental_factors(
    *,
    fundamental_base: pd.DataFrame,
    pit_snapshots: pd.DataFrame,
    config: FundamentalFactorConfig,
) -> pd.DataFrame:
    """Build raw point-in-time fundamental ratios."""
    required_base_columns = (
        "as_of_date",
        "ticker",
        "close_price",
        "assets",
        "cash",
        "equity",
        "shares_outstanding",
        "current_assets",
        "current_liabilities",
        "debt_current",
        "debt_noncurrent",
        "revenue_ttm",
        "gross_profit_ttm",
        "operating_income_ttm",
        "net_income_ttm",
        "operating_cash_flow_ttm",
        "capex_ttm",
        "interest_expense_ttm",
    )

    _require_columns(
        fundamental_base,
        required_base_columns,
        dataset_name="Fundamental base",
    )

    result = _normalize_tickers(fundamental_base)

    result = _normalize_date(
        result,
        "as_of_date",
    )

    duplicates = result.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    )

    if duplicates.any():
        raise FundamentalFactorError("Fundamental base contains duplicate date-ticker rows.")

    diluted = _prepare_diluted_share_fallback(
        pit_snapshots,
        config=config,
    )

    result = result.merge(
        diluted,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    shares_outstanding = pd.to_numeric(
        result["shares_outstanding"],
        errors="coerce",
    )

    diluted_shares = pd.to_numeric(
        result["diluted_shares_quarter"],
        errors="coerce",
    )

    outstanding_valid = shares_outstanding.notna() & shares_outstanding.gt(0.0)

    diluted_valid = diluted_shares.notna() & diluted_shares.gt(0.0)

    result["valuation_share_count"] = np.nan

    result.loc[
        outstanding_valid,
        "valuation_share_count",
    ] = shares_outstanding.loc[outstanding_valid]

    fallback_mask = ~outstanding_valid & diluted_valid

    result.loc[
        fallback_mask,
        "valuation_share_count",
    ] = diluted_shares.loc[fallback_mask]

    source = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    source.loc[outstanding_valid] = "shares_outstanding"

    source.loc[fallback_mask] = "diluted_shares_quarter"

    result["valuation_share_count_source"] = source

    result["valuation_share_count_available_date"] = pd.NaT

    if "shares_outstanding_available_date" in result.columns:
        outstanding_dates = pd.to_datetime(
            result["shares_outstanding_available_date"],
            errors="coerce",
        ).dt.normalize()

        result.loc[
            outstanding_valid,
            "valuation_share_count_available_date",
        ] = outstanding_dates.loc[outstanding_valid]

    diluted_dates = pd.to_datetime(
        result["diluted_shares_available_date"],
        errors="coerce",
    ).dt.normalize()

    result.loc[
        fallback_mask,
        "valuation_share_count_available_date",
    ] = diluted_dates.loc[fallback_mask]

    future_share_information = result["valuation_share_count_available_date"].notna() & (
        result["valuation_share_count_available_date"] > result["as_of_date"]
    )

    if future_share_information.any():
        raise FundamentalFactorError("Valuation share count contains future information.")

    close_price = pd.to_numeric(
        result["close_price"],
        errors="coerce",
    )

    valid_price = close_price.notna() & close_price.gt(0.0)

    if not valid_price.all():
        raise FundamentalFactorError("Fundamental base contains invalid close prices.")

    result["market_cap_proxy"] = close_price * result["valuation_share_count"]

    operating_cash_flow = pd.to_numeric(
        result["operating_cash_flow_ttm"],
        errors="coerce",
    )

    capex = pd.to_numeric(
        result["capex_ttm"],
        errors="coerce",
    )

    if config.capex_positive_outflow:
        result["free_cash_flow_ttm"] = operating_cash_flow - capex
    else:
        result["free_cash_flow_ttm"] = operating_cash_flow + capex

    debt_current = pd.to_numeric(
        result["debt_current"],
        errors="coerce",
    )

    debt_noncurrent = pd.to_numeric(
        result["debt_noncurrent"],
        errors="coerce",
    )

    both_debt_components = debt_current.notna() & debt_noncurrent.notna()

    result["total_debt"] = np.nan

    result.loc[
        both_debt_components,
        "total_debt",
    ] = debt_current.loc[both_debt_components] + debt_noncurrent.loc[both_debt_components]

    cash = pd.to_numeric(
        result["cash"],
        errors="coerce",
    )

    result["net_debt"] = result["total_debt"] - cash

    epsilon = config.min_abs_denominator

    market_cap = result["market_cap_proxy"]

    assets = result["assets"]
    equity = result["equity"]
    revenue = result["revenue_ttm"]
    gross_profit = result["gross_profit_ttm"]
    operating_income = result["operating_income_ttm"]
    net_income = result["net_income_ttm"]
    interest_expense = result["interest_expense_ttm"]

    result["earnings_yield"] = _safe_divide(
        net_income,
        market_cap,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["sales_yield"] = _safe_divide(
        revenue,
        market_cap,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["book_to_market"] = _safe_divide(
        equity,
        market_cap,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["fcf_yield"] = _safe_divide(
        result["free_cash_flow_ttm"],
        market_cap,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["roe"] = _safe_divide(
        net_income,
        equity,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["roa"] = _safe_divide(
        net_income,
        assets,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["gross_profitability"] = _safe_divide(
        gross_profit,
        assets,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["gross_margin"] = _safe_divide(
        gross_profit,
        revenue,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["operating_margin"] = _safe_divide(
        operating_income,
        revenue,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["net_margin"] = _safe_divide(
        net_income,
        revenue,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["cash_conversion"] = _safe_divide(
        operating_cash_flow,
        net_income,
        epsilon=epsilon,
    )

    result["debt_to_assets"] = _safe_divide(
        result["total_debt"],
        assets,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["net_debt_to_assets"] = _safe_divide(
        result["net_debt"],
        assets,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["current_ratio"] = _safe_divide(
        result["current_assets"],
        result["current_liabilities"],
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["interest_coverage"] = _safe_divide(
        operating_income,
        interest_expense,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["capex_to_assets"] = _safe_divide(
        capex,
        assets,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    result["accruals"] = _safe_divide(
        (
            pd.to_numeric(
                net_income,
                errors="coerce",
            )
            - operating_cash_flow
        ),
        assets,
        epsilon=epsilon,
        require_positive_denominator=True,
    )

    factor_frame = result.loc[
        :,
        list(RAW_FACTOR_COLUMNS),
    ]

    infinite_values = int(np.isinf(factor_frame.to_numpy(dtype=float)).sum())

    if infinite_values:
        raise FundamentalFactorError(
            f"Raw fundamental factors contain {infinite_values} infinite values."
        )

    result["raw_factor_count"] = factor_frame.notna().sum(axis=1)

    return result.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)
