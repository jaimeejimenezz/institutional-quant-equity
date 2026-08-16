"""Transaction execution cost models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class ExecutionCostError(ValueError):
    """Raised when execution costs cannot be calculated safely."""


@dataclass(frozen=True)
class ExecutionCostConfig:
    """Configuration for transaction execution costs."""

    commission_bps: float = 0.5
    half_spread_bps: float = 2.0
    slippage_bps: float = 2.5

    market_impact_coefficient: float = 0.10

    annualization_factor: int = 252

    @property
    def linear_cost_bps(self) -> float:
        """Return total linear execution cost in basis points."""
        return self.commission_bps + self.half_spread_bps + self.slippage_bps

    @property
    def linear_cost_rate(self) -> float:
        """Return total linear execution cost as a decimal rate."""
        return self.linear_cost_bps / 10_000.0

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> ExecutionCostConfig:
        """Create execution-cost settings from project configuration."""
        config = cls(
            commission_bps=float(
                values.get(
                    "commission_bps",
                    0.5,
                )
            ),
            half_spread_bps=float(
                values.get(
                    "half_spread_bps",
                    2.0,
                )
            ),
            slippage_bps=float(
                values.get(
                    "slippage_bps",
                    2.5,
                )
            ),
            market_impact_coefficient=float(
                values.get(
                    "market_impact_coefficient",
                    0.10,
                )
            ),
            annualization_factor=int(
                values.get(
                    "annualization_factor",
                    252,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate execution-cost assumptions."""
        if self.commission_bps < 0.0:
            raise ExecutionCostError("commission_bps cannot be negative.")

        if self.half_spread_bps < 0.0:
            raise ExecutionCostError("half_spread_bps cannot be negative.")

        if self.slippage_bps < 0.0:
            raise ExecutionCostError("slippage_bps cannot be negative.")

        if self.market_impact_coefficient < 0.0:
            raise ExecutionCostError("market_impact_coefficient cannot be negative.")

        if self.annualization_factor < 1:
            raise ExecutionCostError("annualization_factor must be positive.")


def estimate_execution_cost_arrays(
    absolute_trade_notional: np.ndarray,
    annualized_volatility: np.ndarray,
    average_dollar_volume: np.ndarray,
    *,
    config: ExecutionCostConfig,
) -> dict[str, np.ndarray]:
    """Estimate execution costs for aligned arrays of trades."""
    config.validate()

    notional = np.asarray(
        absolute_trade_notional,
        dtype=float,
    ).reshape(-1)

    volatility = np.asarray(
        annualized_volatility,
        dtype=float,
    ).reshape(-1)

    dollar_volume = np.asarray(
        average_dollar_volume,
        dtype=float,
    ).reshape(-1)

    if not (notional.shape == volatility.shape == dollar_volume.shape):
        raise ExecutionCostError("Execution-cost arrays must have identical shapes.")

    if not (
        np.isfinite(notional).all()
        and np.isfinite(volatility).all()
        and np.isfinite(dollar_volume).all()
    ):
        raise ExecutionCostError("Execution-cost arrays must contain finite values.")

    if (notional < 0.0).any():
        raise ExecutionCostError("absolute_trade_notional cannot contain negative values.")

    if (volatility < 0.0).any():
        raise ExecutionCostError("annualized_volatility cannot contain negative values.")

    if (dollar_volume <= 0.0).any():
        raise ExecutionCostError("average_dollar_volume must contain positive values.")

    commission_cost = notional * config.commission_bps / 10_000.0

    spread_cost = notional * config.half_spread_bps / 10_000.0

    slippage_cost = notional * config.slippage_bps / 10_000.0

    daily_volatility = volatility / np.sqrt(config.annualization_factor)

    order_adv_fraction = notional / dollar_volume

    market_impact_rate = (
        config.market_impact_coefficient * daily_volatility * np.sqrt(order_adv_fraction)
    )

    market_impact_cost = notional * market_impact_rate

    total_execution_cost = commission_cost + spread_cost + slippage_cost + market_impact_cost

    effective_cost_bps = np.zeros_like(notional)

    positive_trade = notional > 0.0

    effective_cost_bps[positive_trade] = (
        total_execution_cost[positive_trade] / notional[positive_trade] * 10_000.0
    )

    return {
        "commission_cost": commission_cost,
        "spread_cost": spread_cost,
        "slippage_cost": slippage_cost,
        "market_impact_cost": market_impact_cost,
        "total_execution_cost": total_execution_cost,
        "market_impact_bps": (market_impact_rate * 10_000.0),
        "effective_cost_bps": (effective_cost_bps),
        "order_adv_fraction": (order_adv_fraction),
    }


def estimate_trade_execution_cost(
    absolute_trade_notional: float,
    annualized_volatility: float,
    average_dollar_volume: float,
    *,
    config: ExecutionCostConfig,
) -> dict[str, float]:
    """Estimate execution costs for one trade."""
    config.validate()

    notional = float(absolute_trade_notional)

    volatility = float(annualized_volatility)

    dollar_volume = float(average_dollar_volume)

    if not np.isfinite(
        [
            notional,
            volatility,
            dollar_volume,
        ]
    ).all():
        raise ExecutionCostError("Execution-cost inputs must be finite.")

    if notional < 0.0:
        raise ExecutionCostError("absolute_trade_notional cannot be negative.")

    if volatility < 0.0:
        raise ExecutionCostError("annualized_volatility cannot be negative.")

    if dollar_volume <= 0.0:
        raise ExecutionCostError("average_dollar_volume must be positive.")

    if notional == 0.0:
        return {
            "commission_cost": 0.0,
            "spread_cost": 0.0,
            "slippage_cost": 0.0,
            "market_impact_cost": 0.0,
            "total_execution_cost": 0.0,
            "linear_cost_bps": (config.linear_cost_bps),
            "market_impact_bps": 0.0,
            "effective_cost_bps": 0.0,
            "order_adv_fraction": 0.0,
        }

    commission_rate = config.commission_bps / 10_000.0

    spread_rate = config.half_spread_bps / 10_000.0

    slippage_rate = config.slippage_bps / 10_000.0

    daily_volatility = volatility / np.sqrt(config.annualization_factor)

    order_adv_fraction = notional / dollar_volume

    market_impact_rate = (
        config.market_impact_coefficient * daily_volatility * np.sqrt(order_adv_fraction)
    )

    commission_cost = notional * commission_rate

    spread_cost = notional * spread_rate

    slippage_cost = notional * slippage_rate

    market_impact_cost = notional * market_impact_rate

    total_execution_cost = commission_cost + spread_cost + slippage_cost + market_impact_cost

    market_impact_bps = market_impact_rate * 10_000.0

    effective_cost_bps = total_execution_cost / notional * 10_000.0

    return {
        "commission_cost": float(commission_cost),
        "spread_cost": float(spread_cost),
        "slippage_cost": float(slippage_cost),
        "market_impact_cost": float(market_impact_cost),
        "total_execution_cost": float(total_execution_cost),
        "linear_cost_bps": float(config.linear_cost_bps),
        "market_impact_bps": float(market_impact_bps),
        "effective_cost_bps": float(effective_cost_bps),
        "order_adv_fraction": float(order_adv_fraction),
    }


def estimate_trade_execution_costs(
    trades: pd.DataFrame,
    risk_estimates: pd.DataFrame,
    *,
    config: ExecutionCostConfig,
) -> pd.DataFrame:
    """Attach point-in-time execution costs to a table of trades."""
    required_trade_columns = {
        "signal_date",
        "ticker",
        "absolute_trade_notional",
    }

    missing_trade_columns = sorted(required_trade_columns.difference(trades.columns))

    if missing_trade_columns:
        raise ExecutionCostError(
            "Trades are missing columns: " + ", ".join(missing_trade_columns) + "."
        )

    required_risk_columns = {
        "as_of_date",
        "ticker",
        "annualized_volatility",
        "average_dollar_volume",
    }

    missing_risk_columns = sorted(required_risk_columns.difference(risk_estimates.columns))

    if missing_risk_columns:
        raise ExecutionCostError(
            "Risk estimates are missing columns: " + ", ".join(missing_risk_columns) + "."
        )

    trade_data = trades.copy()

    trade_data["signal_date"] = pd.to_datetime(trade_data["signal_date"]).dt.normalize()

    trade_data["ticker"] = trade_data["ticker"].astype(str)

    risk_data = risk_estimates.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "annualized_volatility",
            "average_dollar_volume",
        ],
    ].copy()

    risk_data["as_of_date"] = pd.to_datetime(risk_data["as_of_date"]).dt.normalize()

    risk_data["ticker"] = risk_data["ticker"].astype(str)

    if risk_data.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    ).any():
        raise ExecutionCostError("Risk estimates contain duplicated date-ticker keys.")

    result = trade_data.merge(
        risk_data,
        left_on=[
            "signal_date",
            "ticker",
        ],
        right_on=[
            "as_of_date",
            "ticker",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_risk = (
        result[
            [
                "annualized_volatility",
                "average_dollar_volume",
            ]
        ]
        .isna()
        .any(axis=1)
    )

    if missing_risk.any():
        sample = result.loc[
            missing_risk,
            [
                "signal_date",
                "ticker",
            ],
        ].iloc[0]

        raise ExecutionCostError(
            "Missing point-in-time risk estimates for "
            f"{sample['ticker']} on "
            f"{sample['signal_date'].date()}."
        )

    cost_rows = []

    for row in result.itertuples(index=False):
        cost_rows.append(
            estimate_trade_execution_cost(
                float(row.absolute_trade_notional),
                float(row.annualized_volatility),
                float(row.average_dollar_volume),
                config=config,
            )
        )

    cost_frame = pd.DataFrame(
        cost_rows,
        index=result.index,
    )

    for column in cost_frame.columns:
        result[column] = cost_frame[column]

    return result.drop(
        columns=[
            "as_of_date",
        ]
    ).reset_index(drop=True)
