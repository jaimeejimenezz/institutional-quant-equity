"""Conditional Value-at-Risk portfolio construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from quant_equity.portfolio.optimizer import (
    PortfolioOptimizerConfig,
    _calculate_realized_turnover,
    _prepare_candidate_risk,
    _prepare_signal,
    _select_candidates,
)


class CvarPortfolioError(ValueError):
    """Raised when CVaR portfolio construction cannot be completed."""


@dataclass(frozen=True)
class CvarRiskConfig:
    """Configuration for historical CVaR estimation."""

    confidence_level: float = 0.95
    horizon_days: int = 21
    scenario_lookback: int = 252
    minimum_scenarios: int = 126
    cvar_penalty: float = 0.05

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> CvarRiskConfig:
        """Build CVaR settings from project configuration."""
        config = cls(
            confidence_level=float(
                values.get(
                    "confidence_level",
                    0.95,
                )
            ),
            horizon_days=int(
                values.get(
                    "horizon_days",
                    21,
                )
            ),
            scenario_lookback=int(
                values.get(
                    "scenario_lookback",
                    252,
                )
            ),
            minimum_scenarios=int(
                values.get(
                    "minimum_scenarios",
                    126,
                )
            ),
            cvar_penalty=float(
                values.get(
                    "cvar_penalty",
                    0.05,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate CVaR configuration."""
        if not 0.0 < self.confidence_level < 1.0:
            raise CvarPortfolioError("confidence_level must be between zero and one.")

        if self.horizon_days < 1:
            raise CvarPortfolioError("horizon_days must be positive.")

        if self.scenario_lookback < 1:
            raise CvarPortfolioError("scenario_lookback must be positive.")

        if self.minimum_scenarios < 1:
            raise CvarPortfolioError("minimum_scenarios must be positive.")

        if self.minimum_scenarios > self.scenario_lookback:
            raise CvarPortfolioError("minimum_scenarios cannot exceed scenario_lookback.")

        if self.cvar_penalty < 0.0:
            raise CvarPortfolioError("cvar_penalty cannot be negative.")


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require expected columns."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise CvarPortfolioError(f"{dataset_name} is missing columns: " + ", ".join(missing) + ".")


def _build_horizon_return_scenarios(
    market_daily: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    tickers: list[str],
    config: CvarRiskConfig,
) -> np.ndarray:
    """Build historical multi-session return scenarios using only past data."""
    _require_columns(
        market_daily,
        (
            "date",
            "ticker",
            "adjusted_close",
        ),
        dataset_name="daily market data",
    )

    data = market_daily.loc[
        :,
        [
            "date",
            "ticker",
            "adjusted_close",
        ],
    ].copy()

    data["date"] = pd.to_datetime(data["date"]).dt.normalize()

    data["ticker"] = data["ticker"].astype(str)

    data["adjusted_close"] = pd.to_numeric(
        data["adjusted_close"],
        errors="coerce",
    )

    data = data.loc[data["date"].le(as_of_date) & data["ticker"].isin(tickers)]

    prices = (
        data.pivot(
            index="date",
            columns="ticker",
            values="adjusted_close",
        )
        .reindex(columns=tickers)
        .sort_index()
        .dropna()
    )

    horizon_returns = (prices / prices.shift(config.horizon_days) - 1.0).dropna()

    scenarios = horizon_returns.tail(config.scenario_lookback)

    if len(scenarios) < config.minimum_scenarios:
        raise CvarPortfolioError(
            "Insufficient historical scenarios for "
            f"{as_of_date.date()}: "
            f"{len(scenarios)} available."
        )

    values = scenarios.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise CvarPortfolioError("CVaR scenarios contain non-finite values.")

    return values


def _solve_cvar_portfolio(
    candidates: pd.DataFrame,
    scenario_returns: np.ndarray,
    candidate_risk: pd.DataFrame,
    *,
    previous_weights: pd.Series | None,
    portfolio_config: PortfolioOptimizerConfig,
    cvar_config: CvarRiskConfig,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    """Solve one alpha-CVaR-turnover portfolio."""
    tickers = candidates["ticker"].astype(str).tolist()

    horizon_alpha = (
        (candidates["percentile_score"].to_numpy(dtype=float) - 0.5)
        * portfolio_config.annualized_alpha_scale
        * (cvar_config.horizon_days / 252.0)
    )

    beta = candidate_risk["beta_vs_spy"].to_numpy(dtype=float)

    average_dollar_volume = candidate_risk["average_dollar_volume"].to_numpy(dtype=float)

    effective_security_limit = (
        portfolio_config.max_security_weight - portfolio_config.constraint_margin
    )

    effective_sector_limit = portfolio_config.max_sector_weight - portfolio_config.constraint_margin

    liquidity_weight_limits = (
        portfolio_config.max_position_adv_fraction
        * average_dollar_volume
        / portfolio_config.reference_portfolio_value
    )

    maximum_feasible_weights = np.minimum(
        effective_security_limit,
        liquidity_weight_limits,
    )

    if maximum_feasible_weights.sum() < 1.0 - portfolio_config.weight_tolerance:
        raise CvarPortfolioError(
            "Security and liquidity constraints cannot support a fully invested CVaR portfolio."
        )

    number_of_assets = len(candidates)

    number_of_scenarios = len(scenario_returns)

    weights = cp.Variable(number_of_assets)

    var_threshold = cp.Variable()

    excess_losses = cp.Variable(
        number_of_scenarios,
        nonneg=True,
    )

    scenario_losses = -scenario_returns @ weights

    cvar_expression = var_threshold + (
        cp.sum(excess_losses) / ((1.0 - cvar_config.confidence_level) * number_of_scenarios)
    )

    constraints = [
        weights >= 0.0,
        weights <= effective_security_limit,
        weights <= liquidity_weight_limits,
        cp.sum(weights) == 1.0,
        excess_losses >= scenario_losses - var_threshold,
        beta @ weights >= portfolio_config.minimum_portfolio_beta,
        beta @ weights <= portfolio_config.maximum_portfolio_beta,
    ]

    sectors = candidates["sector"].astype(str).to_numpy()

    for sector in sorted(set(sectors)):
        indices = np.where(sectors == sector)[0]

        constraints.append(cp.sum(weights[indices]) <= effective_sector_limit)

    if previous_weights is None:
        turnover_expression = 0.0
        previous_current = np.zeros(
            number_of_assets,
            dtype=float,
        )
    else:
        previous_current = previous_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)

        dropped_weight = float(previous_weights.loc[~previous_weights.index.isin(tickers)].sum())

        turnover_expression = 0.5 * (cp.norm1(weights - previous_current) + dropped_weight)

    objective = cp.Maximize(
        horizon_alpha @ weights
        - (cvar_config.cvar_penalty * cvar_expression)
        - (portfolio_config.turnover_penalty * turnover_expression)
    )

    problem = cp.Problem(
        objective,
        constraints,
    )

    installed_solvers = set(cp.installed_solvers())

    if "CLARABEL" in installed_solvers:
        problem.solve(
            solver=cp.CLARABEL,
            max_iter=500,
            tol_gap_abs=1e-8,
            tol_gap_rel=1e-8,
            tol_feas=1e-8,
        )
    elif "SCS" in installed_solvers:
        problem.solve(
            solver=cp.SCS,
            max_iters=100_000,
            eps=1e-6,
        )
    else:
        raise CvarPortfolioError("CVaR optimization requires CLARABEL or SCS.")

    if problem.status not in {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    }:
        raise CvarPortfolioError(f"CVaR optimization failed with status {problem.status}.")

    solved_weights = np.asarray(
        weights.value,
        dtype=float,
    ).reshape(-1)

    solved_weights[np.abs(solved_weights) < portfolio_config.weight_tolerance] = 0.0

    solved_weights = np.clip(
        solved_weights,
        0.0,
        None,
    )

    total_weight = float(solved_weights.sum())

    normalization_tolerance = max(
        100.0 * portfolio_config.solver_tolerance,
        10.0 * portfolio_config.weight_tolerance,
    )

    if abs(total_weight - 1.0) > normalization_tolerance:
        raise CvarPortfolioError(
            f"CVaR portfolio weights do not sum sufficiently close to one: {total_weight:.12f}."
        )

    solved_weights = solved_weights / total_weight

    if (
        solved_weights.max()
        > portfolio_config.max_security_weight + portfolio_config.weight_tolerance
    ):
        raise CvarPortfolioError("Normalized CVaR portfolio violates the maximum security weight.")

    normalized_sector_weights = (
        candidates.assign(solved_weight=solved_weights).groupby("sector")["solved_weight"].sum()
    )

    if (
        normalized_sector_weights.max()
        > portfolio_config.max_sector_weight + portfolio_config.weight_tolerance
    ):
        raise CvarPortfolioError("Normalized CVaR portfolio violates the maximum sector weight.")

    normalized_beta = float(beta @ solved_weights)

    if (
        normalized_beta
        < portfolio_config.minimum_portfolio_beta - portfolio_config.weight_tolerance
    ):
        raise CvarPortfolioError("Normalized CVaR portfolio violates the minimum beta constraint.")

    if (
        normalized_beta
        > portfolio_config.maximum_portfolio_beta + portfolio_config.weight_tolerance
    ):
        raise CvarPortfolioError("Normalized CVaR portfolio violates the maximum beta constraint.")

    normalized_adv_fractions = (
        portfolio_config.reference_portfolio_value * solved_weights / average_dollar_volume
    )

    if (
        normalized_adv_fractions.max()
        > portfolio_config.max_position_adv_fraction + portfolio_config.weight_tolerance
    ):
        raise CvarPortfolioError("Normalized CVaR portfolio violates the liquidity constraint.")

    current_weights = pd.Series(
        solved_weights,
        index=tickers,
        dtype=float,
    )

    one_way_turnover = _calculate_realized_turnover(
        previous_weights,
        current_weights,
    )

    realized_scenario_returns = scenario_returns @ solved_weights

    realized_losses = -realized_scenario_returns

    var_loss = float(
        np.quantile(
            realized_losses,
            cvar_config.confidence_level,
        )
    )

    tail_losses = realized_losses[realized_losses >= var_loss - portfolio_config.weight_tolerance]

    empirical_cvar = float(tail_losses.mean())

    portfolio_beta = float(beta @ solved_weights)

    position_adv_fractions = (
        portfolio_config.reference_portfolio_value * solved_weights / average_dollar_volume
    )

    sector_weights = (
        candidates.assign(solved_weight=solved_weights).groupby("sector")["solved_weight"].sum()
    )

    diagnostics = {
        "solver_status": str(problem.status),
        "objective_value": float(problem.value),
        "scenario_count": int(number_of_scenarios),
        "confidence_level": float(cvar_config.confidence_level),
        "horizon_days": int(cvar_config.horizon_days),
        "predicted_alpha_proxy": float(horizon_alpha @ solved_weights),
        "var_loss": var_loss,
        "cvar_loss": empirical_cvar,
        "mean_scenario_return": float(realized_scenario_returns.mean()),
        "worst_scenario_return": float(realized_scenario_returns.min()),
        "one_way_turnover": (one_way_turnover),
        "portfolio_beta_vs_spy": (portfolio_beta),
        "maximum_position_adv_fraction": float(position_adv_fractions.max()),
        "positions": int((solved_weights > portfolio_config.weight_tolerance).sum()),
        "maximum_weight": float(solved_weights.max()),
        "maximum_sector_weight": float(sector_weights.max()),
        "solver": str(problem.solver_stats.solver_name),
    }

    return (
        solved_weights,
        diagnostics,
    )


def build_cvar_portfolios(
    final_signal: pd.DataFrame,
    market_daily: pd.DataFrame,
    risk_estimates: pd.DataFrame,
    *,
    portfolio_config: PortfolioOptimizerConfig,
    cvar_config: CvarRiskConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build historical CVaR portfolios through time."""
    portfolio_config.validate()
    cvar_config.validate()

    signal = _prepare_signal(final_signal)

    portfolio_blocks = []
    diagnostic_rows = []

    previous_weights: pd.Series | None = None

    for (
        as_of_date,
        date_signal,
    ) in signal.groupby(
        "as_of_date",
        sort=True,
    ):
        normalized_date = pd.Timestamp(as_of_date).normalize()

        candidates = _select_candidates(
            date_signal,
            config=portfolio_config,
        )

        tickers = candidates["ticker"].astype(str).tolist()

        scenario_returns = _build_horizon_return_scenarios(
            market_daily,
            as_of_date=normalized_date,
            tickers=tickers,
            config=cvar_config,
        )

        candidate_risk = _prepare_candidate_risk(
            risk_estimates,
            as_of_date=normalized_date,
            tickers=tickers,
        )

        (
            solved_weights,
            diagnostics,
        ) = _solve_cvar_portfolio(
            candidates,
            scenario_returns,
            candidate_risk,
            previous_weights=previous_weights,
            portfolio_config=portfolio_config,
            cvar_config=cvar_config,
        )

        if previous_weights is None:
            previous_candidate_weights = np.zeros(
                len(candidates),
                dtype=float,
            )
        else:
            previous_candidate_weights = (
                previous_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
            )

        candidates = candidates.copy()

        candidates["horizon_alpha_proxy"] = (
            (candidates["percentile_score"] - 0.5)
            * portfolio_config.annualized_alpha_scale
            * (cvar_config.horizon_days / 252.0)
        )

        candidates["previous_weight"] = previous_candidate_weights

        candidates["weight"] = solved_weights

        candidates["method"] = "cvar"

        candidates["beta_vs_spy"] = candidate_risk["beta_vs_spy"].to_numpy(dtype=float)

        candidates["average_dollar_volume"] = candidate_risk["average_dollar_volume"].to_numpy(
            dtype=float
        )

        candidates["position_adv_fraction"] = (
            portfolio_config.reference_portfolio_value
            * solved_weights
            / candidates["average_dollar_volume"].to_numpy(dtype=float)
        )

        portfolio_blocks.append(
            candidates.loc[
                :,
                [
                    "as_of_date",
                    "ticker",
                    "sector",
                    "method",
                    "rank",
                    "percentile_score",
                    "horizon_alpha_proxy",
                    "previous_weight",
                    "beta_vs_spy",
                    "average_dollar_volume",
                    "position_adv_fraction",
                    "weight",
                ],
            ]
        )

        diagnostic_rows.append(
            {
                "as_of_date": (normalized_date),
                **diagnostics,
            }
        )

        previous_weights = pd.Series(
            solved_weights,
            index=tickers,
            dtype=float,
        )

    weights = (
        pd.concat(
            portfolio_blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    diagnostics = pd.DataFrame(diagnostic_rows).sort_values("as_of_date").reset_index(drop=True)

    return (
        weights,
        diagnostics,
    )


def validate_cvar_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    portfolio_config: PortfolioOptimizerConfig,
    cvar_config: CvarRiskConfig,
) -> pd.DataFrame:
    """Audit CVaR-specific portfolio diagnostics."""
    checks = [
        (
            "finite_cvar",
            int((~np.isfinite(diagnostics["cvar_loss"].to_numpy(dtype=float))).sum()),
            "CVaR estimates must be finite.",
        ),
        (
            "cvar_not_below_var",
            int(
                (
                    diagnostics["cvar_loss"] + portfolio_config.weight_tolerance
                    < diagnostics["var_loss"]
                ).sum()
            ),
            "CVaR loss must not be below the corresponding VaR loss.",
        ),
        (
            "minimum_scenarios",
            int(diagnostics["scenario_count"].lt(cvar_config.minimum_scenarios).sum()),
            "Every optimization must use enough historical scenarios.",
        ),
        (
            "portfolio_beta_lower_limit",
            int(
                diagnostics["portfolio_beta_vs_spy"]
                .lt(portfolio_config.minimum_portfolio_beta - portfolio_config.weight_tolerance)
                .sum()
            ),
            "CVaR portfolio beta must remain above its lower bound.",
        ),
        (
            "portfolio_beta_upper_limit",
            int(
                diagnostics["portfolio_beta_vs_spy"]
                .gt(portfolio_config.maximum_portfolio_beta + portfolio_config.weight_tolerance)
                .sum()
            ),
            "CVaR portfolio beta must remain below its upper bound.",
        ),
        (
            "position_liquidity_limit",
            int(
                diagnostics["maximum_position_adv_fraction"]
                .gt(portfolio_config.max_position_adv_fraction + portfolio_config.weight_tolerance)
                .sum()
            ),
            "CVaR positions must respect the configured liquidity limit.",
        ),
    ]

    return pd.DataFrame(
        [
            {
                "check": name,
                "status": ("PASS" if violations == 0 else "FAIL"),
                "violations": violations,
                "description": description,
            }
            for (
                name,
                violations,
                description,
            ) in checks
        ]
    )
