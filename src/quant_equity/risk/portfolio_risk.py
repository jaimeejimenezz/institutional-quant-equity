"""Portfolio-level risk, concentration, sector and liquidity analytics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class PortfolioRiskError(ValueError):
    """Raised when portfolio risk analytics cannot be calculated."""


@dataclass(frozen=True)
class PortfolioRiskConfig:
    """Configuration for portfolio-level risk analytics."""

    portfolio_value: float = 1_000_000.0
    max_daily_adv_participation: float = 0.10
    weight_sum_tolerance: float = 1e-8

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> PortfolioRiskConfig:
        """Build configuration from project settings."""
        return cls(
            portfolio_value=float(
                values.get(
                    "reference_portfolio_value",
                    1_000_000.0,
                )
            ),
            max_daily_adv_participation=float(
                values.get(
                    "max_daily_adv_participation",
                    0.10,
                )
            ),
            weight_sum_tolerance=float(
                values.get(
                    "weight_sum_tolerance",
                    1e-8,
                )
            ),
        )

    def validate(self) -> None:
        """Validate portfolio-risk configuration."""
        if self.portfolio_value <= 0.0:
            raise PortfolioRiskError("portfolio_value must be positive.")

        if not (0.0 < self.max_daily_adv_participation <= 1.0):
            raise PortfolioRiskError("max_daily_adv_participation must be between zero and one.")

        if self.weight_sum_tolerance <= 0.0:
            raise PortfolioRiskError("weight_sum_tolerance must be positive.")


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require a set of columns to be present."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise PortfolioRiskError(f"{dataset_name} is missing columns: " + ", ".join(missing) + ".")


def build_top_n_equal_weights(
    final_signal: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    """Build transparent equal weights for the highest-ranked securities."""
    _require_columns(
        final_signal,
        (
            "as_of_date",
            "ticker",
            "rank",
        ),
        dataset_name="final alpha signal",
    )

    if top_n < 1:
        raise PortfolioRiskError("top_n must be positive.")

    signal = final_signal.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "rank",
        ],
    ].copy()

    signal["as_of_date"] = pd.to_datetime(signal["as_of_date"]).dt.normalize()

    signal["ticker"] = signal["ticker"].astype(str)

    if signal.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    ).any():
        raise PortfolioRiskError("Final alpha signal contains duplicate keys.")

    cross_section_sizes = signal.groupby("as_of_date")["ticker"].nunique()

    if (cross_section_sizes < top_n).any():
        raise PortfolioRiskError("top_n exceeds the available cross-section.")

    selected = signal.loc[signal["rank"].le(top_n)].copy()

    selected_counts = selected.groupby("as_of_date")["ticker"].size()

    if not selected_counts.eq(top_n).all():
        raise PortfolioRiskError("Each date must contain exactly top_n selected securities.")

    selected["weight"] = 1.0 / float(top_n)

    return (
        selected.loc[
            :,
            [
                "as_of_date",
                "ticker",
                "weight",
            ],
        ]
        .sort_values(
            [
                "as_of_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )


def _validate_portfolio_weights(
    weights: pd.DataFrame,
    *,
    tolerance: float,
) -> pd.DataFrame:
    """Normalize and validate long-only fully-invested portfolio weights."""
    _require_columns(
        weights,
        (
            "as_of_date",
            "ticker",
            "weight",
        ),
        dataset_name="portfolio weights",
    )

    data = weights.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "weight",
        ],
    ].copy()

    data["as_of_date"] = pd.to_datetime(data["as_of_date"]).dt.normalize()

    data["ticker"] = data["ticker"].astype(str)

    data["weight"] = pd.to_numeric(
        data["weight"],
        errors="coerce",
    )

    if data.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    ).any():
        raise PortfolioRiskError("Portfolio weights contain duplicate keys.")

    if data["weight"].isna().any():
        raise PortfolioRiskError("Portfolio weights contain missing values.")

    if not np.isfinite(data["weight"].to_numpy(dtype=float)).all():
        raise PortfolioRiskError("Portfolio weights contain non-finite values.")

    if data["weight"].lt(-tolerance).any():
        raise PortfolioRiskError("Portfolio weights must be long-only.")

    weight_sums = data.groupby("as_of_date")["weight"].sum()

    if not np.allclose(
        weight_sums.to_numpy(dtype=float),
        1.0,
        atol=tolerance,
        rtol=0.0,
    ):
        raise PortfolioRiskError("Portfolio weights must sum to one on every date.")

    return data


def calculate_portfolio_risk(
    weights: pd.DataFrame,
    risk_estimates: pd.DataFrame,
    covariance_matrices: pd.DataFrame,
    *,
    config: PortfolioRiskConfig | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Calculate portfolio, security and sector risk analytics."""
    if config is None:
        config = PortfolioRiskConfig()

    config.validate()

    portfolio_weights = _validate_portfolio_weights(
        weights,
        tolerance=(config.weight_sum_tolerance),
    )

    _require_columns(
        risk_estimates,
        (
            "as_of_date",
            "ticker",
            "sector",
            "annualized_volatility",
            "beta_vs_spy",
            "average_dollar_volume",
        ),
        dataset_name="risk estimates",
    )

    _require_columns(
        covariance_matrices,
        (
            "as_of_date",
            "ticker_a",
            "ticker_b",
            "annualized_covariance",
        ),
        dataset_name="covariance matrices",
    )

    estimates = risk_estimates.copy()
    covariance = covariance_matrices.copy()

    estimates["as_of_date"] = pd.to_datetime(estimates["as_of_date"]).dt.normalize()

    covariance["as_of_date"] = pd.to_datetime(covariance["as_of_date"]).dt.normalize()

    estimates["ticker"] = estimates["ticker"].astype(str)

    covariance["ticker_a"] = covariance["ticker_a"].astype(str)

    covariance["ticker_b"] = covariance["ticker_b"].astype(str)

    summary_rows: list[dict[str, Any]] = []

    contribution_rows: list[dict[str, Any]] = []

    sector_rows: list[dict[str, Any]] = []

    for (
        as_of_date,
        date_weights,
    ) in portfolio_weights.groupby(
        "as_of_date",
        sort=True,
    ):
        date_estimates = estimates.loc[estimates["as_of_date"].eq(as_of_date)].copy()

        date_covariance = covariance.loc[covariance["as_of_date"].eq(as_of_date)].copy()

        if date_estimates.empty:
            raise PortfolioRiskError(f"Risk estimates are missing for {as_of_date.date()}.")

        if date_covariance.empty:
            raise PortfolioRiskError(f"Covariance matrix is missing for {as_of_date.date()}.")

        tickers = sorted(date_weights["ticker"].tolist())

        metadata = (
            date_weights.merge(
                date_estimates[
                    [
                        "ticker",
                        "sector",
                        "annualized_volatility",
                        "beta_vs_spy",
                        "average_dollar_volume",
                    ]
                ],
                on="ticker",
                how="left",
                validate="one_to_one",
            )
            .set_index("ticker")
            .reindex(tickers)
        )

        required_metadata = [
            "weight",
            "sector",
            "annualized_volatility",
            "beta_vs_spy",
            "average_dollar_volume",
        ]

        if metadata[required_metadata].isna().any().any():
            raise PortfolioRiskError("Portfolio securities are missing risk estimates.")

        covariance_wide = date_covariance.pivot(
            index="ticker_a",
            columns="ticker_b",
            values="annualized_covariance",
        ).reindex(
            index=tickers,
            columns=tickers,
        )

        if covariance_wide.isna().any().any():
            raise PortfolioRiskError("Portfolio covariance matrix is incomplete.")

        covariance_values = covariance_wide.to_numpy(dtype=float)

        weight_values = metadata["weight"].to_numpy(dtype=float)

        portfolio_variance = float(weight_values @ covariance_values @ weight_values)

        if portfolio_variance < -1e-10:
            raise PortfolioRiskError("Portfolio variance is negative.")

        portfolio_variance = max(
            portfolio_variance,
            0.0,
        )

        portfolio_volatility = float(np.sqrt(portfolio_variance))

        if portfolio_volatility <= 0.0:
            raise PortfolioRiskError("Portfolio volatility must be positive.")

        covariance_times_weights = covariance_values @ weight_values

        marginal_risk = covariance_times_weights / portfolio_volatility

        component_risk = weight_values * marginal_risk

        risk_contribution_share = component_risk / portfolio_volatility

        portfolio_beta = float(
            np.dot(
                weight_values,
                metadata["beta_vs_spy"].to_numpy(dtype=float),
            )
        )

        hhi = float(np.square(weight_values).sum())

        effective_positions = float(1.0 / hhi)

        position_notional = weight_values * config.portfolio_value

        average_dollar_volume = metadata["average_dollar_volume"].to_numpy(dtype=float)

        position_adv_fraction = position_notional / average_dollar_volume

        liquidation_days = position_notional / (
            config.max_daily_adv_participation * average_dollar_volume
        )

        for index, ticker in enumerate(tickers):
            contribution_rows.append(
                {
                    "as_of_date": (pd.Timestamp(as_of_date)),
                    "ticker": ticker,
                    "sector": metadata.loc[
                        ticker,
                        "sector",
                    ],
                    "weight": float(weight_values[index]),
                    "annualized_volatility": float(
                        metadata.loc[
                            ticker,
                            "annualized_volatility",
                        ]
                    ),
                    "beta_vs_spy": float(
                        metadata.loc[
                            ticker,
                            "beta_vs_spy",
                        ]
                    ),
                    "marginal_risk": float(marginal_risk[index]),
                    "component_risk": float(component_risk[index]),
                    "risk_contribution_share": float(risk_contribution_share[index]),
                    "position_notional": float(position_notional[index]),
                    "average_dollar_volume": float(average_dollar_volume[index]),
                    "position_adv_fraction": float(position_adv_fraction[index]),
                    "liquidation_days": float(liquidation_days[index]),
                }
            )

        contribution_frame = pd.DataFrame(contribution_rows)

        current_contributions = contribution_frame.loc[
            contribution_frame["as_of_date"].eq(as_of_date)
        ]

        sector_exposure = current_contributions.groupby(
            "sector",
            as_index=False,
        ).agg(
            portfolio_weight=(
                "weight",
                "sum",
            ),
            component_risk=(
                "component_risk",
                "sum",
            ),
            risk_contribution_share=(
                "risk_contribution_share",
                "sum",
            ),
            positions=(
                "ticker",
                "nunique",
            ),
        )

        universe_sector_weights = (
            date_estimates.groupby("sector")["ticker"]
            .nunique()
            .div(date_estimates["ticker"].nunique())
        )

        sector_exposure["universe_equal_weight"] = (
            sector_exposure["sector"].map(universe_sector_weights).fillna(0.0)
        )

        sector_exposure["active_weight"] = (
            sector_exposure["portfolio_weight"] - sector_exposure["universe_equal_weight"]
        )

        sector_exposure["as_of_date"] = pd.Timestamp(as_of_date)

        sector_rows.extend(
            sector_exposure[
                [
                    "as_of_date",
                    "sector",
                    "portfolio_weight",
                    "universe_equal_weight",
                    "active_weight",
                    "positions",
                    "component_risk",
                    "risk_contribution_share",
                ]
            ].to_dict(orient="records")
        )

        summary_rows.append(
            {
                "as_of_date": (pd.Timestamp(as_of_date)),
                "positions": int(len(tickers)),
                "portfolio_value": float(config.portfolio_value),
                "predicted_volatility": (portfolio_volatility),
                "predicted_variance": (portfolio_variance),
                "portfolio_beta_vs_spy": (portfolio_beta),
                "maximum_weight": float(weight_values.max()),
                "minimum_weight": float(weight_values.min()),
                "concentration_hhi": (hhi),
                "effective_positions": (effective_positions),
                "maximum_sector_weight": float(sector_exposure["portfolio_weight"].max()),
                "maximum_active_sector_weight": float(sector_exposure["active_weight"].abs().max()),
                "maximum_position_adv_fraction": float(position_adv_fraction.max()),
                "weighted_position_adv_fraction": float(
                    np.dot(
                        weight_values,
                        position_adv_fraction,
                    )
                ),
                "maximum_liquidation_days": float(liquidation_days.max()),
                "weighted_liquidation_days": float(
                    np.dot(
                        weight_values,
                        liquidation_days,
                    )
                ),
                "risk_contribution_sum": float(component_risk.sum()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("as_of_date").reset_index(drop=True)

    contributions = (
        pd.DataFrame(contribution_rows)
        .sort_values(
            [
                "as_of_date",
                "risk_contribution_share",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    sectors = (
        pd.DataFrame(sector_rows)
        .sort_values(
            [
                "as_of_date",
                "portfolio_weight",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return (
        summary,
        contributions,
        sectors,
    )


def validate_portfolio_risk(
    summary: pd.DataFrame,
    contributions: pd.DataFrame,
    sectors: pd.DataFrame,
    *,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Run readiness checks on portfolio-level risk analytics."""
    _require_columns(
        summary,
        (
            "as_of_date",
            "predicted_volatility",
            "portfolio_beta_vs_spy",
            "concentration_hhi",
            "effective_positions",
            "risk_contribution_sum",
        ),
        dataset_name="portfolio risk summary",
    )

    _require_columns(
        contributions,
        (
            "as_of_date",
            "ticker",
            "weight",
            "component_risk",
            "risk_contribution_share",
            "liquidation_days",
        ),
        dataset_name="risk contributions",
    )

    _require_columns(
        sectors,
        (
            "as_of_date",
            "sector",
            "portfolio_weight",
            "risk_contribution_share",
        ),
        dataset_name="sector exposures",
    )

    contribution_weight_sums = contributions.groupby("as_of_date")["weight"].sum()

    contribution_risk_sums = contributions.groupby("as_of_date")["risk_contribution_share"].sum()

    sector_weight_sums = sectors.groupby("as_of_date")["portfolio_weight"].sum()

    risk_identity = summary["risk_contribution_sum"].to_numpy(dtype=float)

    volatility = summary["predicted_volatility"].to_numpy(dtype=float)

    checks = [
        (
            "unique_summary_dates",
            int(summary["as_of_date"].duplicated().sum()),
            ("Portfolio summary must contain one row per date."),
        ),
        (
            "unique_contribution_keys",
            int(
                contributions.duplicated(
                    [
                        "as_of_date",
                        "ticker",
                    ]
                ).sum()
            ),
            ("Security risk contributions must have unique date-ticker keys."),
        ),
        (
            "weights_sum_to_one",
            int((np.abs(contribution_weight_sums.to_numpy() - 1.0) > tolerance).sum()),
            ("Security weights must sum to one on every date."),
        ),
        (
            "sector_weights_sum_to_one",
            int((np.abs(sector_weight_sums.to_numpy() - 1.0) > tolerance).sum()),
            ("Sector weights must sum to one on every date."),
        ),
        (
            "risk_contributions_sum_to_one",
            int((np.abs(contribution_risk_sums.to_numpy() - 1.0) > tolerance).sum()),
            ("Security risk contribution shares must sum to one."),
        ),
        (
            "euler_risk_identity",
            int((np.abs(risk_identity - volatility) > tolerance).sum()),
            ("Component risk contributions must reconstruct portfolio volatility."),
        ),
        (
            "positive_portfolio_volatility",
            int(summary["predicted_volatility"].le(0.0).sum()),
            ("Predicted portfolio volatility must be positive."),
        ),
        (
            "positive_effective_positions",
            int(summary["effective_positions"].le(0.0).sum()),
            ("Effective position count must be positive."),
        ),
        (
            "non_negative_liquidation_days",
            int(contributions["liquidation_days"].lt(0.0).sum()),
            ("Estimated liquidation days must be non-negative."),
        ),
        (
            "finite_summary_values",
            int(
                (
                    ~np.isfinite(
                        summary[
                            [
                                "predicted_volatility",
                                "portfolio_beta_vs_spy",
                                "concentration_hhi",
                                "effective_positions",
                            ]
                        ].to_numpy(dtype=float)
                    )
                ).sum()
            ),
            ("Core portfolio risk statistics must be finite."),
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
