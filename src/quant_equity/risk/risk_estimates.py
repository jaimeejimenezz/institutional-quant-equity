"""Point-in-time security-level risk and liquidity estimates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class RiskEstimateError(ValueError):
    """Raised when security-level risk estimates cannot be constructed."""


@dataclass(frozen=True)
class RiskEstimateConfig:
    """Configuration for monthly security-level risk estimates."""

    volatility_window_sessions: int = 252
    beta_window_sessions: int = 252
    liquidity_window_sessions: int = 60
    minimum_return_observations: int = 126
    minimum_liquidity_observations: int = 20
    annualization_factor: int = 252

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> RiskEstimateConfig:
        """Build configuration from a mapping."""
        return cls(
            volatility_window_sessions=int(
                values.get(
                    "volatility_window_sessions",
                    252,
                )
            ),
            beta_window_sessions=int(
                values.get(
                    "beta_window_sessions",
                    252,
                )
            ),
            liquidity_window_sessions=int(
                values.get(
                    "liquidity_window_sessions",
                    60,
                )
            ),
            minimum_return_observations=int(
                values.get(
                    "minimum_return_observations",
                    126,
                )
            ),
            minimum_liquidity_observations=int(
                values.get(
                    "minimum_liquidity_observations",
                    20,
                )
            ),
            annualization_factor=int(
                values.get(
                    "annualization_factor",
                    252,
                )
            ),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        positive_fields = {
            "volatility_window_sessions": (self.volatility_window_sessions),
            "beta_window_sessions": (self.beta_window_sessions),
            "liquidity_window_sessions": (self.liquidity_window_sessions),
            "minimum_return_observations": (self.minimum_return_observations),
            "minimum_liquidity_observations": (self.minimum_liquidity_observations),
            "annualization_factor": (self.annualization_factor),
        }

        for name, value in positive_fields.items():
            if value < 1:
                raise RiskEstimateError(f"{name} must be positive.")

        if self.minimum_return_observations > min(
            self.volatility_window_sessions,
            self.beta_window_sessions,
        ):
            raise RiskEstimateError("minimum_return_observations cannot exceed the return windows.")

        if self.minimum_liquidity_observations > self.liquidity_window_sessions:
            raise RiskEstimateError(
                "minimum_liquidity_observations cannot exceed the liquidity window."
            )


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require a set of columns to be present."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise RiskEstimateError(f"{dataset_name} is missing columns: " + ", ".join(missing) + ".")


def _prepare_market_history(
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize security market data and calculate daily returns."""
    _require_columns(
        market_data,
        (
            "date",
            "ticker",
            "close",
            "adjusted_close",
            "volume",
        ),
        dataset_name="market data",
    )

    data = market_data.loc[
        :,
        [
            "date",
            "ticker",
            "close",
            "adjusted_close",
            "volume",
        ],
    ].copy()

    data["date"] = pd.to_datetime(data["date"]).dt.normalize()

    data["ticker"] = data["ticker"].astype(str)

    for column in (
        "close",
        "adjusted_close",
        "volume",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data.duplicated(
        [
            "date",
            "ticker",
        ]
    ).any():
        raise RiskEstimateError("Market data contain duplicate date-ticker rows.")

    if (
        data[
            [
                "date",
                "ticker",
                "close",
                "adjusted_close",
                "volume",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise RiskEstimateError("Market data contain missing required values.")

    if (
        data[
            [
                "close",
                "adjusted_close",
            ]
        ]
        .le(0.0)
        .any()
        .any()
    ):
        raise RiskEstimateError("Market data contain non-positive prices.")

    if data["volume"].lt(0.0).any():
        raise RiskEstimateError("Market data contain negative volume.")

    data = data.sort_values(
        [
            "ticker",
            "date",
        ]
    ).reset_index(drop=True)

    data["daily_return"] = data.groupby(
        "ticker",
        sort=False,
    )["adjusted_close"].pct_change(fill_method=None)

    data["dollar_volume"] = data["close"] * data["volume"]

    return data


def _prepare_spy_history(
    spy_data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize SPY market data and calculate daily returns."""
    _require_columns(
        spy_data,
        (
            "date",
            "adjusted_close",
        ),
        dataset_name="SPY data",
    )

    data = spy_data.loc[
        :,
        [
            "date",
            "adjusted_close",
        ],
    ].copy()

    data["date"] = pd.to_datetime(data["date"]).dt.normalize()

    data["adjusted_close"] = pd.to_numeric(
        data["adjusted_close"],
        errors="coerce",
    )

    if data["date"].duplicated().any():
        raise RiskEstimateError("SPY data contain duplicate dates.")

    if data.isna().any().any():
        raise RiskEstimateError("SPY data contain missing required values.")

    if data["adjusted_close"].le(0.0).any():
        raise RiskEstimateError("SPY data contain non-positive prices.")

    data = data.sort_values("date").reset_index(drop=True)

    data["spy_return"] = data["adjusted_close"].pct_change(fill_method=None)

    return data


def _downside_volatility(
    returns: pd.Series,
    *,
    annualization_factor: int,
) -> float:
    """Calculate annualized downside deviation relative to zero."""
    values = returns.to_numpy(dtype=float)

    downside = np.minimum(
        values,
        0.0,
    )

    return float(np.sqrt(np.mean(np.square(downside)) * annualization_factor))


def build_risk_estimates(
    market_data: pd.DataFrame,
    spy_data: pd.DataFrame,
    signal_universe: pd.DataFrame,
    *,
    config: RiskEstimateConfig | None = None,
) -> pd.DataFrame:
    """Build point-in-time monthly security risk and liquidity estimates."""
    if config is None:
        config = RiskEstimateConfig()

    config.validate()

    _require_columns(
        signal_universe,
        (
            "as_of_date",
            "ticker",
            "sector",
        ),
        dataset_name="signal universe",
    )

    signal = signal_universe.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "sector",
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
        raise RiskEstimateError("Signal universe contains duplicate date-ticker rows.")

    analysis_end_date = signal["as_of_date"].max()

    market_dates = pd.to_datetime(market_data["date"]).dt.normalize()

    spy_dates = pd.to_datetime(spy_data["date"]).dt.normalize()

    relevant_market_data = market_data.loc[market_dates.le(analysis_end_date)].copy()

    relevant_spy_data = spy_data.loc[spy_dates.le(analysis_end_date)].copy()

    market = _prepare_market_history(relevant_market_data)

    spy = _prepare_spy_history(relevant_spy_data)

    market_by_ticker = {
        ticker: group.reset_index(drop=True)
        for ticker, group in market.groupby(
            "ticker",
            sort=False,
        )
    }

    spy_returns = (
        spy.loc[
            :,
            [
                "date",
                "spy_return",
            ],
        ]
        .dropna()
        .copy()
    )

    annualization_scale = float(np.sqrt(config.annualization_factor))

    rows: list[dict[str, Any]] = []

    for (
        ticker,
        ticker_signal,
    ) in signal.groupby(
        "ticker",
        sort=True,
    ):
        if ticker not in market_by_ticker:
            raise RiskEstimateError(f"Market history is missing ticker {ticker}.")

        ticker_history = market_by_ticker[ticker]

        aligned_returns = (
            ticker_history.loc[
                :,
                [
                    "date",
                    "daily_return",
                ],
            ]
            .merge(
                spy_returns,
                on="date",
                how="inner",
                validate="one_to_one",
            )
            .dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )

        for observation in ticker_signal.sort_values("as_of_date").itertuples(index=False):
            as_of_date = pd.Timestamp(observation.as_of_date)

            available_history = ticker_history.loc[ticker_history["date"].le(as_of_date)]

            if available_history.empty:
                raise RiskEstimateError(
                    f"No market history is available for {ticker} at {as_of_date.date()}."
                )

            return_window = (
                available_history.loc[
                    available_history["daily_return"].notna(),
                    [
                        "date",
                        "daily_return",
                    ],
                ]
                .tail(config.volatility_window_sessions)
                .copy()
            )

            if len(return_window) < config.minimum_return_observations:
                raise RiskEstimateError(
                    f"Insufficient return history for {ticker} at {as_of_date.date()}."
                )

            beta_window = (
                aligned_returns.loc[aligned_returns["date"].le(as_of_date)]
                .tail(config.beta_window_sessions)
                .copy()
            )

            if len(beta_window) < config.minimum_return_observations:
                raise RiskEstimateError(
                    f"Insufficient SPY-aligned history for {ticker} at {as_of_date.date()}."
                )

            liquidity_window = available_history.tail(config.liquidity_window_sessions)

            if len(liquidity_window) < config.minimum_liquidity_observations:
                raise RiskEstimateError(
                    f"Insufficient liquidity history for {ticker} at {as_of_date.date()}."
                )

            asset_returns = return_window["daily_return"].astype(float)

            spy_returns_window = beta_window["spy_return"].astype(float)

            asset_beta_returns = beta_window["daily_return"].astype(float)

            market_variance = float(spy_returns_window.var(ddof=1))

            if not np.isfinite(market_variance) or market_variance <= 0.0:
                raise RiskEstimateError(f"SPY variance is invalid at {as_of_date.date()}.")

            covariance = float(
                np.cov(
                    asset_beta_returns.to_numpy(),
                    spy_returns_window.to_numpy(),
                    ddof=1,
                )[
                    0,
                    1,
                ]
            )

            beta = covariance / market_variance

            correlation = float(asset_beta_returns.corr(spy_returns_window))

            annualized_volatility = float(asset_returns.std(ddof=1) * annualization_scale)

            annualized_downside_volatility = _downside_volatility(
                asset_returns,
                annualization_factor=(config.annualization_factor),
            )

            latest_row = available_history.iloc[-1]

            rows.append(
                {
                    "as_of_date": (as_of_date),
                    "ticker": ticker,
                    "sector": (observation.sector),
                    "annualized_volatility": (annualized_volatility),
                    "annualized_downside_volatility": (annualized_downside_volatility),
                    "beta_vs_spy": float(beta),
                    "correlation_vs_spy": (correlation),
                    "average_dollar_volume": float(liquidity_window["dollar_volume"].mean()),
                    "median_dollar_volume": float(liquidity_window["dollar_volume"].median()),
                    "latest_close": float(latest_row["close"]),
                    "latest_volume": float(latest_row["volume"]),
                    "return_observations": int(len(return_window)),
                    "beta_observations": int(len(beta_window)),
                    "liquidity_observations": int(len(liquidity_window)),
                    "risk_window_start_date": (pd.Timestamp(return_window["date"].min())),
                    "risk_window_end_date": (pd.Timestamp(return_window["date"].max())),
                    "latest_market_date": (pd.Timestamp(latest_row["date"])),
                    "latest_spy_date": (pd.Timestamp(beta_window["date"].max())),
                }
            )

    result = pd.DataFrame(rows)

    numeric_columns = [
        "annualized_volatility",
        "annualized_downside_volatility",
        "beta_vs_spy",
        "correlation_vs_spy",
        "average_dollar_volume",
        "median_dollar_volume",
        "latest_close",
        "latest_volume",
    ]

    if not np.isfinite(result[numeric_columns].to_numpy(dtype=float)).all():
        raise RiskEstimateError("Risk estimates contain non-finite values.")

    return result.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)


def validate_risk_estimates(
    estimates: pd.DataFrame,
    signal_universe: pd.DataFrame,
    *,
    config: RiskEstimateConfig | None = None,
) -> pd.DataFrame:
    """Return deterministic readiness checks for stored risk estimates."""
    if config is None:
        config = RiskEstimateConfig()

    config.validate()

    _require_columns(
        estimates,
        (
            "as_of_date",
            "ticker",
            "sector",
            "annualized_volatility",
            "annualized_downside_volatility",
            "beta_vs_spy",
            "correlation_vs_spy",
            "average_dollar_volume",
            "median_dollar_volume",
            "latest_close",
            "latest_volume",
            "return_observations",
            "beta_observations",
            "liquidity_observations",
            "risk_window_start_date",
            "risk_window_end_date",
            "latest_market_date",
            "latest_spy_date",
        ),
        dataset_name="risk estimates",
    )

    estimates = estimates.copy()

    signal = signal_universe.loc[
        :,
        [
            "as_of_date",
            "ticker",
        ],
    ].copy()

    for frame in (
        estimates,
        signal,
    ):
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.normalize()

        frame["ticker"] = frame["ticker"].astype(str)

    date_columns = (
        "risk_window_start_date",
        "risk_window_end_date",
        "latest_market_date",
        "latest_spy_date",
    )

    for column in date_columns:
        estimates[column] = pd.to_datetime(estimates[column]).dt.normalize()

    expected_keys = signal.drop_duplicates()

    observed_keys = estimates.loc[
        :,
        [
            "as_of_date",
            "ticker",
        ],
    ].drop_duplicates()

    key_comparison = expected_keys.merge(
        observed_keys,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="outer",
        indicator=True,
    )

    coverage_violations = int(key_comparison["_merge"].ne("both").sum())

    numeric_columns = [
        "annualized_volatility",
        "annualized_downside_volatility",
        "beta_vs_spy",
        "correlation_vs_spy",
        "average_dollar_volume",
        "median_dollar_volume",
        "latest_close",
        "latest_volume",
    ]

    checks = [
        (
            "unique_keys",
            int(
                estimates.duplicated(
                    [
                        "as_of_date",
                        "ticker",
                    ]
                ).sum()
            ),
            ("Risk estimates must have one row per date and ticker."),
        ),
        (
            "signal_coverage",
            coverage_violations,
            ("Risk estimates must exactly match the final signal universe."),
        ),
        (
            "market_point_in_time",
            int(estimates["latest_market_date"].gt(estimates["as_of_date"]).sum()),
            ("Market observations must not extend beyond as_of_date."),
        ),
        (
            "spy_point_in_time",
            int(estimates["latest_spy_date"].gt(estimates["as_of_date"]).sum()),
            ("SPY observations must not extend beyond as_of_date."),
        ),
        (
            "risk_window_point_in_time",
            int(estimates["risk_window_end_date"].gt(estimates["as_of_date"]).sum()),
            ("Risk estimation windows must end on or before as_of_date."),
        ),
        (
            "return_history",
            int(estimates["return_observations"].lt(config.minimum_return_observations).sum()),
            ("Every row must have enough trailing returns."),
        ),
        (
            "beta_history",
            int(estimates["beta_observations"].lt(config.minimum_return_observations).sum()),
            ("Every row must have enough SPY-aligned returns."),
        ),
        (
            "liquidity_history",
            int(
                estimates["liquidity_observations"].lt(config.minimum_liquidity_observations).sum()
            ),
            ("Every row must have enough liquidity observations."),
        ),
        (
            "finite_numeric_values",
            int((~np.isfinite(estimates[numeric_columns].to_numpy(dtype=float))).sum()),
            ("Stored risk and liquidity values must be finite."),
        ),
        (
            "non_negative_risk",
            int(
                estimates[
                    [
                        "annualized_volatility",
                        "annualized_downside_volatility",
                    ]
                ]
                .lt(0.0)
                .sum()
                .sum()
            ),
            ("Volatility estimates must be non-negative."),
        ),
        (
            "positive_liquidity_and_price",
            int(
                estimates[
                    [
                        "average_dollar_volume",
                        "median_dollar_volume",
                        "latest_close",
                    ]
                ]
                .le(0.0)
                .sum()
                .sum()
            ),
            ("Liquidity and price measures must be positive."),
        ),
        (
            "valid_market_correlation",
            int(estimates["correlation_vs_spy"].abs().gt(1.0 + 1e-12).sum()),
            ("Correlation with SPY must remain between -1 and 1."),
        ),
    ]

    return pd.DataFrame(
        [
            {
                "check": name,
                "status": ("PASS" if violations == 0 else "FAIL"),
                "violations": (violations),
                "description": (description),
            }
            for (
                name,
                violations,
                description,
            ) in checks
        ]
    )
