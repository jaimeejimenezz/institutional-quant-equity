"""Point-in-time technical feature construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_MARKET_COLUMNS = (
    "date",
    "ticker",
    "close",
    "adjusted_close",
    "volume",
)

TECHNICAL_FEATURE_COLUMNS = (
    "momentum_12_1",
    "momentum_6_1",
    "return_3m",
    "return_1m",
    "return_1w",
    "reversal_1m",
    "volatility_20d",
    "volatility_60d",
    "downside_volatility_60d",
    "beta_60d_market",
    "max_drawdown_126d",
    "distance_sma_50d",
    "distance_sma_200d",
    "sma_50_200_spread",
    "positive_day_ratio_60d",
    "average_dollar_volume_20d",
    "dollar_volume_change_20d_60d",
    "amihud_illiquidity_20d",
    "zero_volume_ratio_60d",
)

TECHNICAL_PANEL_COLUMNS = (
    "as_of_date",
    "ticker",
    "latest_market_date",
    "observations_available",
    *TECHNICAL_FEATURE_COLUMNS,
)


class TechnicalFeatureError(ValueError):
    """Raised when technical features cannot be constructed."""


@dataclass(frozen=True)
class TechnicalFeatureConfig:
    """Window configuration for the first technical feature set."""

    annualization_sessions: int = 252
    momentum_12m_sessions: int = 252
    momentum_6m_sessions: int = 126
    skip_recent_sessions: int = 21
    return_3m_sessions: int = 63
    return_1m_sessions: int = 21
    return_1w_sessions: int = 5
    volatility_short_sessions: int = 20
    volatility_long_sessions: int = 60
    drawdown_sessions: int = 126
    moving_average_short_sessions: int = 50
    moving_average_long_sessions: int = 200
    liquidity_short_sessions: int = 20
    liquidity_long_sessions: int = 60

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> TechnicalFeatureConfig:
        """Build settings from the technical-features YAML block."""
        config = cls(**{field.name: int(values[field.name]) for field in fields(cls)})

        config.validate()

        return config

    def validate(self) -> None:
        """Validate the rolling-window decisions."""
        invalid = [
            field.name
            for field in fields(self)
            if getattr(
                self,
                field.name,
            )
            < 1
        ]

        if invalid:
            raise TechnicalFeatureError(
                "Technical feature windows must be positive: " + ", ".join(sorted(invalid)) + "."
            )

        if self.momentum_12m_sessions <= self.skip_recent_sessions:
            raise TechnicalFeatureError("momentum_12m_sessions must exceed skip_recent_sessions.")

        if self.momentum_6m_sessions <= self.skip_recent_sessions:
            raise TechnicalFeatureError("momentum_6m_sessions must exceed skip_recent_sessions.")

        if self.moving_average_long_sessions <= self.moving_average_short_sessions:
            raise TechnicalFeatureError(
                "The long moving average must exceed the short moving average."
            )

        if self.liquidity_long_sessions <= self.liquidity_short_sessions:
            raise TechnicalFeatureError(
                "The long liquidity window must exceed the short liquidity window."
            )


def _prepare_market_data(
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and prepare daily market observations."""
    missing = sorted(set(REQUIRED_MARKET_COLUMNS).difference(market_data.columns))

    if missing:
        raise TechnicalFeatureError(
            "Market data is missing required columns: " + ", ".join(missing) + "."
        )

    if market_data.empty:
        raise TechnicalFeatureError("Market data is empty.")

    data = market_data.loc[
        :,
        REQUIRED_MARKET_COLUMNS,
    ].copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    ).dt.normalize()

    data["ticker"] = data["ticker"].astype("string").str.strip().str.upper()

    for column in (
        "close",
        "adjusted_close",
        "volume",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    invalid_identifier_rows = data["date"].isna() | data["ticker"].isna() | data["ticker"].eq("")

    if invalid_identifier_rows.any():
        raise TechnicalFeatureError(
            "Market data contains "
            f"{int(invalid_identifier_rows.sum())} rows "
            "with invalid dates or tickers."
        )

    non_positive_price_rows = (data["close"].notna() & data["close"].le(0)) | (
        data["adjusted_close"].notna() & data["adjusted_close"].le(0)
    )

    if non_positive_price_rows.any():
        raise TechnicalFeatureError(
            "Market data contains "
            f"{int(non_positive_price_rows.sum())} rows "
            "with non-positive prices."
        )

    invalid_volume_rows = data["volume"].isna() | data["volume"].lt(0)

    if invalid_volume_rows.any():
        raise TechnicalFeatureError(
            f"Market data contains {int(invalid_volume_rows.sum())} rows with invalid volume."
        )

    missing_price_rows = (
        data[
            [
                "close",
                "adjusted_close",
            ]
        ]
        .isna()
        .any(axis=1)
    )

    data = data.loc[~missing_price_rows].copy()

    if data.empty:
        raise TechnicalFeatureError("Market data contains no valid price observations.")

    duplicate_rows = int(
        data.duplicated(
            [
                "date",
                "ticker",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_rows:
        raise TechnicalFeatureError(
            f"Market data contains {duplicate_rows} duplicated date-ticker rows."
        )

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

    data["market_return"] = data.groupby(
        "date",
        sort=False,
    )["daily_return"].transform("mean")

    return data


def _prepare_rebalance_dates(
    calendar: pd.DataFrame,
) -> pd.DatetimeIndex:
    """Validate and extract the monthly as-of dates."""
    if "as_of_date" not in calendar.columns:
        raise TechnicalFeatureError("Rebalance calendar is missing required column: as_of_date.")

    dates = pd.to_datetime(
        calendar["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    if dates.isna().any():
        raise TechnicalFeatureError("Rebalance calendar contains invalid dates.")

    duplicate_dates = int(dates.duplicated(keep=False).sum())

    if duplicate_dates:
        raise TechnicalFeatureError(
            f"Rebalance calendar contains {duplicate_dates} duplicated dates."
        )

    return pd.DatetimeIndex(dates.sort_values().to_numpy())


def _period_return(
    prices: pd.Series,
    start_lag: int,
    end_lag: int = 0,
) -> float:
    """Calculate a return between two historical lags."""
    if len(prices) < start_lag + 1:
        return np.nan

    start_price = float(prices.iloc[-(start_lag + 1)])

    end_price = float(prices.iloc[-(end_lag + 1)])

    return end_price / start_price - 1.0


def _complete_tail(
    values: pd.Series,
    window: int,
) -> pd.Series | None:
    """Return a complete fixed window or None."""
    selected = values.tail(window).dropna()

    if len(selected) != window:
        return None

    return selected


def _volatility(
    returns: pd.Series,
    window: int,
    annualization: int,
) -> float:
    """Calculate annualized sample volatility."""
    values = _complete_tail(
        returns,
        window,
    )

    if values is None:
        return np.nan

    return float(values.std(ddof=1) * np.sqrt(annualization))


def _downside_volatility(
    returns: pd.Series,
    window: int,
    annualization: int,
) -> float:
    """Calculate annualized downside volatility."""
    values = _complete_tail(
        returns,
        window,
    )

    if values is None:
        return np.nan

    downside = np.minimum(
        values.to_numpy(dtype=float),
        0.0,
    )

    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(annualization))


def _beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int,
) -> float:
    """Calculate beta against the equal-weight market proxy."""
    aligned = (
        pd.DataFrame(
            {
                "stock": stock_returns,
                "market": market_returns,
            }
        )
        .tail(window)
        .dropna()
    )

    if len(aligned) != window:
        return np.nan

    variance = float(aligned["market"].var(ddof=1))

    if not np.isfinite(variance) or variance <= 0:
        return np.nan

    covariance = float(aligned["stock"].cov(aligned["market"]))

    return covariance / variance


def _max_drawdown(
    prices: pd.Series,
    window: int,
) -> float:
    """Calculate the maximum drawdown in a price window."""
    values = prices.tail(window)

    if len(values) != window:
        return np.nan

    running_maximum = values.cummax()

    drawdown = values / running_maximum - 1.0

    return float(drawdown.min())


def _distance_to_sma(
    prices: pd.Series,
    window: int,
) -> float:
    """Calculate current-price distance from an SMA."""
    values = prices.tail(window)

    if len(values) != window:
        return np.nan

    moving_average = float(values.mean())

    return float(values.iloc[-1] / moving_average - 1.0)


def _sma_spread(
    prices: pd.Series,
    short_window: int,
    long_window: int,
) -> float:
    """Calculate the relative short-versus-long SMA spread."""
    if len(prices) < long_window:
        return np.nan

    short_average = float(prices.tail(short_window).mean())

    long_average = float(prices.tail(long_window).mean())

    return short_average / long_average - 1.0


def _positive_ratio(
    returns: pd.Series,
    window: int,
) -> float:
    """Calculate the proportion of positive sessions."""
    values = _complete_tail(
        returns,
        window,
    )

    if values is None:
        return np.nan

    return float(values.gt(0).mean())


def _window_mean(
    values: pd.Series,
    window: int,
) -> float:
    """Calculate a mean over a complete fixed window."""
    selected = _complete_tail(
        values,
        window,
    )

    if selected is None:
        return np.nan

    return float(selected.mean())


def _dollar_volume_change(
    values: pd.Series,
    short_window: int,
    long_window: int,
) -> float:
    """Compare recent dollar volume with preceding volume."""
    selected = _complete_tail(
        values,
        long_window,
    )

    if selected is None:
        return np.nan

    previous = selected.iloc[: long_window - short_window]

    previous_mean = float(previous.mean())

    if previous_mean <= 0:
        return np.nan

    recent_mean = float(selected.tail(short_window).mean())

    return recent_mean / previous_mean - 1.0


def _amihud(
    returns: pd.Series,
    dollar_volume: pd.Series,
    window: int,
) -> float:
    """Calculate Amihud's absolute-return illiquidity proxy."""
    aligned = (
        pd.DataFrame(
            {
                "return": returns,
                "dollar_volume": (dollar_volume),
            }
        )
        .tail(window)
        .dropna()
    )

    if len(aligned) != window:
        return np.nan

    if aligned["dollar_volume"].le(0).any():
        return np.nan

    ratio = aligned["return"].abs() / aligned["dollar_volume"]

    return float(ratio.mean())


def _zero_volume_ratio(
    volume: pd.Series,
    window: int,
) -> float:
    """Calculate the proportion of zero-volume sessions."""
    values = _complete_tail(
        volume,
        window,
    )

    if values is None:
        return np.nan

    return float(values.eq(0).mean())


def _feature_row(
    history: pd.DataFrame,
    as_of_date: pd.Timestamp,
    config: TechnicalFeatureConfig,
) -> dict[str, Any]:
    """Calculate all raw signals for one company and date."""
    prices = history["adjusted_close"]

    returns = history["daily_return"]

    dollar_volume = history["dollar_volume"]

    return_1m = _period_return(
        prices,
        config.return_1m_sessions,
    )

    return {
        "as_of_date": as_of_date,
        "ticker": str(history["ticker"].iloc[-1]),
        "latest_market_date": (history["date"].iloc[-1]),
        "observations_available": (len(history)),
        "momentum_12_1": _period_return(
            prices,
            config.momentum_12m_sessions,
            config.skip_recent_sessions,
        ),
        "momentum_6_1": _period_return(
            prices,
            config.momentum_6m_sessions,
            config.skip_recent_sessions,
        ),
        "return_3m": _period_return(
            prices,
            config.return_3m_sessions,
        ),
        "return_1m": return_1m,
        "return_1w": _period_return(
            prices,
            config.return_1w_sessions,
        ),
        "reversal_1m": (-return_1m if np.isfinite(return_1m) else np.nan),
        "volatility_20d": _volatility(
            returns,
            config.volatility_short_sessions,
            config.annualization_sessions,
        ),
        "volatility_60d": _volatility(
            returns,
            config.volatility_long_sessions,
            config.annualization_sessions,
        ),
        "downside_volatility_60d": (
            _downside_volatility(
                returns,
                config.volatility_long_sessions,
                config.annualization_sessions,
            )
        ),
        "beta_60d_market": _beta(
            returns,
            history["market_return"],
            config.volatility_long_sessions,
        ),
        "max_drawdown_126d": (
            _max_drawdown(
                prices,
                config.drawdown_sessions,
            )
        ),
        "distance_sma_50d": (
            _distance_to_sma(
                prices,
                config.moving_average_short_sessions,
            )
        ),
        "distance_sma_200d": (
            _distance_to_sma(
                prices,
                config.moving_average_long_sessions,
            )
        ),
        "sma_50_200_spread": (
            _sma_spread(
                prices,
                config.moving_average_short_sessions,
                config.moving_average_long_sessions,
            )
        ),
        "positive_day_ratio_60d": (
            _positive_ratio(
                returns,
                config.volatility_long_sessions,
            )
        ),
        "average_dollar_volume_20d": (
            _window_mean(
                dollar_volume,
                config.liquidity_short_sessions,
            )
        ),
        "dollar_volume_change_20d_60d": (
            _dollar_volume_change(
                dollar_volume,
                config.liquidity_short_sessions,
                config.liquidity_long_sessions,
            )
        ),
        "amihud_illiquidity_20d": (
            _amihud(
                returns,
                dollar_volume,
                config.liquidity_short_sessions,
            )
        ),
        "zero_volume_ratio_60d": (
            _zero_volume_ratio(
                history["volume"],
                config.liquidity_long_sessions,
            )
        ),
    }


def build_raw_technical_features(
    market_data: pd.DataFrame,
    rebalance_calendar: pd.DataFrame,
    *,
    feature_config: (TechnicalFeatureConfig | None) = None,
) -> pd.DataFrame:
    """Build raw monthly features without future data."""
    config = feature_config or TechnicalFeatureConfig()

    config.validate()

    data = _prepare_market_data(market_data)

    rebalance_dates = _prepare_rebalance_dates(rebalance_calendar)

    rows: list[dict[str, Any]] = []

    for _, ticker_data in data.groupby(
        "ticker",
        sort=True,
    ):
        ticker_data = ticker_data.sort_values("date").reset_index(drop=True)

        for as_of_date in rebalance_dates:
            history = ticker_data.loc[ticker_data["date"].le(as_of_date)]

            if history.empty:
                continue

            rows.append(
                _feature_row(
                    history,
                    as_of_date,
                    config,
                )
            )

    if not rows:
        return pd.DataFrame(columns=(TECHNICAL_PANEL_COLUMNS))

    features = pd.DataFrame(rows).loc[
        :,
        TECHNICAL_PANEL_COLUMNS,
    ]

    violations = int(features["latest_market_date"].gt(features["as_of_date"]).sum())

    if violations:
        raise TechnicalFeatureError(f"{violations} feature rows use future market observations.")

    return features.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)
