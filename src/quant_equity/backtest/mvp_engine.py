"""Daily execution engine for the MVP equity portfolios."""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.backtest.execution import (
    ExecutionCostConfig,
    estimate_execution_cost_arrays,
)


class MVPBacktestError(ValueError):
    """Raised when the MVP backtest cannot be executed safely."""


@dataclass(frozen=True)
class MVPBacktestConfig:
    """Configuration for the daily MVP execution engine."""

    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 10.0

    final_holding_sessions: int = 21

    date_column: str = "date"
    ticker_column: str = "ticker"
    open_column: str = "open"
    close_column: str = "close"
    adjusted_close_column: str = "adjusted_close"

    minimum_trade_notional: float = 0.01
    cash_tolerance: float = 0.05
    weight_tolerance: float = 1.0e-8
    share_tolerance: float = 1.0e-12

    bisection_tolerance: float = 1.0e-8
    bisection_max_iterations: int = 200

    @property
    def transaction_cost_rate(self) -> float:
        """Return transaction costs as a decimal rate."""
        return self.transaction_cost_bps / 10_000.0

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> MVPBacktestConfig:
        """Create configuration from YAML values."""
        config = cls(
            initial_capital=float(
                values.get(
                    "initial_capital",
                    1_000_000.0,
                )
            ),
            transaction_cost_bps=float(
                values.get(
                    "transaction_cost_bps",
                    10.0,
                )
            ),
            final_holding_sessions=int(
                values.get(
                    "final_holding_sessions",
                    21,
                )
            ),
            date_column=str(
                values.get(
                    "date_column",
                    "date",
                )
            ),
            ticker_column=str(
                values.get(
                    "ticker_column",
                    "ticker",
                )
            ),
            open_column=str(
                values.get(
                    "open_column",
                    "open",
                )
            ),
            close_column=str(
                values.get(
                    "close_column",
                    "close",
                )
            ),
            adjusted_close_column=str(
                values.get(
                    "adjusted_close_column",
                    "adjusted_close",
                )
            ),
            minimum_trade_notional=float(
                values.get(
                    "minimum_trade_notional",
                    0.01,
                )
            ),
            cash_tolerance=float(
                values.get(
                    "cash_tolerance",
                    0.05,
                )
            ),
            weight_tolerance=float(
                values.get(
                    "weight_tolerance",
                    1.0e-8,
                )
            ),
            share_tolerance=float(
                values.get(
                    "share_tolerance",
                    1.0e-12,
                )
            ),
            bisection_tolerance=float(
                values.get(
                    "bisection_tolerance",
                    1.0e-8,
                )
            ),
            bisection_max_iterations=int(
                values.get(
                    "bisection_max_iterations",
                    200,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate execution settings."""
        if self.initial_capital <= 0.0:
            raise MVPBacktestError("initial_capital must be positive.")

        if not 0.0 <= self.transaction_cost_bps < 10_000.0:
            raise MVPBacktestError("transaction_cost_bps must be in [0, 10000).")

        if self.final_holding_sessions < 1:
            raise MVPBacktestError("final_holding_sessions must be positive.")

        for column in (
            self.date_column,
            self.ticker_column,
            self.open_column,
            self.close_column,
            self.adjusted_close_column,
        ):
            if not column:
                raise MVPBacktestError("Market-data column names cannot be empty.")

        if self.minimum_trade_notional < 0.0:
            raise MVPBacktestError("minimum_trade_notional cannot be negative.")

        if self.cash_tolerance <= 0.0:
            raise MVPBacktestError("cash_tolerance must be positive.")

        if self.weight_tolerance <= 0.0:
            raise MVPBacktestError("weight_tolerance must be positive.")

        if self.share_tolerance <= 0.0:
            raise MVPBacktestError("share_tolerance must be positive.")

        if self.bisection_tolerance <= 0.0:
            raise MVPBacktestError("bisection_tolerance must be positive.")

        if self.bisection_max_iterations < 1:
            raise MVPBacktestError("bisection_max_iterations must be positive.")


@dataclass(frozen=True)
class MVPBacktestOutputs:
    """Tables produced by the daily MVP backtest."""

    execution_schedule: pd.DataFrame
    daily_performance: pd.DataFrame
    daily_positions: pd.DataFrame
    trades: pd.DataFrame
    rebalance_summary: pd.DataFrame
    execution_summary: pd.DataFrame


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Require dataframe columns."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise MVPBacktestError(f"{dataset_name} is missing columns: " + ", ".join(missing) + ".")


def _validate_target_weights(
    target_weights: pd.DataFrame,
    *,
    config: MVPBacktestConfig,
) -> pd.DataFrame:
    """Validate and normalize monthly target portfolios."""
    required_columns = (
        "as_of_date",
        "strategy_name",
        "ticker",
        "sector",
        "target_weight",
    )

    _require_columns(
        target_weights,
        required_columns,
        dataset_name="MVP target weights",
    )

    data = target_weights.copy()

    data["as_of_date"] = pd.to_datetime(
        data["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    if data["as_of_date"].isna().any():
        raise MVPBacktestError("Target weights contain invalid as_of_date values.")

    for column in (
        "strategy_name",
        "ticker",
        "sector",
    ):
        data[column] = data[column].astype("string").str.strip()

        if data[column].isna().any() or data[column].eq("").any():
            raise MVPBacktestError(f"Target weights contain missing {column} values.")

    data["target_weight"] = pd.to_numeric(
        data["target_weight"],
        errors="coerce",
    )

    if data["target_weight"].isna().any():
        raise MVPBacktestError("Target weights contain invalid numeric weights.")

    if np.isinf(data["target_weight"].to_numpy(dtype=float)).any():
        raise MVPBacktestError("Target weights contain infinite values.")

    if (data["target_weight"] < -config.weight_tolerance).any():
        raise MVPBacktestError("Target weights contain negative positions.")

    duplicate_count = int(
        data.duplicated(
            [
                "as_of_date",
                "strategy_name",
                "ticker",
            ]
        ).sum()
    )

    if duplicate_count:
        raise MVPBacktestError("Target weights contain duplicated date-strategy-ticker rows.")

    sector_counts = data.groupby("ticker")["sector"].nunique()

    if sector_counts.gt(1).any():
        invalid_ticker = str(sector_counts.loc[sector_counts.gt(1)].index[0])

        raise MVPBacktestError(
            f"At least one ticker is assigned to multiple sectors: {invalid_ticker}."
        )

    weight_sums = data.groupby(
        [
            "strategy_name",
            "as_of_date",
        ]
    )["target_weight"].sum()

    invalid_sums = weight_sums.loc[weight_sums.sub(1.0).abs().gt(config.weight_tolerance)]

    if not invalid_sums.empty:
        first_index = invalid_sums.index[0]

        raise MVPBacktestError(
            "At least one target portfolio does not sum to one. "
            f"First invalid portfolio: {first_index}, "
            f"sum={invalid_sums.iloc[0]:.12f}."
        )

    dates_by_strategy = {
        strategy_name: set(strategy_data["as_of_date"])
        for strategy_name, strategy_data in (
            data.groupby(
                "strategy_name",
                sort=True,
            )
        )
    }

    reference_strategy = sorted(dates_by_strategy)[0]

    reference_dates = dates_by_strategy[reference_strategy]

    for (
        strategy_name,
        strategy_dates,
    ) in dates_by_strategy.items():
        if strategy_dates != reference_dates:
            raise MVPBacktestError(
                "Strategies do not contain the same signal dates. "
                f"Misaligned strategy: {strategy_name}."
            )

    return data.sort_values(
        [
            "as_of_date",
            "strategy_name",
            "ticker",
        ]
    ).reset_index(drop=True)


def _validate_market_data(
    market_data: pd.DataFrame,
    *,
    required_tickers: set[str],
    config: MVPBacktestConfig,
) -> pd.DataFrame:
    """Validate market data and calculate adjusted opening prices."""
    required_columns = (
        config.date_column,
        config.ticker_column,
        config.open_column,
        config.close_column,
        config.adjusted_close_column,
    )

    _require_columns(
        market_data,
        required_columns,
        dataset_name="Daily market data",
    )

    data = market_data.copy()

    data[config.date_column] = pd.to_datetime(
        data[config.date_column],
        errors="coerce",
    ).dt.normalize()

    if data[config.date_column].isna().any():
        raise MVPBacktestError("Market data contain invalid dates.")

    data[config.ticker_column] = data[config.ticker_column].astype("string").str.strip()

    if data[config.ticker_column].isna().any() or data[config.ticker_column].eq("").any():
        raise MVPBacktestError("Market data contain missing ticker values.")

    data = data.loc[data[config.ticker_column].isin(required_tickers)].copy()

    available_tickers = set(data[config.ticker_column].unique())

    missing_tickers = sorted(required_tickers.difference(available_tickers))

    if missing_tickers:
        raise MVPBacktestError(
            "Market data are missing required tickers: " + ", ".join(missing_tickers) + "."
        )

    for column in (
        config.open_column,
        config.close_column,
        config.adjusted_close_column,
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        invalid = data[column].notna() & (
            ~np.isfinite(data[column].to_numpy(dtype=float)) | data[column].le(0.0)
        )

        if invalid.any():
            raise MVPBacktestError(
                f"Market data contain non-positive or infinite values in {column}."
            )

    duplicate_count = int(
        data.duplicated(
            [
                config.date_column,
                config.ticker_column,
            ]
        ).sum()
    )

    if duplicate_count:
        raise MVPBacktestError("Market data contain duplicated date-ticker rows.")

    data["adjusted_open"] = (
        data[config.open_column] * data[config.adjusted_close_column] / data[config.close_column]
    )

    invalid_adjusted_open = data["adjusted_open"].notna() & (
        ~np.isfinite(data["adjusted_open"].to_numpy(dtype=float)) | data["adjusted_open"].le(0.0)
    )

    if invalid_adjusted_open.any():
        raise MVPBacktestError("Calculated adjusted opening prices are invalid.")

    return data.sort_values(
        [
            config.date_column,
            config.ticker_column,
        ]
    ).reset_index(drop=True)


def build_execution_schedule(
    signal_dates: Sequence[pd.Timestamp],
    market_dates: Sequence[pd.Timestamp],
    *,
    final_holding_sessions: int,
) -> pd.DataFrame:
    """Map signal dates to the following trading sessions."""
    normalized_signals = pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).normalize()

    normalized_market_dates = pd.DatetimeIndex(pd.to_datetime(list(market_dates))).normalize()

    normalized_signals = normalized_signals.sort_values()
    normalized_market_dates = normalized_market_dates.drop_duplicates().sort_values()

    if normalized_signals.empty:
        raise MVPBacktestError("No signal dates were provided.")

    if normalized_market_dates.empty:
        raise MVPBacktestError("No market dates were provided.")

    market_date_set = set(normalized_market_dates)

    missing_signal_dates = [
        signal_date for signal_date in normalized_signals if signal_date not in market_date_set
    ]

    if missing_signal_dates:
        raise MVPBacktestError(
            f"At least one signal date is not a market session: {missing_signal_dates[0].date()}."
        )

    rows: list[dict[str, Any]] = []

    for index, signal_date in enumerate(normalized_signals):
        signal_position = int(
            normalized_market_dates.searchsorted(
                signal_date,
                side="left",
            )
        )

        execution_position = signal_position + 1

        if execution_position >= len(normalized_market_dates):
            raise MVPBacktestError(
                f"No following market session exists for signal date {signal_date.date()}."
            )

        execution_date = normalized_market_dates[execution_position]

        if index + 1 < len(normalized_signals):
            next_signal_date = normalized_signals[index + 1]

            next_signal_position = int(
                normalized_market_dates.searchsorted(
                    next_signal_date,
                    side="left",
                )
            )

            next_execution_position = next_signal_position + 1

            holding_end_position = next_execution_position - 1

            holding_end_date = normalized_market_dates[holding_end_position]
        else:
            next_signal_date = pd.NaT

            holding_end_position = signal_position + final_holding_sessions

            if holding_end_position >= len(normalized_market_dates):
                raise MVPBacktestError(
                    "Insufficient market sessions after the final "
                    "signal to complete the requested holding period."
                )

            holding_end_date = normalized_market_dates[holding_end_position]

        rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "next_signal_date": (next_signal_date),
                "holding_end_date": (holding_end_date),
            }
        )

    schedule = pd.DataFrame(rows)

    if schedule["execution_date"].duplicated().any():
        raise MVPBacktestError("Multiple signals map to the same execution date.")

    if not (schedule["execution_date"] > schedule["signal_date"]).all():
        raise MVPBacktestError("Execution dates must be after signal dates.")

    return schedule


def _prepare_execution_risk_lookup(
    risk_estimates: pd.DataFrame,
    *,
    signal_dates: Sequence[pd.Timestamp],
    tickers: Sequence[str],
) -> dict[
    pd.Timestamp,
    tuple[
        np.ndarray,
        np.ndarray,
    ],
]:
    """Align point-in-time volatility and liquidity with execution tickers."""
    _require_columns(
        risk_estimates,
        (
            "as_of_date",
            "ticker",
            "annualized_volatility",
            "average_dollar_volume",
        ),
        dataset_name="Execution risk estimates",
    )

    data = risk_estimates.loc[
        :,
        [
            "as_of_date",
            "ticker",
            "annualized_volatility",
            "average_dollar_volume",
        ],
    ].copy()

    data["as_of_date"] = pd.to_datetime(
        data["as_of_date"],
        errors="coerce",
    ).dt.normalize()

    data["ticker"] = data["ticker"].astype("string").str.strip()

    for column in (
        "annualized_volatility",
        "average_dollar_volume",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data["as_of_date"].isna().any() or data["ticker"].isna().any():
        raise MVPBacktestError("Execution risk estimates contain invalid keys.")

    if data.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    ).any():
        raise MVPBacktestError("Execution risk estimates contain duplicated date-ticker rows.")

    ordered_tickers = [str(ticker) for ticker in tickers]

    lookup: dict[
        pd.Timestamp,
        tuple[
            np.ndarray,
            np.ndarray,
        ],
    ] = {}

    for signal_date in pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).normalize():
        date_data = (
            data.loc[data["as_of_date"].eq(signal_date)]
            .set_index("ticker")
            .reindex(ordered_tickers)
        )

        missing = (
            date_data[
                [
                    "annualized_volatility",
                    "average_dollar_volume",
                ]
            ]
            .isna()
            .any(axis=1)
        )

        if missing.any():
            missing_ticker = str(date_data.index[missing][0])

            raise MVPBacktestError(
                f"Missing execution risk estimate for {missing_ticker} on {signal_date.date()}."
            )

        volatility = date_data["annualized_volatility"].to_numpy(dtype=float)

        dollar_volume = date_data["average_dollar_volume"].to_numpy(dtype=float)

        if not np.isfinite(volatility).all() or (volatility < 0.0).any():
            raise MVPBacktestError("Execution volatility estimates are invalid.")

        if not np.isfinite(dollar_volume).all() or (dollar_volume <= 0.0).any():
            raise MVPBacktestError("Execution liquidity estimates are invalid.")

        lookup[pd.Timestamp(signal_date)] = (
            volatility,
            dollar_volume,
        )

    return lookup


def _solve_post_cost_investable_value(
    *,
    pre_trade_nav: float,
    current_values: np.ndarray,
    target_weights: np.ndarray,
    transaction_cost_rate: float,
    tolerance: float,
    maximum_iterations: int,
    execution_cost_function: (
        Callable[
            [
                np.ndarray,
            ],
            np.ndarray,
        ]
        | None
    ) = None,
) -> float:
    """Solve the self-financing portfolio value after costs."""
    if execution_cost_function is None and transaction_cost_rate == 0.0:
        return pre_trade_nav

    def calculate_cost(
        absolute_trade_notional: np.ndarray,
    ) -> float:
        if execution_cost_function is None:
            return float(transaction_cost_rate * absolute_trade_notional.sum())

        cost_vector = np.asarray(
            execution_cost_function(absolute_trade_notional),
            dtype=float,
        ).reshape(-1)

        if cost_vector.shape != absolute_trade_notional.shape:
            raise MVPBacktestError("Execution cost function returned an invalid shape.")

        if not np.isfinite(cost_vector).all() or (cost_vector < 0.0).any():
            raise MVPBacktestError("Execution cost function returned invalid costs.")

        return float(cost_vector.sum())

    def equation(
        investable_value: float,
    ) -> float:
        target_values = investable_value * target_weights

        absolute_trade_notional = np.abs(target_values - current_values)

        transaction_cost = calculate_cost(absolute_trade_notional)

        return float(investable_value + transaction_cost - pre_trade_nav)

    lower = 0.0
    upper = pre_trade_nav

    lower_value = equation(lower)

    upper_value = equation(upper)

    if lower_value > tolerance:
        raise MVPBacktestError("Could not bracket the post-cost portfolio value.")

    if upper_value <= tolerance:
        return upper

    for _ in range(maximum_iterations):
        midpoint = 0.5 * (lower + upper)

        midpoint_value = equation(midpoint)

        if abs(midpoint_value) <= tolerance:
            return midpoint

        if midpoint_value > 0.0:
            upper = midpoint
        else:
            lower = midpoint

    result = 0.5 * (lower + upper)

    if abs(equation(result)) > max(
        tolerance,
        pre_trade_nav * 1.0e-10,
    ):
        raise MVPBacktestError("Post-cost portfolio value did not converge.")

    return result


def _run_single_strategy(
    target_weights: pd.DataFrame,
    *,
    strategy_name: str,
    schedule: pd.DataFrame,
    backtest_dates: pd.DatetimeIndex,
    adjusted_open_prices: pd.DataFrame,
    adjusted_close_prices: pd.DataFrame,
    sector_by_ticker: Mapping[str, str],
    config: MVPBacktestConfig,
    execution_risk_lookup: (
        Mapping[
            pd.Timestamp,
            tuple[
                np.ndarray,
                np.ndarray,
            ],
        ]
        | None
    ) = None,
    execution_cost_config: (ExecutionCostConfig | None) = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Execute one strategy through the complete daily calendar."""
    tickers = sorted(target_weights["ticker"].astype(str).unique())

    ticker_index = {ticker: index for index, ticker in enumerate(tickers)}

    shares = np.zeros(
        len(tickers),
        dtype=float,
    )

    current_target_weights = np.zeros(
        len(tickers),
        dtype=float,
    )

    cash = float(config.initial_capital)

    previous_end_nav = float(config.initial_capital)

    cumulative_transaction_cost = 0.0

    active_signal_date = pd.NaT
    last_rebalance_date = pd.NaT

    target_by_signal: dict[
        pd.Timestamp,
        np.ndarray,
    ] = {}

    for signal_date, signal_data in target_weights.groupby(
        "as_of_date",
        sort=True,
    ):
        weights = np.zeros(
            len(tickers),
            dtype=float,
        )

        for row in signal_data.itertuples(index=False):
            weights[ticker_index[str(row.ticker)]] = float(row.target_weight)

        if abs(float(weights.sum()) - 1.0) > config.weight_tolerance:
            raise MVPBacktestError(
                f"Strategy target weights do not sum to one: {strategy_name}, {signal_date}."
            )

        target_by_signal[pd.Timestamp(signal_date)] = weights

    signal_by_execution = {
        pd.Timestamp(row.execution_date): pd.Timestamp(row.signal_date)
        for row in schedule.itertuples(index=False)
    }

    daily_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    rebalance_rows: list[dict[str, Any]] = []

    for date in backtest_dates:
        is_rebalance = date in signal_by_execution

        transaction_cost = 0.0
        traded_notional = 0.0
        buy_notional = 0.0
        sell_notional = 0.0
        two_way_turnover = 0.0
        one_way_turnover = 0.0

        if is_rebalance:
            signal_date = signal_by_execution[date]

            target_weight_vector = target_by_signal[signal_date]

            opening_prices = (
                adjusted_open_prices.loc[date].reindex(tickers).astype(float).to_numpy()
            )

            held_before = np.abs(shares) > config.share_tolerance

            targeted = target_weight_vector > config.weight_tolerance

            required_open_prices = held_before | targeted

            invalid_open_prices = required_open_prices & (
                ~np.isfinite(opening_prices) | (opening_prices <= 0.0)
            )

            if invalid_open_prices.any():
                invalid_ticker = tickers[int(np.flatnonzero(invalid_open_prices)[0])]

                raise MVPBacktestError(
                    "Missing execution price for "
                    f"{invalid_ticker} on {date.date()} "
                    f"in strategy {strategy_name}."
                )

            current_values = np.zeros(
                len(tickers),
                dtype=float,
            )

            current_values[held_before] = shares[held_before] * opening_prices[held_before]

            cash_before = cash

            pre_trade_nav = float(cash + current_values.sum())

            if pre_trade_nav <= 0.0:
                raise MVPBacktestError(
                    "Portfolio NAV became non-positive before "
                    f"rebalance: {strategy_name}, {date.date()}."
                )

            execution_cost_function = None

            execution_volatility = None
            execution_dollar_volume = None

            if execution_cost_config is not None:
                if execution_risk_lookup is None or signal_date not in execution_risk_lookup:
                    raise MVPBacktestError(
                        "Advanced execution costs require point-in-time risk estimates."
                    )

                (
                    execution_volatility,
                    execution_dollar_volume,
                ) = execution_risk_lookup[signal_date]

                def execution_cost_function(
                    absolute_notional: np.ndarray,
                    volatility: np.ndarray = execution_volatility,
                    dollar_volume: np.ndarray = execution_dollar_volume,
                    cost_config: ExecutionCostConfig = execution_cost_config,
                ) -> np.ndarray:
                    return estimate_execution_cost_arrays(
                        absolute_notional,
                        volatility,
                        dollar_volume,
                        config=cost_config,
                    )["total_execution_cost"]

            post_cost_investable_value = _solve_post_cost_investable_value(
                pre_trade_nav=pre_trade_nav,
                current_values=current_values,
                target_weights=(target_weight_vector),
                transaction_cost_rate=(
                    0.0 if execution_cost_config is not None else config.transaction_cost_rate
                ),
                tolerance=(config.bisection_tolerance),
                maximum_iterations=(config.bisection_max_iterations),
                execution_cost_function=(execution_cost_function),
            )

            target_values = post_cost_investable_value * target_weight_vector

            target_shares = np.zeros(
                len(tickers),
                dtype=float,
            )

            target_shares[targeted] = target_values[targeted] / opening_prices[targeted]

            trade_notional_vector = target_values - current_values

            trade_shares_vector = target_shares - shares

            absolute_trade_notional = np.abs(trade_notional_vector)

            traded_notional = float(absolute_trade_notional.sum())

            buy_notional = float(trade_notional_vector[trade_notional_vector > 0.0].sum())

            sell_notional = float(-trade_notional_vector[trade_notional_vector < 0.0].sum())

            if execution_cost_config is None:
                transaction_cost = config.transaction_cost_rate * traded_notional

                transaction_cost_allocations = np.zeros(
                    len(tickers),
                    dtype=float,
                )

                if traded_notional > 0.0:
                    transaction_cost_allocations = (
                        transaction_cost * absolute_trade_notional / traded_notional
                    )
            else:
                if execution_volatility is None or execution_dollar_volume is None:
                    raise MVPBacktestError("Advanced execution inputs are unavailable.")

                execution_components = estimate_execution_cost_arrays(
                    absolute_trade_notional,
                    execution_volatility,
                    execution_dollar_volume,
                    config=execution_cost_config,
                )

                transaction_cost_allocations = execution_components["total_execution_cost"]

                transaction_cost = float(transaction_cost_allocations.sum())

            two_way_turnover = traded_notional / pre_trade_nav

            one_way_turnover = (
                max(
                    buy_notional,
                    sell_notional,
                )
                / pre_trade_nav
            )

            cash = float(pre_trade_nav - target_values.sum() - transaction_cost)

            acceptable_cash_error = max(
                config.cash_tolerance,
                pre_trade_nav * config.weight_tolerance,
            )

            if abs(cash) > acceptable_cash_error:
                raise MVPBacktestError(
                    "Unexpected residual cash after rebalance: "
                    f"{cash:.8f} for {strategy_name} "
                    f"on {date.date()}."
                )

            pre_trade_weights = current_values / pre_trade_nav

            output_trade_mask = absolute_trade_notional >= config.minimum_trade_notional

            for ticker_position in np.flatnonzero(output_trade_mask):
                trade_notional_value = float(trade_notional_vector[ticker_position])

                trade_rows.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": date,
                        "strategy_name": strategy_name,
                        "ticker": tickers[ticker_position],
                        "sector": sector_by_ticker[tickers[ticker_position]],
                        "side": ("BUY" if trade_notional_value > 0.0 else "SELL"),
                        "execution_price": float(opening_prices[ticker_position]),
                        "pre_trade_shares": float(shares[ticker_position]),
                        "target_shares": float(target_shares[ticker_position]),
                        "trade_shares": float(trade_shares_vector[ticker_position]),
                        "pre_trade_weight": float(pre_trade_weights[ticker_position]),
                        "target_weight": float(target_weight_vector[ticker_position]),
                        "trade_notional": (trade_notional_value),
                        "absolute_trade_notional": float(absolute_trade_notional[ticker_position]),
                        "transaction_cost": float(transaction_cost_allocations[ticker_position]),
                    }
                )

            holdings_before = int(held_before.sum())

            holdings_after = int(targeted.sum())

            sector_target_weights = (
                pd.DataFrame(
                    {
                        "sector": [sector_by_ticker[ticker] for ticker in tickers],
                        "target_weight": (target_weight_vector),
                    }
                )
                .groupby("sector")["target_weight"]
                .sum()
            )

            rebalance_rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": date,
                    "strategy_name": strategy_name,
                    "pre_trade_nav": pre_trade_nav,
                    "post_cost_investable_value": (post_cost_investable_value),
                    "cash_before": cash_before,
                    "cash_after": cash,
                    "transaction_cost": (transaction_cost),
                    "transaction_cost_fraction": (transaction_cost / pre_trade_nav),
                    "buy_notional": buy_notional,
                    "sell_notional": sell_notional,
                    "traded_notional": traded_notional,
                    "two_way_turnover": (two_way_turnover),
                    "one_way_turnover": (one_way_turnover),
                    "holdings_before": holdings_before,
                    "holdings_after": holdings_after,
                    "maximum_target_weight": float(target_weight_vector.max()),
                    "maximum_sector_target_weight": float(sector_target_weights.max()),
                }
            )

            shares = target_shares
            current_target_weights = target_weight_vector.copy()

            active_signal_date = signal_date
            last_rebalance_date = date

            cumulative_transaction_cost += transaction_cost

        closing_prices = adjusted_close_prices.loc[date].reindex(tickers).astype(float).to_numpy()

        held_at_close = np.abs(shares) > config.share_tolerance

        invalid_close_prices = held_at_close & (
            ~np.isfinite(closing_prices) | (closing_prices <= 0.0)
        )

        if invalid_close_prices.any():
            invalid_ticker = tickers[int(np.flatnonzero(invalid_close_prices)[0])]

            raise MVPBacktestError(
                "Missing valuation price for "
                f"{invalid_ticker} on {date.date()} "
                f"in strategy {strategy_name}."
            )

        market_values = np.zeros(
            len(tickers),
            dtype=float,
        )

        market_values[held_at_close] = shares[held_at_close] * closing_prices[held_at_close]

        invested_value = float(market_values.sum())

        portfolio_value = float(cash + invested_value)

        if portfolio_value <= 0.0:
            raise MVPBacktestError(
                f"Portfolio value became non-positive: {strategy_name}, {date.date()}."
            )

        daily_return = float(portfolio_value / previous_end_nav - 1.0)

        actual_weights = market_values / portfolio_value

        cash_weight = float(cash / portfolio_value)

        gross_exposure = float(invested_value / portfolio_value)

        holdings = int(held_at_close.sum())

        daily_rows.append(
            {
                "date": date,
                "strategy_name": strategy_name,
                "active_signal_date": (active_signal_date),
                "last_rebalance_date": (last_rebalance_date),
                "is_rebalance": is_rebalance,
                "portfolio_value": portfolio_value,
                "daily_return": daily_return,
                "cash": cash,
                "cash_weight": cash_weight,
                "invested_value": invested_value,
                "gross_exposure": gross_exposure,
                "holdings": holdings,
                "transaction_cost": (transaction_cost),
                "cumulative_transaction_cost": (cumulative_transaction_cost),
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "traded_notional": traded_notional,
                "two_way_turnover": (two_way_turnover),
                "one_way_turnover": (one_way_turnover),
            }
        )

        active_position_mask = held_at_close | (current_target_weights > config.weight_tolerance)

        for ticker_position in np.flatnonzero(active_position_mask):
            position_rows.append(
                {
                    "date": date,
                    "strategy_name": strategy_name,
                    "active_signal_date": (active_signal_date),
                    "last_rebalance_date": (last_rebalance_date),
                    "ticker": tickers[ticker_position],
                    "sector": sector_by_ticker[tickers[ticker_position]],
                    "shares": float(shares[ticker_position]),
                    "valuation_price": float(closing_prices[ticker_position]),
                    "market_value": float(market_values[ticker_position]),
                    "actual_weight": float(actual_weights[ticker_position]),
                    "target_weight": float(current_target_weights[ticker_position]),
                    "weight_drift": float(
                        actual_weights[ticker_position] - current_target_weights[ticker_position]
                    ),
                }
            )

        previous_end_nav = portfolio_value

    return (
        pd.DataFrame(daily_rows),
        pd.DataFrame(position_rows),
        pd.DataFrame(trade_rows),
        pd.DataFrame(rebalance_rows),
    )


def summarize_mvp_execution(
    daily_performance: pd.DataFrame,
    rebalance_summary: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    """Summarize execution and accounting by strategy."""
    rows: list[dict[str, Any]] = []

    for strategy_name, strategy_daily in daily_performance.groupby(
        "strategy_name",
        sort=True,
    ):
        strategy_daily = strategy_daily.sort_values("date")

        strategy_rebalances = rebalance_summary.loc[
            rebalance_summary["strategy_name"].eq(strategy_name)
        ]

        final_value = float(strategy_daily["portfolio_value"].iloc[-1])

        rows.append(
            {
                "strategy_name": strategy_name,
                "start_date": strategy_daily["date"].iloc[0],
                "end_date": strategy_daily["date"].iloc[-1],
                "trading_days": len(strategy_daily),
                "rebalances": len(strategy_rebalances),
                "initial_capital": (initial_capital),
                "final_portfolio_value": (final_value),
                "preliminary_total_return": (final_value / initial_capital - 1.0),
                "total_transaction_cost": float(strategy_daily["transaction_cost"].sum()),
                "total_traded_notional": float(strategy_daily["traded_notional"].sum()),
                "mean_two_way_turnover": float(strategy_rebalances["two_way_turnover"].mean()),
                "mean_one_way_turnover": float(strategy_rebalances["one_way_turnover"].mean()),
                "mean_holdings": float(strategy_daily["holdings"].mean()),
                "minimum_holdings": int(strategy_daily["holdings"].min()),
                "maximum_holdings": int(strategy_daily["holdings"].max()),
                "maximum_absolute_cash": float(strategy_daily["cash"].abs().max()),
                "maximum_absolute_cash_weight": float(strategy_daily["cash_weight"].abs().max()),
            }
        )

    return pd.DataFrame(rows).sort_values("strategy_name").reset_index(drop=True)


def run_mvp_backtest(
    target_weights: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    config: MVPBacktestConfig,
    risk_estimates: (pd.DataFrame | None) = None,
    execution_cost_config: (ExecutionCostConfig | None) = None,
) -> MVPBacktestOutputs:
    """Run the complete daily MVP execution backtest."""
    if (risk_estimates is None) != (execution_cost_config is None):
        raise MVPBacktestError(
            "risk_estimates and execution_cost_config "
            "must either both be provided or both be omitted."
        )

    if execution_cost_config is not None:
        execution_cost_config.validate()

    targets = _validate_target_weights(
        target_weights,
        config=config,
    )

    required_tickers = set(targets["ticker"].astype(str))

    market = _validate_market_data(
        market_data,
        required_tickers=required_tickers,
        config=config,
    )

    market_dates = pd.DatetimeIndex(market[config.date_column].unique()).sort_values()

    signal_dates = pd.DatetimeIndex(targets["as_of_date"].unique()).sort_values()

    schedule = build_execution_schedule(
        signal_dates,
        market_dates,
        final_holding_sessions=(config.final_holding_sessions),
    )

    first_execution_date = pd.Timestamp(schedule["execution_date"].min())

    final_backtest_date = pd.Timestamp(schedule["holding_end_date"].iloc[-1])

    backtest_dates = market_dates[
        (market_dates >= first_execution_date) & (market_dates <= final_backtest_date)
    ]

    adjusted_open_prices = market.pivot(
        index=config.date_column,
        columns=config.ticker_column,
        values="adjusted_open",
    )

    adjusted_close_prices = market.pivot(
        index=config.date_column,
        columns=config.ticker_column,
        values=config.adjusted_close_column,
    )

    adjusted_open_prices = adjusted_open_prices.reindex(backtest_dates)

    adjusted_close_prices = adjusted_close_prices.reindex(backtest_dates)

    sector_by_ticker = (
        targets.drop_duplicates("ticker").set_index("ticker")["sector"].astype(str).to_dict()
    )

    daily_frames: list[pd.DataFrame] = []

    position_frames: list[pd.DataFrame] = []

    trade_frames: list[pd.DataFrame] = []

    rebalance_frames: list[pd.DataFrame] = []

    for (
        strategy_name,
        strategy_targets,
    ) in targets.groupby(
        "strategy_name",
        sort=True,
    ):
        execution_risk_lookup = None

        if execution_cost_config is not None:
            if risk_estimates is None:
                raise MVPBacktestError("Advanced execution costs require risk estimates.")

            strategy_tickers = sorted(strategy_targets["ticker"].astype(str).unique())

            execution_risk_lookup = _prepare_execution_risk_lookup(
                risk_estimates,
                signal_dates=(schedule["signal_date"]),
                tickers=(strategy_tickers),
            )

        (
            strategy_daily,
            strategy_positions,
            strategy_trades,
            strategy_rebalances,
        ) = _run_single_strategy(
            strategy_targets,
            strategy_name=str(strategy_name),
            schedule=schedule,
            backtest_dates=backtest_dates,
            adjusted_open_prices=(adjusted_open_prices),
            adjusted_close_prices=(adjusted_close_prices),
            sector_by_ticker=(sector_by_ticker),
            config=config,
            execution_risk_lookup=(execution_risk_lookup),
            execution_cost_config=(execution_cost_config),
        )

        daily_frames.append(strategy_daily)

        position_frames.append(strategy_positions)

        trade_frames.append(strategy_trades)

        rebalance_frames.append(strategy_rebalances)

    daily_performance = (
        pd.concat(
            daily_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "date",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )

    daily_positions = (
        pd.concat(
            position_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "date",
                "strategy_name",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    trades = (
        pd.concat(
            trade_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "execution_date",
                "strategy_name",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    rebalance_summary = (
        pd.concat(
            rebalance_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "execution_date",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )

    expected_rebalances = targets["strategy_name"].nunique() * len(schedule)

    if len(rebalance_summary) != expected_rebalances:
        raise MVPBacktestError(
            "The number of executed rebalances does not match the number expected."
        )

    execution_summary = summarize_mvp_execution(
        daily_performance,
        rebalance_summary,
        initial_capital=(config.initial_capital),
    )

    return MVPBacktestOutputs(
        execution_schedule=schedule,
        daily_performance=(daily_performance),
        daily_positions=(daily_positions),
        trades=trades,
        rebalance_summary=(rebalance_summary),
        execution_summary=(execution_summary),
    )
