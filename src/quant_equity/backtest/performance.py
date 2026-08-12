"""Financial performance evaluation for the MVP backtest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class PerformanceEvaluationError(ValueError):
    """Raised when financial performance cannot be evaluated."""


@dataclass(frozen=True)
class PerformanceEvaluationConfig:
    """Configuration for MVP performance evaluation."""

    benchmark_name: str = "spy_buy_and_hold"
    benchmark_ticker: str = "SPY"

    annualization_periods: int = 252
    risk_free_rate: float = 0.0

    cost_scenarios_bps: tuple[float, ...] = (
        0.0,
        5.0,
        10.0,
        20.0,
        50.0,
    )

    numerical_tolerance: float = 1.0e-12

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> PerformanceEvaluationConfig:
        """Create configuration from YAML values."""
        raw_cost_scenarios = values.get(
            "cost_scenarios_bps",
            [
                0.0,
                5.0,
                10.0,
                20.0,
                50.0,
            ],
        )

        config = cls(
            benchmark_name=str(
                values.get(
                    "benchmark_name",
                    "spy_buy_and_hold",
                )
            ),
            benchmark_ticker=str(
                values.get(
                    "benchmark_ticker",
                    "SPY",
                )
            ),
            annualization_periods=int(
                values.get(
                    "annualization_periods",
                    252,
                )
            ),
            risk_free_rate=float(
                values.get(
                    "risk_free_rate",
                    0.0,
                )
            ),
            cost_scenarios_bps=tuple(float(value) for value in raw_cost_scenarios),
            numerical_tolerance=float(
                values.get(
                    "numerical_tolerance",
                    1.0e-12,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        """Validate evaluation settings."""
        if not self.benchmark_name:
            raise PerformanceEvaluationError("benchmark_name cannot be empty.")

        if not self.benchmark_ticker:
            raise PerformanceEvaluationError("benchmark_ticker cannot be empty.")

        if self.annualization_periods < 1:
            raise PerformanceEvaluationError("annualization_periods must be positive.")

        if self.risk_free_rate <= -1.0:
            raise PerformanceEvaluationError("risk_free_rate must be greater than -100%.")

        if not self.cost_scenarios_bps:
            raise PerformanceEvaluationError("At least one cost scenario is required.")

        if any(cost < 0.0 for cost in self.cost_scenarios_bps):
            raise PerformanceEvaluationError("Cost scenarios cannot be negative.")

        if len(set(self.cost_scenarios_bps)) != len(self.cost_scenarios_bps):
            raise PerformanceEvaluationError("Cost scenarios cannot contain duplicates.")

        if self.numerical_tolerance <= 0.0:
            raise PerformanceEvaluationError("numerical_tolerance must be positive.")


@dataclass(frozen=True)
class PerformanceEvaluationOutputs:
    """Tables produced by financial evaluation."""

    combined_daily: pd.DataFrame
    performance_summary: pd.DataFrame
    yearly_summary: pd.DataFrame
    monthly_returns: pd.DataFrame
    drawdowns: pd.DataFrame


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    """Require dataframe columns."""
    missing = sorted(set(columns).difference(data.columns))

    if missing:
        raise PerformanceEvaluationError(
            f"{dataset_name} is missing columns: " + ", ".join(missing) + "."
        )


def _validate_daily_performance(
    daily_performance: pd.DataFrame,
) -> pd.DataFrame:
    """Validate daily strategy or benchmark results."""
    required_columns = (
        "date",
        "strategy_name",
        "portfolio_value",
        "daily_return",
    )

    _require_columns(
        daily_performance,
        required_columns,
        dataset_name="Daily performance",
    )

    data = daily_performance.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    ).dt.normalize()

    if data["date"].isna().any():
        raise PerformanceEvaluationError("Daily performance contains invalid dates.")

    data["strategy_name"] = data["strategy_name"].astype("string").str.strip()

    if data["strategy_name"].isna().any() or data["strategy_name"].eq("").any():
        raise PerformanceEvaluationError("Daily performance contains missing strategy names.")

    for column in (
        "portfolio_value",
        "daily_return",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if data[column].isna().any():
            raise PerformanceEvaluationError(f"Daily performance contains invalid {column} values.")

        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise PerformanceEvaluationError(
                f"Daily performance contains infinite {column} values."
            )

    if data["portfolio_value"].le(0.0).any():
        raise PerformanceEvaluationError("Portfolio values must be positive.")

    duplicate_count = int(
        data.duplicated(
            [
                "date",
                "strategy_name",
            ]
        ).sum()
    )

    if duplicate_count:
        raise PerformanceEvaluationError(
            "Daily performance contains duplicated date-strategy rows."
        )

    optional_numeric_columns = (
        "transaction_cost",
        "traded_notional",
        "two_way_turnover",
        "one_way_turnover",
    )

    for column in optional_numeric_columns:
        if column not in data.columns:
            data[column] = 0.0
        else:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            ).fillna(0.0)

    return data.sort_values(
        [
            "strategy_name",
            "date",
        ]
    ).reset_index(drop=True)


def build_buy_and_hold_benchmark(
    benchmark_market_data: pd.DataFrame,
    *,
    strategy_name: str,
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Build a buy-and-hold benchmark using adjusted prices."""
    required_columns = (
        "date",
        "ticker",
        "open",
        "close",
        "adjusted_close",
    )

    _require_columns(
        benchmark_market_data,
        required_columns,
        dataset_name="Benchmark market data",
    )

    data = benchmark_market_data.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    ).dt.normalize()

    data["ticker"] = data["ticker"].astype("string").str.strip()

    data = data.loc[data["ticker"].eq(ticker)].copy()

    if data.empty:
        raise PerformanceEvaluationError(f"No benchmark data found for {ticker}.")

    for column in (
        "open",
        "close",
        "adjusted_close",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if data[column].isna().any() or data[column].le(0.0).any():
            raise PerformanceEvaluationError(f"Benchmark data contain invalid {column} values.")

    data["adjusted_open"] = data["open"] * data["adjusted_close"] / data["close"]

    data = (
        data.loc[
            data["date"].between(
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
            )
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if data.empty:
        raise PerformanceEvaluationError("No benchmark observations exist in the backtest period.")

    if data["date"].iloc[0] != pd.Timestamp(start_date):
        raise PerformanceEvaluationError(
            "Benchmark data do not begin on the first backtest execution date."
        )

    if data["date"].iloc[-1] != pd.Timestamp(end_date):
        raise PerformanceEvaluationError("Benchmark data do not reach the final backtest date.")

    transaction_cost_rate = transaction_cost_bps / 10_000.0

    investable_capital = initial_capital / (1.0 + transaction_cost_rate)

    initial_transaction_cost = initial_capital - investable_capital

    first_execution_price = float(data["adjusted_open"].iloc[0])

    shares = investable_capital / first_execution_price

    portfolio_values = shares * data["adjusted_close"]

    daily_returns = portfolio_values.pct_change()

    daily_returns.iloc[0] = portfolio_values.iloc[0] / initial_capital - 1.0

    result = pd.DataFrame(
        {
            "date": data["date"],
            "strategy_name": strategy_name,
            "portfolio_value": (portfolio_values.astype(float)),
            "daily_return": (daily_returns.astype(float)),
            "transaction_cost": 0.0,
            "traded_notional": 0.0,
            "two_way_turnover": 0.0,
            "one_way_turnover": 0.0,
            "holdings": 1,
        }
    )

    result.loc[
        result.index[0],
        "transaction_cost",
    ] = initial_transaction_cost

    result.loc[
        result.index[0],
        "traded_notional",
    ] = investable_capital

    result.loc[
        result.index[0],
        "two_way_turnover",
    ] = investable_capital / initial_capital

    result.loc[
        result.index[0],
        "one_way_turnover",
    ] = investable_capital / initial_capital

    return result


def _calculate_drawdown(
    portfolio_values: pd.Series,
    *,
    initial_capital: float,
) -> pd.Series:
    """Calculate drawdown including initial capital as the first peak."""
    values = portfolio_values.to_numpy(dtype=float)

    extended_values = np.concatenate(
        [
            np.array(
                [initial_capital],
                dtype=float,
            ),
            values,
        ]
    )

    running_maximum = np.maximum.accumulate(extended_values)[1:]

    drawdown = values / running_maximum - 1.0

    return pd.Series(
        drawdown,
        index=portfolio_values.index,
        dtype=float,
    )


def _annualized_sharpe(
    returns: pd.Series,
    *,
    risk_free_rate: float,
    annualization_periods: int,
) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2:
        return float("nan")

    daily_risk_free_rate = (1.0 + risk_free_rate) ** (1.0 / annualization_periods) - 1.0

    excess_returns = returns.astype(float) - daily_risk_free_rate

    volatility = float(excess_returns.std(ddof=1))

    if volatility <= 0.0:
        return float("nan")

    return float(excess_returns.mean() / volatility * np.sqrt(annualization_periods))


def _annualized_sortino(
    returns: pd.Series,
    *,
    risk_free_rate: float,
    annualization_periods: int,
) -> float:
    """Calculate annualized Sortino ratio."""
    daily_risk_free_rate = (1.0 + risk_free_rate) ** (1.0 / annualization_periods) - 1.0

    excess_returns = returns.astype(float) - daily_risk_free_rate

    downside_returns = np.minimum(
        excess_returns.to_numpy(dtype=float),
        0.0,
    )

    downside_deviation = float(np.sqrt(np.mean(np.square(downside_returns))))

    if downside_deviation <= 0.0:
        return float("nan")

    return float(excess_returns.mean() / downside_deviation * np.sqrt(annualization_periods))


def evaluate_performance(
    strategy_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    *,
    initial_capital: float,
    config: PerformanceEvaluationConfig,
) -> PerformanceEvaluationOutputs:
    """Evaluate strategies and benchmark together."""
    strategies = _validate_daily_performance(strategy_daily)

    benchmark = _validate_daily_performance(benchmark_daily)

    if config.benchmark_name not in set(benchmark["strategy_name"]):
        raise PerformanceEvaluationError("The configured benchmark name is missing.")

    combined = (
        pd.concat(
            [
                strategies,
                benchmark,
            ],
            ignore_index=True,
            sort=False,
        )
        .sort_values(
            [
                "strategy_name",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    date_sets = {
        strategy_name: set(group["date"])
        for strategy_name, group in (combined.groupby("strategy_name"))
    }

    benchmark_dates = date_sets[config.benchmark_name]

    for (
        strategy_name,
        strategy_dates,
    ) in date_sets.items():
        if strategy_dates != benchmark_dates:
            raise PerformanceEvaluationError(
                "Strategies and benchmark do not contain "
                "identical trading dates. "
                f"Misaligned strategy: {strategy_name}."
            )

    benchmark_returns = benchmark.loc[
        benchmark["strategy_name"].eq(config.benchmark_name),
        [
            "date",
            "daily_return",
        ],
    ].rename(columns={"daily_return": ("benchmark_return")})

    benchmark_final_value = float(benchmark["portfolio_value"].iloc[-1])

    benchmark_total_return = benchmark_final_value / initial_capital - 1.0

    benchmark_years = len(benchmark) / config.annualization_periods

    benchmark_cagr = (benchmark_final_value / initial_capital) ** (1.0 / benchmark_years) - 1.0

    performance_rows: list[dict[str, Any]] = []

    drawdown_frames: list[pd.DataFrame] = []

    monthly_frames: list[pd.DataFrame] = []

    yearly_rows: list[dict[str, Any]] = []

    for strategy_name, group in combined.groupby(
        "strategy_name",
        sort=True,
    ):
        group = group.sort_values("date").reset_index(drop=True)

        returns = group["daily_return"].astype(float)

        values = group["portfolio_value"].astype(float)

        final_value = float(values.iloc[-1])

        trading_days = len(group)

        years = trading_days / config.annualization_periods

        total_return = final_value / initial_capital - 1.0

        cagr = (final_value / initial_capital) ** (1.0 / years) - 1.0

        annualized_volatility = float(returns.std(ddof=1) * np.sqrt(config.annualization_periods))

        sharpe = _annualized_sharpe(
            returns,
            risk_free_rate=(config.risk_free_rate),
            annualization_periods=(config.annualization_periods),
        )

        sortino = _annualized_sortino(
            returns,
            risk_free_rate=(config.risk_free_rate),
            annualization_periods=(config.annualization_periods),
        )

        drawdown = _calculate_drawdown(
            values,
            initial_capital=initial_capital,
        )

        maximum_drawdown = float(drawdown.min())

        calmar = cagr / abs(maximum_drawdown) if maximum_drawdown < 0.0 else float("nan")

        relative_data = group.loc[
            :,
            [
                "date",
                "daily_return",
            ],
        ].merge(
            benchmark_returns,
            on="date",
            how="inner",
            validate="one_to_one",
        )

        strategy_returns = relative_data["daily_return"].astype(float)

        aligned_benchmark_returns = relative_data["benchmark_return"].astype(float)

        benchmark_variance = float(aligned_benchmark_returns.var(ddof=1))

        beta = (
            float(strategy_returns.cov(aligned_benchmark_returns) / benchmark_variance)
            if benchmark_variance > 0.0
            else float("nan")
        )

        daily_risk_free_rate = (1.0 + config.risk_free_rate) ** (
            1.0 / config.annualization_periods
        ) - 1.0

        alpha = (
            float(
                (
                    (strategy_returns - daily_risk_free_rate).mean()
                    - beta * (aligned_benchmark_returns - daily_risk_free_rate).mean()
                )
                * config.annualization_periods
            )
            if np.isfinite(beta)
            else float("nan")
        )

        active_returns = strategy_returns - aligned_benchmark_returns

        tracking_error = float(active_returns.std(ddof=1) * np.sqrt(config.annualization_periods))

        information_ratio = (
            float(active_returns.mean() * config.annualization_periods / tracking_error)
            if tracking_error > 0.0
            else float("nan")
        )

        correlation = float(strategy_returns.corr(aligned_benchmark_returns))

        performance_rows.append(
            {
                "strategy_name": strategy_name,
                "start_date": group["date"].iloc[0],
                "end_date": group["date"].iloc[-1],
                "trading_days": (trading_days),
                "initial_capital": (initial_capital),
                "final_portfolio_value": (final_value),
                "total_return": (total_return),
                "cagr": cagr,
                "annualized_volatility": (annualized_volatility),
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "maximum_drawdown": (maximum_drawdown),
                "calmar_ratio": calmar,
                "positive_day_ratio": float(returns.gt(0.0).mean()),
                "beta_vs_spy": beta,
                "annualized_alpha_vs_spy": (alpha),
                "correlation_vs_spy": (correlation),
                "tracking_error_vs_spy": (tracking_error),
                "information_ratio_vs_spy": (information_ratio),
                "benchmark_total_return": (benchmark_total_return),
                "excess_total_return": (total_return - benchmark_total_return),
                "benchmark_cagr": (benchmark_cagr),
                "excess_cagr": (cagr - benchmark_cagr),
                "total_transaction_cost": float(group["transaction_cost"].sum()),
                "total_traded_notional": float(group["traded_notional"].sum()),
                "total_two_way_turnover": float(group["two_way_turnover"].sum()),
                "mean_rebalance_two_way_turnover": float(
                    group.loc[
                        group["two_way_turnover"].gt(0.0),
                        "two_way_turnover",
                    ].mean()
                ),
            }
        )

        drawdown_frames.append(
            pd.DataFrame(
                {
                    "date": group["date"],
                    "strategy_name": (strategy_name),
                    "portfolio_value": (values),
                    "drawdown": drawdown,
                }
            )
        )

        monthly = group.copy()

        monthly["month"] = monthly["date"].dt.to_period("M").dt.to_timestamp("M")

        monthly_result = (
            monthly.groupby(
                "month",
                as_index=False,
            )["daily_return"]
            .apply(lambda values: (1.0 + values).prod() - 1.0)
            .rename(columns={"daily_return": ("monthly_return")})
        )

        monthly_result["strategy_name"] = strategy_name

        monthly_frames.append(
            monthly_result.loc[
                :,
                [
                    "month",
                    "strategy_name",
                    "monthly_return",
                ],
            ]
        )

        yearly_data = group.copy()

        yearly_data["year"] = yearly_data["date"].dt.year

        for year, year_group in yearly_data.groupby(
            "year",
            sort=True,
        ):
            year_returns = year_group["daily_return"].astype(float)

            year_total_return = float((1.0 + year_returns).prod() - 1.0)

            year_equity = (1.0 + year_returns).cumprod()

            year_running_maximum = (
                pd.concat(
                    [
                        pd.Series([1.0]),
                        year_equity,
                    ],
                    ignore_index=True,
                )
                .cummax()
                .iloc[1:]
                .reset_index(drop=True)
            )

            year_drawdown = year_equity.reset_index(drop=True) / year_running_maximum - 1.0

            yearly_rows.append(
                {
                    "strategy_name": (strategy_name),
                    "year": int(year),
                    "trading_days": len(year_group),
                    "total_return": (year_total_return),
                    "annualized_volatility": float(
                        year_returns.std(ddof=1) * np.sqrt(config.annualization_periods)
                    ),
                    "sharpe_ratio": (
                        _annualized_sharpe(
                            year_returns,
                            risk_free_rate=(config.risk_free_rate),
                            annualization_periods=(config.annualization_periods),
                        )
                    ),
                    "maximum_drawdown": float(year_drawdown.min()),
                    "transaction_cost": float(year_group["transaction_cost"].sum()),
                    "two_way_turnover": float(year_group["two_way_turnover"].sum()),
                }
            )

    performance_summary = (
        pd.DataFrame(performance_rows)
        .sort_values(
            [
                "sharpe_ratio",
                "cagr",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    yearly_summary = (
        pd.DataFrame(yearly_rows)
        .sort_values(
            [
                "year",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )

    monthly_returns = (
        pd.concat(
            monthly_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "month",
                "strategy_name",
            ]
        )
        .reset_index(drop=True)
    )

    drawdowns = (
        pd.concat(
            drawdown_frames,
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

    return PerformanceEvaluationOutputs(
        combined_daily=combined,
        performance_summary=(performance_summary),
        yearly_summary=yearly_summary,
        monthly_returns=monthly_returns,
        drawdowns=drawdowns,
    )
