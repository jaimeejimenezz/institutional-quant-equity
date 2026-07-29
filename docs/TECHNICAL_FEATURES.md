# Technical Feature Dictionary

## Purpose

This document describes the first point-in-time market features used by the Institutional Quant Equity Research Platform.

Each observation represents one company at one monthly rebalance date. Every calculation uses only market information available on or before `as_of_date`.

The raw feature dataset is stored at:

`data/interim/features_technical_raw_monthly.parquet`

Cross-sectional winsorization, standardization and optional sector neutralization are applied in a later stage.

## Temporal convention

For a rebalance date `t`:

* Prices and volume up to the close of `t` may be used.
* No observation after `t` may be used.
* The signal is considered available after the close of `t`.
* Any eventual portfolio operation starts from the following market session.
* `latest_market_date` must always be less than or equal to `as_of_date`.

## Return and momentum features

### `momentum_12_1`

Return from 252 sessions before the rebalance date to 21 sessions before the rebalance date:

`P(t-21) / P(t-252) - 1`

The most recent month is excluded.

### `momentum_6_1`

Return from 126 sessions before the rebalance date to 21 sessions before the rebalance date:

`P(t-21) / P(t-126) - 1`

### `return_3m`

Adjusted-close return over the latest 63 sessions:

`P(t) / P(t-63) - 1`

### `return_1m`

Adjusted-close return over the latest 21 sessions:

`P(t) / P(t-21) - 1`

### `return_1w`

Adjusted-close return over the latest five sessions:

`P(t) / P(t-5) - 1`

### `reversal_1m`

Negative one-month return:

`-return_1m`

This feature represents the short-term reversal hypothesis. Because it is perfectly inversely related to `return_1m`, both are retained initially and their redundancy will be evaluated during factor research.

## Risk features

### `volatility_20d`

Sample standard deviation of the latest 20 daily adjusted-close returns, annualized using 252 sessions.

### `volatility_60d`

Sample standard deviation of the latest 60 daily adjusted-close returns, annualized using 252 sessions.

### `downside_volatility_60d`

Square root of the mean squared negative component of the latest 60 daily returns, annualized using 252 sessions.

Positive returns contribute zero to downside volatility.

### `beta_60d_market`

Beta over the latest 60 sessions:

`cov(stock_return, market_return) / var(market_return)`

During the MVP, `market_return` is the equal-weight daily return of the available 50-company universe.

This avoids inserting SPY into the company dataset and altering the label universe. An explicit SPY-based beta will be incorporated in the dedicated risk-model stage.

### `max_drawdown_126d`

Worst peak-to-trough adjusted-price loss observed during the latest 126 sessions.

## Trend features

### `distance_sma_50d`

Current adjusted price divided by its 50-session simple moving average, minus one.

### `distance_sma_200d`

Current adjusted price divided by its 200-session simple moving average, minus one.

### `sma_50_200_spread`

Relative spread between the 50-session and 200-session simple moving averages:

`SMA50 / SMA200 - 1`

### `positive_day_ratio_60d`

Proportion of the latest 60 daily returns that are strictly positive.

## Liquidity features

### `average_dollar_volume_20d`

Average of:

`close × volume`

during the latest 20 sessions.

The unadjusted close is used for dollar-volume estimation, while adjusted close is used for return calculations.

### `dollar_volume_change_20d_60d`

Average dollar volume during the latest 20 sessions divided by the average dollar volume during the preceding 40 sessions, minus one.

### `amihud_illiquidity_20d`

Average absolute return per unit of dollar volume during the latest 20 sessions:

`mean(abs(return) / dollar_volume)`

Larger values indicate that relatively small traded amounts are associated with larger price movements.

### `zero_volume_ratio_60d`

Proportion of the latest 60 sessions with reported volume equal to zero.

## Missing values

Missing values are expected during the first part of the sample.

For example:

* A 20-session feature requires at least 21 prices to calculate 20 returns.
* `distance_sma_200d` requires 200 price observations.
* `momentum_12_1` requires at least 253 price observations.

These initial missing values must not be filled using future data. Their treatment will be defined inside the temporal modeling pipeline.
