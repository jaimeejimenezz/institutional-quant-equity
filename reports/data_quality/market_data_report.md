# Market Data Quality Report

- Generated UTC: 2026-07-28T19:25:08+00:00
- Status: **PASS**
- Provider: `yfinance`
- Requested start: `2014-01-01`
- Exclusive end: `2026-07-28`

## Summary

| metric | value |
| --- | --- |
| rows | 157950 |
| tickers | 50 |
| unique_dates | 3159 |
| first_date | 2014-01-02 |
| last_date | 2026-07-27 |
| duplicate_rows | 0 |
| invalid_date_rows | 0 |
| invalid_price_rows | 0 |
| negative_volume_rows | 0 |
| missing_core_values | 80 |
| missing_ratio | 0.000084 |
| out_of_range_rows | 0 |
| ohlc_inconsistency_rows | 0 |
| extreme_return_rows | 2 |
| adjustment_anomaly_rows | 0 |

## Validation issues

- No blocking validation issues.

## Warnings

- 80 missing core values were found.
- 2 extreme adjusted returns require review.

## Coverage by ticker

| ticker | observations | first_date | last_date | coverage_ratio | missing_core_values | extreme_return_count | adjustment_anomaly_count | short_history | low_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| ABBV | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| ABT | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| AMZN | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| APD | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| AVGO | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| BAC | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| BRK-B | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| CAT | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| COP | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| COST | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| CSCO | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| CVX | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| DIS | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| EOG | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| GOOGL | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| GS | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| HD | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| HON | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| JNJ | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| JPM | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| KO | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| LLY | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| LOW | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| MA | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| MCD | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| META | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| MRK | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| MSFT | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| NEE | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| NFLX | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 1 | 0 | False | False |
| NKE | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| NVDA | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| ORCL | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 1 | 0 | False | False |
| PEP | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| PG | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| PLD | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| QCOM | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| RTX | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| SHW | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| SO | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| TMO | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| TSLA | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| TXN | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| UNH | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| UNP | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 5 | 0 | 0 | False | False |
| UPS | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| V | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| WFC | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |
| WMT | 3159 | 2014-01-02 | 2026-07-27 | 1.000000 | 0 | 0 | 0 | False | False |

## Largest absolute adjusted returns

| date | ticker | adjusted_close | return |
| --- | --- | --- | --- |
| 2025-09-10 | ORCL | 324.629974 | 0.359488 |
| 2022-04-20 | NFLX | 22.618999 | -0.351166 |

## Adjustment-factor anomalies

_No observations._

## Interpretation

Extreme returns and adjustment-factor changes are reported for manual review. Their presence does not automatically imply an error because corporate events and genuine market movements may produce large changes.

A passing report confirms the configured mechanical quality checks. It does not certify that the provider is institutionally complete or point-in-time perfect.
