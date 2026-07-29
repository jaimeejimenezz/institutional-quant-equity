# Technical Features Quality Report

- Generated UTC: 2026-07-29T18:57:23+00:00
- Status: **PASS**
- Winsorization: 1.00% to 99.00%
- Minimum cross-section size: `10`
- Sector neutralization: `True`

## Summary

| metric | value |
| --- | --- |
| rows | 7550 |
| dates | 151 |
| tickers | 50 |
| sectors | 11 |
| raw_features | 19 |
| model_features | 19 |
| first_as_of_date | 2014-01-31 |
| last_as_of_date | 2026-07-27 |
| duplicate_rows | 0 |
| temporal_violations | 0 |
| missing_sector_rows | 0 |
| changed_raw_values | 0 |
| non_finite_values | 0 |
| incorrect_winsorized_values | 0 |
| invalid_standardized_dates | 0 |
| invalid_sector_neutral_groups | 0 |
| incomplete_universe_dates | 0 |
| small_sector_groups | 151 |

## Blocking issues

- No blocking validation issues.

## Warnings

- 151 date-sector groups are too small for sector neutralization and retain their global z-score.

## Coverage by feature

| feature | raw_non_missing | raw_missing_ratio | winsorized_non_missing | zscore_non_missing | sector_neutral_non_missing | clipped_observations | first_available_date | last_available_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| momentum_12_1 | 6950 | 0.079470198675 | 6950 | 6950 | 6950 | 278 | 2015-01-30 | 2026-07-27 |
| momentum_6_1 | 7250 | 0.039735099338 | 7250 | 7250 | 7250 | 290 | 2014-07-31 | 2026-07-27 |
| return_3m | 7400 | 0.019867549669 | 7400 | 7400 | 7400 | 296 | 2014-04-30 | 2026-07-27 |
| return_1m | 7500 | 0.006622516556 | 7500 | 7500 | 7500 | 300 | 2014-02-28 | 2026-07-27 |
| return_1w | 7550 | 0.000000000000 | 7550 | 7550 | 7550 | 302 | 2014-01-31 | 2026-07-27 |
| reversal_1m | 7500 | 0.006622516556 | 7500 | 7500 | 7500 | 300 | 2014-02-28 | 2026-07-27 |
| volatility_20d | 7550 | 0.000000000000 | 7550 | 7550 | 7550 | 302 | 2014-01-31 | 2026-07-27 |
| volatility_60d | 7450 | 0.013245033113 | 7450 | 7450 | 7450 | 298 | 2014-03-31 | 2026-07-27 |
| downside_volatility_60d | 7450 | 0.013245033113 | 7450 | 7450 | 7450 | 298 | 2014-03-31 | 2026-07-27 |
| beta_60d_market | 7450 | 0.013245033113 | 7450 | 7450 | 7450 | 298 | 2014-03-31 | 2026-07-27 |
| max_drawdown_126d | 7250 | 0.039735099338 | 7250 | 7250 | 7250 | 290 | 2014-07-31 | 2026-07-27 |
| distance_sma_50d | 7450 | 0.013245033113 | 7450 | 7450 | 7450 | 298 | 2014-03-31 | 2026-07-27 |
| distance_sma_200d | 7100 | 0.059602649007 | 7100 | 7100 | 7100 | 284 | 2014-10-31 | 2026-07-27 |
| sma_50_200_spread | 7100 | 0.059602649007 | 7100 | 7100 | 7100 | 284 | 2014-10-31 | 2026-07-27 |
| positive_day_ratio_60d | 7450 | 0.013245033113 | 7450 | 7450 | 7450 | 212 | 2014-03-31 | 2026-07-27 |
| average_dollar_volume_20d | 7550 | 0.000000000000 | 7550 | 7550 | 7550 | 302 | 2014-01-31 | 2026-07-27 |
| dollar_volume_change_20d_60d | 7450 | 0.013245033113 | 7450 | 7450 | 7450 | 298 | 2014-03-31 | 2026-07-27 |
| amihud_illiquidity_20d | 7550 | 0.000000000000 | 7550 | 7550 | 7550 | 147 | 2014-01-31 | 2026-07-27 |
| zero_volume_ratio_60d | 7450 | 0.013245033113 | 7450 | 7450 | 7450 | 0 | 2014-03-31 | 2026-07-27 |

## Processing interpretation

Winsorization, standardization and sector adjustment are calculated independently for every rebalance date.

No future month contributes to the transformation of a previous month.

Raw feature values are retained in the processed dataset for auditability.

Missing feature values remain missing. No future-aware imputation is applied during this processing stage.
