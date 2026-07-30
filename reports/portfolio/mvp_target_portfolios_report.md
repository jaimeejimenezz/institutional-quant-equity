# MVP Target Portfolios — Step 8A

## Objective

Transform the out-of-sample model predictions into monthly long-only target portfolios before running the daily execution backtest.

## Frozen portfolio constraints

- Top-N holdings: `20`
- Score-weighted candidate count: `25`
- Maximum company weight: `5.00%`
- Maximum sector weight: `25.00%`
- Primary model: `elastic_net`
- Challenger model: `ridge`

## Strategy summary

| strategy_name | months | mean_holdings | minimum_holdings | maximum_holdings | mean_maximum_weight | maximum_observed_weight | mean_maximum_sector_weight | maximum_observed_sector_weight | mean_top_ten_concentration | constraint_pass_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elastic_net_score_weighted | 77 | 25.000000 | 25 | 25 | 0.050000 | 0.050000 | 0.167991 | 0.229593 | 0.499599 | 1.000000 |
| elastic_net_top20_equal_weight | 77 | 20.000000 | 20 | 20 | 0.050000 | 0.050000 | 0.173377 | 0.250000 | 0.500000 | 1.000000 |
| momentum_3m_top20_equal_weight | 77 | 20.000000 | 20 | 20 | 0.050000 | 0.050000 | 0.180519 | 0.250000 | 0.500000 | 1.000000 |
| ridge_top20_equal_weight | 77 | 20.000000 | 20 | 20 | 0.050000 | 0.050000 | 0.177273 | 0.250000 | 0.500000 | 1.000000 |
| universe_equal_weight | 77 | 50.000000 | 50 | 50 | 0.020000 | 0.020000 | 0.160000 | 0.160000 | 0.200000 | 1.000000 |

## Latest target portfolios

| strategy_name | as_of_date | holdings | maximum_weight | maximum_sector_weight | constraints_pass |
| --- | --- | --- | --- | --- | --- |
| elastic_net_score_weighted | 2026-05-29 | 25 | 0.050000 | 0.200000 | True |
| elastic_net_top20_equal_weight | 2026-05-29 | 20 | 0.050000 | 0.200000 | True |
| momentum_3m_top20_equal_weight | 2026-05-29 | 20 | 0.050000 | 0.250000 | True |
| ridge_top20_equal_weight | 2026-05-29 | 20 | 0.050000 | 0.150000 | True |
| universe_equal_weight | 2026-05-29 | 50 | 0.020000 | 0.160000 | True |

## Interpretation

These are target weights, not realized daily positions. The following step must map every signal date to the next trading session, create orders, model shares and cash, apply transaction costs, and allow weights to drift between rebalances.
