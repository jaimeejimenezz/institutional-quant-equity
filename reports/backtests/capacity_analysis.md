# Capacity Analysis

## Execution assumptions

- Linear execution cost: `5.00` bps.
- Market impact: coefficient × daily volatility × square root of order notional divided by ADV.
- Capital levels: $100k, $1M, $10M and $100M.
- Portfolio weights, rebalance dates and signals are held constant across capital scenarios.

## Capacity results

| capital | strategy_name | gross_cagr | net_cagr | cagr_cost_drag | net_sharpe_ratio | net_maximum_drawdown | net_excess_cagr | effective_cost_bps | market_impact_cost | maximum_order_adv_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100000.000000 | score_weighted | 0.248167 | 0.244342 | 0.003825 | 1.087137 | -0.343496 | 0.087800 | 5.043505 | 35.584925 | 0.000043 |
| 100000.000000 | top_n_equal_weight | 0.236032 | 0.232500 | 0.003532 | 1.067180 | -0.337133 | 0.075958 | 5.047528 | 35.002870 | 0.000045 |
| 100000.000000 | median_mad_de | 0.227046 | 0.222710 | 0.004336 | 1.050983 | -0.327323 | 0.066168 | 5.050489 | 42.990953 | 0.000052 |
| 100000.000000 | alpha_risk_turnover | 0.217495 | 0.213717 | 0.003778 | 1.011494 | -0.341793 | 0.057175 | 5.050181 | 37.975228 | 0.000051 |
| 100000.000000 | cvar | 0.208046 | 0.204896 | 0.003150 | 0.979675 | -0.326286 | 0.048354 | 5.049908 | 30.295202 | 0.000048 |
| 1000000.000000 | score_weighted | 0.248167 | 0.244270 | 0.003896 | 1.086881 | -0.343498 | 0.087729 | 5.137560 | 1124.912465 | 0.000434 |
| 1000000.000000 | top_n_equal_weight | 0.236032 | 0.232428 | 0.003604 | 1.066914 | -0.337135 | 0.075886 | 5.150280 | 1106.510001 | 0.000449 |
| 1000000.000000 | median_mad_de | 0.227046 | 0.222616 | 0.004430 | 1.050621 | -0.327325 | 0.066074 | 5.159636 | 1358.888099 | 0.000521 |
| 1000000.000000 | alpha_risk_turnover | 0.217495 | 0.213635 | 0.003860 | 1.011181 | -0.341795 | 0.057093 | 5.158666 | 1200.417036 | 0.000507 |
| 1000000.000000 | cvar | 0.208046 | 0.204828 | 0.003218 | 0.979412 | -0.326287 | 0.048287 | 5.157806 | 957.709909 | 0.000483 |
| 10000000.000000 | score_weighted | 0.248167 | 0.244044 | 0.004122 | 1.086072 | -0.343505 | 0.087502 | 5.434847 | 35534.732109 | 0.004332 |
| 10000000.000000 | top_n_equal_weight | 0.236032 | 0.232201 | 0.003831 | 1.066071 | -0.337141 | 0.075660 | 5.475054 | 34953.166788 | 0.004489 |
| 10000000.000000 | median_mad_de | 0.227046 | 0.222318 | 0.004727 | 1.049479 | -0.327332 | 0.065777 | 5.504577 | 42911.386345 | 0.005197 |
| 10000000.000000 | alpha_risk_turnover | 0.217495 | 0.213378 | 0.004118 | 1.010191 | -0.341804 | 0.056836 | 5.501544 | 37914.067794 | 0.005066 |
| 10000000.000000 | cvar | 0.208046 | 0.204614 | 0.003432 | 0.978580 | -0.326294 | 0.048072 | 5.498862 | 30254.632007 | 0.004825 |
| 100000000.000000 | score_weighted | 0.248167 | 0.243331 | 0.004836 | 1.083516 | -0.343526 | 0.086789 | 6.373544 | 1119908.175587 | 0.043166 |
| 100000000.000000 | top_n_equal_weight | 0.236032 | 0.231485 | 0.004547 | 1.063409 | -0.337160 | 0.074943 | 6.500532 | 1101554.646755 | 0.044739 |
| 100000000.000000 | median_mad_de | 0.227046 | 0.221381 | 0.005665 | 1.045873 | -0.327353 | 0.064839 | 6.593260 | 1350962.974777 | 0.051720 |
| 100000000.000000 | alpha_risk_turnover | 0.217495 | 0.212564 | 0.004931 | 1.007064 | -0.341831 | 0.056022 | 6.583991 | 1194322.073544 | 0.050465 |
| 100000000.000000 | cvar | 0.208046 | 0.203937 | 0.004109 | 0.975951 | -0.326314 | 0.047395 | 6.575897 | 953664.667818 | 0.048096 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_rows | PASS | 0 | Capacity analysis must contain one row per method and capital level. |
| five_methods_per_capital | PASS | 0 | Every capital level must contain all five portfolio methods. |
| positive_final_values | PASS | 0 | All capacity scenarios must retain positive portfolio value. |
| finite_capacity_metrics | PASS | 0 | Key capacity metrics must remain finite. |
| nonnegative_costs | PASS | 0 | Transaction costs cannot be negative. |
| cash_accounting | PASS | 0 | Residual cash must remain within the same absolute-or-relative accounting tolerance used by the backtest engine. |
| cost_bps_non_decreasing | PASS | 0 | Effective execution cost should not decline as capital increases. |
| adv_participation_non_decreasing | PASS | 0 | Maximum order-to-ADV participation should not decline as capital increases. |
