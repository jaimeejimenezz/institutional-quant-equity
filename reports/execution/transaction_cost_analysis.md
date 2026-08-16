# Transaction Cost Analysis

## Scope

- Reference capital for sensitivity analysis: `$1,000,000`.
- Linear scenarios: `0`, `5`, `10`, `20` and `50` bps per dollar traded.
- Liquidity model: commission + half spread + slippage + volatility/ADV-dependent market impact.
- Liquidity-model linear component: `5.00` bps.
- Market-impact coefficient: `0.1000`.

## Cost sensitivity

| scenario | strategy_name | cagr | sharpe_ratio | maximum_drawdown | excess_cagr | mean_one_way_turnover | total_transaction_cost | realized_effective_cost_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_0bps | alpha_risk_turnover | 0.217495 | 1.025992 | -0.341612 | 0.060863 | 0.261831 | 0.000000 | 0.000000 |
| linear_5bps | alpha_risk_turnover | 0.213755 | 1.011639 | -0.341791 | 0.057213 | 0.261905 | 37842.654054 | 5.000000 |
| linear_10bps | alpha_risk_turnover | 0.210026 | 0.997274 | -0.341971 | 0.053574 | 0.261979 | 74799.797067 | 10.000000 |
| linear_20bps | alpha_risk_turnover | 0.202603 | 0.968507 | -0.342328 | 0.046332 | 0.262129 | 146133.548333 | 20.000000 |
| linear_50bps | alpha_risk_turnover | 0.180615 | 0.881981 | -0.343399 | 0.024885 | 0.262580 | 340787.006210 | 50.000000 |
| liquidity_model | alpha_risk_turnover | 0.213635 | 1.011181 | -0.341795 | 0.057093 | 0.261907 | 39028.737778 | 5.158666 |
| linear_0bps | cvar | 0.208046 | 0.991894 | -0.326130 | 0.051414 | 0.220972 | 0.000000 | 0.000000 |
| linear_5bps | cvar | 0.204928 | 0.979797 | -0.326285 | 0.048386 | 0.221036 | 30354.211816 | 5.000000 |
| linear_10bps | cvar | 0.201818 | 0.967691 | -0.326439 | 0.045366 | 0.221099 | 60113.749787 | 10.000000 |
| linear_20bps | cvar | 0.195622 | 0.943455 | -0.326749 | 0.039352 | 0.221227 | 117891.834106 | 20.000000 |
| linear_50bps | cvar | 0.177234 | 0.870587 | -0.327677 | 0.021504 | 0.221613 | 278039.840323 | 50.000000 |
| liquidity_model | cvar | 0.204828 | 0.979412 | -0.326287 | 0.048287 | 0.221038 | 31302.259949 | 5.157806 |
| linear_0bps | median_mad_de | 0.227046 | 1.067616 | -0.327156 | 0.070413 | 0.297272 | 0.000000 | 0.000000 |
| linear_5bps | median_mad_de | 0.222753 | 1.051150 | -0.327322 | 0.066211 | 0.297355 | 42580.589926 | 5.000000 |
| linear_10bps | median_mad_de | 0.218476 | 1.034666 | -0.327488 | 0.062025 | 0.297438 | 84022.398427 | 10.000000 |
| linear_20bps | median_mad_de | 0.209967 | 1.001652 | -0.327819 | 0.053697 | 0.297605 | 163601.513650 | 20.000000 |
| linear_50bps | median_mad_de | 0.184802 | 0.902310 | -0.328812 | 0.029072 | 0.298113 | 377791.415305 | 50.000000 |
| liquidity_model | median_mad_de | 0.222616 | 1.050621 | -0.327325 | 0.066074 | 0.297358 | 43920.972431 | 5.159636 |
| linear_0bps | score_weighted | 0.248167 | 1.100819 | -0.343324 | 0.091534 | 0.258935 | 0.000000 | 0.000000 |
| linear_5bps | score_weighted | 0.244375 | 1.087255 | -0.343495 | 0.087833 | 0.259019 | 40901.461687 | 5.000000 |
| linear_10bps | score_weighted | 0.240595 | 1.073679 | -0.343665 | 0.084143 | 0.259102 | 80837.344002 | 10.000000 |
| linear_20bps | score_weighted | 0.233069 | 1.046491 | -0.344006 | 0.076799 | 0.259269 | 157894.709245 | 20.000000 |
| linear_50bps | score_weighted | 0.210766 | 0.964681 | -0.345026 | 0.055037 | 0.259772 | 367957.004976 | 50.000000 |
| liquidity_model | score_weighted | 0.244270 | 1.086881 | -0.343498 | 0.087729 | 0.259021 | 42012.903422 | 5.137560 |
| linear_0bps | top_n_equal_weight | 0.236032 | 1.080286 | -0.336983 | 0.079400 | 0.241693 | 0.000000 | 0.000000 |
| linear_5bps | top_n_equal_weight | 0.232533 | 1.067303 | -0.337132 | 0.075992 | 0.241763 | 36827.176034 | 5.000000 |
| linear_10bps | top_n_equal_weight | 0.229045 | 1.054310 | -0.337281 | 0.072594 | 0.241833 | 72846.573068 | 10.000000 |
| linear_20bps | top_n_equal_weight | 0.222099 | 1.028292 | -0.337579 | 0.065828 | 0.241974 | 142526.716399 | 20.000000 |
| linear_50bps | top_n_equal_weight | 0.201502 | 0.950033 | -0.338471 | 0.045772 | 0.242400 | 333805.448872 | 50.000000 |
| liquidity_model | top_n_equal_weight | 0.232428 | 1.066914 | -0.337135 | 0.075886 | 0.241765 | 37921.482317 | 5.150280 |

## Capacity

| capital | strategy_name | net_cagr | cagr_cost_drag | net_sharpe_ratio | effective_cost_bps | maximum_order_adv_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 100000.000000 | score_weighted | 0.244342 | 0.003825 | 1.087137 | 5.043505 | 0.000043 |
| 100000.000000 | top_n_equal_weight | 0.232500 | 0.003532 | 1.067180 | 5.047528 | 0.000045 |
| 100000.000000 | median_mad_de | 0.222710 | 0.004336 | 1.050983 | 5.050489 | 0.000052 |
| 100000.000000 | alpha_risk_turnover | 0.213717 | 0.003778 | 1.011494 | 5.050181 | 0.000051 |
| 100000.000000 | cvar | 0.204896 | 0.003150 | 0.979675 | 5.049908 | 0.000048 |
| 1000000.000000 | score_weighted | 0.244270 | 0.003896 | 1.086881 | 5.137560 | 0.000434 |
| 1000000.000000 | top_n_equal_weight | 0.232428 | 0.003604 | 1.066914 | 5.150280 | 0.000449 |
| 1000000.000000 | median_mad_de | 0.222616 | 0.004430 | 1.050621 | 5.159636 | 0.000521 |
| 1000000.000000 | alpha_risk_turnover | 0.213635 | 0.003860 | 1.011181 | 5.158666 | 0.000507 |
| 1000000.000000 | cvar | 0.204828 | 0.003218 | 0.979412 | 5.157806 | 0.000483 |
| 10000000.000000 | score_weighted | 0.244044 | 0.004122 | 1.086072 | 5.434847 | 0.004332 |
| 10000000.000000 | top_n_equal_weight | 0.232201 | 0.003831 | 1.066071 | 5.475054 | 0.004489 |
| 10000000.000000 | median_mad_de | 0.222318 | 0.004727 | 1.049479 | 5.504577 | 0.005197 |
| 10000000.000000 | alpha_risk_turnover | 0.213378 | 0.004118 | 1.010191 | 5.501544 | 0.005066 |
| 10000000.000000 | cvar | 0.204614 | 0.003432 | 0.978580 | 5.498862 | 0.004825 |
| 100000000.000000 | score_weighted | 0.243331 | 0.004836 | 1.083516 | 6.373544 | 0.043166 |
| 100000000.000000 | top_n_equal_weight | 0.231485 | 0.004547 | 1.063409 | 6.500532 | 0.044739 |
| 100000000.000000 | median_mad_de | 0.221381 | 0.005665 | 1.045873 | 6.593260 | 0.051720 |
| 100000000.000000 | alpha_risk_turnover | 0.212564 | 0.004931 | 1.007064 | 6.583991 | 0.050465 |
| 100000000.000000 | cvar | 0.203937 | 0.004109 | 0.975951 | 6.575897 | 0.048096 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_rows | PASS | 0 | Sensitivity analysis must contain one row per method and cost scenario. |
| five_methods_per_scenario | PASS | 0 | Every cost scenario must contain all five portfolio methods. |
| finite_metrics | PASS | 0 | Key transaction-cost sensitivity metrics must remain finite. |
| positive_final_values | PASS | 0 | All strategies must retain positive final portfolio value. |
| nonnegative_transaction_costs | PASS | 0 | Transaction costs cannot be negative. |
| linear_cost_reduces_value | PASS | 0 | Final value must not increase when the linear cost assumption rises. |
| realized_cost_non_decreasing | PASS | 0 | Realized effective cost must not decline as the linear cost assumption rises. |
