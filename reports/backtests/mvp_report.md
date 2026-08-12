# MVP Backtest Report — Step 8C

## Evaluation design

- All model portfolios use genuinely out-of-sample predictions.
- Signals are executed at the adjusted opening price of the following session.
- Positions drift between monthly rebalances.
- Fractional shares are allowed.
- The primary comparison uses transaction costs of `10.00` basis points.
- SPY is evaluated as a buy-and-hold benchmark from the first execution date.

## Main performance comparison

| strategy_name | total_return | cagr | annualized_volatility | sharpe_ratio | sortino_ratio | maximum_drawdown | calmar_ratio | beta_vs_spy | information_ratio_vs_spy | total_transaction_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elastic_net_top20_equal_weight | 3.868540 | 0.281129 | 0.222526 | 1.225304 | 1.782977 | -0.338997 | 0.829297 | 1.043980 | 1.639567 | 93482.426063 |
| ridge_top20_equal_weight | 3.377238 | 0.259975 | 0.220211 | 1.160157 | 1.686935 | -0.339768 | 0.765154 | 1.033050 | 1.393843 | 103356.085849 |
| elastic_net_score_weighted | 3.353621 | 0.258908 | 0.220933 | 1.153278 | 1.672699 | -0.335369 | 0.772008 | 1.041392 | 1.451471 | 81174.579399 |
| universe_equal_weight | 1.984878 | 0.186687 | 0.194807 | 0.976647 | 1.395862 | -0.329057 | 0.567341 | 0.927611 | 0.496458 | 8532.951189 |
| momentum_3m_top20_equal_weight | 1.615747 | 0.162419 | 0.199070 | 0.856247 | 1.204647 | -0.345021 | 0.470752 | 0.919736 | 0.061254 | 91593.732369 |
| spy_buy_and_hold | 1.531127 | 0.156451 | 0.204068 | 0.814763 | 1.153478 | -0.337173 | 0.464010 | 1.000000 |  | 999.000999 |

## Calendar-year returns

| year | elastic_net_score_weighted | elastic_net_top20_equal_weight | momentum_3m_top20_equal_weight | ridge_top20_equal_weight | spy_buy_and_hold | universe_equal_weight |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 | 0.353532 | 0.335434 | 0.335845 | 0.327550 | 0.176687 | 0.264177 |
| 2021 | 0.437237 | 0.468725 | 0.362614 | 0.478836 | 0.287288 | 0.355953 |
| 2022 | -0.112537 | -0.081553 | -0.163740 | -0.091794 | -0.181754 | -0.098484 |
| 2023 | 0.398219 | 0.388405 | 0.197224 | 0.424909 | 0.261758 | 0.241448 |
| 2024 | 0.352636 | 0.378433 | 0.228765 | 0.303076 | 0.248865 | 0.244293 |
| 2025 | 0.217817 | 0.260841 | 0.101598 | 0.222991 | 0.177191 | 0.159975 |
| 2026 | 0.094877 | 0.120004 | 0.060371 | 0.081097 | 0.100919 | 0.077955 |

## Cost sensitivity

| cost_bps | strategy_name | cagr | sharpe_ratio | maximum_drawdown | final_portfolio_value | total_transaction_cost |
| --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | elastic_net_score_weighted | 0.266658 | 1.181022 | -0.334924 | 4527719.371363 | 0.000000 |
| 0.000000 | elastic_net_top20_equal_weight | 0.289785 | 1.255545 | -0.338443 | 5082570.590591 | 0.000000 |
| 0.000000 | momentum_3m_top20_equal_weight | 0.171792 | 0.896552 | -0.344867 | 2753460.313566 | 0.000000 |
| 0.000000 | ridge_top20_equal_weight | 0.269353 | 1.193819 | -0.339212 | 4589617.952480 | 0.000000 |
| 0.000000 | spy_buy_and_hold | 0.156632 | 0.815528 | -0.337173 | 2533658.387733 | 0.000000 |
| 0.000000 | universe_equal_weight | 0.187628 | 0.980700 | -0.329028 | 3000018.596032 | 0.000000 |
| 5.000000 | elastic_net_score_weighted | 0.262777 | 1.167158 | -0.335147 | 4439815.337110 | 41084.703629 |
| 5.000000 | elastic_net_top20_equal_weight | 0.285450 | 1.240435 | -0.338720 | 4974401.219434 | 47380.137010 |
| 5.000000 | momentum_3m_top20_equal_weight | 0.167097 | 0.876413 | -0.344944 | 2683724.057985 | 46469.202663 |
| 5.000000 | ridge_top20_equal_weight | 0.264655 | 1.177000 | -0.339490 | 4482165.874780 | 52447.736615 |
| 5.000000 | spy_buy_and_hold | 0.156542 | 0.815146 | -0.337173 | 2532392.191637 | 499.750125 |
| 5.000000 | universe_equal_weight | 0.187157 | 0.978674 | -0.329043 | 2992438.033806 | 4273.181676 |
| 10.000000 | elastic_net_score_weighted | 0.258908 | 1.153278 | -0.335369 | 4353621.353674 | 81174.579399 |
| 10.000000 | elastic_net_top20_equal_weight | 0.281129 | 1.225304 | -0.338997 | 4868540.214600 | 93482.426063 |
| 10.000000 | momentum_3m_top20_equal_weight | 0.162419 | 0.856247 | -0.345021 | 2615746.903777 | 91593.732369 |
| 10.000000 | ridge_top20_equal_weight | 0.259975 | 1.160157 | -0.339768 | 4377238.067522 | 103356.085849 |
| 10.000000 | spy_buy_and_hold | 0.156451 | 0.814763 | -0.337173 | 2531127.260472 | 999.000999 |
| 10.000000 | universe_equal_weight | 0.186687 | 0.976647 | -0.329057 | 2984877.591689 | 8532.951189 |
| 20.000000 | elastic_net_score_weighted | 0.251206 | 1.125470 | -0.335815 | 4186230.053618 | 158456.518219 |
| 20.000000 | elastic_net_top20_equal_weight | 0.272532 | 1.194984 | -0.339550 | 4663545.224508 | 181977.007035 |
| 20.000000 | momentum_3m_top20_equal_weight | 0.153119 | 0.815841 | -0.345175 | 2484892.537161 | 177953.329719 |
| 20.000000 | ridge_top20_equal_weight | 0.250667 | 1.126409 | -0.340322 | 4174717.224430 | 200716.591657 |
| 20.000000 | spy_buy_and_hold | 0.156271 | 0.813997 | -0.337173 | 2528601.185362 | 1996.007984 |
| 20.000000 | universe_equal_weight | 0.185748 | 0.972591 | -0.329085 | 2969816.822631 | 17012.429247 |
| 50.000000 | elastic_net_score_weighted | 0.228384 | 1.041715 | -0.337148 | 3721740.523743 | 368601.321409 |
| 50.000000 | elastic_net_top20_equal_weight | 0.247089 | 1.103612 | -0.341204 | 4099001.680836 | 419873.622606 |
| 50.000000 | momentum_3m_top20_equal_weight | 0.125650 | 0.694182 | -0.345636 | 2130145.567784 | 408341.826122 |
| 50.000000 | ridge_top20_equal_weight | 0.223161 | 1.024741 | -0.341982 | 3621793.021105 | 459881.007669 |
| 50.000000 | spy_buy_and_hold | 0.155730 | 0.811691 | -0.337173 | 2521053.122122 | 4975.124378 |
| 50.000000 | universe_equal_weight | 0.182936 | 0.960406 | -0.329169 | 2925111.149796 | 42133.408418 |

## Interpretation rule

The preferred MVP strategy should not be chosen only from final capital. It should combine a competitive CAGR, Sharpe and Sortino ratio, controlled drawdown, reasonable turnover, positive performance across several years and robustness under higher transaction costs.
