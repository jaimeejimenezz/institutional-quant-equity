# Portfolio Execution Comparison

## Assumptions

- Initial capital: `$1,000,000`.
- Linear execution costs: `5.00` bps per dollar traded.
- Market impact coefficient: `0.1000`.
- Market impact uses daily volatility and the square root of order size divided by ADV.
- Gross portfolios use zero transaction costs. Net portfolios use the advanced execution model.
- SPY net performance includes the same linear execution cost on its initial buy-and-hold trade; no market-impact term is applied to SPY.

## Gross versus net performance

| strategy_name | gross_cagr | net_cagr | cagr_cost_drag | net_sharpe_ratio | net_sortino_ratio | net_maximum_drawdown | net_beta_vs_spy | net_excess_cagr | mean_one_way_turnover | total_transaction_cost | effective_cost_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_weighted | 0.248167 | 0.244270 | 0.003896 | 1.086881 | 1.555182 | -0.343498 | 1.062284 | 0.087729 | 0.259021 | 42012.903422 | 5.137560 |
| top_n_equal_weight | 0.236032 | 0.232428 | 0.003604 | 1.066914 | 1.527611 | -0.337135 | 1.036552 | 0.075886 | 0.241765 | 37921.482317 | 5.150280 |
| median_mad_de | 0.227046 | 0.222616 | 0.004430 | 1.050621 | 1.504663 | -0.327325 | 1.001235 | 0.066074 | 0.297358 | 43920.972431 | 5.159636 |
| alpha_risk_turnover | 0.217495 | 0.213635 | 0.003860 | 1.011181 | 1.439904 | -0.341795 | 1.003290 | 0.057093 | 0.261907 | 39028.737778 | 5.158666 |
| cvar | 0.208046 | 0.204828 | 0.003218 | 0.979412 | 1.395714 | -0.326287 | 1.004282 | 0.048287 | 0.221038 | 31302.259949 | 5.157806 |

## Execution-cost decomposition

| strategy_name | trades | traded_notional | commission_cost | spread_cost | slippage_cost | market_impact_cost | total_execution_cost | maximum_order_adv_fraction | effective_cost_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alpha_risk_turnover | 2004 | 75656641.482621 | 3782.832074 | 15131.328297 | 18914.160371 | 1200.417036 | 39028.737778 | 0.000507 | 5.158666 |
| cvar | 2117 | 60689099.961127 | 3034.454998 | 12137.819992 | 15172.274990 | 957.709909 | 31302.259889 | 0.000483 | 5.157806 |
| median_mad_de | 2055 | 85124168.664478 | 4256.208433 | 17024.833733 | 21281.042166 | 1358.888099 | 43920.972431 | 0.000521 | 5.159636 |
| score_weighted | 2314 | 81775981.914862 | 4088.799096 | 16355.196383 | 20443.995479 | 1124.912465 | 42012.903422 | 0.000434 | 5.137560 |
| top_n_equal_weight | 2314 | 73629944.631301 | 3681.497232 | 14725.988926 | 18407.486158 | 1106.510001 | 37921.482317 | 0.000449 | 5.150280 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_methods | PASS | 0 | Gross and net backtests must contain all five portfolio methods. |
| aligned_backtest_dates | PASS | 0 | All methods must use identical trading dates. |
| expected_rebalances | PASS | 0 | Every method must execute every scheduled rebalance. |
| positive_portfolio_values | PASS | 0 | Gross and net portfolio values must remain positive. |
| nonnegative_transaction_costs | PASS | 0 | Execution costs cannot be negative. |
| finite_transaction_costs | PASS | 0 | Execution costs must be finite. |
| cash_accounting | PASS | 0 | Residual cash must remain within the configured accounting tolerance. |
| comparison_rows | PASS | 0 | The final comparison must contain one row per portfolio method. |
