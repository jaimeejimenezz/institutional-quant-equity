# Rebalance Frequency Sensitivity

## Methodology

- Monthly signal dates available: `77`.
- Calendar quarter-end signal dates used: `25`.
- Quarterly signals are the March, June, September and December month-end portfolios from the frozen OOS weights.
- Both frequencies are evaluated on the identical daily window `2020-04-01` to `2026-04-30`.
- Portfolio construction, final alpha signal, risk inputs and advanced execution-cost assumptions are unchanged.
- This is a sensitivity analysis only; the frozen production baseline is not changed based on these OOS results.

## Results

| strategy_name | monthly_cagr | quarterly_cagr | quarterly_minus_monthly_cagr | monthly_sharpe_ratio | quarterly_sharpe_ratio | quarterly_minus_monthly_sharpe | monthly_maximum_drawdown | quarterly_maximum_drawdown | monthly_rebalances | quarterly_rebalances | monthly_total_transaction_cost | quarterly_total_transaction_cost | transaction_cost_reduction_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alpha_risk_turnover | 0.266331 | 0.269913 | 0.003582 | 1.380371 | 1.387538 | 0.007167 | -0.209722 | -0.231602 | 73 | 25 | 36339.633105 | 22869.613680 | 0.370670 |
| cvar | 0.254002 | 0.268648 | 0.014646 | 1.310075 | 1.371211 | 0.061136 | -0.228194 | -0.249485 | 73 | 25 | 29098.575478 | 20084.026498 | 0.309793 |
| median_mad_de | 0.271371 | 0.253729 | -0.017642 | 1.405976 | 1.314289 | -0.091687 | -0.208140 | -0.256399 | 73 | 25 | 41136.093929 | 21997.685561 | 0.465246 |
| score_weighted | 0.297695 | 0.302221 | 0.004526 | 1.437589 | 1.457132 | 0.019543 | -0.231467 | -0.258762 | 73 | 25 | 39008.671047 | 21593.095267 | 0.446454 |
| top_n_equal_weight | 0.285221 | 0.281431 | -0.003791 | 1.427193 | 1.408487 | -0.018706 | -0.227030 | -0.255374 | 73 | 25 | 35310.237268 | 19557.693037 | 0.446118 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_methods | PASS | 0 | Monthly and quarterly scenarios must contain all five portfolio methods. |
| calendar_quarter_signals | PASS | 0 | Quarterly portfolios must use March, June, September and December signals only. |
| quarterly_has_fewer_signals | PASS | 0 | Quarterly rebalancing must use fewer signal dates than monthly rebalancing. |
| fully_invested_monthly | PASS | 0 | Monthly target portfolios must sum to one. |
| fully_invested_quarterly | PASS | 0 | Quarterly target portfolios must sum to one. |
| identical_comparison_window | PASS | 0 | Monthly and quarterly performance must be compared on identical daily dates. |
| comparison_rows | PASS | 0 | The final comparison must contain one row per portfolio method. |
| finite_comparison_metrics | PASS | 0 | Key rebalance-frequency metrics must remain finite. |
| quarterly_fewer_rebalances | PASS | 0 | Quarterly scenarios must execute fewer rebalances than monthly scenarios. |
| positive_final_values | PASS | 0 | Monthly and quarterly strategies must retain positive final value. |
