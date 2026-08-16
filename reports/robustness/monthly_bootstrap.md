# Monthly Bootstrap Robustness

## Methodology

- `10,000` paired bootstrap replications over `77` aligned calendar months.
- The same sampled month indices are used for every strategy and SPY, preserving cross-strategy comparability.
- Equal-tailed `95%` confidence intervals.
- Bootstrap Sharpe is calculated from monthly returns, annualized by multiplying mean/std by sqrt(12), with zero risk-free rate for this robustness diagnostic.
- These intervals quantify sampling uncertainty inside the observed out-of-sample period; they are not guarantees of future performance.

## Strategy bootstrap summary

| strategy_name | observed_annualized_return | annualized_return_ci_lower | annualized_return_ci_upper | observed_excess_annualized_return | excess_annualized_return_ci_lower | excess_annualized_return_ci_upper | probability_excess_annualized_return_positive | observed_sharpe | sharpe_ci_lower | sharpe_ci_upper | observed_positive_month_ratio | observed_outperform_spy_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_weighted | 0.243094 | 0.066543 | 0.439937 | 0.087280 | 0.040398 | 0.141414 | 1.000000 | 1.235600 | 0.428067 | 2.148224 | 0.675325 | 0.688312 |
| top_n_equal_weight | 0.231314 | 0.061211 | 0.420378 | 0.075500 | 0.031905 | 0.124119 | 0.999700 | 1.211959 | 0.405764 | 2.112679 | 0.688312 | 0.662338 |
| median_mad_de | 0.221552 | 0.063499 | 0.401349 | 0.065738 | 0.019179 | 0.117638 | 0.997700 | 1.210115 | 0.426217 | 2.075499 | 0.649351 | 0.623377 |
| alpha_risk_turnover | 0.212619 | 0.052248 | 0.393517 | 0.056805 | 0.010630 | 0.106580 | 0.992400 | 1.160484 | 0.365638 | 2.056351 | 0.662338 | 0.558442 |
| cvar | 0.203857 | 0.039456 | 0.390964 | 0.048043 | -0.004572 | 0.106422 | 0.963700 | 1.081479 | 0.294145 | 1.949939 | 0.662338 | 0.571429 |

## Pairwise comparison

| method_a | method_b | mean_cagr_difference_a_minus_b | cagr_difference_ci_lower | cagr_difference_ci_upper | probability_cagr_a_greater_b | mean_sharpe_difference_a_minus_b | sharpe_difference_ci_lower | sharpe_difference_ci_upper | probability_sharpe_a_greater_b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alpha_risk_turnover | cvar | 0.008424 | -0.020192 | 0.035896 | 0.726300 | 0.083162 | -0.051386 | 0.225383 | 0.886400 |
| alpha_risk_turnover | median_mad_de | -0.009131 | -0.041253 | 0.022254 | 0.286500 | -0.046322 | -0.188995 | 0.102452 | 0.264400 |
| alpha_risk_turnover | score_weighted | -0.030644 | -0.067449 | 0.002693 | 0.036000 | -0.075502 | -0.220093 | 0.060259 | 0.138300 |
| alpha_risk_turnover | top_n_equal_weight | -0.018759 | -0.053311 | 0.012708 | 0.129800 | -0.050657 | -0.192350 | 0.088722 | 0.236200 |
| cvar | median_mad_de | -0.017556 | -0.046338 | 0.013131 | 0.120900 | -0.129485 | -0.253478 | -0.005193 | 0.021900 |
| cvar | score_weighted | -0.039068 | -0.079746 | -0.002391 | 0.018500 | -0.158664 | -0.339206 | 0.001723 | 0.026300 |
| cvar | top_n_equal_weight | -0.027183 | -0.063737 | 0.008131 | 0.059800 | -0.133820 | -0.293615 | 0.013176 | 0.037200 |
| median_mad_de | score_weighted | -0.021512 | -0.072036 | 0.023026 | 0.182400 | -0.029180 | -0.228688 | 0.146658 | 0.393400 |
| median_mad_de | top_n_equal_weight | -0.009627 | -0.055102 | 0.031602 | 0.336400 | -0.004335 | -0.182321 | 0.156043 | 0.498300 |
| score_weighted | top_n_equal_weight | 0.011885 | -0.000500 | 0.024989 | 0.969800 | 0.024845 | -0.027674 | 0.081474 | 0.813200 |

## Rank stability

| strategy_name | probability_rank_1_by_annualized_return | probability_rank_1_by_sharpe |
| --- | --- | --- |
| score_weighted | 0.795400 | 0.517200 |
| median_mad_de | 0.175500 | 0.360600 |
| top_n_equal_weight | 0.014200 | 0.071700 |
| alpha_risk_turnover | 0.011300 | 0.049700 |
| cvar | 0.003600 | 0.000800 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| minimum_months | PASS | 0 | Bootstrap analysis should contain at least 60 aligned months. |
| expected_methods | PASS | 0 | Bootstrap summary must contain all five portfolio methods. |
| expected_replications | PASS | 0 | Every method must use the configured number of bootstrap replications. |
| finite_summary_metrics | PASS | 0 | Bootstrap summary metrics must be finite. |
| confidence_interval_order | PASS | 0 | Bootstrap confidence-interval lower bounds must not exceed upper bounds. |
| valid_summary_probabilities | PASS | 0 | Bootstrap strategy probabilities must lie between zero and one. |
| valid_pairwise_probabilities | PASS | 0 | Pairwise bootstrap probabilities must lie between zero and one. |
| rank_probabilities_sum_to_one | PASS | 0 | Bootstrap rank-one probabilities must sum to one for each metric. |
