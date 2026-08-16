# Prediction Horizon Robustness

## Methodology

- The production alpha signal remains completely frozen. No model is retrained or re-tuned for this experiment.
- Realized cross-sectional excess returns are reconstructed at 10, 21 and 42 market-session horizons.
- All horizon comparisons use the same `76` out-of-sample dates and the same 50 securities per date.
- This tests temporal persistence of the existing signal. It does not claim that a separately trained 10- or 42-session model would produce identical results.
- Confidence intervals use `10,000` circular block-bootstrap replications with `3`-month blocks.

## Horizon summary

| horizon_sessions | months | mean_ic | mean_ic_ci_lower | mean_ic_ci_upper | probability_mean_ic_positive | annualized_ic_ir | positive_ic_ratio | mean_top_bottom_spread | spread_ci_lower | spread_ci_upper | probability_mean_spread_positive | positive_spread_ratio | mean_top_quintile_precision | mean_ic_difference_vs_21d | spread_difference_vs_21d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 76 | 0.050158 | 0.008207 | 0.089955 | 0.989500 | 0.850727 | 0.605263 | 0.008544 | 0.002771 | 0.013921 | 0.998300 | 0.592105 | 0.280263 | 0.006286 | -0.004808 |
| 21 | 76 | 0.043872 | 0.000989 | 0.085528 | 0.977600 | 0.730163 | 0.552632 | 0.013352 | 0.004422 | 0.022067 | 0.998100 | 0.605263 | 0.260526 | 0.000000 | 0.000000 |
| 42 | 76 | 0.052615 | 0.008796 | 0.095579 | 0.988900 | 1.043328 | 0.618421 | 0.027831 | 0.012544 | 0.043065 | 0.999600 | 0.671053 | 0.284211 | 0.008743 | 0.014479 |

## Calendar-year stability

| horizon_sessions | year | months | mean_ic | positive_ic_ratio | mean_top_bottom_spread | positive_spread_ratio | mean_top_quintile_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 2020 | 12 | 0.081353 | 0.583333 | 0.016278 | 0.666667 | 0.275000 |
| 21 | 2020 | 12 | 0.080352 | 0.666667 | 0.028384 | 0.750000 | 0.283333 |
| 42 | 2020 | 12 | 0.053245 | 0.666667 | 0.054318 | 0.916667 | 0.291667 |
| 10 | 2021 | 12 | 0.033309 | 0.583333 | 0.003217 | 0.583333 | 0.316667 |
| 21 | 2021 | 12 | 0.037279 | 0.500000 | 0.003348 | 0.500000 | 0.266667 |
| 42 | 2021 | 12 | 0.067251 | 0.583333 | 0.017484 | 0.666667 | 0.325000 |
| 10 | 2022 | 12 | 0.026572 | 0.416667 | 0.002535 | 0.416667 | 0.208333 |
| 21 | 2022 | 12 | 0.031906 | 0.500000 | 0.009296 | 0.500000 | 0.216667 |
| 42 | 2022 | 12 | 0.042946 | 0.500000 | 0.011434 | 0.500000 | 0.266667 |
| 10 | 2023 | 12 | -0.006435 | 0.416667 | 0.001637 | 0.416667 | 0.308333 |
| 21 | 2023 | 12 | -0.000480 | 0.416667 | 0.005123 | 0.500000 | 0.250000 |
| 42 | 2023 | 12 | 0.037567 | 0.583333 | 0.020591 | 0.666667 | 0.225000 |
| 10 | 2024 | 12 | 0.095214 | 0.916667 | 0.013508 | 0.750000 | 0.283333 |
| 21 | 2024 | 12 | 0.077999 | 0.750000 | 0.019983 | 0.750000 | 0.275000 |
| 42 | 2024 | 12 | 0.075630 | 0.750000 | 0.032645 | 0.750000 | 0.308333 |
| 10 | 2025 | 12 | 0.065646 | 0.666667 | 0.014363 | 0.666667 | 0.300000 |
| 21 | 2025 | 12 | 0.040684 | 0.500000 | 0.009514 | 0.583333 | 0.300000 |
| 42 | 2025 | 12 | 0.029240 | 0.666667 | 0.022808 | 0.583333 | 0.300000 |
| 10 | 2026 | 4 | 0.066026 | 0.750000 | 0.007725 | 0.750000 | 0.250000 |
| 21 | 2026 | 4 | 0.030348 | 0.500000 | 0.026739 | 0.750000 | 0.175000 |
| 42 | 2026 | 4 | 0.082041 | 0.500000 | 0.050949 | 0.500000 | 0.250000 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_horizons | PASS | 0 | The analysis must contain 10-, 21- and 42-session horizons. |
| minimum_common_dates | PASS | 0 | Horizon comparison must use at least 60 common OOS dates. |
| same_dates_per_horizon | PASS | 0 | Every horizon must be evaluated on identical OOS dates. |
| complete_cross_sections | PASS | 0 | Every horizon-date observation must use the full 50-stock cross-section. |
| finite_summary_metrics | PASS | 0 | Key horizon robustness metrics must remain finite. |
| valid_bootstrap_probabilities | PASS | 0 | Bootstrap probabilities must lie between zero and one. |
| confidence_interval_order | PASS | 0 | Confidence-interval lower bounds must not exceed upper bounds. |
| baseline_horizon_present | PASS | 0 | The frozen 21-session baseline must be present exactly once. |
