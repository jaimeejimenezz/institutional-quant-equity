# Technical Factor Research Report

## Research design

- Research start date: `2014-01-31`
- Research end date: `2019-12-31`
- Prediction target: `target_21d_excess`
- Quantiles: `5`
- Minimum cross-section: `20`
- Technical signals: `19`
- Research rows: `3600`
- Research dates: `72`

## Temporal interpretation

All technical signals use market information available on or before `as_of_date`.

The future return begins strictly after `as_of_date`. Future returns are used only as evaluation targets and never as model inputs.

Initial factor selection is restricted to the configured research period ending in 2019. Observations from 2020 onward are preserved for subsequent out-of-sample evaluation.

## Factor overview

| signal | months_ic | mean_ic | annualized_ic_ir | positive_month_ratio | preferred_direction | mean_top_bottom_spread | positive_spread_ratio | mean_quintile_monotonicity | mean_turnover | selected_quantile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amihud_illiquidity_20d_sector_neutral | 72 | 0.031462 | 0.654009 | 0.569444 | higher_is_better | 0.004125 | 0.625000 | 0.136111 | 0.270423 | 5.000000 |
| reversal_1m_sector_neutral | 71 | 0.030308 | 0.672029 | 0.563380 | higher_is_better | 0.006441 | 0.563380 | 0.132394 | 0.784286 | 5.000000 |
| return_1m_sector_neutral | 71 | -0.030308 | -0.672029 | 0.436620 | lower_is_better | -0.006441 | 0.436620 | -0.132394 | 0.784286 | 1.000000 |
| volatility_60d_sector_neutral | 70 | 0.022079 | 0.427523 | 0.528571 | higher_is_better | 0.008044 | 0.557143 | 0.055714 | 0.194203 | 5.000000 |
| distance_sma_50d_sector_neutral | 70 | -0.021385 | -0.482519 | 0.457143 | lower_is_better | -0.005682 | 0.471429 | -0.041429 | 0.615942 | 1.000000 |
| beta_60d_market_sector_neutral | 70 | 0.021056 | 0.427605 | 0.542857 | higher_is_better | 0.009421 | 0.614286 | 0.130000 | 0.269565 | 5.000000 |
| return_3m_sector_neutral | 69 | 0.018595 | 0.403866 | 0.608696 | higher_is_better | 0.001764 | 0.536232 | 0.037681 | 0.436765 | 5.000000 |
| distance_sma_200d_sector_neutral | 63 | 0.016829 | 0.292954 | 0.571429 | higher_is_better | -0.002991 | 0.476190 | 0.031746 | 0.291935 | 5.000000 |
| max_drawdown_126d_sector_neutral | 66 | -0.016724 | -0.299758 | 0.454545 | lower_is_better | -0.009363 | 0.469697 | -0.092424 | 0.210769 | 1.000000 |
| sma_50_200_spread_sector_neutral | 63 | 0.014086 | 0.228524 | 0.476190 | higher_is_better | 0.000360 | 0.507937 | 0.012698 | 0.209677 | 5.000000 |
| volatility_20d_sector_neutral | 72 | 0.012973 | 0.276813 | 0.513889 | higher_is_better | 0.007206 | 0.611111 | 0.100000 | 0.522535 | 5.000000 |
| momentum_6_1_sector_neutral | 66 | 0.011356 | 0.189963 | 0.454545 | higher_is_better | 0.000849 | 0.530303 | 0.021212 | 0.350769 | 5.000000 |
| average_dollar_volume_20d_sector_neutral | 72 | -0.010516 | -0.227007 | 0.472222 | lower_is_better | -0.001297 | 0.444444 | -0.025000 | 0.163380 | 1.000000 |
| return_1w_sector_neutral | 72 | -0.010237 | -0.253190 | 0.444444 | lower_is_better | -0.003300 | 0.430556 | -0.081944 | 0.770423 | 1.000000 |
| downside_volatility_60d_sector_neutral | 70 | 0.009442 | 0.188155 | 0.514286 | higher_is_better | 0.007856 | 0.585714 | 0.038571 | 0.272464 | 5.000000 |
| dollar_volume_change_20d_60d_sector_neutral | 70 | -0.001908 | -0.049466 | 0.471429 | lower_is_better | -0.004371 | 0.371429 | -0.058571 | 0.785507 | 1.000000 |
| momentum_12_1_sector_neutral | 60 | -0.001343 | -0.023124 | 0.450000 | lower_is_better | -0.000959 | 0.516667 | -0.025000 | 0.227119 | 1.000000 |
| positive_day_ratio_60d_sector_neutral | 70 | -0.000059 | -0.001173 | 0.528571 | lower_is_better | -0.003621 | 0.471429 | -0.028571 | 0.457971 | 1.000000 |
| zero_volume_ratio_60d_sector_neutral | 0 |  |  |  | lower_is_better |  |  |  |  |  |

## Average excess return by quintile

| signal | Q1 | Q2 | Q3 | Q4 | Q5 |
| --- | --- | --- | --- | --- | --- |
| amihud_illiquidity_20d_sector_neutral | 0.003472 | -0.004832 | 0.000285 | -0.000108 | 0.007597 |
| average_dollar_volume_20d_sector_neutral | 0.005939 | -0.000655 | -0.000116 | -0.003394 | 0.004641 |
| beta_60d_market_sector_neutral | -0.001311 | 0.002296 | -0.001460 | -0.001280 | 0.008110 |
| distance_sma_200d_sector_neutral | 0.006921 | -0.004291 | -0.001454 | 0.002387 | 0.003930 |
| distance_sma_50d_sector_neutral | 0.005492 | -0.000087 | 0.002113 | -0.000974 | -0.000190 |
| dollar_volume_change_20d_60d_sector_neutral | 0.006295 | -0.002844 | 0.000822 | 0.000157 | 0.001924 |
| downside_volatility_60d_sector_neutral | -0.000101 | -0.000330 | 0.000276 | -0.001247 | 0.007755 |
| max_drawdown_126d_sector_neutral | 0.008285 | -0.001893 | -0.001498 | 0.002795 | -0.001078 |
| momentum_12_1_sector_neutral | 0.005937 | 0.002563 | -0.003449 | -0.000722 | 0.004978 |
| momentum_6_1_sector_neutral | 0.003584 | -0.000793 | 0.000039 | -0.000650 | 0.004432 |
| positive_day_ratio_60d_sector_neutral | 0.007433 | -0.002208 | -0.001992 | -0.000691 | 0.003812 |
| return_1m_sector_neutral | 0.006267 | 0.001282 | -0.000645 | -0.001594 | -0.000174 |
| return_1w_sector_neutral | 0.004424 | 0.003034 | -0.002048 | -0.000119 | 0.001124 |
| return_3m_sector_neutral | 0.004921 | -0.001460 | -0.002246 | -0.001281 | 0.006685 |
| reversal_1m_sector_neutral | -0.000174 | -0.001594 | -0.000645 | 0.001282 | 0.006267 |
| sma_50_200_spread_sector_neutral | 0.005976 | -0.000589 | -0.005105 | 0.000876 | 0.006336 |
| volatility_20d_sector_neutral | -0.000372 | 0.001314 | 0.000186 | -0.001547 | 0.006834 |
| volatility_60d_sector_neutral | 0.001486 | -0.001108 | -0.001962 | -0.001593 | 0.009530 |

## Metric interpretation

- `mean_ic`: average monthly Spearman correlation between the signal ranking and future relative returns.
- `annualized_ic_ir`: average IC divided by its volatility and annualized with square root of 12.
- `preferred_direction`: whether high or low values of the signal were associated with better subsequent returns.
- `mean_top_bottom_spread`: average return difference between Q5 and Q1 before applying any direction reversal.
- `mean_turnover`: proportion of members replaced in the economically preferred quintile between consecutive months.

## Current status

This report is descriptive. No final feature selection is made in Step 6A.

Step 6B will add correlation and redundancy analysis, yearly and sector results, figures and the final selection of technical signals.

## Step 6B — Stability, redundancy and sector diagnostics

### Objective

This section evaluates temporal stability, sector breadth and redundancy before the final technical-feature selection.

All diagnostics remain restricted to the 2014-2019 research window. Data from 2020 onward are not used for feature selection.

### Preliminary selection diagnostics

| signal | preliminary_status | abs_mean_ic | directional_month_ratio | directional_spread | directional_positive_spread_ratio | mean_turnover | positive_year_ratio | positive_subperiod_ratio | positive_sector_ratio | strongest_correlated_signal | strongest_mean_absolute_correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amihud_illiquidity_20d_sector_neutral | candidate | 0.031462 | 0.569444 | 0.004125 | 0.625000 | 0.270423 | 0.666667 | 0.500000 | 0.875000 | average_dollar_volume_20d_sector_neutral | 0.653910 |
| reversal_1m_sector_neutral | review_redundancy | 0.030308 | 0.563380 | 0.006441 | 0.563380 | 0.784286 | 0.833333 | 1.000000 | 0.750000 | return_1m_sector_neutral | 1.000000 |
| return_1m_sector_neutral | review_redundancy | 0.030308 | 0.563380 | 0.006441 | 0.563380 | 0.784286 | 0.833333 | 1.000000 | 0.750000 | reversal_1m_sector_neutral | 1.000000 |
| volatility_60d_sector_neutral | candidate | 0.022079 | 0.528571 | 0.008044 | 0.557143 | 0.194203 | 0.500000 | 1.000000 | 0.500000 | downside_volatility_60d_sector_neutral | 0.826671 |
| distance_sma_50d_sector_neutral | candidate | 0.021385 | 0.542857 | 0.005682 | 0.528571 | 0.615942 | 0.666667 | 1.000000 | 0.750000 | reversal_1m_sector_neutral | 0.846768 |
| beta_60d_market_sector_neutral | candidate | 0.021056 | 0.542857 | 0.009421 | 0.614286 | 0.269565 | 0.666667 | 1.000000 | 0.750000 | volatility_60d_sector_neutral | 0.662074 |
| return_3m_sector_neutral | candidate | 0.018595 | 0.608696 | 0.001764 | 0.536232 | 0.436765 | 0.666667 | 1.000000 | 0.375000 | distance_sma_200d_sector_neutral | 0.780331 |
| distance_sma_200d_sector_neutral | candidate_unstable | 0.016829 | 0.571429 | -0.002991 | 0.476190 | 0.291935 | 0.500000 | 1.000000 | 0.500000 | sma_50_200_spread_sector_neutral | 0.852703 |
| max_drawdown_126d_sector_neutral | candidate | 0.016724 | 0.545455 | 0.009363 | 0.530303 | 0.210769 | 0.666667 | 0.500000 | 0.625000 | downside_volatility_60d_sector_neutral | 0.643156 |
| sma_50_200_spread_sector_neutral | review_redundancy | 0.014086 | 0.476190 | 0.000360 | 0.507937 | 0.209677 | 0.500000 | 0.500000 | 0.500000 | momentum_6_1_sector_neutral | 0.903337 |
| volatility_20d_sector_neutral | candidate_unstable | 0.012973 | 0.513889 | 0.007206 | 0.611111 | 0.522535 | 0.666667 | 1.000000 | 0.375000 | volatility_60d_sector_neutral | 0.751012 |
| momentum_6_1_sector_neutral | review_redundancy | 0.011356 | 0.454545 | 0.000849 | 0.530303 | 0.350769 | 0.500000 | 0.500000 | 0.625000 | sma_50_200_spread_sector_neutral | 0.903337 |
| average_dollar_volume_20d_sector_neutral | candidate | 0.010516 | 0.527778 | 0.001297 | 0.555556 | 0.163380 | 0.500000 | 0.500000 | 0.625000 | amihud_illiquidity_20d_sector_neutral | 0.653910 |
| return_1w_sector_neutral | candidate_high_turnover | 0.010237 | 0.555556 | 0.003300 | 0.569444 | 0.770423 | 0.500000 | 0.500000 | 0.625000 | distance_sma_50d_sector_neutral | 0.434026 |
| downside_volatility_60d_sector_neutral | weak_candidate | 0.009442 | 0.514286 | 0.007856 | 0.585714 | 0.272464 | 0.333333 | 1.000000 | 0.625000 | volatility_60d_sector_neutral | 0.826671 |
| dollar_volume_change_20d_60d_sector_neutral | drop_very_weak | 0.001908 | 0.528571 | 0.004371 | 0.628571 | 0.785507 | 0.500000 | 0.500000 | 0.750000 | volatility_20d_sector_neutral | 0.311719 |
| momentum_12_1_sector_neutral | drop_very_weak | 0.001343 | 0.550000 | 0.000959 | 0.483333 | 0.227119 | 0.600000 | 0.500000 | 0.375000 | sma_50_200_spread_sector_neutral | 0.731321 |
| positive_day_ratio_60d_sector_neutral | drop_very_weak | 0.000059 | 0.471429 | 0.003621 | 0.528571 | 0.457971 | 0.666667 | 0.500000 | 0.750000 | return_3m_sector_neutral | 0.491450 |
| zero_volume_ratio_60d_sector_neutral | drop_no_variation |  |  |  |  |  |  |  |  |  |  |

### Highly correlated signal pairs

Pairs shown below have an average absolute monthly correlation of at least `0.90`.

| first_signal | second_signal | months | mean_correlation | median_correlation | mean_absolute_correlation | maximum_absolute_correlation |
| --- | --- | --- | --- | --- | --- | --- |
| return_1m_sector_neutral | reversal_1m_sector_neutral | 71 | -1.000000 | -1.000000 | 1.000000 | 1.000000 |
| momentum_6_1_sector_neutral | sma_50_200_spread_sector_neutral | 63 | 0.903337 | 0.912797 | 0.903337 | 0.969268 |

### Direction-adjusted IC by year

Positive values mean that the signal worked in its preferred economic direction.

| signal | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 |
| --- | --- | --- | --- | --- | --- | --- |
| amihud_illiquidity_20d_sector_neutral | 0.080696 | 0.071244 | 0.039080 | -0.008651 | 0.049812 | -0.043409 |
| beta_60d_market_sector_neutral | -0.011909 | 0.020904 | 0.070756 | 0.029572 | -0.056214 | 0.067731 |
| distance_sma_200d_sector_neutral | 0.111517 | 0.069268 | -0.019552 | 0.032765 | -0.015974 | -0.006034 |
| distance_sma_50d_sector_neutral | 0.006732 | 0.009692 | 0.066234 | -0.004442 | 0.078719 | -0.031068 |
| max_drawdown_126d_sector_neutral | -0.069308 | -0.094398 | 0.117511 | 0.031244 | 0.019704 | 0.052573 |
| return_1m_sector_neutral | 0.059810 | 0.004898 | 0.063858 | 0.011108 | 0.055686 | -0.011052 |
| return_3m_sector_neutral | 0.043340 | 0.095086 | -0.040232 | 0.045530 | -0.029564 | 0.003593 |
| reversal_1m_sector_neutral | 0.059810 | 0.004898 | 0.063858 | 0.011108 | 0.055686 | -0.011052 |
| sma_50_200_spread_sector_neutral | 0.057831 | 0.086843 | -0.004114 | 0.018279 | -0.002497 | -0.039016 |
| volatility_60d_sector_neutral | 0.001930 | -0.004994 | 0.102305 | -0.012717 | -0.010532 | 0.053125 |

### Direction-adjusted IC by subperiod

| period | signal | months | directional_mean_ic | directional_month_ratio |
| --- | --- | --- | --- | --- |
| 2014-2016 | amihud_illiquidity_20d_sector_neutral | 36 | 0.063673 | 0.694444 |
| 2014-2016 | beta_60d_market_sector_neutral | 34 | 0.028848 | 0.617647 |
| 2014-2016 | distance_sma_200d_sector_neutral | 27 | 0.034487 | 0.629630 |
| 2014-2016 | distance_sma_50d_sector_neutral | 34 | 0.028778 | 0.529412 |
| 2014-2016 | max_drawdown_126d_sector_neutral | 30 | -0.004616 | 0.500000 |
| 2014-2016 | return_1m_sector_neutral | 35 | 0.042371 | 0.657143 |
| 2014-2016 | return_3m_sector_neutral | 33 | 0.031767 | 0.606061 |
| 2014-2016 | reversal_1m_sector_neutral | 35 | 0.042371 | 0.657143 |
| 2014-2016 | sma_50_200_spread_sector_neutral | 27 | 0.043194 | 0.518519 |
| 2014-2016 | volatility_60d_sector_neutral | 34 | 0.034913 | 0.617647 |
| 2017-2019 | amihud_illiquidity_20d_sector_neutral | 36 | -0.000750 | 0.444444 |
| 2017-2019 | beta_60d_market_sector_neutral | 36 | 0.013696 | 0.472222 |
| 2017-2019 | distance_sma_200d_sector_neutral | 36 | 0.003585 | 0.527778 |
| 2017-2019 | distance_sma_50d_sector_neutral | 36 | 0.014403 | 0.555556 |
| 2017-2019 | max_drawdown_126d_sector_neutral | 36 | 0.034507 | 0.583333 |
| 2017-2019 | return_1m_sector_neutral | 36 | 0.018581 | 0.472222 |
| 2017-2019 | return_3m_sector_neutral | 36 | 0.006520 | 0.611111 |
| 2017-2019 | reversal_1m_sector_neutral | 36 | 0.018581 | 0.472222 |
| 2017-2019 | sma_50_200_spread_sector_neutral | 36 | -0.007744 | 0.444444 |
| 2017-2019 | volatility_60d_sector_neutral | 36 | 0.009959 | 0.444444 |

### Sector results for the strongest signals

| signal | sector | months | mean_ic | median_ic | preferred_direction | directional_mean_ic | directional_month_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amihud_illiquidity_20d_sector_neutral | Health Care | 72 | 0.090774 | 0.178571 | higher_is_better | 0.090774 | 0.583333 |
| amihud_illiquidity_20d_sector_neutral | Consumer Staples | 72 | 0.068056 | 0.050000 | higher_is_better | 0.068056 | 0.500000 |
| amihud_illiquidity_20d_sector_neutral | Financials | 72 | 0.031746 | 0.142857 | higher_is_better | 0.031746 | 0.555556 |
| amihud_illiquidity_20d_sector_neutral | Information Technology | 72 | 0.008598 | -0.011905 | higher_is_better | 0.008598 | 0.486111 |
| amihud_illiquidity_20d_sector_neutral | Communication Services | 72 | 0.008333 | 0.100000 | higher_is_better | 0.008333 | 0.500000 |
| amihud_illiquidity_20d_sector_neutral | Consumer Discretionary | 72 | 0.006349 | -0.028571 | higher_is_better | 0.006349 | 0.458333 |
| amihud_illiquidity_20d_sector_neutral | Industrials | 72 | 0.005556 | -0.050000 | higher_is_better | 0.005556 | 0.472222 |
| amihud_illiquidity_20d_sector_neutral | Energy | 72 | -0.013889 | -0.500000 | higher_is_better | -0.013889 | 0.486111 |
| beta_60d_market_sector_neutral | Communication Services | 70 | 0.154286 | 0.200000 | higher_is_better | 0.154286 | 0.542857 |
| beta_60d_market_sector_neutral | Information Technology | 70 | 0.099320 | 0.142857 | higher_is_better | 0.099320 | 0.628571 |
| beta_60d_market_sector_neutral | Consumer Staples | 70 | 0.045714 | 0.100000 | higher_is_better | 0.045714 | 0.528571 |
| beta_60d_market_sector_neutral | Financials | 70 | 0.043367 | 0.035714 | higher_is_better | 0.043367 | 0.500000 |
| beta_60d_market_sector_neutral | Health Care | 70 | 0.022959 | 0.053571 | higher_is_better | 0.022959 | 0.528571 |
| beta_60d_market_sector_neutral | Consumer Discretionary | 70 | 0.017143 | -0.028571 | higher_is_better | 0.017143 | 0.471429 |
| beta_60d_market_sector_neutral | Industrials | 70 | -0.042857 | -0.050000 | higher_is_better | -0.042857 | 0.428571 |
| beta_60d_market_sector_neutral | Energy | 70 | -0.157143 | -0.500000 | higher_is_better | -0.157143 | 0.385714 |
| distance_sma_200d_sector_neutral | Information Technology | 63 | 0.119803 | 0.166667 | higher_is_better | 0.119803 | 0.666667 |
| distance_sma_200d_sector_neutral | Financials | 63 | 0.069728 | 0.142857 | higher_is_better | 0.069728 | 0.587302 |
| distance_sma_200d_sector_neutral | Energy | 63 | 0.063492 | 0.500000 | higher_is_better | 0.063492 | 0.555556 |
| distance_sma_200d_sector_neutral | Health Care | 63 | 0.035714 | 0.000000 | higher_is_better | 0.035714 | 0.492063 |
| distance_sma_200d_sector_neutral | Communication Services | 63 | -0.025397 | -0.200000 | higher_is_better | -0.025397 | 0.444444 |
| distance_sma_200d_sector_neutral | Consumer Discretionary | 63 | -0.030385 | -0.028571 | higher_is_better | -0.030385 | 0.476190 |
| distance_sma_200d_sector_neutral | Consumer Staples | 63 | -0.031746 | -0.100000 | higher_is_better | -0.031746 | 0.396825 |
| distance_sma_200d_sector_neutral | Industrials | 63 | -0.058730 | -0.100000 | higher_is_better | -0.058730 | 0.460317 |
| distance_sma_50d_sector_neutral | Energy | 70 | -0.100000 | -0.500000 | lower_is_better | 0.100000 | 0.600000 |
| distance_sma_50d_sector_neutral | Communication Services | 70 | -0.065714 | -0.200000 | lower_is_better | 0.065714 | 0.600000 |
| distance_sma_50d_sector_neutral | Health Care | 70 | -0.027041 | 0.000000 | lower_is_better | 0.027041 | 0.485714 |
| distance_sma_50d_sector_neutral | Industrials | 70 | -0.025714 | -0.100000 | lower_is_better | 0.025714 | 0.528571 |
| distance_sma_50d_sector_neutral | Information Technology | 70 | -0.023129 | 0.011905 | lower_is_better | 0.023129 | 0.485714 |
| distance_sma_50d_sector_neutral | Financials | 70 | -0.001531 | 0.035714 | lower_is_better | 0.001531 | 0.485714 |
| distance_sma_50d_sector_neutral | Consumer Staples | 70 | 0.008571 | 0.000000 | lower_is_better | -0.008571 | 0.485714 |
| distance_sma_50d_sector_neutral | Consumer Discretionary | 70 | 0.012245 | -0.028571 | lower_is_better | -0.012245 | 0.528571 |
| max_drawdown_126d_sector_neutral | Industrials | 66 | -0.077273 | -0.200000 | lower_is_better | 0.077273 | 0.590909 |
| max_drawdown_126d_sector_neutral | Information Technology | 66 | -0.076479 | -0.083333 | lower_is_better | 0.076479 | 0.575758 |
| max_drawdown_126d_sector_neutral | Communication Services | 66 | -0.054545 | -0.200000 | lower_is_better | 0.054545 | 0.515152 |
| max_drawdown_126d_sector_neutral | Financials | 66 | -0.033009 | -0.035714 | lower_is_better | 0.033009 | 0.500000 |
| max_drawdown_126d_sector_neutral | Consumer Discretionary | 66 | -0.032900 | -0.028571 | lower_is_better | 0.032900 | 0.500000 |
| max_drawdown_126d_sector_neutral | Consumer Staples | 66 | 0.031818 | 0.100000 | lower_is_better | -0.031818 | 0.454545 |
| max_drawdown_126d_sector_neutral | Health Care | 66 | 0.093074 | 0.107143 | lower_is_better | -0.093074 | 0.424242 |
| max_drawdown_126d_sector_neutral | Energy | 66 | 0.174242 | 0.500000 | lower_is_better | -0.174242 | 0.378788 |
| return_1m_sector_neutral | Energy | 71 | -0.119718 | -0.500000 | lower_is_better | 0.119718 | 0.577465 |
| return_1m_sector_neutral | Consumer Staples | 71 | -0.069014 | -0.100000 | lower_is_better | 0.069014 | 0.549296 |
| return_1m_sector_neutral | Health Care | 71 | -0.042254 | 0.000000 | lower_is_better | 0.042254 | 0.478873 |
| return_1m_sector_neutral | Communication Services | 71 | -0.039437 | -0.200000 | lower_is_better | 0.039437 | 0.507042 |
| return_1m_sector_neutral | Information Technology | 71 | -0.022133 | 0.047619 | lower_is_better | 0.022133 | 0.478873 |
| return_1m_sector_neutral | Financials | 71 | -0.006036 | -0.071429 | lower_is_better | 0.006036 | 0.507042 |
| return_1m_sector_neutral | Consumer Discretionary | 71 | 0.001207 | 0.085714 | lower_is_better | -0.001207 | 0.450704 |
| return_1m_sector_neutral | Industrials | 71 | 0.014085 | 0.100000 | lower_is_better | -0.014085 | 0.436620 |
| return_3m_sector_neutral | Information Technology | 69 | 0.110421 | 0.119048 | higher_is_better | 0.110421 | 0.623188 |
| return_3m_sector_neutral | Financials | 69 | 0.069358 | 0.035714 | higher_is_better | 0.069358 | 0.521739 |
| return_3m_sector_neutral | Communication Services | 69 | 0.043478 | 0.000000 | higher_is_better | 0.043478 | 0.492754 |
| return_3m_sector_neutral | Industrials | 69 | -0.002899 | 0.100000 | higher_is_better | -0.002899 | 0.521739 |
| return_3m_sector_neutral | Health Care | 69 | -0.016563 | 0.000000 | higher_is_better | -0.016563 | 0.492754 |
| return_3m_sector_neutral | Energy | 69 | -0.021739 | -0.500000 | higher_is_better | -0.021739 | 0.463768 |
| return_3m_sector_neutral | Consumer Staples | 69 | -0.030435 | 0.000000 | higher_is_better | -0.030435 | 0.434783 |
| return_3m_sector_neutral | Consumer Discretionary | 69 | -0.035197 | -0.085714 | higher_is_better | -0.035197 | 0.478261 |
| reversal_1m_sector_neutral | Energy | 71 | 0.119718 | 0.500000 | higher_is_better | 0.119718 | 0.577465 |
| reversal_1m_sector_neutral | Consumer Staples | 71 | 0.069014 | 0.100000 | higher_is_better | 0.069014 | 0.549296 |
| reversal_1m_sector_neutral | Health Care | 71 | 0.042254 | 0.000000 | higher_is_better | 0.042254 | 0.478873 |
| reversal_1m_sector_neutral | Communication Services | 71 | 0.039437 | 0.200000 | higher_is_better | 0.039437 | 0.507042 |
| reversal_1m_sector_neutral | Information Technology | 71 | 0.022133 | -0.047619 | higher_is_better | 0.022133 | 0.478873 |
| reversal_1m_sector_neutral | Financials | 71 | 0.006036 | 0.071429 | higher_is_better | 0.006036 | 0.507042 |
| reversal_1m_sector_neutral | Consumer Discretionary | 71 | -0.001207 | -0.085714 | higher_is_better | -0.001207 | 0.450704 |
| reversal_1m_sector_neutral | Industrials | 71 | -0.014085 | -0.100000 | higher_is_better | -0.014085 | 0.436620 |
| sma_50_200_spread_sector_neutral | Information Technology | 63 | 0.148148 | 0.238095 | higher_is_better | 0.148148 | 0.634921 |
| sma_50_200_spread_sector_neutral | Energy | 63 | 0.111111 | 0.500000 | higher_is_better | 0.111111 | 0.555556 |
| sma_50_200_spread_sector_neutral | Health Care | 63 | 0.060091 | 0.071429 | higher_is_better | 0.060091 | 0.555556 |
| sma_50_200_spread_sector_neutral | Financials | 63 | 0.019274 | 0.071429 | higher_is_better | 0.019274 | 0.539683 |
| sma_50_200_spread_sector_neutral | Communication Services | 63 | -0.015873 | 0.000000 | higher_is_better | -0.015873 | 0.460317 |
| sma_50_200_spread_sector_neutral | Consumer Staples | 63 | -0.055556 | -0.100000 | higher_is_better | -0.055556 | 0.460317 |
| sma_50_200_spread_sector_neutral | Industrials | 63 | -0.100000 | -0.100000 | higher_is_better | -0.100000 | 0.380952 |
| sma_50_200_spread_sector_neutral | Consumer Discretionary | 63 | -0.104762 | -0.085714 | higher_is_better | -0.104762 | 0.396825 |
| volatility_60d_sector_neutral | Communication Services | 70 | 0.120000 | 0.200000 | higher_is_better | 0.120000 | 0.585714 |
| volatility_60d_sector_neutral | Information Technology | 70 | 0.063605 | 0.083333 | higher_is_better | 0.063605 | 0.571429 |
| volatility_60d_sector_neutral | Health Care | 70 | 0.060714 | 0.071429 | higher_is_better | 0.060714 | 0.571429 |
| volatility_60d_sector_neutral | Financials | 70 | 0.056122 | 0.142857 | higher_is_better | 0.056122 | 0.557143 |
| volatility_60d_sector_neutral | Consumer Discretionary | 70 | -0.011429 | -0.028571 | higher_is_better | -0.011429 | 0.471429 |
| volatility_60d_sector_neutral | Consumer Staples | 70 | -0.020000 | -0.100000 | higher_is_better | -0.020000 | 0.442857 |
| volatility_60d_sector_neutral | Industrials | 70 | -0.031429 | -0.100000 | higher_is_better | -0.031429 | 0.428571 |
| volatility_60d_sector_neutral | Energy | 70 | -0.071429 | -0.500000 | higher_is_better | -0.071429 | 0.471429 |

### Interpretation of preliminary statuses

- `candidate`: useful preliminary evidence without an immediate major warning.
- `candidate_high_turnover`: predictive evidence exists, but trading intensity is high.
- `candidate_unstable`: signal strength or spread direction is not sufficiently consistent.
- `review_redundancy`: the signal overlaps strongly with another variable.
- `weak_candidate`: limited individual predictive evidence.
- `drop_very_weak`: near-zero mean IC.
- `drop_no_variation`: the signal cannot meaningfully rank the current universe.

### Current status

The statuses are diagnostic labels, not an automatic final decision. The final list must also consider economic interpretation and which member of each redundant family should be retained.

## Final technical feature selection

### Frozen decision

The technical feature set was selected exclusively using the 2014-2019 research sample.

No observations from 2020 onward were used to choose, remove or replace these variables.

The selection is frozen before any out-of-sample model evaluation. Future results may be used to evaluate the selection, but not to rewrite this historical decision.

### Selected features

| selected_order | signal | family | selection_reason | main_risk |
| --- | --- | --- | --- | --- |
| 1 | amihud_illiquidity_20d_sector_neutral | liquidity | Highest absolute mean IC, broad sector coverage and reasonable turnover. | Performance differs between temporal subperiods. |
| 2 | reversal_1m_sector_neutral | short_term_reversal | Strong IC, positive results in both subperiods and broad sector evidence. | Very high ranking turnover and potential trading costs. |
| 3 | volatility_60d_sector_neutral | risk | High directional spread combined with low turnover. | Limited consistency across individual years and sectors. |
| 4 | distance_sma_50d_sector_neutral | short_term_trend | Useful IC and reasonable temporal and sector stability. | High turnover and partial overlap with one-month reversal. |
| 5 | beta_60d_market_sector_neutral | market_risk | Largest directional spread, moderate turnover and positive results in both subperiods. | Partial correlation with total volatility. |
| 6 | return_3m_sector_neutral | momentum | Positive results in both subperiods and useful exposure to the momentum family. | Predictive evidence is concentrated in relatively few sectors. |
| 7 | max_drawdown_126d_sector_neutral | deep_reversal | Large directional spread and relatively low turnover. | Evidence is dependent on the market regime. |
| 8 | average_dollar_volume_20d_sector_neutral | liquidity_capacity | Low turnover and complementary liquidity information. | Weak individual predictive strength. |

### Excluded features

| signal | family | selection_reason |
| --- | --- | --- |
| distance_sma_200d_sector_neutral | long_term_trend | Excluded because IC and quintile-spread evidence point in conflicting directions. |
| dollar_volume_change_20d_60d_sector_neutral | liquidity_change | Excluded because its mean IC is close to zero. |
| downside_volatility_60d_sector_neutral | downside_risk | Excluded because it is weak and largely overlaps with volatility_60d. |
| momentum_12_1_sector_neutral | long_term_momentum | Excluded because its mean IC is close to zero in the research sample. |
| momentum_6_1_sector_neutral | medium_term_momentum | Excluded because the evidence is weak and redundant with other trend variables. |
| positive_day_ratio_60d_sector_neutral | trend_breadth | Excluded because its mean IC is effectively zero. |
| return_1m_sector_neutral | short_term_return | Excluded because it is the exact inverse representation of reversal_1m. |
| return_1w_sector_neutral | very_short_term_return | Excluded because predictive strength is small relative to its turnover. |
| sma_50_200_spread_sector_neutral | moving_average_trend | Excluded because it is weak and highly redundant with other trend variables. |
| volatility_20d_sector_neutral | short_term_risk | Excluded in favour of volatility_60d, which provides better IC and lower turnover. |
| zero_volume_ratio_60d_sector_neutral | trading_activity | Excluded because it has no variation in the current universe. |

### Final counts

- Signals evaluated: `19`
- Signals selected: `8`
- Signals excluded: `11`
- Research period: `2014-2019`
- Selection status: `FROZEN`
