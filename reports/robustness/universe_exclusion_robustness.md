# Frozen-Universe Exclusion Robustness

## Methodology

- The production final alpha signal remains frozen. No model is retrained or tuned for these exclusions.
- The score-weighted portfolio is reconstructed after removing selected groups from the existing 50-stock universe.
- Tests include full universe, exclusion of Information Technology, leave-one-sector-out scenarios, and exclusion of the five highest-ADV stocks on each date.
- The ADV test is a liquidity-concentration proxy only. It is not presented as a market-cap test.
- A genuine expanded-universe test still requires adding new securities and rebuilding the upstream point-in-time data pipeline.
- Portfolio constraints and advanced transaction-cost assumptions remain unchanged.

## Results

| strategy_name | scenario_type | excluded_group | minimum_eligible_stocks | eligible_sectors | cagr | cagr_difference_vs_full | sharpe_ratio | sharpe_difference_vs_full | maximum_drawdown | drawdown_difference_vs_full | excess_cagr | mean_one_way_turnover | effective_cost_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_universe | baseline |  | 50 | 11 | 0.244270 | 0.000000 | 1.086881 | 0.000000 | -0.343498 | 0.000000 | 0.087729 | 0.259021 | 5.137560 |
| leave_out_sector_communication_services | leave_one_sector_out | Communication Services | 46 | 10 | 0.241950 | -0.002321 | 1.094666 | 0.007785 | -0.354284 | -0.010786 | 0.085408 | 0.242786 | 5.131863 |
| leave_out_sector_consumer_discretionary | leave_one_sector_out | Consumer Discretionary | 44 | 10 | 0.230179 | -0.014092 | 1.076233 | -0.010648 | -0.336968 | 0.006530 | 0.073637 | 0.239125 | 5.127288 |
| leave_out_sector_consumer_staples | leave_one_sector_out | Consumer Staples | 45 | 10 | 0.242247 | -0.002024 | 1.064937 | -0.021944 | -0.342481 | 0.001017 | 0.085705 | 0.240050 | 5.134969 |
| leave_out_sector_energy | leave_one_sector_out | Energy | 47 | 10 | 0.240012 | -0.004258 | 1.071095 | -0.015786 | -0.331882 | 0.011616 | 0.083470 | 0.251776 | 5.128853 |
| leave_out_sector_financials | leave_one_sector_out | Financials | 43 | 10 | 0.239711 | -0.004559 | 1.097813 | 0.010932 | -0.328841 | 0.014657 | 0.083169 | 0.235794 | 5.131935 |
| leave_out_sector_health_care | leave_one_sector_out | Health Care | 43 | 10 | 0.235861 | -0.008410 | 1.027826 | -0.059055 | -0.353733 | -0.010235 | 0.079319 | 0.230317 | 5.125873 |
| leave_out_sector_industrials | leave_one_sector_out | Industrials | 45 | 10 | 0.233901 | -0.010369 | 1.052614 | -0.034267 | -0.350556 | -0.007058 | 0.077360 | 0.238528 | 5.127093 |
| leave_out_sector_materials | leave_one_sector_out | Materials | 48 | 10 | 0.240341 | -0.003929 | 1.070901 | -0.015980 | -0.343650 | -0.000152 | 0.083799 | 0.248792 | 5.129488 |
| leave_out_sector_real_estate | leave_one_sector_out | Real Estate | 49 | 10 | 0.243328 | -0.000943 | 1.087461 | 0.000579 | -0.342686 | 0.000812 | 0.086786 | 0.257810 | 5.133926 |
| leave_out_sector_utilities | leave_one_sector_out | Utilities | 48 | 10 | 0.246443 | 0.002173 | 1.090034 | 0.003153 | -0.341638 | 0.001860 | 0.089902 | 0.252330 | 5.134698 |
| exclude_top5_adv | liquidity_concentration_proxy | Top 5 stocks by point-in-time ADV | 45 | 11 | 0.197976 | -0.046295 | 0.964686 | -0.122195 | -0.344737 | -0.001239 | 0.041434 | 0.250414 | 5.126582 |
| exclude_information_technology | named_sector_exclusion | Information Technology | 42 | 10 | 0.186690 | -0.057580 | 0.936890 | -0.149992 | -0.353455 | -0.009957 | 0.030148 | 0.239002 | 5.119909 |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_scenarios | PASS | 0 | Every defined universe-exclusion scenario must be backtested. |
| expected_oos_dates | PASS | 0 | Every exclusion scenario must contain all 77 frozen OOS dates. |
| fully_invested | PASS | 0 | Every universe-exclusion portfolio must sum to one. |
| long_only | PASS | 0 | Universe-exclusion portfolios must remain long-only. |
| security_cap | PASS | 0 | Universe-exclusion portfolios must respect the 5% security cap. |
| sector_cap | PASS | 0 | Universe-exclusion portfolios must respect the 25% sector cap. |
| minimum_eligible_universe | PASS | 0 | Every scenario must retain at least 25 eligible stocks on every date. |
| finite_performance | PASS | 0 | Key universe-robustness metrics must remain finite. |
| positive_final_values | PASS | 0 | Every universe-exclusion strategy must retain positive final value. |
| one_baseline | PASS | 0 | Exactly one full-universe baseline must be present. |
