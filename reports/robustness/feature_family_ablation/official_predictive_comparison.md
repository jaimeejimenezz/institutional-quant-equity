# Official Feature-Family Ablation Predictive Comparison

- Frozen ensemble candidate: `core_percentile_ensemble`
- Evaluation function: `evaluate_model_predictions`
- All scenarios use the same 77 OOS dates and 50 names per date.

## Aggregate comparison

       scenario               model_name  months  valid_ic_months  mean_ic  median_ic   std_ic  annualized_ic_ir  positive_ic_ratio  mean_top_bottom_spread  positive_spread_ratio  mean_top_quintile_precision  mean_top_quintile_turnover
           full core_percentile_ensemble      77               77 0.046380   0.060510 0.169420          0.948316           0.584416                0.013566               0.636364                     0.254545                    0.409211
no_fundamentals core_percentile_ensemble      77               77 0.048489   0.036309 0.154376          1.088059           0.610390                0.017495               0.636364                     0.275325                    0.419737
    no_momentum core_percentile_ensemble      77               77 0.045648   0.057689 0.172864          0.914766           0.597403                0.011684               0.636364                     0.250649                    0.314474

## Deltas versus FULL

       scenario  delta_mean_ic_vs_full  delta_annualized_ic_ir_vs_full  delta_top_bottom_spread_vs_full  delta_top_quintile_precision_vs_full  delta_top_quintile_turnover_vs_full  ic_retention_ratio  spread_retention_ratio
no_fundamentals               0.002109                        0.139743                         0.003929                              0.020779                             0.010526            1.045478                1.289650
    no_momentum              -0.000731                       -0.033550                        -0.001882                             -0.003896                            -0.094737            0.984229                0.861297

## Frozen FULL checks

                           check status  observed  expected  absolute_difference  tolerance
                    full_mean_ic   PASS  0.046380  0.046380         4.777341e-07   0.000005
           full_annualized_ic_ir   PASS  0.948316  0.948316         8.902347e-08   0.000005
     full_mean_top_bottom_spread   PASS  0.013566  0.013566         4.744600e-07   0.000005
full_mean_top_quintile_precision   PASS  0.254545  0.254545         4.545455e-07   0.000005
 full_mean_top_quintile_turnover   PASS  0.409211  0.409211         4.736842e-07   0.000005

## OOS key checks

       scenario status  key_differences_vs_full
           full   PASS                        0
no_fundamentals   PASS                        0
    no_momentum   PASS                        0
