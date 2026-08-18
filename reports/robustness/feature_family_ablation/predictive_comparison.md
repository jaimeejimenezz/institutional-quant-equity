# Feature-Family Ablation Predictive Comparison

## Aggregate comparison

       scenario  oos_dates  mean_ic   ic_std  annualized_ic_ir  positive_ic_frequency  mean_top_bottom_spread  positive_spread_frequency  mean_top_quintile_precision  mean_top_quintile_turnover
           full         77 0.040514 0.208857          0.671967               0.545455                0.012308                   0.597403                     0.259740                    0.369737
no_fundamentals         77 0.049172 0.191155          0.891100               0.597403                0.016053                   0.623377                     0.274026                    0.367105
    no_momentum         77 0.054859 0.203494          0.933875               0.584416                0.017296                   0.649351                     0.280519                    0.313158

## Deltas versus FULL

       scenario  delta_mean_ic_vs_full  delta_annualized_ic_ir_vs_full  delta_top_bottom_spread_vs_full  delta_top_quintile_precision_vs_full  delta_top_quintile_turnover_vs_full  ic_retention_ratio  spread_retention_ratio
no_fundamentals               0.008658                        0.219133                         0.003745                              0.014286                            -0.002632            1.213709                1.304259
    no_momentum               0.014345                        0.261908                         0.004987                              0.020779                            -0.056579            1.354075                1.405200

## Frozen FULL metric cross-check

                           check status  observed  expected  absolute_difference  tolerance
                    full_mean_ic   FAIL  0.040514  0.046380             0.005866   0.000005
           full_annualized_ic_ir   FAIL  0.671967  0.948316             0.276349   0.000005
     full_mean_top_bottom_spread   FAIL  0.012308  0.013566             0.001258   0.000005
full_mean_top_quintile_precision   FAIL  0.259740  0.254545             0.005195   0.000005
 full_mean_top_quintile_turnover   FAIL  0.369737  0.409211             0.039474   0.000005

## OOS key checks

       scenario status  key_differences_vs_full
           full   PASS                        0
no_fundamentals   PASS                        0
    no_momentum   PASS                        0

## Methodology

- Monthly IC is the cross-sectional Spearman correlation between `percentile_score` and `target_21d_excess`.
- Annualized IC IR is mean monthly IC divided by the sample standard deviation of monthly IC, multiplied by sqrt(12).
- Top-bottom spread is mean realized excess return of the top predicted quintile minus the bottom predicted quintile.
- Top-quintile precision is the share of the predicted top 10 names that belong to the realized top quintile.
- Top-quintile turnover is 1 minus the overlap between consecutive top-10 sets divided by 10.
- All three scenarios are evaluated on exactly the same 77 OOS dates and 50 companies per date.
