# Baseline Portfolio Construction

## Methods

- Sector-controlled top-N equal weight
- Constrained alpha score-weighted portfolio

## Construction diagnostics

```text
            method  mean_positions  mean_maximum_weight  maximum_sector_weight  mean_effective_positions  mean_one_way_turnover
    score_weighted            25.0             0.049999                   0.25                 24.176025               0.233225
top_n_equal_weight            25.0             0.040000                   0.24                 25.000000               0.204737
```

## Predicted risk comparison

```text
            method  mean_predicted_volatility  mean_beta_vs_spy  maximum_predicted_volatility  mean_maximum_sector_weight
    score_weighted                   0.204894          1.066027                      0.371895                    0.212648
top_n_equal_weight                   0.199920          1.040742                      0.366926                    0.204156
```

## Readiness checks

```text
                check status  violations                                              description
   unique_weight_keys   PASS           0 Target weights must have unique method-date-ticker keys.
       fully_invested   PASS           0                         Every portfolio must sum to one.
            long_only   PASS           0                     Portfolio weights must be long-only.
security_weight_limit   PASS           0               No security may exceed its maximum weight.
  sector_weight_limit   PASS           0       No sector may exceed its maximum portfolio weight.
    minimum_positions   PASS           0    Every portfolio must contain enough active positions.
       finite_weights   PASS           0                       All target weights must be finite.
```
