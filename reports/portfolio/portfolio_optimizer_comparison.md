# Portfolio Optimizer Comparison

## Construction methods

- Sector-controlled top-N equal weight
- Constrained score-weighted
- Alpha-risk-turnover optimizer

## Construction diagnostics

```text
             method  mean_positions  maximum_weight  maximum_sector_weight  mean_effective_positions  mean_one_way_turnover
alpha_risk_turnover       21.402597        0.049999               0.249999                 20.424987               0.227074
     score_weighted       25.000000        0.050000               0.250000                 24.176025               0.233225
 top_n_equal_weight       25.000000        0.040000               0.240000                 25.000000               0.204737
```

## Predicted risk

```text
             method  mean_predicted_volatility  mean_beta_vs_spy  maximum_sector_weight
alpha_risk_turnover                   0.190236          0.981225               0.249999
     score_weighted                   0.204894          1.066027               0.250000
 top_n_equal_weight                   0.199920          1.040742               0.240000
```

## Optimizer diagnostics

```text
       predicted_alpha_proxy  predicted_volatility  one_way_turnover  positions  maximum_weight  maximum_sector_weight
count              77.000000             77.000000         76.000000  77.000000    7.700000e+01              77.000000
mean                0.028527              0.190236          0.227074  21.402597    4.999900e-02               0.202579
std                 0.001343              0.065169          0.068195   0.590712    5.859721e-10               0.029586
min                 0.025940              0.114261          0.099998  21.000000    4.999900e-02               0.149997
25%                 0.027503              0.137303          0.178686  21.000000    4.999900e-02               0.194335
50%                 0.028673              0.180834          0.216764  21.000000    4.999900e-02               0.199996
75%                 0.029795              0.212570          0.272595  22.000000    4.999900e-02               0.215780
max                 0.030612              0.349315          0.439305  23.000000    4.999900e-02               0.249999
```

## Constraint checks

```text
                     check status  violations                                                                  description
        unique_weight_keys   PASS           0                     Target weights must have unique method-date-ticker keys.
            fully_invested   PASS           0                                             Every portfolio must sum to one.
                 long_only   PASS           0                                         Portfolio weights must be long-only.
     security_weight_limit   PASS           0                                   No security may exceed its maximum weight.
       sector_weight_limit   PASS           0                           No sector may exceed its maximum portfolio weight.
         minimum_positions   PASS           0                        Every portfolio must contain enough active positions.
            finite_weights   PASS           0                                           All target weights must be finite.
portfolio_beta_lower_limit   PASS           0                  Optimized portfolio beta must remain above its lower bound.
portfolio_beta_upper_limit   PASS           0                  Optimized portfolio beta must remain below its upper bound.
  position_liquidity_limit   PASS           0 No target position may exceed the configured share of Average Dollar Volume.
```
