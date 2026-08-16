# CVaR Portfolio Comparison

## Construction comparison

```text
                     positions  maximum_weight  maximum_sector_weight  effective_positions  one_way_turnover
method                                                                                                      
alpha_risk_turnover  21.402597        0.049999               0.249999            20.424987          0.227074
cvar                 23.402597        0.049999               0.249999            20.383506          0.184383
score_weighted       25.000000        0.050000               0.250000            24.176025          0.233225
top_n_equal_weight   25.000000        0.040000               0.240000            25.000000          0.204737
```

## Predicted risk

```text
                     predicted_volatility  beta_vs_spy  maximum_sector_weight
method                                                                       
alpha_risk_turnover              0.190236     0.981225               0.249999
cvar                             0.195521     1.011956               0.249999
score_weighted                   0.204894     1.066027               0.250000
top_n_equal_weight               0.199920     1.040742               0.240000
```

## CVaR diagnostics

```text
                       as_of_date  objective_value  scenario_count  confidence_level  horizon_days  predicted_alpha_proxy   var_loss  cvar_loss  mean_scenario_return  worst_scenario_return  one_way_turnover  portfolio_beta_vs_spy  maximum_position_adv_fraction  positions  maximum_weight  maximum_sector_weight
count                          77        77.000000            77.0      7.700000e+01          77.0              77.000000  77.000000  77.000000             77.000000              77.000000         76.000000              77.000000                      77.000000  77.000000    7.700000e+01              77.000000
mean   2023-03-31 06:51:25.714000        -0.004206           252.0      9.500000e-01          21.0               0.002269   0.068556   0.092207              0.021405              -0.125161          0.184383               1.011956                       0.000153  23.402597    4.999900e-02               0.215518
min           2020-01-31 00:00:00        -0.014243           252.0      9.500000e-01          21.0               0.002013   0.009908   0.017834             -0.002343              -0.338420          0.049999               0.867914                       0.000076  21.000000    4.999900e-02               0.125246
25%           2021-08-31 00:00:00        -0.005044           252.0      9.500000e-01          21.0               0.002185   0.036579   0.048839              0.012515              -0.141020          0.139285               0.977227                       0.000125  23.000000    4.999900e-02               0.199996
50%           2023-03-31 00:00:00        -0.003747           252.0      9.500000e-01          21.0               0.002278   0.070471   0.086304              0.021496              -0.115064          0.175165               1.024270                       0.000144  23.000000    4.999900e-02               0.211834
75%           2024-10-31 00:00:00        -0.001754           252.0      9.500000e-01          21.0               0.002343   0.089697   0.104017              0.029756              -0.064599          0.243610               1.049200                       0.000179  24.000000    4.999900e-02               0.249995
max           2026-05-29 00:00:00         0.000668           252.0      9.500000e-01          21.0               0.002508   0.164797   0.253728              0.050829              -0.026536          0.368052               1.124058                       0.000232  25.000000    4.999900e-02               0.249999
std                           NaN         0.003451             0.0      3.352510e-16           0.0               0.000116   0.041277   0.061615              0.012826               0.081606          0.073919               0.058438                       0.000038   1.054561    3.788098e-10               0.035439
```

## Readiness checks

```text
              scope                      check status  violations                                                                  description
        all_methods         unique_weight_keys   PASS           0                     Target weights must have unique method-date-ticker keys.
        all_methods             fully_invested   PASS           0                                             Every portfolio must sum to one.
        all_methods                  long_only   PASS           0                                         Portfolio weights must be long-only.
        all_methods      security_weight_limit   PASS           0                                   No security may exceed its maximum weight.
        all_methods        sector_weight_limit   PASS           0                           No sector may exceed its maximum portfolio weight.
        all_methods          minimum_positions   PASS           0                        Every portfolio must contain enough active positions.
        all_methods             finite_weights   PASS           0                                           All target weights must be finite.
alpha_risk_turnover portfolio_beta_lower_limit   PASS           0                  Optimized portfolio beta must remain above its lower bound.
alpha_risk_turnover portfolio_beta_upper_limit   PASS           0                  Optimized portfolio beta must remain below its upper bound.
alpha_risk_turnover   position_liquidity_limit   PASS           0 No target position may exceed the configured share of Average Dollar Volume.
               cvar                finite_cvar   PASS           0                                               CVaR estimates must be finite.
               cvar         cvar_not_below_var   PASS           0                      CVaR loss must not be below the corresponding VaR loss.
               cvar          minimum_scenarios   PASS           0                     Every optimization must use enough historical scenarios.
               cvar portfolio_beta_lower_limit   PASS           0                       CVaR portfolio beta must remain above its lower bound.
               cvar portfolio_beta_upper_limit   PASS           0                       CVaR portfolio beta must remain below its upper bound.
               cvar   position_liquidity_limit   PASS           0                  CVaR positions must respect the configured liquidity limit.
```
