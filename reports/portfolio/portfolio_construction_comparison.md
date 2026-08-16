# Portfolio Construction Comparison

## Construction comparison

```text
                     positions  maximum_weight  maximum_sector_weight  effective_positions  one_way_turnover
method                                                                                                      
alpha_risk_turnover  21.402597        0.049999               0.249999            20.424987          0.227074
cvar                 23.402597        0.049999               0.249999            20.383506          0.184383
median_mad_de        21.285714        0.049999               0.249995            20.457288          0.267476
score_weighted       25.000000        0.050000               0.250000            24.176025          0.233225
top_n_equal_weight   25.000000        0.040000               0.240000            25.000000          0.204737
```

## Predicted risk

```text
                     predicted_volatility  beta_vs_spy  maximum_sector_weight
method                                                                       
alpha_risk_turnover              0.190236     0.981225               0.249999
cvar                             0.195521     1.011956               0.249999
median_mad_de                    0.190582     0.991972               0.249995
score_weighted                   0.204894     1.066027               0.250000
top_n_equal_weight               0.199920     1.040742               0.240000
```

## Median-MAD diagnostics

```text
       median_daily_return  mad_daily  mad_violation  turnover_l1  one_way_turnover  portfolio_beta_vs_spy
count            77.000000  77.000000      77.000000    77.000000         76.000000              77.000000
mean              0.001770   0.008588       0.001014     0.528004          0.267476               0.991972
std               0.001004   0.001965       0.001628     0.168876          0.079267               0.075621
min              -0.001031   0.006033       0.000000     0.000000          0.090275               0.850025
25%               0.001390   0.007190       0.000000     0.424605          0.212942               0.937307
50%               0.002137   0.007916       0.000000     0.517213          0.259631               1.002343
75%               0.002394   0.010011       0.002011     0.635409          0.318298               1.043973
max               0.003269   0.013763       0.005763     0.998696          0.499348               1.142855
```

MAD limit exceedance rate: 37.66%

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
      median_mad_de           finite_objective   PASS           0                                  Median-MAD objective values must be finite.
      median_mad_de              finite_median   PASS           0                                      Median return estimates must be finite.
      median_mad_de                 finite_mad   PASS           0                                                MAD estimates must be finite.
      median_mad_de       minimum_observations   PASS           0                  Every optimization must use enough historical observations.
      median_mad_de             fully_invested   PASS           0                            Median-MAD portfolios must remain fully invested.
      median_mad_de      security_weight_limit   PASS           0                 Median-MAD positions must respect the security weight limit.
      median_mad_de        sector_weight_limit   PASS           0                     Median-MAD sector weights must respect the sector limit.
      median_mad_de portfolio_beta_lower_limit   PASS           0                 Median-MAD portfolio beta must remain above its lower bound.
      median_mad_de portfolio_beta_upper_limit   PASS           0                 Median-MAD portfolio beta must remain below its upper bound.
      median_mad_de   position_liquidity_limit   PASS           0                       Median-MAD positions must respect the liquidity limit.
      median_mad_de       nonnegative_turnover   PASS           0                                    Median-MAD turnover must be non-negative.
```
