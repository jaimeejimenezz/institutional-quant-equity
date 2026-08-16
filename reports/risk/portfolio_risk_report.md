# Portfolio Risk Model

## Reference portfolio

The diagnostic portfolio selects the top 20 securities from the final alpha ranking and assigns equal weights.

The reference portfolio is used only to validate the risk engine. Portfolio construction methods are evaluated separately downstream.

## Risk methodology

- Portfolio volatility is calculated from the Ledoit-Wolf annualized covariance matrix.
- Portfolio beta is the weighted average of security betas versus SPY.
- Security risk contributions use the Euler decomposition of portfolio volatility.
- Sector exposure is compared with the equal-weight universe sector allocation.
- Liquidity diagnostics compare position notional with trailing Average Dollar Volume.

## Readiness checks

```text
                        check status  violations                                                         description
         unique_summary_dates   PASS           0                    Portfolio summary must contain one row per date.
     unique_contribution_keys   PASS           0      Security risk contributions must have unique date-ticker keys.
           weights_sum_to_one   PASS           0                     Security weights must sum to one on every date.
    sector_weights_sum_to_one   PASS           0                       Sector weights must sum to one on every date.
risk_contributions_sum_to_one   PASS           0                  Security risk contribution shares must sum to one.
          euler_risk_identity   PASS           0 Component risk contributions must reconstruct portfolio volatility.
positive_portfolio_volatility   PASS           0                    Predicted portfolio volatility must be positive.
 positive_effective_positions   PASS           0                          Effective position count must be positive.
non_negative_liquidation_days   PASS           0                    Estimated liquidation days must be non-negative.
        finite_summary_values   PASS           0                      Core portfolio risk statistics must be finite.
```

## Distribution across dates

```text
       predicted_volatility  portfolio_beta_vs_spy  concentration_hhi  effective_positions  maximum_sector_weight  maximum_position_adv_fraction  maximum_liquidation_days
count             77.000000              77.000000              77.00         7.700000e+01              77.000000                      77.000000                 77.000000
mean               0.208903               1.081214               0.05         2.000000e+01               0.227273                       0.000151                  0.001509
std                0.071797               0.055392               0.00         3.576010e-15               0.041790                       0.000037                  0.000370
min                0.117207               0.942773               0.05         2.000000e+01               0.150000                       0.000076                  0.000761
25%                0.149124               1.043219               0.05         2.000000e+01               0.200000                       0.000125                  0.001254
50%                0.201691               1.084407               0.05         2.000000e+01               0.200000                       0.000141                  0.001409
75%                0.240031               1.117422               0.05         2.000000e+01               0.250000                       0.000170                  0.001698
max                0.387598               1.211194               0.05         2.000000e+01               0.350000                       0.000259                  0.002593
```

## Latest portfolio risk (2026-05-29)

```text
as_of_date  positions  portfolio_value  predicted_volatility  predicted_variance  portfolio_beta_vs_spy  maximum_weight  minimum_weight  concentration_hhi  effective_positions  maximum_sector_weight  maximum_active_sector_weight  maximum_position_adv_fraction  weighted_position_adv_fraction  maximum_liquidation_days  weighted_liquidation_days  risk_contribution_sum
2026-05-29         20        1000000.0              0.129667            0.016814                0.99695            0.05            0.05               0.05                 20.0                    0.2                          0.05                       0.000141                        0.000037                   0.00141                   0.000373               0.129667
```

## Largest security risk contributions (2026-05-29)

```text
ticker                 sector  weight  beta_vs_spy  risk_contribution_share  position_adv_fraction  liquidation_days
  ORCL Information Technology    0.05     1.719083                 0.091169               0.000011          0.000109
  TSLA Consumer Discretionary    0.05     2.064194                 0.086617               0.000002          0.000021
  QCOM Information Technology    0.05     1.742001                 0.084689               0.000014          0.000144
  AVGO Information Technology    0.05     2.070752                 0.076660               0.000006          0.000057
   CAT            Industrials    0.05     1.597735                 0.073376               0.000025          0.000248
    GS             Financials    0.05     1.526586                 0.062963               0.000027          0.000267
   NKE Consumer Discretionary    0.05     0.940956                 0.062144               0.000047          0.000474
   TXN Information Technology    0.05     0.984352                 0.057356               0.000027          0.000273
   UNH            Health Care    0.05     0.664508                 0.052520               0.000019          0.000187
   UPS            Industrials    0.05     0.830882                 0.051870               0.000078          0.000784
```

## Sector exposures (2026-05-29)

```text
                sector  portfolio_weight  universe_equal_weight  active_weight  positions  risk_contribution_share
Information Technology              0.20                   0.16           0.04          4                 0.309874
Consumer Discretionary              0.15                   0.12           0.03          3                 0.192762
            Financials              0.15                   0.14           0.01          3                 0.154787
           Health Care              0.15                   0.14           0.01          3                 0.112918
Communication Services              0.10                   0.08           0.02          2                 0.063829
           Industrials              0.10                   0.10           0.00          2                 0.125246
      Consumer Staples              0.05                   0.10          -0.05          1                 0.008579
                Energy              0.05                   0.06          -0.01          1                 0.005501
             Materials              0.05                   0.04           0.01          1                 0.026504
```
