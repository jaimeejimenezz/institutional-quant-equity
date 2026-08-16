# Shrinkage Covariance Model

## Methodology

Covariance matrices are estimated independently for every monthly signal date.

Only daily returns available on or before the corresponding as-of date are used.

Ledoit-Wolf shrinkage is applied to reduce sampling instability in the covariance matrix.

Daily covariance estimates are annualized using 252 trading sessions.

## Coverage

```text
dates: 77
matrix_rows: 192500
assets_per_date_min: 50
assets_per_date_max: 50
observations_min: 252
observations_max: 252
```

## Readiness checks

```text
                 check status  violations                                                                description
    unique_matrix_keys   PASS           0                       Every date and ticker pair must appear exactly once.
  signal_date_coverage   PASS           0          Covariance dates must exactly match the final alpha signal dates.
     matrix_dimensions   PASS           0 Every covariance matrix must contain the complete cross-product of assets.
       ticker_coverage   PASS           0       Every matrix must contain exactly the signal universe for that date.
       matrix_symmetry   PASS           0                                 Every covariance matrix must be symmetric.
non_negative_variances   PASS           0                  Covariance diagonals must contain non-negative variances.
 positive_semidefinite   PASS           0               Shrinkage covariance matrices must be positive semidefinite.
       valid_shrinkage   PASS           0                    Ledoit-Wolf shrinkage must remain between zero and one.
 point_in_time_windows   PASS           0       Every covariance estimation window must end on or before as_of_date.
    finite_covariances   PASS           0                                   All covariance estimates must be finite.
    valid_correlations   PASS           0                             All correlations must remain between -1 and 1.
```

## Diagnostic distribution

```text
       observations  shrinkage  minimum_eigenvalue  maximum_eigenvalue  sample_condition_number  shrinkage_condition_number  mean_pairwise_correlation
count          77.0  77.000000           77.000000           77.000000                77.000000                   77.000000                  77.000000
mean          252.0   0.073078            0.011123            2.107409               555.523650                  187.761286                   0.304879
std             0.0   0.030770            0.004455            1.706461               362.419138                  111.074594                   0.133207
min           252.0   0.026057            0.005288            0.589721               152.135126                   39.322916                   0.094124
25%           252.0   0.055965            0.007428            0.874765               271.266963                   96.583974                   0.219861
50%           252.0   0.066482            0.009256            1.531495               486.620972                  126.961261                   0.260322
75%           252.0   0.089823            0.015456            2.429889               674.089435                  301.865679                   0.394966
max           252.0   0.144294            0.019838            6.201682              1543.348634                  423.376345                   0.569639
```

## Latest-date diagnostics (2026-05-29)

```text
as_of_date  assets  observations window_start_date window_end_date  shrinkage  minimum_eigenvalue  maximum_eigenvalue  sample_condition_number  shrinkage_condition_number  mean_pairwise_correlation  median_pairwise_correlation  maximum_pairwise_correlation  minimum_pairwise_correlation
2026-05-29      50           252        2025-05-29      2026-05-29   0.140736            0.015621            0.614271               152.135126                   39.322916                   0.094124                     0.089142                      0.718555                      -0.23936
```

## Strongest latest-date correlations (2026-05-29)

```text
ticker_a ticker_b  correlation
     COP      EOG     0.718555
     EOG      COP     0.718555
      HD      LOW     0.717716
     LOW       HD     0.717716
      MA        V     0.668909
       V       MA     0.668909
     COP      CVX     0.653009
     CVX      COP     0.653009
     BAC      WFC     0.637238
     WFC      BAC     0.637238
     CVX      EOG     0.597492
     EOG      CVX     0.597492
      HD      SHW     0.573935
     SHW       HD     0.573935
     LOW      SHW     0.572544
```
