# Security-Level Risk Estimates

## Methodology

- Volatility uses the latest 252 available daily adjusted-close returns.
- Beta uses up to 252 daily returns aligned with SPY.
- Liquidity uses a 60-session average dollar volume window.
- Every estimation window ends on or before the signal as-of date.
- No forward return or future target is used in this artifact.

## Coverage

```text
rows: 3850
dates: 77
tickers: 50
first_date: 2020-01-31
last_date: 2026-05-29
```

## Readiness checks

```text
                       check status  violations                                                  description
                 unique_keys   PASS           0        Risk estimates must have one row per date and ticker.
             signal_coverage   PASS           0 Risk estimates must exactly match the final signal universe.
        market_point_in_time   PASS           0       Market observations must not extend beyond as_of_date.
           spy_point_in_time   PASS           0          SPY observations must not extend beyond as_of_date.
   risk_window_point_in_time   PASS           0    Risk estimation windows must end on or before as_of_date.
              return_history   PASS           0                 Every row must have enough trailing returns.
                beta_history   PASS           0              Every row must have enough SPY-aligned returns.
           liquidity_history   PASS           0           Every row must have enough liquidity observations.
       finite_numeric_values   PASS           0             Stored risk and liquidity values must be finite.
           non_negative_risk   PASS           0                   Volatility estimates must be non-negative.
positive_liquidity_and_price   PASS           0               Liquidity and price measures must be positive.
    valid_market_correlation   PASS           0           Correlation with SPY must remain between -1 and 1.
```

## Distribution summary

```text
       annualized_volatility  annualized_downside_volatility  beta_vs_spy  correlation_vs_spy  average_dollar_volume
count            3850.000000                     3850.000000  3850.000000         3850.000000           3.850000e+03
mean                0.304016                        0.207731     0.906951            0.549816           2.688846e+09
std                 0.119126                        0.082599     0.444727            0.205951           5.135620e+09
min                 0.120600                        0.074172    -0.317660           -0.157984           1.928093e+08
25%                 0.220213                        0.150706     0.631499            0.429314           6.939450e+08
50%                 0.278652                        0.189533     0.879171            0.576554           1.040645e+09
75%                 0.358384                        0.245643     1.157208            0.709108           1.763614e+09
max                 0.894612                        0.565631     2.841143            0.928313           4.496035e+10
```

## Highest volatility on 2026-05-29

```text
ticker                 sector  annualized_volatility  beta_vs_spy  average_dollar_volume
  ORCL Information Technology               0.634883     1.719083           4.580332e+09
  TSLA Consumer Discretionary               0.462328     2.064194           2.372976e+10
  QCOM Information Technology               0.446418     1.742001           3.472912e+09
  AVGO Information Technology               0.427123     2.070752           8.771385e+09
   UNH            Health Care               0.398779     0.664508           2.670232e+09
   TXN Information Technology               0.387763     0.984352           1.828904e+09
   NKE Consumer Discretionary               0.379494     0.940956           1.055827e+09
   LLY            Health Care               0.378154     0.565562           2.999942e+09
  META Communication Services               0.347949     1.498642           9.835987e+09
  NVDA Information Technology               0.337608     1.800133           3.245455e+10
```

## Lowest dollar volume on 2026-05-29

```text
ticker                 sector  annualized_volatility  beta_vs_spy  average_dollar_volume
   APD              Materials               0.247250     0.332214           3.545047e+08
   PLD            Real Estate               0.208829     0.647939           4.830842e+08
    SO              Utilities               0.156000    -0.144381           4.898901e+08
   SHW              Materials               0.244823     0.836230           6.013697e+08
   EOG                 Energy               0.260982    -0.317660           6.090688e+08
   UPS            Industrials               0.291220     0.830882           6.377340e+08
   LOW Consumer Discretionary               0.255124     0.810765           6.381164e+08
   UNP            Industrials               0.213127     0.463654           7.799748e+08
   HON            Industrials               0.219352     0.735915           9.025093e+08
   NEE              Utilities               0.235676     0.159309           9.427304e+08
```
