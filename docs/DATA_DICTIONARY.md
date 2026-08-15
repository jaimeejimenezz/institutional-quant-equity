# Data Dictionary — Modeling Panel

## Dataset

`data/processed/modeling_panel.parquet` is the canonical monthly dataset used by the predictive-modeling stages.

- Rows: `7550`
- Rebalance dates: `151`
- Companies: `50`
- Candidate model features: `91`
- Modeling rows: `7450`
- Inference-only rows: `100`

## Temporal contract

- `as_of_date` is the information cutoff for every predictor.
- Technical features use market data no later than `technical_latest_market_date`, and that date must be <= `as_of_date`.
- Fundamental factors use SEC accounting information that was available on or before `as_of_date`.
- `first_future_date`, `target_end_date`, `target_21d`, `target_21d_excess` and `label_top_quintile` are future information and never belong to `MODEL_FEATURE_COLUMNS`.
- Rows without a complete future horizon are preserved as `inference_only`.

## Training-time transformation boundary

The master dataset intentionally does not perform global imputation, fitted scaling, PCA, model-based feature selection or hyperparameter fitting.

Any operation that learns parameters from observations must be fitted only inside the training sample of each walk-forward fold.

## Predictor groups

- Technical sector-neutral signals: `19`
- Fundamental global z-scores: `24`
- Fundamental sector z-scores: `24`
- Fundamental missingness indicators: `24`
- Total candidate predictors: `91`

## Predictor coverage

| feature_group | features | mean_coverage | minimum_coverage | latest_mean_coverage |
| --- | --- | --- | --- | --- |
| fundamental_global | 24 | 79.91% | 44.56% | 87.58% |
| fundamental_missing | 24 | 100.00% | 100.00% | 100.00% |
| fundamental_sector | 24 | 69.63% | 37.01% | 77.58% |
| technical | 19 | 97.87% | 92.05% | 100.00% |

## Column index

| column | role | feature_group | family | source_column | overall_coverage | latest_coverage |
| --- | --- | --- | --- | --- | --- | --- |
| as_of_date | identifier | identifier | Calendar | as_of_date | 100.00% | 100.00% |
| ticker | identifier | identifier | Security | ticker | 100.00% | 100.00% |
| sector | metadata | metadata | Security | sector | 100.00% | 100.00% |
| technical_latest_market_date | provenance | provenance | Technical | latest_market_date | 100.00% | 100.00% |
| observations_available | provenance | provenance | Technical | observations_available | 100.00% | 100.00% |
| tech__momentum_12_1_sector_neutral | predictor | technical | Technical | momentum_12_1_sector_neutral | 92.05% | 100.00% |
| tech__momentum_6_1_sector_neutral | predictor | technical | Technical | momentum_6_1_sector_neutral | 96.03% | 100.00% |
| tech__return_3m_sector_neutral | predictor | technical | Technical | return_3m_sector_neutral | 98.01% | 100.00% |
| tech__return_1m_sector_neutral | predictor | technical | Technical | return_1m_sector_neutral | 99.34% | 100.00% |
| tech__return_1w_sector_neutral | predictor | technical | Technical | return_1w_sector_neutral | 100.00% | 100.00% |
| tech__reversal_1m_sector_neutral | predictor | technical | Technical | reversal_1m_sector_neutral | 99.34% | 100.00% |
| tech__volatility_20d_sector_neutral | predictor | technical | Technical | volatility_20d_sector_neutral | 100.00% | 100.00% |
| tech__volatility_60d_sector_neutral | predictor | technical | Technical | volatility_60d_sector_neutral | 98.68% | 100.00% |
| tech__downside_volatility_60d_sector_neutral | predictor | technical | Technical | downside_volatility_60d_sector_neutral | 98.68% | 100.00% |
| tech__beta_60d_market_sector_neutral | predictor | technical | Technical | beta_60d_market_sector_neutral | 98.68% | 100.00% |
| tech__max_drawdown_126d_sector_neutral | predictor | technical | Technical | max_drawdown_126d_sector_neutral | 96.03% | 100.00% |
| tech__distance_sma_50d_sector_neutral | predictor | technical | Technical | distance_sma_50d_sector_neutral | 98.68% | 100.00% |
| tech__distance_sma_200d_sector_neutral | predictor | technical | Technical | distance_sma_200d_sector_neutral | 94.04% | 100.00% |
| tech__sma_50_200_spread_sector_neutral | predictor | technical | Technical | sma_50_200_spread_sector_neutral | 94.04% | 100.00% |
| tech__positive_day_ratio_60d_sector_neutral | predictor | technical | Technical | positive_day_ratio_60d_sector_neutral | 98.68% | 100.00% |
| tech__average_dollar_volume_20d_sector_neutral | predictor | technical | Technical | average_dollar_volume_20d_sector_neutral | 100.00% | 100.00% |
| tech__dollar_volume_change_20d_60d_sector_neutral | predictor | technical | Technical | dollar_volume_change_20d_60d_sector_neutral | 98.68% | 100.00% |
| tech__amihud_illiquidity_20d_sector_neutral | predictor | technical | Technical | amihud_illiquidity_20d_sector_neutral | 100.00% | 100.00% |
| tech__zero_volume_ratio_60d_sector_neutral | predictor | technical | Technical | zero_volume_ratio_60d_sector_neutral | 98.68% | 100.00% |
| fund__earnings_yield_zscore | predictor | fundamental_global | Value | earnings_yield_zscore | 92.87% | 96.00% |
| fund__sales_yield_zscore | predictor | fundamental_global | Value | sales_yield_zscore | 90.25% | 94.00% |
| fund__book_to_market_zscore | predictor | fundamental_global | Value | book_to_market_zscore | 94.08% | 96.00% |
| fund__fcf_yield_zscore | predictor | fundamental_global | Value | fcf_yield_zscore | 69.72% | 80.00% |
| fund__roe_zscore | predictor | fundamental_global | Quality | roe_zscore | 93.48% | 94.00% |
| fund__roa_zscore | predictor | fundamental_global | Quality | roa_zscore | 96.91% | 100.00% |
| fund__gross_profitability_zscore | predictor | fundamental_global | Quality | gross_profitability_zscore | 44.56% | 46.00% |
| fund__gross_margin_zscore | predictor | fundamental_global | Quality | gross_margin_zscore | 44.56% | 46.00% |
| fund__operating_margin_zscore | predictor | fundamental_global | Quality | operating_margin_zscore | 76.38% | 82.00% |
| fund__net_margin_zscore | predictor | fundamental_global | Quality | net_margin_zscore | 93.34% | 98.00% |
| fund__cash_conversion_zscore | predictor | fundamental_global | Quality | cash_conversion_zscore | 89.46% | 98.00% |
| fund__debt_to_assets_zscore | predictor | fundamental_global | Leverage | debt_to_assets_zscore | 67.21% | 74.00% |
| fund__net_debt_to_assets_zscore | predictor | fundamental_global | Leverage | net_debt_to_assets_zscore | 67.21% | 74.00% |
| fund__current_ratio_zscore | predictor | fundamental_global | Solvency | current_ratio_zscore | 86.17% | 88.00% |
| fund__interest_coverage_zscore | predictor | fundamental_global | Solvency | interest_coverage_zscore | 65.85% | 68.00% |
| fund__capex_to_assets_zscore | predictor | fundamental_global | Investment | capex_to_assets_zscore | 81.07% | 86.00% |
| fund__accruals_zscore | predictor | fundamental_global | Accruals | accruals_zscore | 89.46% | 98.00% |
| fund__revenue_growth_yoy_zscore | predictor | fundamental_global | Growth | revenue_growth_yoy_zscore | 86.50% | 98.00% |
| fund__net_income_growth_yoy_zscore | predictor | fundamental_global | Growth | net_income_growth_yoy_zscore | 88.83% | 100.00% |
| fund__operating_cash_flow_growth_yoy_zscore | predictor | fundamental_global | Growth | operating_cash_flow_growth_yoy_zscore | 79.81% | 96.00% |
| fund__asset_growth_yoy_zscore | predictor | fundamental_global | Investment | asset_growth_yoy_zscore | 90.23% | 100.00% |
| fund__revenue_growth_acceleration_zscore | predictor | fundamental_global | Growth | revenue_growth_acceleration_zscore | 78.72% | 98.00% |
| fund__net_income_growth_acceleration_zscore | predictor | fundamental_global | Growth | net_income_growth_acceleration_zscore | 80.81% | 100.00% |
| fund__operating_cash_flow_growth_acceleration_zscore | predictor | fundamental_global | Growth | operating_cash_flow_growth_acceleration_zscore | 70.42% | 92.00% |
| fund__earnings_yield_sector_zscore | predictor | fundamental_sector | Value | earnings_yield_sector_zscore | 82.45% | 86.00% |
| fund__sales_yield_sector_zscore | predictor | fundamental_sector | Value | sales_yield_sector_zscore | 80.24% | 84.00% |
| fund__book_to_market_sector_zscore | predictor | fundamental_sector | Value | book_to_market_sector_zscore | 83.42% | 86.00% |
| fund__fcf_yield_sector_zscore | predictor | fundamental_sector | Value | fcf_yield_sector_zscore | 57.46% | 68.00% |
| fund__roe_sector_zscore | predictor | fundamental_sector | Quality | roe_sector_zscore | 83.06% | 84.00% |
| fund__roa_sector_zscore | predictor | fundamental_sector | Quality | roa_sector_zscore | 86.49% | 90.00% |
| fund__gross_profitability_sector_zscore | predictor | fundamental_sector | Quality | gross_profitability_sector_zscore | 37.01% | 40.00% |
| fund__gross_margin_sector_zscore | predictor | fundamental_sector | Quality | gross_margin_sector_zscore | 37.01% | 40.00% |
| fund__operating_margin_sector_zscore | predictor | fundamental_sector | Quality | operating_margin_sector_zscore | 64.66% | 70.00% |
| fund__net_margin_sector_zscore | predictor | fundamental_sector | Quality | net_margin_sector_zscore | 83.72% | 88.00% |
| fund__cash_conversion_sector_zscore | predictor | fundamental_sector | Quality | cash_conversion_sector_zscore | 80.00% | 90.00% |
| fund__debt_to_assets_sector_zscore | predictor | fundamental_sector | Leverage | debt_to_assets_sector_zscore | 52.87% | 60.00% |
| fund__net_debt_to_assets_sector_zscore | predictor | fundamental_sector | Leverage | net_debt_to_assets_sector_zscore | 52.87% | 60.00% |
| fund__current_ratio_sector_zscore | predictor | fundamental_sector | Solvency | current_ratio_sector_zscore | 73.62% | 76.00% |
| fund__interest_coverage_sector_zscore | predictor | fundamental_sector | Solvency | interest_coverage_sector_zscore | 52.81% | 56.00% |
| fund__capex_to_assets_sector_zscore | predictor | fundamental_sector | Investment | capex_to_assets_sector_zscore | 70.94% | 76.00% |
| fund__accruals_sector_zscore | predictor | fundamental_sector | Accruals | accruals_sector_zscore | 80.00% | 90.00% |
| fund__revenue_growth_yoy_sector_zscore | predictor | fundamental_sector | Growth | revenue_growth_yoy_sector_zscore | 77.28% | 88.00% |
| fund__net_income_growth_yoy_sector_zscore | predictor | fundamental_sector | Growth | net_income_growth_yoy_sector_zscore | 79.26% | 90.00% |
| fund__operating_cash_flow_growth_yoy_sector_zscore | predictor | fundamental_sector | Growth | operating_cash_flow_growth_yoy_sector_zscore | 71.05% | 88.00% |
| fund__asset_growth_yoy_sector_zscore | predictor | fundamental_sector | Investment | asset_growth_yoy_sector_zscore | 80.46% | 90.00% |
| fund__revenue_growth_acceleration_sector_zscore | predictor | fundamental_sector | Growth | revenue_growth_acceleration_sector_zscore | 70.29% | 88.00% |
| fund__net_income_growth_acceleration_sector_zscore | predictor | fundamental_sector | Growth | net_income_growth_acceleration_sector_zscore | 72.03% | 90.00% |
| fund__operating_cash_flow_growth_acceleration_sector_zscore | predictor | fundamental_sector | Growth | operating_cash_flow_growth_acceleration_sector_zscore | 62.24% | 84.00% |
| fund__earnings_yield_missing | predictor | fundamental_missing | Value | earnings_yield_missing | 100.00% | 100.00% |
| fund__sales_yield_missing | predictor | fundamental_missing | Value | sales_yield_missing | 100.00% | 100.00% |
| fund__book_to_market_missing | predictor | fundamental_missing | Value | book_to_market_missing | 100.00% | 100.00% |
| fund__fcf_yield_missing | predictor | fundamental_missing | Value | fcf_yield_missing | 100.00% | 100.00% |
| fund__roe_missing | predictor | fundamental_missing | Quality | roe_missing | 100.00% | 100.00% |
| fund__roa_missing | predictor | fundamental_missing | Quality | roa_missing | 100.00% | 100.00% |
| fund__gross_profitability_missing | predictor | fundamental_missing | Quality | gross_profitability_missing | 100.00% | 100.00% |
| fund__gross_margin_missing | predictor | fundamental_missing | Quality | gross_margin_missing | 100.00% | 100.00% |
| fund__operating_margin_missing | predictor | fundamental_missing | Quality | operating_margin_missing | 100.00% | 100.00% |
| fund__net_margin_missing | predictor | fundamental_missing | Quality | net_margin_missing | 100.00% | 100.00% |
| fund__cash_conversion_missing | predictor | fundamental_missing | Quality | cash_conversion_missing | 100.00% | 100.00% |
| fund__debt_to_assets_missing | predictor | fundamental_missing | Leverage | debt_to_assets_missing | 100.00% | 100.00% |
| fund__net_debt_to_assets_missing | predictor | fundamental_missing | Leverage | net_debt_to_assets_missing | 100.00% | 100.00% |
| fund__current_ratio_missing | predictor | fundamental_missing | Solvency | current_ratio_missing | 100.00% | 100.00% |
| fund__interest_coverage_missing | predictor | fundamental_missing | Solvency | interest_coverage_missing | 100.00% | 100.00% |
| fund__capex_to_assets_missing | predictor | fundamental_missing | Investment | capex_to_assets_missing | 100.00% | 100.00% |
| fund__accruals_missing | predictor | fundamental_missing | Accruals | accruals_missing | 100.00% | 100.00% |
| fund__revenue_growth_yoy_missing | predictor | fundamental_missing | Growth | revenue_growth_yoy_missing | 100.00% | 100.00% |
| fund__net_income_growth_yoy_missing | predictor | fundamental_missing | Growth | net_income_growth_yoy_missing | 100.00% | 100.00% |
| fund__operating_cash_flow_growth_yoy_missing | predictor | fundamental_missing | Growth | operating_cash_flow_growth_yoy_missing | 100.00% | 100.00% |
| fund__asset_growth_yoy_missing | predictor | fundamental_missing | Investment | asset_growth_yoy_missing | 100.00% | 100.00% |
| fund__revenue_growth_acceleration_missing | predictor | fundamental_missing | Growth | revenue_growth_acceleration_missing | 100.00% | 100.00% |
| fund__net_income_growth_acceleration_missing | predictor | fundamental_missing | Growth | net_income_growth_acceleration_missing | 100.00% | 100.00% |
| fund__operating_cash_flow_growth_acceleration_missing | predictor | fundamental_missing | Growth | operating_cash_flow_growth_acceleration_missing | 100.00% | 100.00% |
| first_future_date | target_metadata | target_metadata | Target | first_future_date | 98.68% | 0.00% |
| target_end_date | target_metadata | target_metadata | Target | target_end_date | 98.68% | 0.00% |
| horizon_sessions | target_metadata | target_metadata | Target | horizon_sessions | 98.68% | 0.00% |
| target_21d | target | target | Target | target_21d | 98.68% | 0.00% |
| target_21d_excess | target | target | Target | target_21d_excess | 98.68% | 0.00% |
| label_top_quintile | target | target | Target | label_top_quintile | 98.68% | 0.00% |
| technical_missing_count | diagnostic | diagnostic | Data quality | Derived | 100.00% | 100.00% |
| fundamental_global_missing_count | diagnostic | diagnostic | Data quality | Derived | 100.00% | 100.00% |
| fundamental_sector_missing_count | diagnostic | diagnostic | Data quality | Derived | 100.00% | 100.00% |
| model_feature_missing_count | diagnostic | diagnostic | Data quality | Derived | 100.00% | 100.00% |
| has_target | diagnostic | diagnostic | Sample | Derived | 100.00% | 100.00% |
| sample_role | diagnostic | diagnostic | Sample | Derived | 100.00% | 100.00% |

## Detailed column definitions

### `as_of_date`

- **Role:** identifier
- **Feature group:** identifier
- **Family:** Calendar
- **Source dataset:** `data/processed/rebalance_calendar.parquet`
- **Source column:** `as_of_date`
- **Model input:** `False`
- **Availability:** Monthly rebalance date; all predictors must be available on or before this date.
- **Calculation:** Last available market session selected for the monthly rebalance observation.
- **Missing values:** Must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `ticker`

- **Role:** identifier
- **Feature group:** identifier
- **Family:** Security
- **Source dataset:** `data/reference/universe_v1.csv`
- **Source column:** `ticker`
- **Model input:** `False`
- **Availability:** Security identifier attached to each monthly observation.
- **Calculation:** Normalized uppercase ticker from the configured equity universe.
- **Missing values:** Must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `sector`

- **Role:** metadata
- **Feature group:** metadata
- **Family:** Security
- **Source dataset:** `data/reference/universe_v1.csv`
- **Source column:** `sector`
- **Model input:** `False`
- **Availability:** Universe metadata known for the security.
- **Calculation:** Sector classification joined by ticker.
- **Missing values:** Must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `technical_latest_market_date`

- **Role:** provenance
- **Feature group:** provenance
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `latest_market_date`
- **Model input:** `False`
- **Availability:** Must be less than or equal to as_of_date.
- **Calculation:** Latest daily market observation used by the technical-feature calculation.
- **Missing values:** Must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `observations_available`

- **Role:** provenance
- **Feature group:** provenance
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `observations_available`
- **Model input:** `False`
- **Availability:** Counts historical observations available up to as_of_date.
- **Calculation:** Number of historical market observations available to technical calculations.
- **Missing values:** Must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `tech__momentum_12_1_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `momentum_12_1_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'momentum_12_1' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 92.05%
- **Latest-date coverage:** 100.00%

### `tech__momentum_6_1_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `momentum_6_1_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'momentum_6_1' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 96.03%
- **Latest-date coverage:** 100.00%

### `tech__return_3m_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `return_3m_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'return_3m' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 98.01%
- **Latest-date coverage:** 100.00%

### `tech__return_1m_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `return_1m_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'return_1m' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 99.34%
- **Latest-date coverage:** 100.00%

### `tech__return_1w_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `return_1w_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'return_1w' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `tech__reversal_1m_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `reversal_1m_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'reversal_1m' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 99.34%
- **Latest-date coverage:** 100.00%

### `tech__volatility_20d_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `volatility_20d_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'volatility_20d' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `tech__volatility_60d_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `volatility_60d_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'volatility_60d' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 100.00%

### `tech__downside_volatility_60d_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `downside_volatility_60d_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'downside_volatility_60d' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 100.00%

### `tech__beta_60d_market_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `beta_60d_market_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'beta_60d_market' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 100.00%

### `tech__max_drawdown_126d_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `max_drawdown_126d_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'max_drawdown_126d' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 96.03%
- **Latest-date coverage:** 100.00%

### `tech__distance_sma_50d_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `distance_sma_50d_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'distance_sma_50d' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 100.00%

### `tech__distance_sma_200d_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `distance_sma_200d_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'distance_sma_200d' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 94.04%
- **Latest-date coverage:** 100.00%

### `tech__sma_50_200_spread_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `sma_50_200_spread_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'sma_50_200_spread' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 94.04%
- **Latest-date coverage:** 100.00%

### `tech__positive_day_ratio_60d_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `positive_day_ratio_60d_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'positive_day_ratio_60d' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 100.00%

### `tech__average_dollar_volume_20d_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `average_dollar_volume_20d_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'average_dollar_volume_20d' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `tech__dollar_volume_change_20d_60d_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `dollar_volume_change_20d_60d_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'dollar_volume_change_20d_60d' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 100.00%

### `tech__amihud_illiquidity_20d_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `amihud_illiquidity_20d_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'amihud_illiquidity_20d' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `tech__zero_volume_ratio_60d_sector_neutral`

- **Role:** predictor
- **Feature group:** technical
- **Family:** Technical
- **Source dataset:** `data/processed/features_technical_monthly.parquet`
- **Source column:** `zero_volume_ratio_60d_sector_neutral`
- **Model input:** `True`
- **Availability:** Uses market observations only through technical_latest_market_date, which must be <= as_of_date.
- **Calculation:** Raw technical factor 'zero_volume_ratio_60d' is winsorized cross-sectionally by date, standardized by date, then sector-neutralized. The raw factor formula is documented in docs/TECHNICAL_FEATURES.md.
- **Missing values:** NaN generally indicates insufficient historical market observations or an unavailable underlying technical calculation.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 100.00%

### `fund__earnings_yield_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `earnings_yield_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Net Income TTM / Market Cap Proxy. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 92.87%
- **Latest-date coverage:** 96.00%

### `fund__sales_yield_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `sales_yield_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Revenue TTM / Market Cap Proxy. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 90.25%
- **Latest-date coverage:** 94.00%

### `fund__book_to_market_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `book_to_market_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Equity / Market Cap Proxy. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 94.08%
- **Latest-date coverage:** 96.00%

### `fund__fcf_yield_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `fcf_yield_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** (Operating Cash Flow TTM - CAPEX TTM) / Market Cap Proxy. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 69.72%
- **Latest-date coverage:** 80.00%

### `fund__roe_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `roe_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Net Income TTM / Equity. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 93.48%
- **Latest-date coverage:** 94.00%

### `fund__roa_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `roa_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Net Income TTM / Assets. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 96.91%
- **Latest-date coverage:** 100.00%

### `fund__gross_profitability_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `gross_profitability_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Gross Profit TTM / Assets. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 44.56%
- **Latest-date coverage:** 46.00%

### `fund__gross_margin_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `gross_margin_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Gross Profit TTM / Revenue TTM. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 44.56%
- **Latest-date coverage:** 46.00%

### `fund__operating_margin_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `operating_margin_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Operating Income TTM / Revenue TTM. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 76.38%
- **Latest-date coverage:** 82.00%

### `fund__net_margin_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_margin_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Net Income TTM / Revenue TTM. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 93.34%
- **Latest-date coverage:** 98.00%

### `fund__cash_conversion_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `cash_conversion_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Operating Cash Flow TTM / Net Income TTM. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 89.46%
- **Latest-date coverage:** 98.00%

### `fund__debt_to_assets_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Leverage
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `debt_to_assets_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** (Current Debt + Non-current Debt) / Assets. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 67.21%
- **Latest-date coverage:** 74.00%

### `fund__net_debt_to_assets_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Leverage
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_debt_to_assets_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** (Total Debt - Cash) / Assets. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 67.21%
- **Latest-date coverage:** 74.00%

### `fund__current_ratio_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Solvency
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `current_ratio_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Current Assets / Current Liabilities. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 86.17%
- **Latest-date coverage:** 88.00%

### `fund__interest_coverage_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Solvency
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `interest_coverage_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Operating Income TTM / Interest Expense TTM. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 65.85%
- **Latest-date coverage:** 68.00%

### `fund__capex_to_assets_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Investment
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `capex_to_assets_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** CAPEX TTM / Assets. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 81.07%
- **Latest-date coverage:** 86.00%

### `fund__accruals_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Accruals
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `accruals_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** (Net Income TTM - Operating Cash Flow TTM) / Assets. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 89.46%
- **Latest-date coverage:** 98.00%

### `fund__revenue_growth_yoy_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `revenue_growth_yoy_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Revenue TTM / Revenue TTM 12M Ago - 1. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 86.50%
- **Latest-date coverage:** 98.00%

### `fund__net_income_growth_yoy_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_income_growth_yoy_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** (Net Income TTM - Net Income TTM 12M Ago) / abs(Net Income TTM 12M Ago). The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 88.83%
- **Latest-date coverage:** 100.00%

### `fund__operating_cash_flow_growth_yoy_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `operating_cash_flow_growth_yoy_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** (Operating Cash Flow TTM - Operating Cash Flow TTM 12M Ago) / abs(Operating Cash Flow TTM 12M Ago). The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 79.81%
- **Latest-date coverage:** 96.00%

### `fund__asset_growth_yoy_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Investment
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `asset_growth_yoy_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Assets / Assets 12M Ago - 1. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 90.23%
- **Latest-date coverage:** 100.00%

### `fund__revenue_growth_acceleration_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `revenue_growth_acceleration_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Current Revenue Growth YoY - Revenue Growth YoY 12M Ago. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 78.72%
- **Latest-date coverage:** 98.00%

### `fund__net_income_growth_acceleration_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_income_growth_acceleration_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Current Net Income Growth YoY - Net Income Growth YoY 12M Ago. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 80.81%
- **Latest-date coverage:** 100.00%

### `fund__operating_cash_flow_growth_acceleration_zscore`

- **Role:** predictor
- **Feature group:** fundamental_global
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `operating_cash_flow_growth_acceleration_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information that was available on or before as_of_date.
- **Calculation:** Current Operating Cash Flow Growth YoY - Operating Cash Flow Growth YoY 12M Ago. The raw factor is winsorized by rebalance date and standardized across available companies on that date.
- **Missing values:** NaN means the underlying accounting factor or sufficient cross-sectional information was unavailable.
- **Overall coverage:** 70.42%
- **Latest-date coverage:** 92.00%

### `fund__earnings_yield_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `earnings_yield_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Net Income TTM / Market Cap Proxy. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 82.45%
- **Latest-date coverage:** 86.00%

### `fund__sales_yield_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `sales_yield_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Revenue TTM / Market Cap Proxy. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 80.24%
- **Latest-date coverage:** 84.00%

### `fund__book_to_market_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `book_to_market_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Equity / Market Cap Proxy. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 83.42%
- **Latest-date coverage:** 86.00%

### `fund__fcf_yield_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `fcf_yield_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** (Operating Cash Flow TTM - CAPEX TTM) / Market Cap Proxy. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 57.46%
- **Latest-date coverage:** 68.00%

### `fund__roe_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `roe_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Net Income TTM / Equity. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 83.06%
- **Latest-date coverage:** 84.00%

### `fund__roa_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `roa_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Net Income TTM / Assets. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 86.49%
- **Latest-date coverage:** 90.00%

### `fund__gross_profitability_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `gross_profitability_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Gross Profit TTM / Assets. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 37.01%
- **Latest-date coverage:** 40.00%

### `fund__gross_margin_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `gross_margin_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Gross Profit TTM / Revenue TTM. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 37.01%
- **Latest-date coverage:** 40.00%

### `fund__operating_margin_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `operating_margin_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Operating Income TTM / Revenue TTM. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 64.66%
- **Latest-date coverage:** 70.00%

### `fund__net_margin_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_margin_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Net Income TTM / Revenue TTM. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 83.72%
- **Latest-date coverage:** 88.00%

### `fund__cash_conversion_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `cash_conversion_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Operating Cash Flow TTM / Net Income TTM. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 80.00%
- **Latest-date coverage:** 90.00%

### `fund__debt_to_assets_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Leverage
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `debt_to_assets_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** (Current Debt + Non-current Debt) / Assets. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 52.87%
- **Latest-date coverage:** 60.00%

### `fund__net_debt_to_assets_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Leverage
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_debt_to_assets_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** (Total Debt - Cash) / Assets. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 52.87%
- **Latest-date coverage:** 60.00%

### `fund__current_ratio_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Solvency
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `current_ratio_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Current Assets / Current Liabilities. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 73.62%
- **Latest-date coverage:** 76.00%

### `fund__interest_coverage_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Solvency
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `interest_coverage_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Operating Income TTM / Interest Expense TTM. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 52.81%
- **Latest-date coverage:** 56.00%

### `fund__capex_to_assets_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Investment
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `capex_to_assets_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** CAPEX TTM / Assets. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 70.94%
- **Latest-date coverage:** 76.00%

### `fund__accruals_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Accruals
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `accruals_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** (Net Income TTM - Operating Cash Flow TTM) / Assets. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 80.00%
- **Latest-date coverage:** 90.00%

### `fund__revenue_growth_yoy_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `revenue_growth_yoy_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Revenue TTM / Revenue TTM 12M Ago - 1. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 77.28%
- **Latest-date coverage:** 88.00%

### `fund__net_income_growth_yoy_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_income_growth_yoy_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** (Net Income TTM - Net Income TTM 12M Ago) / abs(Net Income TTM 12M Ago). The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 79.26%
- **Latest-date coverage:** 90.00%

### `fund__operating_cash_flow_growth_yoy_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `operating_cash_flow_growth_yoy_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** (Operating Cash Flow TTM - Operating Cash Flow TTM 12M Ago) / abs(Operating Cash Flow TTM 12M Ago). The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 71.05%
- **Latest-date coverage:** 88.00%

### `fund__asset_growth_yoy_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Investment
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `asset_growth_yoy_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Assets / Assets 12M Ago - 1. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 80.46%
- **Latest-date coverage:** 90.00%

### `fund__revenue_growth_acceleration_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `revenue_growth_acceleration_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Current Revenue Growth YoY - Revenue Growth YoY 12M Ago. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 70.29%
- **Latest-date coverage:** 88.00%

### `fund__net_income_growth_acceleration_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_income_growth_acceleration_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Current Net Income Growth YoY - Net Income Growth YoY 12M Ago. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 72.03%
- **Latest-date coverage:** 90.00%

### `fund__operating_cash_flow_growth_acceleration_sector_zscore`

- **Role:** predictor
- **Feature group:** fundamental_sector
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `operating_cash_flow_growth_acceleration_sector_zscore`
- **Model input:** `True`
- **Availability:** Uses only fundamental information available on or before as_of_date and compares companies within the same sector and rebalance date.
- **Calculation:** Current Operating Cash Flow Growth YoY - Operating Cash Flow Growth YoY 12M Ago. The factor is winsorized and standardized relative to available companies in the same sector and rebalance date.
- **Missing values:** NaN means the raw factor was unavailable or the sector cross-section was insufficient.
- **Overall coverage:** 62.24%
- **Latest-date coverage:** 84.00%

### `fund__earnings_yield_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `earnings_yield_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'earnings_yield' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__sales_yield_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `sales_yield_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'sales_yield' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__book_to_market_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `book_to_market_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'book_to_market' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__fcf_yield_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Value
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `fcf_yield_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'fcf_yield' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__roe_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `roe_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'roe' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__roa_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `roa_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'roa' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__gross_profitability_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `gross_profitability_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'gross_profitability' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__gross_margin_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `gross_margin_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'gross_margin' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__operating_margin_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `operating_margin_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'operating_margin' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__net_margin_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_margin_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'net_margin' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__cash_conversion_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Quality
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `cash_conversion_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'cash_conversion' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__debt_to_assets_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Leverage
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `debt_to_assets_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'debt_to_assets' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__net_debt_to_assets_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Leverage
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_debt_to_assets_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'net_debt_to_assets' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__current_ratio_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Solvency
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `current_ratio_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'current_ratio' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__interest_coverage_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Solvency
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `interest_coverage_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'interest_coverage' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__capex_to_assets_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Investment
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `capex_to_assets_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'capex_to_assets' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__accruals_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Accruals
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `accruals_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'accruals' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__revenue_growth_yoy_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `revenue_growth_yoy_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'revenue_growth_yoy' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__net_income_growth_yoy_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_income_growth_yoy_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'net_income_growth_yoy' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__operating_cash_flow_growth_yoy_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `operating_cash_flow_growth_yoy_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'operating_cash_flow_growth_yoy' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__asset_growth_yoy_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Investment
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `asset_growth_yoy_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'asset_growth_yoy' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__revenue_growth_acceleration_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `revenue_growth_acceleration_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'revenue_growth_acceleration' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__net_income_growth_acceleration_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `net_income_growth_acceleration_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'net_income_growth_acceleration' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fund__operating_cash_flow_growth_acceleration_missing`

- **Role:** predictor
- **Feature group:** fundamental_missing
- **Family:** Growth
- **Source dataset:** `data/processed/features_fundamental_monthly.parquet`
- **Source column:** `operating_cash_flow_growth_acceleration_missing`
- **Model input:** `True`
- **Availability:** Derived solely from availability of the point-in-time raw fundamental factor.
- **Calculation:** Binary flag equal to 1 when 'operating_cash_flow_growth_acceleration' is unavailable and 0 otherwise.
- **Missing values:** The flag itself must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `first_future_date`

- **Role:** target_metadata
- **Feature group:** target_metadata
- **Family:** Target
- **Source dataset:** `data/processed/labels_monthly.parquet`
- **Source column:** `first_future_date`
- **Model input:** `False`
- **Availability:** Future information. Never permitted inside the predictor matrix.
- **Calculation:** First market session strictly after as_of_date.
- **Missing values:** Missing for inference-only observations.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 0.00%

### `target_end_date`

- **Role:** target_metadata
- **Feature group:** target_metadata
- **Family:** Target
- **Source dataset:** `data/processed/labels_monthly.parquet`
- **Source column:** `target_end_date`
- **Model input:** `False`
- **Availability:** Future information. Never permitted inside the predictor matrix.
- **Calculation:** Market date corresponding to the end of the configured forward horizon.
- **Missing values:** Missing when a complete future horizon is not yet available.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 0.00%

### `horizon_sessions`

- **Role:** target_metadata
- **Feature group:** target_metadata
- **Family:** Target
- **Source dataset:** `data/processed/labels_monthly.parquet`
- **Source column:** `horizon_sessions`
- **Model input:** `False`
- **Availability:** Target definition; does not enter the predictor matrix.
- **Calculation:** Configured number of future market sessions used by the target.
- **Missing values:** Missing for inference-only observations.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 0.00%

### `target_21d`

- **Role:** target
- **Feature group:** target
- **Family:** Target
- **Source dataset:** `data/processed/labels_monthly.parquet`
- **Source column:** `target_21d`
- **Model input:** `False`
- **Availability:** Known only after the complete future 21-session horizon has occurred.
- **Calculation:** Adjusted-close return from as_of_date to target_end_date.
- **Missing values:** Missing for inference-only observations.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 0.00%

### `target_21d_excess`

- **Role:** target
- **Feature group:** target
- **Family:** Target
- **Source dataset:** `data/processed/labels_monthly.parquet`
- **Source column:** `target_21d_excess`
- **Model input:** `False`
- **Availability:** Known only after the complete future 21-session horizon has occurred.
- **Calculation:** target_21d minus the cross-sectional median target_21d on the same as_of_date.
- **Missing values:** Missing for inference-only observations.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 0.00%

### `label_top_quintile`

- **Role:** target
- **Feature group:** target
- **Family:** Target
- **Source dataset:** `data/processed/labels_monthly.parquet`
- **Source column:** `label_top_quintile`
- **Model input:** `False`
- **Availability:** Known only after the complete future 21-session horizon has occurred.
- **Calculation:** Binary label equal to 1 for securities inside the top 20% of future returns on the same rebalance date.
- **Missing values:** Missing for inference-only observations.
- **Overall coverage:** 98.68%
- **Latest-date coverage:** 0.00%

### `technical_missing_count`

- **Role:** diagnostic
- **Feature group:** diagnostic
- **Family:** Data quality
- **Source dataset:** `data/processed/modeling_panel.parquet`
- **Source column:** `Derived`
- **Model input:** `False`
- **Availability:** Calculated from predictor availability at as_of_date.
- **Calculation:** Number of missing technical predictor values in the row.
- **Missing values:** Must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fundamental_global_missing_count`

- **Role:** diagnostic
- **Feature group:** diagnostic
- **Family:** Data quality
- **Source dataset:** `data/processed/modeling_panel.parquet`
- **Source column:** `Derived`
- **Model input:** `False`
- **Availability:** Calculated from predictor availability at as_of_date.
- **Calculation:** Number of missing global fundamental z-scores in the row.
- **Missing values:** Must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `fundamental_sector_missing_count`

- **Role:** diagnostic
- **Feature group:** diagnostic
- **Family:** Data quality
- **Source dataset:** `data/processed/modeling_panel.parquet`
- **Source column:** `Derived`
- **Model input:** `False`
- **Availability:** Calculated from predictor availability at as_of_date.
- **Calculation:** Number of missing sector-relative fundamental z-scores in the row.
- **Missing values:** Must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `model_feature_missing_count`

- **Role:** diagnostic
- **Feature group:** diagnostic
- **Family:** Data quality
- **Source dataset:** `data/processed/modeling_panel.parquet`
- **Source column:** `Derived`
- **Model input:** `False`
- **Availability:** Calculated from all candidate predictors available at as_of_date.
- **Calculation:** Total number of missing values among the candidate model features.
- **Missing values:** Must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `has_target`

- **Role:** diagnostic
- **Feature group:** diagnostic
- **Family:** Sample
- **Source dataset:** `data/processed/modeling_panel.parquet`
- **Source column:** `Derived`
- **Model input:** `False`
- **Availability:** Describes whether the full future target exists; never used as a predictor.
- **Calculation:** 1 when all target fields are available, otherwise 0.
- **Missing values:** Must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%

### `sample_role`

- **Role:** diagnostic
- **Feature group:** diagnostic
- **Family:** Sample
- **Source dataset:** `data/processed/modeling_panel.parquet`
- **Source column:** `Derived`
- **Model input:** `False`
- **Availability:** Describes sample eligibility; never used as a predictor.
- **Calculation:** 'modeling' when has_target=1 and 'inference_only' otherwise.
- **Missing values:** Must never be missing.
- **Overall coverage:** 100.00%
- **Latest-date coverage:** 100.00%
