# Feature-Family Ablation Contract

## Purpose

This document freezes the predictor families used by the feature ablation experiments before any ablation model is trained.
The classification changes no prediction, portfolio or backtest result.

## Rules

- Technical predictors are identified from the canonical `TECHNICAL_MODEL_FEATURE_COLUMNS` contract created by the technical processing pipeline.
- Fundamental predictors are the remaining frozen model predictors after the complete technical contract is mapped.
- Momentum predictors are frozen inside the technical family as return, momentum and reversal roots: return_1w, return_1m, return_3m, momentum_6_1, momentum_12_1, reversal_1m.
- `no_fundamentals` removes all fundamental predictors and retains every technical predictor.
- `no_momentum` removes only the explicitly frozen momentum predictors and retains fundamentals and other technical predictors.
- These families must not be changed after viewing the ablation results.

## Family counts

| family | feature_count |
| --- | --- |
| fundamental | 72 |
| technical_momentum | 6 |
| technical_other | 13 |

## Fundamental predictors removed

| feature |
| --- |
| fund__earnings_yield_zscore |
| fund__sales_yield_zscore |
| fund__book_to_market_zscore |
| fund__fcf_yield_zscore |
| fund__roe_zscore |
| fund__roa_zscore |
| fund__gross_profitability_zscore |
| fund__gross_margin_zscore |
| fund__operating_margin_zscore |
| fund__net_margin_zscore |
| fund__cash_conversion_zscore |
| fund__debt_to_assets_zscore |
| fund__net_debt_to_assets_zscore |
| fund__current_ratio_zscore |
| fund__interest_coverage_zscore |
| fund__capex_to_assets_zscore |
| fund__accruals_zscore |
| fund__revenue_growth_yoy_zscore |
| fund__net_income_growth_yoy_zscore |
| fund__operating_cash_flow_growth_yoy_zscore |
| fund__asset_growth_yoy_zscore |
| fund__revenue_growth_acceleration_zscore |
| fund__net_income_growth_acceleration_zscore |
| fund__operating_cash_flow_growth_acceleration_zscore |
| fund__earnings_yield_sector_zscore |
| fund__sales_yield_sector_zscore |
| fund__book_to_market_sector_zscore |
| fund__fcf_yield_sector_zscore |
| fund__roe_sector_zscore |
| fund__roa_sector_zscore |
| fund__gross_profitability_sector_zscore |
| fund__gross_margin_sector_zscore |
| fund__operating_margin_sector_zscore |
| fund__net_margin_sector_zscore |
| fund__cash_conversion_sector_zscore |
| fund__debt_to_assets_sector_zscore |
| fund__net_debt_to_assets_sector_zscore |
| fund__current_ratio_sector_zscore |
| fund__interest_coverage_sector_zscore |
| fund__capex_to_assets_sector_zscore |
| fund__accruals_sector_zscore |
| fund__revenue_growth_yoy_sector_zscore |
| fund__net_income_growth_yoy_sector_zscore |
| fund__operating_cash_flow_growth_yoy_sector_zscore |
| fund__asset_growth_yoy_sector_zscore |
| fund__revenue_growth_acceleration_sector_zscore |
| fund__net_income_growth_acceleration_sector_zscore |
| fund__operating_cash_flow_growth_acceleration_sector_zscore |
| fund__earnings_yield_missing |
| fund__sales_yield_missing |
| fund__book_to_market_missing |
| fund__fcf_yield_missing |
| fund__roe_missing |
| fund__roa_missing |
| fund__gross_profitability_missing |
| fund__gross_margin_missing |
| fund__operating_margin_missing |
| fund__net_margin_missing |
| fund__cash_conversion_missing |
| fund__debt_to_assets_missing |
| fund__net_debt_to_assets_missing |
| fund__current_ratio_missing |
| fund__interest_coverage_missing |
| fund__capex_to_assets_missing |
| fund__accruals_missing |
| fund__revenue_growth_yoy_missing |
| fund__net_income_growth_yoy_missing |
| fund__operating_cash_flow_growth_yoy_missing |
| fund__asset_growth_yoy_missing |
| fund__revenue_growth_acceleration_missing |
| fund__net_income_growth_acceleration_missing |
| fund__operating_cash_flow_growth_acceleration_missing |

## Momentum predictors removed

| feature |
| --- |
| tech__momentum_12_1_sector_neutral |
| tech__momentum_6_1_sector_neutral |
| tech__return_3m_sector_neutral |
| tech__return_1m_sector_neutral |
| tech__return_1w_sector_neutral |
| tech__reversal_1m_sector_neutral |

## Full contract

| feature | canonical_technical_feature | family | is_fundamental | is_technical | is_momentum | included_full_model | included_no_fundamentals | included_no_momentum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tech__momentum_12_1_sector_neutral | momentum_12_1_sector_neutral | technical_momentum | False | True | True | True | True | False |
| tech__momentum_6_1_sector_neutral | momentum_6_1_sector_neutral | technical_momentum | False | True | True | True | True | False |
| tech__return_3m_sector_neutral | return_3m_sector_neutral | technical_momentum | False | True | True | True | True | False |
| tech__return_1m_sector_neutral | return_1m_sector_neutral | technical_momentum | False | True | True | True | True | False |
| tech__return_1w_sector_neutral | return_1w_sector_neutral | technical_momentum | False | True | True | True | True | False |
| tech__reversal_1m_sector_neutral | reversal_1m_sector_neutral | technical_momentum | False | True | True | True | True | False |
| tech__volatility_20d_sector_neutral | volatility_20d_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__volatility_60d_sector_neutral | volatility_60d_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__downside_volatility_60d_sector_neutral | downside_volatility_60d_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__beta_60d_market_sector_neutral | beta_60d_market_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__max_drawdown_126d_sector_neutral | max_drawdown_126d_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__distance_sma_50d_sector_neutral | distance_sma_50d_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__distance_sma_200d_sector_neutral | distance_sma_200d_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__sma_50_200_spread_sector_neutral | sma_50_200_spread_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__positive_day_ratio_60d_sector_neutral | positive_day_ratio_60d_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__average_dollar_volume_20d_sector_neutral | average_dollar_volume_20d_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__dollar_volume_change_20d_60d_sector_neutral | dollar_volume_change_20d_60d_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__amihud_illiquidity_20d_sector_neutral | amihud_illiquidity_20d_sector_neutral | technical_other | False | True | False | True | True | True |
| tech__zero_volume_ratio_60d_sector_neutral | zero_volume_ratio_60d_sector_neutral | technical_other | False | True | False | True | True | True |
| fund__earnings_yield_zscore |  | fundamental | True | False | False | True | False | True |
| fund__sales_yield_zscore |  | fundamental | True | False | False | True | False | True |
| fund__book_to_market_zscore |  | fundamental | True | False | False | True | False | True |
| fund__fcf_yield_zscore |  | fundamental | True | False | False | True | False | True |
| fund__roe_zscore |  | fundamental | True | False | False | True | False | True |
| fund__roa_zscore |  | fundamental | True | False | False | True | False | True |
| fund__gross_profitability_zscore |  | fundamental | True | False | False | True | False | True |
| fund__gross_margin_zscore |  | fundamental | True | False | False | True | False | True |
| fund__operating_margin_zscore |  | fundamental | True | False | False | True | False | True |
| fund__net_margin_zscore |  | fundamental | True | False | False | True | False | True |
| fund__cash_conversion_zscore |  | fundamental | True | False | False | True | False | True |
| fund__debt_to_assets_zscore |  | fundamental | True | False | False | True | False | True |
| fund__net_debt_to_assets_zscore |  | fundamental | True | False | False | True | False | True |
| fund__current_ratio_zscore |  | fundamental | True | False | False | True | False | True |
| fund__interest_coverage_zscore |  | fundamental | True | False | False | True | False | True |
| fund__capex_to_assets_zscore |  | fundamental | True | False | False | True | False | True |
| fund__accruals_zscore |  | fundamental | True | False | False | True | False | True |
| fund__revenue_growth_yoy_zscore |  | fundamental | True | False | False | True | False | True |
| fund__net_income_growth_yoy_zscore |  | fundamental | True | False | False | True | False | True |
| fund__operating_cash_flow_growth_yoy_zscore |  | fundamental | True | False | False | True | False | True |
| fund__asset_growth_yoy_zscore |  | fundamental | True | False | False | True | False | True |
| fund__revenue_growth_acceleration_zscore |  | fundamental | True | False | False | True | False | True |
| fund__net_income_growth_acceleration_zscore |  | fundamental | True | False | False | True | False | True |
| fund__operating_cash_flow_growth_acceleration_zscore |  | fundamental | True | False | False | True | False | True |
| fund__earnings_yield_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__sales_yield_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__book_to_market_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__fcf_yield_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__roe_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__roa_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__gross_profitability_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__gross_margin_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__operating_margin_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__net_margin_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__cash_conversion_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__debt_to_assets_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__net_debt_to_assets_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__current_ratio_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__interest_coverage_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__capex_to_assets_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__accruals_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__revenue_growth_yoy_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__net_income_growth_yoy_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__operating_cash_flow_growth_yoy_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__asset_growth_yoy_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__revenue_growth_acceleration_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__net_income_growth_acceleration_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__operating_cash_flow_growth_acceleration_sector_zscore |  | fundamental | True | False | False | True | False | True |
| fund__earnings_yield_missing |  | fundamental | True | False | False | True | False | True |
| fund__sales_yield_missing |  | fundamental | True | False | False | True | False | True |
| fund__book_to_market_missing |  | fundamental | True | False | False | True | False | True |
| fund__fcf_yield_missing |  | fundamental | True | False | False | True | False | True |
| fund__roe_missing |  | fundamental | True | False | False | True | False | True |
| fund__roa_missing |  | fundamental | True | False | False | True | False | True |
| fund__gross_profitability_missing |  | fundamental | True | False | False | True | False | True |
| fund__gross_margin_missing |  | fundamental | True | False | False | True | False | True |
| fund__operating_margin_missing |  | fundamental | True | False | False | True | False | True |
| fund__net_margin_missing |  | fundamental | True | False | False | True | False | True |
| fund__cash_conversion_missing |  | fundamental | True | False | False | True | False | True |
| fund__debt_to_assets_missing |  | fundamental | True | False | False | True | False | True |
| fund__net_debt_to_assets_missing |  | fundamental | True | False | False | True | False | True |
| fund__current_ratio_missing |  | fundamental | True | False | False | True | False | True |
| fund__interest_coverage_missing |  | fundamental | True | False | False | True | False | True |
| fund__capex_to_assets_missing |  | fundamental | True | False | False | True | False | True |
| fund__accruals_missing |  | fundamental | True | False | False | True | False | True |
| fund__revenue_growth_yoy_missing |  | fundamental | True | False | False | True | False | True |
| fund__net_income_growth_yoy_missing |  | fundamental | True | False | False | True | False | True |
| fund__operating_cash_flow_growth_yoy_missing |  | fundamental | True | False | False | True | False | True |
| fund__asset_growth_yoy_missing |  | fundamental | True | False | False | True | False | True |
| fund__revenue_growth_acceleration_missing |  | fundamental | True | False | False | True | False | True |
| fund__net_income_growth_acceleration_missing |  | fundamental | True | False | False | True | False | True |
| fund__operating_cash_flow_growth_acceleration_missing |  | fundamental | True | False | False | True | False | True |

## Readiness checks

| check | status | violations | description |
| --- | --- | --- | --- |
| expected_model_feature_count | PASS | 0 | The frozen modeling contract should contain 91 predictors. |
| unique_model_feature_names | PASS | 0 | Frozen model feature names must be unique. |
| all_model_features_in_panel | PASS | 0 | Every frozen model feature must exist in modeling_panel.parquet. |
| technical_contract_complete | PASS | 0 | Every canonical technical model feature must map to exactly one frozen modeling-panel predictor. |
| expected_technical_feature_count | PASS | 0 | The technical family must match the canonical technical-model contract. |
| fundamental_family_nonempty | PASS | 0 | The no-fundamentals ablation requires fundamental predictors. |
| momentum_family_nonempty | PASS | 0 | The no-momentum ablation requires explicit momentum predictors. |
| families_partition_model_features | PASS | 0 | Technical and fundamental families must partition the model predictors. |
| no_fundamentals_retains_only_technical | PASS | 0 | The no-fundamentals contract must retain only technical predictors. |
| no_momentum_removes_all_momentum | PASS | 0 | The no-momentum contract must remove every frozen momentum predictor. |
| ablations_retain_predictors | PASS | 0 | Each ablation must retain a usable predictor set. |
