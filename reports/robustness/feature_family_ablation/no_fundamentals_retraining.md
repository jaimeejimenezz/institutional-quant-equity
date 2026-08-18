# Feature-Family Ablation Retraining

- Scenario: `no_fundamentals`
- Predictor count: `19`
- Technical composite components: `8`
- OOS dates: `77`
- OOS companies: `50`
- Elapsed time: `401.2 seconds (6.7 minutes)`

## Methodology

- Uses the frozen feature-family contract and the original 77 walk-forward folds.
- Elastic Net and LightGBM Ranker retain their original candidate hyperparameter grids.
- Hyperparameters continue to be selected only inside the validation interval of each fold.
- The no-momentum technical composite removes the frozen momentum inputs and retains the signed equal-weight construction over the six surviving components.

## OOS prediction rows

                      model_name  prediction_rows
                     elastic_net             3850
                 lightgbm_ranker             3850
technical_equal_weight_composite             3850

## Readiness checks

                        check status  violations                                                                                      description
       expected_feature_count   PASS           0                                                  no_fundamentals must use exactly 19 predictors.
expected_composite_components   PASS           0                             The technical composite must use the frozen surviving component set.
              expected_models   PASS           0 Ablation ensemble predictions must contain technical composite, Elastic Net and LightGBM Ranker.
           composite_oos_rows   PASS           0                                           Technical composite must contain 3850 OOS predictions.
         elastic_net_oos_rows   PASS           0                                                   Elastic Net must contain 3850 OOS predictions.
     lightgbm_ranker_oos_rows   PASS           0                                               LightGBM Ranker must contain 3850 OOS predictions.
        oos_dates_match_folds   PASS           0                          Prediction dates must match the frozen walk-forward test dates exactly.
       unique_prediction_keys   PASS           0                                            Predictions must be unique by fold, ticker and model.
           finite_predictions   PASS           0                                                       All stored OOS predictions must be finite.
