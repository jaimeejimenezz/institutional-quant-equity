# Feature-Family Ablation Ensemble

- Scenario: `no_fundamentals`
- Predictor count: `19`
- Technical composite components used in validation: `8`
- Validation-weight rows: `77`
- Component-score rows: `3850`
- Final alpha rows: `3850`

## Methodology

- Uses the same `EnsembleConfig` and public ensemble functions as the frozen full model.
- Elastic Net and LightGBM validation IC values come from each ablation's own hyperparameter-selection run.
- For `no_momentum`, the two removed composite inputs are represented as missing only inside validation-weight construction. Because the frozen composite requires at least six available components and averages with `skipna=True`, this exactly reproduces the six surviving signed components without changing ensemble code.
- Final OOS scores are built from ablation-specific OOS component predictions and fold-specific validation weights.

## Validation-weight summary

             component     mean      std      min      max
      composite_weight 0.281453 0.102098 0.166667 0.540239
    elastic_net_weight 0.330377 0.073389 0.166667 0.536386
lightgbm_ranker_weight 0.388170 0.120463 0.166667 0.663894

## Readiness checks

                          check status  violations                                                      description
         validation_weight_rows   PASS           0 There must be exactly one validation-weight row per frozen fold.
      validation_weights_finite   PASS           0                             All ensemble weights must be finite.
  validation_weights_sum_to_one   PASS           0                  Fold-specific ensemble weights must sum to one.
 validation_weights_nonnegative   PASS           0             Fold-specific ensemble weights must be non-negative.
 component_scores_required_keys   PASS           0                       Output must contain as_of_date and ticker.
          component_scores_rows   PASS           0                               Output must contain 3850 OOS rows.
         component_scores_dates   PASS           0                                Output must contain 77 OOS dates.
component_scores_cross_sections   PASS           0                  Every OOS date must contain exactly 50 tickers.
   component_scores_unique_keys   PASS           0                  Output must be unique by as_of_date and ticker.
     final_signal_required_keys   PASS           0                       Output must contain as_of_date and ticker.
              final_signal_rows   PASS           0                               Output must contain 3850 OOS rows.
             final_signal_dates   PASS           0                                Output must contain 77 OOS dates.
    final_signal_cross_sections   PASS           0                  Every OOS date must contain exactly 50 tickers.
       final_signal_unique_keys   PASS           0                  Output must be unique by as_of_date and ticker.
