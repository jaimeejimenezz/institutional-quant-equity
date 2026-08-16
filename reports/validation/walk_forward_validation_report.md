# Definitive Walk-Forward Validation

## Step

Step 12D — Final documentation and approval of the definitive walk-forward protocol.

## Status

**READY FOR MODEL TRAINING**

## Configuration

```text
mode: expanding
minimum training dates: 60
validation dates: 12
test window: 1 monthly cross-section
```

## Summary

```text
                     metric               value
                      folds                  77
                       mode           expanding
        minimum_train_dates                  60
           validation_dates                  12
            first_test_date 2020-01-31 00:00:00
             last_test_date 2026-05-29 00:00:00
             oos_test_dates                  77
expected_cross_section_size                  50
         candidate_features                  91
    minimum_active_features                  91
    maximum_active_features                  91
       maximum_purged_dates                   1
           readiness_checks                  19
    failed_readiness_checks                   0
```

## Temporal protocol

- Every security from the same as_of_date remains in the same partition.
- Training always precedes validation and validation always precedes test.
- The test set contains exactly one historical monthly cross-section.
- Historical observations whose forward 21-session target had not matured by the test date are purged from fitting.
- Every historical OOS month from the first test date onward is evaluated exactly once.

## Preprocessing protocol

- Feature availability is determined using training data only.
- Missing-value imputation parameters are estimated using training data only.
- Scaling parameters are estimated using training data only.
- Validation and test data are transformed without refitting preprocessing parameters.
- Missing-indicator features retain their binary 0/1 interpretation.

## Hyperparameter rule

Hyperparameters and model-selection decisions must use training and validation information only. The monthly test cross-section must remain untouched until final OOS prediction.

## Final readiness checks

```text
                         check status  violations                                                                                           description
                   folds_exist   PASS           0                                                 At least one definitive walk-forward fold must exist.
           sequential_fold_ids   PASS           0                                                        Fold IDs must be sequential and deterministic.
             unique_test_dates   PASS           0                                                Every out-of-sample month must be tested exactly once.
strictly_increasing_test_dates   PASS           0                                          Out-of-sample test dates must advance strictly through time.
    complete_oos_date_coverage   PASS           0                         Every completed modeling month from the first OOS date onward must be tested.
      disjoint_fold_partitions   PASS           0                         Train, validation and test dates must be mutually exclusive inside each fold.
 chronological_partition_order   PASS           0                                                    Every fold must satisfy train < validation < test.
      minimum_training_history   PASS           0                                      Every fold must satisfy the configured minimum training history.
        validation_window_size   PASS           0                                    Every fold must contain the configured number of validation dates.
  complete_test_cross_sections   PASS           0                                          Every OOS month must contain the complete security universe.
 historical_test_targets_exist   PASS           0                                  Historical OOS test rows must have completed targets for evaluation.
        fitting_label_maturity   PASS           0                                    Train and validation targets must be fully known by the test date.
    expanding_training_history   PASS           0                                    Expanding mode must preserve previously eligible training history.
stored_fold_metadata_alignment   PASS           0                    Persisted Step 12A fold metadata must match the regenerated walk-forward contract.
  preprocessing_fold_alignment   PASS           0                              Step 12B preprocessing results must align one-to-one with the OOS folds.
preprocessing_feature_contract   PASS           0                  Every fold must account for all candidate features using training-only availability.
no_missing_after_preprocessing   PASS           0                                No active predictor may remain missing after fold-local preprocessing.
  finite_preprocessed_features   PASS           0                                 Preprocessed features must not contain positive or negative infinity.
     training_scaling_contract   PASS           0 Non-constant continuous training features must be centered and scaled using training-only statistics.
```

## Blocking issues

- None.

## Artifacts

- `data/processed/modeling_panel.parquet`
- `data/processed/walk_forward_folds.parquet`
- `data/processed/walk_forward_preprocessing.parquet`
- `reports/tables/walk_forward_preprocessing_audit.csv`
- `reports/tables/walk_forward_readiness_checks.csv`

## Step 12 conclusion

The definitive validation framework is approved for out-of-sample model training.

All models evaluated from Step 13 onward must consume this temporal protocol rather than creating independent train/test splits.