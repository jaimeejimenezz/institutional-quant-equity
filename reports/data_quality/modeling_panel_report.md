# Modeling Panel Quality Report

## Status

**PASS**

## Summary

```text
                    metric  value
                panel_rows   7550
               panel_dates    151
             panel_tickers     50
             panel_sectors     11
  candidate_model_features     91
             modeling_rows   7450
       inference_only_rows    100
minimum_cross_section_size     50
maximum_cross_section_size     50
                  ttm_rows  42766
               ttm_metrics      7
              audit_checks     21
             failed_checks      0
```

## Leakage and point-in-time checks

```text
                           check status  violations                                                                 description
               unique_panel_keys   PASS           0                Every modeling row must have a unique as_of_date-ticker key.
         valid_panel_identifiers   PASS           0                                   Dates, tickers and sectors must be valid.
         technical_point_in_time   PASS           0             Technical market information must not extend beyond as_of_date.
               binary_has_target   PASS           0                                        has_target must contain only 0 or 1.
          complete_target_tuples   PASS           0   Modeling rows need complete targets and inference rows must contain none.
         sample_role_consistency   PASS           0                            sample_role must agree with target availability.
       target_starts_after_as_of   PASS           0                            The target must begin strictly after as_of_date.
                valid_target_end   PASS           0                           The target end must not precede the target start.
                  target_horizon   PASS           0           Every modeling target must use the configured 21-session horizon.
cross_section_target_consistency   PASS           0 All companies on the same rebalance date must share target timing metadata.
           unique_calendar_dates   PASS           0                 The rebalance calendar must contain one row per as_of_date.
         calendar_date_alignment   PASS           0                      Every panel date must exist in the rebalance calendar.
    calendar_target_availability   PASS           0                Panel target availability must match the rebalance calendar.
        calendar_target_metadata   PASS           0          Stored modeling targets must preserve the calendar timing exactly.
       feature_target_separation   PASS           0       Predictor columns must not contain targets or future-return metadata.
           finite_model_features   PASS           0             Model features may be missing, but they must never be infinite.
            unique_ttm_snapshots   PASS           0                    TTM snapshots must be unique by date, ticker and metric.
               ttm_point_in_time   PASS           0                  All TTM components must have been available by as_of_date.
                  ttm_period_end   PASS           0                The latest accounting quarter must not end after as_of_date.
               ttm_four_quarters   PASS           0                   Every TTM observation must contain exactly four quarters.
             ttm_panel_alignment   PASS           0         Every TTM date-ticker key must belong to the master modeling panel.
```

## Blocking issues

- None.

## Modeling boundary

The master panel does not perform training-sample imputation, fitted scaling, PCA, model-based feature selection or hyperparameter optimization.

Any transformation that learns parameters from observations must be fitted inside the training portion of each walk-forward fold.

Cross-sectional technical and fundamental scores are date-local transformations and do not use observations from future dates.