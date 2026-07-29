# Technical Feature Processing

## Purpose

The raw technical feature panel contains market signals expressed in their original units. These values cannot always be compared directly because features have different scales and may contain extreme observations.

The processing stage transforms every feature independently within each monthly rebalance date.

The final dataset is stored at:

`data/processed/features_technical_monthly.parquet`

## Processing sequence

For each feature and rebalance date:

1. Preserve the original raw value.
2. Winsorize the cross-section.
3. Calculate a cross-sectional z-score.
4. Optionally remove the mean score of each sector.

No information from a future rebalance date is used.

## Winsorization

The initial configuration uses the 1st and 99th percentiles.

Values below the lower percentile are replaced by the lower boundary. Values above the upper percentile are replaced by the upper boundary.

Winsorization reduces the influence of extreme observations without deleting companies.

It is calculated separately for every feature and date.

## Cross-sectional standardization

After winsorization, each feature is transformed into a z-score:

`z = (x - cross_sectional_mean) / cross_sectional_standard_deviation`

A positive z-score means that the company has a value above the universe average for that feature on the same date.

A negative z-score means that it is below the universe average.

The transformation does not imply that a high value is economically desirable. For example, high momentum may be interpreted positively, while high volatility or high illiquidity may be interpreted negatively. Signal direction will be evaluated during factor research.

## Sector neutralization

For sector groups containing at least two valid companies, the mean sector z-score is subtracted:

`sector_neutral_score = zscore - sector_mean_zscore`

This reduces the possibility that a factor is merely selecting a sector instead of identifying differences between companies.

If a sector contains only one valid company on a date, internal sector comparison is impossible. In that case, the global z-score is retained and the situation is reported as a warning.

## Missing values

Missing raw values remain missing throughout the processing stage.

No backward filling, forward filling, global mean imputation or future-aware imputation is performed.

Any later imputation required by a predictive model must be fitted only with training data inside the temporal validation pipeline.

## Stored representations

For every raw technical feature, the final dataset stores:

* The original raw value.
* The winsorized value.
* The cross-sectional z-score.
* The sector-neutral score.

This structure preserves auditability while providing model-ready representations.

## Temporal safety

All transformations are performed using companies observed on the same `as_of_date`.

No future month contributes to:

* Percentile boundaries.
* Cross-sectional means.
* Cross-sectional standard deviations.
* Sector means.

Therefore, the processing stage does not introduce temporal leakage.
