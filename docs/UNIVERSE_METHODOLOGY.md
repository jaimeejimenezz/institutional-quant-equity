# Equity Universe Methodology

## Purpose

The equity universe defines the companies that may be evaluated,
ranked and selected by the investment strategy.

Version `v1` contains 50 large and liquid US-listed companies.

The universe is deliberately limited during the MVP so that the
complete research pipeline can be implemented and audited before
expanding to 100-150 companies.

## Selection criteria

Companies were selected according to the following qualitative
criteria:

1. Large US-listed operating company.
2. High historical trading liquidity.
3. Sufficient price history for research beginning in 2014.
4. Availability of an SEC Central Index Key.
5. Representation of the eleven main economic sectors.
6. Expected availability of market and fundamental information.
7. Avoidance of highly obscure or illiquid securities.

The universe is a research sample. It is not intended to replicate
the exact holdings or weights of a commercial equity index.

## Universe schema

| Column | Description |
|---|---|
| `ticker` | Trading ticker used by the market-data provider. |
| `company_name` | Human-readable company name. |
| `sector` | Broad economic sector. |
| `industry` | More specific business classification. |
| `cik` | Ten-digit SEC Central Index Key. |
| `start_date` | First assumed eligibility date in the MVP. |
| `end_date` | Final eligibility date when applicable. |
| `is_active` | Whether the security remains active in this universe version. |
| `inclusion_source` | Source or method used to select the company. |

## Meaning of start_date

In version `v1`, `start_date` is the date from which the company is
assumed to be eligible for the research universe.

It is not evidence that the company belonged to a particular index
on that date.

All version-one companies are assigned an eligibility date of
2014-01-01 so that the MVP can use a stable cross-sectional sample.

## Survivorship bias

Version `v1` is constructed using companies selected with information
available in 2026 and applies that list retrospectively from 2014.

This introduces survivorship bias.

Companies that failed, were acquired, were delisted or became
financially distressed before the universe was created are not
adequately represented.

As a result, historical performance obtained with this universe may
be overstated relative to a genuinely point-in-time investable
universe.

This limitation must be disclosed in every research report or
backtest based on `universe_v1.csv`.

The results must not be described as free from survivorship bias or
as a fully institutional point-in-time simulation.

## Other limitations

The first universe version may also contain:

- Historical ticker changes.
- Corporate reorganizations.
- Mergers and spin-offs.
- Changes in sector classifications.
- Differences between current and historical company structures.
- Unequal sector representation.
- Selection bias caused by manual inclusion criteria.

These issues will be reviewed during market-data and fundamental-data
validation.

## Future universe versions

A future `v2` should consider:

1. Historical index constituent membership.
2. Delisted companies.
3. Historical ticker mappings.
4. Effective inclusion and exclusion dates.
5. Historical sector classifications.
6. Explicit liquidity and market-capitalization screens.
7. Point-in-time company eligibility.
8. A larger universe of approximately 100-150 companies.

Universe files are versioned and must not be silently overwritten.

Any material change to membership requires a new file such as
`universe_v2.csv`.