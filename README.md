# Institutional Quant Equity Research Platform

An end-to-end quantitative equity research platform for building,
evaluating and explaining systematic equity portfolios.

## Project objective

The project studies whether market, fundamental and machine-learning
signals can identify US equities with superior future relative returns.

The platform will cover the complete research process:

1. Market and fundamental data ingestion.
2. Point-in-time data validation.
3. Feature engineering.
4. Cross-sectional return prediction.
5. Walk-forward validation.
6. Risk-aware portfolio construction.
7. Transaction-cost-aware backtesting.
8. Performance and risk reporting.
9. Interactive dashboard.

## Initial research design

- Market: United States.
- Initial universe: approximately 50 liquid large-cap equities.
- Rebalancing: monthly.
- Prediction horizon: 21 trading sessions.
- Portfolio: long-only.
- Target positions: approximately 20.
- Maximum weight per asset: 5%.
- Maximum weight per sector: 25%.
- Main benchmark: SPY.
- Main transaction-cost assumption: 10 basis points per unit of turnover.

## Repository structure

```text
config/       Project configuration.
data/         Raw, interim, processed and reference data.
docs/         Methodological documentation.
notebooks/    Exploratory research notebooks.
reports/      Figures, tables and research reports.
scripts/      Executable pipeline entry points.
src/          Reusable Python source code.
tests/        Automated tests.
app/          Streamlit dashboard.