# Replication package: Does Complexity Pay? A Pseudo Real-Time Evaluation of Econometric, Machine Learning and Hybrid Models for Forecasting Turkish Inflation

This repository contains the data and code needed to reproduce every number, table and figure in the paper.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

Approximate run time on a standard laptop CPU: 1.5–2.5 hours, most of it for the deep learning models (step 3). Results are deterministic given the fixed random seeds; minor floating-point differences may occur across hardware and library versions.

## Data (`data/`)

Monthly data, 2012:M1–2025:M8, retrieved in November 2025 from the Turkish Statistical Institute (TUIK) and the Central Bank of the Republic of Turkey (CBRT) Electronic Data Delivery System (EVDS). Appendix A of the paper lists every variable, its source and its publication lag.

| File | Contents |
|---|---|
| `tufe_yillik_aylik.txt` | Annual (target) and monthly CPI inflation |
| `ufe.txt` | Producer price index and sub-indices |
| `sanayi_uretim.txt` | Industrial production index and sub-indices |
| `tuketici_guven_endeksi.txt` | Consumer confidence index and sub-components |
| `faiz_gecelik.txt` | CBRT overnight borrowing and lending rates (decision dates) |
| `all.txt` | BIST 100 (`bist`), Brent oil price in USD (`usd`), short-term external debt (`dis_borc`), unemployment rates (`isgucu_*`), house price index (`konut_fiyat_endeksi`, unavailable for 2024:M6–2025:M8), consumer and housing loans (`kredi_*`), CBRT indicative exchange rates (`*_alis`, `*_satis`, monthly averages; JPY per 100), real effective exchange rates (`usd_tufe`: CPI-based, `usd_ufe`: PPI-based) |

## Design (see Section 4 of the paper)

- **Forecast origin:** immediately after the release of CPI for month t−1. Inputs are the last 12 months of the 21 predictors most correlated with the change in inflation, selected within each training sample.
- **Publication lags:** t−1 for prices, exchange rates, interest rates, credit, equity prices and consumer confidence; t−2 for unemployment and the house price index; t−3 for industrial production and short-term external debt (`pipeline.py`).
- **Expanding window:** predictor selection, scaling and estimation are repeated at every forecast origin. Deep learning models are re-trained every 12 months (5 seeds, averaged). ML hyperparameters are re-selected every 12 months on the last 12 training observations.
- **Evaluation windows:** last 12, 36 and 60 months of the sample (one-step-ahead forecasts in all cases).

## Code

| Script | Purpose |
|---|---|
| `pipeline.py` | Data preparation, publication lags; random walk, AR, ARIMA, Holt, VAR, ML (default settings) and ARIMA–XGBoost; evaluation functions (DM with HLN correction, MCS) |
| `tuned.py` | ML models with rolling hyperparameter selection |
| `dl.py`, `dl_chunk.py` | LSTM, GRU, CNN, CNN–LSTM and N-BEATS (one 12-month block per call) |
| `merge.py` | Combines all forecasts; seed variability of DL models |
| `analysis.py` | Final metrics, sub-periods, descriptive statistics, permutation importance |
| `extra_metrics.py` | MASE and directional accuracy |
| `select_params.py`, `make_tables_extra.py` | Selected hyperparameters (Appendix E) and appendix tables |
| `make_outputs.py` | Tables 2–3 and Figures 1–4 |

## Outputs

`results/` contains all forecasts (`final_forecasts_*.csv`) and metrics; `paper/` contains the LaTeX tables and PDF figures used in the manuscript.
