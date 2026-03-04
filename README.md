# 📈 ML Asset Pricing — S&P 500 (2000–2026)

A full replication and extension of **Gu, Kelly & Xiu (2020)** — *"Empirical Asset Pricing via Machine Learning"* — applied to S&P 500 stocks over a 26-year horizon. Covers the complete pipeline from data collection to portfolio backtesting.

---

## Data

Pre-parsed data (prices + macro, 1990–2026) is available here:

>  **[Google Drive — Parsed Dataset](https://drive.google.com/drive/folders/1W45UnZAWATJ302yeNce4NeXjQbpyUQEa?usp=sharing)**

| File | Description |
|------|-------------|
| `stocks_adj_close_1990_2026.csv` | Adjusted closing prices for ~500 S&P 500 constituents |
| `macro_data_1990_2026.csv` | Macro features: VIX, yield curve, risk-free rate, credit spread, etc. |

---

## Notebook Structure

### Part 1 — Data Collection
- Dynamic S&P 500 ticker list (Wikipedia scraper + hardcoded fallback, ~503 tickers)
- Batch download via `yfinance` with retry logic
- Macro indicators: VIX, S&P 500, 13W/5Y/10Y Treasuries via Yahoo Finance; BAA–AAA credit spread via FRED
- Derived features: term spread, log market return, VIX change, daily risk-free rate

### Part 2 — Feature Engineering
- **Momentum**: cumulative log returns over 5, 10, 21, 63, 252 days
- **Volatility**: realized std over 5, 21, 63 days
- **Price ratios**: price-to-MA (10/21/50/200d), distance from 52-week high/low
- **Reversal**: lagged returns (lags 1–5)
- **Cross-sectional ranks**: percentile ranks of key features within each date
- Final panel: ~500 stocks × ~6500 days ≈ **3M+ observations**

### Part 3 — Model Horse Race

**Classical models** (sklearn / XGBoost / CatBoost):

| Model | Notes |
|-------|-------|
| OLS | Baseline linear |
| Ridge / LASSO / ElasticNet | Regularized linear |
| Random Forest / Extra Trees | Bagged trees, `min_samples_leaf=200` |
| XGBoost | GPU-accelerated |
| CatBoost | Bernoulli bootstrap |
| NN1 / NN3 | 1- and 3-layer MLPs via `skorch` |

**Deep learning models** (pure PyTorch, GPU + AMP):

| Model | Reference |
|-------|-----------|
| **Transformer** | Kelly, Kuznetsov, Malamud & Xu (2025), NBER WP 33351 |
| **CNN** | Jiang, Kelly & Xiu (2023), *Journal of Finance* |
| **GNN** | Korangi et al. (2024); Uddin et al. (2023) — correlation-based adjacency |

All models evaluated on:
- **OOS R²** (Gu et al. definition: `1 − SS_res / SS_tot`)
- **RMSE**
- **Annualized Sharpe ratio** (long-short quintile portfolio)

Temporal splits: **Train** 2000–2019 · **Val** 2020–2022 · **Test** 2023–2026

Both **daily** (`ret_1d_fwd`) and **weekly** (`ret_5d_fwd`) targets are evaluated.

### Part 4 — Ridge + Polynomial Interactions (AIPT-style)
- Degree-2 polynomial feature expansion
- Grid search over `alpha ∈ [1e-2, 1e6]` on validation OOS R²
- "Virtue of complexity" curve: OOS R² vs feature dimension P
- Comparison of standard Ridge vs Ridge+Poly cumulative L/S returns

### Part 5 — Economic Interpretation
- Feature importance (tree-based models)
- SDF portfolio analysis

---

## Requirements

```bash
pip install yfinance pandas numpy scikit-learn xgboost catboost torch skorch tqdm matplotlib seaborn requests
```

GPU training is automatic — falls back to CPU if CUDA is unavailable.

---

## Quickstart

```python
# Skip Part 1 if using pre-parsed data from Google Drive
# Place CSVs in the working directory, then run Parts 2–5
period_tag = '1990_2026'
prices = pd.read_csv(f'stocks_adj_close_{period_tag}.csv', index_col=0, parse_dates=True)
macro  = pd.read_csv(f'macro_data_{period_tag}.csv',       index_col=0, parse_dates=True)
```

---

## References

- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Kelly, B., Kuznetsov, I., Malamud, S., & Xu, T. (2025). Artificial Intelligence Asset Pricing Models. *NBER WP 33351*.
- Jiang, J., Kelly, B., & Xiu, D. (2023). (Re-)Imag(in)ing Price Trends. *Journal of Finance*, 78(6), 3193–3249.
- Korangi, K., Mues, C., & Bravo, C. (2024). Large-scale Portfolio Optimisation using Graph Attention Networks. *arXiv:2407.15532*.
