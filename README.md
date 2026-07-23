# StockPredictionAI Pro (2025 Edition)

**A complete GAN-based stock price prediction system built with PyTorch.**  
Based on the ideas of [Boris Banushev](https://github.com/borisbanushev/stockpredictionai) — rebuilt from scratch and extended with a Transformer generator, quantile regression, automatic hyperparameter tuning, and comprehensive feature engineering.

---

## Features

### Data & Feature Engineering
- **40+ correlated assets** via Yahoo Finance: peers (JPM, BAC, MS, C, WFC...), indices (S&P 500, DJIA, NASDAQ, Russell 2000, FTSE, Nikkei, Hang Seng, DAX...), currencies (EUR/USD, GBP/USD, USD/JPY...), commodities (gold, silver, oil, gas), volatility (VIX), bonds (TNX, TLT)
- **13 technical indicators**: SMA(7/21), EMA(21), MACD + Signal + Histogram, Bollinger Bands (upper/mid/lower), RSI(14), Momentum(10), Log Momentum(10)
- **Multi-Fourier** (k=3, 6, 9): trend decomposition into long-, medium-, and short-term components
- **ARIMA(5,1,0)**: in-sample approximation as an extra feature
- **Stacked Autoencoder** (GELU) or **VAE** (variational, KL term; `ae_variant="vae"`): non-linear latent features
- **Eigen-portfolio (PCA)**: linear latent features
- **FinBERT sentiment** (opt-in): news sentiment analysis
- **XGBoost Feature Importance**: feature ranking

### Models (WGAN-GP)
- **Generators**: LSTM (with dropout) or Transformer (TST) — switchable with a single flag
- **Discriminator**: 3-layer Conv1D (32→64→128) with BatchNorm, FC(220→220→1)
- **Multi-task output**: ΔP regression, direction classification, quantile regression (q10/q50/q90)
- **Gradient Penalty** (WGAN-GP): stable training
- **MHGAN inference** (opt-in, `--mhgan`): Metropolis-Hastings sampling — the trained critic filters k stochastic generator samples per input (Uber MHGAN)

### Training & Optimization
- **LR Scheduler** for both optimizers (G and D): CosineAnnealing (default) or cyclical Triangular (`lr_scheduler_type="triangular"`)
- **Xavier (Glorot) initialization** for G, D, and autoencoder weights
- **Early Stopping** on validation MAE with best-model restore
- **Optuna**: automatic hyperparameter tuning (generator, lr, batch_size, loss weights, etc.)
- **Walk-forward evaluation**: chronology-aware validation
- **StandardScaler**: fit-on-train-only normalization (no data leakage)

### Diagnostics & Visualization
- **Statistical tests**: ADF (stationarity), VIF (multicollinearity), Ljung-Box (autocorrelation), Breusch-Pagan (heteroscedasticity)
- **Plots**: predictions vs reality with quantile bands, training curves (G/D loss), technical indicators, Fourier decomposition
- **Metrics**: MAE, MAPE, Directional Accuracy, Pinball Loss

### Trading Strategy (Block E)
- **Momentum-Trend strategy** (`src/strategy/`): ported from MultiCharts EasyLanguage, backtester, GPU parameter sweep, optimizer
- Local MOEX data (`src/data_local.py`): M1 bars and ticks

### Engineering
- `torch.compile` + AMP (mixed precision) on CUDA
- WandB logging (optional)
- 294 automated tests (pytest)

---

## Project Structure

```
stockpredictionai-pro/
├── src/
│   ├── config.py              # All parameters (dataclass Config)
│   ├── data.py                # Data loading (Yahoo Finance), indicators, asset panel
│   ├── data_local.py          # Local MOEX data (M1 bars, ticks)
│   ├── dataset.py             # Train/test split, make_sequences, walk-forward
│   ├── train.py               # Main training pipeline
│   ├── features/
│   │   ├── fourier.py         # Fourier approximation (single + multi)
│   │   ├── arima_feat.py      # ARIMA in-sample
│   │   ├── autoencoder.py     # Stacked Autoencoder + VAE variant
│   │   ├── pca_eigen.py       # PCA / eigen-portfolio
│   │   └── sentiment.py       # FinBERT sentiment analysis
│   ├── models/
│   │   ├── base.py            # Shared model blocks
│   │   ├── generator.py       # LSTMGenerator + TransformerGenerator
│   │   ├── discriminator.py   # CNNDiscriminator (Conv1D + BatchNorm)
│   │   ├── gan.py             # WGAN-GP + LR scheduler
│   │   ├── tft.py             # Temporal Fusion Transformer
│   │   ├── classifier.py      # Direction classification
│   │   ├── supervised.py      # Supervised baselines
│   │   └── cross_attention.py # Cross-attention blocks
│   ├── strategy/
│   │   ├── momentum_trend.py  # Momentum-Trend strategy (EasyLanguage port)
│   │   ├── backtester.py      # Backtester
│   │   ├── optimizer.py       # Strategy parameter tuning
│   │   └── gpu_batch.py       # GPU batch parameter sweep
│   └── utils/
│       ├── indicators.py      # Technical indicators (SMA, EMA, RSI, MACD, Bollinger, Momentum, Log Momentum)
│       ├── metrics.py         # MAE, MAPE, sMAPE, Direction Accuracy, Pinball Loss
│       ├── stat_checks.py     # Statistical tests (ADF, VIF, Ljung-Box, Breusch-Pagan)
│       └── visualization.py   # Plots (predictions, training curves, indicators, Fourier)
├── scripts/
│   ├── run_all.py             # Full-cycle run
│   ├── optuna_tune.py         # Hyperparameter tuning
│   ├── feature_importance.py  # XGBoost feature importance
│   ├── baselines.py           # Baselines (naive, ARIMA, supervised)
│   ├── benchmarks.py          # Model comparison
│   ├── sweep_experiment.py    # Hyperparameter sweeps
│   ├── multi_ticker_test.py   # Multi-ticker runs
│   ├── stress_test.py         # Market-regime stress tests
│   ├── run_block_e.py         # Momentum-Trend strategy backtest (+ _wf, _combined)
│   └── konkop_analysis.py     # Strategy trade analysis
├── tests/                     # 294 pytest tests (unit + integration)
├── outputs/                   # Run artifacts (generated, not tracked in git)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/deecalov/stockpredictionai-pro.git
cd stockpredictionai-pro

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch >= 2.1` | Neural networks, GAN, AMP |
| `numpy, pandas` | Data, computation |
| `yfinance >= 0.2` | Quote downloads |
| `scikit-learn` | StandardScaler, metrics |
| `statsmodels >= 0.14` | ARIMA, statistical tests |
| `xgboost >= 2.0` | Feature importance |
| `matplotlib >= 3.7` | Visualization |
| `optuna >= 3.0` | Hyperparameters |
| `transformers >= 4.40` | FinBERT sentiment |
| `pytest >= 7.0` | Testing |
| `wandb >= 0.16` | Logging (opt.) |

---

## Quick Start

### Standard run (LSTM generator)

```bash
python -m src.train --ticker GS --start 2010-01-01 --end 2018-12-31 --test_years 2
```

Full pipeline: data loading → statistical checks → feature engineering → GAN training with early stopping → evaluation → plots and CSV in `outputs/`.

### Transformer (TST) generator

```bash
python -m src.train --ticker GS --start 2010-01-01 --end 2018-12-31 --generator tst
```

### Walk-forward validation (5 splits)

```bash
python -m src.train --ticker GS --start 2010-01-01 --end 2018-12-31 --walk_forward
```

### MHGAN inference (critic-filtered sampling)

```bash
python -m src.train --ticker GS --start 2010-01-01 --end 2018-12-31 --mhgan --mhgan_k 32
```

### Hyperparameter tuning (Optuna)

```bash
python scripts/optuna_tune.py --ticker GS --start 2010-01-01 --end 2018-12-31 --trials 50
```

### Feature Importance (XGBoost)

```bash
python scripts/feature_importance.py --ticker GS --start 2010-01-01 --end 2018-12-31
```

### Tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

All parameters live in `src/config.py` (dataclass `Config`):

| Group | Parameters | Defaults |
|-------|------------|----------|
| **General** | `ticker`, `start`, `end`, `test_years` | `GS`, `2010-01-01`, `2018-12-31`, `2` |
| **GAN** | `seq_len`, `batch_size`, `lr_g`, `lr_d`, `n_epochs`, `critic_steps`, `hidden_size` | `17`, `64`, `1e-3`, `1e-4`, `20`, `5`, `64` |
| **LSTM** | `num_layers`, `dropout_lstm` | `1`, `0.1` |
| **Transformer** | `generator='tst'`, `d_model`, `nhead`, `num_layers_tst`, `dropout_tst` | `64`, `4`, `2`, `0.1` |
| **Loss weights** | `adv_weight`, `l1_weight`, `cls_weight`, `q_weight` | `1.0`, `0.4`, `0.2`, `0.3` |
| **Quantiles** | `quantiles` | `(0.1, 0.5, 0.9)` |
| **Fourier** | `fourier_components` | `(3, 6, 9)` |
| **ARIMA** | `arima_order` | `(5, 1, 0)` |
| **AE / PCA** | `ae_hidden`, `ae_bottleneck`, `ae_epochs`, `ae_variant`, `pca_components` | `64`, `32`, `10`, `'stacked'`, `12` |
| **LR Scheduler** | `use_lr_scheduler`, `lr_scheduler_type`, `lr_scheduler_min_factor`, `lr_cycle_length` | `True`, `'cosine'`, `0.1`, `0` (auto) |
| **Weight init** | `use_xavier_init` | `True` |
| **MHGAN** | `use_mhgan`, `mhgan_k` | `False`, `32` |
| **Early Stopping** | `early_stopping_patience` | `5` |
| **Sentiment** | `use_sentiment` | `False` |

---

## Output Files (`outputs/`)

The `outputs/` directory is generated at runtime and not tracked in git (listed in `.gitignore`).

| File | Description |
|------|-------------|
| `test_predictions.csv` | `y_true`, `y_pred`, `y_logit`, `q10`, `q50`, `q90` |
| `test_predictions_split{i}.csv` | Predictions for each walk-forward split |
| `metrics_walk_forward.csv` | Aggregated metrics across splits |
| `feature_importance_xgb.csv` | Feature importance (XGBoost) |
| `optuna_best.json` | Best hyperparameters (Optuna) |
| `pred_vs_real.png` | Prediction plot with quantile bands |
| `training_curves.png` | Training curves (G loss, D loss) |
| `technical_indicators.png` | Technical indicators dashboard |
| `fourier_components.png` | Fourier decomposition (k=3, 6, 9) |

---

## GAN Architecture

```
         ┌─────────────────────────────────────────────┐
         │  Generator (LSTM or Transformer)            │
         │  Input: [B, T, F+1]  (features + noise)     │
         │  Outputs:                                   │
         │    y_reg     [B]     — ΔP regression        │
         │    y_cls     [B]     — direction (logit)    │
         │    y_q       [B, 3]  — quantiles (q10/50/90)│
         └───────────────┬─────────────────────────────┘
                         │
         ┌───────────────▼─────────────────────────────┐
         │  Discriminator (3× Conv1D + BatchNorm)      │
         │  Input: [B, T, F] concat y → [B, T, F+1]    │
         │  Conv1D: 32→64→128, FC: 220→220→1           │
         │  Output: WGAN score (no sigmoid)            │
         └─────────────────────────────────────────────┘

 Loss_G = adv_weight * (-D(G(x))) + l1_weight * L1 + cls_weight * BCE + q_weight * Pinball
 Loss_D = -(D(real) - D(fake)) + λ * GP
```

---

## Data Processing Pipeline

```
 Yahoo Finance (40+ tickers)
       │
       ▼
 Technical indicators (13)
       │
       ▼
 Multi-Fourier (k=3, 6, 9) + ARIMA
       │
       ▼
 Statistical checks (ADF, VIF, Ljung-Box, Breusch-Pagan)
       │
       ▼
 Autoencoder → latent features
       │
       ▼
 PCA / eigen-portfolio
       │
       ▼
 StandardScaler (fit on train only)
       │
       ▼
 Sliding window sequences [B, T, F]
       │
       ▼
 WGAN-GP training (LR scheduler + early stopping)
       │
       ▼
 Evaluation (MAE, MAPE, DirAcc, Pinball) + visualization
```

---

## Testing

The project is covered by **294 automated tests** (pytest):

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_data` | 4 | Data loading, indicators, NaN |
| `test_data_local` | 99 | Local MOEX data (M1, ticks, aggregation) |
| `test_dataset` | 4 | Train/test split, sequences, walk-forward |
| `test_features` | 20 | Fourier (single + multi), ARIMA, AE + VAE, PCA, sentiment, log_momentum |
| `test_indicators` | 31 | Technical indicators |
| `test_metrics` | 19 | MAE, MAPE, sMAPE, DirAcc, Pinball |
| `test_models` | 39 | LSTM, Transformer, TFT, Discriminator, WGAN-GP, MHGAN, LR schedulers, Xavier init |
| `test_baselines` | 23 | Naive/ARIMA/supervised baselines |
| `test_pipeline` | 6 | Full pipeline (LSTM + TST), walk-forward, CSV output |
| `test_pipeline_local` | 8 | Pipeline on local MOEX data |
| `test_stat_checks` | 14 | ADF, VIF, Ljung-Box, Breusch-Pagan |
| `test_train_helpers` | 16 | Training helper functions |
| `test_visualization` | 11 | Plots |

---

## License

MIT License.

---

## Acknowledgements

- [Boris Banushev](https://github.com/borisbanushev/stockpredictionai) — original idea and notebook
- PyTorch — framework (`torch.compile`, AMP)
- Time series community — inspiration (TST, N-BEATS, N-HiTS)
- Paul Deecalov — 2025
