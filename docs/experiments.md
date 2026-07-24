# Experiment Log

## 2026-07-24 — New feature benchmarks: MHGAN, AE variants, LR schedulers, GPSampler

**Setup:** GS daily bars 2010–2018, `test_years=2` (1806 train / 520 test bars after
feature engineering), seq_len 12, TST generator defaults, seed 42, CPU (torch 2.13).
Panel of 56 assets cached to CSV (includes the newly added `^NYA` NYSE Composite and
`^IRX` 13-week T-bill). Runs executed on the gpu Linux box (`~/work/spai`); the local
workstation was out of physical RAM at the time.

Raw artifacts (not tracked in git): `outputs/experiments/mhgan_benchmark_{tst,lstm}[_summary].csv`,
`sweep_ae_variant.csv`, `sweep_lr_scheduler.csv`, `optuna_trials.csv`, `outputs/optuna_best.json`.

### 1. MHGAN inference vs plain sampling

One model per generator trained once; each inference mode repeated 5 times
(inference is stochastic — noise is injected in `G.forward`). `infer_sec` is per
test-set pass.

TST generator:

| mode | MAE | DirAcc | Pinball | infer_sec |
|---|---|---|---|---|
| plain | 1.17034 ± 0.00034 | 0.51282 | 0.10542 | 0.01 |
| mhgan_k16 | 1.16487 ± 0.00020 | 0.51282 | 0.10551 | 0.17 |
| mhgan_k32 | 1.16487 ± 0.00034 | 0.51282 | 0.10550 | 0.32 |
| mhgan_k64 | 1.16485 ± 0.00011 | 0.51282 | 0.10552 | 0.64 |

LSTM generator:

| mode | MAE | DirAcc | Pinball | infer_sec |
|---|---|---|---|---|
| plain | 0.36375 ± 0.00054 | 0.49704 | 0.14834 | 0.00 |
| mhgan_k16 | 0.35891 ± 0.00049 | 0.49507 | 0.14812 | 0.09 |
| mhgan_k32 | 0.35937 ± 0.00037 | 0.49507 | 0.14811 | 0.18 |
| mhgan_k64 | 0.35902 ± 0.00057 | 0.49428 | 0.14822 | 0.36 |

**Findings:** consistent small MAE improvement on both generators (~0.5% TST, ~1.3%
LSTM), well outside the repeat noise; Pinball flat; DirAcc unchanged (TST) or within
noise (LSTM). k=16 already captures the full benefit — larger k only costs time.

**Recommendation:** use `--mhgan --mhgan_k 16` for final test evaluation; keep plain
`predict` inside training/validation loops (k× inference cost).

### 2. Autoencoder variant ablation (single seed — indicative only)

| experiment | MAE | DirAcc | Pinball | features |
|---|---|---|---|---|
| ae_none (default) | 1.2930 | 0.5128 | 0.1516 | 57 |
| ae_stacked | 0.5484 | 0.4536 | 0.1247 | 89 |
| ae_vae | 1.3198 | 0.5128 | **0.0768** | 89 |

**Caveat:** single-seed GAN runs are noisy. Note that every run with MAE ≈ 1.25–1.32
has DirAcc pinned at exactly 0.5128 — the test-set base rate of up-days — i.e. the
generator collapsed to a near-constant prediction. The stacked-AE run escaped the
collapse in this seed (best MAE); the VAE run collapsed on point forecasts but gave by
far the best quantile calibration (Pinball 0.077). Inconclusive for changing the
`ae_epochs=0` default; needs a multi-seed rerun.

### 3. LR scheduler comparison (single seed — indicative only)

| experiment | MAE | DirAcc | Pinball |
|---|---|---|---|
| lrsched_cosine (default) | 1.2930 | 0.5128 | 0.1516 |
| lrsched_triangular | **0.4355** | 0.4714 | 0.1384 |
| lrsched_none | 1.2517 | 0.5128 | 0.1529 |

Same collapse pattern: cosine and no-scheduler runs collapsed, the triangular run did
not and produced a 3× better MAE. A strong signal worth a multi-seed follow-up before
making triangular the default.

### 4. Optuna GPSampler sanity run (5 trials)

`--sampler gp` works (optuna emits an `ExperimentalWarning`; GPSampler is supported
from optuna 3.6). Best of just 5 trials reached **MAE 0.0930** — an order of magnitude
better than the defaults used above:

```
tst, seq_len=24, batch_size=256, hidden_size=64, lr_g=3.2e-3, lr_d=1.3e-5,
critic_steps=10, d_model=32, nhead=2, num_layers_tst=3, dropout_tst=0.23,
l1_weight=0.17, cls_weight=0.21, q_weight=0.50, grad_clip=4.4
```

### Conclusions

1. **MHGAN**: enable for final evaluations (`--mhgan --mhgan_k 16`) — small, consistent,
   essentially free quality gain at k=16.
2. **Default hyperparameters frequently collapse the TST GAN** (DirAcc pinned at the
   base rate). The GP-found parameters avoid the collapse and dominate every sweep
   result above; a full GP tuning run (50+ trials) and adopting its best parameters is
   the highest-leverage next step.
3. **AE variants and triangular LR**: promising but single-seed; rerun with 3–5 seeds
   after the hyperparameter refresh.
