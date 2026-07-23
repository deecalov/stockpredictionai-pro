# TODO

## Next up

- [ ] Benchmark MHGAN inference vs plain sampling on GS (MAE / DirAcc / Pinball, `--mhgan`, sweep `mhgan_k` = 16 / 32 / 64)
- [ ] Ablation: VAE vs stacked autoencoder (`ae_epochs > 0`, `ae_variant=vae`) — the stacked AE previously added noise, check whether VAE latents behave better
- [ ] Sweep: triangular vs cosine LR schedule (`--lr_scheduler triangular`, tune `lr_cycle_length`)

## Ideas (from the source notebook, not yet implemented)

- [ ] Optuna `GPSampler` option in `scripts/optuna_tune.py` (Bayesian optimization with Gaussian processes, as in the source)
- [ ] Correlated assets: add SOFR (LIBOR is discontinued) and NYSE Composite (`^NYA`)
- [ ] Anomaly detection in options pricing via Self-Organizing Maps (declared in the source, never implemented there)
- [ ] RL-driven hyperparameter re-tuning (Rainbow / PPO deciding *when* to re-run Optuna) — research-only, low practical value

