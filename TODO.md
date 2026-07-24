# TODO

## Next up

- [x] Benchmark MHGAN inference vs plain sampling on GS — done 2026-07-24, see
      `docs/experiments.md`: consistent MAE gain, k=16 is enough; use for final evals
- [x] Ablation: VAE vs stacked autoencoder — done 2026-07-24 (single seed,
      inconclusive: stacked escaped GAN collapse, VAE best Pinball; needs multi-seed)
- [x] Sweep: triangular vs cosine LR schedule — done 2026-07-24 (single seed:
      triangular escaped collapse, 3x better MAE; needs multi-seed)
- [ ] Full GP tuning run (50+ trials, `--sampler gp`) and adopt best params — 5-trial
      sanity already hit MAE 0.093 vs ~1.17 with defaults
- [ ] Multi-seed (3-5) rerun of ae_variant and lr_scheduler sweeps after the
      hyperparameter refresh

## Ideas (from the source notebook, not yet implemented)

- [x] Optuna `GPSampler` option in `scripts/optuna_tune.py` — done 2026-07-24
- [x] Correlated assets: NYSE Composite (`^NYA`) and `^IRX` 13-week T-bill as the
      short-rate proxy (Yahoo has no SOFR series; LIBOR discontinued) — done 2026-07-24
- [ ] Anomaly detection in options pricing via Self-Organizing Maps (declared in the
      source, never implemented there; feasible via synthetic Black-Scholes pricing)
- [ ] RL-driven hyperparameter re-tuning (Rainbow / PPO deciding *when* to re-run
      Optuna) — research-only, low practical value

## Housekeeping

- [ ] Decide the fate of the C# port (`RTSF_Strategy_ML`) — extract from
      `pre-cleanup-backup.bundle` into a separate repo or drop
- [ ] Delete `e:\stockpredictionai-pro\pre-cleanup-backup.bundle` once the cleaned
      history is confirmed good
- [ ] gpu box: `~/work/spai` holds a copy of this repo + deps for remote benchmarks
      (reusable; delete when no longer needed)
