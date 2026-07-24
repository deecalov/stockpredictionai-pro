"""Benchmark MHGAN inference vs plain sampling on a single trained GAN.

The model is trained once (seeded); then the test set is evaluated repeatedly:
plain stochastic sampling and MHGAN (Metropolis-Hastings over k critic-scored
generator samples) for several k values. Inference is stochastic (noise is
injected inside G.forward), so each mode is repeated and reported as mean/std.

Usage:
    python scripts/benchmark_mhgan.py --ticker GS --start 2010-01-01 --end 2018-12-31
    python scripts/benchmark_mhgan.py --generator lstm
    python scripts/benchmark_mhgan.py --panel_csv tests/fixtures/panel_cache.csv
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

from src.config import Config
from src.data import build_panel_auto
from src.dataset import train_test_split_by_years, make_sequences
from src.models.gan import WGAN_GP
from src.train import (
    build_features_safe, fit_transforms, run_one_split,
    evaluate_model, compute_metrics, set_global_seed, _log_device_info,
)

SEED = 42


def load_panel(args, cfg):
    if args.panel_csv:
        print(f"Loading panel from CSV: {args.panel_csv}")
        return pd.read_csv(args.panel_csv, index_col=0, parse_dates=True)
    return build_panel_auto(cfg)


def main():
    parser = argparse.ArgumentParser(description="MHGAN vs plain sampling benchmark")
    parser.add_argument("--ticker", default="GS")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2018-12-31")
    parser.add_argument("--test_years", type=int, default=2)
    parser.add_argument("--generator", default="tst", choices=["lstm", "tst"])
    parser.add_argument("--data_source", default="yfinance", choices=["yfinance", "local"])
    parser.add_argument("--timeframe", default="D1")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--panel_csv", default="",
                        help="Load panel from a cached CSV instead of downloading")
    parser.add_argument("--k_values", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="outputs/experiments/mhgan_benchmark.csv")
    args = parser.parse_args()

    cfg = Config(ticker=args.ticker, start=args.start, end=args.end,
                 test_years=args.test_years, generator=args.generator,
                 data_source=args.data_source, timeframe=args.timeframe,
                 data_path=args.data_path)
    if cfg.data_source == "local" or cfg.timeframe != "D1":
        cfg.apply_timeframe_defaults()
    # Single-process DataLoader: spawn workers on Windows re-import the heavy
    # module stack per worker, which is slower than the D1-sized data needs
    cfg.num_workers = 0

    set_global_seed(SEED, deterministic=cfg.deterministic)
    _log_device_info()

    print(f"Loading panel for {cfg.ticker}...")
    panel = load_panel(args, cfg)
    print(f"Panel: {panel.shape}")

    train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)
    tr_feat, te_feat = build_features_safe(train_panel, test_panel, cfg)
    tr_all, te_all = fit_transforms(tr_feat, te_feat, cfg)

    print(f"Training {cfg.generator.upper()} GAN once (seed {SEED})...")
    t0 = time.time()
    met_train, _, _, _, _, model = run_one_split(tr_all, te_all, cfg, verbose=True)
    print(f"Trained in {time.time() - t0:.1f}s, baseline test metrics: {met_train}")
    assert isinstance(model, WGAN_GP), "MHGAN benchmark requires the GAN model type"

    Xte, yte = make_sequences(te_all, target_col=cfg.ticker, seq_len=cfg.seq_len)
    modes = [("plain", 0)] + [(f"mhgan_k{k}", k) for k in args.k_values]

    rows = []
    for mode_name, k in modes:
        for rep in range(args.repeats):
            torch.manual_seed(1234 + rep)
            t1 = time.time()
            yp, _, yq = evaluate_model(model, Xte, yte, cfg.batch_size, cfg.quantiles,
                                       use_mhgan=(k > 0), mhgan_k=k)
            met = compute_metrics(yte, yp, yq, cfg.quantiles)
            met.update({"mode": mode_name, "k": k, "rep": rep,
                        "infer_sec": round(time.time() - t1, 2)})
            rows.append(met)
        last = [r for r in rows if r["mode"] == mode_name]
        mae_mean = sum(r["MAE"] for r in last) / len(last)
        print(f"  {mode_name:<12} MAE(mean over {args.repeats} reps) = {mae_mean:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved per-repeat results: {args.output}")

    summary = (df.groupby("mode", sort=False)
                 .agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                      DirAcc_mean=("DirAcc", "mean"), DirAcc_std=("DirAcc", "std"),
                      Pinball_mean=("PinballLoss", "mean"),
                      Pinball_std=("PinballLoss", "std"),
                      infer_sec=("infer_sec", "mean"))
                 .round(5))
    summary_path = args.output.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved summary: {summary_path}\n")
    print(f"Generator: {cfg.generator}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
