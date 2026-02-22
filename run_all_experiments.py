"""Run all 8 Wafer2Spike experiments with one shared dataloader build.

Usage (Colab/local):
  python run_all_experiments.py --data_path /content/drive/MyDrive/WM-811k/LSWMD.pkl
"""

import argparse
import contextlib
import datetime as dt
import json
import math
import os
import random
import statistics
import time
from typing import Dict, List

import numpy as np
import torch

from wafer2spike_original_train_test import (
    build_dataloaders,
    training as original_training,
    CurrentBasedSNN as OriginalBase,
)
from wafer2spike_improved_train_test import (
    training as improved_training,
    CurrentBasedSNN as ImprovedBase,
)
from wafer2spike_original_02_residual_prenorm_train_test import (
    CurrentBasedSNNResidual as OriginalResidual,
    CurrentBasedSNNResidualBernoulli as OriginalResidualBernoulli,
)
from wafer2spike_improved_02_residual_prenorm_train_test import CurrentBasedSNNResidual as ImprovedResidual
from wafer2spike_original_03_recurrent_conv_train_test import CurrentBasedSNNRecurrent as OriginalRecurrent
from wafer2spike_improved_03_recurrent_conv_train_test import CurrentBasedSNNRecurrent as ImprovedRecurrent
from wafer2spike_original_04_spike_self_attention_train_test import CurrentBasedSNNAttn as OriginalAttention
from wafer2spike_improved_04_spike_self_attention_train_test import CurrentBasedSNNAttn as ImprovedAttention


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convergence_epoch(val_acc_history, ratio=0.95):
    if not val_acc_history:
        return None
    target = max(val_acc_history) * ratio
    for idx, acc in enumerate(val_acc_history, start=1):
        if acc >= target:
            return idx
    return None


def run_one(exp: Dict, dataloaders, log_dir: str, repeat_idx: int, seed: int) -> Dict[str, str]:
    exp_name = exp["name"]
    log_path = os.path.join(log_dir, f"{exp_name}.run{repeat_idx}.log")
    print(f"\n===== Running: {exp_name} | run={repeat_idx} | seed={seed} =====")
    print(f"Log file: {log_path}")

    status = "success"
    message = ""
    metrics = None
    elapsed_sec = 0.0
    with open(log_path, "w", encoding="utf-8") as f:
        tee = Tee(os.sys.stdout, f)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            try:
                set_seed(seed)
                t0 = time.perf_counter()
                metrics = exp["runner"](dataloaders)
                elapsed_sec = time.perf_counter() - t0
            except Exception as exc:
                status = "failed"
                message = f"{type(exc).__name__}: {exc}"
                print(f"[ERROR] {exp_name} failed - {message}")

    if status == "success":
        print(f"===== Done: {exp_name} =====")
    else:
        print(f"===== Failed: {exp_name} =====")

    return {
        "name": exp_name,
        "status": status,
        "message": message,
        "log": log_path,
        "run": repeat_idx,
        "seed": seed,
        "elapsed_sec": elapsed_sec,
        "metrics": metrics,
    }


def summarize_experiment_runs(exp_name: str, runs: List[Dict]) -> Dict:
    total = len(runs)
    failed = [r for r in runs if r["status"] == "failed"]
    success = [r for r in runs if r["status"] == "success"]
    failure_rate = (len(failed) / total) if total else math.nan

    test_accs = [r["metrics"]["test_acc"] for r in success if r.get("metrics") and "test_acc" in r["metrics"]]
    test_losses = [r["metrics"]["test_loss"] for r in success if r.get("metrics") and "test_loss" in r["metrics"]]
    converge_epochs = [
        convergence_epoch(r["metrics"].get("val_acc_history", []))
        for r in success if r.get("metrics")
    ]
    converge_epochs = [c for c in converge_epochs if c is not None]
    elapsed = [r["elapsed_sec"] for r in success]

    def stat_pack(values):
        if not values:
            return {"mean": None, "std": None}
        return {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        }

    return {
        "name": exp_name,
        "total_runs": total,
        "successful_runs": len(success),
        "failed_runs": len(failed),
        "failure_rate": failure_rate,
        "test_acc": stat_pack(test_accs),
        "test_loss": stat_pack(test_losses),
        "convergence_epoch": stat_pack(converge_epochs),
        "elapsed_sec": stat_pack(elapsed),
    }


def main():
    parser = argparse.ArgumentParser(description="Run all 8 wafer2spike experiments with one shared dataloader")
    parser.add_argument("--data_path", type=str, default="/content/drive/MyDrive/WM-811k/LSWMD.pkl")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_dir", type=str, default="experiment_logs")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional experiment name filters (substring match). Example: --only improved_03 recurrent",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated runs per experiment.")
    parser.add_argument("--seed_base", type=int, default=42, help="Base seed used for repeated runs.")
    parser.add_argument(
        "--compare_residual_encodings",
        action="store_true",
        help="Shortcut to run only original residual constant/bernoulli with shared dataloaders.",
    )
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    print("Building shared dataloaders once...")
    dataloaders = build_dataloaders(data_path=args.data_path, batch_size=args.batch_size)

    experiments: List[Dict] = [
        {
            "name": "original_01_glif_alif",
            "runner": lambda dl: original_training(network=OriginalBase, params=[0.05, 0.10, 0.08, 0.30], dataloaders=dl, return_metrics=True),
        },
        {
            "name": "improved_01_glif_alif",
            "runner": lambda dl: improved_training(
                network=ImprovedBase,
                params=[0.05, 0.10, 0.08, 5.0],
                dataloaders=dl,
                spike_ts=10,
                batch_size=args.batch_size,
                epochs=20,
                lr=1e-4,
                dropout_fc=0.20,
            ),
        },
        {
            "name": "original_02_residual_prenorm_constant",
            "runner": lambda dl: original_training(network=OriginalResidual, params=[0.05, 0.10, 0.08, 0.30], dataloaders=dl, return_metrics=True),
        },
        {
            "name": "original_02_residual_prenorm_bernoulli",
            "runner": lambda dl: original_training(network=OriginalResidualBernoulli, params=[0.05, 0.10, 0.08, 0.30], dataloaders=dl, return_metrics=True),
        },
        {
            "name": "original_02_residual_prenorm_bernoulli",
            "runner": lambda dl: original_training(network=OriginalResidualBernoulli, params=[0.05, 0.10, 0.08, 0.30], dataloaders=dl),
        },
        {
            "name": "improved_02_residual_prenorm",
            "runner": lambda dl: improved_training(
                network=ImprovedResidual,
                params=[0.05, 0.10, 0.08, 5.0],
                dataloaders=dl,
                spike_ts=10,
                batch_size=args.batch_size,
                epochs=20,
                lr=1e-4,
                dropout_fc=0.20,
            ),
        },
        {
            "name": "original_03_recurrent_conv",
            "runner": lambda dl: original_training(network=OriginalRecurrent, params=[0.05, 0.10, 0.08, 0.30], dataloaders=dl, return_metrics=True),
        },
        {
            "name": "improved_03_recurrent_conv",
            "runner": lambda dl: improved_training(
                network=ImprovedRecurrent,
                params=[0.05, 0.10, 0.08, 5.0],
                dataloaders=dl,
                spike_ts=10,
                batch_size=args.batch_size,
                epochs=20,
                lr=1e-4,
                dropout_fc=0.20,
            ),
        },
        {
            "name": "original_04_spike_self_attention",
            "runner": lambda dl: original_training(network=OriginalAttention, params=[0.05, 0.10, 0.08, 0.30], dataloaders=dl, return_metrics=True),
        },
        {
            "name": "improved_04_spike_self_attention",
            "runner": lambda dl: improved_training(
                network=ImprovedAttention,
                params=[0.05, 0.10, 0.08, 5.0],
                dataloaders=dl,
                spike_ts=10,
                batch_size=args.batch_size,
                epochs=20,
                lr=1e-4,
                dropout_fc=0.20,
            ),
        },
    ]

    if args.compare_residual_encodings:
        filters = ["original_02_residual_prenorm_constant", "original_02_residual_prenorm_bernoulli"]
        experiments = [e for e in experiments if e["name"] in filters]
    elif args.only:
        filters = [f.lower() for f in args.only]
        experiments = [e for e in experiments if any(f in e["name"].lower() for f in filters)]

    if not experiments:
        raise ValueError("No experiments selected. Check --only filters.")

    summary_path = os.path.join(log_dir, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("Run started\n")
        sf.write(f"data_path={args.data_path}\n")
        sf.write(f"batch_size={args.batch_size}\n")
        sf.write(f"repeats={args.repeats}\n")
        sf.write(f"seed_base={args.seed_base}\n")
        sf.write("experiments:\n")
        for exp in experiments:
            sf.write(f"- {exp['name']}\n")

    results = []
    for exp in experiments:
        for run_idx in range(1, args.repeats + 1):
            seed = args.seed_base + run_idx - 1
            results.append(run_one(exp, dataloaders, log_dir, run_idx, seed))

    grouped = {}
    for res in results:
        grouped.setdefault(res["name"], []).append(res)
    aggregate = [summarize_experiment_runs(name, runs) for name, runs in grouped.items()]

    aggregate_path = os.path.join(log_dir, "aggregate_metrics.json")
    with open(aggregate_path, "w", encoding="utf-8") as af:
        json.dump(aggregate, af, indent=2)

    with open(summary_path, "a", encoding="utf-8") as sf:
        sf.write("\nresults:\n")
        for res in results:
            if res["status"] == "success":
                sf.write(f"- {res['name']} run={res['run']} seed={res['seed']}: success\n")
            else:
                sf.write(f"- {res['name']} run={res['run']} seed={res['seed']}: failed ({res['message']})\n")
        sf.write("\naggregate:\n")
        for row in aggregate:
            sf.write(
                f"- {row['name']}: failure_rate={row['failure_rate']:.3f}, "
                f"test_acc_mean={row['test_acc']['mean']}, test_acc_std={row['test_acc']['std']}, "
                f"conv_epoch_mean={row['convergence_epoch']['mean']}, elapsed_mean={row['elapsed_sec']['mean']}\n"
            )

    failed = [r for r in results if r["status"] == "failed"]
    print("\nAll selected experiments completed.")
    print(f"Summary: {summary_path}")
    print(f"Aggregate metrics: {aggregate_path}")
    if failed:
        print("Failed experiments:")
        for r in failed:
            print(f"- {r['name']}: {r['message']} (log: {r['log']})")


if __name__ == "__main__":
    main()
