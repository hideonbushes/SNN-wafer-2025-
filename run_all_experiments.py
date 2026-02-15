"""Run all 8 Wafer2Spike experiments with one shared dataloader build.

Usage (Colab/local):
  python run_all_experiments.py --data_path /content/drive/MyDrive/WM-811k/LSWMD.pkl
"""

import argparse
import contextlib
import datetime as dt
import os
from typing import Callable, Dict, List

from wafer2spike_original_train_test import (
    build_dataloaders,
    training as original_training,
    CurrentBasedSNN as OriginalBase,
)
from wafer2spike_improved_train_test import (
    training as improved_training,
    CurrentBasedSNN as ImprovedBase,
)
from wafer2spike_original_02_residual_prenorm_train_test import CurrentBasedSNNResidual as OriginalResidual
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


def run_one(exp: Dict, dataloaders, log_dir: str):
    exp_name = exp["name"]
    log_path = os.path.join(log_dir, f"{exp_name}.log")
    print(f"\n===== Running: {exp_name} =====")
    print(f"Log file: {log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        tee = Tee(os.sys.stdout, f)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            exp["runner"](dataloaders)

    print(f"===== Done: {exp_name} =====")


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
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    print("Building shared dataloaders once...")
    dataloaders = build_dataloaders(data_path=args.data_path, batch_size=args.batch_size)

    experiments: List[Dict] = [
        {
            "name": "original_01_glif_alif",
            "runner": lambda dl: original_training(network=OriginalBase, params=[0.05, 0.10, 0.08, 0.30], dataloaders=dl),
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
            "name": "original_02_residual_prenorm",
            "runner": lambda dl: original_training(network=OriginalResidual, params=[0.05, 0.10, 0.08, 0.30], dataloaders=dl),
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
            "runner": lambda dl: original_training(network=OriginalRecurrent, params=[0.05, 0.10, 0.08, 0.30], dataloaders=dl),
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
            "runner": lambda dl: original_training(network=OriginalAttention, params=[0.05, 0.10, 0.08, 0.30], dataloaders=dl),
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

    if args.only:
        filters = [f.lower() for f in args.only]
        experiments = [e for e in experiments if any(f in e["name"].lower() for f in filters)]

    if not experiments:
        raise ValueError("No experiments selected. Check --only filters.")

    summary_path = os.path.join(log_dir, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("Run started\n")
        sf.write(f"data_path={args.data_path}\n")
        sf.write(f"batch_size={args.batch_size}\n")
        sf.write("experiments:\n")
        for exp in experiments:
            sf.write(f"- {exp['name']}\n")

    for exp in experiments:
        run_one(exp, dataloaders, log_dir)

    print("\nAll selected experiments completed.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
