#!/usr/bin/env python3
"""Minimal training stub used by the web UI.

It just prints dummy epoch logs so that the UI can stream something
while we implement the full meta-model logic later.
"""
import argparse
import time
from pathlib import Path
import random
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--sample-size", dest="sample_size", default="base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--configs-per-sample", type=int, default=10,
                   help="Number of hyperparameter configurations to sample per iteration")
    p.add_argument("--perturbations", type=int, default=10,
                   help="Number of weight perturbations per configuration")
    p.add_argument("--iterations", type=int, default=3,
                   help="Number of meta-model training iterations")
    p.add_argument("--outdir", default=None,
                   help="Optional explicit output directory (defaults to reports/<dataset>_<sample_size>/)")
    return p.parse_args()

def main():
    args = parse_args()

    run_dir = Path(args.outdir) if args.outdir else Path(__file__).parent / "reports" / f"{args.dataset}_{args.sample_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training with configs_per_sample={args.configs_per_sample}, "
          f"perturbations={args.perturbations}, iterations={args.iterations}")
    print("This is a stub - real meta-model training will be implemented here.", flush=True)

    # Dummy training loop
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(1, args.epochs + 1):
        tl = random.uniform(0.5, 1.0) / epoch
        vl = tl + random.uniform(-0.05, 0.05)
        ta = random.uniform(0.5, 0.7) + epoch * 0.02
        va = ta - random.uniform(0.0, 0.05)
        train_loss.append(tl)
        val_loss.append(vl)
        train_acc.append(min(ta, 1.0))
        val_acc.append(min(va, 1.0))

        print(f"Epoch {epoch}/{args.epochs} - "
              f"train_loss: {tl:.4f} val_loss: {vl:.4f} "
              f"train_acc: {ta:.4f} val_acc: {va:.4f}", flush=True)
        time.sleep(0.5)

    # Save dummy model file
    (run_dir / "model.pth").write_text("dummy model contents")

    # Save metrics json
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
    }
    with open(run_dir / "meta_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot history
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc, label="train")
    plt.plot(val_acc, label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_history.png")

    print("Training stub finished.")

if __name__ == "__main__":
    main()
