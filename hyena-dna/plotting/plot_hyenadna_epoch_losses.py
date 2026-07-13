#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot epoch-level HyenaDNA training losses.")
    parser.add_argument("--log", required=True, help="Slurm stdout containing JSON epoch rows.")
    parser.add_argument("--out-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    final = None
    for line in Path(args.log).read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if {"epoch", "train_5mC_loss", "val_5mC_loss"}.issubset(obj):
            rows.append(obj)
        elif obj.get("status") == "ok":
            final = obj

    if not rows:
        raise SystemExit(f"No epoch loss rows found in {args.log}")

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = out_prefix.with_suffix(".epoch_losses.tsv")
    png_path = out_prefix.with_suffix(".epoch_losses.png")

    with tsv_path.open("w") as handle:
        handle.write("epoch\ttrain_5mC_loss\tval_5mC_loss\n")
        for row in rows:
            handle.write(
                f"{row['epoch']}\t{row['train_5mC_loss']:.10f}\t{row['val_5mC_loss']:.10f}\n"
            )

    epochs = [row["epoch"] for row in rows]
    train = [row["train_5mC_loss"] for row in rows]
    val = [row["val_5mC_loss"] for row in rows]

    best_epoch = None
    if final is not None:
        best_epoch = final.get("best_epoch")
    if best_epoch is None:
        best_epoch = min(rows, key=lambda row: row["val_5mC_loss"])["epoch"]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train, marker="o", label="train BCE")
    plt.plot(epochs, val, marker="o", label="validation BCE")
    plt.axvline(best_epoch, color="black", linestyle="--", linewidth=1, label=f"best epoch {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("5mC masked BCE loss")
    plt.title("P(5mC | DNA) training curve")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)

    print(
        json.dumps(
            {
                "rows": len(rows),
                "best_epoch": best_epoch,
                "tsv": str(tsv_path),
                "plot": str(png_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
