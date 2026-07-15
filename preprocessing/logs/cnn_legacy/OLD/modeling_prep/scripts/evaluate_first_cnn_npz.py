#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_first_cnn_npz import ChunkedNPZDataset, FirstKmerCNN, binary_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained first CNN on NPZ chunks.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--max-chunks", type=int, default=16)
    p.add_argument("--selection-mode", choices=["first", "uniform"], default="uniform")
    return p.parse_args()


def metrics_at_threshold(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
    shifted_logits = logits - torch.logit(torch.tensor(threshold))
    return binary_metrics(shifted_logits, labels)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    model = FirstKmerCNN(
        embedding_dim=int(config["embedding_dim"]),
        conv_channels=int(config["conv_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ChunkedNPZDataset(
        args.data_dir,
        shuffle_chunks=False,
        shuffle_rows=False,
        max_chunks=args.max_chunks,
        seed=int(config.get("seed", 42)),
        selection_mode=args.selection_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    probs_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["ref_kmer"].to(device),
                batch["query_kmer"].to(device),
                batch["pos_norm"].to(device),
                batch["ref_strand"].to(device),
                batch["ref_mod_strand"].to(device),
            )
            probs_all.append(torch.sigmoid(logits).cpu())
            labels_all.append(batch["label"].cpu())

    probs = torch.cat(probs_all)
    labels = torch.cat(labels_all)
    probs_np = probs.numpy()

    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    threshold_metrics = {
        str(t): metrics_at_threshold(probs, labels, t)
        for t in thresholds
    }

    result = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "rows": int(labels.numel()),
        "label_counts": {
            "0": int((labels == 0).sum().item()),
            "1": int((labels == 1).sum().item()),
        },
        "probability_summary": {
            "min": float(np.min(probs_np)),
            "p01": float(np.percentile(probs_np, 1)),
            "p05": float(np.percentile(probs_np, 5)),
            "p25": float(np.percentile(probs_np, 25)),
            "median": float(np.percentile(probs_np, 50)),
            "p75": float(np.percentile(probs_np, 75)),
            "p95": float(np.percentile(probs_np, 95)),
            "p99": float(np.percentile(probs_np, 99)),
            "max": float(np.max(probs_np)),
            "mean": float(np.mean(probs_np)),
        },
        "threshold_metrics": threshold_metrics,
    }

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
