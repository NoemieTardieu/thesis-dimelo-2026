#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_balanced_cnn_v2 import BalancedCNNDataset, ImprovedKmerCNN
from train_first_cnn_npz import binary_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a balanced CNN v2 checkpoint on one balanced NPZ file.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8192)
    return p.parse_args()


def metrics_at_threshold(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
    shifted_logits = logits - torch.logit(torch.tensor(threshold))
    return binary_metrics(shifted_logits, labels)


def roc_auc_score_np(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(probs)
    sorted_probs = probs[order]
    ranks = np.empty(labels.size, dtype=np.float64)

    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_probs[end] == sorted_probs[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_np(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-probs)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(sorted_labels.size) + 1)
    return float(precision[sorted_labels == 1].sum() / n_pos)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    model = ImprovedKmerCNN(
        embedding_dim=int(config["embedding_dim"]),
        conv_channels=int(config["conv_channels"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = BalancedCNNDataset(args.data_npz)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probs_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []

    with torch.no_grad():
        for ref_kmer, query_kmer, pos_norm, ref_strand, ref_mod_strand, labels in loader:
            logits = model(
                ref_kmer.to(device),
                query_kmer.to(device),
                pos_norm.to(device),
                ref_strand.to(device),
                ref_mod_strand.to(device),
            )
            probs_all.append(torch.sigmoid(logits).cpu())
            labels_all.append(labels.cpu())

    probs = torch.cat(probs_all)
    labels = torch.cat(labels_all)
    probs_np = probs.numpy()
    labels_np = labels.numpy().astype(np.int64)

    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    threshold_metrics = {str(t): metrics_at_threshold(probs, labels, t) for t in thresholds}

    fine_thresholds = np.linspace(0.01, 0.99, 99)
    fine_metrics = [(float(t), metrics_at_threshold(probs, labels, float(t))) for t in fine_thresholds]
    best_f1_threshold, best_f1_metrics = max(fine_metrics, key=lambda item: item[1]["f1"])
    best_accuracy_threshold, best_accuracy_metrics = max(fine_metrics, key=lambda item: item[1]["accuracy"])

    result = {
        "checkpoint": args.checkpoint,
        "data_npz": args.data_npz,
        "rows": int(labels.numel()),
        "label_counts": {
            "0": int((labels == 0).sum().item()),
            "1": int((labels == 1).sum().item()),
        },
        "roc_auc": roc_auc_score_np(labels_np, probs_np),
        "average_precision_pr_auc": average_precision_np(labels_np, probs_np),
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
        "best_f1": {
            "threshold": best_f1_threshold,
            "metrics": best_f1_metrics,
        },
        "best_accuracy": {
            "threshold": best_accuracy_threshold,
            "metrics": best_accuracy_metrics,
        },
    }

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
