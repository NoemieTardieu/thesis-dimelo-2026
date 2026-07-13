#!/usr/bin/env python3
"""Plot ROC and precision-recall curves for P(Reg|D,M).

Predictions are generated with the same read-overlap aggregation used by the
main evaluator. Targets are binarized as target_6mA >= label_threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_reg_given_m_threshold_metrics import (
    Dimelo6mAMethylConditionedDataset,
    collect_overlap_aggregated_predictions,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", nargs=3, action="append", metavar=("LABEL", "NPZ", "METADATA"), required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--label-threshold-6ma", type=float, default=0.5)
    parser.add_argument("--max-curve-points", type=int, default=5000)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def curve_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return idx


def trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility replacement for np.trapz/np.trapezoid."""
    if x.size < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def roc_pr_curves(target: np.ndarray, pred: np.ndarray, label_threshold: float, max_points: int) -> dict[str, object]:
    y = target >= label_threshold
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Need both positive and negative labels to compute ROC/PR curves.")

    order = np.argsort(-pred, kind="mergesort")
    y_sorted = y[order].astype(np.float64)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)

    tpr = tp / positives
    fpr = fp / negatives
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tpr.copy()

    roc_fpr = np.r_[0.0, fpr, 1.0]
    roc_tpr = np.r_[0.0, tpr, 1.0]
    auroc = trapezoid_area(roc_tpr, roc_fpr)

    # Average precision style AUPRC, matching the project evaluator.
    auprc = float((precision * y_sorted).sum() / positives)

    youden = tpr - fpr
    roc_best_idx = int(np.argmax(youden))
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    pr_best_idx = int(np.argmax(f1))

    roc_keep = curve_indices(roc_fpr.size, max_points)
    pr_keep = curve_indices(recall.size, max_points)

    return {
        "positive_fraction": float(y.mean()),
        "positives": positives,
        "negatives": negatives,
        "auroc": auroc,
        "auprc": auprc,
        "roc": pd.DataFrame({"fpr": roc_fpr[roc_keep], "tpr": roc_tpr[roc_keep]}),
        "pr": pd.DataFrame({"recall": recall[pr_keep], "precision": precision[pr_keep]}),
        "roc_operating_point": {
            "fpr": float(fpr[roc_best_idx]),
            "tpr": float(tpr[roc_best_idx]),
            "threshold": float(pred[order][roc_best_idx]),
        },
        "pr_operating_point": {
            "recall": float(recall[pr_best_idx]),
            "precision": float(precision[pr_best_idx]),
            "f1": float(f1[pr_best_idx]),
            "threshold": float(pred[order][pr_best_idx]),
        },
    }


def plot_individual(label: str, result: dict[str, object], out_prefix: Path) -> None:
    roc = result["roc"]
    pr = result["pr"]
    roc_point = result["roc_operating_point"]
    pr_point = result["pr_operating_point"]
    pos_frac = float(result["positive_fraction"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(roc["fpr"], roc["tpr"], color="#f59e0b", lw=2.5, label=f"AUROC = {result['auroc']:.3f}")
    ax.fill_between(roc["fpr"], roc["tpr"], 0, color="#f59e0b", alpha=0.18)
    ax.plot([0, 1], [0, 1], "k--", lw=1.6, label="Random classifier")
    ax.scatter([roc_point["fpr"]], [roc_point["tpr"]], color="#ef233c", s=55, zorder=4)
    ax.annotate(
        f"({roc_point['fpr']:.2f}, {roc_point['tpr']:.2f})",
        (roc_point["fpr"], roc_point["tpr"]),
        textcoords="offset points",
        xytext=(10, -4),
        color="#ef233c",
        fontsize=10,
    )
    ax.set_title(f"{label}: ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=True, loc="lower right")

    ax = axes[1]
    ax.plot(pr["recall"], pr["precision"], color="#2563eb", lw=2.5, label=f"AUPRC = {result['auprc']:.3f}")
    ax.fill_between(pr["recall"], pr["precision"], pos_frac, color="#2563eb", alpha=0.15)
    ax.axhline(pos_frac, color="k", ls="--", lw=1.6, label=f"Random baseline = {pos_frac:.3f}")
    ax.scatter([pr_point["recall"]], [pr_point["precision"]], color="#ef233c", s=55, zorder=4)
    ax.annotate(
        f"F1={pr_point['f1']:.2f}",
        (pr_point["recall"], pr_point["precision"]),
        textcoords="offset points",
        xytext=(10, -4),
        color="#ef233c",
        fontsize=10,
    )
    ax.set_title(f"{label}: precision-recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.35, min(1.02, float(pr["precision"].max()) + 0.05)))
    ax.legend(frameon=True, loc="upper right")

    fig.suptitle("P(Reg|D,M) overlap-aggregated 6mA classification curves", fontsize=13)
    safe = label.replace("/", "_").replace(" ", "_")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".{safe}.roc_pr.{ext}"), dpi=300)
    plt.close(fig)


def plot_combined(results: dict[str, dict[str, object]], out_prefix: Path) -> None:
    colors = ["#48639c", "#4c956c", "#d08c60", "#9d4edd", "#2563eb", "#ef476f"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    ax = axes[0]
    for (label, result), color in zip(results.items(), colors):
        roc = result["roc"]
        ax.plot(roc["fpr"], roc["tpr"], lw=2.1, color=color, label=f"{label} ({result['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_title("ROC curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=True, fontsize=8, loc="lower right")

    ax = axes[1]
    for (label, result), color in zip(results.items(), colors):
        pr = result["pr"]
        ax.plot(pr["recall"], pr["precision"], lw=2.1, color=color, label=f"{label} ({result['auprc']:.3f})")
        ax.axhline(float(result["positive_fraction"]), color=color, ls=":", lw=1.0, alpha=0.8)
    ax.set_title("Precision-recall curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.35)
    ax.legend(frameon=True, fontsize=8, loc="upper right")

    fig.suptitle("Cross-chromosome P(Reg|D,M) 6mA classification curves", fontsize=13)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".combined_roc_pr.{ext}"), dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    first_dataset = Dimelo6mAMethylConditionedDataset(args.dataset[0][1], args.max_length)
    model, device, _ = load_model(args, first_dataset)

    results = {}
    summary_rows = []
    for label, npz, metadata in args.dataset:
        target, pred, agg = collect_overlap_aggregated_predictions(npz, metadata, model, device, args)
        result = roc_pr_curves(target, pred, args.label_threshold_6ma, args.max_curve_points)
        results[label] = result
        plot_individual(label, result, out_prefix)
        summary_rows.append(
            {
                "label": label,
                **agg,
                "positives": result["positives"],
                "negatives": result["negatives"],
                "positive_fraction": result["positive_fraction"],
                "auroc": result["auroc"],
                "auprc": result["auprc"],
                "auprc_enrichment": result["auprc"] / result["positive_fraction"],
                "roc_operating_threshold": result["roc_operating_point"]["threshold"],
                "roc_operating_fpr": result["roc_operating_point"]["fpr"],
                "roc_operating_tpr": result["roc_operating_point"]["tpr"],
                "pr_operating_threshold": result["pr_operating_point"]["threshold"],
                "pr_operating_precision": result["pr_operating_point"]["precision"],
                "pr_operating_recall": result["pr_operating_point"]["recall"],
                "pr_operating_f1": result["pr_operating_point"]["f1"],
            }
        )

    plot_combined(results, out_prefix)
    summary = pd.DataFrame(summary_rows)
    summary_path = out_prefix.with_suffix(".curve_summary.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    json_path = out_prefix.with_suffix(".curve_summary.json")
    json_path.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
