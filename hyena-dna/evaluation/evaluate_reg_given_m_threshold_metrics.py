#!/usr/bin/env python3
"""Threshold-dependent metrics for P(Reg|D,M).

The main evaluation reports threshold-free ranking metrics such as AUROC and
AUPRC. This helper adds binary-call metrics by selecting the prediction
threshold that maximizes F1 on validation, then applying that same threshold to
test. Targets are binarized with the usual 6mA label threshold, default 0.5.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_region_split_hyenadna_6ma_methyl_conditioned_nosample_overlap_aggregated import (
    Dimelo6mAMethylConditionedDataset,
    Hyena6mAMethylConditionedNoSample,
    add_observations,
    materialize_aggregates,
    move_batch,
    read_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--val-metadata", required=True)
    parser.add_argument("--test-npz", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--label-threshold-6ma", type=float, default=0.5)
    parser.add_argument("--threshold-grid-size", type=int, default=1001)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def load_model(args: argparse.Namespace, example_dataset: Dimelo6mAMethylConditionedDataset):
    sys.path.insert(0, str(Path(args.hyena_root)))
    from huggingface import HyenaDNAPreTrainedModel

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    loader = DataLoader(example_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    backbone = HyenaDNAPreTrainedModel.from_pretrained(
        args.checkpoint_dir,
        args.model_name,
        download=False,
        device=device,
        use_head=False,
    ).to(device)
    backbone.eval()
    probe = move_batch(next(iter(loader)), device)
    with torch.inference_mode():
        hidden_dim = int(backbone(probe["input_ids"]).shape[-1])

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("sample_conditioned"):
        raise SystemExit(f"{args.checkpoint} is sample-conditioned; use a no-sample checkpoint.")
    model = Hyena6mAMethylConditionedNoSample(
        backbone,
        hidden_dim,
        methyl_feature_dim=int(checkpoint.get("methyl_feature_dim", 2)),
        decoder_hidden_dim=int(checkpoint.get("decoder_hidden_dim", 0)),
        decoder_dropout=float(checkpoint.get("decoder_dropout", 0.0)),
    ).to(device)
    if checkpoint.get("trainable_backbone_state_dict"):
        model.backbone.load_state_dict(checkpoint["trainable_backbone_state_dict"], strict=False)
    model.head_6ma.load_state_dict(checkpoint["head_6mA_state_dict"])
    model.eval()
    return model, device, checkpoint


def collect_overlap_aggregated_predictions(
    npz_path: str,
    metadata_path: str,
    model,
    device: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    metadata = read_metadata(metadata_path)
    dataset = Dimelo6mAMethylConditionedDataset(npz_path, args.max_length)
    if len(metadata) != len(dataset):
        raise SystemExit(f"Metadata rows ({len(metadata)}) != tensor rows ({len(dataset)})")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    aggregates = {}
    raw_valid_positions = 0
    with torch.inference_mode():
        for batch in loader:
            row_numbers = batch["row_number"].numpy().tolist()
            batch = move_batch(batch, device)
            logits = model(batch["input_ids"], batch["methyl_value"], batch["methyl_observed"])
            pred = torch.sigmoid(logits).detach().cpu().numpy()
            target = batch["target_6mA"].detach().cpu().numpy()
            mask = batch["mask_6mA"].detach().cpu().numpy().astype(bool)
            for i, row_number in enumerate(row_numbers):
                positions = np.flatnonzero(mask[i])
                if positions.size:
                    raw_valid_positions += int(positions.size)
                    add_observations(
                        aggregates,
                        metadata[row_number],
                        positions,
                        target[i][positions],
                        pred[i][positions],
                    )
    target, pred, _, duplicates_removed = materialize_aggregates(aggregates)
    stats = {
        "reads": len(dataset),
        "raw_window_valid_positions": raw_valid_positions,
        "aggregated_read_positions": int(target.size),
        "duplicate_overlap_observations_removed": int(duplicates_removed),
    }
    return target, pred, stats


def binary_metrics(target: np.ndarray, pred: np.ndarray, label_threshold: float, pred_threshold: float) -> dict[str, float | int]:
    y_true = target >= label_threshold
    y_pred = pred >= pred_threshold
    tp = int(np.logical_and(y_true, y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())
    tn = int(np.logical_and(~y_true, ~y_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(pred_threshold),
        "n": int(target.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "predicted_positive_fraction": float(np.mean(y_pred)) if target.size else 0.0,
        "true_positive_fraction": float(np.mean(y_true)) if target.size else 0.0,
    }


def scan_thresholds(target: np.ndarray, pred: np.ndarray, args: argparse.Namespace) -> list[dict[str, float | int]]:
    thresholds = np.linspace(0.0, 1.0, args.threshold_grid_size)
    return [binary_metrics(target, pred, args.label_threshold_6ma, float(t)) for t in thresholds]


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    val_dataset = Dimelo6mAMethylConditionedDataset(args.val_npz, args.max_length)
    model, device, checkpoint = load_model(args, val_dataset)
    val_target, val_pred, val_stats = collect_overlap_aggregated_predictions(
        args.val_npz, args.val_metadata, model, device, args
    )
    test_target, test_pred, test_stats = collect_overlap_aggregated_predictions(
        args.test_npz, args.test_metadata, model, device, args
    )

    val_scan = scan_thresholds(val_target, val_pred, args)
    best = max(
        val_scan,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["recall"]),
            -abs(float(row["predicted_positive_fraction"]) - float(row["true_positive_fraction"])),
        ),
    )
    chosen_threshold = float(best["threshold"])
    val_at_best = binary_metrics(val_target, val_pred, args.label_threshold_6ma, chosen_threshold)
    test_at_val_threshold = binary_metrics(test_target, test_pred, args.label_threshold_6ma, chosen_threshold)
    test_at_05 = binary_metrics(test_target, test_pred, args.label_threshold_6ma, 0.5)

    scan_path = out_prefix.with_suffix(".val_threshold_scan.tsv")
    with scan_path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(val_scan[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(val_scan)

    summary = {
        "checkpoint": args.checkpoint,
        "model_type": checkpoint.get("model_type", "P(Reg|D,M)"),
        "sample_conditioned": bool(checkpoint.get("sample_conditioned", False)),
        "decoder_hidden_dim": int(checkpoint.get("decoder_hidden_dim", 0)),
        "decoder_dropout": float(checkpoint.get("decoder_dropout", 0.0)),
        "label_threshold_6ma": args.label_threshold_6ma,
        "chosen_prediction_threshold": chosen_threshold,
        "validation": {"aggregation": val_stats, "metrics_at_chosen_threshold": val_at_best},
        "test": {
            "aggregation": test_stats,
            "metrics_at_validation_threshold": test_at_val_threshold,
            "metrics_at_fixed_0.5_threshold": test_at_05,
        },
        "outputs": {"validation_threshold_scan": str(scan_path)},
    }
    summary_path = out_prefix.with_suffix(".threshold_metrics.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    compact_path = out_prefix.with_suffix(".threshold_metrics.tsv")
    rows = [
        {"split": "val", "threshold_source": "best_on_val", **val_at_best},
        {"split": "test", "threshold_source": "best_on_val", **test_at_val_threshold},
        {"split": "test", "threshold_source": "fixed_0.5", **test_at_05},
    ]
    with compact_path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
