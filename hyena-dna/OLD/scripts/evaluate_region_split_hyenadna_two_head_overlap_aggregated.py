#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DimeloTensorDataset(Dataset):
    def __init__(self, npz_path: str | Path, max_length: int) -> None:
        self.path = Path(npz_path)
        self.data = np.load(self.path)
        self.max_length = max_length
        self.n = int(self.data["input_ids"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end = self.max_length
        return {
            "row_number": torch.tensor(idx, dtype=torch.long),
            "input_ids": torch.as_tensor(self.data["input_ids"][idx, :end], dtype=torch.long),
            "target_5mC": torch.as_tensor(self.data["target_5mC"][idx, :end], dtype=torch.float32),
            "mask_5mC": torch.as_tensor(self.data["mask_5mC"][idx, :end], dtype=torch.bool),
            "target_6mA": torch.as_tensor(self.data["target_6mA"][idx, :end], dtype=torch.float32),
            "mask_6mA": torch.as_tensor(self.data["mask_6mA"][idx, :end], dtype=torch.bool),
        }


class HyenaTwoHead(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head_5mc = nn.Linear(hidden_dim, 1)
        self.head_6ma = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(input_ids)
        return {
            "logits_5mC": self.head_5mc(hidden).squeeze(-1),
            "logits_6mA": self.head_6ma(hidden).squeeze(-1),
        }


def read_metadata(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"read_id", "window_start"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise SystemExit(f"Metadata is missing required columns: {sorted(missing)}")
    return rows


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if key != "row_number" else value
        for key, value in batch.items()
    }


def base_metric_row(target: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    if target.size == 0:
        return {
            "valid_positions": 0,
            "bce": None,
            "mse": None,
            "mae": None,
            "mean_pred": None,
            "mean_target": None,
            "pearson": None,
            "spearman": None,
            "auroc": None,
            "auprc": None,
            "positive_fraction": None,
        }

    eps = 1e-7
    pred_clip = np.clip(pred, eps, 1.0 - eps)
    bce = -np.mean(target * np.log(pred_clip) + (1.0 - target) * np.log(1.0 - pred_clip))
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))

    pearson = None
    spearman = None
    if target.size >= 2 and np.std(target) > 0 and np.std(pred) > 0:
        pearson = float(pearsonr(target, pred).statistic)
        spearman = float(spearmanr(target, pred).statistic)

    binary = target >= threshold
    auroc = None
    auprc = None
    if np.unique(binary).size == 2:
        auroc = float(roc_auc_score(binary, pred))
        auprc = float(average_precision_score(binary, pred))

    return {
        "valid_positions": int(target.size),
        "bce": float(bce),
        "mse": float(mse),
        "mae": float(mae),
        "mean_pred": float(np.mean(pred)),
        "mean_target": float(np.mean(target)),
        "pearson": pearson,
        "spearman": spearman,
        "auroc": auroc,
        "auprc": auprc,
        "positive_fraction": float(np.mean(binary)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a no-sample HyenaDNA two-head checkpoint after aggregating "
            "overlapping windows from the same original read."
        )
    )
    p.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    p.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    p.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    p.add_argument("--npz", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split-name", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--threshold-5mc", type=float, default=0.5)
    p.add_argument("--threshold-6ma", type=float, default=0.5)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return p.parse_args()


def add_observations(
    aggregates: dict[tuple[str, str, int], list[float | int | str]],
    row_meta: dict[str, str],
    local_positions: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
) -> None:
    sample = row_meta.get("sample") or row_meta.get("sample_id") or "unknown"
    read_id = row_meta["read_id"]
    region = row_meta.get("region_name") or row_meta.get("region_id") or "unknown"
    window_start = int(row_meta["window_start"])

    for local_pos, target, pred in zip(local_positions, targets, preds):
        read_pos = window_start + int(local_pos)
        key = (sample, read_id, read_pos)
        if key not in aggregates:
            aggregates[key] = [0.0, 0.0, 0, region]
        aggregates[key][0] += float(target)
        aggregates[key][1] += float(pred)
        aggregates[key][2] += 1


def materialize_aggregates(
    aggregates: dict[tuple[str, str, int], list[float | int | str]]
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]], int]:
    targets = []
    preds = []
    per_region_targets: dict[str, list[float]] = defaultdict(list)
    per_region_preds: dict[str, list[float]] = defaultdict(list)
    duplicate_observations_removed = 0

    for target_sum, pred_sum, count, region in aggregates.values():
        count_int = int(count)
        target = float(target_sum) / count_int
        pred = float(pred_sum) / count_int
        region_name = str(region)
        targets.append(target)
        preds.append(pred)
        per_region_targets[region_name].append(target)
        per_region_preds[region_name].append(pred)
        duplicate_observations_removed += max(0, count_int - 1)

    per_region = {
        region: (
            np.asarray(per_region_targets[region], dtype=np.float32),
            np.asarray(per_region_preds[region], dtype=np.float32),
        )
        for region in per_region_targets
    }
    return (
        np.asarray(targets, dtype=np.float32),
        np.asarray(preds, dtype=np.float32),
        per_region,
        duplicate_observations_removed,
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.hyena_root)))

    from huggingface import HyenaDNAPreTrainedModel

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    metadata = read_metadata(args.metadata)
    dataset = DimeloTensorDataset(args.npz, args.max_length)
    if len(metadata) != len(dataset):
        raise SystemExit(f"Metadata rows ({len(metadata)}) != tensor rows ({len(dataset)})")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

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
        hidden_probe = backbone(probe["input_ids"])
    hidden_dim = int(hidden_probe.shape[-1])

    model = HyenaTwoHead(backbone, hidden_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("trainable_backbone_state_dict"):
        model.backbone.load_state_dict(checkpoint["trainable_backbone_state_dict"], strict=False)
    model.head_5mc.load_state_dict(checkpoint["head_5mC_state_dict"])
    model.head_6ma.load_state_dict(checkpoint["head_6mA_state_dict"])
    model.eval()

    aggregates = {"5mC": {}, "6mA": {}}
    raw_valid_positions = {"5mC": 0, "6mA": 0}

    with torch.inference_mode():
        for batch in loader:
            row_numbers = batch["row_number"].numpy().tolist()
            batch = move_batch(batch, device)
            outputs = model(batch["input_ids"])

            pred_5mc = torch.sigmoid(outputs["logits_5mC"]).detach().cpu().numpy()
            pred_6ma = torch.sigmoid(outputs["logits_6mA"]).detach().cpu().numpy()
            target_5mc = batch["target_5mC"].detach().cpu().numpy()
            target_6ma = batch["target_6mA"].detach().cpu().numpy()
            mask_5mc = batch["mask_5mC"].detach().cpu().numpy().astype(bool)
            mask_6ma = batch["mask_6mA"].detach().cpu().numpy().astype(bool)

            for i, row_number in enumerate(row_numbers):
                row_meta = metadata[row_number]

                positions_5mc = np.flatnonzero(mask_5mc[i])
                if positions_5mc.size:
                    raw_valid_positions["5mC"] += int(positions_5mc.size)
                    add_observations(
                        aggregates["5mC"],
                        row_meta,
                        positions_5mc,
                        target_5mc[i, positions_5mc],
                        pred_5mc[i, positions_5mc],
                    )

                positions_6ma = np.flatnonzero(mask_6ma[i])
                if positions_6ma.size:
                    raw_valid_positions["6mA"] += int(positions_6ma.size)
                    add_observations(
                        aggregates["6mA"],
                        row_meta,
                        positions_6ma,
                        target_6ma[i, positions_6ma],
                        pred_6ma[i, positions_6ma],
                    )

    metrics = {}
    per_region_materialized = {}
    aggregation = {}
    for task, threshold in [("5mC", args.threshold_5mc), ("6mA", args.threshold_6ma)]:
        target, pred, per_region, duplicates_removed = materialize_aggregates(aggregates[task])
        metrics[task] = base_metric_row(target, pred, threshold)
        per_region_materialized[task] = per_region
        aggregation[task] = {
            "raw_window_valid_positions": raw_valid_positions[task],
            "aggregated_read_positions": int(target.size),
            "duplicate_overlap_observations_removed": int(duplicates_removed),
        }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_region_path = out_prefix.with_suffix(".per_region_metrics.tsv")

    with open(per_region_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "region",
            "task",
            "valid_positions",
            "bce",
            "mse",
            "mae",
            "mean_pred",
            "mean_target",
            "pearson",
            "spearman",
            "auroc",
            "auprc",
            "positive_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        regions = sorted(
            set(per_region_materialized["5mC"]) | set(per_region_materialized["6mA"])
        )
        for region in regions:
            for task, threshold in [("5mC", args.threshold_5mc), ("6mA", args.threshold_6ma)]:
                if region in per_region_materialized[task]:
                    target, pred = per_region_materialized[task][region]
                else:
                    target = np.asarray([])
                    pred = np.asarray([])
                row = {"region": region, "task": task}
                row.update(base_metric_row(target, pred, threshold))
                writer.writerow(row)

    summary = {
        "npz": args.npz,
        "metadata": args.metadata,
        "split_name": args.split_name,
        "checkpoint": args.checkpoint,
        "model_name": args.model_name,
        "device": device,
        "reads": len(dataset),
        "batch_size": args.batch_size,
        "max_length_used": args.max_length,
        "aggregation_unit": "sample/read_id/read_position",
        "thresholds": {"5mC": args.threshold_5mc, "6mA": args.threshold_6ma},
        "aggregation": aggregation,
        "metrics": metrics,
        "per_region_metrics": str(per_region_path),
    }
    with open(out_prefix.with_suffix(".summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
