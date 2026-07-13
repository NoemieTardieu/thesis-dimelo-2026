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
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Dimelo6mAMethylConditionedDataset(Dataset):
    def __init__(self, npz_path: str | Path, max_length: int) -> None:
        self.path = Path(npz_path)
        self.data = np.load(self.path)
        self.max_length = max_length
        self.n = int(self.data["input_ids"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end = self.max_length
        target_5mc = self.data["target_5mC"][idx, :end]
        mask_5mc = self.data["mask_5mC"][idx, :end].astype(bool)
        methyl_value = np.where(mask_5mc, target_5mc, 0.0).astype(np.float32)
        methyl_observed = mask_5mc.astype(np.float32)
        return {
            "row_number": torch.tensor(idx, dtype=torch.long),
            "input_ids": torch.as_tensor(self.data["input_ids"][idx, :end], dtype=torch.long),
            "methyl_value": torch.as_tensor(methyl_value, dtype=torch.float32),
            "methyl_observed": torch.as_tensor(methyl_observed, dtype=torch.float32),
            "target_6mA": torch.as_tensor(self.data["target_6mA"][idx, :end], dtype=torch.float32),
            "mask_6mA": torch.as_tensor(self.data["mask_6mA"][idx, :end], dtype=torch.bool),
        }


class Hyena6mAMethylConditionedNoSample(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        methyl_feature_dim: int = 2,
        decoder_hidden_dim: int = 0,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        head_dim = hidden_dim + methyl_feature_dim
        if decoder_hidden_dim > 0:
            self.head_6ma = nn.Sequential(
                nn.Linear(head_dim, decoder_hidden_dim),
                nn.GELU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(decoder_hidden_dim, 1),
            )
        else:
            self.head_6ma = nn.Linear(head_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        methyl_value: torch.Tensor,
        methyl_observed: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(input_ids)
        methyl = torch.stack([methyl_value, methyl_observed], dim=-1)
        conditioned = torch.cat([hidden, methyl], dim=-1)
        return self.head_6ma(conditioned).squeeze(-1)


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


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return pearson_corr(rankdata(x), rankdata(y))


def binary_ranking_metrics(binary: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None]:
    positives = int(binary.sum())
    negatives = int(binary.size - positives)
    if positives == 0 or negatives == 0:
        return None, None
    ranks = rankdata(scores)
    auroc = (float(ranks[binary].sum()) - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    order = np.argsort(-scores, kind="mergesort")
    sorted_binary = binary[order].astype(np.float64)
    tp = np.cumsum(sorted_binary)
    fp = np.cumsum(1.0 - sorted_binary)
    precision = tp / np.maximum(tp + fp, 1.0)
    auprc = float((precision * sorted_binary).sum() / positives)
    return float(auroc), auprc


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
            "auprc_enrichment": None,
        }
    eps = 1e-7
    pred_clip = np.clip(pred, eps, 1.0 - eps)
    binary = target >= threshold
    auroc, auprc = binary_ranking_metrics(binary, pred)
    positive_fraction = float(np.mean(binary))
    return {
        "valid_positions": int(target.size),
        "bce": float(-np.mean(target * np.log(pred_clip) + (1.0 - target) * np.log(1.0 - pred_clip))),
        "mse": float(np.mean((pred - target) ** 2)),
        "mae": float(np.mean(np.abs(pred - target))),
        "mean_pred": float(np.mean(pred)),
        "mean_target": float(np.mean(target)),
        "pearson": pearson_corr(target, pred),
        "spearman": spearman_corr(target, pred),
        "auroc": auroc,
        "auprc": auprc,
        "positive_fraction": positive_fraction,
        "auprc_enrichment": float(auprc / positive_fraction) if auprc is not None and positive_fraction > 0 else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate clean no-sample P(6mA | DNA, observed 5mC) with overlap aggregation."
    )
    parser.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--threshold-6ma", type=float, default=0.5)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


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
    duplicates_removed = 0
    for target_sum, pred_sum, count, region in aggregates.values():
        count_int = int(count)
        target = float(target_sum) / count_int
        pred = float(pred_sum) / count_int
        region_name = str(region)
        targets.append(target)
        preds.append(pred)
        per_region_targets[region_name].append(target)
        per_region_preds[region_name].append(pred)
        duplicates_removed += max(0, count_int - 1)
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
        duplicates_removed,
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.hyena_root)))

    from huggingface import HyenaDNAPreTrainedModel

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    metadata = read_metadata(args.metadata)
    dataset = Dimelo6mAMethylConditionedDataset(args.npz, args.max_length)
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

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("sample_conditioned"):
        raise SystemExit(f"{args.checkpoint} is sample-conditioned; use the clean no-sample checkpoint.")
    decoder_hidden_dim = int(checkpoint.get("decoder_hidden_dim", 0))
    decoder_dropout = float(checkpoint.get("decoder_dropout", 0.0))
    methyl_feature_dim = int(checkpoint.get("methyl_feature_dim", 2))
    model = Hyena6mAMethylConditionedNoSample(
        backbone,
        hidden_dim,
        methyl_feature_dim=methyl_feature_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_dropout=decoder_dropout,
    ).to(device)
    if checkpoint.get("trainable_backbone_state_dict"):
        model.backbone.load_state_dict(checkpoint["trainable_backbone_state_dict"], strict=False)
    model.head_6ma.load_state_dict(checkpoint["head_6mA_state_dict"])
    model.eval()

    aggregates: dict[tuple[str, str, int], list[float | int | str]] = {}
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

    target, pred, per_region, duplicates_removed = materialize_aggregates(aggregates)
    metrics = {"6mA": base_metric_row(target, pred, args.threshold_6ma)}

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_region_path = out_prefix.with_suffix(".per_region_metrics.tsv")
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
        "auprc_enrichment",
    ]
    with open(per_region_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for region in sorted(per_region):
            region_target, region_pred = per_region[region]
            row = {"region": region, "task": "6mA"}
            row.update(base_metric_row(region_target, region_pred, args.threshold_6ma))
            writer.writerow(row)

    summary = {
        "npz": args.npz,
        "metadata": args.metadata,
        "split_name": args.split_name,
        "checkpoint": args.checkpoint,
        "model_type": "P(Reg|D,M)",
        "model_name": args.model_name,
        "device": device,
        "reads": len(dataset),
        "batch_size": args.batch_size,
        "max_length_used": args.max_length,
        "sample_conditioned": False,
        "decoder_hidden_dim": decoder_hidden_dim,
        "decoder_dropout": decoder_dropout,
        "aggregation_unit": "sample/read_id/read_position",
        "thresholds": {"6mA": args.threshold_6ma},
        "aggregation": {
            "6mA": {
                "raw_window_valid_positions": raw_valid_positions,
                "aggregated_read_positions": int(target.size),
                "duplicate_overlap_observations_removed": duplicates_removed,
            }
        },
        "metrics": metrics,
        "per_region_metrics": str(per_region_path),
    }
    with open(out_prefix.with_suffix(".summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
