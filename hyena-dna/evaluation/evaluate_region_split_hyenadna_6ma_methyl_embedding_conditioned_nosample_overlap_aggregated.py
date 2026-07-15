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


class Dimelo6mAMethylEmbeddingDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        max_length: int,
        c_token_id: int,
        g_token_id: int,
    ) -> None:
        self.path = Path(npz_path)
        self.data = np.load(self.path)
        self.max_length = max_length
        self.c_token_id = int(c_token_id)
        self.g_token_id = int(g_token_id)
        self.n = int(self.data["input_ids"].shape[0])
        self.has_is_cpg = "is_cpg" in self.data.files

    def __len__(self) -> int:
        return self.n

    def _derive_cpg_mask(self, input_ids: np.ndarray) -> np.ndarray:
        is_cpg = np.zeros_like(input_ids, dtype=bool)
        if input_ids.shape[0] > 1:
            is_cpg[:-1] = (input_ids[:-1] == self.c_token_id) & (
                input_ids[1:] == self.g_token_id
            )
        return is_cpg

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end = self.max_length
        input_ids = self.data["input_ids"][idx, :end]
        target_5mc = self.data["target_5mC"][idx, :end].astype(np.float32)
        mask_5mc = self.data["mask_5mC"][idx, :end].astype(bool)
        if self.has_is_cpg:
            is_cpg = self.data["is_cpg"][idx, :end].astype(bool)
        else:
            is_cpg = self._derive_cpg_mask(input_ids)
        return {
            "row_number": torch.tensor(idx, dtype=torch.long),
            "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "methyl_value": torch.as_tensor(target_5mc, dtype=torch.float32),
            "methyl_observed": torch.as_tensor(mask_5mc, dtype=torch.bool),
            "is_cpg": torch.as_tensor(is_cpg, dtype=torch.bool),
            "target_6mA": torch.as_tensor(
                self.data["target_6mA"][idx, :end], dtype=torch.float32
            ),
            "mask_6mA": torch.as_tensor(
                self.data["mask_6mA"][idx, :end], dtype=torch.bool
            ),
        }


class Hyena6mAMethylEmbeddingConditionedNoSample(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        decoder_hidden_dim: int = 0,
        decoder_dropout: float = 0.0,
        unknown_methylation_value: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.unknown_methylation_value = float(unknown_methylation_value)
        self.unmethylated_cpg_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.methylated_cpg_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.non_cpg_embedding = nn.Parameter(torch.zeros(hidden_dim))
        if decoder_hidden_dim > 0:
            self.head_6ma = nn.Sequential(
                nn.Linear(hidden_dim, decoder_hidden_dim),
                nn.GELU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(decoder_hidden_dim, 1),
            )
        else:
            self.head_6ma = nn.Linear(hidden_dim, 1)

    def methylation_embedding(
        self,
        methyl_value: torch.Tensor,
        methyl_observed: torch.Tensor,
        is_cpg: torch.Tensor,
    ) -> torch.Tensor:
        methyl_prob = torch.where(
            methyl_observed,
            methyl_value,
            torch.full_like(methyl_value, self.unknown_methylation_value),
        ).clamp(0.0, 1.0)
        cpg_embedding = (
            (1.0 - methyl_prob).unsqueeze(-1) * self.unmethylated_cpg_embedding
            + methyl_prob.unsqueeze(-1) * self.methylated_cpg_embedding
        )
        non_cpg_embedding = self.non_cpg_embedding.view(1, 1, -1).expand_as(cpg_embedding)
        return torch.where(is_cpg.unsqueeze(-1), cpg_embedding, non_cpg_embedding)

    def forward(
        self,
        input_ids: torch.Tensor,
        methyl_value: torch.Tensor,
        methyl_observed: torch.Tensor,
        is_cpg: torch.Tensor,
    ) -> torch.Tensor:
        lm_backbone = self.backbone.backbone
        hidden_states = lm_backbone.embeddings(input_ids)
        hidden_states = hidden_states + self.methylation_embedding(
            methyl_value, methyl_observed, is_cpg
        ).to(dtype=hidden_states.dtype)
        residual = None
        for layer in lm_backbone.layers:
            hidden_states, residual = layer(hidden_states, residual)
        dropped = lm_backbone.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = lm_backbone.ln_f(residual.to(dtype=lm_backbone.ln_f.weight.dtype))
        return self.head_6ma(hidden_states).squeeze(-1)


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
        description=(
            "Evaluate no-sample P(6mA | DNA, observed 5mC) methylation-embedding "
            "HyenaDNA model with overlap aggregation."
        )
    )
    parser.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/upstream_hyena_dna")
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

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    c_token_id = int(checkpoint.get("c_token_id", checkpoint.get("args", {}).get("c_token_id", 8)))
    g_token_id = int(checkpoint.get("g_token_id", checkpoint.get("args", {}).get("g_token_id", 9)))
    dataset = Dimelo6mAMethylEmbeddingDataset(args.npz, args.max_length, c_token_id, g_token_id)
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

    if not checkpoint.get("methylation_conditioning") == "input_embedding_addition":
        raise SystemExit(f"{args.checkpoint} is not a methylation-embedding checkpoint.")
    decoder_hidden_dim = int(checkpoint.get("decoder_hidden_dim", 0))
    decoder_dropout = float(checkpoint.get("decoder_dropout", 0.0))
    unknown_methylation_value = float(checkpoint.get("unknown_methylation_value", 0.5))
    model = Hyena6mAMethylEmbeddingConditionedNoSample(
        backbone,
        hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_dropout=decoder_dropout,
        unknown_methylation_value=unknown_methylation_value,
    ).to(device)
    if checkpoint.get("trainable_backbone_state_dict"):
        model.backbone.load_state_dict(checkpoint["trainable_backbone_state_dict"], strict=False)
    model.head_6ma.load_state_dict(checkpoint["head_6mA_state_dict"])
    embeddings = checkpoint["methylation_embedding_state_dict"]
    model.unmethylated_cpg_embedding.data.copy_(embeddings["unmethylated_cpg_embedding"].to(device))
    model.methylated_cpg_embedding.data.copy_(embeddings["methylated_cpg_embedding"].to(device))
    model.non_cpg_embedding.data.copy_(embeddings["non_cpg_embedding"].to(device))
    model.eval()

    aggregates: dict[tuple[str, str, int], list[float | int | str]] = {}
    raw_valid_positions = 0
    with torch.inference_mode():
        for batch in loader:
            row_numbers = batch["row_number"].numpy().tolist()
            batch = move_batch(batch, device)
            logits = model(
                batch["input_ids"],
                batch["methyl_value"],
                batch["methyl_observed"],
                batch["is_cpg"],
            )
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
        "model_type": "P(Reg|D,M_embedding)",
        "model_name": args.model_name,
        "device": device,
        "reads": len(dataset),
        "batch_size": args.batch_size,
        "max_length_used": args.max_length,
        "sample_conditioned": False,
        "decoder_hidden_dim": decoder_hidden_dim,
        "decoder_dropout": decoder_dropout,
        "unknown_methylation_value": unknown_methylation_value,
        "c_token_id": c_token_id,
        "g_token_id": g_token_id,
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
