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
        if "sample_id" not in self.data.files:
            raise SystemExit(f"{npz_path} does not contain sample_id.")
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
            "sample_id": torch.as_tensor(self.data["sample_id"][idx], dtype=torch.long),
            "methyl_value": torch.as_tensor(methyl_value, dtype=torch.float32),
            "methyl_observed": torch.as_tensor(methyl_observed, dtype=torch.float32),
            "target_6mA": torch.as_tensor(self.data["target_6mA"][idx, :end], dtype=torch.float32),
            "mask_6mA": torch.as_tensor(self.data["mask_6mA"][idx, :end], dtype=torch.bool),
        }


class MethylConditionedHyena6mA(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        num_samples: int,
        sample_embedding_dim: int,
        methyl_feature_dim: int = 2,
        decoder_hidden_dim: int = 0,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.sample_embedding = nn.Embedding(num_samples, sample_embedding_dim)
        head_dim = hidden_dim + sample_embedding_dim + methyl_feature_dim
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
        sample_id: torch.Tensor,
        methyl_value: torch.Tensor,
        methyl_observed: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(input_ids)
        sample = self.sample_embedding(sample_id)
        sample = sample[:, None, :].expand(-1, hidden.shape[1], -1)
        methyl = torch.stack([methyl_value, methyl_observed], dim=-1)
        conditioned = torch.cat([hidden, sample, methyl], dim=-1)
        return self.head_6ma(conditioned).squeeze(-1)


def read_metadata(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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
            "auprc_enrichment": None,
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
    positive_fraction = float(np.mean(binary))
    auroc = None
    auprc = None
    auprc_enrichment = None
    if np.unique(binary).size == 2:
        auroc = float(roc_auc_score(binary, pred))
        auprc = float(average_precision_score(binary, pred))
        if positive_fraction > 0:
            auprc_enrichment = float(auprc / positive_fraction)

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
        "positive_fraction": positive_fraction,
        "auprc_enrichment": auprc_enrichment,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extended metrics for P(Reg | D,C,M) HyenaDNA checkpoint."
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
    p.add_argument("--threshold-6ma", type=float, default=0.5)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return p.parse_args()


def append_metrics(
    targets_store: dict[str, list[np.ndarray]],
    preds_store: dict[str, list[np.ndarray]],
    target: np.ndarray,
    pred: np.ndarray,
) -> None:
    if target.size:
        targets_store["6mA"].append(target)
        preds_store["6mA"].append(pred)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.hyena_root)))

    from huggingface import HyenaDNAPreTrainedModel

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

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

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not checkpoint.get("methylation_conditioned"):
        raise SystemExit(f"{args.checkpoint} is not a methylation-conditioned checkpoint.")
    num_samples = int(checkpoint.get("num_samples", 2))
    sample_embedding_dim = int(checkpoint.get("sample_embedding_dim", 16))
    methyl_feature_dim = int(checkpoint.get("methyl_feature_dim", 2))
    decoder_hidden_dim = int(checkpoint.get("decoder_hidden_dim", 0))
    decoder_dropout = float(checkpoint.get("decoder_dropout", 0.0))

    model = MethylConditionedHyena6mA(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_samples=num_samples,
        sample_embedding_dim=sample_embedding_dim,
        methyl_feature_dim=methyl_feature_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_dropout=decoder_dropout,
    ).to(device)
    if checkpoint.get("trainable_backbone_state_dict"):
        model.backbone.load_state_dict(checkpoint["trainable_backbone_state_dict"], strict=False)
    model.sample_embedding.load_state_dict(checkpoint["sample_embedding_state_dict"])
    model.head_6ma.load_state_dict(checkpoint["head_6mA_state_dict"])
    model.eval()

    all_targets = {"6mA": []}
    all_preds = {"6mA": []}
    per_region_targets: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"6mA": []})
    per_region_preds: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"6mA": []})
    per_sample_targets: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"6mA": []})
    per_sample_preds: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"6mA": []})
    per_chrom_targets: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"6mA": []})
    per_chrom_preds: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"6mA": []})

    batches_evaluated = 0
    reads_evaluated = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            if args.max_batches is not None and batch_idx > args.max_batches:
                break
            row_numbers = batch["row_number"].numpy().tolist()
            batch = move_batch(batch, device)
            logits = model(
                batch["input_ids"],
                batch["sample_id"],
                batch["methyl_value"],
                batch["methyl_observed"],
            )
            batches_evaluated += 1
            reads_evaluated += len(row_numbers)
            if args.progress_every and batches_evaluated % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "progress": "eval",
                            "split": args.split_name,
                            "batches_evaluated": batches_evaluated,
                            "reads_evaluated": reads_evaluated,
                        }
                    ),
                    flush=True,
                )

            pred_6ma = torch.sigmoid(logits).detach().cpu().numpy()
            target_6ma = batch["target_6mA"].detach().cpu().numpy()
            mask_6ma = batch["mask_6mA"].detach().cpu().numpy().astype(bool)

            for i, row_number in enumerate(row_numbers):
                meta = metadata[row_number]
                region = meta.get("region_name") or meta.get("region_id") or "unknown"
                sample = meta.get("sample") or meta.get("sample_id") or str(int(batch["sample_id"][i]))
                chrom = meta.get("chrom") or "unknown"

                target = target_6ma[i][mask_6ma[i]]
                pred = pred_6ma[i][mask_6ma[i]]
                append_metrics(all_targets, all_preds, target, pred)
                append_metrics(per_region_targets[region], per_region_preds[region], target, pred)
                append_metrics(per_sample_targets[sample], per_sample_preds[sample], target, pred)
                append_metrics(per_chrom_targets[chrom], per_chrom_preds[chrom], target, pred)

    target = np.concatenate(all_targets["6mA"]) if all_targets["6mA"] else np.asarray([])
    pred = np.concatenate(all_preds["6mA"]) if all_preds["6mA"] else np.asarray([])
    metrics = {"6mA": base_metric_row(target, pred, args.threshold_6ma)}

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_region_path = out_prefix.with_suffix(".per_region_metrics.tsv")
    per_sample_path = out_prefix.with_suffix(".per_sample_metrics.tsv")
    per_chrom_path = out_prefix.with_suffix(".per_chrom_metrics.tsv")

    fieldnames = [
        "group",
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
    for path, grouped_targets, grouped_preds in [
        (per_region_path, per_region_targets, per_region_preds),
        (per_sample_path, per_sample_targets, per_sample_preds),
        (per_chrom_path, per_chrom_targets, per_chrom_preds),
    ]:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for group in sorted(grouped_targets):
                target = (
                    np.concatenate(grouped_targets[group]["6mA"])
                    if grouped_targets[group]["6mA"]
                    else np.asarray([])
                )
                pred = (
                    np.concatenate(grouped_preds[group]["6mA"])
                    if grouped_preds[group]["6mA"]
                    else np.asarray([])
                )
                row = {"group": group, "task": "6mA"}
                row.update(base_metric_row(target, pred, args.threshold_6ma))
                writer.writerow(row)

    summary = {
        "npz": args.npz,
        "metadata": args.metadata,
        "split_name": args.split_name,
        "checkpoint": args.checkpoint,
        "model_type": "P(Reg|D,C,M)",
        "model_name": args.model_name,
        "device": device,
        "reads": len(dataset),
        "reads_evaluated": reads_evaluated,
        "batches_evaluated": batches_evaluated,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "max_length_used": args.max_length,
        "num_samples": num_samples,
        "sample_embedding_dim": sample_embedding_dim,
        "decoder_hidden_dim": decoder_hidden_dim,
        "decoder_dropout": decoder_dropout,
        "methylation_features": ["target_5mC_filled_zero", "mask_5mC_observed"],
        "thresholds": {"6mA": args.threshold_6ma},
        "metrics": metrics,
        "per_region_metrics": str(per_region_path),
        "per_sample_metrics": str(per_sample_path),
        "per_chrom_metrics": str(per_chrom_path),
    }
    with open(out_prefix.with_suffix(".summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
