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
        return {
            "row_number": torch.tensor(idx, dtype=torch.long),
            "input_ids": torch.as_tensor(self.data["input_ids"][idx, :end], dtype=torch.long),
            "sample_id": torch.as_tensor(self.data["sample_id"][idx], dtype=torch.long),
            "target_5mC": torch.as_tensor(self.data["target_5mC"][idx, :end], dtype=torch.float32),
            "mask_5mC": torch.as_tensor(self.data["mask_5mC"][idx, :end], dtype=torch.bool),
            "target_6mA": torch.as_tensor(self.data["target_6mA"][idx, :end], dtype=torch.float32),
            "mask_6mA": torch.as_tensor(self.data["mask_6mA"][idx, :end], dtype=torch.bool),
        }


class ResidualConvDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.gate = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        residual = x
        y = self.norm(x)
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        value, gate = self.gate(y).chunk(2, dim=-1)
        y = value * torch.sigmoid(gate)
        return residual + self.dropout(y)


class SampleConditionedHyenaTwoHead(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        num_samples: int,
        sample_embedding_dim: int,
        decoder_type: str = "auto",
        decoder_hidden_dim: int = 0,
        decoder_dropout: float = 0.0,
        decoder_kernel_size: int = 9,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.sample_embedding = nn.Embedding(num_samples, sample_embedding_dim)
        head_dim = hidden_dim + sample_embedding_dim
        if decoder_type == "auto":
            decoder_type = "mlp" if decoder_hidden_dim > 0 else "linear"
        if decoder_type == "linear":
            self.shared_decoder = nn.Identity()
            decoder_out_dim = head_dim
        elif decoder_type == "mlp":
            self.shared_decoder = nn.Sequential(
                nn.Linear(head_dim, decoder_hidden_dim),
                nn.GELU(),
                nn.Dropout(decoder_dropout),
            )
            decoder_out_dim = decoder_hidden_dim
        elif decoder_type == "conv":
            self.shared_decoder = ResidualConvDecoder(
                input_dim=head_dim,
                hidden_dim=decoder_hidden_dim,
                dropout=decoder_dropout,
                kernel_size=decoder_kernel_size,
            )
            decoder_out_dim = decoder_hidden_dim
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
        self.head_5mc = nn.Linear(decoder_out_dim, 1)
        self.head_6ma = nn.Linear(decoder_out_dim, 1)

    def forward(self, input_ids: torch.Tensor, sample_id: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(input_ids)
        sample = self.sample_embedding(sample_id)
        sample = sample[:, None, :].expand(-1, hidden.shape[1], -1)
        conditioned = torch.cat([hidden, sample], dim=-1)
        decoded = self.shared_decoder(conditioned)
        return {
            "logits_5mC": self.head_5mc(decoded).squeeze(-1),
            "logits_6mA": self.head_6ma(decoded).squeeze(-1),
        }


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
        description="Extended metrics for sample-conditioned HyenaDNA two-head checkpoint."
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
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Evaluate at most this many batches. Useful for large smoke/partial evaluations.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N evaluated batches. Use 0 to disable.",
    )
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return p.parse_args()


def append_metrics(
    targets_store: dict[str, list[np.ndarray]],
    preds_store: dict[str, list[np.ndarray]],
    task: str,
    target: np.ndarray,
    pred: np.ndarray,
) -> None:
    if target.size:
        targets_store[task].append(target)
        preds_store[task].append(pred)


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
    num_samples = int(checkpoint.get("num_samples", 2))
    sample_embedding_dim = int(checkpoint.get("sample_embedding_dim", 16))
    decoder_type = str(checkpoint.get("decoder_type", "auto"))
    decoder_hidden_dim = int(checkpoint.get("decoder_hidden_dim", 0))
    decoder_dropout = float(checkpoint.get("decoder_dropout", 0.0))
    decoder_kernel_size = int(checkpoint.get("decoder_kernel_size", 9))

    model = SampleConditionedHyenaTwoHead(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_samples=num_samples,
        sample_embedding_dim=sample_embedding_dim,
        decoder_type=decoder_type,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_dropout=decoder_dropout,
        decoder_kernel_size=decoder_kernel_size,
    ).to(device)
    if checkpoint.get("trainable_backbone_state_dict"):
        model.backbone.load_state_dict(checkpoint["trainable_backbone_state_dict"], strict=False)
    model.sample_embedding.load_state_dict(checkpoint["sample_embedding_state_dict"])
    if checkpoint.get("shared_decoder_state_dict") is not None:
        model.shared_decoder.load_state_dict(checkpoint["shared_decoder_state_dict"])
    model.head_5mc.load_state_dict(checkpoint["head_5mC_state_dict"])
    model.head_6ma.load_state_dict(checkpoint["head_6mA_state_dict"])
    model.eval()

    all_targets = {"5mC": [], "6mA": []}
    all_preds = {"5mC": [], "6mA": []}
    per_region_targets: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"5mC": [], "6mA": []})
    per_region_preds: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"5mC": [], "6mA": []})
    per_sample_targets: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"5mC": [], "6mA": []})
    per_sample_preds: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"5mC": [], "6mA": []})
    per_chrom_targets: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"5mC": [], "6mA": []})
    per_chrom_preds: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"5mC": [], "6mA": []})

    batches_evaluated = 0
    reads_evaluated = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            if args.max_batches is not None and batch_idx > args.max_batches:
                break
            row_numbers = batch["row_number"].numpy().tolist()
            batch = move_batch(batch, device)
            outputs = model(batch["input_ids"], batch["sample_id"])
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

            pred_5mc = torch.sigmoid(outputs["logits_5mC"]).detach().cpu().numpy()
            pred_6ma = torch.sigmoid(outputs["logits_6mA"]).detach().cpu().numpy()
            target_5mc = batch["target_5mC"].detach().cpu().numpy()
            target_6ma = batch["target_6mA"].detach().cpu().numpy()
            mask_5mc = batch["mask_5mC"].detach().cpu().numpy().astype(bool)
            mask_6ma = batch["mask_6mA"].detach().cpu().numpy().astype(bool)

            for i, row_number in enumerate(row_numbers):
                meta = metadata[row_number]
                region = meta.get("region_name") or meta.get("region_id") or "unknown"
                sample = meta.get("sample") or meta.get("sample_id") or str(int(batch["sample_id"][i]))
                chrom = meta.get("chrom") or "unknown"

                t5 = target_5mc[i][mask_5mc[i]]
                p5 = pred_5mc[i][mask_5mc[i]]
                t6 = target_6ma[i][mask_6ma[i]]
                p6 = pred_6ma[i][mask_6ma[i]]

                append_metrics(all_targets, all_preds, "5mC", t5, p5)
                append_metrics(all_targets, all_preds, "6mA", t6, p6)
                append_metrics(per_region_targets[region], per_region_preds[region], "5mC", t5, p5)
                append_metrics(per_region_targets[region], per_region_preds[region], "6mA", t6, p6)
                append_metrics(per_sample_targets[sample], per_sample_preds[sample], "5mC", t5, p5)
                append_metrics(per_sample_targets[sample], per_sample_preds[sample], "6mA", t6, p6)
                append_metrics(per_chrom_targets[chrom], per_chrom_preds[chrom], "5mC", t5, p5)
                append_metrics(per_chrom_targets[chrom], per_chrom_preds[chrom], "6mA", t6, p6)

    metrics = {}
    for task, threshold in [("5mC", args.threshold_5mc), ("6mA", args.threshold_6ma)]:
        target = np.concatenate(all_targets[task]) if all_targets[task] else np.asarray([])
        pred = np.concatenate(all_preds[task]) if all_preds[task] else np.asarray([])
        metrics[task] = base_metric_row(target, pred, threshold)

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
    ]
    for path, grouped_targets, grouped_preds, group_name in [
        (per_region_path, per_region_targets, per_region_preds, "region"),
        (per_sample_path, per_sample_targets, per_sample_preds, "sample"),
        (per_chrom_path, per_chrom_targets, per_chrom_preds, "chrom"),
    ]:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for group in sorted(grouped_targets):
                for task, threshold in [("5mC", args.threshold_5mc), ("6mA", args.threshold_6ma)]:
                    target = (
                        np.concatenate(grouped_targets[group][task])
                        if grouped_targets[group][task]
                        else np.asarray([])
                    )
                    pred = (
                        np.concatenate(grouped_preds[group][task])
                        if grouped_preds[group][task]
                        else np.asarray([])
                    )
                    row = {"group": group, "task": task}
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
        "reads_evaluated": reads_evaluated,
        "batches_evaluated": batches_evaluated,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "max_length_used": args.max_length,
        "num_samples": num_samples,
        "sample_embedding_dim": sample_embedding_dim,
        "decoder_type": decoder_type,
        "decoder_hidden_dim": decoder_hidden_dim,
        "decoder_dropout": decoder_dropout,
        "decoder_kernel_size": decoder_kernel_size,
        "thresholds": {"5mC": args.threshold_5mc, "6mA": args.threshold_6ma},
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
