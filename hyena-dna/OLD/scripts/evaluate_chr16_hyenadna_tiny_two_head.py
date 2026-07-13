#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


class DimeloTensorDataset(Dataset):
    def __init__(self, npz_path: str | Path, max_length: int = 1024) -> None:
        self.path = Path(npz_path)
        self.data = np.load(self.path)
        self.max_length = max_length
        self.n = int(self.data["input_ids"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end = self.max_length
        return {
            "input_ids": torch.as_tensor(self.data["input_ids"][idx, :end], dtype=torch.long),
            "target_5mC": torch.as_tensor(self.data["target_5mC"][idx, :end], dtype=torch.float32),
            "mask_5mC": torch.as_tensor(self.data["mask_5mC"][idx, :end], dtype=torch.bool),
            "target_6mA": torch.as_tensor(self.data["target_6mA"][idx, :end], dtype=torch.float32),
            "mask_6mA": torch.as_tensor(self.data["mask_6mA"][idx, :end], dtype=torch.bool),
        }


class HyenaTinyTwoHead(nn.Module):
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


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def update_metrics(
    store: dict[str, float],
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
) -> None:
    if int(mask.sum()) == 0:
        return

    pred = torch.sigmoid(logits[mask])
    target = targets[mask]
    bce = nn.functional.binary_cross_entropy(pred, target, reduction="sum")
    mse = nn.functional.mse_loss(pred, target, reduction="sum")
    mae = nn.functional.l1_loss(pred, target, reduction="sum")
    count = int(mask.sum().item())

    store[f"{prefix}_valid_positions"] += count
    store[f"{prefix}_bce_sum"] += float(bce.item())
    store[f"{prefix}_mse_sum"] += float(mse.item())
    store[f"{prefix}_mae_sum"] += float(mae.item())
    store[f"{prefix}_pred_sum"] += float(pred.sum().item())
    store[f"{prefix}_target_sum"] += float(target.sum().item())


def finalize_metrics(store: dict[str, float], prefix: str) -> dict[str, float | int | None]:
    count = int(store[f"{prefix}_valid_positions"])
    if count == 0:
        return {
            "valid_positions": 0,
            "bce": None,
            "mse": None,
            "mae": None,
            "mean_pred": None,
            "mean_target": None,
        }
    return {
        "valid_positions": count,
        "bce": store[f"{prefix}_bce_sum"] / count,
        "mse": store[f"{prefix}_mse_sum"] / count,
        "mae": store[f"{prefix}_mae_sum"] / count,
        "mean_pred": store[f"{prefix}_pred_sum"] / count,
        "mean_target": store[f"{prefix}_target_sum"] / count,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate saved HyenaDNA tiny two-head checkpoint on validation batches."
    )
    p.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    p.add_argument("--model-name", default="hyenadna-tiny-1k-seqlen")
    p.add_argument(
        "--checkpoint-dir",
        default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints",
    )
    p.add_argument("--npz", default="outputs/merged_e5b_chr16_first1000.npz")
    p.add_argument(
        "--checkpoint",
        default="outputs/hyenadna_tiny_two_head_debug_3epochs_25batches.pt",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--max-val-batches", type=int, default=25)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.hyena_root)))

    from huggingface import HyenaDNAPreTrainedModel

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = DimeloTensorDataset(args.npz, max_length=args.max_length)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_fraction))
    val_indices = indices[:n_val]

    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    backbone = HyenaDNAPreTrainedModel.from_pretrained(
        args.checkpoint_dir,
        args.model_name,
        download=False,
        device=device,
        use_head=False,
    ).to(device)
    backbone.eval()

    probe = move_batch(next(iter(val_loader)), device)
    with torch.inference_mode():
        hidden_probe = backbone(probe["input_ids"])
    hidden_dim = int(hidden_probe.shape[-1])

    model = HyenaTinyTwoHead(backbone=backbone, hidden_dim=hidden_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.head_5mc.load_state_dict(checkpoint["head_5mC_state_dict"])
    model.head_6ma.load_state_dict(checkpoint["head_6mA_state_dict"])
    model.eval()

    store = {
        "5mC_valid_positions": 0.0,
        "5mC_bce_sum": 0.0,
        "5mC_mse_sum": 0.0,
        "5mC_mae_sum": 0.0,
        "5mC_pred_sum": 0.0,
        "5mC_target_sum": 0.0,
        "6mA_valid_positions": 0.0,
        "6mA_bce_sum": 0.0,
        "6mA_mse_sum": 0.0,
        "6mA_mae_sum": 0.0,
        "6mA_pred_sum": 0.0,
        "6mA_target_sum": 0.0,
    }

    batches_seen = 0
    with torch.inference_mode():
        for step, batch in enumerate(val_loader, start=1):
            if step > args.max_val_batches:
                break
            batch = move_batch(batch, device)
            outputs = model(batch["input_ids"])
            update_metrics(
                store,
                outputs["logits_5mC"],
                batch["target_5mC"],
                batch["mask_5mC"],
                "5mC",
            )
            update_metrics(
                store,
                outputs["logits_6mA"],
                batch["target_6mA"],
                batch["mask_6mA"],
                "6mA",
            )
            batches_seen += 1

    result = {
        "npz": args.npz,
        "checkpoint": args.checkpoint,
        "model_name": args.model_name,
        "device": device,
        "dataset_length": len(dataset),
        "val_reads_total": len(val_indices),
        "val_batches_evaluated": batches_seen,
        "batch_size": args.batch_size,
        "max_length_used": args.max_length,
        "metrics": {
            "5mC": finalize_metrics(store, "5mC"),
            "6mA": finalize_metrics(store, "6mA"),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
