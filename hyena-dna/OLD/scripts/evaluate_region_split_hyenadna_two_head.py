#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
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


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


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
    count = int(mask.sum().item())
    store[f"{prefix}_valid_positions"] += count
    store[f"{prefix}_bce_sum"] += float(nn.functional.binary_cross_entropy(pred, target, reduction="sum").item())
    store[f"{prefix}_mse_sum"] += float(nn.functional.mse_loss(pred, target, reduction="sum").item())
    store[f"{prefix}_mae_sum"] += float(nn.functional.l1_loss(pred, target, reduction="sum").item())
    store[f"{prefix}_pred_sum"] += float(pred.sum().item())
    store[f"{prefix}_target_sum"] += float(target.sum().item())


def finalize(store: dict[str, float], prefix: str) -> dict[str, float | int | None]:
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
    p = argparse.ArgumentParser(description="Evaluate HyenaDNA two-head checkpoint on one explicit split .npz.")
    p.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    p.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    p.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    p.add_argument("--npz", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split-name", default="val")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--max-batches", type=int, default=0, help="0 means all batches.")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.hyena_root)))

    from huggingface import HyenaDNAPreTrainedModel

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dataset = DimeloTensorDataset(args.npz, args.max_length)
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
    batches = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            if args.max_batches and step > args.max_batches:
                break
            batch = move_batch(batch, device)
            outputs = model(batch["input_ids"])
            update_metrics(store, outputs["logits_5mC"], batch["target_5mC"], batch["mask_5mC"], "5mC")
            update_metrics(store, outputs["logits_6mA"], batch["target_6mA"], batch["mask_6mA"], "6mA")
            batches += 1

    print(
        json.dumps(
            {
                "npz": args.npz,
                "split_name": args.split_name,
                "checkpoint": args.checkpoint,
                "model_name": args.model_name,
                "device": device,
                "reads": len(dataset),
                "batches_evaluated": batches,
                "batch_size": args.batch_size,
                "max_length_used": args.max_length,
                "metrics": {
                    "5mC": finalize(store, "5mC"),
                    "6mA": finalize(store, "6mA"),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
