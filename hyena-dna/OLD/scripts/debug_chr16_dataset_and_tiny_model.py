#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DimeloTensorDataset(Dataset):
    def __init__(self, npz_path: str | Path, max_length: int | None = None) -> None:
        self.path = Path(npz_path)
        self.data = np.load(self.path)
        self.max_length = max_length
        self.n = int(self.data["input_ids"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self.data["input_ids"][idx],
            "target_5mC": self.data["target_5mC"][idx],
            "mask_5mC": self.data["mask_5mC"][idx],
            "target_6mA": self.data["target_6mA"][idx],
            "mask_6mA": self.data["mask_6mA"][idx],
        }
        if self.max_length is not None:
            item = {k: v[: self.max_length] for k, v in item.items()}
        return {
            "input_ids": torch.as_tensor(item["input_ids"], dtype=torch.long),
            "target_5mC": torch.as_tensor(item["target_5mC"], dtype=torch.float32),
            "mask_5mC": torch.as_tensor(item["mask_5mC"], dtype=torch.bool),
            "target_6mA": torch.as_tensor(item["target_6mA"], dtype=torch.float32),
            "mask_6mA": torch.as_tensor(item["mask_6mA"], dtype=torch.bool),
        }


class TinyTwoHeadModel(nn.Module):
    def __init__(self, vocab_size: int = 12, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.head_5mc = nn.Linear(hidden_dim, 1)
        self.head_6ma = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.embedding(input_ids)
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        return {
            "logits_5mC": self.head_5mc(x).squeeze(-1),
            "logits_6mA": self.head_6ma(x).squeeze(-1),
        }


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if int(mask.sum()) == 0:
        return logits.sum() * 0.0
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], reduction="mean"
    )
    return loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect chr16 DiMeLo tensors with a PyTorch Dataset/DataLoader and tiny two-head model."
    )
    p.add_argument(
        "--npz",
        default="outputs/merged_e5b_chr16_first1000.npz",
        help="Tensor .npz file from make_chr16_dimelo_tensors.py.",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Use a prefix of each read for this quick CPU debug. Use 32768 for full length.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DimeloTensorDataset(args.npz, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    batch = next(iter(loader))
    model = TinyTwoHeadModel().to(args.device)
    batch = {k: v.to(args.device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(batch["input_ids"])
        loss_5mc = masked_bce_with_logits(
            outputs["logits_5mC"], batch["target_5mC"], batch["mask_5mC"]
        )
        loss_6ma = masked_bce_with_logits(
            outputs["logits_6mA"], batch["target_6mA"], batch["mask_6mA"]
        )
        total_loss = loss_5mc + loss_6ma

    summary = {
        "npz": str(args.npz),
        "dataset_length": len(dataset),
        "batch_size": args.batch_size,
        "max_length_used": args.max_length,
        "input_ids_shape": list(batch["input_ids"].shape),
        "target_5mC_shape": list(batch["target_5mC"].shape),
        "target_6mA_shape": list(batch["target_6mA"].shape),
        "valid_5mC_in_batch": int(batch["mask_5mC"].sum().item()),
        "valid_6mA_in_batch": int(batch["mask_6mA"].sum().item()),
        "logits_5mC_shape": list(outputs["logits_5mC"].shape),
        "logits_6mA_shape": list(outputs["logits_6mA"].shape),
        "loss_5mC": float(loss_5mc.item()),
        "loss_6mA": float(loss_6ma.item()),
        "total_loss": float(total_loss.item()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
