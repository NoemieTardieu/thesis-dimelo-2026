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
            "hidden": hidden,
            "logits_5mC": self.head_5mc(hidden).squeeze(-1),
            "logits_6mA": self.head_6ma(hidden).squeeze(-1),
        }


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if int(mask.sum()) == 0:
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], reduction="mean"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run chr16 tensors through HyenaDNA tiny-1k plus two per-position heads."
    )
    p.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    p.add_argument("--model-name", default="hyenadna-tiny-1k-seqlen")
    p.add_argument(
        "--checkpoint-dir",
        default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints",
    )
    p.add_argument("--npz", default="outputs/merged_e5b_chr16_first1000.npz")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hyena_root = Path(args.hyena_root)
    sys.path.insert(0, str(hyena_root))

    from huggingface import HyenaDNAPreTrainedModel

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dataset = DimeloTensorDataset(args.npz, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    backbone = HyenaDNAPreTrainedModel.from_pretrained(
        args.checkpoint_dir,
        args.model_name,
        download=False,
        device=device,
        use_head=False,
    )
    backbone = backbone.to(device)
    backbone.eval()

    with torch.inference_mode():
        hidden_probe = backbone(batch["input_ids"])
    hidden_dim = int(hidden_probe.shape[-1])

    model = HyenaTinyTwoHead(backbone=backbone, hidden_dim=hidden_dim).to(device)
    model.eval()

    with torch.inference_mode():
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
        "model_name": args.model_name,
        "device": device,
        "dataset_length": len(dataset),
        "batch_size": args.batch_size,
        "max_length_used": args.max_length,
        "input_ids_shape": list(batch["input_ids"].shape),
        "hidden_shape": list(outputs["hidden"].shape),
        "logits_5mC_shape": list(outputs["logits_5mC"].shape),
        "logits_6mA_shape": list(outputs["logits_6mA"].shape),
        "valid_5mC_in_batch": int(batch["mask_5mC"].sum().item()),
        "valid_6mA_in_batch": int(batch["mask_6mA"].sum().item()),
        "loss_5mC": float(loss_5mc.item()),
        "loss_6mA": float(loss_6ma.item()),
        "total_loss": float(total_loss.item()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
