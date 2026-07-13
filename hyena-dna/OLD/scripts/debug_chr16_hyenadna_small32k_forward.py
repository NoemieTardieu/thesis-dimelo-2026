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
    def __init__(self, npz_path: str | Path, max_length: int = 32768) -> None:
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
        description="Smoke test HyenaDNA-small-32k on one chr16 DiMeLo tensor batch."
    )
    p.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    p.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    p.add_argument(
        "--checkpoint-dir",
        default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints",
    )
    p.add_argument("--npz", default="outputs/merged_e5b_chr16_first1000.npz")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--download",
        action="store_true",
        help="Download checkpoint with huggingface_hub if not already present.",
    )
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

    checkpoint_root = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_root / args.model_name
    if args.download:
        from huggingface_hub import snapshot_download

        checkpoint_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=f"LongSafari/{args.model_name}",
            local_dir=str(checkpoint_path),
            local_dir_use_symlinks=False,
        )

    dataset = DimeloTensorDataset(args.npz, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    backbone = HyenaDNAPreTrainedModel.from_pretrained(
        args.checkpoint_dir,
        args.model_name,
        download=False,
        device=device,
        use_head=False,
    ).to(device)
    backbone.eval()

    with torch.inference_mode():
        hidden = backbone(batch["input_ids"])
        head_5mc = nn.Linear(int(hidden.shape[-1]), 1).to(device)
        head_6ma = nn.Linear(int(hidden.shape[-1]), 1).to(device)
        logits_5mc = head_5mc(hidden).squeeze(-1)
        logits_6ma = head_6ma(hidden).squeeze(-1)
        loss_5mc = masked_bce_with_logits(logits_5mc, batch["target_5mC"], batch["mask_5mC"])
        loss_6ma = masked_bce_with_logits(logits_6ma, batch["target_6mA"], batch["mask_6mA"])

    summary = {
        "npz": args.npz,
        "model_name": args.model_name,
        "checkpoint_path": str(checkpoint_path),
        "device": device,
        "batch_size": args.batch_size,
        "max_length_used": args.max_length,
        "input_ids_shape": list(batch["input_ids"].shape),
        "hidden_shape": list(hidden.shape),
        "logits_5mC_shape": list(logits_5mc.shape),
        "logits_6mA_shape": list(logits_6ma.shape),
        "valid_5mC_in_batch": int(batch["mask_5mC"].sum().item()),
        "valid_6mA_in_batch": int(batch["mask_6mA"].sum().item()),
        "loss_5mC": float(loss_5mc.item()),
        "loss_6mA": float(loss_6ma.item()),
        "total_loss": float((loss_5mc + loss_6ma).item()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
