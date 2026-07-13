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


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if int(mask.sum()) == 0:
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], reduction="mean"
    )


def batch_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    outputs = model(batch["input_ids"])
    loss_5mc = masked_bce_with_logits(
        outputs["logits_5mC"], batch["target_5mC"], batch["mask_5mC"]
    )
    loss_6ma = masked_bce_with_logits(
        outputs["logits_6mA"], batch["target_6mA"], batch["mask_6mA"]
    )
    return {
        "loss_5mC": loss_5mc,
        "loss_6mA": loss_6ma,
        "total_loss": loss_5mc + loss_6ma,
    }


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train HyenaDNA backbone + two heads using explicit region-level train/val tensors."
    )
    p.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    p.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    p.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    p.add_argument("--train-npz", required=True)
    p.add_argument("--val-npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Learning rate for unfrozen backbone parameters. Defaults to lr if unset.",
    )
    p.add_argument(
        "--unfreeze-prefix",
        action="append",
        default=[],
        help=(
            "Backbone parameter/module name prefix to unfreeze. Can be passed multiple times. "
            "Example: --unfreeze-prefix backbone.layers.3 --unfreeze-prefix backbone.ln_f"
        ),
    )
    p.add_argument("--max-train-batches", type=int, default=200)
    p.add_argument("--max-val-batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return p.parse_args()


def configure_trainable_parameters(
    model: nn.Module, unfreeze_prefixes: list[str]
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[str]]:
    for param in model.backbone.parameters():
        param.requires_grad = False

    unfrozen_names = []
    if unfreeze_prefixes:
        for name, param in model.backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in unfreeze_prefixes):
                param.requires_grad = True
                unfrozen_names.append(f"backbone.{name}")

    head_params = list(model.head_5mc.parameters()) + list(model.head_6ma.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    return head_params, backbone_params, unfrozen_names


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

    train_dataset = DimeloTensorDataset(args.train_npz, args.max_length)
    val_dataset = DimeloTensorDataset(args.val_npz, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
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

    probe = move_batch(next(iter(train_loader)), device)
    with torch.inference_mode():
        hidden_probe = backbone(probe["input_ids"])
    hidden_dim = int(hidden_probe.shape[-1])

    model = HyenaTwoHead(backbone, hidden_dim).to(device)
    head_params, backbone_params, unfrozen_names = configure_trainable_parameters(
        model, args.unfreeze_prefix
    )
    param_groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": args.backbone_lr if args.backbone_lr is not None else args.lr,
            }
        )
    optimizer = torch.optim.AdamW(param_groups)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if backbone_params:
            model.backbone.train()
        else:
            model.backbone.eval()
        train_total = []
        train_5mc = []
        train_6ma = []
        for step, batch in enumerate(train_loader, start=1):
            if step > args.max_train_batches:
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = batch_loss(model, batch)
            losses["total_loss"].backward()
            optimizer.step()
            train_total.append(float(losses["total_loss"].item()))
            train_5mc.append(float(losses["loss_5mC"].item()))
            train_6ma.append(float(losses["loss_6mA"].item()))

        model.eval()
        val_total = []
        val_5mc = []
        val_6ma = []
        with torch.inference_mode():
            for step, batch in enumerate(val_loader, start=1):
                if step > args.max_val_batches:
                    break
                batch = move_batch(batch, device)
                losses = batch_loss(model, batch)
                val_total.append(float(losses["total_loss"].item()))
                val_5mc.append(float(losses["loss_5mC"].item()))
                val_6ma.append(float(losses["loss_6mA"].item()))

        row = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(train_total)),
            "train_5mC_loss": float(np.mean(train_5mc)),
            "train_6mA_loss": float(np.mean(train_6ma)),
            "val_total_loss": float(np.mean(val_total)),
            "val_5mC_loss": float(np.mean(val_5mc)),
            "val_6mA_loss": float(np.mean(val_6ma)),
        }
        history.append(row)
        print(json.dumps(row))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "hidden_dim": hidden_dim,
            "head_5mC_state_dict": model.head_5mc.state_dict(),
            "head_6mA_state_dict": model.head_6ma.state_dict(),
            "trainable_backbone_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.backbone.state_dict().items()
                if any(name.startswith(prefix) for prefix in args.unfreeze_prefix)
            },
            "unfrozen_backbone_parameter_names": unfrozen_names,
            "history": history,
        },
        out_path,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "model_name": args.model_name,
                "train_npz": args.train_npz,
                "val_npz": args.val_npz,
                "train_reads": len(train_dataset),
                "val_reads": len(val_dataset),
                "max_length_used": args.max_length,
                "checkpoint": str(out_path),
                "unfreeze_prefix": args.unfreeze_prefix,
                "n_unfrozen_backbone_parameters": int(
                    sum(p.numel() for p in backbone_params)
                ),
                "final": history[-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
