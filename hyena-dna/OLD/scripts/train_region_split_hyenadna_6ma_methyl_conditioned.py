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


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if int(mask.sum()) == 0:
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], reduction="mean"
    )


def batch_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = model(
        batch["input_ids"],
        batch["sample_id"],
        batch["methyl_value"],
        batch["methyl_observed"],
    )
    return masked_bce_with_logits(logits, batch["target_6mA"], batch["mask_6mA"])


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train P(Reg | D,C,M): HyenaDNA + sample_id + observed 5mC -> 6mA."
    )
    p.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    p.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    p.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    p.add_argument("--train-npz", required=True)
    p.add_argument("--val-npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=None)
    p.add_argument("--sample-embedding-dim", type=int, default=16)
    p.add_argument("--decoder-hidden-dim", type=int, default=0)
    p.add_argument("--decoder-dropout", type=float, default=0.0)
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--unfreeze-prefix", action="append", default=[])
    p.add_argument("--max-train-batches", type=int, default=200)
    p.add_argument("--max-val-batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return p.parse_args()


def configure_trainable_parameters(
    model: MethylConditionedHyena6mA, unfreeze_prefixes: list[str]
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[str]]:
    for param in model.backbone.parameters():
        param.requires_grad = False

    unfrozen_names = []
    if unfreeze_prefixes:
        for name, param in model.backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in unfreeze_prefixes):
                param.requires_grad = True
                unfrozen_names.append(f"backbone.{name}")

    head_params = list(model.head_6ma.parameters()) + list(model.sample_embedding.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    return head_params, backbone_params, unfrozen_names


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.hyena_root)))

    from huggingface import HyenaDNAPreTrainedModel

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

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

    model = MethylConditionedHyena6mA(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_samples=args.num_samples,
        sample_embedding_dim=args.sample_embedding_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_dropout=args.decoder_dropout,
    ).to(device)
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

        train_losses = []
        for step, batch in enumerate(train_loader, start=1):
            if step > args.max_train_batches:
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.inference_mode():
            for step, batch in enumerate(val_loader, start=1):
                if step > args.max_val_batches:
                    break
                batch = move_batch(batch, device)
                val_losses.append(float(batch_loss(model, batch).item()))

        row = {
            "epoch": epoch,
            "train_6mA_loss": float(np.mean(train_losses)),
            "val_6mA_loss": float(np.mean(val_losses)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "model_type": "hyenadna_6ma_methyl_conditioned",
            "methylation_conditioned": True,
            "hidden_dim": hidden_dim,
            "num_samples": args.num_samples,
            "sample_embedding_dim": args.sample_embedding_dim,
            "methyl_feature_dim": 2,
            "decoder_hidden_dim": args.decoder_hidden_dim,
            "decoder_dropout": args.decoder_dropout,
            "sample_embedding_state_dict": model.sample_embedding.state_dict(),
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
                "model_type": "P(Reg|D,C,M)",
                "model_name": args.model_name,
                "train_npz": args.train_npz,
                "val_npz": args.val_npz,
                "train_reads": len(train_dataset),
                "val_reads": len(val_dataset),
                "max_length_used": args.max_length,
                "checkpoint": str(out_path),
                "num_samples": args.num_samples,
                "sample_embedding_dim": args.sample_embedding_dim,
                "decoder_hidden_dim": args.decoder_hidden_dim,
                "decoder_dropout": args.decoder_dropout,
                "methylation_features": ["target_5mC_filled_zero", "mask_5mC_observed"],
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
