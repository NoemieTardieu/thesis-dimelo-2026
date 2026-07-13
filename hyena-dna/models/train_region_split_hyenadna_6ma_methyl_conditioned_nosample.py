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


class Dimelo6mAMethylConditionedDataset(Dataset):
    def __init__(self, npz_path: str | Path, max_length: int) -> None:
        self.path = Path(npz_path)
        self.data = np.load(self.path)
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
            "methyl_value": torch.as_tensor(methyl_value, dtype=torch.float32),
            "methyl_observed": torch.as_tensor(methyl_observed, dtype=torch.float32),
            "target_6mA": torch.as_tensor(self.data["target_6mA"][idx, :end], dtype=torch.float32),
            "mask_6mA": torch.as_tensor(self.data["mask_6mA"][idx, :end], dtype=torch.bool),
        }


class Hyena6mAMethylConditionedNoSample(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        methyl_feature_dim: int = 2,
        decoder_hidden_dim: int = 0,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        head_dim = hidden_dim + methyl_feature_dim
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
        methyl_value: torch.Tensor,
        methyl_observed: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(input_ids)
        methyl = torch.stack([methyl_value, methyl_observed], dim=-1)
        conditioned = torch.cat([hidden, methyl], dim=-1)
        return self.head_6ma(conditioned).squeeze(-1)


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    if int(mask.sum()) == 0:
        return logits.sum() * 0.0
    valid_logits = logits[mask]
    valid_targets = targets[mask]
    weight = None
    if pos_weight != 1.0:
        weight = torch.where(
            valid_targets >= 0.5,
            torch.full_like(valid_targets, float(pos_weight)),
            torch.ones_like(valid_targets),
        )
    loss = nn.functional.binary_cross_entropy_with_logits(
        valid_logits, valid_targets, reduction="none"
    )
    if focal_gamma > 0.0:
        prob = torch.sigmoid(valid_logits)
        p_t = prob * valid_targets + (1.0 - prob) * (1.0 - valid_targets)
        loss = loss * torch.pow(1.0 - p_t, float(focal_gamma))
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train clean no-sample P(6mA | DNA, observed 5mC) with HyenaDNA."
    )
    parser.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--best-out", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-6)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze-prefix", action="append", default=[])
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0005)
    parser.add_argument(
        "--pos-weight-6ma",
        type=float,
        default=1.0,
        help="Optional per-position weight for positive/high 6mA targets in the masked loss.",
    )
    parser.add_argument(
        "--focal-gamma-6ma",
        type=float,
        default=0.0,
        help="Optional focal-loss gamma for 6mA. 0 keeps ordinary BCE.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def configure_trainable_parameters(
    model: Hyena6mAMethylConditionedNoSample, unfreeze_prefixes: list[str]
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[str]]:
    for param in model.backbone.parameters():
        param.requires_grad = False

    unfrozen_names = []
    for name, param in model.backbone.named_parameters():
        if any(name.startswith(prefix) for prefix in unfreeze_prefixes):
            param.requires_grad = True
            unfrozen_names.append(f"backbone.{name}")

    head_params = list(model.head_6ma.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    return head_params, backbone_params, unfrozen_names


def checkpoint_payload(
    args: argparse.Namespace,
    model: Hyena6mAMethylConditionedNoSample,
    hidden_dim: int,
    unfrozen_names: list[str],
    history: list[dict],
) -> dict:
    return {
        "model_type": "P(6mA|DNA,5mC)",
        "methylation_conditioned": True,
        "sample_conditioned": False,
        "args": vars(args),
        "hidden_dim": hidden_dim,
        "methyl_feature_dim": 2,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "decoder_dropout": args.decoder_dropout,
        "head_6mA_state_dict": model.head_6ma.state_dict(),
        "trainable_backbone_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.backbone.state_dict().items()
            if any(name.startswith(prefix) for prefix in args.unfreeze_prefix)
        },
        "unfrozen_backbone_parameter_names": unfrozen_names,
        "history": history,
    }


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

    train_dataset = Dimelo6mAMethylConditionedDataset(args.train_npz, args.max_length)
    val_dataset = Dimelo6mAMethylConditionedDataset(args.val_npz, args.max_length)
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

    model = Hyena6mAMethylConditionedNoSample(
        backbone,
        hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_dropout=args.decoder_dropout,
    ).to(device)
    head_params, backbone_params, unfrozen_names = configure_trainable_parameters(
        model, args.unfreeze_prefix
    )

    param_groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups)

    history = []
    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    out_path = Path(args.out)
    best_out_path = Path(args.best_out) if args.best_out else out_path.with_suffix(".best.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.train() if backbone_params else model.backbone.eval()
        train_losses = []
        for step, batch in enumerate(train_loader, start=1):
            if args.max_train_batches > 0 and step > args.max_train_batches:
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["methyl_value"], batch["methyl_observed"])
            loss = masked_bce_with_logits(
                logits,
                batch["target_6mA"],
                batch["mask_6mA"],
                pos_weight=args.pos_weight_6ma,
                focal_gamma=args.focal_gamma_6ma,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            if args.log_every_batches > 0 and step % args.log_every_batches == 0:
                print(
                    json.dumps(
                        {
                            "progress": "train",
                            "epoch": epoch,
                            "batch": step,
                            "train_6mA_loss_recent": mean_or_none(
                                train_losses[-args.log_every_batches :]
                            ),
                            "train_6mA_loss_running": mean_or_none(train_losses),
                        }
                    ),
                    flush=True,
                )

        model.eval()
        val_losses = []
        with torch.inference_mode():
            for step, batch in enumerate(val_loader, start=1):
                if args.max_val_batches > 0 and step > args.max_val_batches:
                    break
                batch = move_batch(batch, device)
                logits = model(batch["input_ids"], batch["methyl_value"], batch["methyl_observed"])
                loss = masked_bce_with_logits(
                    logits,
                    batch["target_6mA"],
                    batch["mask_6mA"],
                    pos_weight=args.pos_weight_6ma,
                    focal_gamma=args.focal_gamma_6ma,
                )
                val_losses.append(float(loss.item()))
                if args.log_every_batches > 0 and step % args.log_every_batches == 0:
                    print(
                        json.dumps(
                            {
                                "progress": "val",
                                "epoch": epoch,
                                "batch": step,
                                "val_6mA_loss_running": mean_or_none(val_losses),
                            }
                        ),
                        flush=True,
                    )

        row = {
            "epoch": epoch,
            "train_6mA_loss": mean_or_none(train_losses),
            "val_6mA_loss": mean_or_none(val_losses),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        val_loss = row["val_6mA_loss"]
        if val_loss is not None and val_loss < (best_val_loss - args.early_stop_min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                checkpoint_payload(args, model, hidden_dim, unfrozen_names, history),
                best_out_path,
            )
            print(
                json.dumps(
                    {
                        "progress": "best_checkpoint",
                        "epoch": epoch,
                        "best_val_6mA_loss": best_val_loss,
                        "checkpoint": str(best_out_path),
                    }
                ),
                flush=True,
            )
        else:
            epochs_without_improvement += 1

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                json.dumps(
                    {
                        "progress": "early_stop",
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_val_6mA_loss": best_val_loss,
                        "epochs_without_improvement": epochs_without_improvement,
                    }
                ),
                flush=True,
            )
            break

    torch.save(checkpoint_payload(args, model, hidden_dim, unfrozen_names, history), out_path)

    print(
        json.dumps(
            {
                "status": "ok",
                "model_type": "P(Reg|D,M)",
                "model_name": args.model_name,
                "train_npz": args.train_npz,
                "val_npz": args.val_npz,
                "train_reads": len(train_dataset),
                "val_reads": len(val_dataset),
                "max_length_used": args.max_length,
                "checkpoint": str(out_path),
                "best_checkpoint": str(best_out_path),
                "best_epoch": best_epoch,
                "best_val_6mA_loss": best_val_loss if best_epoch is not None else None,
                "sample_conditioned": False,
                "decoder_hidden_dim": args.decoder_hidden_dim,
                "decoder_dropout": args.decoder_dropout,
                "methylation_features": ["target_5mC_filled_zero", "mask_5mC_observed"],
                "unfreeze_prefix": args.unfreeze_prefix,
                "n_unfrozen_backbone_parameters": int(sum(p.numel() for p in backbone_params)),
                "final": history[-1],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
