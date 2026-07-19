#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DimeloJointDataset(Dataset):
    """Read-level tensors for a joint latent P(5mC, 6mA | DNA) model."""

    def __init__(self, npz_path: str | Path, max_length: int) -> None:
        self.path = Path(npz_path)
        self.data = np.load(self.path)
        self.max_length = int(max_length)
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


class HyenaJointLatent5mC6mA(nn.Module):
    """Shared HyenaDNA representation with a trainable latent state and two heads."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.head_5mc = nn.Linear(latent_dim, 1)
        self.head_6ma = nn.Linear(latent_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(input_ids)
        z = self.latent(hidden)
        logits_5mc = self.head_5mc(z).squeeze(-1)
        logits_6ma = self.head_6ma(z).squeeze(-1)
        return logits_5mc, logits_6ma


def ensure_transformers_stub() -> None:
    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("transformers")
        module.PreTrainedModel = nn.Module
        sys.modules["transformers"] = module


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
    loss = nn.functional.binary_cross_entropy_with_logits(
        valid_logits, valid_targets, reduction="none"
    )
    if pos_weight != 1.0:
        weights = torch.where(
            valid_targets >= 0.5,
            torch.full_like(valid_targets, float(pos_weight)),
            torch.ones_like(valid_targets),
        )
        loss = loss * weights
    if focal_gamma > 0.0:
        prob = torch.sigmoid(valid_logits)
        p_t = prob * valid_targets + (1.0 - prob) * (1.0 - valid_targets)
        loss = loss * torch.pow(1.0 - p_t, float(focal_gamma))
    return loss.mean()


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a joint latent-state HyenaDNA P(5mC,6mA|DNA) model."
    )
    parser.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/upstream_hyena_dna")
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/server_artifacts/upstream_checkpoints")
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--best-out", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=100)
    parser.add_argument("--max-val-batches", type=int, default=100)
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-6)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--loss-weight-5mc", type=float, default=0.5)
    parser.add_argument("--loss-weight-6ma", type=float, default=1.0)
    parser.add_argument("--pos-weight-6ma", type=float, default=1.0)
    parser.add_argument("--focal-gamma-6ma", type=float, default=0.0)
    parser.add_argument("--unfreeze-prefix", action="append", default=[])
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def configure_trainable_parameters(
    model: HyenaJointLatent5mC6mA, unfreeze_prefixes: list[str]
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[str]]:
    for param in model.backbone.parameters():
        param.requires_grad = False
    unfrozen_names = []
    for name, param in model.backbone.named_parameters():
        if any(name.startswith(prefix) for prefix in unfreeze_prefixes):
            param.requires_grad = True
            unfrozen_names.append(f"backbone.{name}")
    new_params = (
        list(model.latent.parameters())
        + list(model.head_5mc.parameters())
        + list(model.head_6ma.parameters())
    )
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    return new_params, backbone_params, unfrozen_names


def checkpoint_payload(
    args: argparse.Namespace,
    model: HyenaJointLatent5mC6mA,
    hidden_dim: int,
    unfrozen_names: list[str],
    history: list[dict],
) -> dict:
    return {
        "model_type": "P(5mC,6mA|DNA)_joint_latent",
        "latent_state_model": True,
        "sample_conditioned": False,
        "args": vars(args),
        "hidden_dim": hidden_dim,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "latent_state_dict": model.latent.state_dict(),
        "head_5mC_state_dict": model.head_5mc.state_dict(),
        "head_6mA_state_dict": model.head_6ma.state_dict(),
        "trainable_backbone_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.backbone.state_dict().items()
            if any(name.startswith(prefix) for prefix in args.unfreeze_prefix)
        },
        "unfrozen_backbone_parameter_names": unfrozen_names,
        "history": history,
    }


def run_validation(
    model: HyenaJointLatent5mC6mA,
    loader: DataLoader,
    args: argparse.Namespace,
    device: str,
    epoch: int,
) -> tuple[list[float], list[float], list[float]]:
    model.eval()
    total_losses: list[float] = []
    losses_5mc: list[float] = []
    losses_6ma: list[float] = []
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            if args.max_val_batches > 0 and step > args.max_val_batches:
                break
            batch = move_batch(batch, device)
            logits_5mc, logits_6ma = model(batch["input_ids"])
            loss_5mc = masked_bce_with_logits(
                logits_5mc, batch["target_5mC"], batch["mask_5mC"]
            )
            loss_6ma = masked_bce_with_logits(
                logits_6ma,
                batch["target_6mA"],
                batch["mask_6mA"],
                pos_weight=args.pos_weight_6ma,
                focal_gamma=args.focal_gamma_6ma,
            )
            total = args.loss_weight_5mc * loss_5mc + args.loss_weight_6ma * loss_6ma
            losses_5mc.append(float(loss_5mc.item()))
            losses_6ma.append(float(loss_6ma.item()))
            total_losses.append(float(total.item()))
            if args.log_every_batches > 0 and step % args.log_every_batches == 0:
                print(
                    json.dumps(
                        {
                            "progress": "val",
                            "epoch": epoch,
                            "batch": step,
                            "val_joint_loss_running": mean_or_none(total_losses),
                            "val_5mC_loss_running": mean_or_none(losses_5mc),
                            "val_6mA_loss_running": mean_or_none(losses_6ma),
                        }
                    ),
                    flush=True,
                )
    return total_losses, losses_5mc, losses_6ma


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.hyena_root)))
    ensure_transformers_stub()

    from huggingface import HyenaDNAPreTrainedModel

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = DimeloJointDataset(args.train_npz, args.max_length)
    val_dataset = DimeloJointDataset(args.val_npz, args.max_length)
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

    model = HyenaJointLatent5mC6mA(
        backbone,
        hidden_dim=hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)
    new_params, backbone_params, unfrozen_names = configure_trainable_parameters(
        model, args.unfreeze_prefix
    )
    param_groups = [{"params": new_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups)

    history: list[dict] = []
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
        train_total: list[float] = []
        train_5mc: list[float] = []
        train_6ma: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            if args.max_train_batches > 0 and step > args.max_train_batches:
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits_5mc, logits_6ma = model(batch["input_ids"])
            loss_5mc = masked_bce_with_logits(
                logits_5mc, batch["target_5mC"], batch["mask_5mC"]
            )
            loss_6ma = masked_bce_with_logits(
                logits_6ma,
                batch["target_6mA"],
                batch["mask_6mA"],
                pos_weight=args.pos_weight_6ma,
                focal_gamma=args.focal_gamma_6ma,
            )
            total = args.loss_weight_5mc * loss_5mc + args.loss_weight_6ma * loss_6ma
            total.backward()
            optimizer.step()
            train_total.append(float(total.item()))
            train_5mc.append(float(loss_5mc.item()))
            train_6ma.append(float(loss_6ma.item()))
            if args.log_every_batches > 0 and step % args.log_every_batches == 0:
                print(
                    json.dumps(
                        {
                            "progress": "train",
                            "epoch": epoch,
                            "batch": step,
                            "train_joint_loss_recent": mean_or_none(
                                train_total[-args.log_every_batches :]
                            ),
                            "train_joint_loss_running": mean_or_none(train_total),
                            "train_5mC_loss_running": mean_or_none(train_5mc),
                            "train_6mA_loss_running": mean_or_none(train_6ma),
                        }
                    ),
                    flush=True,
                )

        val_total, val_5mc, val_6ma = run_validation(model, val_loader, args, device, epoch)
        row = {
            "epoch": epoch,
            "train_joint_loss": mean_or_none(train_total),
            "train_5mC_loss": mean_or_none(train_5mc),
            "train_6mA_loss": mean_or_none(train_6ma),
            "val_joint_loss": mean_or_none(val_total),
            "val_5mC_loss": mean_or_none(val_5mc),
            "val_6mA_loss": mean_or_none(val_6ma),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        val_loss = row["val_joint_loss"]
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
                        "best_val_joint_loss": best_val_loss,
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
                        "best_val_joint_loss": best_val_loss,
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
                "model_type": "P(5mC,6mA|DNA)_joint_latent",
                "model_name": args.model_name,
                "train_npz": args.train_npz,
                "val_npz": args.val_npz,
                "train_reads": len(train_dataset),
                "val_reads": len(val_dataset),
                "checkpoint": str(out_path),
                "best_checkpoint": str(best_out_path),
                "best_epoch": best_epoch,
                "best_val_joint_loss": best_val_loss if best_epoch is not None else None,
                "latent_dim": args.latent_dim,
                "loss_weight_5mc": args.loss_weight_5mc,
                "loss_weight_6ma": args.loss_weight_6ma,
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
