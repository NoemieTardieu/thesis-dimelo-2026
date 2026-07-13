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


class Dimelo5mCDataset(Dataset):
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
        }


class Hyena5mCOnly(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        decoder_hidden_dim: int = 0,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if decoder_hidden_dim > 0:
            self.head_5mc = nn.Sequential(
                nn.Linear(hidden_dim, decoder_hidden_dim),
                nn.GELU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(decoder_hidden_dim, 1),
            )
        else:
            self.head_5mc = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids)
        return self.head_5mc(hidden).squeeze(-1)


def ensure_transformers_stub() -> None:
    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("transformers")
        module.PreTrainedModel = nn.Module
        sys.modules["transformers"] = module


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if int(mask.sum()) == 0:
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], reduction="mean"
    )


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a final no-sample HyenaDNA P(5mC | DNA) model."
    )
    parser.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-hidden-dim", type=int, default=0)
    parser.add_argument("--decoder-dropout", type=float, default=0.0)
    parser.add_argument("--unfreeze-prefix", action="append", default=[])
    parser.add_argument("--max-train-batches", type=int, default=100)
    parser.add_argument("--max-val-batches", type=int, default=100)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--best-out", default=None)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def configure_trainable_parameters(
    model: Hyena5mCOnly, unfreeze_prefixes: list[str]
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[str]]:
    for param in model.backbone.parameters():
        param.requires_grad = False

    unfrozen_names = []
    for name, param in model.backbone.named_parameters():
        if any(name.startswith(prefix) for prefix in unfreeze_prefixes):
            param.requires_grad = True
            unfrozen_names.append(f"backbone.{name}")

    head_params = list(model.head_5mc.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    return head_params, backbone_params, unfrozen_names


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def checkpoint_payload(
    args: argparse.Namespace,
    model: Hyena5mCOnly,
    hidden_dim: int,
    unfrozen_names: list[str],
    history: list[dict],
) -> dict:
    return {
        "model_type": "P(5mC|DNA)",
        "args": vars(args),
        "hidden_dim": hidden_dim,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "decoder_dropout": args.decoder_dropout,
        "head_5mC_state_dict": model.head_5mc.state_dict(),
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
    ensure_transformers_stub()

    from huggingface import HyenaDNAPreTrainedModel

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = Dimelo5mCDataset(args.train_npz, args.max_length)
    val_dataset = Dimelo5mCDataset(args.val_npz, args.max_length)
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

    model = Hyena5mCOnly(
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
            logits = model(batch["input_ids"])
            loss = masked_bce_with_logits(logits, batch["target_5mC"], batch["mask_5mC"])
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
                            "train_5mC_loss_recent": mean_or_none(
                                train_losses[-args.log_every_batches :]
                            ),
                            "train_5mC_loss_running": mean_or_none(train_losses),
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
                logits = model(batch["input_ids"])
                loss = masked_bce_with_logits(logits, batch["target_5mC"], batch["mask_5mC"])
                val_losses.append(float(loss.item()))
                if args.log_every_batches > 0 and step % args.log_every_batches == 0:
                    print(
                        json.dumps(
                            {
                                "progress": "val",
                                "epoch": epoch,
                                "batch": step,
                                "val_5mC_loss_running": mean_or_none(val_losses),
                            }
                        ),
                        flush=True,
                    )

        row = {
            "epoch": epoch,
            "train_5mC_loss": mean_or_none(train_losses),
            "val_5mC_loss": mean_or_none(val_losses),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        val_loss = row["val_5mC_loss"]
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
                        "best_val_5mC_loss": best_val_loss,
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
                        "best_val_5mC_loss": best_val_loss,
                        "epochs_without_improvement": epochs_without_improvement,
                    }
                ),
                flush=True,
            )
            break

    torch.save(
        checkpoint_payload(args, model, hidden_dim, unfrozen_names, history),
        out_path,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "model_type": "P(5mC|DNA)",
                "model_name": args.model_name,
                "train_npz": args.train_npz,
                "val_npz": args.val_npz,
                "train_reads": len(train_dataset),
                "val_reads": len(val_dataset),
                "max_length_used": args.max_length,
                "checkpoint": str(out_path),
                "best_checkpoint": str(best_out_path),
                "best_epoch": best_epoch,
                "best_val_5mC_loss": best_val_loss if best_epoch is not None else None,
                "decoder_hidden_dim": args.decoder_hidden_dim,
                "decoder_dropout": args.decoder_dropout,
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
