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


class Dimelo6mAMethylEmbeddingDataset(Dataset):
    """Load DNA, observed CpG 5mC context, and 6mA targets for P(Reg|D,M)."""

    def __init__(
        self,
        npz_path: str | Path,
        max_length: int,
        c_token_id: int,
        g_token_id: int,
    ) -> None:
        self.path = Path(npz_path)
        self.data = np.load(self.path)
        self.max_length = max_length
        self.c_token_id = int(c_token_id)
        self.g_token_id = int(g_token_id)
        self.n = int(self.data["input_ids"].shape[0])
        self.has_is_cpg = "is_cpg" in self.data.files

    def __len__(self) -> int:
        return self.n

    def _derive_cpg_mask(self, input_ids: np.ndarray) -> np.ndarray:
        is_cpg = np.zeros_like(input_ids, dtype=bool)
        if input_ids.shape[0] > 1:
            is_cpg[:-1] = (input_ids[:-1] == self.c_token_id) & (
                input_ids[1:] == self.g_token_id
            )
        return is_cpg

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        end = self.max_length
        input_ids = self.data["input_ids"][idx, :end]
        target_5mc = self.data["target_5mC"][idx, :end].astype(np.float32)
        mask_5mc = self.data["mask_5mC"][idx, :end].astype(bool)
        if self.has_is_cpg:
            is_cpg = self.data["is_cpg"][idx, :end].astype(bool)
        else:
            is_cpg = self._derive_cpg_mask(input_ids)

        return {
            "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "methyl_value": torch.as_tensor(target_5mc, dtype=torch.float32),
            "methyl_observed": torch.as_tensor(mask_5mc, dtype=torch.bool),
            "is_cpg": torch.as_tensor(is_cpg, dtype=torch.bool),
            "target_6mA": torch.as_tensor(
                self.data["target_6mA"][idx, :end], dtype=torch.float32
            ),
            "mask_6mA": torch.as_tensor(
                self.data["mask_6mA"][idx, :end], dtype=torch.bool
            ),
        }


class Hyena6mAMethylEmbeddingConditionedNoSample(nn.Module):
    """Inject trainable methylation-state embeddings before HyenaDNA layers."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        decoder_hidden_dim: int = 0,
        decoder_dropout: float = 0.0,
        unknown_methylation_value: float = 0.5,
        methyl_embedding_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = int(hidden_dim)
        self.unknown_methylation_value = float(unknown_methylation_value)

        self.unmethylated_cpg_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.methylated_cpg_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.non_cpg_embedding = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.unmethylated_cpg_embedding, mean=0.0, std=methyl_embedding_init_std)
        nn.init.normal_(self.methylated_cpg_embedding, mean=0.0, std=methyl_embedding_init_std)
        nn.init.normal_(self.non_cpg_embedding, mean=0.0, std=methyl_embedding_init_std)

        if decoder_hidden_dim > 0:
            self.head_6ma = nn.Sequential(
                nn.Linear(hidden_dim, decoder_hidden_dim),
                nn.GELU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(decoder_hidden_dim, 1),
            )
        else:
            self.head_6ma = nn.Linear(hidden_dim, 1)

    def methylation_embedding(
        self,
        methyl_value: torch.Tensor,
        methyl_observed: torch.Tensor,
        is_cpg: torch.Tensor,
    ) -> torch.Tensor:
        methyl_prob = torch.where(
            methyl_observed,
            methyl_value,
            torch.full_like(methyl_value, self.unknown_methylation_value),
        ).clamp(0.0, 1.0)

        cpg_embedding = (
            (1.0 - methyl_prob).unsqueeze(-1) * self.unmethylated_cpg_embedding
            + methyl_prob.unsqueeze(-1) * self.methylated_cpg_embedding
        )
        non_cpg_embedding = self.non_cpg_embedding.view(1, 1, -1).expand_as(cpg_embedding)
        return torch.where(is_cpg.unsqueeze(-1), cpg_embedding, non_cpg_embedding)

    def encode_from_augmented_embeddings(
        self,
        input_ids: torch.Tensor,
        methyl_value: torch.Tensor,
        methyl_observed: torch.Tensor,
        is_cpg: torch.Tensor,
    ) -> torch.Tensor:
        # HyenaDNAModel -> LMBackbone. We use the upstream layers unchanged.
        lm_backbone = self.backbone.backbone
        hidden_states = lm_backbone.embeddings(input_ids)
        hidden_states = hidden_states + self.methylation_embedding(
            methyl_value, methyl_observed, is_cpg
        ).to(dtype=hidden_states.dtype)

        residual = None
        for layer in lm_backbone.layers:
            hidden_states, residual = layer(hidden_states, residual)

        dropped = lm_backbone.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = lm_backbone.ln_f(residual.to(dtype=lm_backbone.ln_f.weight.dtype))
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        methyl_value: torch.Tensor,
        methyl_observed: torch.Tensor,
        is_cpg: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encode_from_augmented_embeddings(
            input_ids, methyl_value, methyl_observed, is_cpg
        )
        return self.head_6ma(hidden).squeeze(-1)


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


def validation_constant_baseline(
    dataset: Dimelo6mAMethylEmbeddingDataset,
    max_batches: int,
    eps: float = 1e-7,
) -> dict[str, float | int]:
    valid_values = []
    n_batches = len(dataset) if max_batches <= 0 else min(len(dataset), max_batches)
    for idx in range(n_batches):
        row = dataset[idx]
        mask = row["mask_6mA"].numpy().astype(bool)
        if mask.any():
            valid_values.append(row["target_6mA"].numpy()[mask].astype(np.float64))
    if not valid_values:
        return {
            "valid_6mA_positions": 0,
            "mean_target": float("nan"),
            "constant_mean_bce": float("nan"),
            "constant_0.5_bce": float("nan"),
        }
    y = np.concatenate(valid_values)
    p = float(y.mean())
    p_clip = float(np.clip(p, eps, 1.0 - eps))
    bce_mean = float(-(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)).mean())
    bce_half = float(-(y * np.log(0.5) + (1.0 - y) * np.log(0.5)).mean())
    return {
        "valid_6mA_positions": int(y.size),
        "mean_target": p,
        "constant_mean_bce": bce_mean,
        "constant_0.5_bce": bce_half,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train no-sample P(6mA | DNA, observed 5mC) with methylation-state "
            "embeddings injected before the HyenaDNA backbone."
        )
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--methyl-embedding-lr", type=float, default=None)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-dropout", type=float, default=0.15)
    parser.add_argument("--unknown-methylation-value", type=float, default=0.5)
    parser.add_argument("--methyl-embedding-init-std", type=float, default=0.02)
    parser.add_argument("--c-token-id", type=int, default=8)
    parser.add_argument("--g-token-id", type=int, default=9)
    parser.add_argument("--unfreeze-prefix", action="append", default=[])
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0005)
    parser.add_argument("--pos-weight-6ma", type=float, default=1.0)
    parser.add_argument("--focal-gamma-6ma", type=float, default=0.0)
    parser.add_argument("--eval-before-training", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def configure_trainable_parameters(
    model: Hyena6mAMethylEmbeddingConditionedNoSample, unfreeze_prefixes: list[str]
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter], list[str]]:
    for param in model.backbone.parameters():
        param.requires_grad = False

    unfrozen_names = []
    for name, param in model.backbone.named_parameters():
        if any(name.startswith(prefix) for prefix in unfreeze_prefixes):
            param.requires_grad = True
            unfrozen_names.append(f"backbone.{name}")

    head_params = list(model.head_6ma.parameters())
    methyl_embedding_params = [
        model.unmethylated_cpg_embedding,
        model.methylated_cpg_embedding,
        model.non_cpg_embedding,
    ]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    return head_params, methyl_embedding_params, backbone_params, unfrozen_names


def checkpoint_payload(
    args: argparse.Namespace,
    model: Hyena6mAMethylEmbeddingConditionedNoSample,
    hidden_dim: int,
    unfrozen_names: list[str],
    history: list[dict],
) -> dict:
    return {
        "model_type": "P(6mA|DNA,5mC_methylation_embeddings)",
        "methylation_conditioned": True,
        "methylation_conditioning": "input_embedding_addition",
        "sample_conditioned": False,
        "args": vars(args),
        "hidden_dim": hidden_dim,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "decoder_dropout": args.decoder_dropout,
        "unknown_methylation_value": args.unknown_methylation_value,
        "c_token_id": args.c_token_id,
        "g_token_id": args.g_token_id,
        "head_6mA_state_dict": model.head_6ma.state_dict(),
        "methylation_embedding_state_dict": {
            "unmethylated_cpg_embedding": model.unmethylated_cpg_embedding.detach().cpu(),
            "methylated_cpg_embedding": model.methylated_cpg_embedding.detach().cpu(),
            "non_cpg_embedding": model.non_cpg_embedding.detach().cpu(),
        },
        "trainable_backbone_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.backbone.state_dict().items()
            if any(name.startswith(prefix) for prefix in args.unfreeze_prefix)
        },
        "unfrozen_backbone_parameter_names": unfrozen_names,
        "history": history,
    }


def evaluate_loss(
    model: Hyena6mAMethylEmbeddingConditionedNoSample,
    loader: DataLoader,
    args: argparse.Namespace,
    device: str,
    epoch: int,
    progress_label: str,
) -> list[float]:
    model.eval()
    losses = []
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            if args.max_val_batches > 0 and step > args.max_val_batches:
                break
            batch = move_batch(batch, device)
            logits = model(
                batch["input_ids"],
                batch["methyl_value"],
                batch["methyl_observed"],
                batch["is_cpg"],
            )
            loss = masked_bce_with_logits(
                logits,
                batch["target_6mA"],
                batch["mask_6mA"],
                pos_weight=args.pos_weight_6ma,
                focal_gamma=args.focal_gamma_6ma,
            )
            losses.append(float(loss.item()))
            if args.log_every_batches > 0 and step % args.log_every_batches == 0:
                print(
                    json.dumps(
                        {
                            "progress": progress_label,
                            "epoch": epoch,
                            "batch": step,
                            "val_6mA_loss_running": mean_or_none(losses),
                        }
                    ),
                    flush=True,
                )
    return losses


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

    train_dataset = Dimelo6mAMethylEmbeddingDataset(
        args.train_npz, args.max_length, args.c_token_id, args.g_token_id
    )
    val_dataset = Dimelo6mAMethylEmbeddingDataset(
        args.val_npz, args.max_length, args.c_token_id, args.g_token_id
    )
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

    print(
        json.dumps(
            {
                "progress": "validation_constant_baseline",
                **validation_constant_baseline(val_dataset, args.max_val_batches),
            }
        ),
        flush=True,
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

    model = Hyena6mAMethylEmbeddingConditionedNoSample(
        backbone,
        hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_dropout=args.decoder_dropout,
        unknown_methylation_value=args.unknown_methylation_value,
        methyl_embedding_init_std=args.methyl_embedding_init_std,
    ).to(device)

    head_params, methyl_embedding_params, backbone_params, unfrozen_names = (
        configure_trainable_parameters(model, args.unfreeze_prefix)
    )

    param_groups = [{"params": head_params, "lr": args.lr}]
    param_groups.append(
        {
            "params": methyl_embedding_params,
            "lr": args.methyl_embedding_lr if args.methyl_embedding_lr else args.lr,
        }
    )
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

    if args.eval_before_training:
        initial_val_losses = evaluate_loss(
            model, val_loader, args, device, epoch=0, progress_label="val_before_training"
        )
        row = {
            "epoch": 0,
            "train_6mA_loss": None,
            "val_6mA_loss": mean_or_none(initial_val_losses),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.train() if backbone_params else model.backbone.eval()
        train_losses = []
        for step, batch in enumerate(train_loader, start=1):
            if args.max_train_batches > 0 and step > args.max_train_batches:
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["input_ids"],
                batch["methyl_value"],
                batch["methyl_observed"],
                batch["is_cpg"],
            )
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

        val_losses = evaluate_loss(model, val_loader, args, device, epoch, "val")
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
                "model_type": "P(Reg|D,M_embedding)",
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
                "methylation_conditioning": {
                    "type": "input_embedding_addition",
                    "cpg": "B*methylated_embedding + (1-B)*unmethylated_embedding",
                    "unknown_cpg_value": args.unknown_methylation_value,
                    "non_cpg": "non_cpg_embedding",
                    "c_token_id": args.c_token_id,
                    "g_token_id": args.g_token_id,
                },
                "unfreeze_prefix": args.unfreeze_prefix,
                "n_unfrozen_backbone_parameters": int(sum(p.numel() for p in backbone_params)),
                "n_methylation_embedding_parameters": int(
                    sum(p.numel() for p in methyl_embedding_params)
                ),
                "final": history[-1],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
