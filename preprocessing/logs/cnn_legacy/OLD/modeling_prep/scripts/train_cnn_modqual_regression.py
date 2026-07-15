#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN regression model for continuous mod_qual prediction.")
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModQualDataset(TensorDataset):
    def __init__(self, path: str | Path) -> None:
        with np.load(path) as arr:
            super().__init__(
                torch.from_numpy(arr["ref_kmer"]).long(),
                torch.from_numpy(arr["query_kmer"]).long(),
                torch.from_numpy(arr["pos_norm"].astype(np.float32)).unsqueeze(1),
                torch.from_numpy(arr["read_length_norm"].astype(np.float32)).unsqueeze(1),
                torch.from_numpy(arr["log_read_length_norm"].astype(np.float32)).unsqueeze(1),
                torch.from_numpy(arr["ref_strand"].astype(np.float32)).unsqueeze(1),
                torch.from_numpy(arr["ref_mod_strand"].astype(np.float32)).unsqueeze(1),
                torch.from_numpy(arr["mod_qual"].astype(np.float32)),
            )


class ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ModQualCNN(nn.Module):
    def __init__(self, embedding_dim: int, conv_channels: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(5, embedding_dim)
        self.input_proj = nn.Sequential(
            nn.Conv1d(embedding_dim, conv_channels, 1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
        )
        self.block3 = ConvBlock(conv_channels, 3, dropout)
        self.block5 = ConvBlock(conv_channels, 5, dropout)
        self.scalar_branch = nn.Sequential(
            nn.Linear(5, 24),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(24, 16),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(conv_channels * 4 + 16, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        ref_kmer: torch.Tensor,
        query_kmer: torch.Tensor,
        pos_norm: torch.Tensor,
        read_length_norm: torch.Tensor,
        log_read_length_norm: torch.Tensor,
        ref_strand: torch.Tensor,
        ref_mod_strand: torch.Tensor,
    ) -> torch.Tensor:
        seq = torch.cat([ref_kmer, query_kmer], dim=1)
        x = self.input_proj(self.embedding(seq).transpose(1, 2))
        x3 = self.block3(x)
        x5 = self.block5(x)
        pooled = torch.cat(
            [torch.amax(x3, 2), torch.mean(x3, 2), torch.amax(x5, 2), torch.mean(x5, 2)],
            dim=1,
        )
        scalar = self.scalar_branch(
            torch.cat([pos_norm, read_length_norm, log_read_length_norm, ref_strand, ref_mod_strand], dim=1)
        )
        return self.head(torch.cat([pooled, scalar], dim=1)).squeeze(1)


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_np = pred.numpy()
    target_np = target.numpy()
    residual = pred_np - target_np
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((target_np - np.mean(target_np)) ** 2))
    pearson = float(np.corrcoef(pred_np, target_np)[0, 1])
    pred_rank = np.argsort(np.argsort(pred_np))
    target_rank = np.argsort(np.argsort(target_np))
    spearman = float(np.corrcoef(pred_rank, target_rank)[0, 1])
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(1 - ss_res / ss_tot) if ss_tot else float("nan"),
        "pearson": pearson,
        "spearman": spearman,
    }


def threshold_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, float]:
    y_true = target >= threshold
    y_pred = pred >= threshold
    tp = int((y_true & y_pred).sum().item())
    tn = int((~y_true & ~y_pred).sum().item())
    fp = int((~y_true & y_pred).sum().item())
    fn = int((y_true & ~y_pred).sum().item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    preds = []
    targets = []
    for ref, query, pos, rln, logrln, rs, rms, target in loader:
        ref = ref.to(device)
        query = query.to(device)
        pos = pos.to(device)
        rln = rln.to(device)
        logrln = logrln.to(device)
        rs = rs.to(device)
        rms = rms.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        pred = model(ref, query, pos, rln, logrln, rs, rms)
        loss = criterion(pred, target)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += loss.item()
        batches += 1
        preds.append(pred.detach().cpu())
        targets.append(target.detach().cpu())
    pred_cat = torch.cat(preds)
    target_cat = torch.cat(targets)
    out = regression_metrics(pred_cat, target_cat)
    out["loss"] = total_loss / batches
    out["batches"] = batches
    out["pred_mean"] = float(pred_cat.mean().item())
    out["target_mean"] = float(target_cat.mean().item())
    out["threshold_0p8"] = threshold_metrics(pred_cat, target_cat, 0.8)
    out["threshold_0p85"] = threshold_metrics(pred_cat, target_cat, 0.85)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = ModQualDataset(args.train_npz)
    val_ds = ModQualDataset(args.val_npz)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModQualCNN(args.embedding_dim, args.conv_channels, args.dropout).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    config = vars(args).copy()
    config.update({"device": str(device), "train_rows": len(train_ds), "val_rows": len(val_ds), "target": "mod_qual"})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best_rmse = float("inf")
    with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as handle:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            val_metrics = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_metrics["rmse"])
            record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record, indent=2))
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, out_dir / "last_model.pt")
            if val_metrics["rmse"] < best_rmse:
                best_rmse = val_metrics["rmse"]
                torch.save(checkpoint, out_dir / "best_model.pt")


if __name__ == "__main__":
    main()
