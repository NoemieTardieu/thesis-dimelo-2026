#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_first_cnn_npz import binary_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an improved CNN on balanced NPZ subsets.")
    p.add_argument("--train-npz", required=True)
    p.add_argument("--val-npz", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--embedding-dim", type=int, default=12)
    p.add_argument("--conv-channels", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BalancedCNNDataset(TensorDataset):
    def __init__(self, path: str | Path) -> None:
        with np.load(path) as arr:
            ref_kmer = torch.from_numpy(arr["ref_kmer"]).long()
            query_kmer = torch.from_numpy(arr["query_kmer"]).long()
            pos_norm = torch.from_numpy(arr["pos_norm"].astype(np.float32)).unsqueeze(1)
            ref_strand = torch.from_numpy(arr["ref_strand"].astype(np.float32)).unsqueeze(1)
            ref_mod_strand = torch.from_numpy(arr["ref_mod_strand"].astype(np.float32)).unsqueeze(1)
            label = torch.from_numpy(arr["label"].astype(np.float32))
        super().__init__(ref_kmer, query_kmer, pos_norm, ref_strand, ref_mod_strand, label)


class ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ImprovedKmerCNN(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 12,
        conv_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim)
        self.input_proj = nn.Sequential(
            nn.Conv1d(embedding_dim, conv_channels, kernel_size=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
        )
        self.block3 = ConvBlock(conv_channels, kernel_size=3, dropout=dropout)
        self.block5 = ConvBlock(conv_channels, kernel_size=5, dropout=dropout)
        self.scalar_branch = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels * 4 + 16, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        ref_kmer: torch.Tensor,
        query_kmer: torch.Tensor,
        pos_norm: torch.Tensor,
        ref_strand: torch.Tensor,
        ref_mod_strand: torch.Tensor,
    ) -> torch.Tensor:
        seq = torch.cat([ref_kmer, query_kmer], dim=1)
        x = self.embedding(seq).transpose(1, 2)
        x = self.input_proj(x)
        x3 = self.block3(x)
        x5 = self.block5(x)

        pooled = torch.cat(
            [
                torch.amax(x3, dim=2),
                torch.mean(x3, dim=2),
                torch.amax(x5, dim=2),
                torch.mean(x5, dim=2),
            ],
            dim=1,
        )
        scalar = torch.cat([pos_norm, ref_strand, ref_mod_strand], dim=1)
        scalar = self.scalar_branch(scalar)
        logits = self.classifier(torch.cat([pooled, scalar], dim=1)).squeeze(1)
        return logits


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    logits_all = []
    labels_all = []

    for ref_kmer, query_kmer, pos_norm, ref_strand, ref_mod_strand, labels in loader:
        ref_kmer = ref_kmer.to(device)
        query_kmer = query_kmer.to(device)
        pos_norm = pos_norm.to(device)
        ref_strand = ref_strand.to(device)
        ref_mod_strand = ref_mod_strand.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(ref_kmer, query_kmer, pos_norm, ref_strand, ref_mod_strand)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        batches += 1
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())

    logits_cat = torch.cat(logits_all)
    labels_cat = torch.cat(labels_all)
    out = binary_metrics(logits_cat, labels_cat)
    probs = torch.sigmoid(logits_cat)
    out["loss"] = total_loss / batches
    out["batches"] = batches
    out["prob_mean"] = float(probs.mean().item())
    out["prob_min"] = float(probs.min().item())
    out["prob_max"] = float(probs.max().item())
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BalancedCNNDataset(args.train_npz)
    val_ds = BalancedCNNDataset(args.val_npz)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedKmerCNN(
        embedding_dim=args.embedding_dim,
        conv_channels=args.conv_channels,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    config = vars(args).copy()
    config["device"] = str(device)
    config["train_rows"] = len(train_ds)
    config["val_rows"] = len(val_ds)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_f1 = -1.0
    with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as handle:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            val_metrics = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_metrics["f1"])
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
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
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save(checkpoint, out_dir / "best_model.pt")


if __name__ == "__main__":
    main()
