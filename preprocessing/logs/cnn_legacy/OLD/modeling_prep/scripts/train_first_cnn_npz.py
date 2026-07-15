#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a first CNN from integer-encoded NPZ chunks.")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embedding-dim", type=int, default=8)
    p.add_argument("--conv-channels", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--train-max-chunks",
        type=int,
        default=None,
        help="Optional limit for number of training chunks per epoch.",
    )
    p.add_argument(
        "--val-max-chunks",
        type=int,
        default=16,
        help="Validation subset size for fast first runs. Use a larger value later.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Start with 0 for stability on HPC; increase later if needed.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("chunk_*.npz"))


class ChunkedNPZDataset(IterableDataset):
    def __init__(
        self,
        directory: str | Path,
        shuffle_chunks: bool = False,
        shuffle_rows: bool = False,
        max_chunks: int | None = None,
        seed: int = 42,
        selection_mode: str = "first",
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_rows = shuffle_rows
        self.max_chunks = max_chunks
        self.seed = seed
        self.selection_mode = selection_mode

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        files = chunk_paths(self.directory)
        rng = np.random.default_rng(self.seed + torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else self.seed)

        if self.shuffle_chunks:
            files = files.copy()
            rng.shuffle(files)

        if self.max_chunks is not None:
            if self.selection_mode == "uniform" and self.max_chunks < len(files):
                indices = np.linspace(0, len(files) - 1, self.max_chunks, dtype=int)
                files = [files[i] for i in indices]
            else:
                files = files[: self.max_chunks]

        for path in files:
            with np.load(path) as arr:
                ref_kmer = arr["ref_kmer"]
                query_kmer = arr["query_kmer"]
                pos_norm = arr["pos_norm"]
                ref_strand = arr["ref_strand"]
                ref_mod_strand = arr["ref_mod_strand"]
                label = arr["label"]

                indices = np.arange(label.shape[0])
                if self.shuffle_rows:
                    rng.shuffle(indices)

                for idx in indices:
                    yield {
                        "ref_kmer": torch.from_numpy(ref_kmer[idx]).long(),
                        "query_kmer": torch.from_numpy(query_kmer[idx]).long(),
                        "pos_norm": torch.tensor([pos_norm[idx]], dtype=torch.float32),
                        "ref_strand": torch.tensor([ref_strand[idx]], dtype=torch.float32),
                        "ref_mod_strand": torch.tensor([ref_mod_strand[idx]], dtype=torch.float32),
                        "label": torch.tensor(label[idx], dtype=torch.float32),
                    }


class FirstKmerCNN(nn.Module):
    def __init__(self, embedding_dim: int = 8, conv_channels: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim)
        self.conv3 = nn.Conv1d(embedding_dim, conv_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embedding_dim, conv_channels, kernel_size=5, padding=2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Sequential(
            nn.Linear(conv_channels * 2 + 3, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        ref_kmer: torch.Tensor,
        query_kmer: torch.Tensor,
        pos_norm: torch.Tensor,
        ref_strand: torch.Tensor,
        ref_mod_strand: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenate both 5-mers into one length-10 sequence.
        seq = torch.cat([ref_kmer, query_kmer], dim=1)
        emb = self.embedding(seq)  # [B, L, E]
        emb = emb.transpose(1, 2)  # [B, E, L]

        x3 = self.act(self.conv3(emb))
        x5 = self.act(self.conv5(emb))

        x3 = torch.amax(x3, dim=2)
        x5 = torch.amax(x5, dim=2)

        features = torch.cat([x3, x5, pos_norm, ref_strand, ref_mod_strand], dim=1)
        features = self.dropout(features)
        logits = self.fc(features).squeeze(1)
        return logits


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": acc,
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
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_batches = 0
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        ref_kmer = batch["ref_kmer"].to(device)
        query_kmer = batch["query_kmer"].to(device)
        pos_norm = batch["pos_norm"].to(device)
        ref_strand = batch["ref_strand"].to(device)
        ref_mod_strand = batch["ref_mod_strand"].to(device)
        labels = batch["label"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(ref_kmer, query_kmer, pos_norm, ref_strand, ref_mod_strand)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())

    if total_batches == 0:
        return {"loss": math.nan, "accuracy": math.nan, "precision": math.nan, "recall": math.nan, "f1": math.nan}

    logits_cat = torch.cat(logits_all)
    labels_cat = torch.cat(labels_all)
    metrics = binary_metrics(logits_cat, labels_cat)
    metrics["loss"] = total_loss / total_batches
    metrics["batches"] = total_batches
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_summary = load_summary(train_dir / "summary.json")
    val_summary = load_summary(val_dir / "summary.json")

    neg = int(train_summary["label_counts"]["0"])
    pos = int(train_summary["label_counts"]["1"])
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ChunkedNPZDataset(
        train_dir,
        shuffle_chunks=True,
        shuffle_rows=True,
        max_chunks=args.train_max_chunks,
        seed=args.seed,
        selection_mode="first",
    )
    val_ds = ChunkedNPZDataset(
        val_dir,
        shuffle_chunks=False,
        shuffle_rows=False,
        max_chunks=args.val_max_chunks,
        seed=args.seed,
        selection_mode="uniform",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = FirstKmerCNN(
        embedding_dim=args.embedding_dim,
        conv_channels=args.conv_channels,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = out_dir / "metrics.jsonl"
    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"
    config_path = out_dir / "config.json"

    config = vars(args).copy()
    config["device"] = str(device)
    config["train_summary"] = train_summary
    config["val_summary"] = val_summary
    config["train_pos_weight"] = float(pos_weight.item())
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val_f1 = -1.0

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = run_epoch(model, val_loader, None, criterion, device)

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

            print(json.dumps(record, indent=2))

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": config,
                },
                last_path,
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "config": config,
                    },
                    best_path,
                )


if __name__ == "__main__":
    main()
