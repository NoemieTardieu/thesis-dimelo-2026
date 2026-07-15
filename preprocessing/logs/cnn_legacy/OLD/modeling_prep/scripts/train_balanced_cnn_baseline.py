#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_first_cnn_npz import FirstKmerCNN, binary_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the first CNN on balanced NPZ subsets.")
    p.add_argument("--train-npz", required=True)
    p.add_argument("--val-npz", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embedding-dim", type=int, default=8)
    p.add_argument("--conv-channels", type=int, default=32)
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
            optimizer.step()

        total_loss += loss.item()
        batches += 1
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())

    out = binary_metrics(torch.cat(logits_all), torch.cat(labels_all))
    out["loss"] = total_loss / batches
    out["batches"] = batches
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
    model = FirstKmerCNN(
        embedding_dim=args.embedding_dim,
        conv_channels=args.conv_channels,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record, indent=2))
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, out_dir / "last_model.pt")
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, out_dir / "best_model.pt")


if __name__ == "__main__":
    main()
