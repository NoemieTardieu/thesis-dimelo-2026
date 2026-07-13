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
    p = argparse.ArgumentParser(description="Train MLP ablations on balanced CNN NPZ subsets.")
    p.add_argument("--train-npz", required=True)
    p.add_argument("--val-npz", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--feature-set",
        choices=[
            "ref",
            "query",
            "ref_query",
            "ref_query_scalar",
            "scalar",
        ],
        required=True,
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_features(arr: np.lib.npyio.NpzFile, feature_set: str) -> np.ndarray:
    pieces = []
    if feature_set in {"ref", "ref_query", "ref_query_scalar"}:
        pieces.append(arr["ref_kmer"].astype(np.float32) / 4.0)
    if feature_set in {"query", "ref_query", "ref_query_scalar"}:
        pieces.append(arr["query_kmer"].astype(np.float32) / 4.0)
    if feature_set in {"scalar", "ref_query_scalar"}:
        scalar = np.column_stack(
            [
                arr["pos_norm"].astype(np.float32),
                arr["ref_strand"].astype(np.float32),
                arr["ref_mod_strand"].astype(np.float32),
            ]
        )
        pieces.append(scalar)
    return np.concatenate(pieces, axis=1)


def load_npz(path: str | Path, feature_set: str) -> TensorDataset:
    with np.load(path) as arr:
        x = build_features(arr, feature_set)
        y = arr["label"].astype(np.float32)
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


class MLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


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

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        batches += 1
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())

    out = metrics(torch.cat(logits_all), torch.cat(labels_all))
    out["loss"] = total_loss / batches
    out["batches"] = batches
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir) / args.feature_set
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_npz(args.train_npz, args.feature_set)
    val_ds = load_npz(args.val_npz, args.feature_set)
    input_dim = train_ds.tensors[0].shape[1]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = vars(args).copy()
    config["device"] = str(device)
    config["input_dim"] = int(input_dim)
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
            print(json.dumps({"feature_set": args.feature_set, **record}, indent=2))
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, out_dir / "last_model.pt")
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, out_dir / "best_model.pt")


if __name__ == "__main__":
    main()
