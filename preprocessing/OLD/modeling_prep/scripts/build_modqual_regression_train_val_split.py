#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARRAYS = [
    "ref_kmer",
    "query_kmer",
    "pos_norm",
    "read_length_norm",
    "log_read_length_norm",
    "ref_strand",
    "ref_mod_strand",
    "mod_qual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create non-overlapping regression train/validation NPZ files.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--train-rows", type=int, default=10_000_000)
    parser.add_argument("--val-rows", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def init_store() -> dict[str, list[np.ndarray]]:
    return {name: [] for name in ARRAYS}


def write_npz(store: dict[str, list[np.ndarray]], output: Path, rng: np.random.Generator) -> int:
    merged = {name: np.concatenate(parts, axis=0) for name, parts in store.items()}
    order = rng.permutation(merged["mod_qual"].shape[0])
    for name in ARRAYS:
        merged[name] = merged[name][order]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    return int(merged["mod_qual"].shape[0])


def allocation(total_needed: int, sizes: np.ndarray) -> np.ndarray:
    raw = total_needed * sizes / sizes.sum()
    alloc = np.floor(raw).astype(int)
    remaining = total_needed - int(alloc.sum())
    if remaining > 0:
        order = np.argsort(-(raw - alloc))
        alloc[order[:remaining]] += 1
    return alloc


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    paths = sorted(Path(args.input_dir).glob("chunk_*.npz"))
    sizes = []
    for path in paths:
        with np.load(path) as arr:
            sizes.append(int(arr["mod_qual"].shape[0]))
    sizes_np = np.asarray(sizes, dtype=np.int64)

    total_needed = args.train_rows + args.val_rows
    if total_needed > int(sizes_np.sum()):
        raise RuntimeError(f"Requested {total_needed} rows but only {int(sizes_np.sum())} available")

    per_chunk_total = allocation(total_needed, sizes_np)
    train_store = init_store()
    val_store = init_store()
    train_count = 0
    val_count = 0

    for path, take_total in zip(paths, per_chunk_total):
        if take_total <= 0:
            continue
        with np.load(path) as arr:
            n = arr["mod_qual"].shape[0]
            idx = rng.choice(n, size=min(take_total, n), replace=False)
            rng.shuffle(idx)

            train_take = min(args.train_rows - train_count, int(round(len(idx) * args.train_rows / total_needed)))
            train_idx = idx[:train_take]
            val_idx = idx[train_take:]
            if val_count + len(val_idx) > args.val_rows:
                val_idx = val_idx[: args.val_rows - val_count]

            for name in ARRAYS:
                if len(train_idx):
                    train_store[name].append(arr[name][train_idx])
                if len(val_idx):
                    val_store[name].append(arr[name][val_idx])
            train_count += len(train_idx)
            val_count += len(val_idx)

    if train_count < args.train_rows or val_count < args.val_rows:
        raise RuntimeError(f"Split too small: train={train_count}, val={val_count}")

    train_rows = write_npz(train_store, Path(args.train_output), rng)
    val_rows = write_npz(val_store, Path(args.val_output), rng)
    summary = {
        "input_dir": args.input_dir,
        "train_output": args.train_output,
        "val_output": args.val_output,
        "train_rows_written": train_rows,
        "val_rows_written": val_rows,
        "seed": args.seed,
        "arrays": ARRAYS,
        "split_method": "Proportional sampling across all chunks to avoid class/order-block bias.",
    }
    Path(args.train_output).with_suffix(".split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
