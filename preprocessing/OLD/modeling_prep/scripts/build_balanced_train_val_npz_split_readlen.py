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
    "label",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--val-output", required=True)
    p.add_argument("--train-rows-per-class", type=int, default=5000000)
    p.add_argument("--val-rows-per-class", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def empty_store():
    return {0: {k: [] for k in ARRAYS}, 1: {k: [] for k in ARRAYS}}

def add_rows(store, counts, label, arr, idx, need):
    if need <= 0 or idx.size == 0:
        return idx
    take = min(need, idx.size)
    chosen = idx[:take]
    for name in ARRAYS:
        store[label][name].append(arr[name][chosen])
    counts[label] += int(take)
    return idx[take:]

def write_npz(store, output, rng):
    merged = {}
    for name in ARRAYS:
        merged[name] = np.concatenate(store[0][name] + store[1][name], axis=0)
    order = rng.permutation(merged["label"].shape[0])
    for name in ARRAYS:
        merged[name] = merged[name][order]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    return int(merged["label"].shape[0])

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    paths = sorted(Path(args.input_dir).glob("chunk_*.npz"))
    train_store = empty_store()
    val_store = empty_store()
    train_counts = {0: 0, 1: 0}
    val_counts = {0: 0, 1: 0}
    chunks_used = 0

    for path in paths:
        if all(train_counts[x] >= args.train_rows_per_class for x in (0, 1)) and all(val_counts[x] >= args.val_rows_per_class for x in (0, 1)):
            break

        with np.load(path) as arr:
            chunks_used += 1
            labels = arr["label"]
            for label in (0, 1):
                idx = np.flatnonzero(labels == label)
                if idx.size == 0:
                    continue
                idx = rng.permutation(idx)

                idx = add_rows(
                    train_store,
                    train_counts,
                    label,
                    arr,
                    idx,
                    args.train_rows_per_class - train_counts[label],
                )
                add_rows(
                    val_store,
                    val_counts,
                    label,
                    arr,
                    idx,
                    args.val_rows_per_class - val_counts[label],
                )

    if train_counts[0] < args.train_rows_per_class or train_counts[1] < args.train_rows_per_class:
        raise RuntimeError(f"Not enough train rows: {train_counts}")
    if val_counts[0] < args.val_rows_per_class or val_counts[1] < args.val_rows_per_class:
        raise RuntimeError(f"Not enough val rows: {val_counts}")

    train_output = Path(args.train_output)
    val_output = Path(args.val_output)

    train_rows = write_npz(train_store, train_output, rng)
    val_rows = write_npz(val_store, val_output, rng)

    summary = {
        "input_dir": args.input_dir,
        "train_output": str(train_output),
        "val_output": str(val_output),
        "train_counts": {str(k): int(v) for k, v in train_counts.items()},
        "val_counts": {str(k): int(v) for k, v in val_counts.items()},
        "train_rows_written": train_rows,
        "val_rows_written": val_rows,
        "chunks_used": chunks_used,
        "seed": args.seed,
        "arrays": ARRAYS,
        "split_method": "Non-overlapping within chunk/class shuffled split.",
    }
    train_output.with_suffix(".split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
