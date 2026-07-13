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
    parser = argparse.ArgumentParser(description="Sample rows from chunked mod_qual regression tensors into one NPZ.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
    if args.rows > int(sizes_np.sum()):
        raise RuntimeError(f"Requested {args.rows} rows but only {int(sizes_np.sum())} available")

    alloc = allocation(args.rows, sizes_np)
    store = {name: [] for name in ARRAYS}
    for path, take in zip(paths, alloc):
        if take <= 0:
            continue
        with np.load(path) as arr:
            idx = rng.choice(arr["mod_qual"].shape[0], size=int(take), replace=False)
            for name in ARRAYS:
                store[name].append(arr[name][idx])

    merged = {name: np.concatenate(parts, axis=0) for name, parts in store.items()}
    order = rng.permutation(merged["mod_qual"].shape[0])
    for name in ARRAYS:
        merged[name] = merged[name][order]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    summary = {
        "input_dir": args.input_dir,
        "output": str(output),
        "rows_requested": args.rows,
        "rows_written": int(merged["mod_qual"].shape[0]),
        "source_rows_available": int(sizes_np.sum()),
        "chunks_used": int(np.sum(alloc > 0)),
        "seed": args.seed,
        "arrays": ARRAYS,
    }
    output.with_suffix(".sample_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
