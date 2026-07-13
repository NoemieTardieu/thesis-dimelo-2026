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
    "ref_strand",
    "ref_mod_strand",
    "label",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build non-overlapping balanced train/validation NPZ subsets from "
            "chunked CNN tensor files."
        )
    )
    p.add_argument("--input-dir", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--val-output", required=True)
    p.add_argument("--train-rows-per-class", type=int, default=5_000_000)
    p.add_argument("--val-rows-per-class", type=int, default=1_000_000)
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def chunk_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("chunk_*.npz"))


def uniformly_spaced(paths: list[Path], max_chunks: int | None) -> list[Path]:
    if max_chunks is None or max_chunks >= len(paths):
        return paths
    idx = np.linspace(0, len(paths) - 1, max_chunks, dtype=int)
    return [paths[i] for i in idx]


def init_store() -> dict[int, dict[str, list[np.ndarray]]]:
    return {
        0: {name: [] for name in ARRAYS},
        1: {name: [] for name in ARRAYS},
    }


def add_rows(
    store: dict[int, dict[str, list[np.ndarray]]],
    counts: dict[int, int],
    label: int,
    arrays: np.lib.npyio.NpzFile,
    idx: np.ndarray,
    need: int,
) -> np.ndarray:
    if need <= 0 or idx.size == 0:
        return idx
    take = min(need, idx.size)
    chosen = idx[:take]
    for name in ARRAYS:
        store[label][name].append(arrays[name][chosen])
    counts[label] += int(take)
    return idx[take:]


def merge_and_write(
    store: dict[int, dict[str, list[np.ndarray]]],
    output: Path,
    rng: np.random.Generator,
) -> int:
    merged = {}
    for name in ARRAYS:
        parts = store[0][name] + store[1][name]
        if not parts:
            raise RuntimeError(f"No arrays collected for {name}")
        merged[name] = np.concatenate(parts, axis=0)

    order = rng.permutation(merged["label"].shape[0])
    for name in ARRAYS:
        merged[name] = merged[name][order]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    return int(merged["label"].shape[0])


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    input_dir = Path(args.input_dir)
    train_output = Path(args.train_output)
    val_output = Path(args.val_output)

    train_store = init_store()
    val_store = init_store()
    train_counts = {0: 0, 1: 0}
    val_counts = {0: 0, 1: 0}
    train_target = args.train_rows_per_class
    val_target = args.val_rows_per_class
    chunks_used = 0

    for path in uniformly_spaced(chunk_paths(input_dir), args.max_chunks):
        train_done = train_counts[0] >= train_target and train_counts[1] >= train_target
        val_done = val_counts[0] >= val_target and val_counts[1] >= val_target
        if train_done and val_done:
            break

        with np.load(path) as arr:
            labels = arr["label"]
            chunks_used += 1

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
                    train_target - train_counts[label],
                )
                add_rows(
                    val_store,
                    val_counts,
                    label,
                    arr,
                    idx,
                    val_target - val_counts[label],
                )

    if train_counts[0] < train_target or train_counts[1] < train_target:
        raise RuntimeError(f"Not enough train rows collected: {train_counts}")
    if val_counts[0] < val_target or val_counts[1] < val_target:
        raise RuntimeError(f"Not enough validation rows collected: {val_counts}")

    train_rows = merge_and_write(train_store, train_output, rng)
    val_rows = merge_and_write(val_store, val_output, rng)

    summary = {
        "input_dir": str(input_dir),
        "train_output": str(train_output),
        "val_output": str(val_output),
        "train_rows_per_class_requested": train_target,
        "val_rows_per_class_requested": val_target,
        "train_counts": {str(k): int(v) for k, v in train_counts.items()},
        "val_counts": {str(k): int(v) for k, v in val_counts.items()},
        "train_rows_written": train_rows,
        "val_rows_written": val_rows,
        "chunks_used": chunks_used,
        "max_chunks": args.max_chunks,
        "seed": args.seed,
        "split_method": "Within each chunk and class, shuffled indices are assigned to train first, then validation; rows are not reused.",
    }
    summary_path = train_output.with_suffix(".split_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
