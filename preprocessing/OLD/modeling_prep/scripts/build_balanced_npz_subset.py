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
    p = argparse.ArgumentParser(description="Build a balanced sampled NPZ subset from chunked NPZ data.")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rows-per-class", type=int, default=1_000_000)
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


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    selected: dict[int, dict[str, list[np.ndarray]]] = {
        0: {name: [] for name in ARRAYS},
        1: {name: [] for name in ARRAYS},
    }
    counts = {0: 0, 1: 0}
    chunks_used = 0

    for path in uniformly_spaced(chunk_paths(input_dir), args.max_chunks):
        if counts[0] >= args.rows_per_class and counts[1] >= args.rows_per_class:
            break

        with np.load(path) as arr:
            labels = arr["label"]
            chunks_used += 1

            for label in (0, 1):
                need = args.rows_per_class - counts[label]
                if need <= 0:
                    continue

                idx = np.flatnonzero(labels == label)
                if idx.size == 0:
                    continue
                if idx.size > need:
                    idx = rng.choice(idx, size=need, replace=False)

                for name in ARRAYS:
                    selected[label][name].append(arr[name][idx])
                counts[label] += idx.size

    if counts[0] == 0 or counts[1] == 0:
        raise RuntimeError(f"Could not sample both classes. Counts: {counts}")

    merged = {}
    for name in ARRAYS:
        merged[name] = np.concatenate(selected[0][name] + selected[1][name], axis=0)

    order = rng.permutation(merged["label"].shape[0])
    for name in ARRAYS:
        merged[name] = merged[name][order]

    np.savez_compressed(output, **merged)

    summary = {
        "input_dir": str(input_dir),
        "output": str(output),
        "rows_per_class_requested": args.rows_per_class,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "rows_written": int(merged["label"].shape[0]),
        "chunks_used": chunks_used,
        "max_chunks": args.max_chunks,
        "seed": args.seed,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
