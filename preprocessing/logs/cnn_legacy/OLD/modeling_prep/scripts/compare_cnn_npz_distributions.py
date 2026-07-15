#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BASES = np.asarray(["A", "C", "G", "T", "N"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare sampled CNN NPZ feature distributions.")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-chunks", type=int, default=32)
    p.add_argument("--max-rows-per-chunk", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def chunk_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("chunk_*.npz"))


def uniformly_spaced(paths: list[Path], n: int) -> list[Path]:
    if n >= len(paths):
        return paths
    idx = np.linspace(0, len(paths) - 1, n, dtype=int)
    return [paths[i] for i in idx]


def decode_kmers(encoded: np.ndarray) -> list[str]:
    clipped = np.clip(encoded, 0, 4)
    return ["".join(BASES[row]) for row in clipped]


def summarize_dataset(
    directory: Path,
    max_chunks: int,
    max_rows_per_chunk: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    paths = uniformly_spaced(chunk_paths(directory), max_chunks)

    label_counts: Counter[int] = Counter()
    ref_strand_counts: Counter[int] = Counter()
    ref_mod_strand_counts: Counter[int] = Counter()
    ref_kmer_counts: Counter[str] = Counter()
    query_kmer_counts: Counter[str] = Counter()
    ref_query_pair_counts: Counter[str] = Counter()
    label_by_ref_kmer: dict[str, Counter[int]] = defaultdict(Counter)
    label_by_query_kmer: dict[str, Counter[int]] = defaultdict(Counter)

    pos_norm_values: list[np.ndarray] = []
    rows_sampled = 0

    for path in paths:
        with np.load(path) as arr:
            n = arr["label"].shape[0]
            if n > max_rows_per_chunk:
                idx = rng.choice(n, size=max_rows_per_chunk, replace=False)
            else:
                idx = np.arange(n)

            labels = arr["label"][idx].astype(int)
            ref_kmers = decode_kmers(arr["ref_kmer"][idx])
            query_kmers = decode_kmers(arr["query_kmer"][idx])
            ref_strands = arr["ref_strand"][idx].astype(int)
            ref_mod_strands = arr["ref_mod_strand"][idx].astype(int)
            pos_norm = arr["pos_norm"][idx]

            rows_sampled += labels.shape[0]
            label_counts.update(labels.tolist())
            ref_strand_counts.update(ref_strands.tolist())
            ref_mod_strand_counts.update(ref_mod_strands.tolist())
            ref_kmer_counts.update(ref_kmers)
            query_kmer_counts.update(query_kmers)
            ref_query_pair_counts.update(f"{r}|{q}" for r, q in zip(ref_kmers, query_kmers))
            pos_norm_values.append(pos_norm)

            for ref, query, label in zip(ref_kmers, query_kmers, labels):
                label_by_ref_kmer[ref][int(label)] += 1
                label_by_query_kmer[query][int(label)] += 1

    pos_norm_all = np.concatenate(pos_norm_values) if pos_norm_values else np.asarray([])

    def top_counter(counter: Counter, n: int = 25) -> list[dict]:
        total = sum(counter.values())
        return [
            {"value": str(k), "count": int(v), "fraction": float(v / total) if total else 0.0}
            for k, v in counter.most_common(n)
        ]

    def top_label_rates(label_by_key: dict[str, Counter[int]], n: int = 25) -> list[dict]:
        rows = []
        for key, counts in label_by_key.items():
            total = counts[0] + counts[1]
            if total == 0:
                continue
            rows.append(
                {
                    "value": key,
                    "count": int(total),
                    "label1_fraction": float(counts[1] / total),
                    "label0": int(counts[0]),
                    "label1": int(counts[1]),
                }
            )
        rows.sort(key=lambda x: x["count"], reverse=True)
        return rows[:n]

    label_total = sum(label_counts.values())
    return {
        "directory": str(directory),
        "chunks_sampled": len(paths),
        "rows_sampled": rows_sampled,
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "label_fractions": {
            str(k): float(v / label_total) if label_total else 0.0
            for k, v in sorted(label_counts.items())
        },
        "ref_strand_counts": {str(k): int(v) for k, v in sorted(ref_strand_counts.items())},
        "ref_mod_strand_counts": {str(k): int(v) for k, v in sorted(ref_mod_strand_counts.items())},
        "pos_norm": {
            "min": float(np.min(pos_norm_all)) if pos_norm_all.size else None,
            "p01": float(np.percentile(pos_norm_all, 1)) if pos_norm_all.size else None,
            "p05": float(np.percentile(pos_norm_all, 5)) if pos_norm_all.size else None,
            "p25": float(np.percentile(pos_norm_all, 25)) if pos_norm_all.size else None,
            "median": float(np.percentile(pos_norm_all, 50)) if pos_norm_all.size else None,
            "p75": float(np.percentile(pos_norm_all, 75)) if pos_norm_all.size else None,
            "p95": float(np.percentile(pos_norm_all, 95)) if pos_norm_all.size else None,
            "p99": float(np.percentile(pos_norm_all, 99)) if pos_norm_all.size else None,
            "max": float(np.max(pos_norm_all)) if pos_norm_all.size else None,
            "mean": float(np.mean(pos_norm_all)) if pos_norm_all.size else None,
        },
        "top_ref_kmers": top_counter(ref_kmer_counts),
        "top_query_kmers": top_counter(query_kmer_counts),
        "top_ref_query_pairs": top_counter(ref_query_pair_counts),
        "top_ref_kmers_label_rates": top_label_rates(label_by_ref_kmer),
        "top_query_kmers_label_rates": top_label_rates(label_by_query_kmer),
    }


def main() -> None:
    args = parse_args()
    train = summarize_dataset(
        Path(args.train_dir),
        args.max_chunks,
        args.max_rows_per_chunk,
        args.seed,
    )
    val = summarize_dataset(
        Path(args.val_dir),
        args.max_chunks,
        args.max_rows_per_chunk,
        args.seed + 1,
    )
    result = {
        "parameters": {
            "max_chunks": args.max_chunks,
            "max_rows_per_chunk": args.max_rows_per_chunk,
            "seed": args.seed,
        },
        "train": train,
        "validation": val,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
