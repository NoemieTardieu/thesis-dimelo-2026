#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np


BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
STRAND_TO_INT = {"+": 1, "-": 0}
READ_LENGTH_CLIP = 100_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode CNN regression tensors with continuous mod_qual target.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    return parser.parse_args()


def empty_arrays() -> dict[str, list]:
    return {
        "ref_kmer": [],
        "query_kmer": [],
        "pos_norm": [],
        "read_length_norm": [],
        "log_read_length_norm": [],
        "ref_strand": [],
        "ref_mod_strand": [],
        "mod_qual": [],
    }


def encode_kmer(kmer: str) -> list[int]:
    return [BASE_TO_INT.get(base, 4) for base in kmer.upper()]


def encode_strand(value: str) -> int:
    return STRAND_TO_INT.get(value, 255)


def flush_chunk(out_dir: Path, chunk_idx: int, arrays: dict[str, list]) -> int:
    row_count = len(arrays["mod_qual"])
    if row_count == 0:
        return 0

    np.savez_compressed(
        out_dir / f"chunk_{chunk_idx:05d}.npz",
        ref_kmer=np.asarray(arrays["ref_kmer"], dtype=np.uint8),
        query_kmer=np.asarray(arrays["query_kmer"], dtype=np.uint8),
        pos_norm=np.asarray(arrays["pos_norm"], dtype=np.float32),
        read_length_norm=np.asarray(arrays["read_length_norm"], dtype=np.float32),
        log_read_length_norm=np.asarray(arrays["log_read_length_norm"], dtype=np.float32),
        ref_strand=np.asarray(arrays["ref_strand"], dtype=np.uint8),
        ref_mod_strand=np.asarray(arrays["ref_mod_strand"], dtype=np.uint8),
        mod_qual=np.asarray(arrays["mod_qual"], dtype=np.float32),
    )
    return row_count


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = empty_arrays()
    chunk_idx = 0
    total_rows = 0
    mod_qual_values: list[float] = []

    with gzip.open(input_path, "rt", encoding="utf-8", newline="") as handle, open(
        out_dir / "manifest.tsv", "w", encoding="utf-8", newline=""
    ) as manifest:
        reader = csv.DictReader(handle, delimiter="\t")
        writer = csv.writer(manifest, delimiter="\t")
        writer.writerow(["chunk_file", "rows"])

        for row in reader:
            ref_kmer = row["ref_kmer"].upper()
            query_kmer = row["query_kmer"].upper()
            if len(ref_kmer) != 5 or len(query_kmer) != 5:
                continue

            read_length = int(row["read_length"])
            forward_pos = int(row["forward_read_position"])
            clipped_read_length = min(float(read_length), READ_LENGTH_CLIP)
            mod_qual = float(row["mod_qual"])

            arrays["ref_kmer"].append(encode_kmer(ref_kmer))
            arrays["query_kmer"].append(encode_kmer(query_kmer))
            arrays["pos_norm"].append(forward_pos / read_length if read_length else 0.0)
            arrays["read_length_norm"].append(clipped_read_length / READ_LENGTH_CLIP)
            arrays["log_read_length_norm"].append(math.log1p(clipped_read_length) / math.log1p(READ_LENGTH_CLIP))
            arrays["ref_strand"].append(encode_strand(row["ref_strand"]))
            arrays["ref_mod_strand"].append(encode_strand(row["ref_mod_strand"]))
            arrays["mod_qual"].append(mod_qual)
            mod_qual_values.append(mod_qual)

            if len(arrays["mod_qual"]) >= args.chunk_size:
                rows = flush_chunk(out_dir, chunk_idx, arrays)
                writer.writerow([f"chunk_{chunk_idx:05d}.npz", rows])
                total_rows += rows
                chunk_idx += 1
                arrays = empty_arrays()

        if arrays["mod_qual"]:
            rows = flush_chunk(out_dir, chunk_idx, arrays)
            writer.writerow([f"chunk_{chunk_idx:05d}.npz", rows])
            total_rows += rows
            chunk_idx += 1

    mod_qual_np = np.asarray(mod_qual_values, dtype=np.float32)
    summary = {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "chunk_size": args.chunk_size,
        "chunks_written": chunk_idx,
        "rows_written": total_rows,
        "target": "mod_qual",
        "mod_qual_summary": {
            "min": float(np.min(mod_qual_np)),
            "p25": float(np.percentile(mod_qual_np, 25)),
            "median": float(np.percentile(mod_qual_np, 50)),
            "p75": float(np.percentile(mod_qual_np, 75)),
            "max": float(np.max(mod_qual_np)),
            "mean": float(np.mean(mod_qual_np)),
        },
        "arrays": list(empty_arrays().keys()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
