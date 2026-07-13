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
    parser = argparse.ArgumentParser(description="Encode mod_qual regression tensors with chromosome split.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--train-chroms", default="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16")
    parser.add_argument("--val-chroms", default="chr17,chr18")
    parser.add_argument("--test-chroms", default="chr19,chr20,chr21,chr22")
    return parser.parse_args()


def split_chroms(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def empty_arrays() -> dict[str, list]:
    return {name: [] for name in ARRAYS}


def encode_kmer(kmer: str) -> list[int]:
    return [BASE_TO_INT.get(base, 4) for base in kmer.upper()]


def encode_strand(value: str) -> int:
    return STRAND_TO_INT.get(value, 255)


def flush_chunk(out_dir: Path, chunk_idx: int, arrays: dict[str, list]) -> int:
    rows = len(arrays["mod_qual"])
    if rows == 0:
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
    return rows


def main() -> None:
    args = parse_args()
    train_chroms = split_chroms(args.train_chroms)
    val_chroms = split_chroms(args.val_chroms)
    test_chroms = split_chroms(args.test_chroms)
    split_sets = {"train": train_chroms, "val": val_chroms, "test": test_chroms}

    output_base = Path(args.output_base)
    dirs = {split: output_base / split for split in split_sets}
    for out_dir in dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {split: empty_arrays() for split in split_sets}
    chunk_idx = {split: 0 for split in split_sets}
    rows_written = {split: 0 for split in split_sets}
    chrom_counts: dict[str, int] = {}
    mod_qual_values = {split: [] for split in split_sets}

    manifests = {
        split: open(dirs[split] / "manifest.tsv", "w", encoding="utf-8", newline="")
        for split in split_sets
    }
    writers = {split: csv.writer(handle, delimiter="\t") for split, handle in manifests.items()}
    for writer in writers.values():
        writer.writerow(["chunk_file", "rows"])

    try:
        with gzip.open(args.input, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                chrom = row["chrom"]
                split = None
                for name, chroms in split_sets.items():
                    if chrom in chroms:
                        split = name
                        break
                if split is None:
                    continue

                ref_kmer = row["ref_kmer"].upper()
                query_kmer = row["query_kmer"].upper()
                if len(ref_kmer) != 5 or len(query_kmer) != 5:
                    continue

                read_length = int(row["read_length"])
                forward_pos = int(row["forward_read_position"])
                clipped_read_length = min(float(read_length), READ_LENGTH_CLIP)
                mod_qual = float(row["mod_qual"])

                a = arrays[split]
                a["ref_kmer"].append(encode_kmer(ref_kmer))
                a["query_kmer"].append(encode_kmer(query_kmer))
                a["pos_norm"].append(forward_pos / read_length if read_length else 0.0)
                a["read_length_norm"].append(clipped_read_length / READ_LENGTH_CLIP)
                a["log_read_length_norm"].append(math.log1p(clipped_read_length) / math.log1p(READ_LENGTH_CLIP))
                a["ref_strand"].append(encode_strand(row["ref_strand"]))
                a["ref_mod_strand"].append(encode_strand(row["ref_mod_strand"]))
                a["mod_qual"].append(mod_qual)

                chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
                mod_qual_values[split].append(mod_qual)

                if len(a["mod_qual"]) >= args.chunk_size:
                    rows = flush_chunk(dirs[split], chunk_idx[split], a)
                    writers[split].writerow([f"chunk_{chunk_idx[split]:05d}.npz", rows])
                    rows_written[split] += rows
                    chunk_idx[split] += 1
                    arrays[split] = empty_arrays()

        for split, a in arrays.items():
            if a["mod_qual"]:
                rows = flush_chunk(dirs[split], chunk_idx[split], a)
                writers[split].writerow([f"chunk_{chunk_idx[split]:05d}.npz", rows])
                rows_written[split] += rows
                chunk_idx[split] += 1
    finally:
        for handle in manifests.values():
            handle.close()

    summary = {
        "input": args.input,
        "output_base": str(output_base),
        "chunk_size": args.chunk_size,
        "chrom_split": {
            "train": sorted(train_chroms),
            "val": sorted(val_chroms),
            "test": sorted(test_chroms),
        },
        "rows_written": rows_written,
        "chunks_written": chunk_idx,
        "chrom_counts": chrom_counts,
        "target": "mod_qual",
        "arrays": ARRAYS,
    }
    for split, values in mod_qual_values.items():
        if values:
            arr = np.asarray(values, dtype=np.float32)
            summary[f"{split}_mod_qual_summary"] = {
                "min": float(np.min(arr)),
                "p25": float(np.percentile(arr, 25)),
                "median": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }
    (output_base / "chrom_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
