#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


BASE_TO_INT = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}

STRAND_TO_INT = {
    "+": 1,
    "-": 0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode CNN-ready kmer TSV rows into compact chunked NPZ files."
    )
    p.add_argument("--input", required=True, help="Path to .tsv.gz CNN feature file")
    p.add_argument("--output-dir", required=True, help="Directory for chunked .npz files")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Number of rows per NPZ chunk",
    )
    return p.parse_args()


def encode_kmer(kmer: str) -> list[int]:
    return [BASE_TO_INT.get(base, 4) for base in kmer]


def encode_strand(value: str) -> int:
    return STRAND_TO_INT.get(value, 255)


def flush_chunk(
    out_dir: Path,
    chunk_idx: int,
    ref_kmers: list[list[int]],
    query_kmers: list[list[int]],
    positions: list[float],
    ref_strands: list[int],
    ref_mod_strands: list[int],
    labels: list[int],
) -> int:
    row_count = len(labels)
    if row_count == 0:
        return 0

    out_path = out_dir / f"chunk_{chunk_idx:05d}.npz"
    np.savez_compressed(
        out_path,
        ref_kmer=np.asarray(ref_kmers, dtype=np.uint8),
        query_kmer=np.asarray(query_kmers, dtype=np.uint8),
        pos_norm=np.asarray(positions, dtype=np.float32),
        ref_strand=np.asarray(ref_strands, dtype=np.uint8),
        ref_mod_strand=np.asarray(ref_mod_strands, dtype=np.uint8),
        label=np.asarray(labels, dtype=np.uint8),
    )
    return row_count


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.tsv"
    summary_path = out_dir / "summary.json"

    total_rows = 0
    chunk_idx = 0
    label_counts: dict[int, int] = {0: 0, 1: 0}

    ref_kmers: list[list[int]] = []
    query_kmers: list[list[int]] = []
    positions: list[float] = []
    ref_strands: list[int] = []
    ref_mod_strands: list[int] = []
    labels: list[int] = []

    with gzip.open(input_path, "rt", encoding="utf-8", newline="") as handle, open(
        manifest_path, "w", encoding="utf-8", newline=""
    ) as manifest_file:
        reader = csv.DictReader(handle, delimiter="\t")
        writer = csv.writer(manifest_file, delimiter="\t")
        writer.writerow(
            [
                "chunk_file",
                "rows",
                "label0_rows",
                "label1_rows",
            ]
        )

        for row in reader:
            ref_kmer = row["ref_kmer"].upper()
            query_kmer = row["query_kmer"].upper()

            # Guard against malformed rows even after upstream QC.
            if len(ref_kmer) != 5 or len(query_kmer) != 5:
                continue

            read_length = int(row["read_length"])
            forward_read_position = int(row["forward_read_position"])
            label = int(row["label"])

            ref_kmers.append(encode_kmer(ref_kmer))
            query_kmers.append(encode_kmer(query_kmer))
            positions.append(forward_read_position / read_length if read_length else 0.0)
            ref_strands.append(encode_strand(row["ref_strand"]))
            ref_mod_strands.append(encode_strand(row["ref_mod_strand"]))
            labels.append(label)

            label_counts[label] = label_counts.get(label, 0) + 1

            if len(labels) >= args.chunk_size:
                chunk_rows = flush_chunk(
                    out_dir,
                    chunk_idx,
                    ref_kmers,
                    query_kmers,
                    positions,
                    ref_strands,
                    ref_mod_strands,
                    labels,
                )
                writer.writerow(
                    [
                        f"chunk_{chunk_idx:05d}.npz",
                        chunk_rows,
                        sum(1 for x in labels if x == 0),
                        sum(1 for x in labels if x == 1),
                    ]
                )
                total_rows += chunk_rows
                chunk_idx += 1

                ref_kmers = []
                query_kmers = []
                positions = []
                ref_strands = []
                ref_mod_strands = []
                labels = []

        if labels:
            chunk_rows = flush_chunk(
                out_dir,
                chunk_idx,
                ref_kmers,
                query_kmers,
                positions,
                ref_strands,
                ref_mod_strands,
                labels,
            )
            writer.writerow(
                [
                    f"chunk_{chunk_idx:05d}.npz",
                    chunk_rows,
                    sum(1 for x in labels if x == 0),
                    sum(1 for x in labels if x == 1),
                ]
            )
            total_rows += chunk_rows
            chunk_idx += 1

    summary = {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "chunk_size": args.chunk_size,
        "chunks_written": chunk_idx,
        "rows_written": total_rows,
        "label_counts": label_counts,
        "encoding": {
            "bases": BASE_TO_INT,
            "ref_strand": STRAND_TO_INT,
            "ref_mod_strand": STRAND_TO_INT,
        },
        "arrays": [
            "ref_kmer",
            "query_kmer",
            "pos_norm",
            "ref_strand",
            "ref_mod_strand",
            "label",
        ],
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
