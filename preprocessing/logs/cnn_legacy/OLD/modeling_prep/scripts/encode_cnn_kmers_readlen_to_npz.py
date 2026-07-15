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
READ_LENGTH_CLIP = 100000.0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chunk-size", type=int, default=2000000)
    return p.parse_args()

def encode_kmer(kmer):
    return [BASE_TO_INT.get(base, 4) for base in kmer.upper()]

def encode_strand(value):
    return STRAND_TO_INT.get(value, 255)

def empty_arrays():
    return {
        "ref_kmer": [],
        "query_kmer": [],
        "pos_norm": [],
        "read_length_norm": [],
        "log_read_length_norm": [],
        "ref_strand": [],
        "ref_mod_strand": [],
        "label": [],
    }

def flush(out_dir, chunk_idx, arrays):
    n = len(arrays["label"])
    if n == 0:
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
        label=np.asarray(arrays["label"], dtype=np.uint8),
    )
    return n

def main():
    args = parse_args()
    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = empty_arrays()
    total = 0
    chunk_idx = 0
    label_counts = {0: 0, 1: 0}

    with gzip.open(inp, "rt", encoding="utf-8", newline="") as handle, open(out_dir / "manifest.tsv", "w", encoding="utf-8", newline="") as mf:
        reader = csv.DictReader(handle, delimiter="\t")
        writer = csv.writer(mf, delimiter="\t")
        writer.writerow(["chunk_file", "rows", "label0_rows", "label1_rows"])

        for row in reader:
            ref = row["ref_kmer"].upper()
            query = row["query_kmer"].upper()
            if len(ref) != 5 or len(query) != 5:
                continue

            read_length = int(row["read_length"])
            forward_pos = int(row["forward_read_position"])
            label = int(row["label"])
            clipped_rl = min(float(read_length), READ_LENGTH_CLIP)

            arrays["ref_kmer"].append(encode_kmer(ref))
            arrays["query_kmer"].append(encode_kmer(query))
            arrays["pos_norm"].append(forward_pos / read_length if read_length else 0.0)
            arrays["read_length_norm"].append(clipped_rl / READ_LENGTH_CLIP)
            arrays["log_read_length_norm"].append(math.log1p(clipped_rl) / math.log1p(READ_LENGTH_CLIP))
            arrays["ref_strand"].append(encode_strand(row["ref_strand"]))
            arrays["ref_mod_strand"].append(encode_strand(row["ref_mod_strand"]))
            arrays["label"].append(label)
            label_counts[label] = label_counts.get(label, 0) + 1

            if len(arrays["label"]) >= args.chunk_size:
                n = flush(out_dir, chunk_idx, arrays)
                writer.writerow([f"chunk_{chunk_idx:05d}.npz", n, sum(x == 0 for x in arrays["label"]), sum(x == 1 for x in arrays["label"])])
                total += n
                chunk_idx += 1
                arrays = empty_arrays()

        if arrays["label"]:
            n = flush(out_dir, chunk_idx, arrays)
            writer.writerow([f"chunk_{chunk_idx:05d}.npz", n, sum(x == 0 for x in arrays["label"]), sum(x == 1 for x in arrays["label"])])
            total += n
            chunk_idx += 1

    summary = {
        "input": str(inp),
        "output_dir": str(out_dir),
        "chunk_size": args.chunk_size,
        "chunks_written": chunk_idx,
        "rows_written": total,
        "label_counts": label_counts,
        "extra_features": ["read_length_norm", "log_read_length_norm"],
        "read_length_clip": READ_LENGTH_CLIP,
        "arrays": ["ref_kmer", "query_kmer", "pos_norm", "read_length_norm", "log_read_length_norm", "ref_strand", "ref_mod_strand", "label"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
