#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pysam


DEFAULT_BINS = [
    0,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    32_000,
    50_000,
    100_000,
    200_000,
    500_000,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute ONT BAM read-length statistics and retention at a target window length."
    )
    p.add_argument("--bam", required=True, help="Input BAM.")
    p.add_argument("--out-prefix", required=True, help="Output prefix for .summary.json and .hist.tsv.")
    p.add_argument("--length-cutoff", type=int, default=32_000, help="Read length cutoff of interest.")
    p.add_argument("--max-reads", type=int, default=None, help="Optional maximum primary reads for quick sampling.")
    p.add_argument(
        "--include-unmapped",
        action="store_true",
        help="Include unmapped primary reads. By default only mapped primary reads are counted.",
    )
    p.add_argument(
        "--include-secondary-supplementary",
        action="store_true",
        help="Include secondary/supplementary alignments. Default excludes them.",
    )
    return p.parse_args()


def read_length(read: pysam.AlignedSegment) -> int:
    if read.query_length is not None:
        return int(read.query_length)
    if read.query_sequence is not None:
        return len(read.query_sequence)
    return 0


def main() -> None:
    args = parse_args()
    bam_path = Path(args.bam)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    lengths: list[int] = []
    counters: Counter[str] = Counter()

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(until_eof=True):
            counters["records_seen"] += 1

            if not args.include_secondary_supplementary and (
                read.is_secondary or read.is_supplementary
            ):
                counters["skip_secondary_supplementary"] += 1
                continue
            if not args.include_unmapped and read.is_unmapped:
                counters["skip_unmapped"] += 1
                continue

            length = read_length(read)
            if length <= 0:
                counters["skip_no_sequence_length"] += 1
                continue

            lengths.append(length)
            counters["primary_reads_counted"] += 1

            if args.max_reads is not None and len(lengths) >= args.max_reads:
                counters["stopped_at_max_reads"] = 1
                break

    if not lengths:
        raise SystemExit("No reads counted after filters.")

    arr = np.asarray(lengths, dtype=np.int64)
    cutoff = int(args.length_cutoff)
    keep = arr <= cutoff

    quantile_probs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quantiles = {
        f"q{int(q * 100):02d}": int(np.quantile(arr, q, method="nearest"))
        for q in quantile_probs
    }

    summary = {
        "bam": str(bam_path),
        "length_cutoff": cutoff,
        "max_reads": args.max_reads,
        "filters": {
            "include_unmapped": args.include_unmapped,
            "include_secondary_supplementary": args.include_secondary_supplementary,
        },
        "counters": dict(counters),
        "n_reads": int(arr.size),
        "total_read_bases": int(arr.sum()),
        "n_reads_le_cutoff": int(keep.sum()),
        "fraction_reads_le_cutoff": float(keep.mean()),
        "bases_in_reads_le_cutoff": int(arr[keep].sum()),
        "fraction_bases_in_reads_le_cutoff": float(arr[keep].sum() / arr.sum()),
        "n_reads_gt_cutoff": int((~keep).sum()),
        "min_length": int(arr.min()),
        "mean_length": float(arr.mean()),
        "median_length": float(np.median(arr)),
        "max_length": int(arr.max()),
        "quantiles": quantiles,
    }

    with open(f"{out_prefix}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    bins = np.asarray(DEFAULT_BINS, dtype=np.int64)
    if cutoff not in bins:
        bins = np.sort(np.unique(np.concatenate([bins, np.asarray([cutoff])])))
    hist, edges = np.histogram(arr, bins=bins)

    with open(f"{out_prefix}.hist.tsv", "w", encoding="utf-8") as handle:
        handle.write("bin_start\tbin_end\tn_reads\n")
        for start, end, count in zip(edges[:-1], edges[1:], hist):
            handle.write(f"{int(start)}\t{int(end)}\t{int(count)}\n")
        handle.write(f"{int(edges[-1])}\tinf\t{int((arr >= edges[-1]).sum())}\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
