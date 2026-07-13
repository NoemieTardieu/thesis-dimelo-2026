#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pyBigWig
import pysam


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select chr16 bins with coverage and heterogeneous DiMeLo signal."
    )
    p.add_argument("--chrom", default="chr16")
    p.add_argument("--chrom-size", type=int, default=90338345)
    p.add_argument("--bin-size", type=int, default=100_000)
    p.add_argument(
        "--exclude",
        default="34000000-47000000",
        help="Comma-separated intervals to exclude, e.g. 34000000-47000000.",
    )
    p.add_argument(
        "--bam",
        default="/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam",
    )
    p.add_argument(
        "--a-bw",
        default="/staging/leuven/stg_00118/BAM_Noemie/pileup_tracks/merged_e5b/merged_e5b_A_a_all_percent.bw",
    )
    p.add_argument(
        "--cpg-bw",
        default="/staging/leuven/stg_00118/BAM_Noemie/pileup_tracks/merged_e5b/merged_e5b_C_combined_cpg_percent.bw",
    )
    p.add_argument("--min-mapq", type=int, default=20)
    p.add_argument("--max-read-length", type=int, default=32768)
    p.add_argument("--min-eligible-reads", type=int, default=20)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument(
        "--out-prefix",
        default="regions/merged_e5b_chr16_selected_100kb_top50",
    )
    return p.parse_args()


def parse_excluded_intervals(raw: str) -> list[tuple[int, int]]:
    intervals = []
    if not raw:
        return intervals
    for item in raw.split(","):
        start_s, end_s = item.split("-")
        intervals.append((int(start_s), int(end_s)))
    return intervals


def overlaps_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start < excl_end and end > excl_start for excl_start, excl_end in intervals)


def finite_stats(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.std(values)), float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))


def eligible_read_count(
    bam: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
    min_mapq: int,
    max_read_length: int,
) -> int:
    count = 0
    seen = set()
    for read in bam.fetch(chrom, start, end):
        if read.query_name in seen:
            continue
        seen.add(read.query_name)
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        if read.query_length is None or read.query_length > max_read_length:
            continue
        count += 1
    return count


def zscore(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = float(arr.std())
    if std == 0.0 or not math.isfinite(std):
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    excluded = parse_excluded_intervals(args.exclude)
    rows = []

    a_bw = pyBigWig.open(args.a_bw)
    cpg_bw = pyBigWig.open(args.cpg_bw)
    bam = pysam.AlignmentFile(args.bam, "rb")

    for start in range(0, args.chrom_size, args.bin_size):
        end = min(start + args.bin_size, args.chrom_size)
        if overlaps_any(start, end, excluded):
            continue

        try:
            a_values = np.asarray(a_bw.values(args.chrom, start, end, numpy=True), dtype=float)
            cpg_values = np.asarray(cpg_bw.values(args.chrom, start, end, numpy=True), dtype=float)
        except RuntimeError:
            continue

        a_mean, a_std, a_range = finite_stats(a_values)
        cpg_mean, cpg_std, cpg_range = finite_stats(cpg_values)
        reads = eligible_read_count(
            bam,
            args.chrom,
            start,
            end,
            args.min_mapq,
            args.max_read_length,
        )

        rows.append(
            {
                "chrom": args.chrom,
                "start": start,
                "end": end,
                "eligible_reads": reads,
                "a_mean": a_mean,
                "a_std": a_std,
                "a_q95_q05": a_range,
                "cpg_mean": cpg_mean,
                "cpg_std": cpg_std,
                "cpg_q95_q05": cpg_range,
            }
        )

    bam.close()
    a_bw.close()
    cpg_bw.close()

    eligible = [r for r in rows if r["eligible_reads"] >= args.min_eligible_reads]
    if not eligible:
        raise SystemExit("No bins passed the eligible read threshold.")

    a_z = zscore([r["a_q95_q05"] for r in eligible])
    cpg_z = zscore([r["cpg_q95_q05"] for r in eligible])
    read_z = zscore([math.log1p(r["eligible_reads"]) for r in eligible])
    for i, row in enumerate(eligible):
        row["score"] = float(a_z[i] + cpg_z[i] + 0.5 * read_z[i])

    eligible.sort(key=lambda r: r["score"], reverse=True)
    selected = eligible[: args.top_n]

    tsv_path = out_prefix.with_suffix(".ranking.tsv")
    bed_path = out_prefix.with_suffix(".bed")

    fieldnames = [
        "chrom",
        "start",
        "end",
        "score",
        "eligible_reads",
        "a_mean",
        "a_std",
        "a_q95_q05",
        "cpg_mean",
        "cpg_std",
        "cpg_q95_q05",
    ]
    with open(tsv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(eligible)

    with open(bed_path, "w", encoding="utf-8") as handle:
        for rank, row in enumerate(selected, start=1):
            name = f"chr16_rank{rank}_score{row['score']:.3f}_reads{row['eligible_reads']}"
            handle.write(f"{row['chrom']}\t{row['start']}\t{row['end']}\t{name}\n")

    print(f"ranked_bins\t{len(eligible)}")
    print(f"selected_bins\t{len(selected)}")
    print(f"ranking_tsv\t{tsv_path}")
    print(f"selected_bed\t{bed_path}")
    print("top_regions")
    for row in selected[:10]:
        print(
            f"{row['chrom']}:{row['start']}-{row['end']}\t"
            f"score={row['score']:.3f}\treads={row['eligible_reads']}\t"
            f"A_range={row['a_q95_q05']:.3f}\tCpG_range={row['cpg_q95_q05']:.3f}"
        )


if __name__ == "__main__":
    main()
