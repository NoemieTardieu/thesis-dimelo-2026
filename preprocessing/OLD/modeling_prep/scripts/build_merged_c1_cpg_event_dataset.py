#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
from collections import Counter
from pathlib import Path

import pysam


TRAIN_CHROMS = {f"chr{i}" for i in range(1, 17)}
VALID_CHROMS = {"chr17", "chr18"}
TEST_CHROMS = {"chr19", "chr20", "chr21", "chr22", "chrX"}
PRIMARY_CHROMS = TRAIN_CHROMS | VALID_CHROMS | TEST_CHROMS
UNMETHYLATED_CODES = {"-"}
METHYLATED_CODES = {"m", "h"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a merged_c1 CpG event dataset for first CNN modeling."
    )
    p.add_argument(
        "--extract-calls",
        default="/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/merged_c1/extract_calls.tsv.gz",
        help="Path to modkit extract-calls table for merged_c1.",
    )
    p.add_argument(
        "--reference",
        default="/data/leuven/383/vsc38330/thesis_dimelo/src/data/hg38.fa",
        help="Reference FASTA used for sequence windows and CpG checks.",
    )
    p.add_argument(
        "--out",
        default="/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/datasets/merged_c1_cpg_event_dataset.tsv.gz",
        help="Output TSV.GZ path.",
    )
    p.add_argument(
        "--summary-out",
        default="/data/leuven/383/vsc38330/thesis_dimelo/src/modeling_prep/datasets/merged_c1_cpg_event_dataset.summary.tsv",
        help="Summary counts output path.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=101,
        help="Odd sequence window size centered on the CpG cytosine.",
    )
    p.add_argument(
        "--sample-id",
        default="merged_c1",
        help="Sample identifier to store in the output table.",
    )
    p.add_argument(
        "--cell-type",
        default="merged_c1",
        help="Cell-type label to store in the output table.",
    )
    return p.parse_args()


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def assign_split(chrom: str) -> str | None:
    if chrom in TRAIN_CHROMS:
        return "train"
    if chrom in VALID_CHROMS:
        return "valid"
    if chrom in TEST_CHROMS:
        return "test"
    return None


def main() -> None:
    args = parse_args()

    if args.window % 2 == 0:
        raise SystemExit("--window must be odd so the CpG cytosine stays centered.")

    extract_calls_path = Path(args.extract_calls)
    reference_path = Path(args.reference)
    out_path = Path(args.out)
    summary_path = Path(args.summary_out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    flank = args.window // 2
    fa = pysam.FastaFile(str(reference_path))

    fieldnames = [
        "sample_id",
        "cell_type",
        "read_id",
        "chrom",
        "pos0",
        "strand",
        "seq_window",
        "m_label",
        "m_prob",
        "split",
        "call_code",
        "canonical_base",
        "modified_primary_base",
        "ref_kmer",
        "query_kmer",
        "source_file",
        "reference_name",
    ]

    counters = Counter()

    with gzip.open(extract_calls_path, "rt", encoding="utf-8", newline="") as inp, gzip.open(
        out_path, "wt", encoding="utf-8", newline=""
    ) as out_handle:
        reader = csv.DictReader(inp, delimiter="\t")
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            counters["rows_seen"] += 1

            chrom = row["chrom"]
            if chrom not in PRIMARY_CHROMS:
                counters["skip_non_primary_chrom"] += 1
                continue

            split = assign_split(chrom)
            if split is None:
                counters["skip_no_split"] += 1
                continue

            if row["canonical_base"] != "C":
                counters["skip_non_c_base"] += 1
                continue

            if parse_bool(row["fail"]):
                counters["skip_failed_call"] += 1
                continue

            ref_position = int(row["ref_position"])
            if ref_position < 0:
                counters["skip_unmapped_ref_position"] += 1
                continue

            call_code = row["call_code"]
            if call_code in UNMETHYLATED_CODES:
                m_label = 0
            elif call_code in METHYLATED_CODES:
                m_label = 1
            else:
                counters["skip_other_call_code"] += 1
                continue

            # Restrict to CpG cytosines on the reference.
            cpg_dinuc = fa.fetch(chrom, ref_position, ref_position + 2).upper()
            if cpg_dinuc != "CG":
                counters["skip_non_cpg"] += 1
                continue

            seq_start = ref_position - flank
            seq_end = ref_position + flank + 1
            if seq_start < 0:
                counters["skip_window_left_edge"] += 1
                continue

            seq_window = fa.fetch(chrom, seq_start, seq_end).upper()
            if len(seq_window) != args.window:
                counters["skip_window_short"] += 1
                continue

            if seq_window[flank : flank + 2] != "CG":
                counters["skip_center_not_cpg"] += 1
                continue

            writer.writerow(
                {
                    "sample_id": args.sample_id,
                    "cell_type": args.cell_type,
                    "read_id": row["read_id"],
                    "chrom": chrom,
                    "pos0": ref_position,
                    "strand": row["ref_strand"],
                    "seq_window": seq_window,
                    "m_label": m_label,
                    "m_prob": row["call_prob"],
                    "split": split,
                    "call_code": call_code,
                    "canonical_base": row["canonical_base"],
                    "modified_primary_base": row["modified_primary_base"],
                    "ref_kmer": row["ref_kmer"],
                    "query_kmer": row["query_kmer"],
                    "source_file": str(extract_calls_path),
                    "reference_name": "hg38",
                }
            )
            counters["rows_written"] += 1
            counters[f"rows_written_{split}"] += 1
            counters[f"rows_written_label_{m_label}"] += 1

    with open(summary_path, "w", encoding="utf-8", newline="") as summary_handle:
        writer = csv.writer(summary_handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        for key in sorted(counters):
            writer.writerow([key, counters[key]])

    print(f"Wrote dataset: {out_path}")
    print(f"Wrote summary: {summary_path}")
    for key in sorted(counters):
        print(f"{key}\t{counters[key]}")


if __name__ == "__main__":
    main()
