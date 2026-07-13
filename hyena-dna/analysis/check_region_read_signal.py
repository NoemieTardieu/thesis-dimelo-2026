#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pysam


RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check read coverage and modkit signal for one genomic interval."
    )
    p.add_argument("--bam", required=True)
    p.add_argument("--extract-full", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--min-mapq", type=int, default=20)
    p.add_argument("--max-reads", type=int, default=None)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def sequence_in_modkit_forward_orientation(read: pysam.AlignedSegment) -> str:
    seq = read.query_sequence.upper()
    if read.is_reverse:
        return seq.translate(RC_TABLE)[::-1].upper()
    return seq


def main() -> None:
    args = parse_args()
    counters: Counter = Counter()
    read_ids: set[str] = set()
    sequences: dict[str, str] = {}
    read_lengths = []
    mapqs = []

    with pysam.AlignmentFile(args.bam, "rb") as bam:
        for read in bam.fetch(args.chrom, args.start, args.end):
            counters["bam_records_seen"] += 1
            if read.is_unmapped:
                counters["skip_unmapped"] += 1
                continue
            if read.is_secondary or read.is_supplementary:
                counters["skip_secondary_supplementary"] += 1
                continue
            if read.mapping_quality < args.min_mapq:
                counters["skip_low_mapq"] += 1
                continue
            if read.query_sequence is None or read.query_length is None:
                counters["skip_no_sequence"] += 1
                continue
            if read.query_name in read_ids:
                counters["skip_duplicate_read_id"] += 1
                continue

            read_ids.add(read.query_name)
            sequences[read.query_name] = sequence_in_modkit_forward_orientation(read)
            read_lengths.append(int(read.query_length))
            mapqs.append(int(read.mapping_quality))
            counters["eligible_reads"] += 1
            if args.max_reads is not None and len(read_ids) >= args.max_reads:
                break

    mod_probs = {"5mC_CpG": [], "5mC_nonCpG": [], "6mA": []}
    reads_with_mod_rows: set[str] = set()

    with gzip.open(args.extract_full, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            read_id = row.get("read_id", "")
            if read_id not in read_ids:
                continue
            counters["extract_rows_for_region_reads"] += 1
            reads_with_mod_rows.add(read_id)

            try:
                pos = int(row["forward_read_position"])
                prob = float(row["mod_qual"])
            except (KeyError, ValueError):
                counters["skip_bad_extract_row"] += 1
                continue
            if prob < 0:
                counters["skip_negative_mod_qual"] += 1
                continue
            prob = min(prob, 1.0)

            mod_code = row.get("mod_code", "")
            canonical_base = row.get("canonical_base", "").upper()
            seq = sequences.get(read_id, "")

            if mod_code == "a" and canonical_base == "A":
                mod_probs["6mA"].append(prob)
                counters["rows_6mA"] += 1
            elif mod_code == "m" and canonical_base == "C":
                is_cpg = pos + 1 < len(seq) and seq[pos : pos + 2] == "CG"
                if is_cpg:
                    mod_probs["5mC_CpG"].append(prob)
                    counters["rows_5mC_CpG"] += 1
                else:
                    mod_probs["5mC_nonCpG"].append(prob)
                    counters["rows_5mC_nonCpG"] += 1
            else:
                counters[f"rows_other_{canonical_base}_{mod_code}"] += 1

    signal_summary = {}
    for key, values in mod_probs.items():
        arr = np.asarray(values, dtype=float)
        signal_summary[key] = {
            "n": int(arr.size),
            "mean": None if arr.size == 0 else float(np.mean(arr)),
            "median": None if arr.size == 0 else float(np.median(arr)),
            "q95": None if arr.size == 0 else float(np.quantile(arr, 0.95)),
        }

    length_arr = np.asarray(read_lengths, dtype=float)
    result = {
        "bam": args.bam,
        "extract_full": args.extract_full,
        "region": f"{args.chrom}:{args.start}-{args.end}",
        "min_mapq": args.min_mapq,
        "max_reads": args.max_reads,
        "eligible_reads": int(len(read_ids)),
        "reads_with_modkit_rows": int(len(reads_with_mod_rows)),
        "read_length": {
            "mean": None if length_arr.size == 0 else float(np.mean(length_arr)),
            "median": None if length_arr.size == 0 else float(np.median(length_arr)),
            "max": None if length_arr.size == 0 else int(np.max(length_arr)),
            "gt_32768": int(np.sum(length_arr > 32768)) if length_arr.size else 0,
        },
        "signal": signal_summary,
        "counters": dict(counters),
    }

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
