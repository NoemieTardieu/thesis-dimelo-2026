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


BASE_TO_ID = {
    "A": 7,
    "C": 8,
    "G": 9,
    "T": 10,
    "N": 11,
}
PAD_ID = 4
UNK_ID = 6
IGNORE_VALUE = -100.0
RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build first dense HyenaDNA-style tensors from merged_e5b chr16 extract-full output."
    )
    p.add_argument(
        "--bam",
        default="/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam",
        help="Input BAM with read sequences.",
    )
    p.add_argument(
        "--extract-full",
        default="/staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/by_chrom/extract_full_chr16.tsv.gz",
        help="modkit extract full TSV.GZ for chr16.",
    )
    p.add_argument("--chrom", default="chr16", help="Chromosome to process.")
    p.add_argument(
        "--regions-bed",
        default=None,
        help="Optional BED file. If set, collect eligible reads overlapping these regions.",
    )
    p.add_argument("--max-length", type=int, default=32768, help="Fixed tensor length.")
    p.add_argument(
        "--long-read-mode",
        choices=["skip", "truncate", "nonoverlap", "overlap"],
        default="skip",
        help=(
            "How to handle reads longer than max-length. "
            "skip preserves the original behavior; truncate keeps the first window; "
            "nonoverlap creates consecutive windows; overlap creates sliding windows."
        ),
    )
    p.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help=(
            "Stride for --long-read-mode overlap. Defaults to max-length / 2. "
            "Ignored for skip/truncate/nonoverlap."
        ),
    )
    p.add_argument("--max-reads", type=int, default=1000, help="Number of eligible reads to collect.")
    p.add_argument(
        "--max-reads-per-region",
        type=int,
        default=None,
        help="Optional cap on reads collected from each BED region.",
    )
    p.add_argument("--min-mapq", type=int, default=20, help="Minimum mapping quality.")
    p.add_argument(
        "--out-prefix",
        default="outputs/merged_e5b_chr16_first1000",
        help="Output prefix for .npz, .metadata.tsv, and .summary.json.",
    )
    p.add_argument(
        "--exclude-inferred",
        action="store_true",
        help="Exclude modkit rows marked inferred=true from target masks.",
    )
    p.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Do not stop early while streaming extract-full after selected read region.",
    )
    return p.parse_args()


def encode_sequence(seq: str, max_length: int) -> np.ndarray:
    out = np.full((max_length,), PAD_ID, dtype=np.uint8)
    seq = seq.upper()[:max_length]
    for i, base in enumerate(seq):
        out[i] = BASE_TO_ID.get(base, UNK_ID)
    return out


def sequence_in_modkit_forward_orientation(read: pysam.AlignedSegment) -> str:
    seq = read.query_sequence.upper()
    if read.is_reverse:
        return seq.translate(RC_TABLE)[::-1].upper()
    return seq


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def load_regions(path: str | None, chrom: str) -> list[tuple[str, int, int]]:
    if path is None:
        return [(chrom, 0, 2**31 - 1)]

    regions = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            region_chrom = parts[0]
            if region_chrom != chrom:
                continue
            regions.append((region_chrom, int(parts[1]), int(parts[2])))

    if not regions:
        raise SystemExit(f"No regions for {chrom} found in BED file: {path}")
    return regions


def read_window_starts(read_length: int, max_length: int, mode: str, stride: int) -> list[int]:
    if read_length <= max_length:
        return [0]
    if mode == "skip":
        return []
    if mode == "truncate":
        return [0]
    if mode == "nonoverlap":
        return list(range(0, read_length, max_length))
    if mode == "overlap":
        starts = list(range(0, read_length, stride))
        last_start = max(0, read_length - max_length)
        if starts[-1] != last_start:
            starts.append(last_start)
        return sorted(set(starts))
    raise ValueError(f"Unsupported long-read mode: {mode}")


def collect_reads(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, list[int]], Counter]:
    reads: list[dict[str, object]] = []
    read_index: dict[str, list[int]] = {}
    seen_windows: set[tuple[str, int]] = set()
    counters: Counter = Counter()
    regions = load_regions(args.regions_bed, args.chrom)
    stride = args.window_stride or max(1, args.max_length // 2)
    if args.long_read_mode == "overlap" and stride <= 0:
        raise SystemExit("--window-stride must be positive for overlap mode.")

    with pysam.AlignmentFile(args.bam, "rb") as bam:
        for region_chrom, region_start, region_end in regions:
            counters["regions_seen"] += 1
            reads_from_region = 0
            for read in bam.fetch(region_chrom, region_start, region_end):
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
                window_starts = read_window_starts(
                    int(read.query_length), args.max_length, args.long_read_mode, stride
                )
                if not window_starts:
                    counters["skip_long_read"] += 1
                    continue

                sequence = sequence_in_modkit_forward_orientation(read)
                for window_start in window_starts:
                    if (read.query_name, window_start) in seen_windows:
                        counters["skip_duplicate_read_window"] += 1
                        continue
                    window_end = min(window_start + args.max_length, int(read.query_length))
                    window_seq = sequence[window_start:window_end]
                    if not window_seq:
                        counters["skip_empty_window"] += 1
                        continue

                    seen_windows.add((read.query_name, window_start))
                    idx = len(reads)
                    read_index.setdefault(read.query_name, []).append(idx)
                    reads.append(
                        {
                            "read_id": read.query_name,
                            "chrom": read.reference_name,
                            "alignment_start": int(read.reference_start),
                            "alignment_end": int(read.reference_end),
                            "read_length": int(read.query_length),
                            "window_start": int(window_start),
                            "window_end": int(window_end),
                            "window_length": int(window_end - window_start),
                            "is_reverse": bool(read.is_reverse),
                            "mapq": int(read.mapping_quality),
                            "sequence": window_seq,
                        }
                    )
                    counters["reads_collected"] += 1
                    counters["windows_collected"] += 1
                    if read.query_length > args.max_length:
                        counters["long_read_windows_collected"] += 1
                    reads_from_region += 1

                    if len(reads) >= args.max_reads:
                        return reads, read_index, counters
                    if (
                        args.max_reads_per_region is not None
                        and reads_from_region >= args.max_reads_per_region
                    ):
                        counters["regions_reached_read_cap"] += 1
                        break
                if (
                    args.max_reads_per_region is not None
                    and reads_from_region >= args.max_reads_per_region
                ):
                    break

    return reads, read_index, counters


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    reads, read_index, counters = collect_reads(args)
    if not reads:
        raise SystemExit("No reads collected. Check BAM path, chromosome, and filters.")

    n = len(reads)
    max_length = args.max_length
    stride = args.window_stride or max(1, args.max_length // 2)

    input_ids = np.full((n, max_length), PAD_ID, dtype=np.uint8)
    target_5mc = np.full((n, max_length), IGNORE_VALUE, dtype=np.float16)
    target_6ma = np.full((n, max_length), IGNORE_VALUE, dtype=np.float16)
    mask_5mc = np.zeros((n, max_length), dtype=np.uint8)
    mask_6ma = np.zeros((n, max_length), dtype=np.uint8)

    sequences = []
    for i, read in enumerate(reads):
        seq = str(read["sequence"])
        sequences.append(seq)
        input_ids[i] = encode_sequence(seq, max_length)

    max_selected_alignment_end = max(int(r["alignment_end"]) for r in reads)
    early_stop_after = max_selected_alignment_end + 100_000

    with gzip.open(args.extract_full, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            counters["extract_rows_seen"] += 1

            read_id = row["read_id"]
            candidate_indices = read_index.get(read_id)
            if candidate_indices is None:
                if not args.no_early_stop:
                    try:
                        aln_start = int(row["alignment_start"])
                    except ValueError:
                        aln_start = -1
                    if aln_start > early_stop_after:
                        counters["early_stop_after_alignment_start"] = aln_start
                        break
                continue

            if args.exclude_inferred and parse_bool(row["inferred"]):
                counters["skip_inferred"] += 1
                continue

            try:
                pos = int(row["forward_read_position"])
            except ValueError:
                counters["skip_bad_position"] += 1
                continue

            mod_code = row["mod_code"]
            canonical_base = row["canonical_base"].upper()
            try:
                prob = float(row["mod_qual"])
            except ValueError:
                counters["skip_bad_mod_qual"] += 1
                continue
            if prob < 0.0:
                counters["skip_negative_mod_qual"] += 1
                continue
            prob = min(prob, 1.0)

            target_set = False
            position_seen_in_any_window = False
            for idx in candidate_indices:
                local_pos = pos - int(reads[idx]["window_start"])
                if local_pos < 0 or local_pos >= max_length or local_pos >= len(sequences[idx]):
                    continue
                position_seen_in_any_window = True

                seq = sequences[idx]
                seq_base = seq[local_pos]
                if seq_base != canonical_base:
                    counters[f"base_mismatch_{canonical_base}_vs_{seq_base}"] += 1

                if mod_code == "a" and canonical_base == "A":
                    target_6ma[idx, local_pos] = prob
                    mask_6ma[idx, local_pos] = 1
                    counters["target_6ma_set"] += 1
                    target_set = True
                elif mod_code == "m" and canonical_base == "C":
                    is_cpg = local_pos + 1 < len(seq) and seq[local_pos : local_pos + 2] == "CG"
                    if is_cpg:
                        target_5mc[idx, local_pos] = prob
                        mask_5mc[idx, local_pos] = 1
                        counters["target_5mc_cpg_set"] += 1
                        target_set = True
                    else:
                        counters["skip_c_non_cpg"] += 1
                else:
                    counters[f"skip_mod_code_{canonical_base}_{mod_code}"] += 1
            if not position_seen_in_any_window:
                counters["skip_position_out_of_range"] += 1
            elif not target_set:
                counters["rows_seen_for_selected_windows_but_no_target_set"] += 1

    metadata_path = out_prefix.with_suffix(".metadata.tsv")
    with open(metadata_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "row_idx",
            "read_id",
            "chrom",
            "alignment_start",
            "alignment_end",
            "read_length",
            "window_start",
            "window_end",
            "window_length",
            "is_reverse",
            "mapq",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for i, read in enumerate(reads):
            row = {k: read[k] for k in fieldnames if k != "row_idx"}
            row["row_idx"] = i
            writer.writerow(row)

    npz_path = out_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        input_ids=input_ids,
        target_5mC=target_5mc,
        mask_5mC=mask_5mc,
        target_6mA=target_6ma,
        mask_6mA=mask_6ma,
    )

    summary = {
        "bam": args.bam,
        "extract_full": args.extract_full,
        "chrom": args.chrom,
        "regions_bed": args.regions_bed,
        "max_length": max_length,
        "long_read_mode": args.long_read_mode,
        "window_stride": stride if args.long_read_mode == "overlap" else None,
        "max_reads": args.max_reads,
        "max_reads_per_region": args.max_reads_per_region,
        "n_reads": n,
        "input_ids_shape": list(input_ids.shape),
        "target_5mC_shape": list(target_5mc.shape),
        "target_6mA_shape": list(target_6ma.shape),
        "valid_5mC_targets": int(mask_5mc.sum()),
        "valid_6mA_targets": int(mask_6ma.sum()),
        "reads_with_5mC_targets": int((mask_5mc.sum(axis=1) > 0).sum()),
        "reads_with_6mA_targets": int((mask_6ma.sum(axis=1) > 0).sum()),
        "counters": dict(counters),
        "outputs": {
            "npz": str(npz_path),
            "metadata": str(metadata_path),
            "summary": str(out_prefix.with_suffix(".summary.json")),
        },
    }

    summary_path = out_prefix.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
