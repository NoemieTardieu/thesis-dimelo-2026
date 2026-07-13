#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam


CIGAR_MATCH = {0, 7, 8}
CIGAR_INS = 1
CIGAR_DEL = 2
CIGAR_REF_SKIP = 3
CIGAR_SOFT_CLIP = 4
CIGAR_HARD_CLIP = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CIGAR/DNA heterogeneity for high- and low-target-variance "
            "loci. This tests whether read-to-read target variance can be "
            "explained by variation in the underlying sequenced DNA/alignment."
        )
    )
    parser.add_argument("--variance-tsv", required=True, help="*.per_locus_variance.tsv.gz")
    parser.add_argument("--bam", required=True)
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--group", default="merged_c1")
    parser.add_argument("--chrom", default="chr16")
    parser.add_argument("--marks", nargs="+", default=["5mC", "6mA"])
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--bottom-n", type=int, default=500)
    parser.add_argument("--min-reads", type=int, default=10)
    parser.add_argument(
        "--max-reads-per-locus",
        type=int,
        default=250,
        help="Cap reads inspected per locus to keep pathological high-depth loci cheap.",
    )
    parser.add_argument(
        "--local-window",
        type=int,
        default=20,
        help="Reference bases on each side of the locus used for local mismatch/indel summaries.",
    )
    parser.add_argument(
        "--insertion-window",
        type=int,
        default=5,
        help="Distance from locus at which an insertion is counted as nearby.",
    )
    return parser.parse_args()


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def select_loci(args: argparse.Namespace) -> list[dict[str, object]]:
    high_candidates: dict[str, list[dict[str, object]]] = {mark: [] for mark in args.marks}
    low_candidates: dict[str, list[dict[str, object]]] = {mark: [] for mark in args.marks}
    columns = [
        "group",
        "mark",
        "chrom",
        "position_0based",
        "n_reads",
        "mean",
        "variance",
        "std",
        "positive_fraction",
    ]
    dtype = {
        "group": "string",
        "mark": "string",
        "chrom": "string",
        "position_0based": "int64",
        "n_reads": "int64",
        "mean": "float64",
        "variance": "float64",
        "std": "float64",
        "positive_fraction": "float64",
    }
    reader = pd.read_csv(
        args.variance_tsv,
        sep="\t",
        usecols=columns,
        dtype=dtype,
        chunksize=750_000,
    )

    batches_seen = 0
    for frame in reader:
        batches_seen += 1
        frame = frame[
            (frame["group"] == args.group)
            & (frame["chrom"] == args.chrom)
            & (frame["n_reads"] >= args.min_reads)
            & np.isfinite(frame["variance"].to_numpy())
        ]
        for mark in args.marks:
            mark_frame = frame[frame["mark"] == mark]
            if mark_frame.empty:
                continue
            high_part = mark_frame.sort_values("variance", ascending=False).head(args.top_n)
            low_part = mark_frame.sort_values("variance", ascending=True).head(args.bottom_n)
            high_candidates[mark].extend(high_part.to_dict("records"))
            low_candidates[mark].extend(low_part.to_dict("records"))
        if batches_seen % 10 == 0:
            print(json.dumps({"progress": "select_loci", "batches_seen": batches_seen}), flush=True)

    selected = []
    seen = set()
    for mark in args.marks:
        high_final = sorted(high_candidates[mark], key=lambda row: float(row["variance"]), reverse=True)[: args.top_n]
        low_final = sorted(low_candidates[mark], key=lambda row: float(row["variance"]))[: args.bottom_n]
        for row in high_final:
            out = {
                "group": row["group"],
                "mark": row["mark"],
                "chrom": row["chrom"],
                "position_0based": int(row["position_0based"]),
                "n_reads": int(row["n_reads"]),
                "target_mean": float(row["mean"]),
                "target_variance": float(row["variance"]),
                "target_std": float(row["std"]),
                "target_positive_fraction": float(row["positive_fraction"]),
            }
            out["variance_class"] = "high"
            key = (out["mark"], out["position_0based"], out["variance_class"])
            if key not in seen:
                selected.append(out)
                seen.add(key)
        for row in low_final:
            out = {
                "group": row["group"],
                "mark": row["mark"],
                "chrom": row["chrom"],
                "position_0based": int(row["position_0based"]),
                "n_reads": int(row["n_reads"]),
                "target_mean": float(row["mean"]),
                "target_variance": float(row["variance"]),
                "target_std": float(row["std"]),
                "target_positive_fraction": float(row["positive_fraction"]),
            }
            out["variance_class"] = "low"
            key = (out["mark"], out["position_0based"], out["variance_class"])
            if key not in seen:
                selected.append(out)
                seen.add(key)
    if not selected:
        raise SystemExit("No loci selected. Check group/chrom/mark/min-read filters.")
    return selected


def base_at_reference(fasta: pysam.FastaFile, chrom: str, pos: int) -> str:
    try:
        return fasta.fetch(chrom, pos, pos + 1).upper()
    except Exception:
        return "N"


def analyze_alignment_at_locus(
    alignment: pysam.AlignedSegment,
    fasta: pysam.FastaFile,
    chrom: str,
    locus: int,
    local_window: int,
    insertion_window: int,
) -> dict[str, object]:
    qpos = alignment.query_alignment_start or 0
    rpos = alignment.reference_start
    query = alignment.query_sequence or ""
    read_length = alignment.query_length or len(query)
    local_start = locus - local_window
    local_end = locus + local_window

    aligned_bases = 0
    mismatch_bases = 0
    insertion_bases = 0
    deletion_bases = 0
    refskip_bases = 0
    insertion_events_near_locus = 0
    deletion_at_locus = False
    refskip_at_locus = False
    locus_query_pos = None
    locus_read_base = ""
    locus_ref_base = base_at_reference(fasta, chrom, locus)
    locus_matches_reference = None
    softclip_bases = 0

    for op, length in alignment.cigartuples or []:
        if op in CIGAR_MATCH:
            for offset in range(length):
                ref = rpos + offset
                query_pos = qpos + offset
                if ref == locus:
                    locus_query_pos = query_pos
                    if 0 <= query_pos < len(query):
                        locus_read_base = query[query_pos].upper()
                        locus_matches_reference = locus_read_base == locus_ref_base
                if local_start <= ref <= local_end and 0 <= query_pos < len(query):
                    aligned_bases += 1
                    ref_base = base_at_reference(fasta, chrom, ref)
                    read_base = query[query_pos].upper()
                    mismatch_bases += int(read_base != ref_base)
            qpos += length
            rpos += length
        elif op == CIGAR_INS:
            if abs(rpos - locus) <= insertion_window:
                insertion_events_near_locus += 1
            if local_start <= rpos <= local_end:
                insertion_bases += length
            qpos += length
        elif op == CIGAR_DEL:
            overlap_start = max(local_start, rpos)
            overlap_end = min(local_end, rpos + length - 1)
            if overlap_start <= overlap_end:
                deletion_bases += overlap_end - overlap_start + 1
            if rpos <= locus < rpos + length:
                deletion_at_locus = True
            rpos += length
        elif op == CIGAR_REF_SKIP:
            overlap_start = max(local_start, rpos)
            overlap_end = min(local_end, rpos + length - 1)
            if overlap_start <= overlap_end:
                refskip_bases += overlap_end - overlap_start + 1
            if rpos <= locus < rpos + length:
                refskip_at_locus = True
            rpos += length
        elif op == CIGAR_SOFT_CLIP:
            softclip_bases += length
            qpos += length
        elif op == CIGAR_HARD_CLIP:
            continue
        else:
            if op in {6}:  # padding
                continue
            qpos += length if op in {4} else 0
            rpos += length if op in {2, 3} else 0

    local_ref_span = 2 * local_window + 1
    nm = alignment.get_tag("NM") if alignment.has_tag("NM") else None
    return {
        "read_id": alignment.query_name,
        "mapq": int(alignment.mapping_quality),
        "is_reverse": bool(alignment.is_reverse),
        "read_length": int(read_length),
        "cigar": alignment.cigarstring or "",
        "nm": nm,
        "locus_ref_base": locus_ref_base,
        "locus_read_base": locus_read_base,
        "locus_has_aligned_base": locus_query_pos is not None,
        "locus_matches_reference": locus_matches_reference,
        "locus_deleted": deletion_at_locus,
        "locus_refskip": refskip_at_locus,
        "insertion_near_locus": insertion_events_near_locus > 0,
        "aligned_bases_local": aligned_bases,
        "mismatch_bases_local": mismatch_bases,
        "insertion_bases_local": insertion_bases,
        "deletion_bases_local": deletion_bases,
        "refskip_bases_local": refskip_bases,
        "local_mismatch_rate": mismatch_bases / aligned_bases if aligned_bases else np.nan,
        "local_indel_bases_per_ref_base": (insertion_bases + deletion_bases) / local_ref_span,
        "softclip_fraction": softclip_bases / read_length if read_length else np.nan,
    }


def bool_mean(values: list[object]) -> float | None:
    valid = [v for v in values if v is not None and v != ""]
    if not valid:
        return None
    return float(np.mean([bool(v) for v in valid]))


def numeric_summary(values: list[object]) -> tuple[float | None, float | None]:
    arr = np.asarray([float(v) for v in values if v is not None and v != "" and not (isinstance(v, float) and np.isnan(v))])
    if arr.size == 0:
        return None, None
    return float(np.mean(arr)), float(np.median(arr))


def summarize_locus(locus: dict[str, object], read_rows: list[dict[str, object]]) -> dict[str, object]:
    out = dict(locus)
    out["reads_inspected"] = len(read_rows)
    out["reads_with_aligned_base_at_locus"] = sum(bool(r["locus_has_aligned_base"]) for r in read_rows)
    out["reads_deleted_at_locus"] = sum(bool(r["locus_deleted"]) for r in read_rows)
    out["reads_with_insertion_near_locus"] = sum(bool(r["insertion_near_locus"]) for r in read_rows)
    out["fraction_locus_matches_reference"] = bool_mean([r["locus_matches_reference"] for r in read_rows])
    out["fraction_locus_mismatch"] = (
        None
        if out["fraction_locus_matches_reference"] is None
        else 1.0 - float(out["fraction_locus_matches_reference"])
    )
    out["fraction_locus_deleted"] = bool_mean([r["locus_deleted"] for r in read_rows])
    out["fraction_insertion_near_locus"] = bool_mean([r["insertion_near_locus"] for r in read_rows])
    out["fraction_reverse"] = bool_mean([r["is_reverse"] for r in read_rows])

    for key in ("mapq", "nm", "local_mismatch_rate", "local_indel_bases_per_ref_base", "softclip_fraction"):
        mean_value, median_value = numeric_summary([r[key] for r in read_rows])
        out[f"mean_{key}"] = mean_value
        out[f"median_{key}"] = median_value
    return out


def inspect_loci(args: argparse.Namespace, loci: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    bam = pysam.AlignmentFile(args.bam, "rb")
    fasta = pysam.FastaFile(args.reference_fasta)
    per_read = []
    per_locus = []

    for i, locus in enumerate(loci, start=1):
        chrom = str(locus["chrom"])
        pos = int(locus["position_0based"])
        read_rows = []
        fetched = 0
        for alignment in bam.fetch(chrom, pos, pos + 1):
            if alignment.is_unmapped or alignment.is_secondary or alignment.is_supplementary:
                continue
            if alignment.reference_start is None or alignment.reference_end is None:
                continue
            if not (alignment.reference_start <= pos < alignment.reference_end):
                continue
            fetched += 1
            row = analyze_alignment_at_locus(
                alignment=alignment,
                fasta=fasta,
                chrom=chrom,
                locus=pos,
                local_window=args.local_window,
                insertion_window=args.insertion_window,
            )
            full_row = dict(locus)
            full_row.update(row)
            per_read.append(full_row)
            read_rows.append(row)
            if len(read_rows) >= args.max_reads_per_locus:
                break
        summary = summarize_locus(locus, read_rows)
        summary["bam_alignments_fetched"] = fetched
        per_locus.append(summary)
        if i % 100 == 0:
            print(json.dumps({"progress": i, "selected_loci": len(loci)}), flush=True)

    bam.close()
    fasta.close()
    return per_read, per_locus


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def group_summary(per_locus: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "target_variance",
        "target_mean",
        "fraction_locus_mismatch",
        "fraction_locus_deleted",
        "fraction_insertion_near_locus",
        "mean_local_mismatch_rate",
        "mean_local_indel_bases_per_ref_base",
        "mean_mapq",
        "mean_nm",
        "mean_softclip_fraction",
    ]
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_locus:
        groups[(str(row["mark"]), str(row["variance_class"]))].append(row)

    output = []
    for (mark, variance_class), rows in sorted(groups.items()):
        summary = {"mark": mark, "variance_class": variance_class, "loci": len(rows)}
        for metric in metrics:
            values = []
            for row in rows:
                value = row.get(metric)
                if value is None or value == "":
                    continue
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value_float):
                    values.append(value_float)
            arr = np.asarray(values, dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else None
            summary[f"{metric}_median"] = float(np.median(arr)) if arr.size else None
        output.append(summary)
    return output


def plot_summary(per_locus: list[dict[str, object]], path: Path) -> None:
    metrics = [
        ("fraction_locus_mismatch", "Fraction locus base mismatches reference"),
        ("fraction_locus_deleted", "Fraction reads deleted at locus"),
        ("fraction_insertion_near_locus", "Fraction reads with insertion nearby"),
        ("mean_local_mismatch_rate", "Mean local mismatch rate"),
        ("mean_local_indel_bases_per_ref_base", "Mean local indel bases per ref base"),
        ("mean_mapq", "Mean MAPQ"),
    ]
    marks = sorted({str(row["mark"]) for row in per_locus})
    fig, axes = plt.subplots(len(metrics), len(marks), figsize=(5 * len(marks), 3.2 * len(metrics)), squeeze=False)
    for col, mark in enumerate(marks):
        for row_idx, (metric, label) in enumerate(metrics):
            axis = axes[row_idx, col]
            data = []
            labels = []
            for variance_class in ("low", "high"):
                values = [
                    float(row[metric])
                    for row in per_locus
                    if row["mark"] == mark
                    and row["variance_class"] == variance_class
                    and row.get(metric) is not None
                    and row.get(metric) != ""
                    and math.isfinite(float(row[metric]))
                ]
                data.append(values)
                labels.append(variance_class)
            axis.boxplot(data, tick_labels=labels, showfliers=False)
            axis.set_title(f"{mark}: {label}")
            axis.set_ylabel(label)
    fig.suptitle("CIGAR/DNA heterogeneity at low- versus high-variance target loci", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    loci = select_loci(args)
    selected_path = Path(f"{out_prefix}.selected_loci.tsv")
    write_table(selected_path, loci)
    print(json.dumps({"selected_loci": len(loci), "selected_loci_tsv": str(selected_path)}), flush=True)

    per_read, per_locus = inspect_loci(args, loci)
    per_read_path = Path(f"{out_prefix}.per_read.tsv.gz")
    per_locus_path = Path(f"{out_prefix}.per_locus_summary.tsv")
    group_summary_path = Path(f"{out_prefix}.group_summary.tsv")
    plot_path = Path(f"{out_prefix}.cigar_dna_plots.png")
    json_path = Path(f"{out_prefix}.summary.json")

    write_table(per_read_path, per_read)
    write_table(per_locus_path, per_locus)
    summary_rows = group_summary(per_locus)
    write_table(group_summary_path, summary_rows)
    plot_summary(per_locus, plot_path)

    summary = {
        "variance_tsv": args.variance_tsv,
        "bam": args.bam,
        "reference_fasta": args.reference_fasta,
        "group": args.group,
        "chrom": args.chrom,
        "marks": args.marks,
        "top_n": args.top_n,
        "bottom_n": args.bottom_n,
        "min_reads": args.min_reads,
        "local_window": args.local_window,
        "insertion_window": args.insertion_window,
        "selected_loci": len(loci),
        "per_read_rows": len(per_read),
        "outputs": {
            "selected_loci": str(selected_path),
            "per_read": str(per_read_path),
            "per_locus_summary": str(per_locus_path),
            "group_summary": str(group_summary_path),
            "plots": str(plot_path),
            "summary": str(json_path),
        },
        "interpretation": (
            "If high-variance loci show higher mismatch/indel/softclip rates or lower MAPQ "
            "than low-variance loci, some target variance may be explained by underlying "
            "read-level DNA/alignment heterogeneity. If not, the variance is more likely "
            "biological or measurement noise rather than sequence differences."
        ),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
