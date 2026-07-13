#!/usr/bin/env python3
"""Audit read-to-read variance summaries and threshold-tail proportions.

This script distinguishes two different quantities that are easy to conflate:

1. variance partition fractions, e.g. within-locus / total variation;
2. proportions of eligible loci whose per-locus variance exceeds thresholds.

The variance histogram and threshold statistics are computed from the exact same
filtered per-locus variance arrays loaded from the per_locus_variance.tsv.gz file.
Optional extract-full input is used only to recover pre-min-read denominators.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MARKS = ("5mC", "6mA")
THRESHOLDS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-locus", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--group", default="merged_c1")
    parser.add_argument("--chrom", default="chr16")
    parser.add_argument("--chrom-length", type=int, default=90338345)
    parser.add_argument("--min-reads", type=int, default=5)
    parser.add_argument("--extract-full", default=None)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--threshold-line", type=float, default=0.02)
    return parser.parse_args()


def load_filtered_variances(path: Path, group: str, chrom: str) -> dict[str, dict[str, np.ndarray]]:
    """Load the exact arrays used by the previous variance histograms."""
    values: dict[str, dict[str, list[float | int]]] = {
        mark: {"variance": [], "n_reads": [], "mean": []} for mark in MARKS
    }
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["group"] != group or row["chrom"] != chrom:
                continue
            mark = row["mark"]
            if mark not in values:
                continue
            values[mark]["variance"].append(float(row["variance"]))
            values[mark]["n_reads"].append(int(row["n_reads"]))
            values[mark]["mean"].append(float(row["mean"]))
    return {
        mark: {
            key: np.asarray(vals, dtype=np.float64 if key != "n_reads" else np.int64)
            for key, vals in mark_values.items()
        }
        for mark, mark_values in values.items()
    }


def count_prefilter_loci_from_extract(path: Path, chrom: str, chrom_length: int, batch_size: int) -> dict[str, dict[str, int]]:
    """Count loci with >=1 and >=2 observations before min-read filtering."""
    counts = {mark: np.zeros(chrom_length, dtype=np.uint32) for mark in MARKS}
    columns = [
        "ref_position",
        "chrom",
        "ref_mod_strand",
        "mod_qual",
        "mod_code",
        "query_kmer",
        "canonical_base",
    ]
    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=columns,
        chunksize=batch_size,
        low_memory=False,
    )
    batches_seen = 0
    for frame in reader:
        batches_seen += 1
        frame = frame[
            (frame["chrom"] == chrom)
            & (frame["ref_position"] >= 0)
            & (frame["ref_position"] < chrom_length)
            & frame["mod_qual"].notna()
        ]

        six_ma = frame[
            (frame["mod_code"] == "a")
            & (frame["canonical_base"] == "A")
        ]
        pos = six_ma["ref_position"].to_numpy(dtype=np.int64, copy=False)
        if pos.size:
            unique, inverse = np.unique(pos, return_inverse=True)
            counts["6mA"][unique] += np.bincount(inverse).astype(np.uint32)

        query_kmer = frame["query_kmer"].fillna("").astype(str)
        five_mc = frame[
            (frame["mod_code"] == "m")
            & (frame["canonical_base"] == "C")
            & (query_kmer.str.slice(2, 4).str.upper() == "CG")
        ]
        pos = five_mc["ref_position"].to_numpy(dtype=np.int64, copy=True)
        reverse = five_mc["ref_mod_strand"].to_numpy() == "-"
        pos[reverse] -= 1
        keep = (pos >= 0) & (pos < chrom_length)
        pos = pos[keep]
        if pos.size:
            unique, inverse = np.unique(pos, return_inverse=True)
            counts["5mC"][unique] += np.bincount(inverse).astype(np.uint32)

        if batches_seen % 50 == 0:
            print(
                json.dumps({"progress": "count_prefilter_loci", "batches_seen": batches_seen}),
                flush=True,
            )

    return {
        mark: {
            "total_genomic_loci_before_filtering": chrom_length,
            "loci_at_least_one_observation": int(np.sum(arr >= 1)),
            "loci_at_least_two_contributing_reads": int(np.sum(arr >= 2)),
            "loci_passing_min_read_threshold": int(np.sum(arr >= 5)),
        }
        for mark, arr in counts.items()
    }


def count_prefilter_loci_from_extract_polars_unused(path: Path, chrom: str, chrom_length: int, batch_size: int) -> dict[str, dict[str, int]]:
    """Deprecated placeholder retained only to document the old Polars approach."""
    raise NotImplementedError(
        "This environment exposes an incomplete polars namespace; use pandas chunked counting."
    )


def pct(numer: int | float, denom: int | float) -> float:
    return float(numer / denom * 100.0) if denom else float("nan")


def summarize_variance_array(variance: np.ndarray, prefilter: dict[str, int] | None, summary_reported_loci: int) -> dict[str, object]:
    """Summarize one filtered variance array with explicit numerators/denominators."""
    denom = int(variance.size)
    out: dict[str, object] = {
        "total_genomic_loci_before_filtering": None if prefilter is None else prefilter["total_genomic_loci_before_filtering"],
        "loci_at_least_one_observation": None if prefilter is None else prefilter["loci_at_least_one_observation"],
        "loci_at_least_two_contributing_reads": None if prefilter is None else prefilter["loci_at_least_two_contributing_reads"],
        "loci_passing_min_read_coverage_threshold": None if prefilter is None else prefilter["loci_passing_min_read_threshold"],
        "loci_included_in_variance_histogram": denom,
        "reported_loci_from_summary_json": int(summary_reported_loci),
        "histogram_array_matches_reported_loci": bool(denom == int(summary_reported_loci)),
    }
    zero = int(np.sum(variance == 0.0))
    gt_zero = int(np.sum(variance > 0.0))
    out["variance_equal_zero"] = {"numerator": zero, "denominator": denom, "percent": pct(zero, denom)}
    out["variance_greater_than_zero"] = {"numerator": gt_zero, "denominator": denom, "percent": pct(gt_zero, denom)}
    for threshold in THRESHOLDS:
        n = int(np.sum(variance >= threshold))
        out[f"variance_ge_{threshold:g}"] = {"numerator": n, "denominator": denom, "percent": pct(n, denom)}
    out["mean_variance"] = float(np.mean(variance)) if denom else float("nan")
    out["median_variance"] = float(np.median(variance)) if denom else float("nan")
    out["quantiles"] = {
        "q25": float(np.quantile(variance, 0.25)) if denom else float("nan"),
        "q75": float(np.quantile(variance, 0.75)) if denom else float("nan"),
        "q90": float(np.quantile(variance, 0.90)) if denom else float("nan"),
        "q95": float(np.quantile(variance, 0.95)) if denom else float("nan"),
        "q99": float(np.quantile(variance, 0.99)) if denom else float("nan"),
    }
    return out


def write_summary_tsv(path: Path, audit: dict[str, object]) -> None:
    """Write a long-form TSV with numerator/denominator for every percentage."""
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["mark", "metric", "value", "numerator", "denominator", "percent"])
        for mark in MARKS:
            stats = audit["marks"][mark]
            for key in [
                "total_genomic_loci_before_filtering",
                "loci_at_least_one_observation",
                "loci_at_least_two_contributing_reads",
                "loci_passing_min_read_coverage_threshold",
                "loci_included_in_variance_histogram",
                "reported_loci_from_summary_json",
                "histogram_array_matches_reported_loci",
                "mean_variance",
                "median_variance",
            ]:
                writer.writerow([mark, key, stats.get(key), "", "", ""])
            for key, value in stats["quantiles"].items():
                writer.writerow([mark, key, value, "", "", ""])
            for key, value in stats.items():
                if isinstance(value, dict) and {"numerator", "denominator", "percent"} <= set(value):
                    writer.writerow([mark, key, "", value["numerator"], value["denominator"], value["percent"]])


def plot_threshold_audit(path_prefix: Path, arrays: dict[str, dict[str, np.ndarray]], audit: dict[str, object], threshold: float) -> dict[str, str]:
    """Save density histograms and ECDFs using the same arrays as the audit."""
    outputs = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    bins = np.linspace(0, max(float(arrays["5mC"]["variance"].max()), float(arrays["6mA"]["variance"].max())), 120)
    for mark, color in [("5mC", "#2878b5"), ("6mA", "#d95f02")]:
        variance = arrays[mark]["variance"]
        axes[0].hist(variance, bins=bins, density=True, alpha=0.45, label=mark, color=color)
        n = audit["marks"][mark][f"variance_ge_{threshold:g}"]["numerator"]
        denom = audit["marks"][mark][f"variance_ge_{threshold:g}"]["denominator"]
        percentage = audit["marks"][mark][f"variance_ge_{threshold:g}"]["percent"]
        axes[0].text(
            threshold,
            axes[0].get_ylim()[1] * (0.90 if mark == "5mC" else 0.80),
            f"{mark}: {n}/{denom} ({percentage:.2f}%) >= {threshold}",
            fontsize=8,
        )
        xs = np.sort(variance)
        ys = np.arange(1, xs.size + 1) / xs.size
        axes[1].plot(xs, ys, label=mark, color=color)
    for ax in axes:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Between-read variance")
        ax.legend()
    axes[0].set_ylabel("Density")
    axes[0].set_title("Density-normalized variance histograms")
    axes[1].set_ylabel("Empirical cumulative probability")
    axes[1].set_title("Variance ECDF")
    png = path_prefix.with_suffix(".threshold_density_ecdf.png")
    svg = path_prefix.with_suffix(".threshold_density_ecdf.svg")
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    outputs["threshold_density_ecdf_png"] = str(png)
    outputs["threshold_density_ecdf_svg"] = str(svg)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, mark, color in zip(axes, ["5mC", "6mA"], ["#2878b5", "#d95f02"]):
        variance = arrays[mark]["variance"]
        ax.hist(variance, bins=100, color=color)
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
        n = audit["marks"][mark][f"variance_ge_{threshold:g}"]["numerator"]
        denom = audit["marks"][mark][f"variance_ge_{threshold:g}"]["denominator"]
        percentage = audit["marks"][mark][f"variance_ge_{threshold:g}"]["percent"]
        ax.annotate(
            f">= {threshold}: {n}/{denom}\n({percentage:.2f}% eligible loci)",
            xy=(threshold, ax.get_ylim()[1] * 0.85),
            xytext=(threshold + 0.02, ax.get_ylim()[1] * 0.85),
            arrowprops={"arrowstyle": "->", "lw": 0.8},
            fontsize=8,
        )
        ax.set_title(f"{mark}: variance distribution")
        ax.set_xlabel("Between-read variance")
        ax.set_ylabel("Number of eligible loci")
    png = path_prefix.with_suffix(".threshold_count_histograms.png")
    svg = path_prefix.with_suffix(".threshold_count_histograms.svg")
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    outputs["threshold_count_histograms_png"] = str(png)
    outputs["threshold_count_histograms_svg"] = str(svg)
    return outputs


def main() -> None:
    args = parse_args()
    per_locus = Path(args.per_locus)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "rt", encoding="utf-8") as handle:
        old_summary = json.load(handle)

    arrays = load_filtered_variances(per_locus, args.group, args.chrom)
    prefilter = None
    if args.extract_full:
        prefilter = count_prefilter_loci_from_extract(
            Path(args.extract_full),
            args.chrom,
            args.chrom_length,
            args.batch_size,
        )

    audit = {
        "group": args.group,
        "chrom": args.chrom,
        "min_reads": args.min_reads,
        "threshold_line": args.threshold_line,
        "important_note": (
            "The previously reported 49.5% for 5mC and 88.2% for 6mA are "
            "within-locus fractions of total variation, not proportions of loci "
            "above a variance threshold."
        ),
        "histogram_consistency": (
            "Threshold percentages and new histograms/ECDFs are computed from "
            "the exact same filtered variance arrays loaded from the per-locus TSV."
        ),
        "old_within_locus_fraction_of_total_variation": {
            mark: old_summary["groups"][args.group][mark]["within_locus_fraction_of_total_variation"]
            for mark in MARKS
        },
        "old_between_locus_fraction_of_total_variation": {
            mark: old_summary["groups"][args.group][mark]["between_locus_fraction_of_total_variation"]
            for mark in MARKS
        },
        "marks": {
            mark: summarize_variance_array(
                arrays[mark]["variance"],
                None if prefilter is None else prefilter[mark],
                old_summary["groups"][args.group][mark]["reported_loci"],
            )
            for mark in MARKS
        },
    }
    plots = plot_threshold_audit(out_prefix, arrays, audit, args.threshold_line)
    audit["outputs"] = plots
    json_path = out_prefix.with_suffix(".audit.json")
    tsv_path = out_prefix.with_suffix(".audit.tsv")
    with json_path.open("wt", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")
    write_summary_tsv(tsv_path, audit)
    print(json.dumps({"audit_json": str(json_path), "audit_tsv": str(tsv_path), **plots}, indent=2))


if __name__ == "__main__":
    main()
