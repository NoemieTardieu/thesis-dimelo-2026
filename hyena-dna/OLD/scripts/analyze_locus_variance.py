#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pysam


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    positive_count: int = 0

    def update(self, value: float, threshold: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.positive_count += int(value >= threshold)

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count >= 2 else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure between-read target variance at reference-genome loci. "
            "Overlapping windows from the same read are collapsed before each "
            "read position is mapped through the BAM CIGAR."
        )
    )
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NPZ", "METADATA"),
        required=True,
        help="Tensor/metadata pair. Repeat for train, validation, and test.",
    )
    parser.add_argument(
        "--bam",
        action="append",
        required=True,
        metavar="SAMPLE=PATH",
        help="Primary alignment BAM for a metadata sample. Repeat per sample.",
    )
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-reads",
        type=int,
        default=2,
        help="Minimum independent reads required to report a locus.",
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        default=None,
        help="Optional debugging limit per dataset.",
    )
    parser.add_argument(
        "--collapse-cpg-dyads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Represent reverse-strand 5mC calls by the preceding reference "
            "coordinate so both strands of one CpG dyad share a locus."
        ),
    )
    return parser.parse_args()


def parse_bams(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --bam value {value!r}; expected SAMPLE=PATH")
        sample, path = value.split("=", 1)
        result[sample] = path
    return result


def read_metadata(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"read_id", "chrom", "window_start", "window_length", "sample"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise SystemExit(f"{path} is missing metadata columns: {sorted(missing)}")
    return rows


def collapse_read_windows(
    targets: np.ndarray,
    masks: np.ndarray,
    rows: list[dict[str, str]],
    row_indices: list[int],
) -> dict[int, float]:
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for row_idx in row_indices:
        row = rows[row_idx]
        window_start = int(row["window_start"])
        window_length = int(row["window_length"])
        valid = np.flatnonzero(masks[row_idx, :window_length])
        for local_pos in valid:
            read_pos = window_start + int(local_pos)
            sums[read_pos] += float(targets[row_idx, local_pos])
            counts[read_pos] += 1
    return {position: sums[position] / counts[position] for position in sums}


def read_to_reference_map(
    alignment: pysam.AlignedSegment, forward_positions: set[int]
) -> dict[int, int]:
    if not forward_positions:
        return {}

    read_length = int(alignment.query_length)
    query_to_forward = {}
    for forward_pos in forward_positions:
        query_pos = read_length - 1 - forward_pos if alignment.is_reverse else forward_pos
        query_to_forward[query_pos] = forward_pos

    mapped = {}
    for query_pos, reference_pos in alignment.get_aligned_pairs(matches_only=False):
        if query_pos is None or reference_pos is None:
            continue
        forward_pos = query_to_forward.get(query_pos)
        if forward_pos is not None:
            mapped[forward_pos] = int(reference_pos)
    return mapped


def process_mark(
    npz_path: str,
    rows: list[dict[str, str]],
    bam_paths: dict[str, str],
    mark: str,
    threshold: float,
    collapse_cpg_dyads: bool,
    max_reads: int | None,
    stats: dict[tuple[str, str, str, int], RunningStats],
    counters: dict[str, int],
) -> None:
    target_key = f"target_{mark}"
    mask_key = f"mask_{mark}"
    with np.load(npz_path) as archive:
        targets = archive[target_key]
        masks = archive[mask_key].astype(bool)

    rows_by_sample_read: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row_idx, row in enumerate(rows):
        rows_by_sample_read[(row["sample"], row["read_id"])].append(row_idx)

    selected_by_sample: dict[str, set[str]] = defaultdict(set)
    bounds_by_sample_chrom_region: dict[tuple[str, str, str], list[int]] = {}
    for sample, read_id in rows_by_sample_read:
        selected_by_sample[sample].add(read_id)
        first_row = rows[rows_by_sample_read[(sample, read_id)][0]]
        chrom = first_row["chrom"]
        region = first_row.get("region_name") or first_row.get("region_id") or read_id
        start = int(first_row["alignment_start"])
        end = int(first_row["alignment_end"])
        bounds = bounds_by_sample_chrom_region.setdefault((sample, chrom, region), [start, end])
        bounds[0] = min(bounds[0], start)
        bounds[1] = max(bounds[1], end)

    for sample, selected_read_ids in selected_by_sample.items():
        bam_path = bam_paths.get(sample)
        if bam_path is None:
            raise SystemExit(f"No --bam path supplied for sample {sample!r}")

        processed = 0
        seen: set[str] = set()
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            sample_bounds = [
                (chrom, bounds)
                for (bound_sample, chrom, _region), bounds in bounds_by_sample_chrom_region.items()
                if bound_sample == sample
            ]
            stop = False
            for fetch_chrom, (fetch_start, fetch_end) in sample_bounds:
                for alignment in bam.fetch(fetch_chrom, fetch_start, fetch_end):
                    read_id = alignment.query_name
                    if read_id not in selected_read_ids or read_id in seen:
                        continue
                    if (
                        alignment.is_unmapped
                        or alignment.is_secondary
                        or alignment.is_supplementary
                        or alignment.query_length is None
                    ):
                        continue

                    seen.add(read_id)
                    row_indices = rows_by_sample_read[(sample, read_id)]
                    observations = collapse_read_windows(targets, masks, rows, row_indices)
                    position_map = read_to_reference_map(alignment, set(observations))
                    counters[f"{mark}_reads_processed"] += 1
                    counters[f"{mark}_read_positions_after_window_collapse"] += len(observations)
                    counters[f"{mark}_positions_mapped_to_reference"] += len(position_map)
                    counters[f"{mark}_positions_unmapped_indel_or_softclip"] += (
                        len(observations) - len(position_map)
                    )

                    chrom = alignment.reference_name
                    for read_pos, reference_pos in position_map.items():
                        locus_pos = reference_pos
                        if mark == "5mC" and collapse_cpg_dyads and alignment.is_reverse:
                            locus_pos -= 1
                        if locus_pos < 0:
                            continue
                        value = observations[read_pos]
                        stats[("all", mark, chrom, locus_pos)].update(value, threshold)
                        stats[(sample, mark, chrom, locus_pos)].update(value, threshold)

                    processed += 1
                    if max_reads is not None and processed >= max_reads:
                        stop = True
                        break
                if stop:
                    break

        counters[f"{mark}_{sample}_selected_reads"] += len(selected_read_ids)
        counters[f"{mark}_{sample}_alignments_found"] += len(seen)

    del targets
    del masks


def write_table(
    output_path: Path,
    stats: dict[tuple[str, str, str, int], RunningStats],
    min_reads: int,
) -> list[dict[str, float | int | str]]:
    rows = []
    for (group, mark, chrom, position), value in stats.items():
        if value.count < min_reads:
            continue
        rows.append(
            {
                "group": group,
                "mark": mark,
                "chrom": chrom,
                "position_0based": position,
                "position_1based": position + 1,
                "n_reads": value.count,
                "mean": value.mean,
                "variance": value.variance,
                "std": np.sqrt(value.variance),
                "min": value.minimum,
                "max": value.maximum,
                "range": value.maximum - value.minimum,
                "positive_fraction": value.positive_count / value.count,
            }
        )
    rows.sort(key=lambda row: (str(row["group"]), str(row["mark"]), str(row["chrom"]), int(row["position_0based"])))

    fieldnames = list(rows[0]) if rows else [
        "group", "mark", "chrom", "position_0based", "position_1based",
        "n_reads", "mean", "variance", "std", "min", "max", "range",
        "positive_fraction",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_rows(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    pooled = [row for row in rows if row["group"] == "all"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    for column, mark in enumerate(("5mC", "6mA")):
        mark_rows = [row for row in pooled if row["mark"] == mark]
        positions = np.asarray([row["position_0based"] for row in mark_rows], dtype=np.int64)
        variances = np.asarray([row["variance"] for row in mark_rows], dtype=float)
        coverage = np.asarray([row["n_reads"] for row in mark_rows], dtype=int)
        means = np.asarray([row["mean"] for row in mark_rows], dtype=float)

        axes[0, column].scatter(
            positions / 1e6, variances, s=2, alpha=0.25, linewidths=0, rasterized=True
        )
        axes[0, column].set(
            title=f"{mark}: variance across reference loci",
            xlabel="Reference position (Mb)",
            ylabel="Between-read variance",
        )

        axes[1, column].hist(variances[np.isfinite(variances)], bins=80, color="#2878b5", alpha=0.85)
        axes[1, column].set(
            title=f"{mark}: variance distribution",
            xlabel="Between-read variance",
            ylabel="Number of loci",
        )

        plot = axes[2, column].hexbin(
            means,
            variances,
            C=coverage,
            reduce_C_function=np.mean,
            gridsize=50,
            mincnt=1,
            cmap="viridis",
        )
        axes[2, column].set(
            title=f"{mark}: mean versus variance",
            xlabel="Mean target probability",
            ylabel="Between-read variance",
        )
        fig.colorbar(plot, ax=axes[2, column], label="Mean read coverage")

    fig.suptitle("Read-to-read modification variance at reference-genome loci", fontsize=15)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summarize(
    rows: list[dict[str, float | int | str]],
    counters: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    result: dict[str, object] = {
        "datasets": [{"npz": pair[0], "metadata": pair[1]} for pair in args.dataset],
        "threshold": args.threshold,
        "min_reads": args.min_reads,
        "collapse_cpg_dyads": args.collapse_cpg_dyads,
        "counters": counters,
        "groups": {},
    }
    groups = sorted({str(row["group"]) for row in rows})
    for group in groups:
        result["groups"][group] = {}
        for mark in ("5mC", "6mA"):
            mark_rows = [row for row in rows if row["group"] == group and row["mark"] == mark]
            values = np.asarray([row["variance"] for row in mark_rows], dtype=float)
            coverage = np.asarray([row["n_reads"] for row in mark_rows], dtype=float)
            means = np.asarray([row["mean"] for row in mark_rows], dtype=float)
            total_observations = float(np.sum(coverage))
            global_mean = (
                float(np.sum(coverage * means) / total_observations)
                if total_observations
                else None
            )
            within_ss = float(np.sum((coverage - 1.0) * values)) if values.size else 0.0
            between_ss = (
                float(np.sum(coverage * (means - global_mean) ** 2))
                if values.size and global_mean is not None
                else 0.0
            )
            total_ss = within_ss + between_ss
            result["groups"][group][mark] = {
                "reported_loci": int(values.size),
                "observations_at_reported_loci": int(total_observations),
                "median_variance": float(np.median(values)) if values.size else None,
                "mean_variance": float(np.mean(values)) if values.size else None,
                "variance_q90": float(np.quantile(values, 0.9)) if values.size else None,
                "variance_q95": float(np.quantile(values, 0.95)) if values.size else None,
                "median_read_coverage": float(np.median(coverage)) if coverage.size else None,
                "global_mean": global_mean,
                "within_locus_fraction_of_total_variation": (
                    within_ss / total_ss if total_ss > 0 else None
                ),
                "between_locus_fraction_of_total_variation": (
                    between_ss / total_ss if total_ss > 0 else None
                ),
            }
    return result


def main() -> None:
    args = parse_args()
    bam_paths = parse_bams(args.bam)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[tuple[str, str, str, int], RunningStats] = defaultdict(RunningStats)
    counters: dict[str, int] = defaultdict(int)

    for npz_path, metadata_path in args.dataset:
        rows = read_metadata(metadata_path)
        for mark in ("5mC", "6mA"):
            process_mark(
                npz_path=npz_path,
                rows=rows,
                bam_paths=bam_paths,
                mark=mark,
                threshold=args.threshold,
                collapse_cpg_dyads=args.collapse_cpg_dyads,
                max_reads=args.max_reads,
                stats=stats,
                counters=counters,
            )

    table_path = Path(f"{out_prefix}.per_locus_variance.tsv")
    plot_path = Path(f"{out_prefix}.variance_plots.png")
    summary_path = Path(f"{out_prefix}.summary.json")
    table_rows = write_table(table_path, stats, args.min_reads)
    plot_rows(table_rows, plot_path)
    summary = summarize(table_rows, dict(counters), args)
    summary["outputs"] = {
        "per_locus_variance": str(table_path),
        "variance_plots": str(plot_path),
        "summary": str(summary_path),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
