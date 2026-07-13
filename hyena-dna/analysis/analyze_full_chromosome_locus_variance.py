#!/usr/bin/env python3
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
import polars as pl


MARKS = ("5mC", "6mA")
POLARS_AVAILABLE = hasattr(pl, "read_csv_batched") and hasattr(pl, "Int64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute read-to-read 5mC and 6mA probability variance at every "
            "covered reference locus in a complete chromosome, directly from "
            "modkit extract-full files."
        )
    )
    parser.add_argument(
        "--extract-full",
        action="append",
        required=True,
        metavar="SAMPLE=PATH",
        help="Sample and chromosome extract-full TSV.GZ. Repeat per sample.",
    )
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--chrom-length", type=int, required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--min-reads", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional smoke-test limit per sample.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=750_000,
        help="Maximum loci rendered in each chromosome scatter panel.",
    )
    return parser.parse_args()


def parse_inputs(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --extract-full value {value!r}; use SAMPLE=PATH")
        sample, path = value.split("=", 1)
        if sample in result:
            raise SystemExit(f"Duplicate sample name: {sample}")
        result[sample] = path
    return result


def allocate_stats(chrom_length: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        mark: {
            "count": np.zeros(chrom_length, dtype=np.uint32),
            "sum": np.zeros(chrom_length, dtype=np.float64),
            "sumsq": np.zeros(chrom_length, dtype=np.float64),
            "positive": np.zeros(chrom_length, dtype=np.uint32),
        }
        for mark in MARKS
    }


def update_arrays(
    arrays: dict[str, np.ndarray],
    positions: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> None:
    if positions.size == 0:
        return
    unique_positions, inverse = np.unique(positions, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values)
    sums_sq = np.bincount(inverse, weights=values * values)
    positives = np.bincount(inverse, weights=(values >= threshold).astype(np.uint8))
    arrays["count"][unique_positions] += counts.astype(np.uint32)
    arrays["sum"][unique_positions] += sums
    arrays["sumsq"][unique_positions] += sums_sq
    arrays["positive"][unique_positions] += positives.astype(np.uint32)


def process_sample(
    sample: str,
    path: str,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    if not POLARS_AVAILABLE:
        return process_sample_pandas(sample, path, args)

    stats = allocate_stats(args.chrom_length)
    counters = {
        "rows_seen": 0,
        "mapped_rows": 0,
        "5mC_rows_used": 0,
        "6mA_rows_used": 0,
    }
    columns = [
        "ref_position",
        "chrom",
        "ref_mod_strand",
        "mod_qual",
        "mod_code",
        "query_kmer",
        "canonical_base",
    ]
    schema = {
        "ref_position": pl.Int64,
        "chrom": pl.String,
        "ref_mod_strand": pl.String,
        "mod_qual": pl.Float32,
        "mod_code": pl.String,
        "query_kmer": pl.String,
        "canonical_base": pl.String,
    }
    reader = pl.read_csv_batched(
        path,
        separator="\t",
        columns=columns,
        schema_overrides=schema,
        batch_size=args.batch_size,
        n_rows=args.max_rows,
        low_memory=True,
    )

    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        frame = batches[0]
        counters["rows_seen"] += frame.height
        frame = frame.filter(
            (pl.col("chrom") == args.chrom)
            & (pl.col("ref_position") >= 0)
            & (pl.col("ref_position") < args.chrom_length)
            & pl.col("mod_qual").is_not_null()
        )
        counters["mapped_rows"] += frame.height

        six_ma = frame.filter(
            (pl.col("mod_code") == "a") & (pl.col("canonical_base") == "A")
        )
        pos = six_ma["ref_position"].to_numpy().astype(np.int64, copy=False)
        values = six_ma["mod_qual"].to_numpy().astype(np.float64, copy=False)
        update_arrays(stats["6mA"], pos, values, args.threshold)
        counters["6mA_rows_used"] += int(pos.size)

        five_mc = frame.filter(
            (pl.col("mod_code") == "m")
            & (pl.col("canonical_base") == "C")
            & (pl.col("query_kmer").str.slice(2, 2).str.to_uppercase() == "CG")
        )
        pos = five_mc["ref_position"].to_numpy().astype(np.int64, copy=True)
        reverse = five_mc["ref_mod_strand"].to_numpy() == "-"
        pos[reverse] -= 1
        values = five_mc["mod_qual"].to_numpy().astype(np.float64, copy=False)
        keep = (pos >= 0) & (pos < args.chrom_length)
        update_arrays(stats["5mC"], pos[keep], values[keep], args.threshold)
        counters["5mC_rows_used"] += int(np.sum(keep))

        if counters["rows_seen"] % (10 * args.batch_size) == 0:
            print(
                json.dumps(
                    {
                        "sample": sample,
                        "rows_seen": counters["rows_seen"],
                        "5mC_rows_used": counters["5mC_rows_used"],
                        "6mA_rows_used": counters["6mA_rows_used"],
                    }
                ),
                flush=True,
            )
    return stats, counters


def process_sample_pandas(
    sample: str,
    path: str,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """Chunked pandas fallback for environments without a usable Polars build."""
    stats = allocate_stats(args.chrom_length)
    counters = {
        "rows_seen": 0,
        "mapped_rows": 0,
        "5mC_rows_used": 0,
        "6mA_rows_used": 0,
    }
    columns = [
        "ref_position",
        "chrom",
        "ref_mod_strand",
        "mod_qual",
        "mod_code",
        "query_kmer",
        "canonical_base",
    ]
    dtype = {
        "chrom": "string",
        "ref_mod_strand": "string",
        "mod_code": "string",
        "query_kmer": "string",
        "canonical_base": "string",
    }
    rows_left = args.max_rows
    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=columns,
        dtype=dtype,
        chunksize=args.batch_size,
        compression="infer",
    )
    for chunk in reader:
        if rows_left is not None:
            if rows_left <= 0:
                break
            if len(chunk) > rows_left:
                chunk = chunk.iloc[:rows_left].copy()
            rows_left -= len(chunk)

        counters["rows_seen"] += int(len(chunk))
        chunk["ref_position"] = pd.to_numeric(chunk["ref_position"], errors="coerce")
        chunk["mod_qual"] = pd.to_numeric(chunk["mod_qual"], errors="coerce")
        frame = chunk[
            (chunk["chrom"] == args.chrom)
            & chunk["ref_position"].notna()
            & (chunk["ref_position"] >= 0)
            & (chunk["ref_position"] < args.chrom_length)
            & chunk["mod_qual"].notna()
        ]
        counters["mapped_rows"] += int(len(frame))

        six_ma = frame[(frame["mod_code"] == "a") & (frame["canonical_base"] == "A")]
        pos = six_ma["ref_position"].to_numpy(dtype=np.int64, copy=False)
        values = six_ma["mod_qual"].to_numpy(dtype=np.float64, copy=False)
        update_arrays(stats["6mA"], pos, values, args.threshold)
        counters["6mA_rows_used"] += int(pos.size)

        kmer_cpg = frame["query_kmer"].astype("string").str.slice(2, 4).str.upper() == "CG"
        five_mc = frame[(frame["mod_code"] == "m") & (frame["canonical_base"] == "C") & kmer_cpg]
        pos = five_mc["ref_position"].to_numpy(dtype=np.int64, copy=False)
        values = five_mc["mod_qual"].to_numpy(dtype=np.float64, copy=False)
        update_arrays(stats["5mC"], pos, values, args.threshold)
        counters["5mC_rows_used"] += int(pos.size)

        if counters["rows_seen"] % (args.batch_size * 10) == 0:
            print(
                json.dumps(
                    {
                        "progress": "pandas_stream",
                        "sample": sample,
                        "rows_seen": counters["rows_seen"],
                        "mapped_rows": counters["mapped_rows"],
                    }
                ),
                flush=True,
            )
    return stats, counters


def combine_stats(
    sample_stats: dict[str, dict[str, dict[str, np.ndarray]]],
    mark: str,
) -> dict[str, np.ndarray]:
    samples = list(sample_stats)
    combined = {
        key: sample_stats[samples[0]][mark][key].copy()
        for key in ("count", "sum", "sumsq", "positive")
    }
    for sample in samples[1:]:
        for key in combined:
            combined[key] += sample_stats[sample][mark][key]
    return combined


def materialize(
    arrays: dict[str, np.ndarray],
    min_reads: int,
) -> dict[str, np.ndarray]:
    positions = np.flatnonzero(arrays["count"] >= min_reads)
    count = arrays["count"][positions].astype(np.float64)
    sums = arrays["sum"][positions]
    sumsq = arrays["sumsq"][positions]
    mean = sums / count
    variance = np.maximum(0.0, (sumsq - sums * sums / count) / (count - 1.0))
    return {
        "position": positions,
        "count": count.astype(np.uint32),
        "mean": mean,
        "variance": variance,
        "std": np.sqrt(variance),
        "positive_fraction": arrays["positive"][positions] / count,
    }


def summarize_loci(loci: dict[str, np.ndarray]) -> dict[str, float | int | None]:
    if loci["position"].size == 0:
        return {"reported_loci": 0}
    count = loci["count"].astype(np.float64)
    mean = loci["mean"]
    variance = loci["variance"]
    observations = float(np.sum(count))
    global_mean = float(np.sum(count * mean) / observations)
    within_ss = float(np.sum((count - 1.0) * variance))
    between_ss = float(np.sum(count * (mean - global_mean) ** 2))
    total_ss = within_ss + between_ss
    return {
        "reported_loci": int(loci["position"].size),
        "observations_at_reported_loci": int(observations),
        "median_coverage": float(np.median(count)),
        "median_variance": float(np.median(variance)),
        "mean_variance": float(np.mean(variance)),
        "variance_q90": float(np.quantile(variance, 0.90)),
        "variance_q95": float(np.quantile(variance, 0.95)),
        "global_mean": global_mean,
        "within_locus_fraction_of_total_variation": within_ss / total_ss if total_ss else None,
        "between_locus_fraction_of_total_variation": between_ss / total_ss if total_ss else None,
    }


def write_loci(
    path: Path,
    groups: dict[str, dict[str, dict[str, np.ndarray]]],
    chrom: str,
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "group",
                "mark",
                "chrom",
                "position_0based",
                "position_1based",
                "n_reads",
                "mean",
                "variance",
                "std",
                "positive_fraction",
            ]
        )
        for group, marks in groups.items():
            for mark, loci in marks.items():
                n = loci["position"].size
                chunk_size = 100_000
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    writer.writerows(
                        zip(
                            [group] * (end - start),
                            [mark] * (end - start),
                            [chrom] * (end - start),
                            loci["position"][start:end],
                            loci["position"][start:end] + 1,
                            loci["count"][start:end],
                            loci["mean"][start:end],
                            loci["variance"][start:end],
                            loci["std"][start:end],
                            loci["positive_fraction"][start:end],
                        )
                    )


def sample_indices(n: int, maximum: int) -> np.ndarray:
    if n <= maximum:
        return np.arange(n)
    return np.linspace(0, n - 1, maximum, dtype=np.int64)


def plot_pooled(
    pooled: dict[str, dict[str, np.ndarray]],
    path: Path,
    chrom: str,
    maximum: int,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    for column, mark in enumerate(MARKS):
        loci = pooled[mark]
        chosen = sample_indices(loci["position"].size, maximum)
        axes[0, column].scatter(
            loci["position"][chosen] / 1e6,
            loci["variance"][chosen],
            s=1,
            alpha=0.2,
            linewidths=0,
            rasterized=True,
        )
        axes[0, column].set(
            title=f"{mark}: variance across {chrom}",
            xlabel="Reference position (Mb)",
            ylabel="Between-read variance",
        )
        axes[1, column].hist(loci["variance"], bins=100, color="#2878b5")
        axes[1, column].set(
            title=f"{mark}: variance distribution",
            xlabel="Between-read variance",
            ylabel="Number of loci",
        )
        plot = axes[2, column].hexbin(
            loci["mean"][chosen],
            loci["variance"][chosen],
            C=loci["count"][chosen],
            reduce_C_function=np.mean,
            gridsize=60,
            mincnt=1,
            cmap="viridis",
        )
        axes[2, column].set(
            title=f"{mark}: mean versus variance",
            xlabel="Mean target probability",
            ylabel="Between-read variance",
        )
        fig.colorbar(plot, ax=axes[2, column], label="Mean read coverage")
    fig.suptitle(f"Read-to-read modification variance across complete {chrom}", fontsize=15)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inputs = parse_inputs(args.extract_full)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    sample_stats = {}
    counters = {}
    for sample, path in inputs.items():
        print(json.dumps({"status": "processing", "sample": sample, "path": path}), flush=True)
        sample_stats[sample], counters[sample] = process_sample(sample, path, args)

    materialized: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for sample, marks in sample_stats.items():
        materialized[sample] = {
            mark: materialize(marks[mark], args.min_reads) for mark in MARKS
        }
    pooled_arrays = {mark: combine_stats(sample_stats, mark) for mark in MARKS}
    materialized["all"] = {
        mark: materialize(pooled_arrays[mark], args.min_reads) for mark in MARKS
    }

    table_path = Path(f"{out_prefix}.per_locus_variance.tsv.gz")
    plot_path = Path(f"{out_prefix}.variance_plots.png")
    summary_path = Path(f"{out_prefix}.summary.json")
    write_loci(table_path, materialized, args.chrom)
    plot_pooled(materialized["all"], plot_path, args.chrom, args.max_scatter_points)

    summary = {
        "chrom": args.chrom,
        "chrom_length": args.chrom_length,
        "extract_full": inputs,
        "min_reads": args.min_reads,
        "threshold": args.threshold,
        "counters": counters,
        "groups": {
            group: {mark: summarize_loci(marks[mark]) for mark in MARKS}
            for group, marks in materialized.items()
        },
        "notes": {
            "5mC_locus": (
                "Only mod_code=m calls at query-sequence CpGs; reverse-strand "
                "calls are shifted to the cytosine coordinate of the CpG dyad."
            ),
            "6mA_locus": "All mapped mod_code=a adenine calls.",
            "labels": (
                "Uses modkit extract-full probabilities, including inferred "
                "zero-probability calls, matching the tensor-label source."
            ),
        },
        "outputs": {
            "per_locus_variance": str(table_path),
            "variance_plots": str(plot_path),
            "summary": str(summary_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
