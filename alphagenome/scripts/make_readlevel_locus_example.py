#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam

from benchmark_utils import CHROMS, load_regions, read_to_reference_map, read_tsv
from evaluate_alphagenome_readlevel import load_cached_tracks


def complete_bins(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(bin_start, bin_start + size) for bin_start in range(start, end - size + 1, size)]


def find_primary_alignment(
    bam: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
    read_id: str,
) -> pysam.AlignedSegment | None:
    for read in bam.fetch(chrom, start, end):
        if (
            read.query_name == read_id
            and not read.is_unmapped
            and not read.is_secondary
            and not read.is_supplementary
            and read.query_length is not None
        ):
            return read
    return None


def collapse_targets_for_read(
    metadata: list[dict[str, str]],
    targets: np.ndarray,
    masks: np.ndarray,
    row_indices: list[int],
) -> dict[int, float]:
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for idx in row_indices:
        row = metadata[idx]
        length = min(int(row["window_length"]), targets.shape[1])
        window_start = int(row["window_start"])
        for local_pos in np.flatnonzero(masks[idx, :length]):
            read_pos = window_start + int(local_pos)
            sums[read_pos] += float(targets[idx, local_pos])
            counts[read_pos] += 1
    return {pos: sums[pos] / counts[pos] for pos in sums}


def alpha_at_read_pos(cache: dict, read_length: int, read_pos: int) -> float | None:
    left_pad = (int(cache["sequence_length"]) - read_length) // 2
    output_index = (left_pad + read_pos) // int(cache["resolution"])
    track = cache["track"]
    if output_index < 0 or output_index >= len(track):
        return None
    return float(track[output_index])


def collect_read_bin_rows(
    regions_path: Path,
    outputs_dir: Path,
    cache_dir: Path,
    bam_paths: dict[str, Path],
    bin_size: int,
    split: str,
) -> pd.DataFrame:
    cached_tracks = load_cached_tracks(cache_dir)
    regions = load_regions(regions_path, split=split)
    regions_by_chrom = {chrom: [r for r in regions if r.chrom == chrom] for chrom in CHROMS}
    rows = []

    for chrom in CHROMS:
        prefix = (
            outputs_dir
            / f"merged_e5b_c1_{chrom}_selected_top100_overlap16k_full5000_region_split.{split}"
        )
        metadata = read_tsv(Path(f"{prefix}.metadata.tsv"))
        archive = np.load(Path(f"{prefix}.npz"))
        targets = archive["target_6mA"]
        masks = archive["mask_6mA"].astype(bool)

        metadata_by_read: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for idx, row in enumerate(metadata):
            sample = row.get("sample") or row.get("sample_id")
            key = (sample, row["read_id"], row.get("region_id") or row.get("region_name") or "")
            if (sample, row["read_id"]) in cached_tracks:
                metadata_by_read[key].append(idx)

        bam_handles = {
            sample: pysam.AlignmentFile(path, "rb")
            for sample, path in bam_paths.items()
        }
        try:
            for region in regions_by_chrom[chrom]:
                region_bins = complete_bins(region.start, region.end, bin_size)
                if not region_bins:
                    continue
                valid_starts = {start for start, _ in region_bins}
                for sample in bam_paths:
                    read_keys = [
                        key for key in metadata_by_read
                        if key[0] == sample and key[2] == region.region_id
                    ]
                    for _, read_id, _ in read_keys:
                        cache = cached_tracks.get((sample, read_id))
                        if cache is None:
                            continue
                        alignment = find_primary_alignment(
                            bam_handles[sample], chrom, region.start, region.end, read_id
                        )
                        if alignment is None or alignment.query_length is None:
                            continue
                        observations = collapse_targets_for_read(
                            metadata, targets, masks, metadata_by_read[(sample, read_id, region.region_id)]
                        )
                        if not observations:
                            continue
                        mapping = read_to_reference_map(alignment, set(observations))
                        per_bin: dict[int, dict[str, list[float]]] = defaultdict(
                            lambda: {"target": [], "alpha": [], "ref_pos": []}
                        )
                        for read_pos, target in observations.items():
                            ref_pos = mapping.get(read_pos)
                            if ref_pos is None or not (region.start <= ref_pos < region.end):
                                continue
                            bin_start = region.start + ((ref_pos - region.start) // bin_size) * bin_size
                            if bin_start not in valid_starts:
                                continue
                            alpha = alpha_at_read_pos(cache, int(alignment.query_length), read_pos)
                            if alpha is None:
                                continue
                            per_bin[bin_start]["target"].append(float(target))
                            per_bin[bin_start]["alpha"].append(float(alpha))
                            per_bin[bin_start]["ref_pos"].append(float(ref_pos))
                        for bin_start, values in per_bin.items():
                            rows.append(
                                {
                                    "sample": sample,
                                    "read_id": read_id,
                                    "chrom": chrom,
                                    "region_id": region.region_id,
                                    "region_name": region.name,
                                    "bin_start": int(bin_start),
                                    "bin_end": int(bin_start + bin_size),
                                    "read_observations_in_bin": len(values["target"]),
                                    "mean_ref_pos": float(np.mean(values["ref_pos"])),
                                    "alphagenome_mean": float(np.mean(values["alpha"])),
                                    "alphagenome_std_within_read": float(np.std(values["alpha"])),
                                    "dimelo_6ma_mean": float(np.mean(values["target"])),
                                    "dimelo_6ma_std_within_read": float(np.std(values["target"])),
                                    "dimelo_positive_fraction_0_5": float(np.mean(np.asarray(values["target"]) >= 0.5)),
                                }
                            )
        finally:
            for bam in bam_handles.values():
                bam.close()
            archive.close()
    return pd.DataFrame(rows)


def select_example(read_bins: pd.DataFrame, min_reads: int, min_observations: int) -> tuple[pd.DataFrame, pd.Series]:
    grouped = read_bins.groupby(["chrom", "region_id", "bin_start", "bin_end"], observed=True)
    summaries = grouped.agg(
        reads=("read_id", "nunique"),
        observations=("read_observations_in_bin", "sum"),
        alpha_mean=("alphagenome_mean", "mean"),
        alpha_std=("alphagenome_mean", "std"),
        alpha_min=("alphagenome_mean", "min"),
        alpha_max=("alphagenome_mean", "max"),
        dimelo_mean=("dimelo_6ma_mean", "mean"),
        dimelo_std=("dimelo_6ma_mean", "std"),
        dimelo_min=("dimelo_6ma_mean", "min"),
        dimelo_max=("dimelo_6ma_mean", "max"),
        region_name=("region_name", "first"),
    ).reset_index()
    summaries["alpha_cv"] = summaries["alpha_std"] / summaries["alpha_mean"].replace(0, np.nan)
    summaries["dimelo_range"] = summaries["dimelo_max"] - summaries["dimelo_min"]
    candidates = summaries[
        (summaries["reads"] >= min_reads)
        & (summaries["observations"] >= min_observations)
        & summaries["alpha_cv"].notna()
    ].copy()
    if candidates.empty:
        raise SystemExit("No candidate locus found; lower --min-reads or --min-observations.")
    candidates["score"] = (
        candidates["dimelo_std"].fillna(0)
        + candidates["dimelo_range"].fillna(0)
        - 0.1 * candidates["alpha_cv"].fillna(0)
    )
    best = candidates.sort_values("score", ascending=False).iloc[0]
    example = read_bins[
        (read_bins["chrom"] == best["chrom"])
        & (read_bins["region_id"].astype(str) == str(best["region_id"]))
        & (read_bins["bin_start"] == int(best["bin_start"]))
        & (read_bins["bin_end"] == int(best["bin_end"]))
    ].copy()
    return example, best


def plot_example(example: pd.DataFrame, best: pd.Series, out: Path) -> None:
    frame = example.sort_values("dimelo_6ma_mean").reset_index(drop=True)
    labels = [f"{row.sample}:{str(row.read_id)[:8]}" for row in frame.itertuples()]
    x = np.arange(len(frame))
    fig, axis1 = plt.subplots(figsize=(max(10, len(frame) * 0.32), 5.2))
    axis1.bar(x, frame["dimelo_6ma_mean"], color="tab:orange", alpha=0.75, label="DiMeLo 6mA per read")
    axis1.set_ylabel("DiMeLo 6mA mean in locus")
    axis1.set_ylim(0, 1)
    axis1.set_xticks(x)
    axis1.set_xticklabels(labels, rotation=90, fontsize=7)
    axis2 = axis1.twinx()
    axis2.plot(x, frame["alphagenome_mean"], color="black", marker="o", linewidth=1.5, label="AlphaGenome H3K4me3")
    axis2.set_ylabel("AlphaGenome H3K4me3 score")
    title = (
        f"{best['chrom']}:{int(best['bin_start'])}-{int(best['bin_end'])} "
        f"({int(best['reads'])} reads): stable AlphaGenome, variable DiMeLo"
    )
    axis1.set_title(title)
    lines1, labels1 = axis1.get_legend_handles_labels()
    lines2, labels2 = axis2.get_legend_handles_labels()
    axis1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_interpretation(path: Path, best: pd.Series, example: pd.DataFrame) -> None:
    text = f"""# Same-Locus Read-Level Example

Selected locus: `{best['chrom']}:{int(best['bin_start'])}-{int(best['bin_end'])}`.

This locus contains `{int(best['reads'])}` reads and `{int(best['observations'])}` read-position observations.

Across reads:

- AlphaGenome mean score: `{best['alpha_mean']:.3f}`
- AlphaGenome read-to-read SD: `{best['alpha_std']:.3f}`
- DiMeLo 6mA mean: `{best['dimelo_mean']:.3f}`
- DiMeLo 6mA read-to-read SD: `{best['dimelo_std']:.3f}`
- DiMeLo read-level range: `{best['dimelo_min']:.3f}` to `{best['dimelo_max']:.3f}`

Interpretation: this is an example where reads map to the same genomic locus and receive broadly similar AlphaGenome regulatory scores, but their observed DiMeLo 6mA signal varies substantially. This illustrates why read-level prediction is difficult: AlphaGenome captures a locus-level regulatory propensity, while individual read-level DiMeLo observations are variable/sparse. Aggregating across reads recovers the locus-level signal.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a same-locus read-level example explaining AlphaGenome read-level failure.")
    parser.add_argument("--regions", type=Path, default=Path("outputs/4chrom_test_regions.tsv"))
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/outputs"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("cache_readseq_10reads"))
    parser.add_argument("--bam-c1", type=Path, default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam"))
    parser.add_argument("--bam-e5b", type=Path, default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/alphagenome_readlevel_locus_example"))
    parser.add_argument("--bin-size", type=int, default=200)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--min-reads", type=int, default=8)
    parser.add_argument("--min-observations", type=int, default=500)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    read_bins = collect_read_bin_rows(
        args.regions,
        args.outputs_dir,
        args.cache_dir,
        {"merged_c1": args.bam_c1, "merged_e5b": args.bam_e5b},
        args.bin_size,
        args.split,
    )
    read_bins.to_csv(args.out_dir / "all_read_bin_values.tsv", sep="\t", index=False)
    example, best = select_example(read_bins, args.min_reads, args.min_observations)
    display = example.sort_values("dimelo_6ma_mean").copy()
    display["same_genomic_locus"] = "yes"
    display.to_csv(args.out_dir / "same_locus_read_table.tsv", sep="\t", index=False)
    plot_example(display, best, args.out_dir / "same_locus_read_example.png")
    write_interpretation(args.out_dir / "interpretation.md", best, display)
    print(f"Wrote example to {args.out_dir}")
    print(best.to_string())


if __name__ == "__main__":
    main()
