#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_alphagenome_vs_dimelo import add_result


KEYS = ["chrom", "region_id", "region_start", "region_end", "bin_start", "bin_end"]
ALPHA_TRACK = "A549_H3K4me3_fixed_mean"


def complete_bins(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(bin_start, bin_start + size) for bin_start in range(start, end - size + 1, size)]


def overlap_weights(
    starts: np.ndarray,
    ends: np.ndarray,
    target_start: int,
    target_end: int,
) -> np.ndarray:
    return np.maximum(
        0,
        np.minimum(ends, target_end) - np.maximum(starts, target_start),
    ).astype(float)


def weighted_mean_from_arrays(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    mask = (~np.isnan(values)) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def rebin_alpha(alpha: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    rows = []
    track_cols = [column for column in alpha.columns if column not in KEYS + ["region_name"]]
    for _, region in alpha[
        ["chrom", "region_id", "region_start", "region_end", "region_name"]
    ].drop_duplicates().iterrows():
        subset = alpha[
            (alpha["chrom"] == region["chrom"])
            & (alpha["region_id"] == region["region_id"])
        ].sort_values("bin_start")
        starts = subset["bin_start"].to_numpy(int)
        ends = subset["bin_end"].to_numpy(int)
        values = {track: subset[track].to_numpy(float) for track in track_cols}
        for bin_start, bin_end in complete_bins(int(region["region_start"]), int(region["region_end"]), bin_size):
            weights = overlap_weights(starts, ends, bin_start, bin_end)
            row = {
                "chrom": region["chrom"],
                "region_id": region["region_id"],
                "region_start": int(region["region_start"]),
                "region_end": int(region["region_end"]),
                "region_name": region["region_name"],
                "bin_start": bin_start,
                "bin_end": bin_end,
            }
            for track in track_cols:
                row[track] = weighted_mean_from_arrays(values[track], weights)
            rows.append(row)
    return pd.DataFrame(rows)


def rebin_signal_track(track: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    rows = []
    for _, region in track[
        ["sample", "chrom", "region_id", "region_start", "region_end", "region_name"]
    ].drop_duplicates().iterrows():
        subset = track[
            (track["sample"] == region["sample"])
            & (track["chrom"] == region["chrom"])
            & (track["region_id"] == region["region_id"])
        ].sort_values("bin_start")
        starts = subset["bin_start"].to_numpy(int)
        ends = subset["bin_end"].to_numpy(int)
        mean_signal = subset["mean_signal"].to_numpy(float)
        positive_fraction = subset["positive_fraction_0_5"].to_numpy(float)
        observed_positions = subset["observed_positions"].fillna(0).to_numpy(float)
        unique_reads = subset["unique_reads"].fillna(0).to_numpy(float)
        for bin_start, bin_end in complete_bins(int(region["region_start"]), int(region["region_end"]), bin_size):
            overlap_bp = overlap_weights(starts, ends, bin_start, bin_end)
            signal_weights = overlap_bp * observed_positions
            overlapping = overlap_bp > 0
            row = {
                "sample": region["sample"],
                "chrom": region["chrom"],
                "region_id": region["region_id"],
                "region_start": int(region["region_start"]),
                "region_end": int(region["region_end"]),
                "region_name": region["region_name"],
                "bin_start": bin_start,
                "bin_end": bin_end,
                "mean_signal": weighted_mean_from_arrays(mean_signal, signal_weights),
                "positive_fraction_0_5": weighted_mean_from_arrays(
                    positive_fraction, signal_weights
                ),
                "unique_reads": int(np.nanmax(unique_reads[overlapping])) if np.any(overlapping) else 0,
                "observed_positions": int(np.nansum(observed_positions[overlapping])) if np.any(overlapping) else 0,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def derive_threshold(dimelo_val: pd.DataFrame, min_reads: int, min_positions: int) -> float:
    covered = dimelo_val[
        (dimelo_val["sample"] == "pooled")
        & (dimelo_val["unique_reads"] >= min_reads)
        & (dimelo_val["observed_positions"] >= min_positions)
        & dimelo_val["mean_signal"].notna()
    ]
    if covered.empty:
        raise SystemExit("No covered pooled validation bins remain after 200 bp rebinning.")
    return float(np.quantile(covered["mean_signal"].to_numpy(float), 0.90))


def evaluate(
    alpha: pd.DataFrame,
    dimelo: pd.DataFrame,
    hyena: pd.DataFrame,
    threshold: float,
    min_reads: int,
    min_positions: int,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    results: list[dict] = []
    for sample in ("merged_c1", "merged_e5b", "pooled"):
        target = dimelo[
            (dimelo["sample"] == sample)
            & (dimelo["unique_reads"] >= min_reads)
            & (dimelo["observed_positions"] >= min_positions)
            & dimelo["mean_signal"].notna()
        ]
        alpha_data = target.merge(alpha, on=KEYS, how="inner", validate="one_to_one")
        hyena_sample = hyena[hyena["sample"] == sample][KEYS + ["mean_signal"]].rename(
            columns={"mean_signal": "HyenaDNA"}
        )
        hyena_data = target.merge(hyena_sample, on=KEYS, how="inner", validate="one_to_one")
        hyena_data = hyena_data[hyena_data["HyenaDNA"].notna()]
        add_result(
            results,
            alpha_data,
            "AlphaGenome",
            ALPHA_TRACK,
            sample,
            "pooled",
            "top_10_percent_primary_200bp",
            threshold,
            bootstrap_replicates,
            seed,
        )
        add_result(
            results,
            hyena_data,
            "HyenaDNA",
            "HyenaDNA",
            sample,
            "pooled",
            "top_10_percent_primary_200bp",
            threshold,
            bootstrap_replicates,
            seed,
        )
    return pd.DataFrame(results)


def plot_regions(alpha: pd.DataFrame, dimelo: pd.DataFrame, hyena: pd.DataFrame, out_dir: Path, max_regions: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = alpha[["chrom", "region_id", "region_name"]].drop_duplicates().head(max_regions)
    colors = {
        ("DiMeLo", "merged_c1"): "tab:blue",
        ("DiMeLo", "merged_e5b"): "tab:orange",
        ("HyenaDNA", "merged_c1"): "tab:purple",
        ("HyenaDNA", "merged_e5b"): "deeppink",
    }
    for _, selected in regions.iterrows():
        chrom = selected["chrom"]
        region_id = selected["region_id"]
        region_name = selected["region_name"]
        a = alpha[(alpha["chrom"] == chrom) & (alpha["region_id"] == region_id)]
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        x = (a["bin_start"] + a["bin_end"]) / 2
        axes[0].plot(x, a[ALPHA_TRACK], color="black", lw=1.2)
        axes[0].set_ylabel("AlphaGenome\nH3K4me3")
        for sample in ("merged_c1", "merged_e5b"):
            d = dimelo[
                (dimelo["chrom"] == chrom)
                & (dimelo["region_id"] == region_id)
                & (dimelo["sample"] == sample)
            ]
            h = hyena[
                (hyena["chrom"] == chrom)
                & (hyena["region_id"] == region_id)
                & (hyena["sample"] == sample)
            ]
            axes[1].plot(
                (d["bin_start"] + d["bin_end"]) / 2,
                d["mean_signal"],
                label=sample,
                color=colors[("DiMeLo", sample)],
            )
            axes[2].plot(
                (h["bin_start"] + h["bin_end"]) / 2,
                h["mean_signal"],
                label=sample,
                color=colors[("HyenaDNA", sample)],
            )
        axes[1].set_ylabel("DiMeLo 6mA")
        axes[2].set_ylabel("HyenaDNA")
        axes[2].set_xlabel(f"{chrom} coordinate (hg38)")
        axes[1].legend(frameon=False)
        axes[2].legend(frameon=False)
        fig.suptitle(f"{chrom} region {region_id}: 200 bp evaluation bins\n{region_name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{chrom}_region{region_id}_200bp.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebin the benchmark to 200 bp genomic evaluation bins.")
    parser.add_argument("--alpha", type=Path, default=Path("outputs/alphagenome_test_128bp.tsv"))
    parser.add_argument("--dimelo-test", type=Path, default=Path("outputs/dimelo_test_128bp.tsv"))
    parser.add_argument("--dimelo-val", type=Path, default=Path("outputs/dimelo_val_128bp.tsv"))
    parser.add_argument("--hyena", type=Path, default=Path("outputs/hyenadna_test_128bp.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/benchmark_200bp"))
    parser.add_argument("--bin-size", type=int, default=200)
    parser.add_argument("--minimum-unique-reads", type=int, default=2)
    parser.add_argument("--minimum-observed-positions", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--max-plotted-regions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    alpha_128 = pd.read_csv(args.alpha, sep="\t")
    dimelo_test_128 = pd.read_csv(args.dimelo_test, sep="\t")
    dimelo_val_128 = pd.read_csv(args.dimelo_val, sep="\t")
    hyena_128 = pd.read_csv(args.hyena, sep="\t")

    alpha = rebin_alpha(alpha_128, args.bin_size)
    dimelo_test = rebin_signal_track(dimelo_test_128, args.bin_size)
    dimelo_val = rebin_signal_track(dimelo_val_128, args.bin_size)
    hyena = rebin_signal_track(hyena_128, args.bin_size)
    threshold = derive_threshold(
        dimelo_val, args.minimum_unique_reads, args.minimum_observed_positions
    )
    summary = evaluate(
        alpha,
        dimelo_test,
        hyena,
        threshold,
        args.minimum_unique_reads,
        args.minimum_observed_positions,
        args.bootstrap_replicates,
        args.seed,
    )

    alpha.to_csv(args.out_dir / "alphagenome_test_200bp.tsv", sep="\t", index=False)
    dimelo_test.to_csv(args.out_dir / "dimelo_test_200bp.tsv", sep="\t", index=False)
    dimelo_val.to_csv(args.out_dir / "dimelo_val_200bp.tsv", sep="\t", index=False)
    hyena.to_csv(args.out_dir / "hyenadna_test_200bp.tsv", sep="\t", index=False)
    summary.to_csv(args.out_dir / "benchmark_200bp.summary.tsv", sep="\t", index=False)
    with open(args.out_dir / "thresholds_200bp.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "bin_size": args.bin_size,
                "threshold_name": "top_10_percent_primary_200bp",
                "threshold": threshold,
                "minimum_unique_reads": args.minimum_unique_reads,
                "minimum_observed_positions": args.minimum_observed_positions,
                "note": "AlphaGenome native 128 bp output was overlap-weighted onto 200 bp genomic evaluation bins.",
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    plot_regions(alpha, dimelo_test, hyena, args.out_dir / "plots", args.max_plotted_regions)
    print(f"Wrote 200 bp benchmark outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
