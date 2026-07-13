#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def minmax(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return values * np.nan
    return (values - lo) / (hi - lo)


def plot_region(
    alpha: pd.DataFrame,
    dimelo: pd.DataFrame,
    hyena: pd.DataFrame,
    chrom: str,
    region_id: int,
    out_dir: Path,
) -> None:
    region_rows = alpha[(alpha["chrom"] == chrom) & (alpha["region_id"] == region_id)]
    if region_rows.empty:
        return
    region_name = region_rows["region_name"].iloc[0]

    fig, axis = plt.subplots(figsize=(13, 5))
    for sample, alpha_color, dimelo_color, hyena_color in (
        ("merged_c1", "dimgray", "tab:blue", "tab:purple"),
        ("merged_e5b", "black", "tab:orange", "deeppink"),
    ):
        a = region_rows[region_rows["sample"] == sample].copy()
        d = dimelo[
            (dimelo["chrom"] == chrom)
            & (dimelo["region_id"] == region_id)
            & (dimelo["sample"] == sample)
        ].copy()
        h = hyena[
            (hyena["chrom"] == chrom)
            & (hyena["region_id"] == region_id)
            & (hyena["sample"] == sample)
        ].copy()
        if a.empty or d.empty or h.empty:
            continue
        axis.plot(
            (a["bin_start"] + a["bin_end"]) / 2,
            minmax(a["mean_signal"]),
            color=alpha_color,
            linewidth=1.6,
            label=f"AlphaGenome ONT-read {sample}",
        )
        axis.plot(
            (d["bin_start"] + d["bin_end"]) / 2,
            minmax(d["mean_signal"]),
            color=dimelo_color,
            linewidth=1.3,
            alpha=0.85,
            label=f"DiMeLo {sample}",
        )
        axis.plot(
            (h["bin_start"] + h["bin_end"]) / 2,
            minmax(h["mean_signal"]),
            color=hyena_color,
            linewidth=1.3,
            linestyle="--",
            alpha=0.9,
            label=f"HyenaDNA {sample}",
        )

    axis.set_xlabel(f"{chrom} coordinate (hg38)")
    axis.set_ylabel("minmax normalized signal")
    axis.set_ylim(-0.05, 1.05)
    axis.legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle(
        f"{chrom} region {region_id}: ONT-read AlphaGenome normalized tracks\n{region_name}"
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{chrom}_region{region_id}_readseq_alphagenome_minmax.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot normalized AlphaGenome ONT-read, DiMeLo, and HyenaDNA tracks."
    )
    parser.add_argument(
        "--alphagenome-readseq",
        type=Path,
        default=Path("outputs/alphagenome_readseq_200bp_10reads.tsv"),
    )
    parser.add_argument(
        "--dimelo",
        type=Path,
        default=Path("outputs/benchmark_200bp/dimelo_test_200bp.tsv"),
    )
    parser.add_argument(
        "--hyena",
        type=Path,
        default=Path("outputs/benchmark_200bp/hyenadna_test_200bp.tsv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/alphagenome_readseq_200bp_10reads_normalized_plots"),
    )
    parser.add_argument("--max-regions", type=int, default=12)
    args = parser.parse_args()

    alpha = pd.read_csv(args.alphagenome_readseq, sep="\t")
    dimelo = pd.read_csv(args.dimelo, sep="\t")
    hyena = pd.read_csv(args.hyena, sep="\t")
    regions = (
        alpha[["chrom", "region_id"]]
        .drop_duplicates()
        .sort_values(["chrom", "region_id"])
        .head(args.max_regions)
    )
    for row in regions.itertuples(index=False):
        plot_region(alpha, dimelo, hyena, row.chrom, int(row.region_id), args.out_dir)
    print(f"Wrote {len(regions)} plots to {args.out_dir}")


if __name__ == "__main__":
    main()
