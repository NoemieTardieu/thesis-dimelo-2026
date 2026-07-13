#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


KEYS = ["chrom", "region_id", "region_start", "region_end", "bin_start", "bin_end"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot regional AlphaGenome, DiMeLo, and HyenaDNA tracks.")
    parser.add_argument("--alphagenome", type=Path, required=True)
    parser.add_argument("--dimelo", type=Path, required=True)
    parser.add_argument("--hyenadna", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--max-regions", type=int, default=12)
    args = parser.parse_args()

    alpha = pd.read_csv(args.alphagenome, sep="\t")
    dimelo = pd.read_csv(args.dimelo, sep="\t")
    hyena = pd.read_csv(args.hyenadna, sep="\t")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    regions = alpha[["chrom", "region_id"]].drop_duplicates().head(args.max_regions)
    for _, selected in regions.iterrows():
        chrom, region_id = selected["chrom"], selected["region_id"]
        a = alpha[(alpha["chrom"] == chrom) & (alpha["region_id"] == region_id)]
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        x = (a["bin_start"] + a["bin_end"]) / 2
        axes[0].plot(x, a["A549_H3K4me3_fixed_mean"], color="black", lw=1)
        axes[0].set_ylabel("AlphaGenome\nH3K4me3")
        for sample, color in (("merged_c1", "tab:blue"), ("merged_e5b", "tab:orange")):
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
            axes[1].plot((d["bin_start"] + d["bin_end"]) / 2, d["mean_signal"], label=sample, color=color)
            axes[2].plot((h["bin_start"] + h["bin_end"]) / 2, h["mean_signal"], label=sample, color=color)
        axes[1].set_ylabel("DiMeLo 6mA")
        axes[2].set_ylabel("HyenaDNA")
        axes[2].set_xlabel(f"{chrom} coordinate (hg38)")
        axes[1].legend(frameon=False)
        axes[2].legend(frameon=False)
        fig.suptitle(f"{chrom} region {region_id}: cross-assay comparison")
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{chrom}_region{region_id}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
