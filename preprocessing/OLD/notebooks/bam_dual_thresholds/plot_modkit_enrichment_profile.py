#!/usr/bin/env python3
"""Plot a DiMeLo-style enrichment profile from two bigWig tracks over BED regions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig


@dataclass
class Region:
    chrom: str
    center: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regions-bed", required=True, help="BED file of regions/features")
    p.add_argument("--a-bigwig", required=True, help="A-channel signal bigWig")
    p.add_argument("--c-bigwig", required=True, help="C-channel signal bigWig")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--title", default="plot_enrichment_profile")
    p.add_argument("--a-label", default="mA")
    p.add_argument("--c-label", default="mCpG")
    p.add_argument("--upstream", type=int, default=5000)
    p.add_argument("--downstream", type=int, default=5000)
    p.add_argument("--bin-size", type=int, default=200)
    p.add_argument("--max-regions", type=int, default=20000)
    return p.parse_args()


def load_regions(path: str, max_regions: int) -> list[Region]:
    regions: list[Region] = []
    with open(path) as handle:
        for i, line in enumerate(handle):
            if max_regions and i >= max_regions:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            start = int(parts[1])
            end = int(parts[2])
            center = (start + end) // 2
            regions.append(Region(parts[0], center))
    return regions


def build_matrix(bw_path: str, regions: list[Region], upstream: int, downstream: int, bin_size: int) -> np.ndarray:
    bins = (upstream + downstream) // bin_size
    matrix = np.zeros((len(regions), bins), dtype=np.float32)
    with pyBigWig.open(bw_path) as bw:
        chrom_sizes = bw.chroms()
        for i, region in enumerate(regions):
            chrom_len = chrom_sizes.get(region.chrom)
            if chrom_len is None:
                continue
            start = max(0, region.center - upstream)
            end = min(chrom_len, region.center + downstream)
            vals = bw.stats(region.chrom, start, end, nBins=bins, type="mean")
            matrix[i, :] = np.array(
                [0.0 if v is None or np.isnan(v) else v for v in vals],
                dtype=np.float32,
            )
    return matrix


def main() -> None:
    args = parse_args()
    regions = load_regions(args.regions_bed, args.max_regions)
    if not regions:
        raise SystemExit("No regions loaded from BED")

    a_matrix = build_matrix(args.a_bigwig, regions, args.upstream, args.downstream, args.bin_size)
    c_matrix = build_matrix(args.c_bigwig, regions, args.upstream, args.downstream, args.bin_size)

    x = np.arange(-args.upstream, args.downstream, args.bin_size)
    if len(x) != a_matrix.shape[1]:
        x = np.linspace(-args.upstream, args.downstream, a_matrix.shape[1], endpoint=False)

    a_profile = a_matrix.mean(axis=0)
    c_profile = c_matrix.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    ax.plot(x, a_profile, color="#1f4f7a", linewidth=2.5, label=args.a_label)
    ax.plot(x, c_profile, color="#c55038", linewidth=2.5, label=args.c_label)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Fraction methylated")
    ax.set_title(args.title, fontsize=18, pad=12)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"regions={len(regions)}")


if __name__ == "__main__":
    main()
