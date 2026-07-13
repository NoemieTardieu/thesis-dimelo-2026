#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["chrom", "region_id", "region_start", "region_end", "bin_start", "bin_end"]


def complete_bins(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(bin_start, bin_start + size) for bin_start in range(start, end - size + 1, size)]


def overlap_weights(starts: np.ndarray, ends: np.ndarray, target_start: int, target_end: int) -> np.ndarray:
    return np.maximum(0, np.minimum(ends, target_end) - np.maximum(starts, target_start)).astype(float)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlap-weight AlphaGenome native bins onto a 200 bp grid.")
    parser.add_argument("--in-tsv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bin-size", type=int, default=200)
    args = parser.parse_args()

    alpha = pd.read_csv(args.in_tsv, sep="\t")
    track_cols = [col for col in alpha.columns if col not in KEYS + ["region_name"]]
    rows = []
    for region in alpha[["chrom", "region_id", "region_start", "region_end", "region_name"]].drop_duplicates().itertuples(index=False):
        subset = alpha[(alpha["chrom"] == region.chrom) & (alpha["region_id"] == region.region_id)].sort_values("bin_start")
        starts = subset["bin_start"].to_numpy(int)
        ends = subset["bin_end"].to_numpy(int)
        values = {track: subset[track].to_numpy(float) for track in track_cols}
        for bin_start, bin_end in complete_bins(int(region.region_start), int(region.region_end), args.bin_size):
            weights = overlap_weights(starts, ends, bin_start, bin_end)
            row = {
                "chrom": region.chrom,
                "region_id": region.region_id,
                "region_name": region.region_name,
                "region_start": int(region.region_start),
                "region_end": int(region.region_end),
                "bin_start": bin_start,
                "bin_end": bin_end,
            }
            for track in track_cols:
                row[track] = weighted_mean(values[track], weights)
            rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(rows)} AlphaGenome bins to {args.out}")


if __name__ == "__main__":
    main()
