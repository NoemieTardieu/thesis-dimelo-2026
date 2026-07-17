#!/usr/bin/env python3
"""Evaluate cached AlphaGenome predictions against ENCODE BigWig.

This is a focused sensitivity/audit script for the suspiciously high
AlphaGenome-vs-ENCODE correlation. It supports both the original selected
DiMeLo/HyenaDNA regions and independent random regions, and reports metrics at
the native 128 bp AlphaGenome grid plus rebinned 200 bp bins.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

from benchmark_utils import BIN_SIZE, cache_name, load_prediction_cache, load_regions, read_tsv


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation."""

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Spearman correlation."""

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(float)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(float)
    return pearson(xr, yr)


def normalized_errors(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Calculate MAE/RMSE after robust global 1-99 percentile scaling."""

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.nan, np.nan

    def scale(values: np.ndarray) -> np.ndarray:
        p01, p99 = np.quantile(values, [0.01, 0.99])
        if p99 <= p01:
            return np.full(values.shape, np.nan)
        return np.clip((values - p01) / (p99 - p01), 0.0, 1.0)

    xs = scale(x)
    ys = scale(y)
    return float(np.nanmean(np.abs(xs - ys))), float(np.sqrt(np.nanmean((xs - ys) ** 2)))


def bigwig_mean(handle: pyBigWig.pyBigWig, chrom: str, start: int, end: int) -> float:
    """Extract exact mean BigWig signal."""

    try:
        value = handle.stats(chrom, start, end, type="mean", exact=True)[0]
    except RuntimeError:
        value = None
    return np.nan if value is None else float(value)


def selected_track_indices(metadata: list[dict], selected_names: list[str]) -> list[int]:
    """Find selected AlphaGenome track indices in one cache metadata object."""

    names = [str(row["name"]) for row in metadata]
    indices = [idx for idx, name in enumerate(names) if name in selected_names]
    if not indices:
        raise SystemExit("No selected A549 H3K4me3 tracks found in AlphaGenome cache.")
    return indices


def export_128bp_bins(regions_path: Path, selected_tracks: Path, cache_dir: Path, bigwig: Path) -> pd.DataFrame:
    """Create paired AlphaGenome/ENCODE bins at native 128 bp resolution."""

    selected = read_tsv(selected_tracks)
    ontology_terms = sorted({row["ontology_curie"] for row in selected})
    selected_names = [row["name"] for row in selected]
    regions = load_regions(regions_path, split="test")
    rows = []

    with pyBigWig.open(str(bigwig)) as bw:
        for region in regions:
            cache_path = cache_dir / cache_name(region, ontology_terms)
            values, metadata, provenance = load_prediction_cache(cache_path)
            resolution = int(provenance["resolution"])
            if resolution != BIN_SIZE:
                raise SystemExit(f"Expected {BIN_SIZE} bp AlphaGenome output in {cache_path}, observed {resolution}")
            returned = provenance["returned_interval"]
            indices = selected_track_indices(metadata, selected_names)
            for bin_index in range(values.shape[0]):
                start = int(returned["start"]) + bin_index * resolution
                end = start + resolution
                if start < region.start or end > region.end:
                    continue
                alpha_values = [float(values[bin_index, idx]) for idx in indices]
                rows.append(
                    {
                        "chrom": region.chrom,
                        "region_id": region.region_id,
                        "region_name": region.name,
                        "region_start": region.start,
                        "region_end": region.end,
                        "start": start,
                        "end": end,
                        "bin_size": resolution,
                        "alphagenome_raw": float(np.mean(alpha_values)),
                        "external_raw": bigwig_mean(bw, region.chrom, start, end),
                    }
                )
    return pd.DataFrame(rows)


def rebin(data: pd.DataFrame, bin_size: int, bigwig: Path) -> pd.DataFrame:
    """Rebin paired 128 bp AlphaGenome bins to a larger grid."""

    if bin_size == BIN_SIZE:
        return data.copy()
    rows = []
    with pyBigWig.open(str(bigwig)) as bw:
        for (chrom, region_id, region_name, region_start, region_end), region in data.groupby(
            ["chrom", "region_id", "region_name", "region_start", "region_end"], sort=False
        ):
            region = region.sort_values("start")
            for start in range(int(region_start), int(region_end) - bin_size + 1, bin_size):
                end = start + bin_size
                overlap = region[(region["start"] < end) & (region["end"] > start)]
                if overlap.empty:
                    continue
                weights = np.minimum(overlap["end"].to_numpy(int), end) - np.maximum(
                    overlap["start"].to_numpy(int), start
                )
                alpha = np.average(overlap["alphagenome_raw"].to_numpy(float), weights=weights)
                rows.append(
                    {
                        "chrom": chrom,
                        "region_id": region_id,
                        "region_name": region_name,
                        "region_start": region_start,
                        "region_end": region_end,
                        "start": start,
                        "end": end,
                        "bin_size": bin_size,
                        "alphagenome_raw": float(alpha),
                        "external_raw": bigwig_mean(bw, chrom, start, end),
                    }
                )
    return pd.DataFrame(rows)


def metric_rows(data: pd.DataFrame, label: str) -> pd.DataFrame:
    """Calculate pooled and chromosome-specific AlphaGenome-vs-ENCODE metrics."""

    rows = []
    for scope, subset in [("all", data), *[(chrom, d) for chrom, d in data.groupby("chrom", sort=True)]]:
        x = subset["alphagenome_raw"].to_numpy(float)
        y = subset["external_raw"].to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        mae, rmse = normalized_errors(x, y)
        rows.append(
            {
                "label": label,
                "scope": scope,
                "bin_size": int(subset["bin_size"].iloc[0]) if not subset.empty else np.nan,
                "n_bins": int(mask.sum()),
                "pearson": pearson(x, y),
                "spearman": spearman(x, y),
                "normalized_mae": mae,
                "normalized_rmse": rmse,
                "alphagenome_mean": float(np.nanmean(x)) if x.size else np.nan,
                "external_mean": float(np.nanmean(y)) if y.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write TSV or TSV.GZ."""

    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(path, sep="\t", index=False, compression=compression)


def write_summary(metrics: pd.DataFrame, out: Path, args: argparse.Namespace) -> None:
    """Write a compact Markdown summary."""

    pooled = metrics[metrics["scope"] == "all"].copy()
    lines = [
        "# AlphaGenome-ENCODE Sensitivity Check",
        "",
        "This analysis re-checks AlphaGenome A549 H3K4me3 versus ENCODE A549 H3K4me3 outside the main DiMeLo/HyenaDNA population benchmark.",
        "",
        f"- Regions: `{args.regions}`",
        f"- Cache directory: `{args.cache_dir}`",
        f"- External BigWig: `{args.bigwig}`",
        "",
        "| label | bin size | n bins | Pearson | Spearman | normalized MAE | normalized RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.bin_size} | {row.n_bins} | {row.pearson:.4f} | "
            f"{row.spearman:.4f} | {row.normalized_mae:.4f} | {row.normalized_rmse:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation note: if the random-region correlation remains near the selected-region value, the high AlphaGenome-ENCODE agreement is likely driven by matched/in-distribution A549 H3K4me3 signal rather than only DiMeLo-region selection. If it drops substantially, the original 0.95 was likely inflated by selected regions and/or 200 bp smoothing.",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--selected-tracks",
        type=Path,
        default=Path("metadata/selected_a549_h3k4me3_tracks.tsv"),
    )
    parser.add_argument(
        "--bigwig",
        type=Path,
        default=Path("server_artifacts/external_tracks/ENCSR203XPU/ENCFF074PND_ENCSR203XPU_A549_H3K4me3_fold_change_GRCh38.bigWig"),
    )
    parser.add_argument("--label", default="random_regions")
    parser.add_argument("--out-dir", type=Path, default=Path("results/alphagenome_encode_sensitivity"))
    parser.add_argument("--write-bins", action="store_true")
    args = parser.parse_args()

    bins128 = export_128bp_bins(args.regions, args.selected_tracks, args.cache_dir, args.bigwig)
    bins200 = rebin(bins128, 200, args.bigwig)
    metrics = pd.concat(
        [
            metric_rows(bins128, f"{args.label}_128bp"),
            metric_rows(bins200, f"{args.label}_200bp"),
        ],
        ignore_index=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_table(metrics, args.out_dir / f"{args.label}.metrics.tsv")
    if args.write_bins:
        write_table(bins128, args.out_dir / f"{args.label}.paired_bins_128bp.tsv.gz")
        write_table(bins200, args.out_dir / f"{args.label}.paired_bins_200bp.tsv.gz")
    provenance = {
        "regions": str(args.regions),
        "cache_dir": str(args.cache_dir),
        "selected_tracks": str(args.selected_tracks),
        "bigwig": str(args.bigwig),
        "label": args.label,
    }
    (args.out_dir / f"{args.label}.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    write_summary(metrics, args.out_dir / f"{args.label}.summary.md", args)
    print(f"Wrote sensitivity metrics to {args.out_dir / f'{args.label}.metrics.tsv'}")


if __name__ == "__main__":
    main()
