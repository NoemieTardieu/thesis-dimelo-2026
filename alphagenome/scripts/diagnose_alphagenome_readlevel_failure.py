#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_utils import CHROMS, read_tsv
from evaluate_alphagenome_readlevel import (
    collect_observations,
    load_cached_tracks,
)


def load_observations(cache_dir: Path, outputs_dir: Path, split: str) -> pd.DataFrame:
    cached_tracks = load_cached_tracks(cache_dir)
    observations = []
    for chrom in CHROMS:
        prefix = (
            outputs_dir
            / f"merged_e5b_c1_{chrom}_selected_top100_overlap16k_full5000_region_split.{split}"
        )
        metadata = read_tsv(Path(f"{prefix}.metadata.tsv"))
        archive = np.load(Path(f"{prefix}.npz"))
        rows, _ = collect_observations(
            metadata,
            archive["target_6mA"],
            archive["mask_6mA"].astype(bool),
            cached_tracks,
        )
        observations.extend(rows)
        archive.close()
    if not observations:
        raise SystemExit("No observations matched the AlphaGenome read-sequence cache.")
    return pd.DataFrame(observations)


def decile_table(obs: pd.DataFrame, positive_cutoff: float) -> pd.DataFrame:
    frame = obs.copy()
    frame["score_decile"] = pd.qcut(
        frame["alphagenome_h3k4me3"].rank(method="first"),
        10,
        labels=False,
    ) + 1
    grouped = frame.groupby("score_decile", observed=True)
    return grouped.agg(
        observations=("target_6ma", "size"),
        reads=("read_id", "nunique"),
        alpha_min=("alphagenome_h3k4me3", "min"),
        alpha_median=("alphagenome_h3k4me3", "median"),
        alpha_max=("alphagenome_h3k4me3", "max"),
        mean_6ma=("target_6ma", "mean"),
        positive_fraction=("target_6ma", lambda x: float((x >= positive_cutoff).mean())),
        std_6ma=("target_6ma", "std"),
    ).reset_index()


def plot_deciles(table: pd.DataFrame, out: Path) -> None:
    fig, axis1 = plt.subplots(figsize=(8, 4.5))
    axis1.plot(table["score_decile"], table["positive_fraction"], marker="o", color="black")
    axis1.set_xlabel("AlphaGenome H3K4me3 score decile")
    axis1.set_ylabel("DiMeLo 6mA positive fraction")
    axis1.set_title("Read-level calibration: AlphaGenome score vs DiMeLo positivity")
    axis1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_hexbin(obs: pd.DataFrame, out: Path, max_points: int, seed: int) -> None:
    frame = obs
    if len(frame) > max_points:
        frame = frame.sample(max_points, random_state=seed)
    fig, axis = plt.subplots(figsize=(7, 5.5))
    hb = axis.hexbin(
        frame["alphagenome_h3k4me3"],
        frame["target_6ma"],
        gridsize=70,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    axis.set_xlabel("AlphaGenome read-sequence H3K4me3 score")
    axis.set_ylabel("DiMeLo read-level 6mA probability")
    axis.set_title("Read-level AlphaGenome vs DiMeLo observations")
    fig.colorbar(hb, ax=axis, label="log10(count)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_per_read(per_read: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, metric in zip(axes.ravel(), ["pearson", "spearman", "auroc", "auprc"]):
        values = pd.to_numeric(per_read[metric], errors="coerce").dropna()
        axis.hist(values, bins=40, color="slateblue", alpha=0.85)
        axis.axvline(values.median(), color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{metric} per read; median={values.median():.3f}")
    fig.suptitle("Distribution of read-level AlphaGenome metrics across reads")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def example_table(obs: pd.DataFrame, positive_cutoff: float) -> pd.DataFrame:
    frame = obs.copy()
    frame["alpha_rounded"] = frame["alphagenome_h3k4me3"].round(1)
    grouped = frame.groupby(["sample", "read_id", "alpha_rounded"], observed=True)
    table = grouped.agg(
        observations=("target_6ma", "size"),
        mean_alpha=("alphagenome_h3k4me3", "mean"),
        mean_6ma=("target_6ma", "mean"),
        std_6ma=("target_6ma", "std"),
        positive_fraction=("target_6ma", lambda x: float((x >= positive_cutoff).mean())),
    ).reset_index()
    table = table[table["observations"] >= 100]
    table["mixed_label_score"] = table["std_6ma"].fillna(0) * table["observations"].pow(0.25)
    return table.sort_values("mixed_label_score", ascending=False).head(30)


def write_interpretation(path: Path, summary: pd.DataFrame, deciles: pd.DataFrame) -> None:
    pooled = summary[summary["sample"] == "pooled"].iloc[0]
    lowest = deciles.iloc[0]
    highest = deciles.iloc[-1]
    text = f"""# AlphaGenome Read-Level Failure Diagnostics

AlphaGenome was run on original ONT read sequences and evaluated against DiMeLo 6mA observations at read positions.

## Main Read-Level Result

- Pooled Pearson: {pooled['pearson']:.3f}
- Pooled Spearman: {pooled['spearman']:.3f}
- Pooled AUROC: {pooled['auroc']:.3f}
- Pooled AUPRC: {pooled['auprc']:.3f}
- Positive fraction: {pooled['positive_fraction']:.3f}

## What The Diagnostics Show

The read-level relationship is weak even for AlphaGenome. The lowest AlphaGenome score decile has a DiMeLo positive fraction of {lowest['positive_fraction']:.3f}, while the highest AlphaGenome score decile has a positive fraction of {highest['positive_fraction']:.3f}. This weak separation explains why AUROC/AUPRC are low at read level.

The most likely interpretation is that AlphaGenome predicts a population/reference-like A549 H3K4me3 regulatory propensity, while single-read DiMeLo 6mA observations are sparse and noisy. The biological signal becomes visible after aggregating across reads and genomic bins, but individual read positions are not reliably predictable from sequence-only H3K4me3 propensity.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create diagnostics explaining weak AlphaGenome read-level metrics.")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache_readseq_10reads"))
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/outputs"),
    )
    parser.add_argument("--summary", type=Path, default=Path("outputs/alphagenome_readlevel_10reads.summary.tsv"))
    parser.add_argument("--per-read", type=Path, default=Path("outputs/alphagenome_readlevel_10reads.per_read.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/alphagenome_readlevel_diagnostics"))
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--positive-cutoff", type=float, default=0.5)
    parser.add_argument("--max-hexbin-points", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    obs = load_observations(args.cache_dir, args.outputs_dir, args.split)
    summary = pd.read_csv(args.summary, sep="\t")
    per_read = pd.read_csv(args.per_read, sep="\t")

    deciles = decile_table(obs, args.positive_cutoff)
    examples = example_table(obs, args.positive_cutoff)
    deciles.to_csv(args.out_dir / "score_deciles.tsv", sep="\t", index=False)
    examples.to_csv(args.out_dir / "mixed_label_examples.tsv", sep="\t", index=False)
    plot_deciles(deciles, args.out_dir / "score_deciles.png")
    plot_hexbin(obs, args.out_dir / "readlevel_hexbin.png", args.max_hexbin_points, args.seed)
    plot_per_read(per_read, args.out_dir / "per_read_metric_histograms.png")
    write_interpretation(args.out_dir / "interpretation.md", summary, deciles)
    print(f"Wrote diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
