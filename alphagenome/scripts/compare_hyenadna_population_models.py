#!/usr/bin/env python3
"""Compare old and methylation-conditioned HyenaDNA population benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["chrom", "region_id", "region_name", "start", "end"]


@dataclass
class RobustParams:
    p01: float
    p99: float


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(path, sep="\t", index=False, compression=compression)


def fit_robust(values: pd.Series) -> RobustParams:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return RobustParams(np.nan, np.nan)
    return RobustParams(float(np.percentile(finite, 1)), float(np.percentile(finite, 99)))


def apply_robust(values: pd.Series, params: RobustParams) -> pd.Series:
    if not np.isfinite(params.p01) or not np.isfinite(params.p99) or params.p99 <= params.p01:
        return pd.Series(np.nan, index=values.index)
    return ((values - params.p01) / (params.p99 - params.p01)).clip(0, 1)


def extract_primary_metrics(path: Path, model_label: str) -> pd.DataFrame:
    metrics = read_tsv(path)
    subset = metrics[
        (metrics["scope"] == "all")
        & (metrics["intersection"] == "pair_specific")
        & metrics["pair"].isin(["H-D", "T-H", "A-H"])
    ].copy()
    subset.insert(0, "hyena_model", model_label)
    return subset


def make_overlay_plots(data: pd.DataFrame, out_dir: Path, max_regions: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = (
        data.groupby(["chrom", "region_id", "region_name"], sort=False)
        .size()
        .reset_index(name="n")
        .head(max_regions)
    )
    colors = {
        "external_norm": "tab:green",
        "alphagenome_norm": "black",
        "dimelo_norm": "tab:orange",
        "hyena_old_norm": "tab:purple",
        "hyena_new_norm": "deeppink",
    }
    labels = {
        "external_norm": "ENCODE A549 H3K4me3",
        "alphagenome_norm": "AlphaGenome A549 H3K4me3",
        "dimelo_norm": "DiMeLo observed 6mA",
        "hyena_old_norm": "HyenaDNA old",
        "hyena_new_norm": "HyenaDNA DNA+5mC",
    }
    for row in regions.itertuples(index=False):
        subset = data[
            (data["chrom"] == row.chrom)
            & (data["region_id"] == row.region_id)
            & (data["region_name"] == row.region_name)
        ].sort_values("start")
        if subset.empty:
            continue
        x = (subset["start"] + subset["end"]) / 2
        fig, ax = plt.subplots(figsize=(12, 4))
        for column in ["external_norm", "alphagenome_norm", "dimelo_norm", "hyena_old_norm", "hyena_new_norm"]:
            ax.plot(x, subset[column], lw=1.2, color=colors[column], label=labels[column])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f"{row.chrom} coordinate (GRCh38)")
        ax.set_ylabel("global robust-normalized signal")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        ax.set_title(f"{row.chrom} region {row.region_id}: old vs DNA+5mC HyenaDNA")
        fig.tight_layout()
        stem = f"{row.chrom}_region{row.region_id}_old_vs_new_hyenadna_overlay"
        fig.savefig(out_dir / f"{stem}.png", dpi=180)
        fig.savefig(out_dir / f"{stem}.svg")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-regions", type=int, default=12)
    args = parser.parse_args()

    old_raw = read_tsv(args.old_dir / "canonical_raw_bins.tsv.gz")
    new_raw = read_tsv(args.new_dir / "canonical_raw_bins.tsv.gz")
    merged = new_raw[
        KEYS
        + [
            "external_raw",
            "alphagenome_raw",
            "dimelo_raw",
            "dimelo_coverage",
            "hyena_raw",
            "hyena_prediction_count",
        ]
    ].rename(
        columns={
            "hyena_raw": "hyena_new_raw",
            "hyena_prediction_count": "hyena_new_prediction_count",
        }
    )
    old_h = old_raw[KEYS + ["hyena_raw", "hyena_prediction_count"]].rename(
        columns={
            "hyena_raw": "hyena_old_raw",
            "hyena_prediction_count": "hyena_old_prediction_count",
        }
    )
    merged = merged.merge(old_h, on=KEYS, how="left")

    for raw_col, norm_col in [
        ("external_raw", "external_norm"),
        ("alphagenome_raw", "alphagenome_norm"),
        ("dimelo_raw", "dimelo_norm"),
        ("hyena_old_raw", "hyena_old_norm"),
        ("hyena_new_raw", "hyena_new_norm"),
    ]:
        merged[norm_col] = apply_robust(merged[raw_col], fit_robust(merged[raw_col]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(merged, args.out_dir / "canonical_raw_and_global_normalized_old_new_hyena.tsv.gz")

    old_metrics = extract_primary_metrics(args.old_dir / "pairwise_metrics.tsv.gz", "old_hyenadna")
    new_metrics = extract_primary_metrics(args.new_dir / "pairwise_metrics.tsv.gz", "dna_plus_5mc_hyenadna")
    metrics = pd.concat([old_metrics, new_metrics], ignore_index=True)
    write_tsv(metrics, args.out_dir / "hyenadna_model_metric_comparison.tsv")

    wide = metrics.pivot_table(
        index=["pair"],
        columns="hyena_model",
        values=["pearson", "spearman", "normalized_mae", "normalized_rmse", "n_bins"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    if {"pearson_dna_plus_5mc_hyenadna", "pearson_old_hyenadna"}.issubset(wide.columns):
        wide["delta_pearson_new_minus_old"] = wide["pearson_dna_plus_5mc_hyenadna"] - wide["pearson_old_hyenadna"]
    if {"spearman_dna_plus_5mc_hyenadna", "spearman_old_hyenadna"}.issubset(wide.columns):
        wide["delta_spearman_new_minus_old"] = wide["spearman_dna_plus_5mc_hyenadna"] - wide["spearman_old_hyenadna"]
    if {"normalized_mae_dna_plus_5mc_hyenadna", "normalized_mae_old_hyenadna"}.issubset(wide.columns):
        wide["delta_normalized_mae_new_minus_old"] = (
            wide["normalized_mae_dna_plus_5mc_hyenadna"] - wide["normalized_mae_old_hyenadna"]
        )
    write_tsv(wide, args.out_dir / "hyenadna_model_metric_deltas.tsv")

    make_overlay_plots(merged, args.out_dir / "overlay_selected_regions_old_new_hyena", args.max_regions)

    with (args.out_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Old vs DNA+5mC HyenaDNA Population Benchmark Comparison\n\n")
        handle.write("This folder compares the previous HyenaDNA population track with the final DNA+5mC-conditioned HyenaDNA track.\n\n")
        handle.write("- `hyenadna_model_metric_comparison.tsv`: old/new pairwise metrics involving HyenaDNA.\n")
        handle.write("- `hyenadna_model_metric_deltas.tsv`: new-minus-old metric deltas.\n")
        handle.write("- `canonical_raw_and_global_normalized_old_new_hyena.tsv.gz`: merged bin table with both HyenaDNA tracks.\n")
        handle.write("- `overlay_selected_regions_old_new_hyena/`: locus plots with both HyenaDNA lines.\n")


if __name__ == "__main__":
    main()
