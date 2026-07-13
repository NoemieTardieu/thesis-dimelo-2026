#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_alphagenome_vs_dimelo import metrics

KEYS = ["chrom", "region_id", "region_start", "region_end", "bin_start", "bin_end"]


def minmax(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return values * np.nan
    return (values - lo) / (hi - lo)


def zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    mean = values.mean(skipna=True)
    std = values.std(skipna=True)
    if pd.isna(mean) or pd.isna(std) or std == 0:
        return values * np.nan
    return (values - mean) / std


def normalize(values: pd.Series, mode: str) -> pd.Series:
    if mode == "zscore":
        return zscore(values)
    if mode == "minmax":
        return minmax(values)
    raise ValueError(f"Unknown normalization mode: {mode}")


def write_promoter_metrics(summary_path: Path, out_path: Path) -> None:
    summary = pd.read_csv(summary_path, sep="\t")
    selected = summary[
        (summary["threshold_name"] == "top_10_percent_primary")
        & summary["track"].isin(["A549_H3K4me3_fixed_mean", "HyenaDNA"])
    ].copy()
    selected = selected[
        [
            "scope",
            "model",
            "track",
            "experimental_sample",
            "number_of_regions",
            "number_of_bins",
            "pearson",
            "pearson_ci_low",
            "pearson_ci_high",
            "spearman",
            "spearman_ci_low",
            "spearman_ci_high",
            "auroc",
            "auroc_ci_low",
            "auroc_ci_high",
            "auprc",
            "auprc_ci_low",
            "auprc_ci_high",
            "positive_fraction",
            "auprc_enrichment",
        ]
    ]
    selected.to_csv(out_path, sep="\t", index=False)
    pooled = selected[selected["scope"] == "pooled"]
    md_path = out_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# AlphaGenome vs HyenaDNA Averaged-Signal Metrics\n\n")
        handle.write(
            "Primary threshold: validation-derived top 10% pooled DiMeLo 6mA signal. "
            "Confidence intervals use region-level bootstrap replicates.\n\n"
        )
        cols = [
            "model",
            "track",
            "experimental_sample",
            "number_of_bins",
            "pearson",
            "spearman",
            "auroc",
            "auprc",
            "positive_fraction",
            "auprc_enrichment",
        ]
        handle.write("| " + " | ".join(cols) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in pooled[cols].iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")
        handle.write("\n")
    print(f"Wrote {out_path} and {md_path}")


def plot_normalized_tracks(
    alpha_path: Path,
    dimelo_path: Path,
    hyena_path: Path,
    out_dir: Path,
    max_regions: int,
    mode: str,
) -> None:
    alpha = pd.read_csv(alpha_path, sep="\t")
    dimelo = pd.read_csv(dimelo_path, sep="\t")
    hyena = pd.read_csv(hyena_path, sep="\t")
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = alpha[["chrom", "region_id", "region_name"]].drop_duplicates().head(max_regions)
    for _, selected in regions.iterrows():
        chrom = selected["chrom"]
        region_id = selected["region_id"]
        region_name = selected["region_name"]
        a = alpha[(alpha["chrom"] == chrom) & (alpha["region_id"] == region_id)].copy()
        a["norm"] = normalize(a["A549_H3K4me3_fixed_mean"], mode)

        fig, axis = plt.subplots(figsize=(13, 5))
        x = (a["bin_start"] + a["bin_end"]) / 2
        axis.plot(x, a["norm"], color="black", lw=1.5, label="AlphaGenome A549 H3K4me3")
        for sample, dimelo_color, hyena_color in (
            ("merged_c1", "tab:blue", "tab:purple"),
            ("merged_e5b", "tab:orange", "deeppink"),
        ):
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
            d["norm"] = normalize(d["mean_signal"], mode)
            h["norm"] = normalize(h["mean_signal"], mode)
            axis.plot(
                (d["bin_start"] + d["bin_end"]) / 2,
                d["norm"],
                color=dimelo_color,
                linestyle="-",
                alpha=0.8,
                label=f"DiMeLo {sample}",
            )
            axis.plot(
                (h["bin_start"] + h["bin_end"]) / 2,
                h["norm"],
                color=hyena_color,
                linestyle="--",
                alpha=0.9,
                label=f"HyenaDNA {sample}",
            )
        axis.set_xlabel(f"{chrom} coordinate (hg38)")
        axis.set_ylabel(f"{mode} normalized signal")
        if mode == "minmax":
            axis.set_ylim(-0.05, 1.05)
        axis.legend(frameon=False, ncol=2, fontsize=9)
        fig.suptitle(f"{chrom} region {region_id}: same-scale normalized tracks\n{region_name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{chrom}_region{region_id}_{mode}_normalized.png", dpi=180)
        plt.close(fig)
    print(f"Wrote normalized plots to {out_dir}")


def coarsen_alpha(alpha: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    frame = alpha.copy()
    frame["coarse_start"] = frame["region_start"] + (
        (frame["bin_start"] - frame["region_start"]) // bin_size
    ) * bin_size
    frame["coarse_end"] = frame["coarse_start"] + bin_size
    expected = bin_size // 128
    grouped = frame.groupby(
        ["chrom", "region_id", "region_name", "region_start", "region_end", "coarse_start", "coarse_end"],
        as_index=False,
    )
    rows = grouped.agg(
        n_subbins=("bin_start", "count"),
        A549_H3K4me3_fixed_mean=("A549_H3K4me3_fixed_mean", "mean"),
    )
    rows = rows[(rows["n_subbins"] == expected) & (rows["coarse_end"] <= rows["region_end"])].copy()
    rows = rows.rename(columns={"coarse_start": "bin_start", "coarse_end": "bin_end"})
    return rows.drop(columns=["n_subbins"])


def coarsen_track(track: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    frame = track.copy()
    frame["coarse_start"] = frame["region_start"] + (
        (frame["bin_start"] - frame["region_start"]) // bin_size
    ) * bin_size
    frame["coarse_end"] = frame["coarse_start"] + bin_size
    expected = bin_size // 128
    frame["observed_positions_filled"] = frame["observed_positions"].fillna(0).astype(float)
    frame["weighted_signal"] = frame["mean_signal"].fillna(0).astype(float) * frame[
        "observed_positions_filled"
    ]
    frame["weighted_positive_fraction"] = frame["positive_fraction_0_5"].fillna(0).astype(float) * frame[
        "observed_positions_filled"
    ]
    group_cols = [
        "split",
        "chrom",
        "region_id",
        "region_name",
        "region_start",
        "region_end",
        "coarse_start",
        "coarse_end",
        "sample",
    ]
    rows = (
        frame.groupby(group_cols, as_index=False)
        .agg(
            n_subbins=("bin_start", "count"),
            weighted_signal=("weighted_signal", "sum"),
            weighted_positive_fraction=("weighted_positive_fraction", "sum"),
            observed_positions=("observed_positions_filled", "sum"),
            unique_reads=("unique_reads", "max"),
        )
    )
    rows["mean_signal"] = np.where(
        rows["observed_positions"] > 0,
        rows["weighted_signal"] / rows["observed_positions"],
        np.nan,
    )
    rows["positive_fraction_0_5"] = np.where(
        rows["observed_positions"] > 0,
        rows["weighted_positive_fraction"] / rows["observed_positions"],
        np.nan,
    )
    rows = rows[(rows["n_subbins"] == expected) & (rows["coarse_end"] <= rows["region_end"])].copy()
    rows = rows.rename(columns={"coarse_start": "bin_start", "coarse_end": "bin_end"})
    return rows.drop(columns=["n_subbins", "weighted_signal", "weighted_positive_fraction"])


def bin_size_sensitivity(
    alpha_path: Path,
    dimelo_test_path: Path,
    hyena_path: Path,
    dimelo_val_path: Path,
    out_path: Path,
    bin_sizes: list[int],
) -> None:
    alpha_128 = pd.read_csv(alpha_path, sep="\t")
    dimelo_test_128 = pd.read_csv(dimelo_test_path, sep="\t")
    dimelo_val_128 = pd.read_csv(dimelo_val_path, sep="\t")
    hyena_128 = pd.read_csv(hyena_path, sep="\t")
    results = []
    key = ["chrom", "region_id", "region_start", "region_end", "bin_start", "bin_end"]
    for bin_size in bin_sizes:
        if bin_size % 128 != 0:
            raise SystemExit(f"Bin size must be a multiple of 128 for AlphaGenome output: {bin_size}")
        alpha = coarsen_alpha(alpha_128, bin_size) if bin_size != 128 else alpha_128
        dimelo_test = coarsen_track(dimelo_test_128, bin_size) if bin_size != 128 else dimelo_test_128
        dimelo_val = coarsen_track(dimelo_val_128, bin_size) if bin_size != 128 else dimelo_val_128
        hyena = coarsen_track(hyena_128, bin_size) if bin_size != 128 else hyena_128

        val_pooled = dimelo_val[
            (dimelo_val["sample"] == "pooled")
            & dimelo_val["mean_signal"].notna()
            & (dimelo_val["observed_positions"] >= 1)
            & (dimelo_val["unique_reads"] >= 1)
        ]
        threshold = float(val_pooled["mean_signal"].quantile(0.90))

        for sample in ("merged_c1", "merged_e5b", "pooled"):
            target = dimelo_test[
                (dimelo_test["sample"] == sample)
                & dimelo_test["mean_signal"].notna()
                & (dimelo_test["observed_positions"] >= 1)
                & (dimelo_test["unique_reads"] >= 1)
            ]
            alpha_data = target.merge(alpha, on=key, how="inner", validate="one_to_one")
            hyena_sample = hyena[hyena["sample"] == sample][key + ["mean_signal"]].rename(
                columns={"mean_signal": "HyenaDNA"}
            )
            hyena_data = target.merge(hyena_sample, on=key, how="inner", validate="one_to_one")
            for model, data, pred_col in (
                ("AlphaGenome", alpha_data, "A549_H3K4me3_fixed_mean"),
                ("HyenaDNA", hyena_data, "HyenaDNA"),
            ):
                row = {
                    "bin_size": bin_size,
                    "model": model,
                    "experimental_sample": sample,
                    "threshold_from_validation_pooled_top10": threshold,
                    "number_of_regions": int(data[["chrom", "region_id"]].drop_duplicates().shape[0]),
                }
                row.update(metrics(data["mean_signal"].to_numpy(float), data[pred_col].to_numpy(float), threshold))
                results.append(row)
    pd.DataFrame(results).to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promoter follow-up tables, normalized plots, and bin-size sensitivity.")
    parser.add_argument("--alpha", type=Path, default=Path("outputs/alphagenome_test_128bp.tsv"))
    parser.add_argument("--dimelo-test", type=Path, default=Path("outputs/dimelo_test_128bp.tsv"))
    parser.add_argument("--dimelo-val", type=Path, default=Path("outputs/dimelo_val_128bp.tsv"))
    parser.add_argument("--hyena", type=Path, default=Path("outputs/hyenadna_test_128bp.tsv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/benchmark.summary.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/promoter_followup"))
    parser.add_argument("--normalization", choices=("minmax", "zscore"), default="minmax")
    parser.add_argument("--max-regions", type=int, default=12)
    parser.add_argument("--bin-sizes", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_promoter_metrics(args.summary, args.out_dir / "alphagenome_vs_hyenadna_primary_metrics.tsv")
    plot_normalized_tracks(
        args.alpha,
        args.dimelo_test,
        args.hyena,
        args.out_dir / f"normalized_{args.normalization}_plots",
        args.max_regions,
        args.normalization,
    )
    bin_size_sensitivity(
        args.alpha,
        args.dimelo_test,
        args.hyena,
        args.dimelo_val,
        args.out_dir / "bin_size_sensitivity.tsv",
        args.bin_sizes,
    )


if __name__ == "__main__":
    main()
