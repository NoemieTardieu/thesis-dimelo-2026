#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


KEYS = ["chrom", "region_id", "region_start", "region_end", "bin_start", "bin_end"]


def metrics(target: np.ndarray, prediction: np.ndarray, threshold: float) -> dict:
    result = {
        "number_of_bins": int(target.size),
        "pearson": None,
        "spearman": None,
        "auroc": None,
        "auprc": None,
        "positive_fraction": None,
        "auprc_enrichment": None,
    }
    if target.size == 0:
        return result
    labels = target >= threshold
    result["positive_fraction"] = float(labels.mean())
    if target.size >= 2 and np.std(target) > 0 and np.std(prediction) > 0:
        result["pearson"] = float(pearsonr(target, prediction).statistic)
        result["spearman"] = float(spearmanr(target, prediction).statistic)
    if np.unique(labels).size == 2:
        result["auroc"] = float(roc_auc_score(labels, prediction))
        result["auprc"] = float(average_precision_score(labels, prediction))
        if result["positive_fraction"]:
            result["auprc_enrichment"] = result["auprc"] / result["positive_fraction"]
    return result


def bootstrap_ci(
    data: pd.DataFrame,
    target_col: str,
    prediction_col: str,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float | None, float | None]]:
    if replicates <= 0:
        return {name: (None, None) for name in ("pearson", "spearman", "auroc", "auprc")}
    region_groups = [group for _, group in data.groupby(["chrom", "region_id"], sort=False)]
    if len(region_groups) < 2:
        return {name: (None, None) for name in ("pearson", "spearman", "auroc", "auprc")}
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("pearson", "spearman", "auroc", "auprc")}
    for _ in range(replicates):
        sampled = [region_groups[i] for i in rng.integers(0, len(region_groups), len(region_groups))]
        frame = pd.concat(sampled, ignore_index=True)
        row = metrics(
            frame[target_col].to_numpy(float),
            frame[prediction_col].to_numpy(float),
            threshold,
        )
        for name in values:
            if row[name] is not None:
                values[name].append(row[name])
    return {
        name: (
            float(np.quantile(samples, 0.025)) if samples else None,
            float(np.quantile(samples, 0.975)) if samples else None,
        )
        for name, samples in values.items()
    }


def add_result(
    output: list[dict],
    data: pd.DataFrame,
    model: str,
    track: str,
    sample: str,
    scope: str,
    threshold_name: str,
    threshold: float,
    replicates: int,
    seed: int,
) -> None:
    row = {
        "scope": scope,
        "model": model,
        "track": track,
        "experimental_sample": sample,
        "threshold_name": threshold_name,
        "threshold": threshold,
        "number_of_regions": int(data[["chrom", "region_id"]].drop_duplicates().shape[0]),
    }
    row.update(metrics(data["mean_signal"].to_numpy(float), data[track].to_numpy(float), threshold))
    ci = bootstrap_ci(data, "mean_signal", track, threshold, replicates, seed)
    for metric_name, (low, high) in ci.items():
        row[f"{metric_name}_ci_low"] = low
        row[f"{metric_name}_ci_high"] = high
    output.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AlphaGenome and HyenaDNA on identical 128 bp bins.")
    parser.add_argument("--alphagenome", type=Path, required=True)
    parser.add_argument("--dimelo", type=Path, required=True)
    parser.add_argument("--hyenadna", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    with open(args.thresholds, "r", encoding="utf-8") as handle:
        threshold_config = json.load(handle)
    min_reads = int(threshold_config["minimum_unique_reads"])
    min_positions = int(threshold_config["minimum_observed_positions"])
    thresholds = threshold_config["thresholds"]

    alpha = pd.read_csv(args.alphagenome, sep="\t")
    dimelo = pd.read_csv(args.dimelo, sep="\t")
    hyena = pd.read_csv(args.hyenadna, sep="\t")
    alpha_tracks = [column for column in alpha.columns if column not in KEYS + ["region_name"]]
    if "A549_H3K4me3_fixed_mean" not in alpha_tracks:
        raise SystemExit("AlphaGenome fixed-mean track is missing.")

    results: list[dict] = []
    per_region: list[dict] = []
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
        hyena_data = target.merge(hyena_sample, on=KEYS, how="inner", suffixes=("", "_hyena"))
        hyena_data = hyena_data[hyena_data["HyenaDNA"].notna()]

        for threshold_name, threshold in thresholds.items():
            for scope in ["pooled", *sorted(alpha_data["chrom"].unique())]:
                subset = alpha_data if scope == "pooled" else alpha_data[alpha_data["chrom"] == scope]
                for track in alpha_tracks:
                    add_result(
                        results, subset, "AlphaGenome", track, sample, scope,
                        threshold_name, float(threshold),
                        args.bootstrap_replicates if threshold_name == "top_10_percent_primary" else 0,
                        args.seed,
                    )
                hyena_subset = hyena_data if scope == "pooled" else hyena_data[hyena_data["chrom"] == scope]
                add_result(
                    results, hyena_subset, "HyenaDNA", "HyenaDNA", sample, scope,
                    threshold_name, float(threshold),
                    args.bootstrap_replicates if threshold_name == "top_10_percent_primary" else 0,
                    args.seed,
                )

            for (chrom, region_id), subset in alpha_data.groupby(["chrom", "region_id"]):
                for track in alpha_tracks:
                    row = {
                        "chrom": chrom, "region_id": region_id, "model": "AlphaGenome",
                        "track": track, "experimental_sample": sample,
                        "threshold_name": threshold_name, "threshold": threshold,
                    }
                    row.update(metrics(subset["mean_signal"].to_numpy(float), subset[track].to_numpy(float), threshold))
                    per_region.append(row)
                matching_hyena = hyena_data[
                    (hyena_data["chrom"] == chrom) & (hyena_data["region_id"] == region_id)
                ]
                if not matching_hyena.empty:
                    row = {
                        "chrom": chrom, "region_id": region_id, "model": "HyenaDNA",
                        "track": "HyenaDNA", "experimental_sample": sample,
                        "threshold_name": threshold_name, "threshold": threshold,
                    }
                    row.update(
                        metrics(
                            matching_hyena["mean_signal"].to_numpy(float),
                            matching_hyena["HyenaDNA"].to_numpy(float),
                            threshold,
                        )
                    )
                    per_region.append(row)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(f"{args.out_prefix}.summary.tsv", sep="\t", index=False)
    pd.DataFrame(per_region).to_csv(f"{args.out_prefix}.per_region.tsv", sep="\t", index=False)
    with open(f"{args.out_prefix}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "alphagenome": str(args.alphagenome),
                "dimelo": str(args.dimelo),
                "hyenadna": str(args.hyenadna),
                "thresholds": threshold_config,
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_unit": "genomic_region",
                "results": results,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"Wrote {args.out_prefix}.summary.tsv")


if __name__ == "__main__":
    main()
