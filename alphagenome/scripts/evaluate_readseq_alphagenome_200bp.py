#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_alphagenome_vs_dimelo import KEYS, add_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reference AlphaGenome, read-sequence AlphaGenome, and HyenaDNA on 200 bp bins."
    )
    parser.add_argument("--dimelo", type=Path, default=Path("outputs/benchmark_200bp/dimelo_test_200bp.tsv"))
    parser.add_argument("--dimelo-val", type=Path, default=Path("outputs/benchmark_200bp/dimelo_val_200bp.tsv"))
    parser.add_argument("--alphagenome-reference", type=Path, default=Path("outputs/benchmark_200bp/alphagenome_test_200bp.tsv"))
    parser.add_argument("--alphagenome-readseq", type=Path, default=Path("outputs/alphagenome_readseq_200bp.tsv"))
    parser.add_argument("--hyenadna", type=Path, default=Path("outputs/benchmark_200bp/hyenadna_test_200bp.tsv"))
    parser.add_argument("--out-prefix", type=Path, default=Path("outputs/alphagenome_readseq_200bp_benchmark"))
    parser.add_argument("--minimum-unique-reads", type=int, default=2)
    parser.add_argument("--minimum-observed-positions", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dimelo = pd.read_csv(args.dimelo, sep="\t")
    dimelo_val = pd.read_csv(args.dimelo_val, sep="\t")
    alpha_ref = pd.read_csv(args.alphagenome_reference, sep="\t")
    alpha_read = pd.read_csv(args.alphagenome_readseq, sep="\t")
    hyena = pd.read_csv(args.hyenadna, sep="\t")
    threshold_data = dimelo_val[
        (dimelo_val["sample"] == "pooled")
        & (dimelo_val["unique_reads"] >= args.minimum_unique_reads)
        & (dimelo_val["observed_positions"] >= args.minimum_observed_positions)
        & dimelo_val["mean_signal"].notna()
    ]
    if threshold_data.empty:
        raise SystemExit("No validation bins available for threshold derivation.")
    threshold = float(threshold_data["mean_signal"].quantile(0.90))

    results: list[dict] = []
    for sample in ("merged_c1", "merged_e5b", "pooled"):
        target = dimelo[
            (dimelo["sample"] == sample)
            & (dimelo["unique_reads"] >= args.minimum_unique_reads)
            & (dimelo["observed_positions"] >= args.minimum_observed_positions)
            & dimelo["mean_signal"].notna()
        ]
        ref_data = target.merge(alpha_ref, on=KEYS, how="inner", validate="one_to_one")
        read_sample = alpha_read[alpha_read["sample"] == sample][KEYS + ["mean_signal"]].rename(
            columns={"mean_signal": "AlphaGenome_readseq"}
        )
        read_data = target.merge(read_sample, on=KEYS, how="inner", validate="one_to_one")
        read_data = read_data[read_data["AlphaGenome_readseq"].notna()]
        hyena_sample = hyena[hyena["sample"] == sample][KEYS + ["mean_signal"]].rename(
            columns={"mean_signal": "HyenaDNA"}
        )
        hyena_data = target.merge(hyena_sample, on=KEYS, how="inner", validate="one_to_one")
        hyena_data = hyena_data[hyena_data["HyenaDNA"].notna()]

        add_result(
            results,
            ref_data,
            "AlphaGenome_reference",
            "A549_H3K4me3_fixed_mean",
            sample,
            "pooled",
            "top_10_percent_primary_200bp",
            threshold,
            args.bootstrap_replicates,
            args.seed,
        )
        add_result(
            results,
            read_data,
            "AlphaGenome_readseq",
            "AlphaGenome_readseq",
            sample,
            "pooled",
            "top_10_percent_primary_200bp",
            threshold,
            args.bootstrap_replicates,
            args.seed,
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
            args.bootstrap_replicates,
            args.seed,
        )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(f"{args.out_prefix}.summary.tsv", sep="\t", index=False)
    with open(f"{args.out_prefix}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": threshold,
                "minimum_unique_reads": args.minimum_unique_reads,
                "minimum_observed_positions": args.minimum_observed_positions,
                "bootstrap_replicates": args.bootstrap_replicates,
                "results": results,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"Wrote {args.out_prefix}.summary.tsv")


if __name__ == "__main__":
    main()
