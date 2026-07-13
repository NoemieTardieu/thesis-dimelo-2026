#!/usr/bin/env python3
"""Combine paired-read interpretability runs across chromosomes/loci."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paired_read_interpretability_analysis import (
    Config,
    pearson_spearman,
    save_global_plots,
    save_locus_associations,
    select_case_pairs,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", required=True, help="Per-run .all_pairs.tsv files.")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--top-n-per-case", type=int, default=5)
    parser.add_argument("--flag-substitution-distance", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in args.pairs:
        df = pd.read_csv(path, sep="\t")
        df["source_pairs_tsv"] = path
        frames.append(df)
    if not frames:
        raise SystemExit("No pair tables supplied.")
    pair_df = pd.concat(frames, ignore_index=True)
    pair_df = pair_df.replace([np.inf, -np.inf], np.nan)
    if "flag_high_pairwise_substitution_distance" not in pair_df:
        pair_df["flag_high_pairwise_substitution_distance"] = (
            pair_df["substitution_distance"].astype(float) > args.flag_substitution_distance
        )

    config = Config(
        datasets=[],
        bam=Path("combined"),
        sample="combined",
        chrom="combined",
        locus_variance=Path("combined"),
        out_prefix=out_prefix,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        top_n_per_case=args.top_n_per_case,
        flag_substitution_distance=args.flag_substitution_distance,
    )
    selected_df = select_case_pairs(pair_df, config)

    all_path = out_prefix.with_suffix(".all_pairs.tsv")
    selected_path = out_prefix.with_suffix(".selected_pairs.tsv")
    pair_df.to_csv(all_path, sep="\t", index=False)
    selected_df.to_csv(selected_path, sep="\t", index=False)
    plots = save_global_plots(pair_df, selected_df, config)
    locus_assoc = save_locus_associations(pair_df, config)
    added = {"status": "combined_descriptive_only", "reason": "Detailed grouped regression is run in the per-analysis script."}
    report = write_report(pair_df, selected_df, plots, added, config)

    read_set = set(pair_df["read_id_1"]).union(set(pair_df["read_id_2"])) if len(pair_df) else set()
    summary = {
        "n_pair_tables": len(args.pairs),
        "n_pairs": int(len(pair_df)),
        "n_unique_reads": int(len(read_set)),
        "n_loci": int(pair_df["locus_id"].nunique()) if len(pair_df) else 0,
        "chromosomes": sorted(pair_df["chrom"].dropna().unique().tolist()) if "chrom" in pair_df else [],
        "selected_cases": selected_df["case"].value_counts().to_dict() if len(selected_df) else {},
        "correlations": {
            "dna_distance_vs_observed_6ma_mae": pearson_spearman(pair_df["dna_distance"], pair_df["observed_6ma_mae"]),
            "observed_5mc_mae_vs_observed_6ma_mae": pearson_spearman(
                pair_df["observed_5mc_mae"], pair_df["observed_6ma_mae"]
            ),
            "observed_vs_predicted_6ma_mae": pearson_spearman(
                pair_df["observed_6ma_mae"], pair_df["predicted_6ma_mae"]
            ),
        },
        "outputs": {
            "all_pairs": str(all_path),
            "selected_pairs": str(selected_path),
            "within_locus_associations": locus_assoc,
            "report": report,
            **plots,
        },
    }
    summary_path = out_prefix.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
