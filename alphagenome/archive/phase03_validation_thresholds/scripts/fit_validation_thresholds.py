#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmark_utils import json_dump


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze experimental positive-bin thresholds from validation data.")
    parser.add_argument("--validation-track", type=Path, required=True)
    parser.add_argument("--min-reads", type=int, default=1)
    parser.add_argument("--min-positions", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("outputs/validation_thresholds.json"))
    args = parser.parse_args()

    data = pd.read_csv(args.validation_track, sep="\t")
    covered = data[
        (data["sample"] == "pooled")
        & (data["unique_reads"] >= args.min_reads)
        & (data["observed_positions"] >= args.min_positions)
        & data["mean_signal"].notna()
    ]
    if covered.empty:
        raise SystemExit("No covered pooled validation bins remain.")
    thresholds = {
        "top_5_percent": float(covered["mean_signal"].quantile(0.95)),
        "top_10_percent_primary": float(covered["mean_signal"].quantile(0.90)),
        "top_20_percent": float(covered["mean_signal"].quantile(0.80)),
    }
    json_dump(
        args.out,
        {
            "source": str(args.validation_track),
            "source_split": "val",
            "selection": "sample == pooled",
            "covered_bins": int(len(covered)),
            "minimum_unique_reads": args.min_reads,
            "minimum_observed_positions": args.min_positions,
            "thresholds": thresholds,
        },
    )
    print(thresholds)


if __name__ == "__main__":
    main()
