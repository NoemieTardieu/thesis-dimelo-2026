#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRACKS = [
    ("external_robust01", "ENCSR203XPU", "tab:green"),
    ("alphagenome_robust01", "AlphaGenome", "black"),
    ("hyena_robust01", "HyenaDNA pooled", "tab:purple"),
    ("dimelo_robust01", "DiMeLo pooled", "tab:orange"),
]


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a high-signal peak agreement example without rerunning the benchmark.")
    parser.add_argument("--canonical", type=Path, default=Path("outputs/population_track_benchmark_ENCSR203XPU_200bp_final/canonical_normalized_bins.tsv.gz"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/population_track_benchmark_ENCSR203XPU_200bp_final/figures/peak_agreement"))
    parser.add_argument("--window-bins", type=int, default=10)
    parser.add_argument("--min-dimelo-coverage", type=int, default=5)
    args = parser.parse_args()

    data = pd.read_csv(args.canonical, sep="\t")
    data = data[data["dimelo_coverage"].fillna(0) >= args.min_dimelo_coverage].copy()
    rows = []
    for (chrom, region_id), region in data.groupby(["chrom", "region_id"], sort=False):
        region = region.sort_values("start").reset_index(drop=True)
        for offset in range(0, max(1, len(region) - args.window_bins + 1), max(1, args.window_bins // 2)):
            window = region.iloc[offset : offset + args.window_bins].copy()
            if len(window) < args.window_bins:
                continue
            track_matrix = window[[track for track, _, _ in TRACKS]].to_numpy(float)
            if not np.isfinite(track_matrix).all():
                continue
            peak_score = float(np.nanmean(window[["external_robust01", "alphagenome_robust01"]].to_numpy(float)))
            all_four_mean = float(np.nanmean(track_matrix))
            disagreement = float(np.nanmean(np.nanstd(track_matrix, axis=1)))
            rows.append(
                {
                    "chrom": chrom,
                    "region_id": region_id,
                    "region_name": window["region_name"].iloc[0],
                    "start": int(window["start"].min()),
                    "end": int(window["end"].max()),
                    "peak_score_external_alpha": peak_score,
                    "all_four_mean_signal": all_four_mean,
                    "mean_all_four_disagreement": disagreement,
                    "selection_score": peak_score + all_four_mean - disagreement,
                }
            )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise SystemExit("No high-signal candidate windows found.")
    selected = candidates.sort_values("selection_score", ascending=False).head(1).iloc[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates.sort_values("selection_score", ascending=False).head(25).to_csv(
        args.out_dir / "peak_agreement_candidates.tsv", sep="\t", index=False
    )
    selected.to_frame().T.to_csv(args.out_dir / "selected_peak_agreement_locus.tsv", sep="\t", index=False)

    subset = data[
        (data["chrom"] == selected["chrom"])
        & (data["start"] >= int(selected["start"]))
        & (data["end"] <= int(selected["end"]))
    ].sort_values("start")
    x = (subset["start"] + subset["end"]) / 2
    fig, axes = plt.subplots(5, 1, figsize=(13, 8), sharex=True)
    for ax, (col, label, color) in zip(axes[:4], TRACKS):
        ax.plot(x, subset[col], color=color, lw=1.3)
        ax.set_ylabel(label)
        ax.set_ylim(-0.05, 1.05)
    axes[4].bar(x, subset["dimelo_coverage"], width=float((subset["end"] - subset["start"]).median()) * 0.9, color="gray")
    axes[4].set_ylabel("DiMeLo\ncoverage")
    axes[4].set_xlabel(f"{selected['chrom']} coordinate (hg38)")
    fig.suptitle(
        "High-signal peak agreement example\n"
        f"{selected['chrom']}:{int(selected['start'])}-{int(selected['end'])} "
        f"{selected['region_name']}"
    )
    fig.tight_layout()
    save_figure(fig, args.out_dir / "high_signal_peak_agreement")
    print(f"Wrote peak agreement example to {args.out_dir}")


if __name__ == "__main__":
    main()
