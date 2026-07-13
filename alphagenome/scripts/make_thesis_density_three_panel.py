#!/usr/bin/env python3
"""Create a thesis-ready three-panel density figure.

Panels:
A. external track vs AlphaGenome
B. HyenaDNA vs DiMeLo
C. AlphaGenome vs DiMeLo

The plot uses globally robust-normalized values for visualization, while the
reported Pearson/Spearman/n values are read from the benchmark metrics table.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PANELS = [
    ("A", "T-A", "external", "alphagenome"),
    ("B", "H-D", "hyena", "dimelo"),
    ("C", "A-D", "alphagenome", "dimelo"),
]

LABELS = {
    "external": "External H3K4me3\nrobust normalized",
    "alphagenome": "AlphaGenome\nrobust normalized",
    "hyena": "HyenaDNA\nrobust normalized",
    "dimelo": "DiMeLo\nrobust normalized",
}


def read_table(path: Path) -> pd.DataFrame:
    """Read a TSV or TSV.GZ table."""

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return pd.read_csv(handle, sep="\t")


def metric_row(metrics: pd.DataFrame, comparison: str, intersection: str) -> pd.Series:
    """Return one pairwise metric row."""

    subset = metrics[
        (metrics["comparison"] == comparison)
        & (metrics["intersection"] == intersection)
        & (metrics["scope"] == "all")
    ]
    if subset.empty:
        raise SystemExit(f"No metrics row for {comparison} / {intersection}")
    return subset.iloc[0]


def valid_subset(data: pd.DataFrame, left: str, right: str, min_dimelo_coverage: int) -> pd.DataFrame:
    """Subset to bins valid for the selected pair."""

    valid = data[f"{left}_valid"] & data[f"{right}_valid"]
    if "dimelo" in {left, right}:
        valid &= data["dimelo_coverage"].fillna(0) >= min_dimelo_coverage
    subset = data.loc[valid].copy()
    cols = [f"{left}_robust01", f"{right}_robust01"]
    return subset[np.isfinite(subset[cols[0]]) & np.isfinite(subset[cols[1]])]


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    metrics: pd.Series,
    panel_label: str,
    comparison: str,
    left: str,
    right: str,
) -> None:
    """Draw one hexbin density panel."""

    x = data[f"{left}_robust01"].to_numpy(float)
    y = data[f"{right}_robust01"].to_numpy(float)

    if x.size:
        ax.hexbin(x, y, gridsize=55, mincnt=1, cmap="viridis", bins="log")
        if np.std(x) > 0 and np.std(y) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(0, 1, 100)
            ax.plot(xx, slope * xx + intercept, color="black", lw=1)
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle=":")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(LABELS[left], fontsize=9)
    ax.set_ylabel(LABELS[right], fontsize=9)
    ax.tick_params(labelsize=8)

    ax.text(
        -0.13,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.text(
        0.04,
        0.96,
        f"{comparison}\n"
        f"r = {metrics['pearson']:.3f}\n"
        f"rho = {metrics['spearman']:.3f}\n"
        f"n = {int(metrics['n_bins']):,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("outputs/population_track_benchmark_GSM2421502_1kb"),
    )
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path("outputs/population_track_benchmark_GSM2421502_1kb/figures/thesis_density_three_panel"),
    )
    parser.add_argument(
        "--intersection",
        default="pair_specific",
        choices=["pair_specific", "common_four_track"],
    )
    parser.add_argument("--min-dimelo-coverage", type=int, default=5)
    args = parser.parse_args()

    data = read_table(args.benchmark_dir / "canonical_normalized_bins.tsv.gz")
    metrics = read_table(args.benchmark_dir / "pairwise_metrics.tsv.gz")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), constrained_layout=True)
    for ax, (panel_label, comparison, left, right) in zip(axes, PANELS):
        subset = valid_subset(data, left, right, args.min_dimelo_coverage)
        row = metric_row(metrics, comparison, args.intersection)
        draw_panel(ax, subset, row, panel_label, comparison, left, right)

    args.out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_base.with_suffix('.png')}")
    print(f"Wrote {args.out_base.with_suffix('.svg')}")
    print(f"Wrote {args.out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
