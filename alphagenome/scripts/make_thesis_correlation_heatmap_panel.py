#!/usr/bin/env python3
"""Create a thesis-ready Pearson/Spearman heatmap panel.

This script regenerates the two correlation heatmaps from the benchmark
metrics table, without subplot titles, and combines them into one figure with
panel labels A and B.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRACK_LABELS = {
    "external": "ENCODE\nH3K4me3",
    "alphagenome": "AlphaGenome",
    "hyena": "HyenaDNA",
    "dimelo": "DiMeLo",
}

TRACK_ORDER = ["external", "alphagenome", "hyena", "dimelo"]


def read_metrics(path: Path, intersection: str) -> pd.DataFrame:
    """Read pairwise metrics and keep one intersection type."""

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        metrics = pd.read_csv(handle, sep="\t")
    metrics = metrics[metrics["intersection"] == intersection].copy()
    if metrics.empty:
        raise SystemExit(f"No rows found with intersection={intersection!r} in {path}")
    return metrics


def correlation_matrix(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Convert pairwise rows into a symmetric 4 x 4 correlation matrix."""

    labels = [TRACK_LABELS[track] for track in TRACK_ORDER]
    matrix = pd.DataFrame(np.eye(len(TRACK_ORDER)), index=labels, columns=labels)
    for row in metrics.itertuples(index=False):
        left = TRACK_LABELS[row.left_track]
        right = TRACK_LABELS[row.right_track]
        value = getattr(row, metric)
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    return matrix


def draw_heatmap(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    """Draw one annotated heatmap on an existing axis."""

    values = matrix.to_numpy(float)
    image = ax.imshow(values, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.tick_params(length=0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "outputs/population_track_benchmark_ENCSR203XPU_200bp_final/"
            "pairwise_metrics.tsv.gz"
        ),
    )
    parser.add_argument(
        "--out-base",
        type=Path,
        default=Path(
            "outputs/population_track_benchmark_ENCSR203XPU_200bp_final/"
            "figures/thesis_correlation_heatmap_panel"
        ),
    )
    parser.add_argument(
        "--intersection",
        default="pair_specific",
        choices=["pair_specific", "common_four_track"],
        help="Which metric rows to use for the heatmaps.",
    )
    args = parser.parse_args()

    metrics = read_metrics(args.metrics, args.intersection)
    pearson = correlation_matrix(metrics, "pearson")
    spearman = correlation_matrix(metrics, "spearman")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), constrained_layout=True)
    image = draw_heatmap(axes[0], pearson)
    draw_heatmap(axes[1], spearman)

    axes[0].text(
        -0.14,
        1.06,
        "A",
        transform=axes[0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    axes[1].text(
        -0.14,
        1.06,
        "B",
        transform=axes[1].transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("correlation coefficient", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

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
