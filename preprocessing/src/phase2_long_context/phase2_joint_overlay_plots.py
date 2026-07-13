#!/usr/bin/env python3
"""Phase 2 continuous joint-distribution overlays from cached 1 kb features.

This script reuses the per-bin feature caches produced by
windowed_oe_enrichment_heatmap.py and generates:

1. Option 2 style overlay:
   Reg normalized to [0,1] vs methylation fraction
2. Option 3 style overlay:
   log1p(Reg density) vs methylation fraction

Each point corresponds to one 1 kb genomic bin that passed the Phase 2
coverage filter.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required for phase2_joint_overlay_plots.py") from exc


MARK_COLORS: Dict[str, str] = {
    "h3k27ac": "#d73027",
    "h3k27me3": "#3182bd",
    "h3k4me3": "#33a02c",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 overlay contour plots from cached 1 kb features.")
    p.add_argument(
        "--cache-dir",
        default="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase2_long_context/windowed_enrichment_1kb_combined",
        help="Directory containing feature_cache_<mark>_*.npz files.",
    )
    p.add_argument(
        "--out-dir",
        default="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase2_long_context/windowed_joint_plots_1kb",
        help="Directory for new overlay joint plots.",
    )
    p.add_argument(
        "--marks",
        default="h3k27ac,h3k27me3,h3k4me3",
        help="Comma-separated marks to plot.",
    )
    p.add_argument(
        "--sample-per-mark",
        type=int,
        default=60000,
        help="Max bins per mark to scatter as background points.",
    )
    p.add_argument(
        "--hist-bins-x",
        type=int,
        default=120,
        help="Number of x bins for contour density estimation.",
    )
    p.add_argument(
        "--hist-bins-y",
        type=int,
        default=120,
        help="Number of y bins for contour density estimation.",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma for contour density.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for scatter subsampling.",
    )
    return p.parse_args()


def cache_path(cache_dir: str, mark: str) -> str:
    return os.path.join(
        cache_dir,
        f"feature_cache_{mark}_bin1000_a_mod_per_kb_c_meth_frac_cov30.npz",
    )


def load_mark(cache_dir: str, mark: str) -> Dict[str, np.ndarray]:
    path = cache_path(cache_dir, mark)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cache for {mark}: {path}")
    arr = np.load(path, allow_pickle=False)
    mask = arr["cov_ok"].astype(bool)
    reg = arr["reg"][mask].astype(np.float64)
    meth = arr["meth"][mask].astype(np.float64)
    valid = np.isfinite(reg) & np.isfinite(meth)
    reg = reg[valid]
    meth = meth[valid]
    return {"reg": reg, "meth": meth}


def normalize_zero_one(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float64)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float64)
    return (x - xmin) / (xmax - xmin)


def scatter_sample(x: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if x.size <= n:
        return x, y
    idx = rng.choice(x.size, size=n, replace=False)
    return x[idx], y[idx]


def smoothed_density(
    x: np.ndarray,
    y: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    bins_x: int,
    bins_y: int,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=[bins_x, bins_y],
        range=[x_range, y_range],
    )
    hist = gaussian_filter(hist, sigma=sigma)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return hist.T, x_centers, y_centers


def contour_levels(z: np.ndarray) -> np.ndarray:
    positive = z[z > 0]
    if positive.size == 0:
        return np.array([])
    q = np.quantile(positive, [0.55, 0.72, 0.84, 0.92, 0.97])
    q = np.unique(q[q > 0])
    return q


def plot_overlay(
    path: str,
    title: str,
    mark_to_xy: Dict[str, Tuple[np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    sample_per_mark: int,
    bins_x: int,
    bins_y: int,
    sigma: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    fig = plt.figure(figsize=(9.0, 7.2))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4.8, 1.3),
        height_ratios=(1.2, 4.8),
        hspace=0.05,
        wspace=0.05,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    all_x = np.concatenate([xy[0] for xy in mark_to_xy.values() if xy[0].size > 0])
    all_y = np.concatenate([xy[1] for xy in mark_to_xy.values() if xy[1].size > 0])
    x_range = (float(np.min(all_x)), float(np.max(all_x)))
    y_range = (0.0, 1.0)

    line_handles = []
    for mark in ["h3k27ac", "h3k27me3", "h3k4me3"]:
        if mark not in mark_to_xy:
            continue
        x, y = mark_to_xy[mark]
        color = MARK_COLORS[mark]

        xs, ys = scatter_sample(x, y, sample_per_mark, rng)
        ax_main.scatter(xs, ys, s=4, alpha=0.12, color=color, edgecolors="none")

        z, xc, yc = smoothed_density(
            x=x,
            y=y,
            x_range=x_range,
            y_range=y_range,
            bins_x=bins_x,
            bins_y=bins_y,
            sigma=sigma,
        )
        levels = contour_levels(z)
        if levels.size > 0:
            ax_main.contour(xc, yc, z, levels=levels, colors=color, linewidths=1.5)

        ax_top.hist(
            x,
            bins=70,
            range=x_range,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            alpha=0.95,
        )
        ax_right.hist(
            y,
            bins=50,
            range=y_range,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            alpha=0.95,
            orientation="horizontal",
        )
        handle = plt.Line2D([0], [0], color=color, lw=2, label=mark)
        line_handles.append(handle)

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_xlim(*x_range)
    ax_main.set_ylim(*y_range)
    ax_main.grid(alpha=0.15, linewidth=0.5)

    ax_top.set_title(title, fontsize=14)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("Density")
    ax_top.grid(alpha=0.12, linewidth=0.5)

    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlabel("Density")
    ax_right.grid(alpha=0.12, linewidth=0.5)

    fig.legend(
        handles=line_handles,
        title="Mark",
        loc="upper right",
        bbox_to_anchor=(0.97, 0.95),
        frameon=True,
    )

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_notes(path: str, option2_path: str, option3_path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Phase 2 Joint Overlay Plots\n\n")
        f.write("- Input source: cached 1 kb Phase 2 feature arrays from `windowed_enrichment_1kb_combined`\n")
        f.write("- Each point corresponds to one 1 kb genomic bin passing the Phase 2 C-coverage filter\n")
        f.write("- Reg feature: `a_mod_per_kb`\n")
        f.write("- M feature: `c_meth_frac`\n")
        f.write("- Option 2 overlay: Reg normalized to [0,1] vs methylation fraction\n")
        f.write("- Option 3 overlay: `log1p(a_mod_per_kb)` vs methylation fraction\n")
        f.write(f"- Output: {option2_path}\n")
        f.write(f"- Output: {option3_path}\n")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    marks = [x.strip() for x in args.marks.split(",") if x.strip()]
    raw_by_mark: Dict[str, Dict[str, np.ndarray]] = {
        mark: load_mark(args.cache_dir, mark) for mark in marks
    }

    option2_xy: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    option3_xy: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for mark, payload in raw_by_mark.items():
        reg = payload["reg"]
        meth = payload["meth"]
        option2_xy[mark] = (normalize_zero_one(reg), meth)
        option3_xy[mark] = (np.log1p(reg), meth)

    option2_path = os.path.join(args.out_dir, "phase2_option2_overlay_regNorm_vs_mfrac_by_mark.svg")
    option3_path = os.path.join(args.out_dir, "phase2_option3_overlay_log1pReg_vs_mfrac_by_mark.svg")

    plot_overlay(
        path=option2_path,
        title="Phase 2 Option 2: Reg normalized vs methylation fraction (1 kb bins)",
        mark_to_xy=option2_xy,
        xlabel="Reg normalized [0,1]",
        ylabel="Methylation fraction [0,1]",
        sample_per_mark=args.sample_per_mark,
        bins_x=args.hist_bins_x,
        bins_y=args.hist_bins_y,
        sigma=args.smooth_sigma,
        seed=args.seed,
    )

    plot_overlay(
        path=option3_path,
        title="Phase 2 Option 3: log1p(Reg density) vs methylation fraction (1 kb bins)",
        mark_to_xy=option3_xy,
        xlabel="log1p(A_mod_density_per_kb)",
        ylabel="Methylation fraction [0,1]",
        sample_per_mark=args.sample_per_mark,
        bins_x=args.hist_bins_x,
        bins_y=args.hist_bins_y,
        sigma=args.smooth_sigma,
        seed=args.seed,
    )

    write_notes(
        path=os.path.join(args.out_dir, "phase2_joint_overlay_notes.md"),
        option2_path=option2_path,
        option3_path=option3_path,
    )

    print(f"Wrote Phase 2 overlay plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
