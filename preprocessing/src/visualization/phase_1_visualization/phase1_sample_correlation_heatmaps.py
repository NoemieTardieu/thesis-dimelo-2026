#!/usr/bin/env python3
"""Cross-sample correlation heatmaps for DiMeLo-seq mC and mA signals.

This script uses common 1 kb genomic bins so that cross-sample Pearson
correlation is defined on aligned genomic loci rather than unmatched reads.

Panel A:
    Pearson correlation of local mC levels across samples
Panel B:
    Pearson correlation of local mA levels across samples

Inputs:
- Phase 2 backend manifests (for mC counts)
- Phase 2 cached 1 kb features (for mA per-kb signal)
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1/2 sample comparability correlation heatmaps.")
    p.add_argument(
        "--manifest-glob-root",
        default="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase2_long_context/backends_all_marks",
        help="Root containing backend_<mark>_C/manifest_<mark>_C.tsv",
    )
    p.add_argument(
        "--feature-cache-dir",
        default="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase2_long_context/windowed_enrichment_1kb_combined",
        help="Directory containing feature_cache_<mark>_bin1000_*.npz",
    )
    p.add_argument(
        "--out-dir",
        default="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase_1_visualization/output_feedback",
        help="Output directory for the correlation figure and summary files.",
    )
    p.add_argument(
        "--marks",
        default="h3k27ac,h3k27me3,h3k4me3",
        help="Comma-separated sample/mark names.",
    )
    p.add_argument(
        "--bin-size",
        type=int,
        default=1000,
        help="Common genomic tile size for the comparison.",
    )
    p.add_argument(
        "--normalize-cpm",
        action="store_true",
        help="Normalize each sample vector to counts per million before correlation.",
    )
    return p.parse_args()


def manifest_path(root: str, mark: str) -> str:
    return os.path.join(root, f"backend_{mark}_C", f"manifest_{mark}_C.tsv")


def load_manifest_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(row)
    return rows


def sum_by_bin(arr: np.ndarray, bin_size: int) -> np.ndarray:
    idx = np.arange(0, arr.shape[0], bin_size, dtype=np.int64)
    return np.add.reduceat(arr.astype(np.int64), idx)


def load_mc_signal(mark: str, manifest_root: str, bin_size: int) -> np.ndarray:
    path = manifest_path(manifest_root, mark)
    rows = load_manifest_rows(path)
    chunks: List[np.ndarray] = []
    for row in rows:
        payload = np.load(row["npz_path"], allow_pickle=False)
        meth_counts = payload["meth_counts"].astype(np.int64)
        chunks.append(sum_by_bin(meth_counts, bin_size).astype(np.float64))
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)


def load_ma_signal(mark: str, cache_dir: str) -> np.ndarray:
    path = os.path.join(
        cache_dir,
        f"feature_cache_{mark}_bin1000_a_mod_per_kb_c_meth_frac_cov30.npz",
    )
    payload = np.load(path, allow_pickle=False)
    return payload["reg"].astype(np.float64)


def normalize_cpm(x: np.ndarray) -> np.ndarray:
    s = float(np.sum(x))
    if s <= 0:
        return np.zeros_like(x, dtype=np.float64)
    return x * (1_000_000.0 / s)


def pearson_corr_matrix(signal_by_mark: Dict[str, np.ndarray]) -> np.ndarray:
    marks = list(signal_by_mark.keys())
    n = len(marks)
    mat = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            x = signal_by_mark[marks[i]]
            y = signal_by_mark[marks[j]]
            mask = np.isfinite(x) & np.isfinite(y)
            mask &= ((x > 0) | (y > 0))
            if int(mask.sum()) < 2:
                r = float("nan")
            else:
                r = float(np.corrcoef(x[mask], y[mask])[0, 1])
            mat[i, j] = r
            mat[j, i] = r
    return mat


def cluster_order(corr: np.ndarray) -> np.ndarray:
    if corr.shape[0] <= 2:
        return np.arange(corr.shape[0])
    dist = 1.0 - np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="average")
    return leaves_list(link)


def plot_cluster_panel(
    fig: plt.Figure,
    outer_spec,
    corr: np.ndarray,
    labels: List[str],
    title_letter: str,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    order = cluster_order(corr)
    corr_ord = corr[np.ix_(order, order)]
    labels_ord = [labels[i] for i in order]

    gs = outer_spec.subgridspec(
        2,
        3,
        width_ratios=[1.2, 4.5, 0.28],
        height_ratios=[1.2, 4.5],
        wspace=0.18,
        hspace=0.04,
    )
    ax_empty = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_hm = fig.add_subplot(gs[1, 1])
    ax_cb = fig.add_subplot(gs[1, 2])

    ax_empty.axis("off")

    if corr.shape[0] > 2:
        dist = 1.0 - np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="average")
        dendrogram(
            link,
            ax=ax_top,
            no_labels=True,
            color_threshold=None,
            link_color_func=lambda _: "black",
        )
        dendrogram(
            link,
            ax=ax_left,
            orientation="left",
            no_labels=True,
            color_threshold=None,
            link_color_func=lambda _: "black",
        )
    else:
        ax_top.axis("off")
        ax_left.axis("off")

    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for spine in ax_left.spines.values():
        spine.set_visible(False)

    im = ax_hm.imshow(corr_ord, cmap="OrRd", vmin=vmin, vmax=vmax, aspect="equal")
    ax_hm.set_xticks(np.arange(len(labels_ord)))
    ax_hm.set_yticks(np.arange(len(labels_ord)))
    ax_hm.set_xticklabels(labels_ord, rotation=90)
    ax_hm.set_yticklabels([])
    ax_hm.tick_params(length=0)
    ax_hm.yaxis.tick_left()
    ax_hm.tick_params(axis="y", labelleft=False, labelright=False)
    for spine in ax_hm.spines.values():
        spine.set_visible(False)

    # Draw vertical row labels manually so they sit close to the matrix
    # without overlapping the left dendrogram lines.
    for i, label in enumerate(labels_ord):
        ax_hm.text(
            -0.72,
            i,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
            clip_on=False,
        )

    cb = fig.colorbar(im, cax=ax_cb)
    cb.outline.set_visible(False)

    ax_hm.text(
        -0.28,
        1.06,
        title_letter,
        transform=ax_hm.transAxes,
        fontsize=20,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax_top.set_title(title, fontsize=12, pad=8)


def write_summary(
    path: str,
    labels: List[str],
    mc_corr: np.ndarray,
    ma_corr: np.ndarray,
) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["signal", "sample_1", "sample_2", "pearson_r"]
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for signal_name, mat in [("mC", mc_corr), ("mA", ma_corr)]:
            for i, a in enumerate(labels):
                for j, b in enumerate(labels):
                    if j < i:
                        continue
                    w.writerow(
                        {
                            "signal": signal_name,
                            "sample_1": a,
                            "sample_2": b,
                            "pearson_r": f"{mat[i, j]:.6f}",
                        }
                    )


def mean_offdiag(corr: np.ndarray) -> float:
    if corr.shape[0] < 2:
        return float("nan")
    vals = corr[np.triu_indices_from(corr, k=1)]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def write_conclusion(path: str, mc_corr: np.ndarray, ma_corr: np.ndarray) -> None:
    mc_mean = mean_offdiag(mc_corr)
    ma_mean = mean_offdiag(ma_corr)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Phase 1 Sample Comparability Conclusion\n\n")
        f.write(
            "Pearson correlation was computed across common 1 kb genomic bins using local mC and mA signal summaries derived from the DiMeLo-seq data.\n\n"
        )
        f.write(f"- Mean off-diagonal mC correlation: {mc_mean:.3f}\n")
        f.write(f"- Mean off-diagonal mA correlation: {ma_mean:.3f}\n\n")
        if np.isfinite(mc_mean) and np.isfinite(ma_mean):
            if mc_mean >= ma_mean:
                f.write(
                    "Across the three DiMeLo-seq samples, local mC levels were more consistent across samples than local mA levels. "
                    "In contrast, the mA signal showed stronger mark-dependent divergence, consistent with the interpretation that the adenine-derived signal is more specific to the targeted chromatin context.\n"
                )
            else:
                f.write(
                    "Across the three DiMeLo-seq samples, local mA levels were at least as correlated as local mC levels. "
                    "This suggests that both signals share substantial genome-wide structure at the current 1 kb resolution, although mark-specific differences may still be present in the detailed spatial pattern.\n"
                )


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    marks = [x.strip() for x in args.marks.split(",") if x.strip()]

    mc_by_mark: Dict[str, np.ndarray] = {}
    ma_by_mark: Dict[str, np.ndarray] = {}

    for mark in marks:
        mc = load_mc_signal(mark, args.manifest_glob_root, args.bin_size)
        ma = load_ma_signal(mark, args.feature_cache_dir)
        if args.normalize_cpm:
            mc = normalize_cpm(mc)
            ma = normalize_cpm(ma)
        mc_by_mark[mark] = mc
        ma_by_mark[mark] = ma

    mc_corr = pearson_corr_matrix(mc_by_mark)
    ma_corr = pearson_corr_matrix(ma_by_mark)

    fig = plt.figure(figsize=(13.6, 6.2), dpi=180)
    outer = fig.add_gridspec(1, 2, wspace=0.28)
    plot_cluster_panel(
        fig=fig,
        outer_spec=outer[0],
        corr=mc_corr,
        labels=marks,
        title_letter="A",
        title="DiMeLo-seq mC signal comparability across samples",
    )
    plot_cluster_panel(
        fig=fig,
        outer_spec=outer[1],
        corr=ma_corr,
        labels=marks,
        title_letter="B",
        title="DiMeLo-seq mA signal comparability across samples",
    )

    out_svg = os.path.join(args.out_dir, "phase1_sample_signal_correlation_heatmaps.svg")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    write_summary(
        path=os.path.join(args.out_dir, "phase1_sample_signal_correlation_summary.tsv"),
        labels=marks,
        mc_corr=mc_corr,
        ma_corr=ma_corr,
    )
    write_conclusion(
        path=os.path.join(args.out_dir, "phase1_sample_signal_correlation_conclusion.md"),
        mc_corr=mc_corr,
        ma_corr=ma_corr,
    )
    print(f"Wrote correlation figure and summaries to: {args.out_dir}")


if __name__ == "__main__":
    main()
