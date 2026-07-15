#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig


CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
COLORS = {
    "h3k27ac": "#d55e00",
    "h3k27me3": "#cc79a7",
    "h3k4me3": "#009e73",
    "merged_c1": "#0072b2",
}


@dataclass
class SampleTracks:
    name: str
    a_bw: str
    c_bw: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create staging-based summary and dependence figures from A/C bigWig tracks."
    )
    p.add_argument("--sample", action="append", required=True, help="NAME=A_BW,C_BW")
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument("--promoters-bed", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bin-size-200", type=int, default=200)
    p.add_argument("--bin-size-50", type=int, default=50)
    p.add_argument("--n-quantiles", type=int, default=10)
    p.add_argument("--max-bins-per-sample", type=int, default=120000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_samples(entries: Sequence[str]) -> List[SampleTracks]:
    out: List[SampleTracks] = []
    for entry in entries:
        if "=" not in entry or "," not in entry.split("=", 1)[1]:
            raise SystemExit(f"Expected NAME=A_BW,C_BW, got: {entry}")
        name, rhs = entry.split("=", 1)
        a_bw, c_bw = rhs.split(",", 1)
        out.append(SampleTracks(name=name.strip(), a_bw=a_bw.strip(), c_bw=c_bw.strip()))
    return out


def load_chrom_sizes(path: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chrom, size = line.rstrip("\n").split("\t")[:2]
            if chrom in CHROMS:
                out[chrom] = int(size)
    return out


def load_bed(path: str) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, end = line.rstrip("\n").split("\t")[:3]
            if chrom not in CHROMS:
                continue
            out.setdefault(chrom, []).append((int(start), int(end)))
    for chrom in out:
        out[chrom].sort()
    return out


def tiled_intervals(chrom_sizes: Dict[str, int], bin_size: int) -> Iterable[Tuple[str, int, int]]:
    for chrom in CHROMS:
        if chrom not in chrom_sizes:
            continue
        size = chrom_sizes[chrom]
        for start in range(0, size, bin_size):
            end = min(start + bin_size, size)
            yield chrom, start, end


def intersect_mask(
    intervals: Sequence[Tuple[str, int, int]], regions: Dict[str, List[Tuple[int, int]]]
) -> np.ndarray:
    mask = np.zeros(len(intervals), dtype=bool)
    by_chrom: Dict[str, List[Tuple[int, int, int]]] = {}
    for idx, (chrom, start, end) in enumerate(intervals):
        by_chrom.setdefault(chrom, []).append((idx, start, end))
    for chrom, items in by_chrom.items():
        if chrom not in regions:
            continue
        reg = regions[chrom]
        j = 0
        for idx, start, end in items:
            while j < len(reg) and reg[j][1] <= start:
                j += 1
            k = j
            while k < len(reg) and reg[k][0] < end:
                if reg[k][1] > start:
                    mask[idx] = True
                    break
                k += 1
    return mask


def fetch_interval_means(bw_path: str, intervals: Sequence[Tuple[str, int, int]]) -> np.ndarray:
    values = np.full(len(intervals), np.nan, dtype=np.float32)
    with pyBigWig.open(bw_path) as bw:
        for i, (chrom, start, end) in enumerate(intervals):
            try:
                stat = bw.stats(chrom, start, end, type="mean", exact=True)[0]
            except RuntimeError:
                stat = None
            if stat is not None and math.isfinite(stat):
                values[i] = float(stat)
    return values


def finite_pair(a: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(c)
    return a[mask], c[mask]


def subsample_pair(a: np.ndarray, c: np.ndarray, max_n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if a.size <= max_n or max_n <= 0:
        return a, c
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(a.size, size=max_n, replace=False))
    return a[idx], c[idx]


def quantile_bin_indices(values: np.ndarray, n_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    for i in range(1, edges.shape[0]):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return np.digitize(values, edges[1:-1], right=False).astype(np.int64)


def binned_counts(a: np.ndarray, c: np.ndarray, n_bins: int) -> np.ndarray:
    a_idx = quantile_bin_indices(a, n_bins)
    c_idx = quantile_bin_indices(c, n_bins)
    flat = c_idx * n_bins + a_idx
    return np.bincount(flat, minlength=n_bins * n_bins).reshape((n_bins, n_bins))


def enrichment_log2(counts: np.ndarray, pseudocount: float = 1e-9) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    p_ij = counts / total
    p_i = p_ij.sum(axis=0, keepdims=True)
    p_j = p_ij.sum(axis=1, keepdims=True)
    expected = p_j @ p_i
    return np.log2((p_ij + pseudocount) / (expected + pseudocount))


def mutual_information(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    p_ij = counts / total
    p_i = p_ij.sum(axis=0, keepdims=True)
    p_j = p_ij.sum(axis=1, keepdims=True)
    expected = p_j @ p_i
    mask = p_ij > 0
    return float(np.sum(p_ij[mask] * np.log2(p_ij[mask] / expected[mask])))


def save_violin_ecdf(samples: Dict[str, np.ndarray], out_png: str, out_tsv: str, ylabel: str, title: str) -> None:
    rows = []
    names = list(samples.keys())
    arrays = [samples[n] for n in names]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180, gridspec_kw={"width_ratios": [1.0, 1.1]})
    parts = ax0.violinplot(arrays, positions=np.arange(1, len(names) + 1), showmeans=False, showmedians=False, showextrema=False)
    for body, name in zip(parts["bodies"], names):
        body.set_facecolor(COLORS.get(name, "#666666"))
        body.set_edgecolor("black")
        body.set_alpha(0.65)
    ax0.boxplot(arrays, positions=np.arange(1, len(names) + 1), widths=0.22, showfliers=False)
    ax0.set_xticks(np.arange(1, len(names) + 1))
    ax0.set_xticklabels(names, rotation=20, ha="right")
    ax0.set_ylabel(ylabel)
    ax0.set_title("Distribution")
    ax0.grid(axis="y", alpha=0.25, linewidth=0.6)
    for name, vals in samples.items():
        xs = np.sort(vals)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax1.plot(xs, ys, color=COLORS.get(name, "#666666"), linewidth=2.0, label=name)
        q10, q50, q90 = np.quantile(vals, [0.1, 0.5, 0.9])
        rows.append((name, vals.size, float(vals.mean()), float(vals.std()), float(q10), float(q50), float(q90)))
    ax1.set_xlabel(ylabel)
    ax1.set_ylabel("Empirical cumulative fraction")
    ax1.set_title("ECDF")
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.legend(frameon=False, loc="lower right")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    with open(out_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample", "n", "mean", "std", "q10", "median", "q90"])
        writer.writerows(rows)


def save_conditional_a(samples: Dict[str, Tuple[np.ndarray, np.ndarray]], out_prefix: str, n_quantiles: int) -> None:
    rows = []
    names = list(samples.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.1 * len(names), 5.2), dpi=180, squeeze=False)
    for ax, name in zip(axes[0], names):
        a_vals, c_vals = samples[name]
        q_idx = quantile_bin_indices(c_vals, n_quantiles)
        groups = [a_vals[q_idx == i] for i in range(n_quantiles)]
        prepared = [g if g.size else np.array([np.nan]) for g in groups]
        parts = ax.violinplot(prepared, positions=np.arange(1, n_quantiles + 1), widths=0.85, showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#9ecae1")
            body.set_edgecolor("#3182bd")
            body.set_alpha(0.55)
        medians = []
        means = []
        q25s = []
        q75s = []
        for i, vals in enumerate(groups, start=1):
            if vals.size:
                q25, q50, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
                mean = float(np.mean(vals))
            else:
                q25 = q50 = q75 = mean = np.nan
            rows.append((name, i, vals.size, mean, q25, q50, q75))
            medians.append(q50)
            means.append(mean)
            q25s.append(q25)
            q75s.append(q75)
        xpos = np.arange(1, n_quantiles + 1)
        ax.scatter(xpos, medians, color="#08519c", s=16, zorder=3)
        ax.vlines(xpos, q25s, q75s, color="#08519c", linewidth=2, alpha=0.9)
        ax.plot(xpos, means, color="#cb181d", linewidth=1.4, alpha=0.9)
        ax.set_title(name)
        ax.set_xlabel("CpG methylation quantile")
        ax.set_ylabel("A signal")
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"Q{i}" for i in xpos], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.suptitle("Conditional A distributions by CpG methylation quantile", y=0.995)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", bbox_inches="tight")
    plt.close(fig)
    with open(out_prefix + ".tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample", "c_quantile", "n", "mean", "q25", "median", "q75"])
        writer.writerows(rows)


def save_joint_density(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, out_png: str, out_tsv: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=180)
    hb = ax.hexbin(a, b, gridsize=120, mincnt=1, bins="log", cmap="viridis")
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    ax.set_title(f"{label_a} vs {label_b}")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log10(bin count)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    with open(out_tsv, "w", encoding="utf-8") as handle:
        handle.write("metric\tvalue\n")
        handle.write(f"label_a\t{label_a}\n")
        handle.write(f"label_b\t{label_b}\n")
        handle.write(f"n_plotted\t{a.size}\n")
        handle.write(f"pearson\t{np.corrcoef(a, b)[0, 1]:.6f}\n")


def save_hexbin_panels(samples: Dict[str, Tuple[np.ndarray, np.ndarray]], out_png: str, out_tsv: str) -> None:
    names = list(samples.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.7 * len(names), 4.8), dpi=180, squeeze=False)
    rows = []
    for ax, name in zip(axes[0], names):
        a_vals, c_vals = samples[name]
        hb = ax.hexbin(c_vals, a_vals, gridsize=80, mincnt=1, bins="log", cmap="viridis")
        ax.set_xlabel("CpG methylation signal")
        ax.set_ylabel("A signal")
        ax.set_title(name)
        rows.append((name, a_vals.size, float(np.corrcoef(c_vals, a_vals)[0, 1])))
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("A signal versus CpG methylation", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    with open(out_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample", "n", "pearson"])
        writer.writerows(rows)


def save_oe_heatmaps(samples: Dict[str, Tuple[np.ndarray, np.ndarray]], out_svg: str, out_tsv: str, n_quantiles: int) -> None:
    names = list(samples.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.2), dpi=180, squeeze=False)
    rows = []
    for ax, name in zip(axes[0], names):
        a_vals, c_vals = samples[name]
        counts = binned_counts(a_vals, c_vals, n_quantiles)
        oe = enrichment_log2(counts)
        im = ax.imshow(oe, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        ax.set_title(name)
        ax.set_xlabel("A quantile")
        ax.set_ylabel("C quantile")
        rows.append((name, mutual_information(counts)))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Observed / expected enrichment by sample", y=1.02)
    fig.tight_layout()
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    with open(out_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample", "mutual_information_bits"])
        writer.writerows(rows)


def save_permutation_oe(samples: Dict[str, Tuple[np.ndarray, np.ndarray]], out_prefix: str, n_quantiles: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    names = list(samples.keys())
    observed = []
    shuffled = []
    for name in names:
        a_vals, c_vals = samples[name]
        counts_obs = binned_counts(a_vals, c_vals, n_quantiles)
        observed.append(mutual_information(counts_obs))
        shuf = c_vals.copy()
        rng.shuffle(shuf)
        shuffled.append(mutual_information(binned_counts(a_vals, shuf, n_quantiles)))
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=180)
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.18, observed, width=0.36, color="#cb181d", label="observed")
    ax.bar(xpos + 0.18, shuffled, width=0.36, color="#9ecae1", label="shuffled")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Mutual information (bits)")
    ax.set_title("Observed versus shuffled dependence")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", bbox_inches="tight")
    plt.close(fig)
    with open(out_prefix + ".tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample", "observed_mi_bits", "shuffled_mi_bits"])
        for name, obs, shuf in zip(names, observed, shuffled):
            writer.writerow([name, f"{obs:.6f}", f"{shuf:.6f}"])


def save_mark_trend_surface(data_50: Dict[str, Tuple[np.ndarray, np.ndarray]], out_prefix: str, n_quantiles: int) -> None:
    names = list(data_50.keys())
    surface = np.full((len(names), n_quantiles), np.nan, dtype=np.float64)
    for i, name in enumerate(names):
        a_vals, c_vals = data_50[name]
        q_idx = quantile_bin_indices(c_vals, n_quantiles)
        for q in range(n_quantiles):
            vals = a_vals[q_idx == q]
            if vals.size:
                surface[i, q] = float(np.mean(vals))
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    im = ax.imshow(surface, aspect="auto", cmap="RdYlBu_r", origin="lower")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks(np.arange(n_quantiles))
    ax.set_xticklabels([f"Q{i}" for i in range(1, n_quantiles + 1)])
    ax.set_xlabel("CpG methylation quantile (50 bp bins)")
    ax.set_ylabel("Sample")
    ax.set_title("Mean A signal trend surface")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", bbox_inches="tight")
    plt.close(fig)
    with open(out_prefix + ".tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample"] + [f"Q{i}" for i in range(1, n_quantiles + 1)])
        for name, row in zip(names, surface):
            writer.writerow([name] + [("nan" if np.isnan(v) else f"{v:.6f}") for v in row])


def save_mi_summary(data_200: Dict[str, Tuple[np.ndarray, np.ndarray]], data_50: Dict[str, Tuple[np.ndarray, np.ndarray]], out_prefix: str, n_quantiles: int) -> None:
    names = list(data_200.keys())
    mi_200 = [mutual_information(binned_counts(*data_200[name], n_quantiles)) for name in names]
    mi_50 = [mutual_information(binned_counts(*data_50[name], n_quantiles)) for name in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.18, mi_200, width=0.36, color="#3182bd", label="200 bp")
    ax.bar(xpos + 0.18, mi_50, width=0.36, color="#e6550d", label="50 bp")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Mutual information (bits)")
    ax.set_title("Dependence summary across resolutions")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", bbox_inches="tight")
    plt.close(fig)
    with open(out_prefix + ".tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample", "mi_200bp_bits", "mi_50bp_bits"])
        for name, a, b in zip(names, mi_200, mi_50):
            writer.writerow([name, f"{a:.6f}", f"{b:.6f}"])


def save_promoter_background(
    data_all: Dict[str, Tuple[np.ndarray, np.ndarray]],
    data_prom: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_prefix: str,
) -> None:
    names = list(data_all.keys())
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=180)
    xpos = np.arange(len(names))
    all_a = [float(np.nanmean(data_all[n][0])) for n in names]
    prom_a = [float(np.nanmean(data_prom[n][0])) for n in names]
    all_c = [float(np.nanmean(data_all[n][1])) for n in names]
    prom_c = [float(np.nanmean(data_prom[n][1])) for n in names]
    axes[0].bar(xpos - 0.18, all_a, width=0.36, color="#9ecae1", label="background")
    axes[0].bar(xpos + 0.18, prom_a, width=0.36, color="#08519c", label="promoter CGI overlap")
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("Mean A signal")
    axes[0].set_title("Promoter versus background: A")
    axes[0].legend(frameon=False)
    axes[1].bar(xpos - 0.18, all_c, width=0.36, color="#fcbba1", label="background")
    axes[1].bar(xpos + 0.18, prom_c, width=0.36, color="#cb181d", label="promoter CGI overlap")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel("Mean CpG methylation signal")
    axes[1].set_title("Promoter versus background: C")
    axes[1].legend(frameon=False)
    for name, aa, ap, ca, cp in zip(names, all_a, prom_a, all_c, prom_c):
        rows.append((name, aa, ap, ca, cp))
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", bbox_inches="tight")
    plt.close(fig)
    with open(out_prefix + ".tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample", "mean_a_background", "mean_a_promoter", "mean_c_background", "mean_c_promoter"])
        writer.writerows(rows)


def save_cluster_contingency(a_vals: np.ndarray, c_vals: np.ndarray, out_prefix: str, n_bins: int) -> None:
    counts = binned_counts(a_vals, c_vals, n_bins)
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=180)
    im = ax.imshow(counts, origin="lower", aspect="auto", cmap="Blues")
    ax.set_xlabel("A quantile cluster")
    ax.set_ylabel("C quantile cluster")
    ax.set_title("h3k4me3 contingency")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", bbox_inches="tight")
    plt.close(fig)
    with open(out_prefix + ".tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["c_quantile"] + [f"a_q{i}" for i in range(1, n_bins + 1)])
        for i, row in enumerate(counts, start=1):
            writer.writerow([f"c_q{i}"] + row.tolist())


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    chrom_sizes = load_chrom_sizes(args.chrom_sizes)
    promoters = load_bed(args.promoters_bed)
    intervals_200 = list(tiled_intervals(chrom_sizes, args.bin_size_200))
    intervals_50 = list(tiled_intervals(chrom_sizes, args.bin_size_50))
    promoter_mask_200 = intersect_mask(intervals_200, promoters)
    samples = parse_samples(args.sample)

    data_200: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    data_50: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    promoter_200: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for i, sample in enumerate(samples):
        a200 = fetch_interval_means(sample.a_bw, intervals_200)
        c200 = fetch_interval_means(sample.c_bw, intervals_200)
        a200, c200 = finite_pair(a200, c200)
        a200, c200 = subsample_pair(a200, c200, args.max_bins_per_sample, args.seed + i)
        data_200[sample.name] = (a200, c200)

        # Promoter/background split is computed before subsampling to preserve overlap mask.
        a200_all = fetch_interval_means(sample.a_bw, intervals_200)
        c200_all = fetch_interval_means(sample.c_bw, intervals_200)
        a_prom = a200_all[promoter_mask_200]
        c_prom = c200_all[promoter_mask_200]
        a_prom, c_prom = finite_pair(a_prom, c_prom)
        promoter_200[sample.name] = subsample_pair(a_prom, c_prom, max(1, args.max_bins_per_sample // 3), args.seed + 100 + i)

        a50 = fetch_interval_means(sample.a_bw, intervals_50)
        c50 = fetch_interval_means(sample.c_bw, intervals_50)
        a50, c50 = finite_pair(a50, c50)
        a50, c50 = subsample_pair(a50, c50, args.max_bins_per_sample, args.seed + 200 + i)
        data_50[sample.name] = (a50, c50)

    save_violin_ecdf(
        {name: vals[1] for name, vals in data_200.items()},
        os.path.join(args.out_dir, "cpg_methylation_by_mark.png"),
        os.path.join(args.out_dir, "cpg_methylation_by_mark.tsv"),
        ylabel="CpG methylation signal per 200 bp bin",
        title="CpG methylation differs across samples",
    )
    save_conditional_a(data_200, os.path.join(args.out_dir, "conditional_a_meth_frac_200bp"), args.n_quantiles)
    save_hexbin_panels(
        data_200,
        os.path.join(args.out_dir, "marks_vs_cpg_methylation_hexbin_a_meth_fraction_200bp.png"),
        os.path.join(args.out_dir, "marks_vs_cpg_methylation_hexbin_a_meth_fraction_200bp.tsv"),
    )
    save_oe_heatmaps(
        data_50,
        os.path.join(args.out_dir, "oe_heatmap_by_mark.svg"),
        os.path.join(args.out_dir, "oe_heatmap_by_mark.tsv"),
        args.n_quantiles,
    )
    save_permutation_oe(data_200, os.path.join(args.out_dir, "permutation_oe_200bp"), args.n_quantiles, args.seed)
    save_mark_trend_surface(data_50, os.path.join(args.out_dir, "mark_trend_surface_50bp"), args.n_quantiles)
    save_mi_summary(data_200, data_50, os.path.join(args.out_dir, "mi_summary"), args.n_quantiles)
    save_promoter_background(data_200, promoter_200, os.path.join(args.out_dir, "promoter_background_200bp"))
    if "h3k4me3" in data_200:
        save_cluster_contingency(*data_200["h3k4me3"], os.path.join(args.out_dir, "h3k4me3_cluster_contingency_200bp"), args.n_quantiles)
    if "h3k4me3" in data_200 and "h3k27me3" in data_200:
        a_h3k4 = data_200["h3k4me3"][0]
        a_h3k27 = data_200["h3k27me3"][0]
        n = min(a_h3k4.size, a_h3k27.size)
        save_joint_density(
            a_h3k4[:n],
            a_h3k27[:n],
            "h3k4me3 A signal",
            "h3k27me3 A signal",
            os.path.join(args.out_dir, "h3k4me3_vs_h3k27me3_a_meth_fraction_joint_density.png"),
            os.path.join(args.out_dir, "h3k4me3_vs_h3k27me3_a_meth_fraction_joint_density.tsv"),
        )

    # Reuse the conditional A figure under a second legacy-style name for convenience.
    # This keeps downstream notes simpler when comparing with the old folder.
    for ext in (".png", ".tsv"):
        src = os.path.join(args.out_dir, "conditional_a_meth_frac_200bp" + ext)
        dst = os.path.join(args.out_dir, "conditional_reg_200bp" + ext)
        if os.path.exists(src):
            with open(src, "rb") as s, open(dst, "wb") as d:
                d.write(s.read())


if __name__ == "__main__":
    main()
