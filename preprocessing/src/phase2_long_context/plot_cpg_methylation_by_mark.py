import argparse
import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CACHE_DIR = (
    "/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/"
    "phase2_long_context/windowed_enrichment_1kb_combined"
)
DEFAULT_OUT_DIR = (
    "/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/"
    "phase2_long_context/cpg_methylation_by_mark"
)
DEFAULT_MARKS = ["h3k27ac", "h3k27me3", "h3k4me3"]
MARK_COLORS = {
    "h3k27ac": "#d55e00",
    "h3k27me3": "#0072b2",
    "h3k4me3": "#009e73",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize and plot 1 kb CpG methylation fractions by histone mark."
    )
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--bin-size", type=int, default=1000)
    p.add_argument("--reg-mode", default="a_mod_per_kb")
    p.add_argument("--m-mode", default="c_meth_frac")
    p.add_argument("--min-cov", type=int, default=30)
    p.add_argument("--sample-size", type=int, default=100000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--marks", nargs="+", default=DEFAULT_MARKS)
    return p.parse_args()


def cache_path(cache_dir: str, mark: str, bin_size: int, reg_mode: str, m_mode: str, min_cov: int) -> str:
    return os.path.join(
        cache_dir,
        f"feature_cache_{mark}_bin{bin_size}_{reg_mode}_{m_mode}_cov{min_cov}.npz",
    )


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(values)
    ys = np.arange(1, xs.size + 1, dtype=np.float64) / xs.size
    return xs, ys


def summary_row(mark: str, values: np.ndarray) -> Dict[str, float | int | str]:
    q10, q25, q50, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "mark": mark,
        "n_bins": int(values.size),
        "mean_meth_frac": float(values.mean()),
        "std_meth_frac": float(values.std()),
        "q10": float(q10),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "q90": float(q90),
        "frac_lt_0_20": float((values < 0.20).mean()),
        "frac_gt_0_80": float((values > 0.80).mean()),
    }


def write_summary_tsv(path: str, rows: List[Dict[str, float | int | str]]) -> None:
    fields = [
        "mark",
        "n_bins",
        "mean_meth_frac",
        "std_meth_frac",
        "q10",
        "q25",
        "median",
        "q75",
        "q90",
        "frac_lt_0_20",
        "frac_gt_0_80",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    full_values: Dict[str, np.ndarray] = {}
    sampled_values: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, float | int | str]] = []

    for mark in args.marks:
        path = cache_path(args.cache_dir, mark, args.bin_size, args.reg_mode, args.m_mode, args.min_cov)
        arr = np.load(path, allow_pickle=False)
        values = arr["meth"][arr["cov_ok"].astype(bool)].astype(np.float32)
        full_values[mark] = values
        rows.append(summary_row(mark, values))

        n = min(args.sample_size, values.size)
        idx = rng.choice(values.size, size=n, replace=False)
        sampled_values[mark] = values[idx]

    rows.sort(key=lambda x: str(x["mark"]))
    write_summary_tsv(os.path.join(args.out_dir, "cpg_methylation_summary.tsv"), rows)

    marks = [str(row["mark"]) for row in rows]
    samples = [sampled_values[m] for m in marks]
    positions = np.arange(1, len(marks) + 1)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12.5, 4.8), dpi=160, gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    parts = ax0.violinplot(
        samples,
        positions=positions,
        widths=0.9,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, mark in zip(parts["bodies"], marks):
        body.set_facecolor(MARK_COLORS.get(mark, "#666666"))
        body.set_edgecolor("black")
        body.set_alpha(0.65)

    box = ax0.boxplot(
        samples,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, mark in zip(box["boxes"], marks):
        patch.set_facecolor(MARK_COLORS.get(mark, "#666666"))
        patch.set_alpha(0.9)

    ax0.set_xticks(positions)
    ax0.set_xticklabels(marks, rotation=20, ha="right")
    ax0.set_ylabel("CpG methylation fraction per 1 kb bin")
    ax0.set_title("Distribution by histone mark")
    ax0.set_ylim(0.0, 1.0)
    ax0.grid(axis="y", alpha=0.25, linewidth=0.6)

    for mark in marks:
        xs, ys = ecdf(full_values[mark])
        ax1.plot(xs, ys, label=mark, color=MARK_COLORS.get(mark, "#666666"), linewidth=2.0)
    ax1.set_xlabel("CpG methylation fraction per 1 kb bin")
    ax1.set_ylabel("Empirical cumulative fraction")
    ax1.set_title("ECDF across all coverage-filtered bins")
    ax1.set_xlim(0.0, 0.5)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.legend(frameon=False, loc="lower right")

    fig.suptitle("CpG methylation differs across histone-mark windows", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cpg_methylation_by_mark.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, "cpg_methylation_by_mark.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
