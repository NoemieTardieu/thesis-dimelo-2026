#!/usr/bin/env python3
"""Plot cross-chromosome AUROC/AUPRC for P(Reg|D,M)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="Input TSV with chromosome, auroc, auprc and pos_frac.")
    parser.add_argument("--out-prefix", required=True, help="Output prefix for PNG/SVG/PDF.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metrics, sep="\t")
    required = {"chromosome", "auroc", "auprc", "pos_frac", "auprc_enrichment"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df["chromosome"] = pd.Categorical(
        df["chromosome"],
        categories=[c for c in ["chr16", "chr11", "chr17", "chr19"] if c in set(df["chromosome"])],
        ordered=True,
    )
    df = df.sort_values("chromosome")

    colors = ["#48639c", "#4c956c", "#d08c60", "#9d4edd"][: len(df)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    x = range(len(df))

    ax = axes[0]
    ax.bar(x, df["auroc"], color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(0.5, color="0.35", linestyle="--", linewidth=1.0, label="random")
    ax.set_xticks(list(x), df["chromosome"].astype(str))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Binary ranking performance")
    ax.legend(frameon=False, fontsize=8)
    for i, value in enumerate(df["auroc"]):
        ax.text(i, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    ax.bar(x, df["auprc"], color=colors, edgecolor="black", linewidth=0.7, label="model AUPRC")
    ax.scatter(x, df["pos_frac"], color="black", zorder=3, label="random baseline = positive fraction")
    ax.vlines(x, df["pos_frac"], df["auprc"], color="0.25", linewidth=1.0, alpha=0.7)
    ax.set_xticks(list(x), df["chromosome"].astype(str))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUPRC")
    ax.set_title("Precision-recall under sparse positives")
    ax.legend(frameon=False, fontsize=8)
    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(
            i,
            row["auprc"] + 0.004,
            f"{row['auprc']:.3f}\n{row['auprc_enrichment']:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Cross-chromosome generalization of P(Reg|D,M)", fontsize=13)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".{ext}"), dpi=300)


if __name__ == "__main__":
    main()
