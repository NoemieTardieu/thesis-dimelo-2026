#!/usr/bin/env python3
"""Follow-up diagnostics for ENCODE H3K4me3 versus aggregated DiMeLo.

This script is intentionally a compact downstream diagnostic on top of the
final population benchmark. It does not regenerate AlphaGenome predictions and
does not use external tracks for training.

Main questions addressed:
  * Is ENCODE-vs-DiMeLo agreement sensitive to DiMeLo coverage?
  * Is DiMeLo enriched in high-ENCODE H3K4me3 bins?
  * Do correlations vary by chromosome, region, or signal/coverage strata?
  * If a reference FASTA is provided, are adenine-rich bins better correlated?
"""

from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRACKS = {
    "external": "ENCODE H3K4me3",
    "alphagenome": "AlphaGenome H3K4me3",
    "hyena": "HyenaDNA 6mA",
    "dimelo": "DiMeLo 6mA",
}

COMPARISONS = [
    ("external", "dimelo", "T-D"),
    ("external", "hyena", "T-H"),
    ("external", "alphagenome", "T-A"),
    ("alphagenome", "dimelo", "A-D"),
    ("hyena", "dimelo", "H-D"),
]


def read_tsv(path: Path) -> pd.DataFrame:
    """Read a TSV or TSV.GZ file."""

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return pd.read_csv(handle, sep="\t")


def finite_pair(data: pd.DataFrame, left: str, right: str, min_coverage: int = 0) -> pd.DataFrame:
    """Return rows with finite values for two raw tracks and sufficient DiMeLo coverage."""

    left_col = f"{left}_raw"
    right_col = f"{right}_raw"
    valid = np.isfinite(data[left_col]) & np.isfinite(data[right_col])
    if "dimelo" in {left, right}:
        valid &= data["dimelo_coverage"].fillna(0) >= min_coverage
    return data.loc[valid].copy()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation without scipy."""

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Spearman correlation through rank correlation."""

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(float)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(float)
    return pearson(xr, yr)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate AUROC from ranks."""

    mask = np.isfinite(scores) & np.isfinite(labels)
    labels = labels[mask].astype(bool)
    scores = scores[mask]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate average precision / AUPRC for binary labels."""

    mask = np.isfinite(scores) & np.isfinite(labels)
    labels = labels[mask].astype(bool)
    scores = scores[mask]
    n_pos = int(labels.sum())
    if n_pos == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    rank = np.arange(1, sorted_labels.size + 1)
    precision = tp / rank
    return float(precision[sorted_labels].sum() / n_pos)


def metrics_for(data: pd.DataFrame, left: str, right: str, min_coverage: int) -> dict[str, float | int | str]:
    """Calculate pairwise metrics for raw values."""

    subset = finite_pair(data, left, right, min_coverage)
    x = subset[f"{left}_raw"].to_numpy(float)
    y = subset[f"{right}_raw"].to_numpy(float)
    return {
        "comparison": f"{left}-{right}",
        "label": next(label for l, r, label in COMPARISONS if l == left and r == right),
        "min_dimelo_coverage": min_coverage if "dimelo" in {left, right} else 0,
        "n_bins": int(subset.shape[0]),
        "pearson": pearson(x, y),
        "spearman": spearman(x, y),
        "left_mean": float(np.nanmean(x)) if x.size else np.nan,
        "right_mean": float(np.nanmean(y)) if y.size else np.nan,
    }


def coverage_sensitivity(data: pd.DataFrame, thresholds: Iterable[int]) -> pd.DataFrame:
    """Evaluate DiMeLo-involving correlations over coverage thresholds."""

    rows = []
    for threshold in thresholds:
        for left, right, _ in COMPARISONS:
            if "dimelo" not in {left, right}:
                continue
            rows.append(metrics_for(data, left, right, threshold))
    return pd.DataFrame(rows)


def chromosome_metrics(data: pd.DataFrame, min_coverage: int) -> pd.DataFrame:
    """Calculate pairwise metrics separately by chromosome."""

    rows = []
    for chrom, subset in data.groupby("chrom", sort=True):
        for left, right, _ in COMPARISONS:
            row = metrics_for(subset, left, right, min_coverage)
            row["chrom"] = chrom
            rows.append(row)
    return pd.DataFrame(rows)


def region_metrics(data: pd.DataFrame, min_coverage: int) -> pd.DataFrame:
    """Calculate ENCODE-vs-DiMeLo metrics separately by region."""

    rows = []
    for (chrom, region_id, region_name), subset in data.groupby(["chrom", "region_id", "region_name"], sort=True):
        row = metrics_for(subset, "external", "dimelo", min_coverage)
        row.update({"chrom": chrom, "region_id": region_id, "region_name": region_name})
        rows.append(row)
    return pd.DataFrame(rows)


def add_quantile_bins(data: pd.DataFrame, column: str, bins: int, label: str) -> pd.DataFrame:
    """Add quantile bin labels for a numeric column."""

    out = data.copy()
    valid = np.isfinite(out[column])
    if valid.sum() < bins:
        out[label] = np.nan
        return out
    ranks = out.loc[valid, column].rank(method="first")
    out.loc[valid, label] = pd.qcut(ranks, q=bins, labels=False, duplicates="drop") + 1
    return out


def stratified_metrics(data: pd.DataFrame, min_coverage: int) -> pd.DataFrame:
    """Calculate ENCODE-vs-DiMeLo metrics by signal/coverage strata."""

    work = data.copy()
    for column, label in [
        ("dimelo_coverage", "dimelo_coverage_decile"),
        ("dimelo_read_count", "dimelo_read_count_decile"),
        ("external_raw", "external_signal_decile"),
    ]:
        if column in work.columns:
            work = add_quantile_bins(work, column, 10, label)

    rows = []
    for label in ["dimelo_coverage_decile", "dimelo_read_count_decile", "external_signal_decile"]:
        if label not in work.columns:
            continue
        for value, subset in work.groupby(label, dropna=True, sort=True):
            row = metrics_for(subset, "external", "dimelo", min_coverage)
            row.update({"stratification": label, "stratum": int(value)})
            rows.append(row)
    return pd.DataFrame(rows)


def peak_proxy_enrichment(data: pd.DataFrame, min_coverage: int, quantiles: Iterable[float]) -> pd.DataFrame:
    """Use top external-signal bins as a peak proxy and test DiMeLo enrichment."""

    rows = []
    base = finite_pair(data, "external", "dimelo", min_coverage)
    for q in quantiles:
        threshold = float(base["external_raw"].quantile(q))
        labels = base["external_raw"].to_numpy(float) >= threshold
        pos = base.loc[labels]
        neg = base.loc[~labels]
        dimelo_scores = base["dimelo_raw"].to_numpy(float)
        hyena_scores = base["hyena_raw"].to_numpy(float) if "hyena_raw" in base.columns else np.full(base.shape[0], np.nan)
        alpha_scores = base["alphagenome_raw"].to_numpy(float)
        external_scores = base["external_raw"].to_numpy(float)
        rows.append(
            {
                "external_quantile_cutoff": q,
                "external_raw_threshold": threshold,
                "n_bins": int(base.shape[0]),
                "n_high_external_bins": int(labels.sum()),
                "positive_fraction": float(labels.mean()),
                "dimelo_mean_high_external": float(pos["dimelo_raw"].mean()),
                "dimelo_mean_other": float(neg["dimelo_raw"].mean()),
                "dimelo_fold_enrichment_high_vs_other": float(pos["dimelo_raw"].mean() / neg["dimelo_raw"].mean())
                if neg["dimelo_raw"].mean() > 0
                else np.nan,
                "hyena_mean_high_external": float(pos["hyena_raw"].mean()),
                "hyena_mean_other": float(neg["hyena_raw"].mean()),
                "dimelo_auroc_for_high_external": auroc(labels, dimelo_scores),
                "dimelo_auprc_for_high_external": auprc(labels, dimelo_scores),
                "hyena_auroc_for_high_external": auroc(labels, hyena_scores),
                "hyena_auprc_for_high_external": auprc(labels, hyena_scores),
                "alphagenome_auroc_for_high_external": auroc(labels, alpha_scores),
                "alphagenome_auprc_for_high_external": auprc(labels, alpha_scores),
                "external_self_auroc_sanity": auroc(labels, external_scores),
            }
        )
    return pd.DataFrame(rows)


def load_fasta_sequence(fasta: Path, chrom: str, start: int, end: int) -> str:
    """Fetch a sequence from a FASTA using pysam.

    `pysam.FastaFile` works well on HPC installs and supports `.fai` indexing.
    Chromosome names are tried both with and without the `chr` prefix.
    """

    try:
        import pysam  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("pysam is required for --reference-fasta adenine-density analysis.") from exc
    if not fasta.exists():
        raise SystemExit(f"Reference FASTA does not exist: {fasta}")
    if not hasattr(load_fasta_sequence, "_cache"):
        try:
            setattr(load_fasta_sequence, "_cache", pysam.FastaFile(str(fasta)))
        except OSError as exc:
            raise SystemExit(
                f"Could not open reference FASTA {fasta}. If the .fai index is missing, run: samtools faidx {fasta}"
            ) from exc
    genome = getattr(load_fasta_sequence, "_cache")
    references = set(genome.references)
    candidates = [chrom, chrom.removeprefix("chr")]
    if not chrom.startswith("chr"):
        candidates.append(f"chr{chrom}")
    for candidate in candidates:
        if candidate in references:
            return str(genome.fetch(candidate, start, end)).upper()
    preview = ", ".join(list(genome.references[:8]))
    raise SystemExit(f"Chromosome {chrom!r} was not found in {fasta}. FASTA references begin with: {preview}")


def adenine_density_table(data: pd.DataFrame, fasta: Path | None, min_coverage: int) -> pd.DataFrame:
    """Calculate ENCODE-vs-DiMeLo metrics by adenine-density decile."""

    if fasta is None:
        return pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "No --reference-fasta supplied; adenine-density stratification was not calculated.",
                }
            ]
        )

    work = data.copy()
    densities = []
    counts = []
    for row in work.itertuples(index=False):
        seq = load_fasta_sequence(fasta, row.chrom, int(row.start), int(row.end))
        count = seq.count("A") + seq.count("a")
        counts.append(count)
        densities.append(count / max(1, len(seq)))
    work["adenine_count"] = counts
    work["adenine_density"] = densities
    work = add_quantile_bins(work, "adenine_density", 10, "adenine_density_decile")

    rows = []
    for decile, subset in work.groupby("adenine_density_decile", dropna=True, sort=True):
        row = metrics_for(subset, "external", "dimelo", min_coverage)
        row.update(
            {
                "adenine_density_decile": int(decile),
                "mean_adenine_density": float(subset["adenine_density"].mean()),
                "mean_adenine_count_per_bin": float(subset["adenine_count"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write a TSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def plot_coverage_sensitivity(table: pd.DataFrame, out: Path) -> None:
    """Plot Pearson/Spearman as a function of DiMeLo coverage threshold."""

    if table.empty:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, metric in zip(axes, ["pearson", "spearman"]):
        for label, subset in table.groupby("label", sort=False):
            ax.plot(subset["min_dimelo_coverage"], subset[metric], marker="o", label=label)
        ax.set_xlabel("minimum DiMeLo observations per bin")
        ax.set_ylabel(metric)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylim(-1, 1)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def plot_stratified(table: pd.DataFrame, out: Path) -> None:
    """Plot ENCODE-vs-DiMeLo correlation by decile strata."""

    if table.empty:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, metric in zip(axes, ["pearson", "spearman"]):
        for strat, subset in table.groupby("stratification", sort=False):
            ax.plot(subset["stratum"], subset[metric], marker="o", label=strat)
        ax.set_xlabel("decile")
        ax.set_ylabel(metric)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylim(-1, 1)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def format_float(value: float) -> str:
    """Format a float for markdown."""

    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def write_summary(
    out: Path,
    data: pd.DataFrame,
    coverage: pd.DataFrame,
    peak: pd.DataFrame,
    chrom: pd.DataFrame,
    adenine: pd.DataFrame,
    min_coverage: int,
) -> None:
    """Write a compact Markdown summary."""

    base_td = metrics_for(data, "external", "dimelo", min_coverage)
    high90 = peak.loc[np.isclose(peak["external_quantile_cutoff"], 0.90)].iloc[0] if not peak.empty else None
    chrom_td = chrom[chrom["label"] == "T-D"].copy()
    lines = [
        "# ChIP-seq versus DiMeLo Follow-Up Diagnostics",
        "",
        "This downstream analysis investigates why the external ENCODE H3K4me3 track and aggregated DiMeLo 6mA show moderate bin-level correlation.",
        "",
        "## Main ENCODE-vs-DiMeLo Result",
        "",
        f"- Minimum DiMeLo coverage: `{min_coverage}` valid observations per bin.",
        f"- Shared bins: `{base_td['n_bins']}`.",
        f"- Pearson: `{format_float(base_td['pearson'])}`.",
        f"- Spearman: `{format_float(base_td['spearman'])}`.",
        "",
        "## Peak-Proxy Enrichment",
        "",
    ]
    if high90 is not None:
        lines.extend(
            [
                "Using the top 10% of ENCODE H3K4me3 bins as a peak-like proxy:",
                "",
                f"- DiMeLo mean in high-ENCODE bins: `{format_float(high90['dimelo_mean_high_external'])}`.",
                f"- DiMeLo mean in other bins: `{format_float(high90['dimelo_mean_other'])}`.",
                f"- DiMeLo fold enrichment: `{format_float(high90['dimelo_fold_enrichment_high_vs_other'])}`.",
                f"- DiMeLo AUROC for high-ENCODE bins: `{format_float(high90['dimelo_auroc_for_high_external'])}`.",
                f"- DiMeLo AUPRC for high-ENCODE bins: `{format_float(high90['dimelo_auprc_for_high_external'])}`.",
                f"- Random AUPRC baseline: `{format_float(high90['positive_fraction'])}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Chromosome Consistency",
            "",
            "| Chromosome | n bins | Pearson | Spearman |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in chrom_td.itertuples(index=False):
        lines.append(f"| {row.chrom} | {row.n_bins} | {format_float(row.pearson)} | {format_float(row.spearman)} |")
    lines.extend(["", "## Adenine Density", ""])
    if "status" in adenine.columns and adenine.iloc[0]["status"] == "skipped":
        lines.append(str(adenine.iloc[0]["reason"]))
    else:
        lines.append("Adenine-density stratification was calculated from the supplied reference FASTA.")
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `coverage_sensitivity.tsv`",
            "- `chromosome_metrics.tsv`",
            "- `region_metrics_external_dimelo.tsv`",
            "- `stratified_external_dimelo_metrics.tsv`",
            "- `external_peak_proxy_enrichment.tsv`",
            "- `adenine_density_external_dimelo.tsv`",
            "",
            "Interpretation note: these are diagnostics for the moderate ENCODE-vs-DiMeLo agreement. The peak-proxy enrichment/AUROC can be stronger than Pearson correlation because enrichment asks whether DiMeLo is higher in ChIP-seq-like regions, whereas correlation asks whether exact bin-level amplitudes match.",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-raw",
        type=Path,
        default=Path(
            "server_artifacts/alphagenome_archive_large_tables/"
            "phase10_encode_final_local_artifacts/canonical_raw_bins.tsv.gz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/chipseq_dimelo_followup"),
    )
    parser.add_argument("--min-coverage", type=int, default=5)
    parser.add_argument(
        "--coverage-thresholds",
        default="1,5,10,20,50,100,200,500,1000",
        help="Comma-separated DiMeLo coverage thresholds.",
    )
    parser.add_argument(
        "--external-quantiles",
        default="0.80,0.90,0.95",
        help="External-signal quantile cutoffs used as peak-like proxies.",
    )
    parser.add_argument(
        "--reference-fasta",
        type=Path,
        default=None,
        help="Optional GRCh38 FASTA for adenine-density stratification.",
    )
    args = parser.parse_args()

    data = read_tsv(args.canonical_raw)
    thresholds = [int(value) for value in args.coverage_thresholds.split(",") if value]
    quantiles = [float(value) for value in args.external_quantiles.split(",") if value]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = pd.DataFrame(
        [
            {
                "track": track,
                "n_missing_raw": int(data[f"{track}_raw"].isna().sum()),
                "n_valid_flag": int(data.get(f"{track}_valid", pd.Series(False, index=data.index)).fillna(False).sum()),
            }
            for track in ["external", "alphagenome", "hyena", "dimelo"]
        ]
    )
    coverage = coverage_sensitivity(data, thresholds)
    chrom = chromosome_metrics(data, args.min_coverage)
    region = region_metrics(data, args.min_coverage)
    stratified = stratified_metrics(data, args.min_coverage)
    peak = peak_proxy_enrichment(data, args.min_coverage, quantiles)
    adenine = adenine_density_table(data, args.reference_fasta, args.min_coverage)

    write_table(missing, args.out_dir / "missingness.tsv")
    write_table(coverage, args.out_dir / "coverage_sensitivity.tsv")
    write_table(chrom, args.out_dir / "chromosome_metrics.tsv")
    write_table(region, args.out_dir / "region_metrics_external_dimelo.tsv")
    write_table(stratified, args.out_dir / "stratified_external_dimelo_metrics.tsv")
    write_table(peak, args.out_dir / "external_peak_proxy_enrichment.tsv")
    write_table(adenine, args.out_dir / "adenine_density_external_dimelo.tsv")
    plot_coverage_sensitivity(coverage, args.out_dir / "figures" / "coverage_sensitivity_external_dimelo")
    plot_stratified(stratified, args.out_dir / "figures" / "stratified_external_dimelo")
    write_summary(args.out_dir / "summary.md", data, coverage, peak, chrom, adenine, args.min_coverage)

    print(f"Wrote follow-up diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
