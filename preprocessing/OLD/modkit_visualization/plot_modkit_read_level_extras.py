#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUTS = {
    'h3k27ac': Path('/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/per_read_summary/h3k27ac.per_read_modkit_summary.tsv'),
    'h3k27me3': Path('/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/per_read_summary/h3k27me3.per_read_modkit_summary.tsv'),
    'h3k4me3': Path('/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/per_read_summary/h3k4me3.per_read_modkit_summary.tsv'),
}

MARK_COLORS = {
    'h3k27ac': '#cc503e',
    'h3k27me3': '#3e6c8f',
    'h3k4me3': '#4d9a6e',
}

CHROMS = [f'chr{i}' for i in range(1, 23)] + ['chrX']


def parse_mark_inputs(entries: List[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for entry in entries:
        if '=' not in entry:
            raise SystemExit(f'Expected MARK=PATH, got {entry!r}')
        mark, path = entry.split('=', 1)
        parsed[mark.strip()] = Path(path.strip()).expanduser()
    return parsed


def load_tables(mark_paths: Dict[str, Path]) -> pd.DataFrame:
    frames = []
    for mark, path in mark_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, sep='\t')
        df['mark'] = mark
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    for col in ['frac_A_mod_pass', 'frac_C_mod_pass', 'frac_A_mod_all', 'frac_C_mod_all', 'read_length']:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors='coerce')
    return all_df


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(x.shape[0], dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < sorted_x.shape[0]:
        j = i + 1
        while j < sorted_x.shape[0] and sorted_x[j] == sorted_x[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float('nan')
    return pearson_corr(rankdata_average(x), rankdata_average(y))


def plot_read_length_vs_burden(df: pd.DataFrame, outdir: Path, max_points: int) -> None:
    marks = list(dict.fromkeys(df['mark']))
    fig, axes = plt.subplots(2, len(marks), figsize=(5.2 * len(marks), 8.8), dpi=180, squeeze=False)
    rng = np.random.default_rng(42)
    rows = []
    for col_idx, mark in enumerate(marks):
        sub = df.loc[df['mark'] == mark].copy()
        sub = sub[['read_length', 'frac_A_mod_pass', 'frac_C_mod_pass']].dropna()
        if sub.shape[0] > max_points:
            idx = np.sort(rng.choice(sub.shape[0], size=max_points, replace=False))
            sub = sub.iloc[idx]

        x = np.log10(sub['read_length'].to_numpy(dtype=float) + 1.0)
        for row_idx, burden_col, label in [(0, 'frac_A_mod_pass', 'A burden'), (1, 'frac_C_mod_pass', 'C burden')]:
            y = sub[burden_col].to_numpy(dtype=float)
            ax = axes[row_idx, col_idx]
            hb = ax.hexbin(x, y, gridsize=60, mincnt=1, bins='log', cmap='viridis')
            ax.set_xlabel('log10(read length + 1)')
            ax.set_ylabel(label)
            ax.set_title(mark)
            cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label('log10(bin count)')
            rows.append({
                'mark': mark,
                'metric': burden_col,
                'pearson': pearson_corr(x, y),
                'spearman': spearman_corr(x, y),
                'n': int(x.size),
            })
    fig.tight_layout()
    fig.savefig(outdir / 'read_length_vs_burden_hexbin.png')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outdir / 'read_length_vs_burden_stats.tsv', sep='\t', index=False)


def summarize_by_chrom(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(['mark', 'chrom'], dropna=False)
        .agg(
            n_reads=('read_id', 'nunique'),
            mean_A_burden=('frac_A_mod_pass', 'mean'),
            mean_C_burden=('frac_C_mod_pass', 'mean'),
            mean_read_length=('read_length', 'mean'),
        )
        .reset_index()
    )
    grouped['chrom'] = pd.Categorical(grouped['chrom'], categories=CHROMS, ordered=True)
    return grouped.sort_values(['mark', 'chrom'])


def plot_chrom_summary(df: pd.DataFrame, outdir: Path) -> None:
    grouped = summarize_by_chrom(df)
    grouped.to_csv(outdir / 'chromosome_level_summary.tsv', sep='\t', index=False)

    marks = list(dict.fromkeys(df['mark']))
    x = np.arange(len(CHROMS))
    width = 0.26

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=180, sharex=True)
    metric_info = [
        ('mean_A_burden', 'Mean A burden per read'),
        ('mean_C_burden', 'Mean C burden per read'),
        ('n_reads', 'Number of reads'),
    ]

    for ax, (metric, ylabel) in zip(axes, metric_info):
        for idx, mark in enumerate(marks):
            sub = grouped.loc[grouped['mark'] == mark].set_index('chrom').reindex(CHROMS)
            vals = sub[metric].to_numpy(dtype=float)
            offset = (idx - 1) * width
            ax.bar(x + offset, vals, width=width, label=mark, color=MARK_COLORS.get(mark, '#777777'))
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
    axes[-1].set_xticks(x, CHROMS, rotation=45, ha='right')
    axes[-1].set_xlabel('Chromosome')
    fig.tight_layout()
    fig.savefig(outdir / 'chromosome_level_summary_by_mark.png')
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Extra modkit read-level QC figures.')
    p.add_argument('--input', action='append', default=[], metavar='MARK=PATH')
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--max-scatter-points', type=int, default=50000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    mark_paths = parse_mark_inputs(args.input) if args.input else DEFAULT_INPUTS
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_tables(mark_paths)
    plot_read_length_vs_burden(df, outdir, args.max_scatter_points)
    plot_chrom_summary(df, outdir)


if __name__ == '__main__':
    main()
