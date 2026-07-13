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

COLORS = {
    'h3k27ac': '#cc503e',
    'h3k27me3': '#3e6c8f',
    'h3k4me3': '#4d9a6e',
}


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
    for col in ['frac_A_mod_pass', 'frac_C_mod_pass', 'frac_A_mod_all', 'frac_C_mod_all']:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors='coerce')
    return all_df


def save_summary_table(df: pd.DataFrame, outdir: Path) -> None:
    summary = (
        df.groupby('mark', dropna=False)
        .agg(
            n_reads=('read_id', 'nunique'),
            mean_read_length=('read_length', 'mean'),
            median_read_length=('read_length', 'median'),
            mean_pass_fraction=('n_pass_calls', lambda s: float(np.nanmean(s / df.loc[s.index, 'n_total_calls']))),
            mean_fail_fraction=('n_fail_calls', lambda s: float(np.nanmean(s / df.loc[s.index, 'n_total_calls']))),
            mean_frac_A_mod_pass=('frac_A_mod_pass', 'mean'),
            mean_frac_C_mod_pass=('frac_C_mod_pass', 'mean'),
            mean_n_A_mod_pass=('n_A_mod_pass', 'mean'),
            mean_n_C_mod_pass=('n_C_mod_pass', 'mean'),
        )
        .reset_index()
    )
    summary.to_csv(outdir / 'modkit_read_level_summary_by_mark.tsv', sep='\t', index=False)


def plot_calls_per_read(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    marks = list(dict.fromkeys(df['mark']))
    for ax, col, title in zip(
        axes,
        ['n_A_pass', 'n_C_pass'],
        ['Passing A calls per read', 'Passing C calls per read'],
    ):
        data = [np.log10(df.loc[df['mark'] == mark, col].to_numpy(dtype=float) + 1.0) for mark in marks]
        parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        for body, mark in zip(parts['bodies'], marks):
            body.set_facecolor(COLORS.get(mark, '#777777'))
            body.set_alpha(0.65)
        parts['cmedians'].set_color('black')
        ax.set_xticks(range(1, len(marks) + 1), marks)
        ax.set_ylabel('log10(count + 1)')
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outdir / 'calls_per_read_by_mark.png')
    plt.close(fig)


def plot_pass_fail(df: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for mark, sub in df.groupby('mark'):
        total = sub['n_total_calls'].replace(0, np.nan)
        rows.append({
            'mark': mark,
            'pass_frac': np.nanmean(sub['n_pass_calls'] / total),
            'fail_frac': np.nanmean(sub['n_fail_calls'] / total),
        })
    summary = pd.DataFrame(rows)
    x = np.arange(summary.shape[0])
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=180)
    ax.bar(x, summary['pass_frac'], color='#4d9a6e', label='pass fraction')
    ax.bar(x, summary['fail_frac'], bottom=summary['pass_frac'], color='#c95d4a', label='fail fraction')
    ax.set_xticks(x, summary['mark'])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Mean fraction of calls per read')
    ax.set_title('Passing vs failing calls by mark')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / 'pass_fail_fraction_by_mark.png')
    plt.close(fig)


def plot_burden_scatter(df: pd.DataFrame, outdir: Path, max_points: int) -> None:
    marks = list(dict.fromkeys(df['mark']))
    fig, axes = plt.subplots(1, len(marks), figsize=(5.2 * len(marks), 4.8), dpi=180, squeeze=False)
    rng = np.random.default_rng(42)
    for ax, mark in zip(axes[0], marks):
        sub = df.loc[df['mark'] == mark, ['frac_A_mod_pass', 'frac_C_mod_pass']].dropna()
        if sub.shape[0] > max_points:
            idx = np.sort(rng.choice(sub.shape[0], size=max_points, replace=False))
            sub = sub.iloc[idx]
        hb = ax.hexbin(sub['frac_A_mod_pass'], sub['frac_C_mod_pass'], gridsize=60, mincnt=1, bins='log', cmap='viridis')
        ax.set_xlabel('A burden per read (modified/pass)')
        ax.set_ylabel('C burden per read (modified/pass)')
        ax.set_title(mark)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('log10(bin count)')
    fig.tight_layout()
    fig.savefig(outdir / 'read_level_A_vs_C_burden_hexbin.png')
    plt.close(fig)


def plot_cooccurrence(df: pd.DataFrame, outdir: Path) -> None:
    marks = list(dict.fromkeys(df['mark']))
    rows = []
    for mark, sub in df.groupby('mark'):
        has_a = sub['n_A_mod_pass'] > 0
        has_c = sub['n_C_mod_pass'] > 0
        rows.extend([
            {'mark': mark, 'class': 'A only', 'fraction': np.mean(has_a & ~has_c)},
            {'mark': mark, 'class': 'C only', 'fraction': np.mean(~has_a & has_c)},
            {'mark': mark, 'class': 'A + C', 'fraction': np.mean(has_a & has_c)},
            {'mark': mark, 'class': 'Neither', 'fraction': np.mean(~has_a & ~has_c)},
        ])
    co = pd.DataFrame(rows)
    classes = ['A only', 'C only', 'A + C', 'Neither']
    fig, ax = plt.subplots(figsize=(8.2, 5), dpi=180)
    bottom = np.zeros(len(marks), dtype=float)
    palette = {'A only': '#4477aa', 'C only': '#cc6677', 'A + C': '#44aa99', 'Neither': '#bbbbbb'}
    for cls in classes:
        vals = [co.loc[(co['mark'] == m) & (co['class'] == cls), 'fraction'].iloc[0] for m in marks]
        ax.bar(marks, vals, bottom=bottom, color=palette[cls], label=cls)
        bottom += np.array(vals)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Fraction of reads')
    ax.set_title('Read-level A/C co-occurrence')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / 'read_level_A_C_cooccurrence_stacked.png')
    plt.close(fig)


def plot_mod_counts(df: pd.DataFrame, outdir: Path) -> None:
    long_rows = []
    for _, row in df.iterrows():
        long_rows.append({'mark': row['mark'], 'channel': 'A', 'count': row['n_A_mod_pass']})
        long_rows.append({'mark': row['mark'], 'channel': 'C', 'count': row['n_C_mod_pass']})
    long_df = pd.DataFrame(long_rows)
    marks = list(dict.fromkeys(df['mark']))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    for ax, channel in zip(axes, ['A', 'C']):
        data = [np.log10(long_df.loc[(long_df['mark'] == mark) & (long_df['channel'] == channel), 'count'].to_numpy(dtype=float) + 1.0) for mark in marks]
        parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        for body, mark in zip(parts['bodies'], marks):
            body.set_facecolor(COLORS.get(mark, '#777777'))
            body.set_alpha(0.65)
        parts['cmedians'].set_color('black')
        ax.set_xticks(range(1, len(marks) + 1), marks)
        ax.set_ylabel('log10(modified pass calls + 1)')
        ax.set_title(f'{channel}-channel modified calls per read')
    fig.tight_layout()
    fig.savefig(outdir / 'modified_calls_per_read_by_mark.png')
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Plot read-level figures from modkit per-read summaries.')
    p.add_argument('--input', action='append', default=[], metavar='MARK=PATH', help='Per-read summary input TSV for one mark.')
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--max-scatter-points', type=int, default=50000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    mark_paths = parse_mark_inputs(args.input) if args.input else DEFAULT_INPUTS
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_tables(mark_paths)
    save_summary_table(df, outdir)
    plot_calls_per_read(df, outdir)
    plot_pass_fail(df, outdir)
    plot_burden_scatter(df, outdir, args.max_scatter_points)
    plot_cooccurrence(df, outdir)
    plot_mod_counts(df, outdir)


if __name__ == '__main__':
    main()
