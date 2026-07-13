import argparse
import csv
import glob
import gzip
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_mark_path_map(entries: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for x in entries:
        if '=' not in x:
            raise ValueError(f'Expected MARK=PATH, got: {x}')
        mark, path = x.split('=', 1)
        out[mark.strip()] = path.strip()
    return out


def parse_bed(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    chrom_to_intervals: Dict[str, List[Tuple[int, int]]] = {}
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            if end <= start:
                continue
            chrom_to_intervals.setdefault(chrom, []).append((start, end))

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for chrom, ivs in chrom_to_intervals.items():
        ivs.sort()
        starts = np.asarray([s for s, _ in ivs], dtype=np.int64)
        ends = np.sort(np.asarray([e for _, e in ivs], dtype=np.int64))
        out[chrom] = (starts, ends)
    return out


def overlap_mask_reads(
    chrom: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    peaks_idx: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    mask = np.zeros(chrom.shape[0], dtype=bool)
    for i in range(chrom.shape[0]):
        c = str(chrom[i])
        if c not in peaks_idx:
            continue
        starts, ends = peaks_idx[c]
        s = int(start[i])
        e = int(end[i])
        n_started = int(np.searchsorted(starts, e, side='left'))
        n_ended = int(np.searchsorted(ends, s, side='right'))
        mask[i] = n_started > n_ended
    return mask


def read_per_read_tsv(path: str) -> Dict[str, np.ndarray]:
    reg = []
    c_conf = []
    c_total = []
    chrom = []
    ref_start = []
    ref_end = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            reg.append(float(row['A_mod_density_per_kb']))
            c_conf.append(float(row['C_mod_conf']))
            c_total.append(float(row['C_mod_total']))
            chrom.append(row['chrom'])
            ref_start.append(int(row['ref_start']))
            ref_end.append(int(row['ref_end']))

    reg_arr = np.asarray(reg, dtype=np.float64)
    c_conf_arr = np.asarray(c_conf, dtype=np.float64)
    c_total_arr = np.asarray(c_total, dtype=np.float64)
    m_bin = (c_conf_arr > 0).astype(np.float64)
    m_frac = np.divide(c_conf_arr, np.maximum(c_total_arr, 1.0))
    m_frac = np.clip(m_frac, 0.0, 1.0)

    return {
        'reg_raw': reg_arr,
        'reg01': rank_normalize_01(reg_arr),
        'm_bin': m_bin,
        'm_frac': m_frac,
        'm_conf_raw': c_conf_arr,
        'chrom': np.asarray(chrom, dtype=object),
        'ref_start': np.asarray(ref_start, dtype=np.int64),
        'ref_end': np.asarray(ref_end, dtype=np.int64),
    }


def rank_normalize_01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    return (ranks - 1.0) / max(float(x.size - 1), 1.0)


def gaussian_smooth_2d(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    radius = int(max(2, round(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (xs / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=0, arr=arr)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=1, arr=out)
    return out


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float('nan')
    xr = rank_normalize_01(x)
    yr = rank_normalize_01(y)
    x0 = xr - xr.mean()
    y0 = yr - yr.mean()
    den = np.sqrt((x0 * x0).sum() * (y0 * y0).sum())
    if den <= 0:
        return float('nan')
    return float((x0 * y0).sum() / den)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float('nan')
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.sqrt((x0 * x0).sum() * (y0 * y0).sum())
    if den <= 0:
        return float('nan')
    return float((x0 * y0).sum() / den)


def mutual_info_bits_continuous_binned(x: np.ndarray, y: np.ndarray, n_x: int = 20, n_y: int = 20) -> float:
    if x.size == 0:
        return float('nan')
    x_edges = np.linspace(np.nanmin(x), np.nanmax(x) + 1e-12, n_x + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y) + 1e-12, n_y + 1)
    xb = np.digitize(x, x_edges[1:-1], right=False)
    yb = np.digitize(y, y_edges[1:-1], right=False)
    mat = np.zeros((n_y, n_x), dtype=np.float64)
    for xi, yi in zip(xb, yb):
        mat[yi, xi] += 1
    if mat.sum() <= 0:
        return float('nan')
    pxy = mat / mat.sum()
    px = pxy.sum(axis=0, keepdims=True)
    py = pxy.sum(axis=1, keepdims=True)
    exp = py @ px
    nz = pxy > 0
    return float((pxy[nz] * np.log2(pxy[nz] / exp[nz])).sum())


def plot_joint_marginal_contours(
    path: str,
    x: np.ndarray,
    y: np.ndarray,
    mark: str,
    color: str,
    xlabel: str,
    ylabel: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    y_is_binary: bool = False,
) -> None:
    rng = np.random.default_rng(7)
    y_plot = y.copy()
    if y_is_binary:
        y_plot = np.clip(y + rng.normal(0.0, 0.05, size=y.size), ylim[0], ylim[1])

    fig = plt.figure(figsize=(7.2, 7.2), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    ax_top = fig.add_subplot(gs[0, 0:3])
    ax_right = fig.add_subplot(gs[1:4, 3])
    ax = fig.add_subplot(gs[1:4, 0:3])

    n = x.size
    take = min(7000, n)
    if n > take:
        idx = rng.choice(n, size=take, replace=False)
        xs = x[idx]
        ys = y_plot[idx]
    else:
        xs = x
        ys = y_plot

    ax.scatter(xs, ys, s=6, alpha=0.12, color=color, linewidths=0)

    H, x_edges, y_edges = np.histogram2d(x, y_plot, bins=[80, 80], range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]])
    Hs = gaussian_smooth_2d(H, sigma=1.2)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(xc, yc, indexing='xy')
    pos = Hs[Hs > 0]
    if pos.size:
        levels = np.unique(np.quantile(pos, [0.65, 0.78, 0.88, 0.94, 0.98]))
        ax.contour(X, Y, Hs.T, levels=levels, colors=[color], linewidths=1.2)

    ax_top.hist(x, bins=30, color=color, alpha=0.8)
    ax_right.hist(y, bins=30 if not y_is_binary else np.array([-0.5, 0.5, 1.5]), orientation='horizontal', color=color, alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax_top.set_xlim(*xlim)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_right.set_ylim(*ylim)
    ax_right.set_xticks([])
    if y_is_binary:
        ax_right.set_yticks([0, 1])
        ax_right.set_yticklabels(['0', '1'])

    ax.set_title(mark)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_overlay_contours(
    path: str,
    mark_data: Dict[str, Dict[str, np.ndarray]],
    x_key: str,
    y_key: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    xlabel: str,
    ylabel: str,
    title: str,
    y_is_binary: bool,
) -> None:
    colors = {'h3k27ac': '#d62728', 'h3k27me3': '#1f77b4', 'h3k4me3': '#2ca02c'}
    fig, ax = plt.subplots(figsize=(8.2, 6.4), dpi=150)
    handles = []
    rng = np.random.default_rng(11)

    for mark in sorted(mark_data.keys()):
        x = mark_data[mark][x_key]
        y = mark_data[mark][y_key]
        y_plot = y.copy()
        if y_is_binary:
            y_plot = np.clip(y + rng.normal(0.0, 0.05, size=y.size), ylim[0], ylim[1])

        H, x_edges, y_edges = np.histogram2d(x, y_plot, bins=[80, 80], range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]])
        Hs = gaussian_smooth_2d(H, sigma=1.2)
        xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        yc = 0.5 * (y_edges[:-1] + y_edges[1:])
        X, Y = np.meshgrid(xc, yc, indexing='xy')
        pos = Hs[Hs > 0]
        if pos.size == 0:
            continue

        levels = np.unique(np.quantile(pos, [0.65, 0.78, 0.88, 0.94, 0.98]))
        c = colors.get(mark, '#444444')
        ax.contour(X, Y, Hs.T, levels=levels, colors=[c], linewidths=1.4)
        handles.append(plt.Line2D([0], [0], color=c, lw=2, label=mark))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    if handles:
        ax.legend(handles=handles, title='Mark', loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.subplots_adjust(right=0.80)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_reg_raw_distributions(path: str, mark_data: Dict[str, Dict[str, np.ndarray]]) -> None:
    colors = {'h3k27ac': '#d62728', 'h3k27me3': '#1f77b4', 'h3k4me3': '#2ca02c'}
    marks = sorted(mark_data.keys())
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.4), dpi=150, sharex=False)

    for mark in marks:
        reg = mark_data[mark]['reg_raw']
        c = colors.get(mark, '#444444')
        axes[0].hist(reg, bins=80, density=True, histtype='step', linewidth=1.8, color=c, label=mark)
        axes[1].hist(np.log1p(reg), bins=80, density=True, histtype='step', linewidth=1.8, color=c, label=mark)

    axes[0].set_title('Raw Reg distribution by mark')
    axes[0].set_xlabel('A_mod_density_per_kb (raw)')
    axes[0].set_ylabel('Density')
    axes[0].legend(title='Mark', loc='upper right')

    axes[1].set_title('Reg distribution by mark (log1p transformed)')
    axes[1].set_xlabel('log1p(A_mod_density_per_kb)')
    axes[1].set_ylabel('Density')

    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def write_metrics(path: str, rows: List[Dict[str, object]]) -> None:
    fields = ['mark', 'scope', 'n', 'm_mean', 'spearman_rho', 'pearson_r', 'mutual_info_bits']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Revised Phase 1 plots after promoter feedback.')
    p.add_argument('--input-dir', default='/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase1_output')
    p.add_argument('--output-dir', default='/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase_1_visualization/output_feedback')
    p.add_argument('--peaks-bed-map', action='append', default=[], help='Optional MARK=/path/to/peaks.bed for Option 1')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, '*.per_read.tsv')))
    if not files:
        raise SystemExit(f'No per_read TSV found in {args.input_dir}')

    peaks_map = parse_mark_path_map(args.peaks_bed_map)
    peaks_idx_by_mark: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for mark, bed in peaks_map.items():
        if os.path.exists(bed):
            peaks_idx_by_mark[mark] = parse_bed(bed)

    mark_data: Dict[str, Dict[str, np.ndarray]] = {}
    mark_data_peak_only: Dict[str, Dict[str, np.ndarray]] = {}
    mark_data_peak_only_binary: Dict[str, Dict[str, np.ndarray]] = {}
    mark_data_peak_only_rawm: Dict[str, Dict[str, np.ndarray]] = {}
    rows: List[Dict[str, object]] = []
    colors = {'h3k27ac': '#d62728', 'h3k27me3': '#1f77b4', 'h3k4me3': '#2ca02c'}

    for path in files:
        mark = os.path.basename(path).replace('.per_read.tsv', '')
        d = read_per_read_tsv(path)
        mark_data[mark] = d

        # existing/base figure (binary M)
        plot_joint_marginal_contours(
            path=os.path.join(args.output_dir, f'feedback_joint_marginal_{mark}.svg'),
            x=d['reg01'],
            y=d['m_bin'],
            mark=f'{mark}: binary M',
            color=colors.get(mark, '#444444'),
            xlabel='Reg normalized [0,1]',
            ylabel='M binary (0/1, jittered for contour)',
            xlim=(0.0, 1.0),
            ylim=(-0.1, 1.1),
            y_is_binary=True,
        )

        # Option 2: methylation fraction instead of binary M
        plot_joint_marginal_contours(
            path=os.path.join(args.output_dir, f'option2_joint_marginal_mfrac_{mark}.svg'),
            x=d['reg01'],
            y=d['m_frac'],
            mark=f'{mark}: Reg norm vs methylation fraction',
            color=colors.get(mark, '#444444'),
            xlabel='Reg normalized [0,1]',
            ylabel='Methylation fraction [0,1]',
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            y_is_binary=False,
        )

        # Option 3: Reg density (raw/log1p) vs methylation fraction
        reg_log = np.log1p(d['reg_raw'])
        x_hi = float(np.quantile(reg_log, 0.995)) if reg_log.size else 1.0
        x_hi = max(x_hi, 1e-6)
        plot_joint_marginal_contours(
            path=os.path.join(args.output_dir, f'option3_joint_marginal_regDensity_vs_mfrac_{mark}.svg'),
            x=reg_log,
            y=d['m_frac'],
            mark=f'{mark}: log1p(Reg density) vs methylation fraction',
            color=colors.get(mark, '#444444'),
            xlabel='log1p(A_mod_density_per_kb)',
            ylabel='Methylation fraction [0,1]',
            xlim=(0.0, x_hi),
            ylim=(0.0, 1.0),
            y_is_binary=False,
        )

        rows.append({
            'mark': mark,
            'scope': 'all_reads',
            'n': int(d['reg01'].size),
            'm_mean': float(d['m_frac'].mean()) if d['m_frac'].size else float('nan'),
            'spearman_rho': spearman_corr(d['reg01'], d['m_frac']),
            'pearson_r': pearson_corr(d['reg01'], d['m_frac']),
            'mutual_info_bits': mutual_info_bits_continuous_binned(d['reg01'], d['m_frac']),
        })

        # Option 1: peak-only (if peaks provided)
        if mark in peaks_idx_by_mark:
            peak_mask = overlap_mask_reads(d['chrom'], d['ref_start'], d['ref_end'], peaks_idx_by_mark[mark])
            if peak_mask.any():
                # Option 1B (existing): peak-only with methylation fraction
                mark_data_peak_only[mark] = {
                    'reg01': d['reg01'][peak_mask],
                    'm_frac': d['m_frac'][peak_mask],
                }
                plot_joint_marginal_contours(
                    path=os.path.join(args.output_dir, f'option1_peak_only_joint_marginal_{mark}.svg'),
                    x=d['reg01'][peak_mask],
                    y=d['m_frac'][peak_mask],
                    mark=f'{mark}: peak-overlapping reads only',
                    color=colors.get(mark, '#444444'),
                    xlabel='Reg normalized [0,1]',
                    ylabel='Methylation fraction [0,1]',
                    xlim=(0.0, 1.0),
                    ylim=(0.0, 1.0),
                    y_is_binary=False,
                )
                rows.append({
                    'mark': mark,
                    'scope': 'peak_only',
                    'n': int(peak_mask.sum()),
                    'm_mean': float(d['m_frac'][peak_mask].mean()),
                    'spearman_rho': spearman_corr(d['reg01'][peak_mask], d['m_frac'][peak_mask]),
                    'pearson_r': pearson_corr(d['reg01'][peak_mask], d['m_frac'][peak_mask]),
                    'mutual_info_bits': mutual_info_bits_continuous_binned(d['reg01'][peak_mask], d['m_frac'][peak_mask]),
                })

                # Option 1A: peak-only with binary methylation (direct 0/1)
                mark_data_peak_only_binary[mark] = {
                    'reg01': d['reg01'][peak_mask],
                    'm_bin': d['m_bin'][peak_mask],
                }
                plot_joint_marginal_contours(
                    path=os.path.join(args.output_dir, f'option1_peak_only_joint_marginal_binary_{mark}.svg'),
                    x=d['reg01'][peak_mask],
                    y=d['m_bin'][peak_mask],
                    mark=f'{mark}: peak-overlapping reads only (binary M)',
                    color=colors.get(mark, '#444444'),
                    xlabel='Reg normalized [0,1]',
                    ylabel='M binary (0/1, jittered for contour)',
                    xlim=(0.0, 1.0),
                    ylim=(-0.1, 1.1),
                    y_is_binary=True,
                )
                rows.append({
                    'mark': mark,
                    'scope': 'peak_only_binary',
                    'n': int(peak_mask.sum()),
                    'm_mean': float(d['m_bin'][peak_mask].mean()),
                    'spearman_rho': spearman_corr(d['reg01'][peak_mask], d['m_bin'][peak_mask]),
                    'pearson_r': pearson_corr(d['reg01'][peak_mask], d['m_bin'][peak_mask]),
                    'mutual_info_bits': mutual_info_bits_continuous_binned(d['reg01'][peak_mask], d['m_bin'][peak_mask]),
                })

                # Option 1C: peak-only with direct methylation signal (raw confident calls; log1p for plotting)
                m_conf_log = np.log1p(d['m_conf_raw'][peak_mask])
                mark_data_peak_only_rawm[mark] = {
                    'reg01': d['reg01'][peak_mask],
                    'm_conf_log': m_conf_log,
                }
                y_hi = float(np.quantile(m_conf_log, 0.995)) if m_conf_log.size else 1.0
                y_hi = max(y_hi, 1e-6)
                plot_joint_marginal_contours(
                    path=os.path.join(args.output_dir, f'option1_peak_only_joint_marginal_rawM_{mark}.svg'),
                    x=d['reg01'][peak_mask],
                    y=m_conf_log,
                    mark=f'{mark}: peak-overlapping reads only (log1p raw M)',
                    color=colors.get(mark, '#444444'),
                    xlabel='Reg normalized [0,1]',
                    ylabel='log1p(C_mod_conf)',
                    xlim=(0.0, 1.0),
                    ylim=(0.0, y_hi),
                    y_is_binary=False,
                )
                rows.append({
                    'mark': mark,
                    'scope': 'peak_only_rawM_log',
                    'n': int(peak_mask.sum()),
                    'm_mean': float(m_conf_log.mean()),
                    'spearman_rho': spearman_corr(d['reg01'][peak_mask], m_conf_log),
                    'pearson_r': pearson_corr(d['reg01'][peak_mask], m_conf_log),
                    'mutual_info_bits': mutual_info_bits_continuous_binned(d['reg01'][peak_mask], m_conf_log),
                })

    if mark_data_peak_only:
        plot_overlay_contours(
            path=os.path.join(args.output_dir, 'option1_overlay_contours_peak_only_by_mark.svg'),
            mark_data=mark_data_peak_only,
            x_key='reg01',
            y_key='m_frac',
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel='Reg normalized [0,1] (peak-only)',
            ylabel='Methylation fraction [0,1] (peak-only)',
            title='Option 1: peak-only contour overlay by mark',
            y_is_binary=False,
        )
    if mark_data_peak_only_binary:
        plot_overlay_contours(
            path=os.path.join(args.output_dir, 'option1_overlay_contours_peak_only_binary_by_mark.svg'),
            mark_data=mark_data_peak_only_binary,
            x_key='reg01',
            y_key='m_bin',
            xlim=(0.0, 1.0),
            ylim=(-0.1, 1.1),
            xlabel='Reg normalized [0,1] (peak-only)',
            ylabel='M binary (0/1, jittered for contour)',
            title='Option 1: peak-only binary-M contour overlay by mark',
            y_is_binary=True,
        )
    if mark_data_peak_only_rawm:
        y_max = 0.0
        for mark in mark_data_peak_only_rawm:
            arr = mark_data_peak_only_rawm[mark]['m_conf_log']
            if arr.size:
                y_max = max(y_max, float(np.quantile(arr, 0.995)))
        y_max = max(y_max, 1e-6)
        plot_overlay_contours(
            path=os.path.join(args.output_dir, 'option1_overlay_contours_peak_only_rawM_by_mark.svg'),
            mark_data=mark_data_peak_only_rawm,
            x_key='reg01',
            y_key='m_conf_log',
            xlim=(0.0, 1.0),
            ylim=(0.0, y_max),
            xlabel='Reg normalized [0,1] (peak-only)',
            ylabel='log1p(C_mod_conf)',
            title='Option 1: peak-only raw-M contour overlay by mark',
            y_is_binary=False,
        )

    plot_overlay_contours(
        path=os.path.join(args.output_dir, 'feedback_overlay_topographic_contours_by_mark.svg'),
        mark_data=mark_data,
        x_key='reg01',
        y_key='m_bin',
        xlim=(0.0, 1.0),
        ylim=(-0.1, 1.1),
        xlabel='Reg normalized [0,1]',
        ylabel='M binary (0/1, jittered for contour)',
        title='Topographic contour overlay by histone mark',
        y_is_binary=True,
    )

    # Option 2 overlay
    plot_overlay_contours(
        path=os.path.join(args.output_dir, 'option2_overlay_contours_regNorm_vs_mfrac_by_mark.svg'),
        mark_data=mark_data,
        x_key='reg01',
        y_key='m_frac',
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel='Reg normalized [0,1]',
        ylabel='Methylation fraction [0,1]',
        title='Option 2: contour overlay by mark (Reg normalized vs M fraction)',
        y_is_binary=False,
    )

    # Option 3 overlay
    mark_data_opt3: Dict[str, Dict[str, np.ndarray]] = {}
    max_x = 0.0
    for mark, d in mark_data.items():
        x = np.log1p(d['reg_raw'])
        mark_data_opt3[mark] = {'x': x, 'y': d['m_frac']}
        if x.size:
            max_x = max(max_x, float(np.quantile(x, 0.995)))
    max_x = max(max_x, 1e-6)
    plot_overlay_contours(
        path=os.path.join(args.output_dir, 'option3_overlay_contours_regDensity_vs_mfrac_by_mark.svg'),
        mark_data=mark_data_opt3,
        x_key='x',
        y_key='y',
        xlim=(0.0, max_x),
        ylim=(0.0, 1.0),
        xlabel='log1p(A_mod_density_per_kb)',
        ylabel='Methylation fraction [0,1]',
        title='Option 3: contour overlay by mark (Reg density vs M fraction)',
        y_is_binary=False,
    )

    plot_reg_raw_distributions(
        path=os.path.join(args.output_dir, 'feedback_reg_raw_distribution_by_mark.svg'),
        mark_data=mark_data,
    )
    write_metrics(os.path.join(args.output_dir, 'feedback_correlation_summary.tsv'), rows)

    with open(os.path.join(args.output_dir, 'feedback_notes.md'), 'w', encoding='utf-8') as f:
        f.write('# Feedback-Oriented Phase 1 Visualizations\n\n')
        f.write('- Base: binary M + Reg normalized [0,1].\n')
        f.write('- Option 2: Reg normalized [0,1] vs methylation fraction [0,1].\n')
        f.write('- Option 3: log1p(Reg density) vs methylation fraction [0,1].\n')
        if peaks_idx_by_mark:
            f.write('- Option 1 peak-only plots were generated in three variants: M fraction, binary M, and log1p(raw M).\n')
        else:
            f.write('- Option 1 peak-only plots were skipped (no --peaks-bed-map files provided/found).\n')

    print(f'Wrote feedback outputs to: {args.output_dir}')


if __name__ == '__main__':
    main()
