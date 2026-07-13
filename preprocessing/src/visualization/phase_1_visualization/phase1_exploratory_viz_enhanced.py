import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_per_mark(tsv_path: str) -> Dict[str, np.ndarray]:
    cols = {
        'read_id': [],
        'chrom': [],
        'ref_start': [],
        'ref_end': [],
        'read_len': [],
        'a_density': [],
        'c_density': [],
    }

    with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cols['read_id'].append(row['read_id'])
            cols['chrom'].append(row['chrom'])
            cols['ref_start'].append(int(row['ref_start']))
            cols['ref_end'].append(int(row['ref_end']))
            cols['read_len'].append(float(row['read_len']))
            cols['a_density'].append(float(row['A_mod_density_per_kb']))
            cols['c_density'].append(float(row['C_mod_density_per_kb']))

    return {
        'read_id': np.asarray(cols['read_id'], dtype=object),
        'chrom': np.asarray(cols['chrom'], dtype=object),
        'ref_start': np.asarray(cols['ref_start'], dtype=np.int64),
        'ref_end': np.asarray(cols['ref_end'], dtype=np.int64),
        'read_len': np.asarray(cols['read_len'], dtype=np.float64),
        'a_density': np.asarray(cols['a_density'], dtype=np.float64),
        'c_density': np.asarray(cols['c_density'], dtype=np.float64),
    }


def quantile_edges(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    return edges


def binned_counts(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = quantile_edges(x, n_bins)
    y_edges = quantile_edges(y, n_bins)

    xi = np.digitize(x, x_edges[1:-1], right=False)
    yi = np.digitize(y, y_edges[1:-1], right=False)

    mat = np.zeros((n_bins, n_bins), dtype=np.int64)
    for a, b in zip(xi, yi):
        mat[b, a] += 1
    return mat, x_edges, y_edges


def enrichment_log2(mat: np.ndarray, pseudocount: float = 1e-9) -> np.ndarray:
    total = float(mat.sum())
    if total <= 0:
        return np.zeros_like(mat, dtype=np.float64)

    p_ij = mat / total
    p_i = p_ij.sum(axis=0, keepdims=True)
    p_j = p_ij.sum(axis=1, keepdims=True)
    expected = p_j @ p_i
    return np.log2((p_ij + pseudocount) / (expected + pseudocount))


def plot_enrichment_heatmap(
    out_path: str,
    enr: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    annotate: bool = False,
) -> None:
    vmax = np.nanpercentile(np.abs(enr), 99)
    vmax = max(vmax, 0.25)

    fig, ax = plt.subplots(figsize=(8.0, 6.2), dpi=150)
    im = ax.imshow(enr, origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(enr.shape[1]))
    ax.set_yticks(np.arange(enr.shape[0]))
    ax.set_xticklabels([f'Q{i+1}' for i in range(enr.shape[1])], rotation=45, ha='right')
    ax.set_yticklabels([f'Q{i+1}' for i in range(enr.shape[0])])

    if annotate:
        for r in range(enr.shape[0]):
            for c in range(enr.shape[1]):
                ax.text(c, r, f'{enr[r, c]:.2f}', ha='center', va='center', fontsize=6)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('log2(observed / expected)')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def gaussian_smooth_2d(arr: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    radius = int(max(2, round(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (xs / sigma) ** 2)
    k /= k.sum()

    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=0, arr=arr)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), axis=1, arr=out)
    return out


def plot_joint_contours(
    out_path: str,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    n_bins: int = 80,
    sample_scatter: int = 6000,
) -> None:
    x_log = np.log1p(x)
    y_log = np.log1p(y)

    H, x_edges, y_edges = np.histogram2d(x_log, y_log, bins=n_bins)
    Hs = gaussian_smooth_2d(H, sigma=1.3)

    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_cent, y_cent, indexing='xy')

    positive = Hs[Hs > 0]
    if positive.size == 0:
        levels = np.array([1.0])
    else:
        levels = np.quantile(positive, [0.70, 0.85, 0.93, 0.97])
        levels = np.unique(levels)

    fig, ax = plt.subplots(figsize=(8.0, 6.2), dpi=150)

    if sample_scatter > 0 and x_log.size > 0:
        if x_log.size > sample_scatter:
            rng = np.random.default_rng(7)
            idx = rng.choice(x_log.size, size=sample_scatter, replace=False)
            xs = x_log[idx]
            ys = y_log[idx]
        else:
            xs = x_log
            ys = y_log
        ax.scatter(xs, ys, s=5, alpha=0.12, color='#5f6c80', linewidths=0)

    if levels.size > 0:
        cs = ax.contour(X, Y, Hs.T, levels=levels, colors=['#08306b', '#2171b5', '#f16913', '#cb181d'], linewidths=1.5)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.2g')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_enrichment_by_mark(out_path: str, mark_to_enr: Dict[str, np.ndarray]) -> None:
    marks = sorted(mark_to_enr.keys())
    n = len(marks)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), dpi=150, squeeze=False)
    axes = axes[0]

    vmax = max(max(np.nanpercentile(np.abs(mark_to_enr[m]), 99), 0.25) for m in marks)

    im = None
    for ax, mark in zip(axes, marks):
        enr = mark_to_enr[mark]
        im = ax.imshow(enr, origin='lower', cmap='bwr', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(mark)
        ax.set_xlabel('Reg quantile (A density)')
        ax.set_ylabel('M quantile (C density)')
        ax.set_xticks(np.arange(enr.shape[1]))
        ax.set_yticks(np.arange(enr.shape[0]))
        ax.set_xticklabels([f'Q{i+1}' for i in range(enr.shape[1])], rotation=45, ha='right')
        ax.set_yticklabels([f'Q{i+1}' for i in range(enr.shape[0])])

    fig.subplots_adjust(right=0.90, wspace=0.26)
    cax = fig.add_axes([0.915, 0.17, 0.012, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('log2(observed / expected)')
    fig.suptitle('Reg vs M Enrichment Heatmaps by Mark', y=1.03)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_joint_contours_by_mark(out_path: str, mark_data: Dict[str, Dict[str, np.ndarray]]) -> None:
    marks = sorted(mark_data.keys())
    n = len(marks)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), dpi=150, squeeze=False)
    axes = axes[0]

    for ax, mark in zip(axes, marks):
        x = mark_data[mark]['a_density']
        y = mark_data[mark]['c_density']
        x_log = np.log1p(x)
        y_log = np.log1p(y)

        H, x_edges, y_edges = np.histogram2d(x_log, y_log, bins=70)
        Hs = gaussian_smooth_2d(H, sigma=1.2)

        x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
        X, Y = np.meshgrid(x_cent, y_cent, indexing='xy')
        positive = Hs[Hs > 0]
        levels = np.quantile(positive, [0.70, 0.85, 0.93, 0.97]) if positive.size else np.array([1.0])
        levels = np.unique(levels)

        if x_log.size > 4000:
            rng = np.random.default_rng(11)
            idx = rng.choice(x_log.size, size=4000, replace=False)
            xs, ys = x_log[idx], y_log[idx]
        else:
            xs, ys = x_log, y_log
        ax.scatter(xs, ys, s=4, alpha=0.12, color='#6b7280', linewidths=0)

        if levels.size > 0:
            ax.contour(X, Y, Hs.T, levels=levels, colors=['#08306b', '#2171b5', '#f16913', '#cb181d'], linewidths=1.2)

        ax.set_title(mark)
        ax.set_xlabel('log1p(Reg proxy: A density)')
        ax.set_ylabel('log1p(M proxy: C density)')

    fig.suptitle('Joint Density Contours by Mark', y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# --------------------- viz3 upgrades ---------------------

def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    sorted_x = x[order]
    n = x.size
    ranks_sorted = np.zeros(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks_sorted[i:j] = avg_rank
        i = j

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float('nan')
    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    x0 = rx - rx.mean()
    y0 = ry - ry.mean()
    denom = np.sqrt((x0 * x0).sum() * (y0 * y0).sum())
    if denom <= 0:
        return float('nan')
    return float((x0 * y0).sum() / denom)


def mutual_information_quantile(x: np.ndarray, y: np.ndarray, n_bins: int = 20, pseudocount: float = 1e-12) -> float:
    if x.size == 0:
        return float('nan')
    mat, _, _ = binned_counts(x, y, n_bins=n_bins)
    pxy = mat.astype(np.float64)
    pxy /= max(pxy.sum(), 1.0)
    px = pxy.sum(axis=0, keepdims=True)
    py = pxy.sum(axis=1, keepdims=True)
    expected = py @ px

    nz = pxy > 0
    return float((pxy[nz] * np.log2((pxy[nz] + pseudocount) / (expected[nz] + pseudocount))).sum())


def write_dependence_metrics(path: str, rows: List[Dict[str, object]]) -> None:
    fields = ['scope', 'mark', 'subset', 'n_reads', 'spearman_rho', 'mutual_info_bits']
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def add_metric_rows(rows: List[Dict[str, object]], scope: str, mark: str, subset: str, x: np.ndarray, y: np.ndarray) -> None:
    rows.append({
        'scope': scope,
        'mark': mark,
        'subset': subset,
        'n_reads': int(x.size),
        'spearman_rho': spearman_corr(x, y),
        'mutual_info_bits': mutual_information_quantile(x, y, n_bins=20),
    })


def plot_joint_contours_two_groups(
    out_path: str,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    t1: str,
    t2: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=150)

    for ax, x, y, tt in [(axes[0], x1, y1, t1), (axes[1], x2, y2, t2)]:
        x_log = np.log1p(x)
        y_log = np.log1p(y)

        if x_log.size > 0:
            H, x_edges, y_edges = np.histogram2d(x_log, y_log, bins=70)
            Hs = gaussian_smooth_2d(H, sigma=1.2)
            x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
            X, Y = np.meshgrid(x_cent, y_cent, indexing='xy')
            pos = Hs[Hs > 0]
            levels = np.quantile(pos, [0.70, 0.85, 0.93, 0.97]) if pos.size else np.array([1.0])
            levels = np.unique(levels)

            if x_log.size > 5000:
                rng = np.random.default_rng(17)
                idx = rng.choice(x_log.size, size=5000, replace=False)
                xs, ys = x_log[idx], y_log[idx]
            else:
                xs, ys = x_log, y_log
            ax.scatter(xs, ys, s=4, alpha=0.10, color='#6b7280', linewidths=0)
            if levels.size > 0:
                ax.contour(X, Y, Hs.T, levels=levels, colors=['#08306b', '#2171b5', '#f16913', '#cb181d'], linewidths=1.2)

        ax.set_title(tt)
        ax.set_xlabel('log1p(Reg proxy: A density)')
        ax.set_ylabel('log1p(M proxy: C density)')

    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def parse_bed(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    chrom_to_intervals: Dict[str, List[Tuple[int, int]]] = {}
    with open(path, 'r', encoding='utf-8') as f:
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


def infer_peak_file(peaks_dir: str, mark: str) -> Optional[str]:
    patterns = [
        f'{mark}*.bed', f'{mark}*.narrowPeak', f'{mark}*.broadPeak',
        f'*{mark}*.bed', f'*{mark}*.narrowPeak', f'*{mark}*.broadPeak',
    ]
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(peaks_dir, pat)))
        if hits:
            return hits[0]
    return None


def write_notes(path: str, notes: List[str]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Enhanced Phase 1 Visualizations\n\n')
        for line in notes:
            f.write(f'- {line}\n')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate enhanced Phase 1 visualizations (new files, old preserved).')
    p.add_argument(
        '--input-dir',
        default='/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase1_output',
        help='Folder containing *.per_read.tsv files.',
    )
    p.add_argument(
        '--output-dir',
        default='/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase_1_visualization/output_enhanced',
        help='Output folder for enhanced figures.',
    )
    p.add_argument(
        '--long-read-threshold',
        type=float,
        default=20000.0,
        help='Read length threshold for long-read stratification (bp).',
    )
    p.add_argument(
        '--peaks-dir',
        default='',
        help='Optional folder with peak BED files (one file per mark, auto-matched by mark name).',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tsv_files = sorted(glob.glob(os.path.join(args.input_dir, '*.per_read.tsv')))
    if not tsv_files:
        raise SystemExit(f'No .per_read.tsv files found in: {args.input_dir}')

    mark_data: Dict[str, Dict[str, np.ndarray]] = {}
    for tsv in tsv_files:
        mark = os.path.basename(tsv).replace('.per_read.tsv', '')
        mark_data[mark] = read_per_mark(tsv)

    marks = sorted(mark_data.keys())
    all_a = np.concatenate([mark_data[m]['a_density'] for m in marks])
    all_c = np.concatenate([mark_data[m]['c_density'] for m in marks])
    all_len = np.concatenate([mark_data[m]['read_len'] for m in marks])

    # viz2 (keep existing files for backward comparison)
    pooled_counts, _, _ = binned_counts(all_a, all_c, n_bins=10)
    pooled_enr = enrichment_log2(pooled_counts)
    plot_enrichment_heatmap(
        out_path=os.path.join(args.output_dir, 'viz2_reg_vs_m_enrichment_heatmap_pooled.svg'),
        enr=pooled_enr,
        title='Reg vs M Enrichment Heatmap (Observed / Expected)',
        xlabel='Reg quantile bin (A density)',
        ylabel='M quantile bin (C density)',
        annotate=False,
    )

    mark_to_enr: Dict[str, np.ndarray] = {}
    for mark, d in mark_data.items():
        counts, _, _ = binned_counts(d['a_density'], d['c_density'], n_bins=10)
        mark_to_enr[mark] = enrichment_log2(counts)
    plot_enrichment_by_mark(
        out_path=os.path.join(args.output_dir, 'viz2_reg_vs_m_enrichment_heatmap_by_mark.svg'),
        mark_to_enr=mark_to_enr,
    )

    plot_joint_contours(
        out_path=os.path.join(args.output_dir, 'viz2_reg_vs_m_joint_contours_pooled.svg'),
        x=all_a,
        y=all_c,
        title='Joint Contour Map: log1p(Reg) vs log1p(M) (Pooled)',
        xlabel='log1p(Reg proxy: A_mod_density_per_kb)',
        ylabel='log1p(M proxy: C_mod_density_per_kb)',
    )

    plot_joint_contours_by_mark(
        out_path=os.path.join(args.output_dir, 'viz2_reg_vs_m_joint_contours_by_mark.svg'),
        mark_data=mark_data,
    )

    # viz3-A: scalar dependence metrics
    metrics_rows: List[Dict[str, object]] = []
    add_metric_rows(metrics_rows, 'pooled', 'all', 'all_reads', all_a, all_c)
    for mark in marks:
        add_metric_rows(metrics_rows, 'per_mark', mark, 'all_reads', mark_data[mark]['a_density'], mark_data[mark]['c_density'])

    # viz3-B: long-read stratification
    long_mask_all = all_len > args.long_read_threshold
    add_metric_rows(metrics_rows, 'pooled', 'all', f'read_len_gt_{int(args.long_read_threshold)}', all_a[long_mask_all], all_c[long_mask_all])

    for mark in marks:
        mlen = mark_data[mark]['read_len']
        mmask = mlen > args.long_read_threshold
        add_metric_rows(
            metrics_rows,
            'per_mark',
            mark,
            f'read_len_gt_{int(args.long_read_threshold)}',
            mark_data[mark]['a_density'][mmask],
            mark_data[mark]['c_density'][mmask],
        )

    write_dependence_metrics(
        os.path.join(args.output_dir, 'viz3_reg_vs_m_dependence_metrics.tsv'),
        metrics_rows,
    )

    plot_joint_contours_two_groups(
        out_path=os.path.join(args.output_dir, 'viz3_reg_vs_m_joint_contours_longread_vs_all.svg'),
        x1=all_a,
        y1=all_c,
        x2=all_a[long_mask_all],
        y2=all_c[long_mask_all],
        t1='All reads',
        t2=f'Reads > {int(args.long_read_threshold)} bp',
        title='Reg vs M Joint Contours: All Reads vs Long Reads',
    )

    # viz3-C: peak-restricted view (if peaks are available)
    notes: List[str] = [
        'viz2 outputs preserved (same filenames) for direct comparison with previous version.',
        'Added viz3-A scalar metrics table: Spearman rho and mutual information.',
        f'Added viz3-B long-read stratification using threshold > {int(args.long_read_threshold)} bp.',
    ]

    if args.peaks_dir and os.path.isdir(args.peaks_dir):
        peak_rows_added = 0
        pooled_in_a: List[np.ndarray] = []
        pooled_in_c: List[np.ndarray] = []
        pooled_out_a: List[np.ndarray] = []
        pooled_out_c: List[np.ndarray] = []

        for mark in marks:
            peak_file = infer_peak_file(args.peaks_dir, mark)
            if not peak_file:
                notes.append(f'No peak file found for mark {mark} in {args.peaks_dir}.')
                continue

            pidx = parse_bed(peak_file)
            d = mark_data[mark]
            in_peak = overlap_mask_reads(d['chrom'], d['ref_start'], d['ref_end'], pidx)
            out_peak = ~in_peak

            add_metric_rows(metrics_rows, 'per_mark', mark, 'in_peak', d['a_density'][in_peak], d['c_density'][in_peak])
            add_metric_rows(metrics_rows, 'per_mark', mark, 'out_peak', d['a_density'][out_peak], d['c_density'][out_peak])

            pooled_in_a.append(d['a_density'][in_peak])
            pooled_in_c.append(d['c_density'][in_peak])
            pooled_out_a.append(d['a_density'][out_peak])
            pooled_out_c.append(d['c_density'][out_peak])
            peak_rows_added += 2

        if peak_rows_added > 0:
            all_in_a = np.concatenate([x for x in pooled_in_a if x.size > 0]) if any(x.size > 0 for x in pooled_in_a) else np.array([], dtype=np.float64)
            all_in_c = np.concatenate([x for x in pooled_in_c if x.size > 0]) if any(x.size > 0 for x in pooled_in_c) else np.array([], dtype=np.float64)
            all_out_a = np.concatenate([x for x in pooled_out_a if x.size > 0]) if any(x.size > 0 for x in pooled_out_a) else np.array([], dtype=np.float64)
            all_out_c = np.concatenate([x for x in pooled_out_c if x.size > 0]) if any(x.size > 0 for x in pooled_out_c) else np.array([], dtype=np.float64)

            add_metric_rows(metrics_rows, 'pooled', 'all', 'in_peak', all_in_a, all_in_c)
            add_metric_rows(metrics_rows, 'pooled', 'all', 'out_peak', all_out_a, all_out_c)

            plot_joint_contours_two_groups(
                out_path=os.path.join(args.output_dir, 'viz3_reg_vs_m_joint_contours_peak_vs_nonpeak.svg'),
                x1=all_in_a,
                y1=all_in_c,
                x2=all_out_a,
                y2=all_out_c,
                t1='Peak-overlapping reads',
                t2='Non-peak reads',
                title='Reg vs M Joint Contours: Peak vs Non-peak',
            )
            notes.append('Added viz3-C peak vs non-peak contour comparison and corresponding metrics rows.')
        else:
            notes.append('Peak directory provided but no usable mark-matched peak files found; viz3-C skipped.')
    else:
        notes.append('No peaks directory provided; viz3-C (peak vs non-peak) skipped.')

    # Rewrite metrics after potential peak rows were added
    write_dependence_metrics(
        os.path.join(args.output_dir, 'viz3_reg_vs_m_dependence_metrics.tsv'),
        metrics_rows,
    )

    write_notes(os.path.join(args.output_dir, 'viz3_enhanced_notes.md'), notes)
    print(f'Wrote enhanced visualizations to: {args.output_dir}')


if __name__ == '__main__':
    main()
