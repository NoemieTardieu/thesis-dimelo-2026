#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig

DEFAULT_COLORS = ['#3b6c9e', '#cc503e', '#4d9a6e', '#8f63a5', '#d9a441', '#5c5c5c']


def parse_region_sets(entries: List[str]) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for entry in entries:
        if '=' not in entry:
            raise SystemExit(f'Expected LABEL=PATH, got {entry!r}')
        label, path = entry.split('=', 1)
        items.append((label.strip(), Path(path.strip()).expanduser()))
    return items


def load_regions(path: Path, max_regions: int | None) -> List[Tuple[str, int, int, str]]:
    regions = []
    with path.open() as handle:
        for i, line in enumerate(handle):
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.rstrip().split('\t')
            if len(fields) < 3:
                continue
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            name = fields[3] if len(fields) >= 4 else f'region_{i+1}'
            regions.append((chrom, start, end, name))
            if max_regions is not None and len(regions) >= max_regions:
                break
    return regions


def centered_profile_values(bw: pyBigWig.pyBigWig, regions, upstream: int, downstream: int, bin_size: int) -> np.ndarray:
    n_bins = (upstream + downstream) // bin_size
    matrix = np.full((len(regions), n_bins), np.nan, dtype=np.float32)
    for i, (chrom, start, end, _name) in enumerate(regions):
        center = (start + end) // 2
        wstart = max(0, center - upstream)
        wend = center + downstream
        try:
            vals = np.array(bw.stats(chrom, wstart, wend, nBins=n_bins, type='mean', exact=False), dtype=np.float32)
        except RuntimeError:
            continue
        matrix[i, :] = vals
    return np.nanmean(matrix, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description='Plot one bigWig over multiple region sets.')
    p.add_argument('--bigwig', type=Path, required=True)
    p.add_argument('--regions', action='append', required=True, metavar='LABEL=PATH')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--title', default='Average profile by region set')
    p.add_argument('--ylabel', default='Signal')
    p.add_argument('--upstream', type=int, default=5000)
    p.add_argument('--downstream', type=int, default=5000)
    p.add_argument('--bin-size', type=int, default=200)
    p.add_argument('--max-regions', type=int, default=20000)
    args = p.parse_args()

    if not args.bigwig.exists():
        raise FileNotFoundError(args.bigwig)
    if (args.upstream + args.downstream) % args.bin_size != 0:
        raise SystemExit('upstream + downstream must be divisible by bin-size')

    region_sets = parse_region_sets(args.regions)
    x = np.arange(-args.upstream, args.downstream, args.bin_size) + args.bin_size / 2.0
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=180)
    with pyBigWig.open(str(args.bigwig)) as bw:
        for idx, (label, path) in enumerate(region_sets):
            if not path.exists():
                raise FileNotFoundError(path)
            regions = load_regions(path, args.max_regions)
            if not regions:
                continue
            prof = centered_profile_values(bw, regions, args.upstream, args.downstream, args.bin_size)
            ax.plot(x, prof, linewidth=2.2, label=label, color=DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
    ax.axvline(0, color='black', linestyle='--', linewidth=1.2)
    ax.set_xlabel('Position relative to center (bp)')
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)


if __name__ == '__main__':
    main()
