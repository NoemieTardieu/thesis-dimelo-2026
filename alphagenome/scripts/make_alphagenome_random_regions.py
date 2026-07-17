#!/usr/bin/env python3
"""Create random regions for AlphaGenome-vs-ENCODE sensitivity analysis.

The goal is to evaluate AlphaGenome on regions that were not selected from the
DiMeLo/HyenaDNA ranked region set. The output follows the same region-manifest
schema used by the existing AlphaGenome query code.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def read_fai(path: Path) -> dict[str, int]:
    """Read chromosome lengths from a FASTA index."""

    lengths = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                lengths[fields[0]] = int(fields[1])
    return lengths


def read_avoid_intervals(paths: list[Path], padding: int) -> dict[str, list[tuple[int, int]]]:
    """Read existing region intervals and add optional padding."""

    intervals: dict[str, list[tuple[int, int]]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            columns = {name: idx for idx, name in enumerate(header)}
            required = {"chrom", "start", "end"}
            if not required.issubset(columns):
                continue
            for line in handle:
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                chrom = fields[columns["chrom"]]
                start = max(0, int(fields[columns["start"]]) - padding)
                end = int(fields[columns["end"]]) + padding
                intervals.setdefault(chrom, []).append((start, end))
    for chrom in intervals:
        intervals[chrom].sort()
    return intervals


def overlaps_any(chrom: str, start: int, end: int, avoid: dict[str, list[tuple[int, int]]]) -> bool:
    """Return whether an interval overlaps any avoided interval."""

    for avoid_start, avoid_end in avoid.get(chrom, []):
        if avoid_end <= start:
            continue
        if avoid_start >= end:
            break
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta-index",
        type=Path,
        default=Path("/staging/leuven/stg_00118/antonella/reference/GRCh38.p13.genome.fa.fai"),
    )
    parser.add_argument("--chroms", default="chr11,chr16,chr17,chr19")
    parser.add_argument("--regions-per-chrom", type=int, default=15)
    parser.add_argument("--region-width", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--avoid-regions",
        type=Path,
        action="append",
        default=[
            Path("metadata/regions/4chrom_test_regions.tsv"),
            Path("metadata/regions/4chrom_val_regions.tsv"),
        ],
        help="Region manifest to avoid. Can be passed more than once.",
    )
    parser.add_argument("--avoid-padding", type=int, default=131_072)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("metadata/regions/4chrom_random_sensitivity_regions.tsv"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    lengths = read_fai(args.fasta_index)
    chroms = [chrom.strip() for chrom in args.chroms.split(",") if chrom.strip()]
    missing = [chrom for chrom in chroms if chrom not in lengths]
    if missing:
        raise SystemExit(f"Chromosomes absent from {args.fasta_index}: {missing}")
    avoid = read_avoid_intervals(args.avoid_regions or [], args.avoid_padding)

    rows = []
    for chrom in chroms:
        selected: list[tuple[int, int]] = []
        attempts = 0
        max_start = lengths[chrom] - args.region_width
        if max_start <= 0:
            raise SystemExit(f"Chromosome {chrom} is shorter than requested width {args.region_width}")
        while len(selected) < args.regions_per_chrom:
            attempts += 1
            if attempts > args.regions_per_chrom * 10000:
                raise SystemExit(f"Could not sample enough non-overlapping regions for {chrom}")
            start = rng.randint(0, max_start)
            start = (start // args.region_width) * args.region_width
            end = start + args.region_width
            if overlaps_any(chrom, start, end, avoid):
                continue
            if any(not (end <= old_start or start >= old_end) for old_start, old_end in selected):
                continue
            selected.append((start, end))
        selected.sort()
        for idx, (start, end) in enumerate(selected, start=1):
            rows.append(
                {
                    "region_id": f"random_{chrom}_{idx:02d}",
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "name": f"{chrom}_random{idx:02d}_seed{args.seed}",
                    "split": "test",
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write("region_id\tchrom\tstart\tend\tname\tsplit\n")
        for row in rows:
            handle.write(
                f"{row['region_id']}\t{row['chrom']}\t{row['start']}\t{row['end']}\t{row['name']}\t{row['split']}\n"
            )
    print(f"Wrote {len(rows)} random regions to {args.out}")


if __name__ == "__main__":
    main()
