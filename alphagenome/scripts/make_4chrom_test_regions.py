#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from benchmark_utils import CHROMS, load_regions, write_tsv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the authoritative four-chromosome region manifest.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("../outputs"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--out", type=Path, default=Path("outputs/4chrom_test_regions.tsv"))
    args = parser.parse_args()

    regions = []
    all_regions = []
    for chrom in CHROMS:
        source = args.outputs_dir / (
            f"merged_e5b_{chrom}_selected_top100_overlap16k_full5000_"
            "region_split.region_splits.tsv"
        )
        source_regions = load_regions(source)
        all_regions.extend(source_regions)
        regions.extend(region for region in source_regions if region.split == args.split)

    counts = Counter(region.chrom for region in regions)
    expected = 15 if args.split in {"val", "test"} else 70
    if counts != Counter({chrom: expected for chrom in CHROMS}):
        raise SystemExit(f"Unexpected region counts for {args.split}: {dict(counts)}")
    coordinates = {(r.chrom, r.start, r.end) for r in regions}
    if len(coordinates) != len(regions):
        raise SystemExit("Duplicate genomic intervals found.")
    split_by_coordinate = {}
    for region in all_regions:
        coordinate = (region.chrom, region.start, region.end)
        previous = split_by_coordinate.setdefault(coordinate, region.split)
        if previous != region.split:
            raise SystemExit(f"Region split leakage detected at {coordinate}.")

    rows = [
        {
            "region_id": region.region_id,
            "chrom": region.chrom,
            "start": region.start,
            "end": region.end,
            "name": region.name,
            "split": region.split,
        }
        for region in regions
    ]
    write_tsv(args.out, rows, ["region_id", "chrom", "start", "end", "name", "split"])
    print(f"Wrote {len(rows)} {args.split} regions to {args.out}")


if __name__ == "__main__":
    main()
