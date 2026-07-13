#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmark_utils import CHROMS, collapse_windows, load_regions, read_tsv
from reference_tracks import build_reference_track


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CIGAR-aware 128 bp DiMeLo 6mA tracks.")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, default=Path("../outputs"))
    parser.add_argument(
        "--bam-c1",
        type=Path,
        default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam"),
    )
    parser.add_argument(
        "--bam-e5b",
        type=Path,
        default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam"),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--grid", type=Path, help="Optional AlphaGenome bin grid TSV for exact test alignment.")
    args = parser.parse_args()

    regions = load_regions(args.regions, split=args.split)
    grid_by_region = None
    if args.grid:
        grid_by_region = {}
        for row in read_tsv(args.grid):
            key = f"{row['chrom']}:{row['region_start']}-{row['region_end']}"
            grid_by_region.setdefault(key, []).append((int(row["bin_start"]), int(row["bin_end"])))
    out = args.out or Path(f"outputs/dimelo_{args.split}_128bp.tsv")
    per_chrom_outputs = []
    for chrom in CHROMS:
        chrom_regions = [region for region in regions if region.chrom == chrom]
        prefix = (
            args.outputs_dir
            / f"merged_e5b_c1_{chrom}_selected_top100_overlap16k_full5000_region_split.{args.split}"
        )
        metadata_path = Path(f"{prefix}.metadata.tsv")
        npz_path = Path(f"{prefix}.npz")
        metadata = read_tsv(metadata_path)
        archive = np.load(npz_path)
        targets = archive["target_6mA"]
        masks = archive["mask_6mA"].astype(bool)
        if len(metadata) != targets.shape[0]:
            raise SystemExit(f"Metadata/tensor row mismatch for {chrom}")

        def getter(indices: list[int]) -> tuple[dict[int, float], int]:
            return collapse_windows(targets, masks, metadata, indices)

        chrom_out = out.with_name(f"{out.stem}.{chrom}{out.suffix}")
        build_reference_track(
            metadata_path,
            chrom_regions,
            {"merged_c1": args.bam_c1, "merged_e5b": args.bam_e5b},
            getter,
            chrom_out,
            chrom_out.with_suffix(".summary.json"),
            grid_by_region,
        )
        per_chrom_outputs.append(chrom_out)
        archive.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as destination:
        for index, path in enumerate(per_chrom_outputs):
            with open(path, "r", encoding="utf-8") as source:
                for line_number, line in enumerate(source):
                    if index and line_number == 0:
                        continue
                    destination.write(line)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
