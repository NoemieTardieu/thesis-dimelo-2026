#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_utils import BIN_SIZE, cache_name, load_prediction_cache, load_regions, read_tsv, write_tsv


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cached A549 H3K4me3 predictions and their exact bin grid.")
    parser.add_argument("--regions", type=Path, default=Path("outputs/4chrom_test_regions.tsv"))
    parser.add_argument(
        "--selected-tracks",
        type=Path,
        default=Path("outputs/metadata/selected_a549_h3k4me3_tracks.tsv"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--out", type=Path, default=Path("outputs/alphagenome_test_128bp.tsv"))
    parser.add_argument(
        "--fixed-mean-column",
        default="A549_H3K4me3_fixed_mean",
        help="Name for the arithmetic mean of all selected AlphaGenome tracks.",
    )
    args = parser.parse_args()

    selected = read_tsv(args.selected_tracks)
    ontology_terms = sorted({row["ontology_curie"] for row in selected})
    selected_names = [row["name"] for row in selected]
    rows = []
    track_columns: list[str] | None = None
    regions = load_regions(args.regions, split="test")
    if len(regions) != 60:
        raise SystemExit(f"Expected 60 test regions, found {len(regions)}")
    for region in regions:
        path = args.cache_dir / cache_name(region, ontology_terms)
        values, metadata, provenance = load_prediction_cache(path)
        if int(provenance["resolution"]) != BIN_SIZE:
            raise SystemExit(f"Expected 128 bp AlphaGenome output in {path}")
        returned = provenance["returned_interval"]
        metadata_names = [str(row["name"]) for row in metadata]
        indices = [idx for idx, name in enumerate(metadata_names) if name in selected_names]
        if not indices:
            raise SystemExit(f"No selected A549 tracks in {path}")
        labels = []
        seen: dict[str, int] = {}
        for idx in indices:
            name = metadata_names[idx]
            seen[name] = seen.get(name, 0) + 1
            labels.append(name if metadata_names.count(name) == 1 else f"{name}__track{seen[name]}")
        if track_columns is None:
            track_columns = labels
        elif labels != track_columns:
            raise SystemExit("Selected track order differs across AlphaGenome caches.")

        for bin_index in range(values.shape[0]):
            bin_start = int(returned["start"]) + bin_index * BIN_SIZE
            bin_end = bin_start + BIN_SIZE
            if bin_start < region.start or bin_end > region.end:
                continue
            row = {
                "chrom": region.chrom,
                "region_id": region.region_id,
                "region_name": region.name,
                "region_start": region.start,
                "region_end": region.end,
                "bin_start": bin_start,
                "bin_end": bin_end,
            }
            selected_values = [float(values[bin_index, idx]) for idx in indices]
            row.update(dict(zip(labels, selected_values)))
            row[args.fixed_mean_column] = sum(selected_values) / len(selected_values)
            rows.append(row)

    fields = [
        "chrom", "region_id", "region_name", "region_start", "region_end",
        "bin_start", "bin_end", *(track_columns or []), args.fixed_mean_column,
    ]
    write_tsv(args.out, rows, fields)
    counts = {}
    region_counts = {}
    for row in rows:
        counts[row["chrom"]] = counts.get(row["chrom"], 0) + 1
        region_counts.setdefault(row["chrom"], set()).add(row["region_id"])
    if set(counts) != {"chr16", "chr11", "chr17", "chr19"}:
        raise SystemExit(f"Incomplete chromosome caches: {counts}")
    if {chrom: len(ids) for chrom, ids in region_counts.items()} != {
        "chr16": 15, "chr11": 15, "chr17": 15, "chr19": 15
    }:
        raise SystemExit(f"Expected 15 cached regions per chromosome: {region_counts}")
    print(f"Wrote {len(rows)} aligned AlphaGenome bins to {args.out}")


if __name__ == "__main__":
    main()
