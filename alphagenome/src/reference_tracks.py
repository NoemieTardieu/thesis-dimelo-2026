#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pysam

from benchmark_utils import (
    BIN_SIZE,
    Region,
    RunningStats,
    complete_bins,
    find_primary_alignments,
    group_metadata,
    json_dump,
    read_to_reference_map,
    read_tsv,
    write_tsv,
)

ObservationGetter = Callable[[list[int]], tuple[dict[int, float], int]]
ReferenceTransform = Callable[[pysam.AlignedSegment, int], int | None]

TRACK_FIELDS = [
    "split",
    "chrom",
    "region_id",
    "region_name",
    "region_start",
    "region_end",
    "bin_start",
    "bin_end",
    "sample",
    "mean_signal",
    "positive_fraction_0_5",
    "unique_reads",
    "observed_positions",
    "variance",
    "standard_error",
]


def build_reference_track(
    metadata_path: Path,
    regions: list[Region],
    bam_paths: dict[str, Path],
    get_observations: ObservationGetter,
    out_path: Path,
    summary_path: Path,
    grid_by_region: dict[str, list[tuple[int, int]]] | None = None,
    bin_size: int = BIN_SIZE,
    reference_transform: ReferenceTransform | None = None,
) -> None:
    metadata = read_tsv(metadata_path)
    groups = group_metadata(metadata)
    regions_by_key = {(r.chrom, r.region_id): r for r in regions}
    groups_by_region_sample: dict[tuple[str, str, str], list[tuple]] = defaultdict(list)
    for key, row_indices in groups.items():
        sample, _, chrom, region_id = key
        groups_by_region_sample[(sample, chrom, region_id)].append((key, row_indices))

    stats: dict[tuple[str, str, int], RunningStats] = defaultdict(RunningStats)
    counters: Counter = Counter()
    bam_handles = {sample: pysam.AlignmentFile(path, "rb") for sample, path in bam_paths.items()}
    try:
        for (sample, chrom, region_id), read_groups in groups_by_region_sample.items():
            region = regions_by_key.get((chrom, region_id))
            if region is None:
                counters["metadata_groups_outside_manifest"] += len(read_groups)
                continue
            bam = bam_handles.get(sample)
            if bam is None:
                raise ValueError(f"No BAM configured for sample {sample!r}")
            read_ids = {key[1] for key, _ in read_groups}
            alignments = find_primary_alignments(
                bam, chrom, region.start, region.end, read_ids
            )
            region_bins = (
                grid_by_region.get(region.key, [])
                if grid_by_region is not None
                else list(complete_bins(region.start, region.end, bin_size))
            )
            valid_starts = {start for start, _ in region_bins}
            grid_start = region_bins[0][0] if region_bins else region.start
            counters["selected_reads"] += len(read_ids)
            counters["primary_alignments_found"] += len(alignments)

            for key, row_indices in read_groups:
                _, read_id, _, _ = key
                alignment = alignments.get(read_id)
                if alignment is None:
                    counters["reads_without_primary_alignment"] += 1
                    continue
                observations, duplicates = get_observations(row_indices)
                mapping = read_to_reference_map(alignment, set(observations))
                counters["raw_overlap_duplicates_removed"] += duplicates
                counters["unique_read_positions"] += len(observations)
                counters["positions_mapped_to_reference"] += len(mapping)
                counters["positions_unmapped"] += len(observations) - len(mapping)
                for read_pos, reference_pos in mapping.items():
                    if reference_transform is not None:
                        reference_pos = reference_transform(alignment, reference_pos)
                        if reference_pos is None:
                            counters["positions_dropped_by_reference_transform"] += 1
                            continue
                    if not (region.start <= reference_pos < region.end):
                        counters["mapped_positions_outside_region"] += 1
                        continue
                    if not region_bins:
                        counters["regions_without_grid"] += 1
                        continue
                    bin_index = (reference_pos - grid_start) // bin_size
                    bin_start = grid_start + bin_index * bin_size
                    if bin_start not in valid_starts:
                        counters["positions_in_partial_edge_bin"] += 1
                        continue
                    value = observations[read_pos]
                    stats[(sample, region.key, bin_start)].update(value, read_id)
                    stats[("pooled", region.key, bin_start)].update(
                        value, f"{sample}:{read_id}"
                    )
    finally:
        for bam in bam_handles.values():
            bam.close()

    rows = []
    samples = list(bam_paths) + ["pooled"]
    for region in regions:
        region_bins = (
            grid_by_region.get(region.key, [])
            if grid_by_region is not None
            else list(complete_bins(region.start, region.end, bin_size))
        )
        for sample in samples:
            for bin_start, bin_end in region_bins:
                row = {
                    "split": region.split,
                    "chrom": region.chrom,
                    "region_id": region.region_id,
                    "region_name": region.name,
                    "region_start": region.start,
                    "region_end": region.end,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "sample": sample,
                }
                row.update(stats[(sample, region.key, bin_start)].row())
                rows.append(row)
    write_tsv(out_path, rows, TRACK_FIELDS)
    json_dump(
        summary_path,
        {
            "metadata": str(metadata_path),
            "regions": len(regions),
            "grid_source": "explicit" if grid_by_region is not None else "region_start_anchored",
            "bin_size": bin_size,
            "samples": samples,
            "counters": dict(counters),
            "output": str(out_path),
        },
    )
