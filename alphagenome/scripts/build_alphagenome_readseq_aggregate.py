#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from alphagenome_query import selected_terms
from benchmark_utils import Region, RunningStats, load_regions, read_to_reference_map, write_tsv


RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")
SUPPORTED_SEQUENCE_LENGTHS = (16_384, 131_072, 524_288, 1_048_576)
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


def reference_oriented_query_sequence(read: pysam.AlignedSegment) -> str:
    sequence = (read.query_sequence or "").upper()
    if read.is_reverse:
        return sequence.translate(RC_TABLE)[::-1].upper()
    return sequence


def embed_sequence(sequence: str, sequence_length: int) -> tuple[str, int, int]:
    if len(sequence) > sequence_length:
        raise ValueError(
            f"Read length {len(sequence)} exceeds AlphaGenome sequence length {sequence_length}."
        )
    left_pad = (sequence_length - len(sequence)) // 2
    right_pad = sequence_length - len(sequence) - left_pad
    return "N" * left_pad + sequence + "N" * right_pad, left_pad, left_pad + len(sequence)


def complete_bins(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(bin_start, bin_start + size) for bin_start in range(start, end - size + 1, size)]


def read_cache_name(sample: str, read_id: str, sequence: str, ontology_terms: list[str]) -> str:
    digest = hashlib.sha256(
        ("\t".join([sample, read_id, ",".join(ontology_terms), sequence])).encode()
    ).hexdigest()[:16]
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in read_id)[:80]
    return f"{sample}_{clean}_{digest}.npz"


def selected_track_mean(output, selected_names: set[str]) -> tuple[np.ndarray, list[str], int]:
    track = output.chip_histone
    names = track.metadata["name"].astype(str).tolist()
    indices = [idx for idx, name in enumerate(names) if name in selected_names]
    if not indices:
        raise ValueError("No selected A549 H3K4me3 tracks returned by AlphaGenome.")
    values = np.asarray(track.values[:, indices], dtype=float)
    return values.mean(axis=1), [names[idx] for idx in indices], int(track.resolution)


def predict_with_retries(fn, retries: int, label: str):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            delay = min(60.0, 2.0 ** (attempt - 1))
            print(f"Retrying {label} after attempt {attempt} in {delay:.1f}s: {exc}")
            time.sleep(delay)
    raise RuntimeError(f"Failed {label} after {retries} attempts") from last_error


def load_or_query_track(
    model,
    dna_client,
    cache_dir: Path,
    sample: str,
    read_id: str,
    sequence: str,
    ontology_terms: list[str],
    selected_names: set[str],
    retries: int,
) -> tuple[np.ndarray, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / read_cache_name(sample, read_id, sequence, ontology_terms)
    if cache_path.exists():
        archive = np.load(cache_path, allow_pickle=True)
        return archive["track"].astype(float), int(archive["resolution"])

    output = predict_with_retries(
        lambda: model.predict_sequence(
            sequence=sequence,
            requested_outputs=[dna_client.OutputType.CHIP_HISTONE],
            ontology_terms=ontology_terms,
        ),
        retries,
        f"read sequence {sample}:{read_id}",
    )
    track, returned_tracks, resolution = selected_track_mean(output, selected_names)
    np.savez_compressed(
        cache_path,
        track=track.astype(np.float32),
        resolution=np.array(resolution, dtype=np.int32),
        provenance_json=np.array(
            json.dumps(
                {
                    "sample": sample,
                    "read_id": read_id,
                    "ontology_terms": ontology_terms,
                    "selected_tracks": returned_tracks,
                    "sequence_length": len(sequence),
                    "resolution": resolution,
                }
            )
        ),
    )
    return track, resolution


def collect_region_reads(
    bam: pysam.AlignmentFile,
    region: Region,
    max_reads: int,
    min_mapq: int,
) -> list[pysam.AlignedSegment]:
    reads = []
    seen = set()
    for read in bam.fetch(region.chrom, region.start, region.end):
        if len(reads) >= max_reads:
            break
        if (
            read.is_unmapped
            or read.is_secondary
            or read.is_supplementary
            or read.mapping_quality < min_mapq
            or read.query_sequence is None
            or read.query_length is None
            or read.query_name in seen
        ):
            continue
        reads.append(read)
        seen.add(read.query_name)
    return reads


def add_read_track_to_stats(
    stats: dict[tuple[str, str, int], RunningStats],
    counters: Counter,
    sample: str,
    region: Region,
    read: pysam.AlignedSegment,
    track: np.ndarray,
    resolution: int,
    left_pad: int,
    bin_size: int,
) -> None:
    read_length = int(read.query_length or 0)
    all_forward_positions = set(range(read_length))
    mapping = read_to_reference_map(read, all_forward_positions)
    counters["mapped_read_positions"] += len(mapping)
    counters["unmapped_read_positions"] += read_length - len(mapping)
    if not mapping:
        return
    bins = complete_bins(region.start, region.end, bin_size)
    valid_starts = {start for start, _ in bins}
    if not bins:
        return
    grid_start = bins[0][0]
    read_id = read.query_name
    for output_index, value in enumerate(track):
        seq_start = output_index * resolution
        seq_end = seq_start + resolution
        read_start = max(0, seq_start - left_pad)
        read_end = min(read_length, seq_end - left_pad)
        if read_start >= read_end:
            counters["alphagenome_bins_in_padding"] += 1
            continue
        for read_pos in range(read_start, read_end):
            reference_pos = mapping.get(read_pos)
            if reference_pos is None:
                continue
            if not (region.start <= reference_pos < region.end):
                continue
            bin_index = (reference_pos - grid_start) // bin_size
            bin_start = grid_start + bin_index * bin_size
            if bin_start not in valid_starts:
                continue
            stats[(sample, region.key, bin_start)].update(float(value), read_id)
            stats[("pooled", region.key, bin_start)].update(float(value), f"{sample}:{read_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AlphaGenome on original read sequences and aggregate predictions to genomic bins."
    )
    parser.add_argument("--regions", type=Path, default=Path("outputs/4chrom_test_regions.tsv"))
    parser.add_argument("--selected-tracks", type=Path, default=Path("outputs/metadata/selected_a549_h3k4me3_tracks.tsv"))
    parser.add_argument("--bam-c1", type=Path, default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam"))
    parser.add_argument("--bam-e5b", type=Path, default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam"))
    parser.add_argument("--out", type=Path, default=Path("outputs/alphagenome_readseq_200bp.tsv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/alphagenome_readseq_200bp.summary.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache_readseq"))
    parser.add_argument("--sequence-length", type=int, default=131_072)
    parser.add_argument("--bin-size", type=int, default=200)
    parser.add_argument("--max-reads-per-region-per-sample", type=int, default=2)
    parser.add_argument("--min-mapq", type=int, default=20)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise SystemExit("ALPHAGENOME_API_KEY is not set.")
    if args.sequence_length not in SUPPORTED_SEQUENCE_LENGTHS:
        raise SystemExit(f"Unsupported sequence length: {args.sequence_length}")

    from alphagenome.models import dna_client

    selected_metadata, ontology_terms = selected_terms(args.selected_tracks)
    selected_names = set(selected_metadata["name"].astype(str))
    regions = load_regions(args.regions, split="test")
    model = dna_client.create(api_key)
    bam_paths = {"merged_c1": args.bam_c1, "merged_e5b": args.bam_e5b}
    stats: dict[tuple[str, str, int], RunningStats] = defaultdict(RunningStats)
    counters: Counter = Counter()

    for sample, bam_path in bam_paths.items():
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for region in regions:
                reads = collect_region_reads(
                    bam, region, args.max_reads_per_region_per_sample, args.min_mapq
                )
                counters[f"{sample}_selected_reads"] += len(reads)
                for read in reads:
                    sequence = reference_oriented_query_sequence(read)
                    if len(sequence) > args.sequence_length:
                        counters["reads_longer_than_sequence_length_skipped"] += 1
                        continue
                    embedded, left_pad, _ = embed_sequence(sequence, args.sequence_length)
                    print(
                        f"querying {sample}:{read.query_name} {region.key} "
                        f"read_len={len(sequence)}"
                    )
                    track, resolution = load_or_query_track(
                        model,
                        dna_client,
                        args.cache_dir,
                        sample,
                        read.query_name,
                        embedded,
                        ontology_terms,
                        selected_names,
                        args.retries,
                    )
                    add_read_track_to_stats(
                        stats,
                        counters,
                        sample,
                        region,
                        read,
                        track,
                        resolution,
                        left_pad,
                        args.bin_size,
                    )
                    counters["read_sequence_predictions_used"] += 1

    rows = []
    samples = list(bam_paths) + ["pooled"]
    for region in regions:
        for sample in samples:
            for bin_start, bin_end in complete_bins(region.start, region.end, args.bin_size):
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
    write_tsv(args.out, rows, TRACK_FIELDS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {
                "out": str(args.out),
                "regions": str(args.regions),
                "bin_size": args.bin_size,
                "sequence_length": args.sequence_length,
                "max_reads_per_region_per_sample": args.max_reads_per_region_per_sample,
                "note": (
                    "AlphaGenome was run on original BAM query sequences. Its read-coordinate "
                    "output bins were expanded across their covered read positions, mapped "
                    "through the CIGAR to reference coordinates, and aggregated into genomic bins."
                ),
                "counters": dict(counters),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
