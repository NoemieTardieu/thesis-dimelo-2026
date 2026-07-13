#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from alphagenome_query import selected_terms
from benchmark_utils import Region, load_regions


RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reference_oriented_query_sequence(read: pysam.AlignedSegment) -> str:
    sequence = (read.query_sequence or "").upper()
    if read.is_reverse:
        return sequence.translate(RC_TABLE)[::-1].upper()
    return sequence


def centered_supported_sequence(sequence: str, center: int, length: int) -> tuple[str, int, int]:
    center = max(0, min(center, len(sequence)))
    start = center - length // 2
    end = start + length
    left_pad = max(0, -start)
    right_pad = max(0, end - len(sequence))
    start = max(0, start)
    end = min(len(sequence), end)
    window = "N" * left_pad + sequence[start:end] + "N" * right_pad
    if len(window) != length:
        raise ValueError(f"Internal sequence padding error: expected {length}, got {len(window)}")
    return window, start, end


def query_pos_for_reference_center(read: pysam.AlignedSegment, reference_center: int) -> int:
    read_length = int(read.query_length or 0)
    best_query = None
    best_distance = None
    for query_pos, reference_pos in read.get_aligned_pairs(matches_only=False):
        if query_pos is None or reference_pos is None:
            continue
        distance = abs(int(reference_pos) - reference_center)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_query = int(query_pos)
    if best_query is None:
        return read_length // 2
    if read.is_reverse:
        return read_length - 1 - best_query
    return best_query


def safe_track_mean(output, selected_names: set[str]) -> tuple[np.ndarray, list[str], int]:
    track = output.chip_histone
    metadata = track.metadata
    names = metadata["name"].astype(str).tolist()
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
        except Exception as exc:  # AlphaGenome client raises transient RPC exceptions.
            last_error = exc
            if attempt == retries:
                break
            delay = min(60.0, 2.0 ** (attempt - 1))
            print(f"Retrying {label} after attempt {attempt} in {delay:.1f}s: {exc}")
            time.sleep(delay)
    raise RuntimeError(f"Failed {label} after {retries} attempts") from last_error


def collect_reads(
    regions: list[Region],
    bam_paths: dict[str, Path],
    max_reads_per_sample: int,
    min_mapq: int,
) -> list[tuple[str, Region, pysam.AlignedSegment]]:
    selected = []
    counts = {sample: 0 for sample in bam_paths}
    for sample, path in bam_paths.items():
        with pysam.AlignmentFile(path, "rb") as bam:
            for region in regions:
                if counts[sample] >= max_reads_per_sample:
                    break
                for read in bam.fetch(region.chrom, region.start, region.end):
                    if counts[sample] >= max_reads_per_sample:
                        break
                    if (
                        read.is_unmapped
                        or read.is_secondary
                        or read.is_supplementary
                        or read.mapping_quality < min_mapq
                        or read.query_sequence is None
                        or read.query_length is None
                    ):
                        continue
                    selected.append((sample, region, read))
                    counts[sample] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AlphaGenome predictions on original read sequences versus reference intervals."
    )
    parser.add_argument("--regions", type=Path, default=Path("outputs/4chrom_test_regions.tsv"))
    parser.add_argument("--selected-tracks", type=Path, default=Path("outputs/metadata/selected_a549_h3k4me3_tracks.tsv"))
    parser.add_argument("--bam-c1", type=Path, default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam"))
    parser.add_argument("--bam-e5b", type=Path, default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/read_sequence_sensitivity"))
    parser.add_argument("--sequence-length", type=int, default=16_384)
    parser.add_argument("--max-reads-per-sample", type=int, default=6)
    parser.add_argument("--min-mapq", type=int, default=20)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise SystemExit("ALPHAGENOME_API_KEY is not set.")

    from alphagenome.data import genome
    from alphagenome.models import dna_client

    if args.sequence_length not in (16_384, 131_072, 524_288, 1_048_576):
        raise SystemExit("AlphaGenome sequence length must be one of 16384, 131072, 524288, 1048576.")

    selected_metadata, ontology_terms = selected_terms(args.selected_tracks)
    selected_names = set(selected_metadata["name"].astype(str))
    regions = load_regions(args.regions, split="test")
    reads = collect_reads(
        regions,
        {"merged_c1": args.bam_c1, "merged_e5b": args.bam_e5b},
        args.max_reads_per_sample,
        args.min_mapq,
    )
    if not reads:
        raise SystemExit("No reads selected for read-sequence sensitivity.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = dna_client.create(api_key)
    rows = []
    for index, (sample, region, read) in enumerate(reads, start=1):
        sequence = reference_oriented_query_sequence(read)
        reference_center = max(region.start, min(region.end - 1, (region.start + region.end) // 2))
        query_center = query_pos_for_reference_center(read, reference_center)
        read_sequence, read_window_start, read_window_end = centered_supported_sequence(
            sequence, query_center, args.sequence_length
        )
        ref_start = max(0, reference_center - args.sequence_length // 2)
        ref_end = ref_start + args.sequence_length
        ref_interval = genome.Interval(chromosome=region.chrom, start=ref_start, end=ref_end)

        label = f"{sample}:{read.query_name}:{region.key}"
        print(f"[{index}/{len(reads)}] querying {label}")
        read_output = predict_with_retries(
            lambda: model.predict_sequence(
                sequence=read_sequence,
                requested_outputs=[dna_client.OutputType.CHIP_HISTONE],
                ontology_terms=ontology_terms,
            ),
            args.retries,
            f"read sequence {label}",
        )
        reference_output = predict_with_retries(
            lambda: model.predict_interval(
                interval=ref_interval,
                requested_outputs=[dna_client.OutputType.CHIP_HISTONE],
                ontology_terms=ontology_terms,
            ),
            args.retries,
            f"reference interval {label}",
        )
        read_track, returned_tracks, read_resolution = safe_track_mean(read_output, selected_names)
        ref_track, _, ref_resolution = safe_track_mean(reference_output, selected_names)
        if read_track.shape != ref_track.shape:
            raise RuntimeError(f"Read/reference output shape mismatch: {read_track.shape} vs {ref_track.shape}")
        if np.std(read_track) > 0 and np.std(ref_track) > 0:
            corr = float(np.corrcoef(read_track, ref_track)[0, 1])
        else:
            corr = np.nan
        delta = read_track - ref_track
        out_npz = args.out_dir / f"readseq_{index:03d}_{sample}_{region.chrom}_{region.region_id}.npz"
        np.savez_compressed(
            out_npz,
            read_track=read_track.astype(np.float32),
            reference_track=ref_track.astype(np.float32),
            delta=delta.astype(np.float32),
            read_sequence=np.array(read_sequence),
            provenance_json=np.array(json.dumps({
                "sample": sample,
                "read_id": read.query_name,
                "region": region.key,
                "sequence_length": args.sequence_length,
                "read_window_start": read_window_start,
                "read_window_end": read_window_end,
                "reference_interval": {
                    "chrom": region.chrom,
                    "start": ref_start,
                    "end": ref_end,
                },
                "read_resolution": read_resolution,
                "reference_resolution": ref_resolution,
                "returned_tracks": returned_tracks,
                "cigar": read.cigarstring,
                "is_reverse": bool(read.is_reverse),
            })),
        )
        rows.append({
            "sample": sample,
            "read_id": read.query_name,
            "chrom": region.chrom,
            "region_id": region.region_id,
            "region_start": region.start,
            "region_end": region.end,
            "alignment_start": int(read.reference_start),
            "alignment_end": int(read.reference_end),
            "mapq": int(read.mapping_quality),
            "is_reverse": bool(read.is_reverse),
            "cigar": read.cigarstring,
            "sequence_length": args.sequence_length,
            "non_n_bases": int(args.sequence_length - read_sequence.count("N")),
            "read_track_mean": float(np.nanmean(read_track)),
            "reference_track_mean": float(np.nanmean(ref_track)),
            "mean_delta_read_minus_reference": float(np.nanmean(delta)),
            "mean_abs_delta": float(np.nanmean(np.abs(delta))),
            "pearson_read_vs_reference": corr,
            "npz": str(out_npz),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "read_sequence_vs_reference_summary.tsv", sep="\t", index=False)
    print(f"Wrote {args.out_dir / 'read_sequence_vs_reference_summary.tsv'}")


if __name__ == "__main__":
    main()
