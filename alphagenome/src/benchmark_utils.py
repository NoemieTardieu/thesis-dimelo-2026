#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pysam

BIN_SIZE = 128
CHROMS = ("chr16", "chr11", "chr17", "chr19")
SAMPLES = ("merged_c1", "merged_e5b")
SUPPORTED_SEQUENCE_LENGTHS = (16_384, 131_072, 524_288, 1_048_576)


@dataclass(frozen=True)
class Region:
    region_id: str
    chrom: str
    start: int
    end: int
    name: str
    split: str

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    positive_count: int = 0
    reads: set[str] | None = None

    def update(self, value: float, read_id: str, threshold: float = 0.5) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.positive_count += int(value >= threshold)
        if self.reads is None:
            self.reads = set()
        self.reads.add(read_id)

    def row(self) -> dict[str, float | int | None]:
        if not self.count:
            return {
                "mean_signal": None,
                "positive_fraction_0_5": None,
                "unique_reads": 0,
                "observed_positions": 0,
                "variance": None,
                "standard_error": None,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "mean_signal": mean,
            "positive_fraction_0_5": self.positive_count / self.count,
            "unique_reads": len(self.reads or ()),
            "observed_positions": self.count,
            "variance": variance,
            "standard_error": (variance / self.count) ** 0.5,
        }


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_regions(path: str | Path, split: str | None = None) -> list[Region]:
    regions = []
    for row in read_tsv(path):
        row_split = row.get("split", "")
        if split is not None and row_split != split:
            continue
        regions.append(
            Region(
                region_id=str(row.get("region_id", "")),
                chrom=row["chrom"],
                start=int(row["start"]),
                end=int(row["end"]),
                name=row.get("name") or row.get("region_name") or "",
                split=row_split,
            )
        )
    return regions


def read_fasta_lengths(path: str | Path) -> dict[str, int]:
    lengths = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                lengths[fields[0]] = int(fields[1])
    return lengths


def supported_enclosing_interval(
    region: Region,
    chromosome_length: int,
    supported_lengths: tuple[int, ...] = SUPPORTED_SEQUENCE_LENGTHS,
) -> tuple[int, int]:
    target_length = next(
        (length for length in supported_lengths if length >= region.end - region.start),
        None,
    )
    if target_length is None:
        raise ValueError(
            f"Region {region.key} is wider than the largest supported sequence length."
        )
    if chromosome_length < target_length:
        raise ValueError(
            f"Chromosome {region.chrom} length {chromosome_length} is shorter than "
            f"the required model input {target_length}."
        )

    center = (region.start + region.end) // 2
    start = center - target_length // 2
    start = max(0, min(start, chromosome_length - target_length))
    end = start + target_length
    if start > region.start or end < region.end:
        raise ValueError(f"Could not enclose benchmark region {region.key}.")
    return start, end


def complete_bins(start: int, end: int, size: int = BIN_SIZE) -> Iterator[tuple[int, int]]:
    for bin_start in range(start, end - size + 1, size):
        yield bin_start, bin_start + size


def read_to_reference_map(
    alignment: pysam.AlignedSegment, forward_positions: set[int]
) -> dict[int, int]:
    if not forward_positions or alignment.query_length is None:
        return {}
    read_length = int(alignment.query_length)
    query_to_forward = {
        read_length - 1 - pos if alignment.is_reverse else pos: pos
        for pos in forward_positions
    }
    mapped = {}
    for query_pos, reference_pos in alignment.get_aligned_pairs(matches_only=False):
        if query_pos is None or reference_pos is None:
            continue
        forward_pos = query_to_forward.get(query_pos)
        if forward_pos is not None:
            mapped[forward_pos] = int(reference_pos)
    return mapped


def collapse_windows(
    targets: np.ndarray,
    masks: np.ndarray,
    metadata: list[dict[str, str]],
    row_indices: list[int],
) -> tuple[dict[int, float], int]:
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    raw = 0
    for row_idx in row_indices:
        row = metadata[row_idx]
        length = min(int(row["window_length"]), targets.shape[1])
        for local_pos in np.flatnonzero(masks[row_idx, :length]):
            read_pos = int(row["window_start"]) + int(local_pos)
            sums[read_pos] += float(targets[row_idx, local_pos])
            counts[read_pos] += 1
            raw += 1
    return ({pos: sums[pos] / counts[pos] for pos in sums}, raw - len(sums))


def group_metadata(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, str], list[int]]:
    grouped: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        sample = row.get("sample") or row.get("sample_id") or "unknown"
        region_id = row.get("region_id") or row.get("region_name") or ""
        grouped[(sample, row["read_id"], row["chrom"], region_id)].append(idx)
    return grouped


def find_primary_alignments(
    bam: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
    read_ids: set[str],
) -> dict[str, pysam.AlignedSegment]:
    found = {}
    for alignment in bam.fetch(chrom, max(0, start), end):
        if (
            alignment.query_name in read_ids
            and alignment.query_name not in found
            and not alignment.is_unmapped
            and not alignment.is_secondary
            and not alignment.is_supplementary
            and alignment.query_length is not None
        ):
            found[alignment.query_name] = alignment
    return found


def atomic_savez(path: str | Path, **arrays: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def cache_name(region: Region, ontology_terms: list[str]) -> str:
    selection = ",".join(sorted(ontology_terms))
    digest = hashlib.sha256(selection.encode("utf-8")).hexdigest()[:12]
    return f"{region.chrom}_{region.start}_{region.end}_chip_histone_{digest}.npz"


def save_prediction_cache(
    path: str | Path,
    values: np.ndarray,
    metadata_records: list[dict],
    provenance: dict,
) -> None:
    atomic_savez(
        path,
        values=np.asarray(values, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata_records, sort_keys=True)),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )


def load_prediction_cache(path: str | Path) -> tuple[np.ndarray, list[dict], dict]:
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["values"])
        metadata = json.loads(str(archive["metadata_json"].item()))
        provenance = json.loads(str(archive["provenance_json"].item()))
    if values.ndim != 2 or values.shape[1] != len(metadata):
        raise ValueError(f"Invalid cache shape or metadata in {path}")
    resolution = int(provenance["resolution"])
    interval = provenance["returned_interval"]
    if values.shape[0] * resolution != int(interval["end"]) - int(interval["start"]):
        raise ValueError(f"Cache interval/resolution mismatch in {path}")
    return values, metadata, provenance


def json_dump(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def region_record(region: Region) -> dict:
    return asdict(region)
