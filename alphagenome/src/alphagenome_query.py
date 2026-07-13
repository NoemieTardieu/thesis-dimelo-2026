#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from benchmark_utils import (
    cache_name,
    load_prediction_cache,
    load_regions,
    read_fasta_lengths,
    region_record,
    save_prediction_cache,
    supported_enclosing_interval,
)


def selected_terms(path: Path, expected_biosample: str | None) -> tuple[pd.DataFrame, list[str]]:
    selected = pd.read_csv(path, sep="\t")
    required = {"name", "ontology_curie", "biosample_name"}
    missing = required - set(selected.columns)
    if missing:
        raise SystemExit(f"Selected-track metadata lacks columns: {sorted(missing)}")
    is_mark = selected.astype(str).agg(" ".join, axis=1).str.contains("H3K4me3", case=False)
    if expected_biosample:
        is_expected_biosample = (
            selected["biosample_name"].astype(str).str.contains(expected_biosample, case=False)
            | selected["name"].astype(str).str.contains(expected_biosample, case=False)
        )
        if not (is_expected_biosample & is_mark).all():
            raise SystemExit(
                f"Selected metadata contains a non-{expected_biosample} or non-H3K4me3 track."
            )
    elif not is_mark.all():
        raise SystemExit("Selected metadata contains a non-H3K4me3 track.")
    return selected, sorted(selected["ontology_curie"].dropna().astype(str).unique())


def run_query(
    regions_path: Path,
    selected_tracks_path: Path,
    cache_dir: Path,
    limit: int | None,
    retries: int,
    fasta_index: Path,
    expected_biosample: str | None,
) -> None:
    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise SystemExit("ALPHAGENOME_API_KEY is not set.")

    from alphagenome.data import genome
    from alphagenome.models import dna_client

    selected, ontology_terms = selected_terms(selected_tracks_path, expected_biosample)
    selected_names = set(selected["name"].astype(str))
    regions = load_regions(regions_path)
    chromosome_lengths = read_fasta_lengths(fasta_index)
    if limit is not None:
        regions = regions[:limit]
    model = dna_client.create(api_key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for index, region in enumerate(regions, start=1):
        cache_path = cache_dir / cache_name(region, ontology_terms)
        if cache_path.exists():
            try:
                load_prediction_cache(cache_path)
                print(f"[{index}/{len(regions)}] cached {region.key}")
                continue
            except (KeyError, ValueError, OSError):
                cache_path.unlink()

        for attempt in range(1, retries + 1):
            try:
                if region.chrom not in chromosome_lengths:
                    raise ValueError(
                        f"Chromosome {region.chrom} is absent from {fasta_index}."
                    )
                query_start, query_end = supported_enclosing_interval(
                    region, chromosome_lengths[region.chrom]
                )
                requested = genome.Interval(
                    chromosome=region.chrom, start=query_start, end=query_end
                )
                outputs = model.predict_interval(
                    interval=requested,
                    ontology_terms=ontology_terms,
                    requested_outputs=[dna_client.OutputType.CHIP_HISTONE],
                )
                track = outputs.chip_histone
                returned_names = set(track.metadata["name"].astype(str))
                missing = selected_names - returned_names
                if missing:
                    raise ValueError(f"Selected tracks absent from response: {sorted(missing)}")
                interval = track.interval
                provenance = {
                    "benchmark_region": region_record(region),
                    "model_input_interval": {
                        "chrom": region.chrom,
                        "start": query_start,
                        "end": query_end,
                        "width": query_end - query_start,
                    },
                    "returned_interval": {
                        "chrom": interval.chromosome,
                        "start": int(interval.start),
                        "end": int(interval.end),
                    },
                    "resolution": int(track.resolution),
                    "ontology_terms": ontology_terms,
                    "selected_track_names": sorted(selected_names),
                    "output_type": "CHIP_HISTONE",
                    "alphagenome_version": importlib.metadata.version("alphagenome"),
                    "query_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                save_prediction_cache(
                    cache_path,
                    track.values,
                    track.metadata.to_dict(orient="records"),
                    provenance,
                )
                print(f"[{index}/{len(regions)}] queried {region.key} -> {cache_path.name}")
                break
            except ValueError:
                raise
            except Exception:
                if attempt == retries:
                    raise
                delay = min(60.0, 2.0 ** (attempt - 1)) + random.random()
                print(f"Retrying {region.key} after attempt {attempt} in {delay:.1f}s")
                time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query and atomically cache AlphaGenome H3K4me3 predictions.")
    parser.add_argument("--regions", type=Path, default=Path("outputs/4chrom_test_regions.tsv"))
    parser.add_argument(
        "--selected-tracks",
        type=Path,
        default=Path("outputs/metadata/selected_a549_h3k4me3_tracks.tsv"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument(
        "--fasta-index",
        type=Path,
        default=Path("/data/leuven/383/vsc38330/thesis_dimelo/src/data/hg38.fa.fai"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--expected-biosample",
        default="A549",
        help="Require selected tracks to match this biosample/name. Use an empty string to disable the biosample check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_query(
        args.regions,
        args.selected_tracks,
        args.cache_dir,
        args.limit,
        args.retries,
        args.fasta_index,
        args.expected_biosample or None,
    )


if __name__ == "__main__":
    main()
