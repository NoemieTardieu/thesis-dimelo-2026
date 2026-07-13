#!/usr/bin/env python3
"""Build A- and C-fraction bedGraph tracks from modkit extract-calls tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import pandas as pd


DEFAULT_INPUTS = {
    "h3k27ac": Path(
        "/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/h3k27ac/extract_calls.tsv.gz"
    ),
    "h3k27me3": Path(
        "/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/h3k27me3/extract_calls.tsv.gz"
    ),
    "h3k4me3": Path(
        "/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/h3k4me3/extract_calls.tsv.gz"
    ),
}

DEFAULT_OUTPUT_DIR = Path(
    "/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/fraction_tracks"
)

TRACKED_BASES = ("A", "C")


def normalize_fail_column(values: pd.Series) -> pd.Series:
    lowered = values.astype(str).str.strip().str.lower()
    return lowered.map({"true": True, "false": False})


def parse_mark_inputs(items: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected MARK=PATH format for --input, got {item!r}.")
        mark, path = item.split("=", 1)
        parsed.append((mark.strip(), Path(path).expanduser()))
    return parsed


def build_position_counts(
    input_path: Path, chunksize: int
) -> DefaultDict[tuple[str, int, str], list[int]]:
    counts: DefaultDict[tuple[str, int, str], list[int]] = defaultdict(lambda: [0, 0])

    usecols = ["chrom", "ref_position", "canonical_base", "call_code", "fail"]
    reader = pd.read_csv(
        input_path,
        sep="\t",
        usecols=usecols,
        chunksize=chunksize,
        compression="infer",
        low_memory=False,
    )

    for chunk in reader:
        chunk = chunk[chunk["canonical_base"].isin(TRACKED_BASES)].copy()
        if chunk.empty:
            continue

        chunk["fail"] = normalize_fail_column(chunk["fail"])
        chunk = chunk.dropna(subset=["fail"])
        chunk = chunk[chunk["fail"] == False].copy()
        if chunk.empty:
            continue

        chunk["is_modified"] = chunk["call_code"].astype(str) != "-"
        grouped = (
            chunk.groupby(["chrom", "ref_position", "canonical_base"], sort=False)
            .agg(modified_sum=("is_modified", "sum"), total_calls=("is_modified", "size"))
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            key = (str(row.chrom), int(row.ref_position), str(row.canonical_base))
            counts[key][0] += int(row.modified_sum)
            counts[key][1] += int(row.total_calls)

    return counts


def write_bedgraph(
    mark: str,
    base: str,
    counts: dict[tuple[str, int, str], list[int]],
    output_dir: Path,
) -> Path:
    output_path = output_dir / f"{mark}_{base}_mod_fraction.bedGraph"
    rows: list[tuple[str, int, int, float]] = []

    for (chrom, ref_position, canonical_base), (modified_sum, total_calls) in counts.items():
        if canonical_base != base or total_calls == 0:
            continue
        frac = modified_sum / total_calls
        rows.append((chrom, ref_position, ref_position + 1, frac))

    rows.sort(key=lambda x: (x[0], x[1]))

    with output_path.open("w") as handle:
        for chrom, start, end, frac in rows:
            handle.write(f"{chrom}\t{start}\t{end}\t{frac:.6f}\n")

    return output_path


def run(mark_inputs: list[tuple[str, Path]], output_dir: Path, chunksize: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for mark, input_path in mark_inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found for {mark}: {input_path}")

        print(f"Building fraction tracks for {mark}: {input_path}")
        counts = build_position_counts(input_path, chunksize=chunksize)

        for base in TRACKED_BASES:
            output_path = write_bedgraph(mark, base, counts, output_dir)
            print(f"Saved {base} track to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert modkit extract-calls tables into A/C fraction bedGraph tracks."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="MARK=PATH",
        help="Input extract-calls file for one mark. Can be given multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for bedGraph fraction tracks (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Number of rows per pandas chunk when reading extract-calls tables.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    mark_inputs = parse_mark_inputs(args.input) if args.input else list(DEFAULT_INPUTS.items())
    run(mark_inputs=mark_inputs, output_dir=args.output_dir, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
