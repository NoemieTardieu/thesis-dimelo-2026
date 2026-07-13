#!/usr/bin/env python3
"""Convert large bedGraph files to bigWig in streaming batches."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyBigWig


def load_chrom_sizes(fai_path: Path) -> list[tuple[str, int]]:
    chrom_sizes: list[tuple[str, int]] = []
    with fai_path.open() as fh:
        for line in fh:
            chrom, size = line.split("\t", 2)[:2]
            chrom_sizes.append((chrom, int(size)))
    return chrom_sizes


def convert_one(bedgraph_path: Path, fai_path: Path, output_path: Path, batch_size: int) -> None:
    chrom_sizes = load_chrom_sizes(fai_path)

    bw = pyBigWig.open(str(output_path), "w")
    bw.addHeader(chrom_sizes)

    chroms: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    values: list[float] = []

    last_chrom: str | None = None
    last_start = -1

    with bedgraph_path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            chrom, start, end, value = line.rstrip().split("\t")
            start_i = int(start)
            end_i = int(end)
            value_f = float(value)

            if chrom == last_chrom and start_i < last_start:
                raise ValueError(
                    f"{bedgraph_path} not sorted at line {line_no}: "
                    f"{chrom}:{start_i} after {last_chrom}:{last_start}"
                )

            chroms.append(chrom)
            starts.append(start_i)
            ends.append(end_i)
            values.append(value_f)

            last_chrom = chrom
            last_start = start_i

            if len(chroms) >= batch_size:
                bw.addEntries(chroms, starts, ends=ends, values=values)
                chroms.clear()
                starts.clear()
                ends.clear()
                values.clear()

    if chroms:
        bw.addEntries(chroms, starts, ends=ends, values=values)

    bw.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bedgraphs", nargs="+", help="Input bedGraph file(s)")
    parser.add_argument("--fai", required=True, help="Reference .fai file")
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for .bw files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500000,
        help="Number of rows per pyBigWig addEntries call",
    )
    args = parser.parse_args()

    fai_path = Path(args.fai)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for bedgraph in args.bedgraphs:
        bedgraph_path = Path(bedgraph)
        output_path = outdir / f"{bedgraph_path.stem}.bw"
        convert_one(bedgraph_path, fai_path, output_path, args.batch_size)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
