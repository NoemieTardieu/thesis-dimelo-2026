#!/usr/bin/env python3
"""Summarize modkit extract-calls tables into per-read TSV files.

The script reads one or more large `modkit extract calls` outputs in chunks,
aggregates A- and C-channel calls per read, and writes one per-read summary TSV
per mark. It is designed for the three GM12878 DiMeLo-seq marks used in this
project, but can also be reused on other modkit extract-calls tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    "/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/per_read_summary"
)

TRACKED_BASES = ("A", "C")


@dataclass
class ReadAccumulator:
    read_id: str
    chrom: str = ""
    read_length: int = 0
    flag: int = 0
    alignment_start: int = 0
    alignment_end: int = 0
    n_total_calls: int = 0
    n_pass_calls: int = 0
    n_fail_calls: int = 0
    n_A_total: int = 0
    n_A_pass: int = 0
    n_A_fail: int = 0
    n_A_mod_pass: int = 0
    n_A_canonical_pass: int = 0
    n_A_mod_all: int = 0
    n_C_total: int = 0
    n_C_pass: int = 0
    n_C_fail: int = 0
    n_C_mod_pass: int = 0
    n_C_canonical_pass: int = 0
    n_C_mod_all: int = 0

    def update(
        self,
        chrom: str,
        read_length: int,
        flag: int,
        alignment_start: int,
        alignment_end: int,
        canonical_base: str,
        call_code: str,
        failed: bool,
    ) -> None:
        if not self.chrom:
            self.chrom = chrom
        self.read_length = max(self.read_length, read_length)
        self.flag = flag
        self.alignment_start = min_nonzero(self.alignment_start, alignment_start)
        self.alignment_end = max(self.alignment_end, alignment_end)

        is_modified = call_code != "-"

        self.n_total_calls += 1
        if failed:
            self.n_fail_calls += 1
        else:
            self.n_pass_calls += 1

        if canonical_base == "A":
            self.n_A_total += 1
            if is_modified:
                self.n_A_mod_all += 1
            if failed:
                self.n_A_fail += 1
            else:
                self.n_A_pass += 1
                if is_modified:
                    self.n_A_mod_pass += 1
                else:
                    self.n_A_canonical_pass += 1
        elif canonical_base == "C":
            self.n_C_total += 1
            if is_modified:
                self.n_C_mod_all += 1
            if failed:
                self.n_C_fail += 1
            else:
                self.n_C_pass += 1
                if is_modified:
                    self.n_C_mod_pass += 1
                else:
                    self.n_C_canonical_pass += 1

    def to_record(self, mark: str) -> Dict[str, object]:
        return {
            "mark": mark,
            "read_id": self.read_id,
            "chrom": self.chrom,
            "read_length": self.read_length,
            "flag": self.flag,
            "alignment_start": self.alignment_start,
            "alignment_end": self.alignment_end,
            "n_total_calls": self.n_total_calls,
            "n_pass_calls": self.n_pass_calls,
            "n_fail_calls": self.n_fail_calls,
            "n_A_total": self.n_A_total,
            "n_A_pass": self.n_A_pass,
            "n_A_fail": self.n_A_fail,
            "n_A_mod_all": self.n_A_mod_all,
            "n_A_mod_pass": self.n_A_mod_pass,
            "n_A_canonical_pass": self.n_A_canonical_pass,
            "frac_A_mod_pass": safe_fraction(self.n_A_mod_pass, self.n_A_pass),
            "frac_A_mod_all": safe_fraction(self.n_A_mod_all, self.n_A_total),
            "n_C_total": self.n_C_total,
            "n_C_pass": self.n_C_pass,
            "n_C_fail": self.n_C_fail,
            "n_C_mod_all": self.n_C_mod_all,
            "n_C_mod_pass": self.n_C_mod_pass,
            "n_C_canonical_pass": self.n_C_canonical_pass,
            "frac_C_mod_pass": safe_fraction(self.n_C_mod_pass, self.n_C_pass),
            "frac_C_mod_all": safe_fraction(self.n_C_mod_all, self.n_C_total),
        }


def min_nonzero(current: int, new_value: int) -> int:
    if current == 0:
        return new_value
    return min(current, new_value)


def safe_fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def normalize_fail_column(values: pd.Series) -> pd.Series:
    lowered = values.astype(str).str.strip().str.lower()
    return lowered.map({"true": True, "false": False})


def parse_mark_inputs(items: Iterable[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Expected MARK=PATH format for --input, got {item!r}."
            )
        mark, path = item.split("=", 1)
        parsed.append((mark.strip(), Path(path).expanduser()))
    return parsed


def summarize_extract_calls(input_path: Path, mark: str, chunksize: int) -> pd.DataFrame:
    columns = [
        "read_id",
        "chrom",
        "alignment_start",
        "alignment_end",
        "read_length",
        "call_code",
        "canonical_base",
        "fail",
        "flag",
    ]

    accumulators: Dict[str, ReadAccumulator] = {}

    chunk_iter = pd.read_csv(
        input_path,
        sep="\t",
        usecols=columns,
        chunksize=chunksize,
        compression="infer",
        low_memory=False,
    )

    for chunk in chunk_iter:
        chunk = chunk[chunk["canonical_base"].isin(TRACKED_BASES)].copy()
        if chunk.empty:
            continue

        chunk["fail"] = normalize_fail_column(chunk["fail"])
        chunk = chunk.dropna(subset=["fail"])
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            read_id = str(row.read_id)
            acc = accumulators.get(read_id)
            if acc is None:
                acc = ReadAccumulator(read_id=read_id)
                accumulators[read_id] = acc
            acc.update(
                chrom=str(row.chrom),
                read_length=int(row.read_length),
                flag=int(row.flag),
                alignment_start=int(row.alignment_start),
                alignment_end=int(row.alignment_end),
                canonical_base=str(row.canonical_base),
                call_code=str(row.call_code),
                failed=bool(row.fail),
            )

    records = [acc.to_record(mark) for acc in accumulators.values()]
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        result = result.sort_values(["chrom", "alignment_start", "read_id"]).reset_index(drop=True)
    return result


def run(mark_inputs: List[Tuple[str, Path]], output_dir: Path, chunksize: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for mark, input_path in mark_inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found for {mark}: {input_path}")

        print(f"Summarizing {mark}: {input_path}")
        summary_df = summarize_extract_calls(input_path, mark=mark, chunksize=chunksize)

        output_path = output_dir / f"{mark}.per_read_modkit_summary.tsv"
        summary_df.to_csv(output_path, sep="\t", index=False)
        print(f"Saved {len(summary_df)} reads to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert modkit extract-calls outputs into per-read summary TSVs."
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
        help=f"Directory for per-read summary TSVs (default: {DEFAULT_OUTPUT_DIR})",
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
