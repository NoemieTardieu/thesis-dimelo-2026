import argparse
import csv
from typing import Dict, Iterable, List, Sequence, Tuple

import pysam


DEFAULT_TRAIN_CHROMS = [
    f"chr{i}" for i in range(1, 17)
]
DEFAULT_VAL_CHROMS = ["chr17", "chr18"]
DEFAULT_TEST_CHROMS = ["chr19", "chr20", "chr21", "chr22", "chrX"]


def _as_set(chroms: Sequence[str]) -> set:
    return {c.strip() for c in chroms if c.strip()}


def load_chrom_sizes_from_bam(bam_path: str) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for chrom, length in zip(bam.references, bam.lengths):
            sizes[chrom] = int(length)
    return sizes


def iter_windows(chrom: str, length: int, window_size: int, stride: int) -> Iterable[Tuple[str, int, int]]:
    if length <= 0:
        return
    start = 0
    while start < length:
        end = min(start + window_size, length)
        if end - start < window_size:
            break
        yield chrom, start, end
        start += stride


def assign_split(chrom: str, train: set, val: set, test: set) -> str:
    if chrom in train:
        return "train"
    if chrom in val:
        return "valid"
    if chrom in test:
        return "test"
    return "exclude"


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate long-context interval TSV from BAM chromosome sizes.")
    parser.add_argument("--bam", required=True, help="Any indexed BAM with desired chromosome naming/sizes.")
    parser.add_argument("--out-tsv", required=True, help="Output intervals TSV with split labels.")
    parser.add_argument("--window-size", type=int, default=1_000_000)
    parser.add_argument("--stride", type=int, default=1_000_000)
    parser.add_argument("--train-chroms", default=",".join(DEFAULT_TRAIN_CHROMS))
    parser.add_argument("--val-chroms", default=",".join(DEFAULT_VAL_CHROMS))
    parser.add_argument("--test-chroms", default=",".join(DEFAULT_TEST_CHROMS))
    parser.add_argument("--include-chrom-prefix", default="chr", help="Only include chromosomes starting with this prefix.")
    args = parser.parse_args()

    chrom_sizes = load_chrom_sizes_from_bam(args.bam)
    train = _as_set(parse_csv_list(args.train_chroms))
    val = _as_set(parse_csv_list(args.val_chroms))
    test = _as_set(parse_csv_list(args.test_chroms))

    fieldnames = ["window_id", "chrom", "start", "end", "split"]
    n_rows = 0
    with open(args.out_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for chrom in sorted(chrom_sizes.keys()):
            if args.include_chrom_prefix and not chrom.startswith(args.include_chrom_prefix):
                continue
            split = assign_split(chrom, train, val, test)
            if split == "exclude":
                continue
            length = chrom_sizes[chrom]
            for chrom_name, start, end in iter_windows(chrom, length, args.window_size, args.stride):
                writer.writerow(
                    {
                        "window_id": f"{chrom_name}:{start}-{end}",
                        "chrom": chrom_name,
                        "start": start,
                        "end": end,
                        "split": split,
                    }
                )
                n_rows += 1

    print(f"Wrote {n_rows} intervals to {args.out_tsv}")


if __name__ == "__main__":
    main()
