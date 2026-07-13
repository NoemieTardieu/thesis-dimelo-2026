#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create region-level train/val/test tensor files from a DiMeLo .npz and metadata."
    )
    p.add_argument("--npz", required=True, help="Input tensor .npz file.")
    p.add_argument("--metadata", required=True, help="Input metadata TSV matching the .npz rows.")
    p.add_argument("--regions-bed", required=True, help="Selected regions BED file.")
    p.add_argument("--out-prefix", required=True, help="Output prefix for split files.")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def read_regions(path: str | Path) -> list[dict[str, object]]:
    regions = []
    with open(path, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            name = parts[3] if len(parts) > 3 else f"region_{i + 1}"
            regions.append(
                {
                    "region_id": len(regions),
                    "chrom": parts[0],
                    "start": int(parts[1]),
                    "end": int(parts[2]),
                    "name": name,
                }
            )
    if not regions:
        raise SystemExit(f"No regions found in {path}")
    return regions


def assign_region_splits(
    regions: list[dict[str, object]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> list[dict[str, object]]:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Fractions must sum to 1.0, got {total}")

    rng = random.Random(seed)
    shuffled = list(regions)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise SystemExit("Split fractions produced no test regions.")

    for i, region in enumerate(shuffled):
        if i < n_train:
            region["split"] = "train"
        elif i < n_train + n_val:
            region["split"] = "val"
        else:
            region["split"] = "test"

    return sorted(shuffled, key=lambda r: int(r["region_id"]))


def read_metadata(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def assign_rows_to_splits(
    metadata: list[dict[str, str]], regions: list[dict[str, object]]
) -> tuple[dict[str, list[int]], list[dict[str, str]], dict[str, int]]:
    by_chrom: dict[str, list[dict[str, object]]] = {}
    for region in regions:
        by_chrom.setdefault(str(region["chrom"]), []).append(region)

    split_to_indices = {"train": [], "val": [], "test": [], "unassigned": []}
    assigned_metadata = []
    counters = {"ambiguous_tie_rows": 0}

    for row in metadata:
        chrom = row["chrom"]
        start = int(row["alignment_start"])
        end = int(row["alignment_end"])
        candidates = []
        for region in by_chrom.get(chrom, []):
            ov = overlap_len(start, end, int(region["start"]), int(region["end"]))
            if ov > 0:
                candidates.append((ov, region))

        row_idx = int(row["row_idx"])
        out_row = dict(row)
        if not candidates:
            split_to_indices["unassigned"].append(row_idx)
            out_row["region_id"] = ""
            out_row["region_name"] = ""
            out_row["split"] = "unassigned"
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)
            if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
                counters["ambiguous_tie_rows"] += 1
            region = candidates[0][1]
            split = str(region["split"])
            split_to_indices[split].append(row_idx)
            out_row["region_id"] = str(region["region_id"])
            out_row["region_name"] = str(region["name"])
            out_row["split"] = split
        assigned_metadata.append(out_row)

    return split_to_indices, assigned_metadata, counters


def write_region_splits(path: Path, regions: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = ["region_id", "chrom", "start", "end", "name", "split"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(regions)


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    regions = read_regions(args.regions_bed)
    regions = assign_region_splits(
        regions, args.train_frac, args.val_frac, args.test_frac, args.seed
    )
    metadata = read_metadata(args.metadata)
    split_to_indices, assigned_metadata, counters = assign_rows_to_splits(metadata, regions)

    tensor_data = np.load(args.npz)
    split_summary = {}

    write_region_splits(out_prefix.with_suffix(".region_splits.tsv"), regions)
    write_metadata(out_prefix.with_suffix(".metadata_with_splits.tsv"), assigned_metadata)

    for split in ["train", "val", "test"]:
        indices = np.asarray(sorted(split_to_indices[split]), dtype=np.int64)
        split_path = out_prefix.parent / f"{out_prefix.name}.{split}.npz"
        np.savez_compressed(
            split_path,
            **{key: tensor_data[key][indices] for key in tensor_data.files},
        )

        split_rows = [
            row for row in assigned_metadata if row["split"] == split
        ]
        write_metadata(out_prefix.parent / f"{out_prefix.name}.{split}.metadata.tsv", split_rows)

        split_summary[split] = {
            "reads": int(indices.size),
            "npz": str(split_path),
            "metadata": str(out_prefix.parent / f"{out_prefix.name}.{split}.metadata.tsv"),
            "valid_5mC_targets": int(tensor_data["mask_5mC"][indices].sum()) if indices.size else 0,
            "valid_6mA_targets": int(tensor_data["mask_6mA"][indices].sum()) if indices.size else 0,
        }

    region_counts = {"train": 0, "val": 0, "test": 0}
    for region in regions:
        region_counts[str(region["split"])] += 1

    summary = {
        "input_npz": args.npz,
        "input_metadata": args.metadata,
        "regions_bed": args.regions_bed,
        "out_prefix": str(out_prefix),
        "seed": args.seed,
        "region_counts": region_counts,
        "unassigned_reads": len(split_to_indices["unassigned"]),
        "counters": counters,
        "splits": split_summary,
        "region_splits": str(out_prefix.with_suffix(".region_splits.tsv")),
        "metadata_with_splits": str(out_prefix.with_suffix(".metadata_with_splits.tsv")),
    }

    with open(out_prefix.with_suffix(".summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
