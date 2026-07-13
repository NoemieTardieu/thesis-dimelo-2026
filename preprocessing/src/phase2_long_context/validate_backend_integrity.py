import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    p = argparse.ArgumentParser(description="Validate interval backend integrity (manifest + NPZ files).")
    p.add_argument("--manifest", required=True, help="Path to manifest_<mark>_<base>.tsv")
    p.add_argument("--intervals-tsv", default=None, help="Optional interval TSV to compare row count.")
    p.add_argument("--max-errors", type=int, default=20)
    args = p.parse_args()

    if not os.path.exists(args.manifest):
        raise SystemExit(f"Manifest not found: {args.manifest}")

    rows = load_rows(args.manifest)
    print(f"manifest_rows={len(rows)}")

    if args.intervals_tsv is not None:
        if not os.path.exists(args.intervals_tsv):
            raise SystemExit(f"Intervals file not found: {args.intervals_tsv}")
        irows = load_rows(args.intervals_tsv)
        print(f"interval_rows={len(irows)}")
        if len(irows) != len(rows):
            print(f"WARN row_count_mismatch intervals={len(irows)} manifest={len(rows)}")

    errors: List[Tuple] = []
    for i, row in enumerate(rows, start=1):
        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])
        expected_len = end - start
        npz_path = row["npz_path"]

        if not os.path.exists(npz_path):
            errors.append((i, "missing_npz", npz_path))
            if len(errors) >= args.max_errors:
                break
            continue

        try:
            z = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            errors.append((i, "npz_read_error", str(e), npz_path))
            if len(errors) >= args.max_errors:
                break
            continue

        for key in ("methyl_ids", "coverage", "meth_counts"):
            if key not in z:
                errors.append((i, "missing_key", key, npz_path))

        if "methyl_ids" in z:
            arr = z["methyl_ids"]
            if len(arr) != expected_len:
                errors.append((i, "methyl_len_mismatch", len(arr), expected_len, npz_path))
            vals = np.unique(arr)
            if np.any((vals < 0) | (vals > 2)):
                errors.append((i, "methyl_value_out_of_range", vals.tolist(), npz_path))

        if "coverage" in z and len(z["coverage"]) != expected_len:
            errors.append((i, "coverage_len_mismatch", len(z["coverage"]), expected_len, npz_path))

        if "meth_counts" in z and len(z["meth_counts"]) != expected_len:
            errors.append((i, "meth_counts_len_mismatch", len(z["meth_counts"]), expected_len, npz_path))

        if len(errors) >= args.max_errors:
            break

    print(f"errors={len(errors)}")
    for e in errors:
        print("ERR", e)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
