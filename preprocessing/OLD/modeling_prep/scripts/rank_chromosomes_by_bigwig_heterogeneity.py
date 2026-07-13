#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pyBigWig


DEFAULT_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank chromosomes by binned signal heterogeneity from A-mod and CpG-mod bigWigs."
    )
    p.add_argument("--a-bw", required=True, help="A-mod/6mA percent bigWig.")
    p.add_argument("--c-bw", required=True, help="CpG C-mod percent bigWig.")
    p.add_argument("--chrom-sizes", required=True, help="Chrom sizes TSV.")
    p.add_argument("--out", required=True, help="Output TSV.")
    p.add_argument("--bin-size", type=int, default=100_000, help="Bin size for chromosome summaries.")
    p.add_argument("--min-valid-bins", type=int, default=100, help="Minimum bins with signal in either track.")
    return p.parse_args()


def load_chrom_sizes(path: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chrom, size = line.rstrip("\n").split("\t")[:2]
            sizes[chrom] = int(size)
    return sizes


def binned_means(bw: pyBigWig.pyBigWig, chrom: str, size: int, bin_size: int) -> np.ndarray:
    values = []
    for start in range(0, size, bin_size):
        end = min(start + bin_size, size)
        value = bw.stats(chrom, start, end, type="mean")[0]
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    valid = np.isfinite(values)
    out: dict[str, float | int] = {
        f"{prefix}_valid_bins": int(valid.sum()),
        f"{prefix}_mean": math.nan,
        f"{prefix}_std": math.nan,
        f"{prefix}_cv": math.nan,
        f"{prefix}_q05": math.nan,
        f"{prefix}_q50": math.nan,
        f"{prefix}_q95": math.nan,
        f"{prefix}_range_q95_q05": math.nan,
        f"{prefix}_nonzero_frac": math.nan,
    }
    if not valid.any():
        return out

    x = values[valid]
    mean = float(np.mean(x))
    std = float(np.std(x))
    q05, q50, q95 = np.quantile(x, [0.05, 0.50, 0.95])
    out.update(
        {
            f"{prefix}_mean": mean,
            f"{prefix}_std": std,
            f"{prefix}_cv": float(std / mean) if mean > 0 else math.nan,
            f"{prefix}_q05": float(q05),
            f"{prefix}_q50": float(q50),
            f"{prefix}_q95": float(q95),
            f"{prefix}_range_q95_q05": float(q95 - q05),
            f"{prefix}_nonzero_frac": float(np.mean(x > 0)),
        }
    )
    return out


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chrom_sizes = load_chrom_sizes(args.chrom_sizes)
    a_bw = pyBigWig.open(args.a_bw)
    c_bw = pyBigWig.open(args.c_bw)

    rows = []
    for chrom in DEFAULT_CHROMS:
        if chrom not in chrom_sizes:
            continue
        size = chrom_sizes[chrom]
        a_vals = binned_means(a_bw, chrom, size, args.bin_size)
        c_vals = binned_means(c_bw, chrom, size, args.bin_size)

        row: dict[str, float | int | str] = {
            "chrom": chrom,
            "chrom_size": size,
            "bin_size": args.bin_size,
            "n_bins": int(math.ceil(size / args.bin_size)),
        }
        row.update(summarize(a_vals, "a_mod"))
        row.update(summarize(c_vals, "cpg_mod"))

        a_valid = int(row["a_mod_valid_bins"])
        c_valid = int(row["cpg_mod_valid_bins"])
        enough = max(a_valid, c_valid) >= args.min_valid_bins
        if enough:
            # Simple combined heterogeneity score; ranks chromosomes with variable
            # signal in both tracks higher, but still allows A-mod to dominate.
            a_component = float(row["a_mod_range_q95_q05"])
            c_component = float(row["cpg_mod_range_q95_q05"])
            if not np.isfinite(a_component):
                a_component = 0.0
            if not np.isfinite(c_component):
                c_component = 0.0
            row["heterogeneity_score"] = a_component + c_component
        else:
            row["heterogeneity_score"] = math.nan
        rows.append(row)

    a_bw.close()
    c_bw.close()

    rows.sort(
        key=lambda r: (
            -1 if not np.isfinite(float(r["heterogeneity_score"])) else -float(r["heterogeneity_score"]),
            str(r["chrom"]),
        )
    )

    fieldnames = [
        "chrom",
        "chrom_size",
        "bin_size",
        "n_bins",
        "heterogeneity_score",
        "a_mod_valid_bins",
        "a_mod_mean",
        "a_mod_std",
        "a_mod_cv",
        "a_mod_q05",
        "a_mod_q50",
        "a_mod_q95",
        "a_mod_range_q95_q05",
        "a_mod_nonzero_frac",
        "cpg_mod_valid_bins",
        "cpg_mod_mean",
        "cpg_mod_std",
        "cpg_mod_cv",
        "cpg_mod_q05",
        "cpg_mod_q50",
        "cpg_mod_q95",
        "cpg_mod_range_q95_q05",
        "cpg_mod_nonzero_frac",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {out_path}")
    print("Top 10 chromosomes by binned heterogeneity:")
    for row in rows[:10]:
        print(
            f"{row['chrom']}\tscore={float(row['heterogeneity_score']):.4g}\t"
            f"A_range={float(row['a_mod_range_q95_q05']):.4g}\t"
            f"CpG_range={float(row['cpg_mod_range_q95_q05']):.4g}\t"
            f"A_valid={row['a_mod_valid_bins']}\tCpG_valid={row['cpg_mod_valid_bins']}"
        )


if __name__ == "__main__":
    main()
