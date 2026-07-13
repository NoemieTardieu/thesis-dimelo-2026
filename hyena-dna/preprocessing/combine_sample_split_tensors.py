#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ARRAY_KEYS = ["input_ids", "target_5mC", "mask_5mC", "target_6mA", "mask_6mA"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine two split NPZ datasets and add a sample_id array/metadata column."
    )
    p.add_argument("--sample-a-name", required=True)
    p.add_argument("--sample-a-id", type=int, required=True)
    p.add_argument("--sample-a-npz", required=True)
    p.add_argument("--sample-a-metadata", required=True)
    p.add_argument("--sample-b-name", required=True)
    p.add_argument("--sample-b-id", type=int, required=True)
    p.add_argument("--sample-b-npz", required=True)
    p.add_argument("--sample-b-metadata", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--out-metadata", required=True)
    p.add_argument("--out-summary", required=True)
    return p.parse_args()


def load_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    missing = [key for key in ARRAY_KEYS if key not in data.files]
    if missing:
        raise SystemExit(f"{path} is missing arrays: {missing}")
    return {key: data[key] for key in ARRAY_KEYS}


def read_metadata(path: str, sample_name: str, sample_id: int, row_offset: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"Metadata file has no header: {path}")
        for local_idx, row in enumerate(reader):
            row = dict(row)
            row["row_idx"] = str(row_offset + local_idx)
            row["sample"] = sample_name
            row["sample_id"] = str(sample_id)
            rows.append(row)
    return rows


def write_metadata(path: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise SystemExit("No metadata rows to write.")
    fieldnames = list(rows[0].keys())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    a = load_npz(args.sample_a_npz)
    b = load_npz(args.sample_b_npz)

    n_a = int(a["input_ids"].shape[0])
    n_b = int(b["input_ids"].shape[0])
    if n_a == 0 or n_b == 0:
        raise SystemExit("Both samples must contain at least one read.")

    combined = {key: np.concatenate([a[key], b[key]], axis=0) for key in ARRAY_KEYS}
    sample_id = np.concatenate(
        [
            np.full((n_a,), args.sample_a_id, dtype=np.int64),
            np.full((n_b,), args.sample_b_id, dtype=np.int64),
        ],
        axis=0,
    )
    combined["sample_id"] = sample_id

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **combined)

    rows_a = read_metadata(args.sample_a_metadata, args.sample_a_name, args.sample_a_id, 0)
    rows_b = read_metadata(args.sample_b_metadata, args.sample_b_name, args.sample_b_id, n_a)
    metadata_rows = rows_a + rows_b
    if len(metadata_rows) != n_a + n_b:
        raise SystemExit("Metadata row count does not match tensor row count.")
    write_metadata(args.out_metadata, metadata_rows)

    summary = {
        "sample_a": {
            "name": args.sample_a_name,
            "sample_id": args.sample_a_id,
            "reads": n_a,
            "npz": args.sample_a_npz,
            "metadata": args.sample_a_metadata,
        },
        "sample_b": {
            "name": args.sample_b_name,
            "sample_id": args.sample_b_id,
            "reads": n_b,
            "npz": args.sample_b_npz,
            "metadata": args.sample_b_metadata,
        },
        "combined_reads": n_a + n_b,
        "arrays": {key: list(value.shape) for key, value in combined.items()},
        "valid_5mC_targets": int(combined["mask_5mC"].sum()),
        "valid_6mA_targets": int(combined["mask_6mA"].sum()),
        "outputs": {
            "npz": args.out_npz,
            "metadata": args.out_metadata,
            "summary": args.out_summary,
        },
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
