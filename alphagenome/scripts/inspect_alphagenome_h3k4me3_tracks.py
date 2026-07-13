#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AlphaGenome human H3K4me3 metadata and A549 tracks.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/metadata"))
    args = parser.parse_args()

    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise SystemExit("ALPHAGENOME_API_KEY is not set.")

    from alphagenome.models import dna_client

    model = dna_client.create(api_key)
    output_metadata = model.output_metadata(organism=dna_client.Organism.HOMO_SAPIENS)
    metadata = output_metadata.chip_histone.copy()
    searchable = metadata.astype(str).agg(" ".join, axis=1)
    h3k4me3 = metadata[searchable.str.contains("H3K4me3", case=False, regex=False)].copy()
    biosample = h3k4me3.get("biosample_name", pd.Series("", index=h3k4me3.index)).astype(str)
    name = h3k4me3.get("name", pd.Series("", index=h3k4me3.index)).astype(str)
    a549 = h3k4me3[biosample.str.contains("A549", case=False) | name.str.contains("A549", case=False)].copy()
    if a549.empty:
        raise SystemExit("No A549 H3K4me3 tracks were found; inspect the complete metadata before proceeding.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.out_dir / "chip_histone_all.tsv", sep="\t", index=False)
    h3k4me3.to_csv(args.out_dir / "chip_histone_h3k4me3.tsv", sep="\t", index=False)
    a549.to_csv(args.out_dir / "selected_a549_h3k4me3_tracks.tsv", sep="\t", index=False)
    print(a549.to_string(index=False))
    print(f"\nSelected {len(a549)} A549 H3K4me3 tracks.")


if __name__ == "__main__":
    main()
