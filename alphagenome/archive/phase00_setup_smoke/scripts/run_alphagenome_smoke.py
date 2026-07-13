#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from alphagenome_query import parse_args, run_query, selected_terms
from benchmark_utils import cache_name, load_prediction_cache, load_regions


def main() -> None:
    args = parse_args()
    run_query(
        args.regions,
        args.selected_tracks,
        args.cache_dir,
        1,
        args.retries,
        args.fasta_index,
    )
    _, ontology_terms = selected_terms(args.selected_tracks)
    region = load_regions(args.regions)[0]
    cache_path = args.cache_dir / cache_name(region, ontology_terms)
    values, metadata, provenance = load_prediction_cache(cache_path)
    import matplotlib.pyplot as plt

    selected_names = set(provenance["selected_track_names"])
    indices = [i for i, row in enumerate(metadata) if str(row["name"]) in selected_names]
    if not indices:
        raise SystemExit("Smoke cache has no selected A549 H3K4me3 tracks.")
    mean_track = values[:, indices].mean(axis=1)
    interval = provenance["returned_interval"]
    resolution = int(provenance["resolution"])
    x = [int(interval["start"]) + (i + 0.5) * resolution for i in range(len(mean_track))]
    output = Path("outputs/plots/alphagenome_smoke.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(13, 4))
    axis.plot(x, mean_track, color="black", linewidth=1)
    axis.set(
        title=f"AlphaGenome A549 H3K4me3 smoke: {region.key}",
        xlabel=f"{region.chrom} coordinate (hg38)",
        ylabel="Fixed track mean",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote diagnostic plot to {output}")


if __name__ == "__main__":
    main()
