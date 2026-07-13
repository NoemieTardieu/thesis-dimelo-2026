from __future__ import annotations

import numpy as np
import pandas as pd

import population_track_benchmark as bench


def test_bin_alignment_normalization_and_pairwise_metrics() -> None:
    regions = pd.DataFrame(
        [
            {
                "chrom": "chr1",
                "region_id": 1,
                "region_name": "toy",
                "region_start": 0,
                "region_end": 4000,
            }
        ]
    )
    bins = bench.canonical_bins(regions, 1000)
    assert bins[["start", "end"]].to_records(index=False).tolist() == [
        (0, 1000),
        (1000, 2000),
        (2000, 3000),
        (3000, 4000),
    ]

    source_rows = []
    for start in range(0, 4000, 500):
        source_rows.append(
            {
                "sample": "pooled",
                "chrom": "chr1",
                "region_id": 1,
                "region_start": 0,
                "region_end": 4000,
                "region_name": "toy",
                "bin_start": start,
                "bin_end": start + 500,
                "mean_signal": start / 4000,
                "unique_reads": 2,
                "observed_positions": 10,
            }
        )
    read_track = pd.DataFrame(source_rows)
    dimelo = bench.aggregate_read_track(read_track, bins, "pooled", "mean_signal", "dimelo")
    hyena = bench.aggregate_read_track(read_track.assign(mean_signal=read_track["mean_signal"] + 0.1), bins, "pooled", "mean_signal", "hyena")
    alpha = bench.aggregate_alpha(
        pd.DataFrame(
            {
                "chrom": ["chr1"] * 4,
                "region_id": [1] * 4,
                "region_name": ["toy"] * 4,
                "region_start": [0] * 4,
                "region_end": [4000] * 4,
                "bin_start": [0, 1000, 2000, 3000],
                "bin_end": [1000, 2000, 3000, 4000],
                "alpha": [0.0, 0.3, 0.6, 0.9],
            }
        ),
        bins,
        "alpha",
    )
    external = bins[["chrom", "region_id", "region_name", "start", "end"]].copy()
    external["external_raw"] = [0.0, 0.3, np.nan, 0.9]
    external["external_covered_fraction"] = [1.0, 1.0, 0.0, 1.0]
    external["external_valid"] = external["external_raw"].notna()

    keys = ["chrom", "region_id", "region_name", "start", "end"]
    canonical = bins.merge(external, on=keys)
    canonical = canonical.merge(alpha, on=keys)
    canonical = canonical.merge(hyena, on=keys)
    canonical = canonical.merge(dimelo, on=keys)
    canonical["all_four_valid"] = (
        canonical["external_valid"]
        & canonical["alphagenome_valid"]
        & canonical["hyena_valid"]
        & canonical["dimelo_valid"]
    )
    normalized, params = bench.add_normalized_columns(canonical)

    assert len(params) == 4
    assert normalized["external_raw"].isna().sum() == 1
    assert normalized.loc[normalized["external_raw"].isna(), "external_valid"].eq(False).all()
    assert normalized["dimelo_coverage"].tolist() == [20.0, 20.0, 20.0, 20.0]

    pair_metrics, _ = bench.pairwise_metrics(
        normalized,
        use_common_intersection=False,
        coverage_threshold=1,
        block_size=1000,
        bootstrap_replicates=0,
        seed=1,
    )
    td = pair_metrics[pair_metrics["comparison"] == "T-D"].iloc[0]
    assert td["n_bins"] == 3
    assert np.isfinite(td["pearson"])
