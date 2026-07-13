#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmark_utils import CHROMS, read_tsv


def load_cached_tracks(cache_dir: Path) -> dict[tuple[str, str], dict]:
    tracks = {}
    for path in sorted(cache_dir.glob("*.npz")):
        archive = np.load(path, allow_pickle=True)
        provenance = json.loads(str(archive["provenance_json"]))
        key = (str(provenance["sample"]), str(provenance["read_id"]))
        tracks[key] = {
            "track": archive["track"].astype(float),
            "resolution": int(archive["resolution"]),
            "sequence_length": int(provenance["sequence_length"]),
            "cache": str(path),
        }
    return tracks


def metric_row(target: np.ndarray, prediction: np.ndarray, positive_cutoff: float) -> dict:
    row = {
        "number_of_observations": int(target.size),
        "pearson": None,
        "spearman": None,
        "auroc": None,
        "auprc": None,
        "positive_fraction": None,
    }
    if target.size == 0:
        return row
    labels = target >= positive_cutoff
    row["positive_fraction"] = float(labels.mean())
    if target.size >= 2 and np.std(target) > 0 and np.std(prediction) > 0:
        row["pearson"] = float(pearsonr(target, prediction).statistic)
        row["spearman"] = float(spearmanr(target, prediction).statistic)
    if np.unique(labels).size == 2:
        row["auroc"] = float(roc_auc_score(labels, prediction))
        row["auprc"] = float(average_precision_score(labels, prediction))
    return row


def collect_observations(
    metadata: list[dict[str, str]],
    targets: np.ndarray,
    masks: np.ndarray,
    cached_tracks: dict[tuple[str, str], dict],
) -> tuple[list[dict], list[dict]]:
    sums: dict[tuple[str, str, int], float] = defaultdict(float)
    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    read_lengths: dict[tuple[str, str], int] = {}
    for idx, row in enumerate(metadata):
        sample = row.get("sample") or row.get("sample_id")
        read_id = row["read_id"]
        key = (sample, read_id)
        if key not in cached_tracks:
            continue
        read_lengths[key] = int(row["read_length"])
        length = min(int(row["window_length"]), targets.shape[1])
        window_start = int(row["window_start"])
        for local_pos in np.flatnonzero(masks[idx, :length]):
            read_pos = window_start + int(local_pos)
            sums[(sample, read_id, read_pos)] += float(targets[idx, local_pos])
            counts[(sample, read_id, read_pos)] += 1

    observations = []
    dropped = []
    for (sample, read_id, read_pos), total in sums.items():
        cache = cached_tracks[(sample, read_id)]
        read_length = read_lengths[(sample, read_id)]
        left_pad = (int(cache["sequence_length"]) - read_length) // 2
        output_index = (left_pad + read_pos) // int(cache["resolution"])
        track = cache["track"]
        if output_index < 0 or output_index >= len(track):
            dropped.append({"sample": sample, "read_id": read_id, "read_pos": read_pos})
            continue
        observations.append(
            {
                "sample": sample,
                "read_id": read_id,
                "read_pos": read_pos,
                "target_6ma": total / counts[(sample, read_id, read_pos)],
                "alphagenome_h3k4me3": float(track[output_index]),
                "duplicate_window_count": counts[(sample, read_id, read_pos)],
            }
        )
    return observations, dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AlphaGenome read-sequence predictions against read-level DiMeLo 6mA observations."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("cache_readseq_10reads"))
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/outputs"),
    )
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--positive-cutoff", type=float, default=0.5)
    parser.add_argument("--out-prefix", type=Path, default=Path("outputs/alphagenome_readlevel_10reads"))
    parser.add_argument("--write-observations", action="store_true")
    args = parser.parse_args()

    cached_tracks = load_cached_tracks(args.cache_dir)
    if not cached_tracks:
        raise SystemExit(f"No AlphaGenome read-sequence caches found in {args.cache_dir}")

    all_observations = []
    all_dropped = []
    for chrom in CHROMS:
        prefix = (
            args.outputs_dir
            / f"merged_e5b_c1_{chrom}_selected_top100_overlap16k_full5000_region_split.{args.split}"
        )
        metadata_path = Path(f"{prefix}.metadata.tsv")
        npz_path = Path(f"{prefix}.npz")
        metadata = read_tsv(metadata_path)
        archive = np.load(npz_path)
        targets = archive["target_6mA"]
        masks = archive["mask_6mA"].astype(bool)
        observations, dropped = collect_observations(metadata, targets, masks, cached_tracks)
        all_observations.extend(observations)
        all_dropped.extend(dropped)
        archive.close()

    obs = pd.DataFrame(all_observations)
    if obs.empty:
        raise SystemExit("No matched DiMeLo observations found for cached AlphaGenome reads.")

    summary_rows = []
    for sample, frame in list(obs.groupby("sample")) + [("pooled", obs)]:
        row = {"sample": sample, "number_of_reads": int(frame["read_id"].nunique())}
        row.update(
            metric_row(
                frame["target_6ma"].to_numpy(float),
                frame["alphagenome_h3k4me3"].to_numpy(float),
                args.positive_cutoff,
            )
        )
        summary_rows.append(row)

    per_read_rows = []
    for (sample, read_id), frame in obs.groupby(["sample", "read_id"]):
        row = {"sample": sample, "read_id": read_id}
        row.update(
            metric_row(
                frame["target_6ma"].to_numpy(float),
                frame["alphagenome_h3k4me3"].to_numpy(float),
                args.positive_cutoff,
            )
        )
        per_read_rows.append(row)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        f"{args.out_prefix}.summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(per_read_rows).to_csv(
        f"{args.out_prefix}.per_read.tsv", sep="\t", index=False
    )
    if args.write_observations:
        obs.to_csv(f"{args.out_prefix}.observations.tsv", sep="\t", index=False)
    with open(f"{args.out_prefix}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_dir": str(args.cache_dir),
                "split": args.split,
                "positive_cutoff": args.positive_cutoff,
                "cached_reads": len(cached_tracks),
                "matched_observations": int(len(obs)),
                "matched_reads": int(obs[["sample", "read_id"]].drop_duplicates().shape[0]),
                "dropped_observations": int(len(all_dropped)),
                "note": (
                    "AlphaGenome predicts A549 H3K4me3 from original read sequence; "
                    "these read-level metrics compare that cross-assay prediction to "
                    "DiMeLo 6mA probabilities at observed read positions."
                ),
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Wrote {args.out_prefix}.summary.tsv")


if __name__ == "__main__":
    main()
