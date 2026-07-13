#!/usr/bin/env python3
"""Population-level genomic track benchmark for external bigWig, AlphaGenome,
HyenaDNA, and DiMeLo-seq.

The script is intentionally evaluation-only: external bigWig and AlphaGenome
signals are never written as training targets. Existing read-level DiMeLo and
HyenaDNA summaries are aggregated onto a shared genomic bin grid before any
comparison with population-level tracks.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


TRACKS = ["external", "alphagenome", "hyena", "dimelo"]
PAIR_LABELS = {
    ("external", "alphagenome"): "T-A",
    ("external", "hyena"): "T-H",
    ("external", "dimelo"): "T-D",
    ("alphagenome", "hyena"): "A-H",
    ("alphagenome", "dimelo"): "A-D",
    ("hyena", "dimelo"): "H-D",
}
PRIMARY_COMPARISONS = {"T-A", "H-D"}
SOURCE_PROTECTED_SUFFIXES = {
    ".bw",
    ".bigwig",
    ".bam",
    ".bai",
    ".cram",
    ".crai",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".yaml",
    ".yml",
    ".json",
}


@dataclass
class TrackMetadata:
    """Human-readable metadata for one benchmark track."""

    track: str
    source_path: str
    genome_assembly: str
    chromosome_naming: str
    histone_mark: str
    cell_type_or_tissue: str
    assay_type: str
    coordinate_convention: str
    native_resolution: str
    value_type_units: str
    value_transform: str
    notes: str


@dataclass
class NormalizationParams:
    """Parameters fitted once per track over all valid evaluation bins."""

    track: str
    mean: float | None
    std: float | None
    p01: float | None
    p99: float | None
    n_valid: int


def now_iso() -> str:
    """Return a timestamp suitable for provenance files."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def extension(path: Path) -> str:
    """Return a useful compound extension for common compressed files."""

    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-1] in {".gz", ".bgz", ".bz2", ".xz"}:
        return "".join(suffixes[-2:])
    return path.suffix


def infer_file_role(path: Path) -> str:
    """Infer a coarse file role for the audit table."""

    name = path.name.lower()
    ext = extension(path).lower()
    if ext in {".bigwig", ".bw"}:
        return "source_external_bigwig"
    if ext in {".bam", ".cram", ".bai", ".crai"}:
        return "source_alignment"
    if ext in {".pt", ".pth", ".ckpt"}:
        return "model_checkpoint"
    if ext in {".yaml", ".yml", ".json", ".toml", ".ini"}:
        return "configuration_or_metadata"
    if ext in {".tsv", ".tsv.gz", ".csv", ".csv.gz", ".parquet"}:
        return "table"
    if ext in {".png", ".svg", ".pdf"}:
        return "figure"
    if ext == ".py":
        return "code"
    if ext in {".pyc"} or "__pycache__" in path.parts:
        return "temporary_python_cache"
    if name.startswith("slurm-") or ext in {".out", ".err", ".log"}:
        return "run_log"
    if ext in {".tar", ".tgz", ".gz"}:
        return "downloaded_archive_or_compressed_file"
    return "unknown"


def cleanup_decision(path: Path, role: str, protected_roots: list[Path]) -> tuple[bool, str]:
    """Return whether a file is a candidate for deletion and why."""

    resolved = path.resolve()
    if any(str(resolved).startswith(str(root.resolve())) for root in protected_roots):
        return False, "protected input or output root"
    low = path.name.lower()
    if extension(path).lower() in SOURCE_PROTECTED_SUFFIXES:
        return False, "protected source, checkpoint, config or final metadata type"
    if role == "temporary_python_cache":
        return True, "temporary Python bytecode/cache"
    if role == "run_log" and (low.endswith(".err") or low.endswith(".out")):
        return True, "scheduler log; safe only after manual inspection"
    if "smoke" in str(path).lower() and role in {"table", "figure", "run_log"}:
        return True, "smoke/intermediate output"
    return False, "retained by default"


def write_inventory(
    roots: list[Path],
    out_dir: Path,
    protected_roots: list[Path],
    apply_cleanup: bool,
) -> None:
    """Recursively inventory files and write a proposed cleanup manifest."""

    rows = []
    deletion_rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            role = infer_file_role(path)
            candidate, reason = cleanup_decision(path, role, protected_roots)
            row = {
                "path": str(path),
                "filename": path.name,
                "extension": extension(path),
                "file_size_bytes": stat.st_size,
                "modification_time": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                "inferred_role": role,
                "required": not candidate,
                "reason": reason,
            }
            rows.append(row)
            if candidate:
                deletion_rows.append(row | {"delete_if_apply_cleanup": True})

    inventory = pd.DataFrame(rows).sort_values(["inferred_role", "path"]) if rows else pd.DataFrame()
    manifest = pd.DataFrame(deletion_rows).sort_values(["inferred_role", "path"]) if deletion_rows else pd.DataFrame()
    write_table(inventory, out_dir / "file_inventory.tsv.gz")
    write_table(manifest, out_dir / "proposed_deletion_manifest.tsv.gz")
    if apply_cleanup:
        for row in deletion_rows:
            path = Path(row["path"])
            if extension(path).lower() in SOURCE_PROTECTED_SUFFIXES:
                continue
            if path.exists() and path.is_file():
                path.unlink()


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a table, using gzip compression for .gz suffixes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(path, sep="\t", index=False, compression=compression)


def read_table(path: Path) -> pd.DataFrame:
    """Read a TSV/TSV.GZ table."""

    return pd.read_csv(path, sep="\t")


def parse_chromosomes(value: str) -> list[str]:
    """Parse comma-separated chromosome names."""

    return [item.strip() for item in value.split(",") if item.strip()]


def chrom_style(chroms: Iterable[str]) -> str:
    """Describe chromosome naming convention."""

    observed = [str(chrom) for chrom in chroms if pd.notna(chrom)]
    if not observed:
        return "unknown"
    has_chr = sum(chrom.startswith("chr") for chrom in observed)
    if has_chr == len(observed):
        return "chr-prefixed"
    if has_chr == 0:
        return "bare"
    return "mixed"


def load_regions(path: Path, chromosomes: list[str], include_alt: bool, include_mito: bool) -> pd.DataFrame:
    """Load held-out regions and restrict them to requested chromosomes."""

    regions = read_table(path)
    required = {"chrom", "start", "end"}
    if not required.issubset(regions.columns):
        raise SystemExit(f"Region table lacks required columns {sorted(required)}: {path}")
    regions = regions[regions["chrom"].isin(chromosomes)].copy()
    if not include_alt:
        regions = regions[~regions["chrom"].astype(str).str.contains("_|random|alt|fix", case=False, regex=True)]
    if not include_mito:
        regions = regions[~regions["chrom"].isin(["chrM", "MT", "M"])]
    if regions.empty:
        raise SystemExit("No evaluation regions remain after chromosome/contig filtering.")
    if "region_id" not in regions.columns:
        regions["region_id"] = np.arange(1, len(regions) + 1)
    if "name" in regions.columns and "region_name" not in regions.columns:
        regions = regions.rename(columns={"name": "region_name"})
    if "region_name" not in regions.columns:
        regions["region_name"] = regions["chrom"] + ":" + regions["start"].astype(str) + "-" + regions["end"].astype(str)
    return regions[["chrom", "region_id", "region_name", "start", "end"]].rename(
        columns={"start": "region_start", "end": "region_end"}
    )


def canonical_bins(regions: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    """Create non-overlapping canonical bins within evaluation regions."""

    rows = []
    for region in regions.itertuples(index=False):
        start = int(region.region_start)
        end = int(region.region_end)
        for bin_start in range(start, end - bin_size + 1, bin_size):
            rows.append(
                {
                    "chrom": region.chrom,
                    "region_id": region.region_id,
                    "region_name": region.region_name,
                    "region_start": start,
                    "region_end": end,
                    "start": bin_start,
                    "end": bin_start + bin_size,
                }
            )
    if not rows:
        raise SystemExit("Canonical bin table is empty. Check region widths and bin size.")
    return pd.DataFrame(rows)


def exclude_blacklist_bins(bins: pd.DataFrame, blacklist_bed: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude bins overlapping a BED blacklist and return an exclusion report."""

    blacklist = read_bed(blacklist_bed)
    keep = np.ones(len(bins), dtype=bool)
    reasons = []
    for chrom, intervals in blacklist.groupby("chrom", sort=False):
        indices = bins.index[bins["chrom"] == chrom].to_numpy()
        if indices.size == 0:
            continue
        chrom_bins = bins.loc[indices]
        for item in intervals.itertuples(index=False):
            overlap = (chrom_bins["start"].to_numpy(int) < int(item.end)) & (
                chrom_bins["end"].to_numpy(int) > int(item.start)
            )
            hit_indices = indices[overlap]
            keep[np.isin(bins.index.to_numpy(), hit_indices)] = False
            for idx in hit_indices:
                reasons.append(
                    {
                        "chrom": bins.loc[idx, "chrom"],
                        "start": int(bins.loc[idx, "start"]),
                        "end": int(bins.loc[idx, "end"]),
                        "reason": f"overlaps blacklist {chrom}:{int(item.start)}-{int(item.end)}",
                    }
                )
    return bins.loc[keep].reset_index(drop=True), pd.DataFrame(reasons)


def interval_overlap(starts: np.ndarray, ends: np.ndarray, target_start: int, target_end: int) -> np.ndarray:
    """Return overlap length between source intervals and one target interval."""

    return np.maximum(0, np.minimum(ends, target_end) - np.maximum(starts, target_start)).astype(float)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a NaN-aware weighted mean."""

    mask = np.isfinite(values) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def aggregate_alpha(alpha: pd.DataFrame, bins: pd.DataFrame, alpha_col: str) -> pd.DataFrame:
    """Overlap-weight AlphaGenome intervals onto canonical bins."""

    if alpha_col not in alpha.columns:
        raise SystemExit(f"AlphaGenome column not found: {alpha_col}")
    rows = []
    alpha = alpha.rename(columns={"bin_start": "source_start", "bin_end": "source_end"}).copy()
    for region_key, region_bins in bins.groupby(["chrom", "region_id"], sort=False):
        chrom, region_id = region_key
        subset = alpha[(alpha["chrom"] == chrom) & (alpha["region_id"] == region_id)]
        starts = subset["source_start"].to_numpy(int)
        ends = subset["source_end"].to_numpy(int)
        values = subset[alpha_col].to_numpy(float)
        for row in region_bins.itertuples(index=False):
            weights = interval_overlap(starts, ends, int(row.start), int(row.end))
            value = weighted_mean(values, weights)
            rows.append(
                {
                    "chrom": row.chrom,
                    "region_id": row.region_id,
                    "region_name": row.region_name,
                    "start": row.start,
                    "end": row.end,
                    "alphagenome_raw": value,
                    "alphagenome_valid": np.isfinite(value),
                }
            )
    return pd.DataFrame(rows)


def aggregate_read_track(
    track: pd.DataFrame,
    bins: pd.DataFrame,
    sample: str,
    value_col: str,
    output_prefix: str,
) -> pd.DataFrame:
    """Aggregate DiMeLo or HyenaDNA binned read summaries onto canonical bins."""

    if sample != "pooled":
        track = track[track["sample"] == sample].copy()
    required = {"chrom", "region_id", "bin_start", "bin_end", value_col, "observed_positions", "unique_reads"}
    missing = required - set(track.columns)
    if missing:
        raise SystemExit(f"{output_prefix} input lacks required columns: {sorted(missing)}")
    rows = []
    source = track.rename(columns={"bin_start": "source_start", "bin_end": "source_end"}).copy()
    source["weight"] = source["observed_positions"].fillna(0).astype(float)
    source["weighted_value"] = source[value_col].astype(float) * source["weight"]
    for region_key, region_bins in bins.groupby(["chrom", "region_id"], sort=False):
        chrom, region_id = region_key
        subset = source[(source["chrom"] == chrom) & (source["region_id"] == region_id)]
        starts = subset["source_start"].to_numpy(int)
        ends = subset["source_end"].to_numpy(int)
        values = subset[value_col].to_numpy(float)
        weights_base = subset["weight"].to_numpy(float)
        reads = subset["unique_reads"].fillna(0).to_numpy(float)
        for row in region_bins.itertuples(index=False):
            overlap = interval_overlap(starts, ends, int(row.start), int(row.end))
            weights = weights_base * overlap
            value = weighted_mean(values, weights)
            overlapping = overlap > 0
            coverage = float(np.nansum(weights_base[overlapping])) if np.any(overlapping) else 0.0
            read_count = float(np.nanmax(reads[overlapping])) if np.any(overlapping) else 0.0
            entry = {
                "chrom": row.chrom,
                "region_id": row.region_id,
                "region_name": row.region_name,
                "start": row.start,
                "end": row.end,
                f"{output_prefix}_raw": value,
                f"{output_prefix}_coverage": coverage,
                f"{output_prefix}_read_count": read_count,
                f"{output_prefix}_valid": np.isfinite(value) and coverage > 0,
            }
            if output_prefix == "hyena":
                finite_values = values[np.isfinite(values) & overlapping] if np.any(overlapping) else np.array([])
                entry["hyena_median_prediction"] = float(np.nanmedian(finite_values)) if finite_values.size else np.nan
                entry["hyena_prediction_count"] = coverage
            rows.append(entry)
    return pd.DataFrame(rows)


def sample_bigwig(
    bins: pd.DataFrame,
    bigwig_path: Path,
    expected_assembly: str,
    allow_assembly_mismatch: bool,
) -> tuple[pd.DataFrame, dict]:
    """Sample mean bigWig signal and base coverage fraction in each bin."""

    try:
        import pyBigWig
    except ImportError as exc:
        raise SystemExit("pyBigWig is required. Install it in the active venv.") from exc

    if not bigwig_path.exists():
        raise SystemExit(f"External bigWig not found: {bigwig_path}")
    values: list[float] = []
    fractions: list[float] = []
    with pyBigWig.open(str(bigwig_path)) as bw:
        chrom_sizes = bw.chroms()
        header = bw.header()
        bw_style = chrom_style(chrom_sizes.keys())
        bin_style = chrom_style(bins["chrom"].unique())
        if bw_style != bin_style:
            missing = sorted(set(bins["chrom"].unique()) - set(chrom_sizes.keys()))
            raise SystemExit(
                "Chromosome naming differs between bins and bigWig. "
                f"bin_style={bin_style}, bigwig_style={bw_style}, missing_examples={missing[:5]}"
            )
        if expected_assembly.lower() not in {"hg38", "grch38"} and not allow_assembly_mismatch:
            raise SystemExit(
                f"Expected assembly {expected_assembly!r} is not hg38/GRCh38. "
                "Provide liftover/conversion before comparison or use --allow-assembly-mismatch."
            )
        for row in bins.itertuples(index=False):
            if row.chrom not in chrom_sizes:
                values.append(np.nan)
                fractions.append(0.0)
                continue
            try:
                stat = bw.stats(row.chrom, int(row.start), int(row.end), type="mean", exact=True)[0]
                cov = bw.stats(row.chrom, int(row.start), int(row.end), type="coverage", exact=True)[0]
            except RuntimeError:
                stat = None
                cov = None
            values.append(float(stat) if stat is not None else np.nan)
            fractions.append(float(cov) if cov is not None else 0.0)
    frame = bins[["chrom", "region_id", "region_name", "start", "end"]].copy()
    frame["external_raw"] = values
    frame["external_covered_fraction"] = fractions
    frame["external_valid"] = np.isfinite(frame["external_raw"].astype(float))
    metadata = {
        "bigwig_header": header,
        "chromosome_naming": bw_style,
        "number_of_chromosomes": len(chrom_sizes),
    }
    return frame, metadata


def normalize_track(values: pd.Series, track: str) -> tuple[pd.DataFrame, NormalizationParams]:
    """Create z-score and robust percentile-normalized columns."""

    numeric = values.astype(float)
    valid = numeric[np.isfinite(numeric)]
    if valid.empty:
        params = NormalizationParams(track, None, None, None, None, 0)
        return pd.DataFrame({f"{track}_zscore": np.nan, f"{track}_robust01": np.nan}), params
    mean = float(valid.mean())
    std = float(valid.std(ddof=0))
    p01 = float(valid.quantile(0.01))
    p99 = float(valid.quantile(0.99))
    z = (numeric - mean) / std if std > 0 else numeric * np.nan
    robust = ((numeric - p01) / (p99 - p01)).clip(0, 1) if p99 > p01 else numeric * np.nan
    params = NormalizationParams(track, mean, std, p01, p99, int(valid.size))
    return pd.DataFrame({f"{track}_zscore": z, f"{track}_robust01": robust}), params


def apply_normalization(values: pd.Series, params: NormalizationParams) -> pd.DataFrame:
    """Apply previously fitted normalization parameters to another track instance."""

    numeric = values.astype(float)
    if params.mean is None or params.std in {None, 0}:
        z = numeric * np.nan
    else:
        z = (numeric - params.mean) / params.std
    if params.p01 is None or params.p99 is None or params.p99 <= params.p01:
        robust = numeric * np.nan
    else:
        robust = ((numeric - params.p01) / (params.p99 - params.p01)).clip(0, 1)
    return pd.DataFrame({"zscore": z, "robust01": robust})


def add_normalized_columns(canonical: pd.DataFrame) -> tuple[pd.DataFrame, list[NormalizationParams]]:
    """Fit normalization once per complete track across all evaluation bins."""

    normalized = canonical.copy()
    params: list[NormalizationParams] = []
    for track in TRACKS:
        cols, param = normalize_track(normalized[f"{track}_raw"], track)
        normalized = pd.concat([normalized, cols], axis=1)
        params.append(param)
    return normalized, params


def params_by_track(params: list[NormalizationParams]) -> dict[str, NormalizationParams]:
    """Index normalization parameters by track name."""

    return {param.track: param for param in params}


def metric_values(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    x_norm: np.ndarray | None = None,
    y_norm: np.ndarray | None = None,
) -> dict:
    """Calculate raw-value correlations and normalized-distance metrics."""

    result = {
        "n_bins": int(x_raw.size),
        "pearson": np.nan,
        "spearman": np.nan,
        "normalized_mae": np.nan,
        "normalized_rmse": np.nan,
        "r2_squared_correlation": np.nan,
        "r2_predictive": np.nan,
    }
    if x_raw.size < 2:
        return result
    if np.std(x_raw) > 0 and np.std(y_raw) > 0:
        pearson = float(pearsonr(x_raw, y_raw).statistic)
        result["pearson"] = pearson
        result["spearman"] = float(spearmanr(x_raw, y_raw).statistic)
        result["r2_squared_correlation"] = pearson**2
    if x_norm is not None and y_norm is not None and x_norm.size:
        result["normalized_mae"] = float(np.mean(np.abs(x_norm - y_norm)))
        result["normalized_rmse"] = float(np.sqrt(np.mean((x_norm - y_norm) ** 2)))
    denominator = float(np.sum((x_raw - np.mean(x_raw)) ** 2))
    if denominator > 0:
        result["r2_predictive"] = float(1.0 - np.sum((x_raw - y_raw) ** 2) / denominator)
    return result


def block_bootstrap_ci(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    block_size: int,
    replicates: int,
    seed: int,
) -> dict:
    """Estimate CIs using genomic block bootstrap."""

    if replicates <= 0:
        return {"pearson_ci_low": np.nan, "pearson_ci_high": np.nan, "spearman_ci_low": np.nan, "spearman_ci_high": np.nan}
    blocks = []
    for chrom, subset in data.groupby("chrom", sort=False):
        block_id = (subset["start"].astype(int) // block_size).to_numpy()
        tmp = subset.assign(_block=[f"{chrom}:{bid}" for bid in block_id])
        blocks.extend([group for _, group in tmp.groupby("_block", sort=False)])
    if len(blocks) < 2:
        return {"pearson_ci_low": np.nan, "pearson_ci_high": np.nan, "spearman_ci_low": np.nan, "spearman_ci_high": np.nan}
    rng = np.random.default_rng(seed)
    pearsons = []
    spearmans = []
    for _ in range(replicates):
        sampled = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        frame = pd.concat(sampled, ignore_index=True)
        row = metric_values(frame[x_col].to_numpy(float), frame[y_col].to_numpy(float))
        if np.isfinite(row["pearson"]):
            pearsons.append(row["pearson"])
        if np.isfinite(row["spearman"]):
            spearmans.append(row["spearman"])
    return {
        "pearson_ci_low": float(np.quantile(pearsons, 0.025)) if pearsons else np.nan,
        "pearson_ci_high": float(np.quantile(pearsons, 0.975)) if pearsons else np.nan,
        "spearman_ci_low": float(np.quantile(spearmans, 0.025)) if spearmans else np.nan,
        "spearman_ci_high": float(np.quantile(spearmans, 0.975)) if spearmans else np.nan,
    }


def pairwise_metrics(
    data: pd.DataFrame,
    use_common_intersection: bool,
    coverage_threshold: int,
    block_size: int,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pairwise and chromosome-specific metrics."""

    rows = []
    chrom_rows = []
    common = np.ones(len(data), dtype=bool)
    if use_common_intersection:
        for track in TRACKS:
            common &= data[f"{track}_valid"].to_numpy(bool)
        common &= data["dimelo_coverage"].fillna(0).to_numpy(float) >= coverage_threshold

    for left, right in PAIR_LABELS:
        pair_valid = data[f"{left}_valid"].to_numpy(bool) & data[f"{right}_valid"].to_numpy(bool)
        if "dimelo" in {left, right}:
            pair_valid &= data["dimelo_coverage"].fillna(0).to_numpy(float) >= coverage_threshold
        if use_common_intersection:
            pair_valid &= common
        subset = data[pair_valid].copy()
        x_raw_col = f"{left}_raw"
        y_raw_col = f"{right}_raw"
        x_norm_col = f"{left}_robust01"
        y_norm_col = f"{right}_robust01"
        subset = subset[
            np.isfinite(subset[x_raw_col])
            & np.isfinite(subset[y_raw_col])
            & np.isfinite(subset[x_norm_col])
            & np.isfinite(subset[y_norm_col])
        ]
        label = PAIR_LABELS[(left, right)]
        row = {
            "comparison": label,
            "comparison_role": "primary_model_to_target" if label in PRIMARY_COMPARISONS else "secondary_cross_assay",
            "left_track": left,
            "right_track": right,
            "scope": "all",
            "intersection": "common_four_track" if use_common_intersection else "pair_specific",
            "min_dimelo_coverage": coverage_threshold,
            "bootstrap_block_size": block_size,
        }
        row.update(
            metric_values(
                subset[x_raw_col].to_numpy(float),
                subset[y_raw_col].to_numpy(float),
                subset[x_norm_col].to_numpy(float),
                subset[y_norm_col].to_numpy(float),
            )
        )
        row.update(block_bootstrap_ci(subset, x_raw_col, y_raw_col, block_size, bootstrap_replicates, seed))
        rows.append(row)
        for chrom, chrom_subset in subset.groupby("chrom", sort=False):
            chrom_row = row.copy()
            chrom_row["scope"] = chrom
            chrom_row.update(
                metric_values(
                    chrom_subset[x_raw_col].to_numpy(float),
                    chrom_subset[y_raw_col].to_numpy(float),
                    chrom_subset[x_norm_col].to_numpy(float),
                    chrom_subset[y_norm_col].to_numpy(float),
                )
            )
            chrom_rows.append(chrom_row)
    return pd.DataFrame(rows), pd.DataFrame(chrom_rows)


def coverage_sensitivity(
    data: pd.DataFrame,
    thresholds: list[int],
    block_size: int,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Repeat DiMeLo-involving comparisons across coverage thresholds."""

    rows = []
    for threshold in thresholds:
        metrics, _ = pairwise_metrics(data, False, threshold, block_size, bootstrap_replicates, seed)
        rows.append(metrics[metrics["comparison"].isin(["T-D", "A-D", "H-D"])])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Calculate weighted Pearson correlation."""

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[mask]
    y = y[mask]
    w = w[mask]
    if x.size < 2:
        return np.nan
    wx = np.average(x, weights=w)
    wy = np.average(y, weights=w)
    cov = np.average((x - wx) * (y - wy), weights=w)
    vx = np.average((x - wx) ** 2, weights=w)
    vy = np.average((y - wy) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(cov / math.sqrt(vx * vy))


def weighted_correlation_table(data: pd.DataFrame, min_coverage: int) -> pd.DataFrame:
    """Calculate DiMeLo-coverage-weighted Pearson correlations."""

    rows = []
    for left, right in PAIR_LABELS:
        valid = data[f"{left}_valid"] & data[f"{right}_valid"] & (data["dimelo_coverage"].fillna(0) >= min_coverage)
        subset = data[valid]
        rows.append(
            {
                "comparison": PAIR_LABELS[(left, right)],
                "weighted_by": "dimelo_coverage",
                "min_dimelo_coverage": min_coverage,
                "n_bins": int(subset.shape[0]),
                "weighted_pearson": weighted_corr(
                    subset[f"{left}_raw"].to_numpy(float),
                    subset[f"{right}_raw"].to_numpy(float),
                    subset["dimelo_coverage"].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(matrix: pd.DataFrame, title: str, out_base: Path) -> None:
    """Plot an annotated correlation heatmap without seaborn."""

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    values = matrix.to_numpy(float)
    image = ax.imshow(values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = "NA" if not np.isfinite(values[i, j]) else f"{values[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, out_base)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    """Save a figure as PNG and SVG."""

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def correlation_matrices(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Convert pairwise metric rows into a square matrix."""

    names = {"external": "external BigWig", "alphagenome": "AlphaGenome", "hyena": "HyenaDNA", "dimelo": "DiMeLo"}
    matrix = pd.DataFrame(np.eye(4), index=[names[t] for t in TRACKS], columns=[names[t] for t in TRACKS])
    for row in metrics.itertuples(index=False):
        left = names[row.left_track]
        right = names[row.right_track]
        value = getattr(row, metric)
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    return matrix


def make_pairwise_density_plots(data: pd.DataFrame, out_dir: Path, min_coverage: int) -> None:
    """Create pairwise hexbin plots for all six track pairs."""

    labels = {"external": "external BigWig", "alphagenome": "AlphaGenome", "hyena": "HyenaDNA", "dimelo": "DiMeLo"}
    for left, right in PAIR_LABELS:
        valid = data[f"{left}_valid"] & data[f"{right}_valid"]
        if "dimelo" in {left, right}:
            valid &= data["dimelo_coverage"].fillna(0) >= min_coverage
        subset = data[valid].copy()
        x = subset[f"{left}_robust01"].to_numpy(float)
        y = subset[f"{right}_robust01"].to_numpy(float)
        x_raw = subset[f"{left}_raw"].to_numpy(float)
        y_raw = subset[f"{right}_raw"].to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(x_raw) & np.isfinite(y_raw)
        x = x[mask]
        y = y[mask]
        x_raw = x_raw[mask]
        y_raw = y_raw[mask]
        stats = metric_values(x_raw, y_raw, x, y)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        if x.size:
            ax.hexbin(x, y, gridsize=55, mincnt=1, cmap="viridis", bins="log")
            if np.std(x) > 0 and np.std(y) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                xx = np.linspace(0, 1, 100)
                ax.plot(xx, slope * xx + intercept, color="black", lw=1)
            ax.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle=":")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(f"{labels[left]} robust normalized")
        ax.set_ylabel(f"{labels[right]} robust normalized")
        ax.set_title(
            f"{PAIR_LABELS[(left, right)]}: r={stats['pearson']:.3f}, "
            f"rho={stats['spearman']:.3f}, n={stats['n_bins']}"
        )
        fig.tight_layout()
        save_figure(fig, out_dir / f"density_{PAIR_LABELS[(left, right)].replace('-', '_')}")


def make_chromosome_plot(chrom_metrics: pd.DataFrame, out_base: Path) -> None:
    """Plot Pearson and Spearman by chromosome and comparison."""

    if chrom_metrics.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, metric in zip(axes, ["pearson", "spearman"]):
        pivot = chrom_metrics.pivot_table(index="scope", columns="comparison", values=metric, aggfunc="first")
        x = np.arange(len(pivot.index))
        width = 0.12
        for idx, col in enumerate(pivot.columns):
            ax.plot(x + (idx - len(pivot.columns) / 2) * width, pivot[col], marker="o", linestyle="", label=col)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel(metric)
        ax.set_xticks(x, pivot.index)
        ax.set_ylim(-1, 1)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("chromosome")
    fig.tight_layout()
    save_figure(fig, out_base)


def make_coverage_plot(coverage: pd.DataFrame, out_base: Path) -> None:
    """Plot correlation as a function of minimum DiMeLo coverage."""

    if coverage.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, metric in zip(axes, ["pearson", "spearman"]):
        for comparison, subset in coverage.groupby("comparison", sort=False):
            ax.plot(subset["min_dimelo_coverage"], subset[metric], marker="o", label=comparison)
        ax.set_xlabel("minimum DiMeLo observations")
        ax.set_ylabel(metric)
        ax.set_ylim(-1, 1)
        ax.set_title(metric)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_base)


def select_representative_loci(data: pd.DataFrame, min_coverage: int, window_bins: int = 10) -> pd.DataFrame:
    """Select representative loci using local normalized MAE/correlation criteria."""

    rows = []
    valid = data[data["dimelo_coverage"].fillna(0) >= min_coverage].copy()
    for (chrom, region_id), region in valid.groupby(["chrom", "region_id"], sort=False):
        region = region.sort_values("start")
        for offset in range(0, max(1, len(region) - window_bins + 1), max(1, window_bins // 2)):
            window = region.iloc[offset : offset + window_bins]
            if len(window) < max(4, window_bins // 2):
                continue
            local = {"chrom": chrom, "region_id": region_id, "start": int(window["start"].min()), "end": int(window["end"].max())}
            for left, right in PAIR_LABELS:
                mask = window[f"{left}_valid"] & window[f"{right}_valid"]
                x = window.loc[mask, f"{left}_robust01"].to_numpy(float)
                y = window.loc[mask, f"{right}_robust01"].to_numpy(float)
                local[f"{PAIR_LABELS[(left, right)]}_mae"] = float(np.mean(np.abs(x - y))) if x.size else np.nan
                local[f"{PAIR_LABELS[(left, right)]}_pearson"] = metric_values(x, y)["pearson"] if x.size >= 2 else np.nan
            rows.append(local)
    if not rows:
        return pd.DataFrame()
    candidates = pd.DataFrame(rows)
    selected = []
    definitions = {
        "all_four_agree": ("sum_mae", True),
        "alphagenome_matches_external": ("T-A_mae_minus_T-H_mae", True),
        "hyena_matches_dimelo": ("H-D_mae_minus_A-D_mae", True),
        "dimelo_external_disagree": ("T-D_mae", False),
    }
    candidates["sum_mae"] = candidates[["T-A_mae", "T-H_mae", "T-D_mae", "A-H_mae", "A-D_mae", "H-D_mae"]].sum(axis=1)
    candidates["T-A_mae_minus_T-H_mae"] = candidates["T-A_mae"] - candidates["T-H_mae"]
    candidates["H-D_mae_minus_A-D_mae"] = candidates["H-D_mae"] - candidates["A-D_mae"]
    for label, (column, ascending) in definitions.items():
        row = candidates.sort_values(column, ascending=ascending).head(1).copy()
        row["selection_reason"] = label
        selected.append(row)
    return pd.concat(selected, ignore_index=True)


def plot_locus(data: pd.DataFrame, locus: pd.Series, out_dir: Path) -> None:
    """Create browser-style four-track plot for one selected locus."""

    subset = data[(data["chrom"] == locus["chrom"]) & (data["start"] >= locus["start"]) & (data["end"] <= locus["end"])].sort_values("start")
    x = (subset["start"] + subset["end"]) / 2
    fig, axes = plt.subplots(5, 1, figsize=(13, 8), sharex=True)
    plot_specs = [
        ("external_robust01", "external BigWig", "tab:green"),
        ("alphagenome_robust01", "AlphaGenome", "black"),
        ("hyena_robust01", "HyenaDNA", "tab:purple"),
        ("dimelo_robust01", "DiMeLo", "tab:orange"),
    ]
    for ax, (col, label, color) in zip(axes[:4], plot_specs):
        ax.plot(x, subset[col], color=color, lw=1.4)
        ax.set_ylabel(label)
        ax.set_ylim(-0.05, 1.05)
    axes[4].bar(x, subset["dimelo_coverage"], width=(subset["end"] - subset["start"]).median() * 0.9, color="gray")
    axes[4].set_ylabel("DiMeLo\ncoverage")
    axes[4].set_xlabel(f"{locus['chrom']} coordinate")
    fig.suptitle(f"{locus['selection_reason']}: {locus['chrom']}:{int(locus['start'])}-{int(locus['end'])}")
    fig.tight_layout()
    save_figure(fig, out_dir / f"{locus['selection_reason']}_{locus['chrom']}_{int(locus['start'])}_{int(locus['end'])}")


def make_stacked_region_plots(
    normalized: pd.DataFrame,
    bins: pd.DataFrame,
    dimelo_source: pd.DataFrame,
    hyena_source: pd.DataFrame,
    norm_params: dict[str, NormalizationParams],
    out_dir: Path,
    max_regions: int,
) -> None:
    """Plot selected regions with one panel per population-level track.

    ENCSR203XPU and AlphaGenome are shown once. DiMeLo and HyenaDNA are shown
    separately for C1 and E5B. All signal panels use normalization parameters
    fitted globally on the pooled evaluation set.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    regions = normalized[["chrom", "region_id", "region_name"]].drop_duplicates().head(max_regions)
    dimelo_by_sample = {}
    hyena_by_sample = {}
    for sample in ("merged_c1", "merged_e5b"):
        d = aggregate_read_track(dimelo_source, bins, sample, "mean_signal", "dimelo")
        h = aggregate_read_track(hyena_source, bins, sample, "mean_signal", "hyena")
        d_norm = apply_normalization(d["dimelo_raw"], norm_params["dimelo"])
        h_norm = apply_normalization(h["hyena_raw"], norm_params["hyena"])
        d["dimelo_robust01"] = d_norm["robust01"]
        h["hyena_robust01"] = h_norm["robust01"]
        dimelo_by_sample[sample] = d
        hyena_by_sample[sample] = h

    for selected in regions.itertuples(index=False):
        subset = normalized[
            (normalized["chrom"] == selected.chrom)
            & (normalized["region_id"] == selected.region_id)
        ].sort_values("start")
        if subset.empty:
            continue
        fig, axes = plt.subplots(7, 1, figsize=(14, 10), sharex=True)
        x = (subset["start"] + subset["end"]) / 2
        panels = [
            (axes[0], x, subset["external_robust01"], "ENCSR203XPU\nH3K4me3", "tab:green", "-"),
            (axes[1], x, subset["alphagenome_robust01"], "AlphaGenome\nA549 H3K4me3", "black", "-"),
        ]
        for ax, xs, ys, label, color, style in panels:
            ax.plot(xs, ys, color=color, linestyle=style, lw=1.3)
            ax.set_ylabel(label)
            ax.set_ylim(-0.05, 1.05)
        plot_specs = [
            ("merged_c1", axes[2], axes[3], "tab:blue", "tab:purple", "C1"),
            ("merged_e5b", axes[4], axes[5], "tab:orange", "deeppink", "E5B"),
        ]
        coverage_frames = []
        for sample, dimelo_ax, hyena_ax, dimelo_color, hyena_color, label in plot_specs:
            d = dimelo_by_sample[sample]
            h = hyena_by_sample[sample]
            d = d[
                (d["chrom"] == selected.chrom)
                & (d["region_id"] == selected.region_id)
                & (d["start"].isin(subset["start"]))
            ].sort_values("start")
            h = h[
                (h["chrom"] == selected.chrom)
                & (h["region_id"] == selected.region_id)
                & (h["start"].isin(subset["start"]))
            ].sort_values("start")
            dimelo_ax.plot((d["start"] + d["end"]) / 2, d["dimelo_robust01"], color=dimelo_color, lw=1.1)
            dimelo_ax.set_ylabel(f"DiMeLo\n{label}")
            dimelo_ax.set_ylim(-0.05, 1.05)
            hyena_ax.plot((h["start"] + h["end"]) / 2, h["hyena_robust01"], color=hyena_color, lw=1.1)
            hyena_ax.set_ylabel(f"HyenaDNA\n{label}")
            hyena_ax.set_ylim(-0.05, 1.05)
            coverage_frames.append(d[["start", "end", "dimelo_coverage"]].assign(sample=sample))
        coverage = pd.concat(coverage_frames, ignore_index=True)
        for sample, color in (("merged_c1", "tab:blue"), ("merged_e5b", "tab:orange")):
            c = coverage[coverage["sample"] == sample]
            axes[6].plot((c["start"] + c["end"]) / 2, c["dimelo_coverage"], color=color, lw=1.0, label=sample)
        axes[6].set_ylabel("DiMeLo\ncoverage")
        axes[6].set_xlabel(f"{selected.chrom} coordinate (hg38)")
        axes[6].legend(frameon=False, ncol=2, fontsize=8)
        fig.suptitle(f"{selected.chrom} region {selected.region_id}: globally normalized population-level tracks\n{selected.region_name}")
        fig.tight_layout()
        save_figure(fig, out_dir / f"{selected.chrom}_region{selected.region_id}_stacked_global_normalized")


def make_overlay_region_plots(
    normalized: pd.DataFrame,
    out_dir: Path,
    max_regions: int,
) -> None:
    """Create optional overlay plots using global robust normalization."""

    out_dir.mkdir(parents=True, exist_ok=True)
    regions = normalized[["chrom", "region_id", "region_name"]].drop_duplicates().head(max_regions)
    for selected in regions.itertuples(index=False):
        subset = normalized[
            (normalized["chrom"] == selected.chrom)
            & (normalized["region_id"] == selected.region_id)
        ].sort_values("start")
        x = (subset["start"] + subset["end"]) / 2
        fig, ax = plt.subplots(figsize=(13, 5))
        for col, label, color, style in [
            ("external_robust01", "ENCSR203XPU", "tab:green", "-"),
            ("alphagenome_robust01", "AlphaGenome", "black", "-"),
            ("dimelo_robust01", "DiMeLo pooled", "tab:orange", "-"),
            ("hyena_robust01", "HyenaDNA pooled", "tab:purple", "--"),
        ]:
            ax.plot(x, subset[col], label=label, color=color, linestyle=style, lw=1.2)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f"{selected.chrom} coordinate (hg38)")
        ax.set_ylabel("global robust normalized signal")
        ax.legend(frameon=False, ncol=2)
        fig.suptitle(f"{selected.chrom} region {selected.region_id}: overlay\n{selected.region_name}")
        fig.tight_layout()
        save_figure(fig, out_dir / f"{selected.chrom}_region{selected.region_id}_overlay_global_normalized")


def read_bed(path: Path) -> pd.DataFrame:
    """Read a BED file with at least three columns."""

    frame = pd.read_csv(path, sep="\t", header=None, comment="#")
    if frame.shape[1] < 3:
        raise SystemExit(f"BED file has fewer than 3 columns: {path}")
    columns = ["chrom", "start", "end"] + [f"col{i}" for i in range(4, frame.shape[1] + 1)]
    frame.columns = columns[: frame.shape[1]]
    return frame


def average_profiles(
    data: pd.DataFrame,
    bed_paths: list[Path],
    flank: int,
    profile_bin_size: int,
    seed: int,
    out_dir: Path,
) -> pd.DataFrame:
    """Create average landmark profiles from canonical binned tracks."""

    if not bed_paths:
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(seed)
    for bed_path in bed_paths:
        if not bed_path.exists():
            continue
        bed = read_bed(bed_path)
        retained = 0
        profiles = {track: [] for track in TRACKS}
        positions = np.arange(-flank, flank, profile_bin_size)
        for landmark in bed.itertuples(index=False):
            center = int((landmark.start + landmark.end) // 2)
            window = data[(data["chrom"] == landmark.chrom) & (data["start"] >= center - flank) & (data["end"] <= center + flank)]
            if window.empty:
                continue
            retained += 1
            for track in TRACKS:
                vals = []
                for rel in positions:
                    start = center + int(rel)
                    end = start + profile_bin_size
                    hit = window[(window["start"] < end) & (window["end"] > start)]
                    vals.append(float(hit[f"{track}_robust01"].mean()) if not hit.empty else np.nan)
                profiles[track].append(vals)
        for idx, rel in enumerate(positions):
            for track in TRACKS:
                matrix = np.asarray(profiles[track], dtype=float)
                values = matrix[:, idx] if matrix.size else np.array([])
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "landmark_file": str(bed_path),
                        "relative_start": int(rel),
                        "relative_end": int(rel + profile_bin_size),
                        "track": track,
                        "mean": float(values.mean()) if values.size else np.nan,
                        "ci_low": float(np.quantile([np.mean(rng.choice(values, size=values.size, replace=True)) for _ in range(200)], 0.025)) if values.size else np.nan,
                        "ci_high": float(np.quantile([np.mean(rng.choice(values, size=values.size, replace=True)) for _ in range(200)], 0.975)) if values.size else np.nan,
                        "retained_regions": retained,
                    }
                )
    profile = pd.DataFrame(rows)
    if profile.empty:
        return profile
    for bed_path, subset in profile.groupby("landmark_file", sort=False):
        fig, ax = plt.subplots(figsize=(8, 4))
        for track, color in zip(TRACKS, ["tab:green", "black", "tab:purple", "tab:orange"]):
            t = subset[subset["track"] == track].sort_values("relative_start")
            x = (t["relative_start"] + t["relative_end"]) / 2
            ax.plot(x, t["mean"], label=track, color=color)
            ax.fill_between(x, t["ci_low"], t["ci_high"], color=color, alpha=0.15)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("position relative to landmark center (bp)")
        ax.set_ylabel("robust normalized signal")
        ax.set_title(Path(str(bed_path)).name)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, out_dir / f"average_profile_{Path(str(bed_path)).stem}")
    return profile


def write_metadata(args: argparse.Namespace, bigwig_meta: dict, canonical: pd.DataFrame, out_dir: Path) -> None:
    """Write track metadata validation table and chromosome mapping report."""

    metadata = [
        TrackMetadata(
            "external",
            str(args.external_bigwig),
            args.genome_assembly,
            bigwig_meta.get("chromosome_naming", "unknown"),
            args.histone_mark,
            args.external_cell_type,
            args.external_assay_type,
            "0-based half-open genomic intervals",
            "bigWig variable-step/summary intervals sampled as bin means",
            args.external_value_type,
            args.external_value_transform,
            "External population-level signal; evaluation only.",
        ),
        TrackMetadata(
            "alphagenome",
            str(args.alphagenome),
            args.genome_assembly,
            chrom_style(canonical["chrom"].unique()),
            args.histone_mark,
            args.alphagenome_cell_type,
            "AlphaGenome CHIP_HISTONE prediction",
            "0-based half-open genomic intervals after export",
            args.alphagenome_native_resolution,
            args.alphagenome_value_type,
            "model output score; raw scale retained",
            "Pretrained AlphaGenome prediction; evaluation only.",
        ),
        TrackMetadata(
            "hyena",
            str(args.hyena),
            args.genome_assembly,
            chrom_style(canonical["chrom"].unique()),
            args.histone_mark,
            args.dimelo_cell_type,
            "HyenaDNA read-level predictions aggregated by genomic bin",
            "read positions mapped to 0-based half-open genomic bins",
            "read-level predictions aggregated to requested bins",
            "predicted 6mA probability",
            "probability/score mean across valid predicted adenine positions",
            "Extended HyenaDNA model output; no external track training.",
        ),
        TrackMetadata(
            "dimelo",
            str(args.dimelo),
            args.genome_assembly,
            chrom_style(canonical["chrom"].unique()),
            args.histone_mark,
            args.dimelo_cell_type,
            "DiMeLo-seq observed read-level 6mA aggregated by genomic bin",
            "read positions mapped to 0-based half-open genomic bins",
            "read-level observations aggregated to requested bins",
            "observed 6mA probability",
            "mean across valid adenine observations; missing observations not treated as unmodified",
            "Experimental read-level signal aggregated for population-level evaluation.",
        ),
    ]
    pd.DataFrame([asdict(item) for item in metadata]).to_csv(out_dir / "validated_track_metadata.tsv", sep="\t", index=False)
    mapping = pd.DataFrame(
        {
            "canonical_chromosome": sorted(canonical["chrom"].unique()),
            "mapping_action": "unchanged",
            "reason": "Chromosome naming matched required evaluation bins.",
        }
    )
    mapping.to_csv(out_dir / "chromosome_name_mapping.tsv", sep="\t", index=False)


def make_summary(metrics: pd.DataFrame, out_path: Path) -> None:
    """Write a concise markdown summary of primary results."""

    primary = metrics[(metrics["scope"] == "all") & (metrics["intersection"] == "pair_specific")]
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("# Population-Level Track Benchmark Summary\n\n")
        handle.write(
            "This is an evaluation-only comparison. External bigWig and AlphaGenome tracks were not used "
            "as HyenaDNA training or pretraining targets.\n\n"
        )
        handle.write(
            "Pearson and Spearman correlations use raw binned values. "
            "MAE/RMSE use globally fitted robust [0,1] normalized values.\n\n"
        )
        handle.write("| Comparison | role | n bins | Pearson | Spearman | normalized MAE | normalized RMSE |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in primary.itertuples(index=False):
            handle.write(
                f"| {row.comparison} | {row.comparison_role} | {row.n_bins} | {row.pearson:.4f} | {row.spearman:.4f} | "
                f"{row.normalized_mae:.4f} | {row.normalized_rmse:.4f} |\n"
            )
        handle.write("\nMissing bigWig values and bins below DiMeLo coverage thresholds were excluded; missing observations were not set to zero.\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-bigwig", type=Path, required=True)
    parser.add_argument("--alphagenome", type=Path, default=Path("outputs/benchmark_200bp/alphagenome_test_200bp.tsv"))
    parser.add_argument("--alphagenome-column", default="A549_H3K4me3_fixed_mean")
    parser.add_argument("--hyena", type=Path, default=Path("outputs/benchmark_200bp/hyenadna_test_200bp.tsv"))
    parser.add_argument("--dimelo", type=Path, default=Path("outputs/benchmark_200bp/dimelo_test_200bp.tsv"))
    parser.add_argument("--regions", type=Path, default=Path("outputs/4chrom_test_regions.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/population_track_benchmark"))
    parser.add_argument("--chromosomes", default="chr16,chr11,chr17,chr19")
    parser.add_argument("--sample", default="pooled", choices=["pooled", "merged_c1", "merged_e5b"])
    parser.add_argument("--bin-size", type=int, default=1000)
    parser.add_argument("--min-dimelo-coverage", type=int, default=5)
    parser.add_argument("--coverage-thresholds", default="1,5,10,20")
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--bootstrap-block-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--genome-assembly", default="GRCh38")
    parser.add_argument("--allow-assembly-mismatch", action="store_true")
    parser.add_argument("--histone-mark", default="H3K4me3")
    parser.add_argument("--external-cell-type", default="unknown")
    parser.add_argument("--external-assay-type", default="experimental bigWig signal")
    parser.add_argument("--external-value-type", default="mean signal of unique reads")
    parser.add_argument("--external-value-transform", default="unknown/raw bigWig scale")
    parser.add_argument("--alphagenome-cell-type", default="A549")
    parser.add_argument("--alphagenome-native-resolution", default="128 bp native output rebinned to canonical bins")
    parser.add_argument("--alphagenome-value-type", default="AlphaGenome CHIP_HISTONE prediction score")
    parser.add_argument("--dimelo-cell-type", default="A549 C1/E5B clones")
    parser.add_argument("--blacklist-bed", type=Path)
    parser.add_argument("--landmark-bed", type=Path, action="append", default=[])
    parser.add_argument("--profile-flank", type=int, default=3000)
    parser.add_argument("--profile-bin-size", type=int, default=50)
    parser.add_argument("--max-stacked-regions", type=int, default=12)
    parser.add_argument("--skip-overlay-plots", action="store_true")
    parser.add_argument("--include-alt-contigs", action="store_true")
    parser.add_argument("--include-mito", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Audit and validate inputs, but do not write heavy benchmark tables/figures.")
    parser.add_argument("--dry-run-cleanup", action="store_true", help="Write cleanup manifest only.")
    parser.add_argument("--apply-cleanup", action="store_true", help="Delete only files listed as cleanup candidates.")
    return parser


def main() -> None:
    """Run the population-level benchmark."""

    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "processing_log.txt"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Started: {now_iso()}\n")
        log.write(f"Command: {' '.join(sys.argv)}\n")
        log.write(f"Python: {sys.version}\n")
        log.write(f"Platform: {platform.platform()}\n")

    protected = [args.external_bigwig, args.alphagenome, args.hyena, args.dimelo, args.regions]
    inventory_roots = [Path.cwd(), args.out_dir, args.external_bigwig.parent]
    write_inventory(inventory_roots, args.out_dir, protected, args.apply_cleanup)

    chromosomes = parse_chromosomes(args.chromosomes)
    regions = load_regions(args.regions, chromosomes, args.include_alt_contigs, args.include_mito)
    bins = canonical_bins(regions, args.bin_size)
    if args.blacklist_bed:
        bins, blacklist_report = exclude_blacklist_bins(bins, args.blacklist_bed)
        write_table(blacklist_report, args.out_dir / "blacklist_excluded_bins.tsv.gz")
        if bins.empty:
            raise SystemExit("All canonical bins were excluded by the blacklist.")

    config = vars(args).copy()
    config["external_bigwig"] = str(args.external_bigwig)
    config["alphagenome"] = str(args.alphagenome)
    config["hyena"] = str(args.hyena)
    config["dimelo"] = str(args.dimelo)
    config["regions"] = str(args.regions)
    config["created_at"] = now_iso()
    config["software_versions"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    with open(args.out_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)
        handle.write("\n")

    if args.dry_run_cleanup:
        print(f"Wrote inventory and cleanup manifest to {args.out_dir}")
        return

    if "A549" not in args.alphagenome_column and "A549" not in args.alphagenome_cell_type:
        raise SystemExit(
            "This benchmark expects an A549 AlphaGenome H3K4me3 output. "
            "Set --alphagenome-column/--alphagenome-cell-type explicitly if this is intentional."
        )
    external, bigwig_meta = sample_bigwig(bins, args.external_bigwig, args.genome_assembly, args.allow_assembly_mismatch)
    alpha_source = read_table(args.alphagenome)
    dimelo_source = read_table(args.dimelo)
    hyena_source = read_table(args.hyena)
    alpha = aggregate_alpha(alpha_source, bins, args.alphagenome_column)
    hyena = aggregate_read_track(hyena_source, bins, args.sample, "mean_signal", "hyena")
    dimelo = aggregate_read_track(dimelo_source, bins, args.sample, "mean_signal", "dimelo")

    merge_keys = ["chrom", "region_id", "region_name", "start", "end"]
    canonical = bins.merge(external, on=merge_keys, how="left")
    canonical = canonical.merge(alpha, on=merge_keys, how="left")
    canonical = canonical.merge(hyena, on=merge_keys, how="left")
    canonical = canonical.merge(dimelo, on=merge_keys, how="left")
    canonical["dimelo_coverage"] = canonical["dimelo_coverage"].fillna(0)
    canonical["hyena_prediction_count"] = canonical["hyena_prediction_count"].fillna(0)
    canonical["all_four_valid"] = (
        canonical["external_valid"].fillna(False)
        & canonical["alphagenome_valid"].fillna(False)
        & canonical["hyena_valid"].fillna(False)
        & canonical["dimelo_valid"].fillna(False)
        & (canonical["dimelo_coverage"] >= args.min_dimelo_coverage)
    )

    normalized, norm_params = add_normalized_columns(canonical)
    write_metadata(args, bigwig_meta, normalized, args.out_dir)
    write_table(canonical, args.out_dir / "canonical_raw_bins.tsv.gz")
    write_table(normalized, args.out_dir / "canonical_normalized_bins.tsv.gz")
    pd.DataFrame([asdict(item) for item in norm_params]).to_csv(args.out_dir / "normalization_parameters.tsv", sep="\t", index=False)
    if args.dry_run:
        print(f"Dry run complete. Wrote audit, metadata, config, and canonical tables to {args.out_dir}")
        return

    pair_metrics, chrom_metrics = pairwise_metrics(
        normalized, False, args.min_dimelo_coverage, args.bootstrap_block_size, args.bootstrap_replicates, args.seed
    )
    common_metrics, common_chrom_metrics = pairwise_metrics(
        normalized, True, args.min_dimelo_coverage, args.bootstrap_block_size, args.bootstrap_replicates, args.seed
    )
    pair_metrics = pd.concat([pair_metrics, common_metrics], ignore_index=True)
    chrom_metrics = pd.concat([chrom_metrics, common_chrom_metrics], ignore_index=True)
    coverage_thresholds = [int(item) for item in args.coverage_thresholds.split(",") if item.strip()]
    coverage = coverage_sensitivity(
        normalized, coverage_thresholds, args.bootstrap_block_size, max(0, args.bootstrap_replicates // 5), args.seed
    )
    weighted = weighted_correlation_table(normalized, args.min_dimelo_coverage)

    write_table(pair_metrics, args.out_dir / "pairwise_metrics.tsv.gz")
    write_table(chrom_metrics, args.out_dir / "chromosome_specific_metrics.tsv.gz")
    write_table(coverage, args.out_dir / "coverage_sensitivity.tsv.gz")
    write_table(weighted, args.out_dir / "weighted_correlations.tsv.gz")

    primary = pair_metrics[(pair_metrics["scope"] == "all") & (pair_metrics["intersection"] == "pair_specific")]
    plot_heatmap(correlation_matrices(primary, "pearson"), "Pearson correlation", args.out_dir / "figures" / "pearson_heatmap")
    plot_heatmap(correlation_matrices(primary, "spearman"), "Spearman correlation", args.out_dir / "figures" / "spearman_heatmap")
    make_pairwise_density_plots(normalized, args.out_dir / "figures" / "density", args.min_dimelo_coverage)
    make_chromosome_plot(chrom_metrics[chrom_metrics["intersection"] == "pair_specific"], args.out_dir / "figures" / "correlation_by_chromosome")
    make_coverage_plot(coverage, args.out_dir / "figures" / "coverage_sensitivity")
    make_stacked_region_plots(
        normalized,
        bins,
        dimelo_source,
        hyena_source,
        params_by_track(norm_params),
        args.out_dir / "figures" / "stacked_selected_regions",
        args.max_stacked_regions,
    )
    if not args.skip_overlay_plots:
        make_overlay_region_plots(
            normalized,
            args.out_dir / "figures" / "overlay_selected_regions",
            args.max_stacked_regions,
        )

    loci = select_representative_loci(normalized, args.min_dimelo_coverage)
    write_table(loci, args.out_dir / "representative_loci.tsv.gz")
    for _, locus in loci.iterrows():
        plot_locus(normalized, locus, args.out_dir / "figures" / "representative_loci")

    profiles = average_profiles(
        normalized, args.landmark_bed, args.profile_flank, args.profile_bin_size, args.seed, args.out_dir / "figures" / "average_profiles"
    )
    if not profiles.empty:
        write_table(profiles, args.out_dir / "average_landmark_profiles.tsv.gz")

    make_summary(pair_metrics, args.out_dir / "text_summary.md")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"Finished: {now_iso()}\n")
        log.write(f"Output directory: {args.out_dir}\n")
    print(f"Wrote population-level benchmark outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
