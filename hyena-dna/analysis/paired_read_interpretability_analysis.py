#!/usr/bin/env python3
"""Paired-read interpretability analysis for read-level 6mA variation.

The script compares read pairs aligned to the same locus/window and asks how
DNA differences, observed 5mC differences and observed 6mA differences relate.
Prediction columns are carried through explicitly as missing values unless
prediction arrays are supplied in the input tensors in a later workflow.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import torch
from torch import nn


ID_TO_BASE = {7: "A", 8: "C", 9: "G", 10: "T", 11: "N", 4: "-", 6: "N"}
BASES = set("ACGT")
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


class Hyena6mAMethylConditionedNoSample(nn.Module):
    """Minimal P(Reg|D,M) wrapper used only for interpretability predictions."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        methyl_feature_dim: int = 2,
        decoder_hidden_dim: int = 0,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        head_dim = hidden_dim + methyl_feature_dim
        if decoder_hidden_dim > 0:
            self.head_6ma = nn.Sequential(
                nn.Linear(head_dim, decoder_hidden_dim),
                nn.GELU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(decoder_hidden_dim, 1),
            )
        else:
            self.head_6ma = nn.Linear(head_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        methyl_value: torch.Tensor,
        methyl_observed: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(input_ids)
        methyl = torch.stack([methyl_value, methyl_observed], dim=-1)
        conditioned = torch.cat([hidden, methyl], dim=-1)
        return self.head_6ma(conditioned).squeeze(-1)


@dataclass(frozen=True)
class DatasetSpec:
    npz: Path
    metadata: Path


@dataclass
class Config:
    datasets: list[DatasetSpec]
    bam: Path
    sample: str
    chrom: str
    locus_variance: Path
    out_prefix: Path
    reference_fasta: Path | None = None
    top_loci: int = 300
    max_candidate_loci: int = 100
    max_reads_per_locus: int = 30
    window_size: int = 1000
    bin_size: int = 50
    min_reads_locus: int = 8
    min_window_coverage_fraction: float = 0.60
    min_pair_comparable_fraction: float = 0.60
    min_base_quality: int = 0
    min_valid_6ma_positions: int = 10
    min_valid_5mc_positions: int = 1
    max_missing_signal_fraction: float = 0.95
    max_mismatch_rate: float | None = None
    min_valid_per_bin: int = 1
    min_shared_bins: int = 2
    low_quantile: float = 0.10
    high_quantile: float = 0.90
    top_n_per_case: int = 3
    seed: int = 7
    reg_checkpoint: Path | None = None
    hyena_root: Path = Path("/data/leuven/383/vsc38330/hyena-dna-main")
    checkpoint_dir: Path = Path("/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    model_name: str = "hyenadna-small-32k-seqlen"
    max_length: int = 32768
    device: str = "auto"
    max_pairs_per_locus: int = 200
    flag_substitution_distance: float = 0.10


@dataclass
class ReadProfile:
    locus_id: str
    read_id: str
    sample: str
    chrom: str
    window_start: int
    window_end: int
    reference_sequence: str
    read_sequence_aligned: np.ndarray
    base_quality: np.ndarray
    observed_6ma: np.ndarray
    observed_6ma_mask: np.ndarray
    predicted_6ma: np.ndarray
    predicted_6ma_mask: np.ndarray
    observed_5mc: np.ndarray
    observed_5mc_mask: np.ndarray
    predicted_5mc: np.ndarray
    predicted_5mc_mask: np.ndarray
    mapq: int
    cigar: str
    is_reverse: bool
    coverage_fraction: float
    mismatch_to_ref_fraction: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs=2, action="append", metavar=("NPZ", "METADATA"), required=False)
    parser.add_argument("--bam")
    parser.add_argument("--sample", default="merged_c1")
    parser.add_argument("--chrom", default="chr16")
    parser.add_argument("--locus-variance", default="outputs/full_chr16_c1_locus_variance.per_locus_variance.tsv.gz")
    parser.add_argument("--reference-fasta", default=None)
    parser.add_argument("--top-loci", type=int, default=300)
    parser.add_argument("--max-candidate-loci", type=int, default=100)
    parser.add_argument("--max-reads-per-locus", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--bin-size", type=int, default=50)
    parser.add_argument("--min-reads-locus", type=int, default=8)
    parser.add_argument("--min-window-coverage-fraction", type=float, default=0.60)
    parser.add_argument("--min-pair-comparable-fraction", type=float, default=0.60)
    parser.add_argument("--min-base-quality", type=int, default=0)
    parser.add_argument("--min-valid-6ma-positions", type=int, default=10)
    parser.add_argument("--min-valid-5mc-positions", type=int, default=1)
    parser.add_argument("--max-missing-signal-fraction", type=float, default=0.95)
    parser.add_argument("--max-mismatch-rate", type=float, default=None)
    parser.add_argument("--min-valid-per-bin", type=int, default=1)
    parser.add_argument("--min-shared-bins", type=int, default=2)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--top-n-per-case", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reg-checkpoint", default=None)
    parser.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    parser.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs-per-locus", type=int, default=200)
    parser.add_argument("--flag-substitution-distance", type=float, default=0.10)
    parser.add_argument("--out-prefix", required=False)
    parser.add_argument("--synthetic-test", action="store_true")
    return parser.parse_args()


def open_text(path: Path):
    """Open plain or gzipped text."""
    return gzip.open(path, "rt", encoding="utf-8", newline="") if str(path).endswith(".gz") else path.open("rt", newline="")


def read_metadata(path: Path) -> list[dict[str, str]]:
    """Read tensor metadata TSV."""
    with path.open("rt", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty metadata file: {path}")
    return rows


def select_high_variance_loci(config: Config) -> list[tuple[str, int]]:
    """Stream the variance table and retain only the top variable loci."""
    heap: list[tuple[float, int, str, int]] = []
    counter = 0
    with open_text(config.locus_variance) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("group") != config.sample:
                continue
            if row.get("mark") != "6mA" or row.get("chrom") != config.chrom:
                continue
            n_reads = int(float(row.get("count", row.get("n_reads", 0))))
            if n_reads < config.min_reads_locus:
                continue
            item = (float(row["variance"]), counter, row["chrom"], int(row["position_0based"]))
            counter += 1
            if len(heap) < config.top_loci:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
    heap.sort(reverse=True)
    return [(chrom, pos) for _, _, chrom, pos in heap[: config.max_candidate_loci]]


def load_tensor_data(config: Config) -> tuple[list[np.lib.npyio.NpzFile], list[list[dict[str, str]]], dict[str, list[tuple[int, int]]]]:
    """Load NPZ tensors/metadata and index rows by read_id."""
    archives: list[np.lib.npyio.NpzFile] = []
    metadata_list: list[list[dict[str, str]]] = []
    rows_by_read: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ds_idx, spec in enumerate(config.datasets):
        archive = np.load(spec.npz)
        metadata = read_metadata(spec.metadata)
        if len(metadata) != archive["input_ids"].shape[0]:
            raise ValueError(f"Metadata rows do not match tensor rows for {spec.npz}")
        archives.append(archive)
        metadata_list.append(metadata)
        for row_idx, row in enumerate(metadata):
            if row.get("sample") != config.sample or row.get("chrom") != config.chrom:
                continue
            rows_by_read[row["read_id"]].append((ds_idx, row_idx))
    return archives, metadata_list, rows_by_read


def forward_read_pos(read: pysam.AlignedSegment, query_pos: int) -> int:
    """Convert query-position orientation into modkit/tensor forward-read position."""
    read_len = int(read.query_length or 0)
    return read_len - 1 - int(query_pos) if read.is_reverse else int(query_pos)


def reference_oriented_query_base(read: pysam.AlignedSegment, query_pos: int, read_sequence: str) -> str:
    """Return the read base projected into reference orientation.

    For reverse-strand alignments, the query base is complemented before DNA
    distance and reference-mismatch audits. Modification labels still use
    forward-read positions separately via :func:`forward_read_pos`.
    """
    if query_pos < 0 or query_pos >= len(read_sequence):
        return "N"
    base = read_sequence[query_pos].upper()
    if read.is_reverse:
        base = base.translate(COMPLEMENT).upper()
    return base if base in BASES else "N"


def build_ref_to_forward_map(read: pysam.AlignedSegment) -> dict[int, int | None]:
    """Map reference positions to forward-read positions; deletions map to None."""
    mapping: dict[int, int | None] = {}
    for query_pos, ref_pos in read.get_aligned_pairs(matches_only=False):
        if ref_pos is None:
            continue
        mapping[int(ref_pos)] = None if query_pos is None else forward_read_pos(read, int(query_pos))
    return mapping


def build_ref_to_query_map(read: pysam.AlignedSegment) -> dict[int, int | None]:
    """Map reference positions to BAM query positions for reference-oriented DNA audit."""
    mapping: dict[int, int | None] = {}
    for query_pos, ref_pos in read.get_aligned_pairs(matches_only=False):
        if ref_pos is None:
            continue
        mapping[int(ref_pos)] = None if query_pos is None else int(query_pos)
    return mapping


def load_reg_model(config: Config, archive: np.lib.npyio.NpzFile):
    """Load P(Reg|D,M) model for optional prediction profiles."""
    if config.reg_checkpoint is None:
        return None, None
    import sys

    sys.path.insert(0, str(config.hyena_root))
    from huggingface import HyenaDNAPreTrainedModel

    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    if device == "auto":
        device = "cpu"
    backbone = HyenaDNAPreTrainedModel.from_pretrained(
        str(config.checkpoint_dir),
        config.model_name,
        download=False,
        device=device,
        use_head=False,
    ).to(device)
    backbone.eval()
    probe_len = min(config.max_length, archive["input_ids"].shape[1])
    probe = torch.as_tensor(archive["input_ids"][0:1, :probe_len], dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden_dim = int(backbone(probe).shape[-1])
    checkpoint = torch.load(config.reg_checkpoint, map_location=device, weights_only=False)
    model = Hyena6mAMethylConditionedNoSample(
        backbone,
        hidden_dim,
        methyl_feature_dim=int(checkpoint.get("methyl_feature_dim", 2)),
        decoder_hidden_dim=int(checkpoint.get("decoder_hidden_dim", 0)),
        decoder_dropout=float(checkpoint.get("decoder_dropout", 0.0)),
    ).to(device)
    if checkpoint.get("trainable_backbone_state_dict"):
        model.backbone.load_state_dict(checkpoint["trainable_backbone_state_dict"], strict=False)
    model.head_6ma.load_state_dict(checkpoint["head_6mA_state_dict"])
    model.eval()
    return model, device


def predict_row_6ma(
    model,
    device: str | None,
    archive: np.lib.npyio.NpzFile,
    row_idx: int,
    max_length: int,
    cache: dict[tuple[int, int], np.ndarray],
    dataset_idx: int,
) -> np.ndarray | None:
    """Return cached sigmoid P(Reg|D,M) predictions for one tensor row."""
    if model is None or device is None:
        return None
    key = (dataset_idx, row_idx)
    if key in cache:
        return cache[key]
    seq_len = min(max_length, archive["input_ids"].shape[1])
    input_ids = torch.as_tensor(archive["input_ids"][row_idx : row_idx + 1, :seq_len], dtype=torch.long, device=device)
    target_5mc = archive["target_5mC"][row_idx : row_idx + 1, :seq_len]
    mask_5mc = archive["mask_5mC"][row_idx : row_idx + 1, :seq_len].astype(bool)
    methyl_value = torch.as_tensor(np.where(mask_5mc, target_5mc, 0.0), dtype=torch.float32, device=device)
    methyl_observed = torch.as_tensor(mask_5mc.astype(np.float32), dtype=torch.float32, device=device)
    with torch.inference_mode():
        pred = torch.sigmoid(model(input_ids, methyl_value, methyl_observed)).detach().cpu().numpy()[0]
    cache[key] = pred
    return pred


def collapse_tensor_value(
    archives: list[np.lib.npyio.NpzFile],
    metadata_list: list[list[dict[str, str]]],
    row_refs: list[tuple[int, int]],
    read_pos: int,
    target_key: str,
    mask_key: str,
) -> tuple[float, bool]:
    """Average duplicate overlapping-window observations for one read position."""
    values: list[float] = []
    for ds_idx, row_idx in row_refs:
        meta = metadata_list[ds_idx][row_idx]
        local = read_pos - int(meta["window_start"])
        archive = archives[ds_idx]
        if local < 0 or local >= int(meta["window_length"]) or local >= archive[target_key].shape[1]:
            continue
        if bool(archive[mask_key][row_idx, local]):
            values.append(float(archive[target_key][row_idx, local]))
    if not values:
        return float("nan"), False
    return float(np.mean(values)), True


def collapse_prediction_value(
    archives: list[np.lib.npyio.NpzFile],
    metadata_list: list[list[dict[str, str]]],
    row_refs: list[tuple[int, int]],
    read_pos: int,
    pred_key: str,
    fallback_mask_key: str,
    pred_cache: dict[tuple[int, int], np.ndarray] | None = None,
    reg_model=None,
    reg_device: str | None = None,
    max_length: int = 32768,
) -> tuple[float, bool]:
    """Average prediction arrays if present; otherwise return missing explicitly."""
    values: list[float] = []
    for ds_idx, row_idx in row_refs:
        archive = archives[ds_idx]
        meta = metadata_list[ds_idx][row_idx]
        local = read_pos - int(meta["window_start"])
        if local < 0 or local >= int(meta["window_length"]):
            continue
        if pred_key in archive.files:
            if local >= archive[pred_key].shape[1]:
                continue
            if bool(archive[fallback_mask_key][row_idx, local]):
                values.append(float(archive[pred_key][row_idx, local]))
            continue
        if pred_cache is None or pred_key != "predicted_6mA":
            continue
        pred = predict_row_6ma(reg_model, reg_device, archive, row_idx, max_length, pred_cache, ds_idx)
        if pred is None or local >= pred.shape[0]:
            continue
        if bool(archive[fallback_mask_key][row_idx, local]):
            values.append(float(pred[local]))
    if not values:
        return float("nan"), False
    return float(np.mean(values)), True


def reference_sequence_for_window(fasta: pysam.FastaFile | None, chrom: str, start: int, end: int) -> str:
    """Fetch reference sequence or use N if no FASTA is supplied."""
    if fasta is None:
        return "N" * (end - start)
    return fasta.fetch(chrom, start, end).upper()


def read_profile_for_window(
    read: pysam.AlignedSegment,
    locus_id: str,
    chrom: str,
    window_start: int,
    window_end: int,
    reference_sequence: str,
    row_refs: list[tuple[int, int]],
    archives: list[np.lib.npyio.NpzFile],
    metadata_list: list[list[dict[str, str]]],
    config: Config,
    pred_cache: dict[tuple[int, int], np.ndarray],
    reg_model=None,
    reg_device: str | None = None,
) -> ReadProfile | None:
    """Convert one aligned read into reference-coordinate signal arrays."""
    n = window_end - window_start
    seq = np.full(n, ".", dtype="<U1")
    qual = np.full(n, -1, dtype=np.int16)
    obs_6ma = np.full(n, np.nan, dtype=np.float32)
    obs_6ma_mask = np.zeros(n, dtype=bool)
    pred_6ma = np.full(n, np.nan, dtype=np.float32)
    pred_6ma_mask = np.zeros(n, dtype=bool)
    obs_5mc = np.full(n, np.nan, dtype=np.float32)
    obs_5mc_mask = np.zeros(n, dtype=bool)
    pred_5mc = np.full(n, np.nan, dtype=np.float32)
    pred_5mc_mask = np.zeros(n, dtype=bool)

    ref_to_forward = build_ref_to_forward_map(read)
    ref_to_query = build_ref_to_query_map(read)
    read_sequence = read.query_sequence or ""
    qualities = read.query_qualities

    covered = 0
    mismatches = 0
    for ref_pos in range(window_start, window_end):
        idx = ref_pos - window_start
        query_pos = ref_to_query.get(ref_pos)
        forward_pos = ref_to_forward.get(ref_pos)
        if query_pos is None:
            if ref_pos in ref_to_query:
                seq[idx] = "-"
            continue
        if query_pos < 0 or query_pos >= len(read_sequence):
            continue
        base = reference_oriented_query_base(read, query_pos, read_sequence)
        base_q = int(qualities[query_pos]) if qualities is not None else 60
        if base_q < config.min_base_quality:
            seq[idx] = "."
            qual[idx] = base_q
            continue
        seq[idx] = base if base in BASES else "N"
        qual[idx] = base_q
        covered += 1
        ref_base = reference_sequence[idx] if idx < len(reference_sequence) else "N"
        if ref_base in BASES and seq[idx] in BASES and ref_base != seq[idx]:
            mismatches += 1
        if forward_pos is None:
            continue
        obs_6ma[idx], obs_6ma_mask[idx] = collapse_tensor_value(
            archives, metadata_list, row_refs, forward_pos, "target_6mA", "mask_6mA"
        )
        obs_5mc[idx], obs_5mc_mask[idx] = collapse_tensor_value(
            archives, metadata_list, row_refs, forward_pos, "target_5mC", "mask_5mC"
        )
        pred_6ma[idx], pred_6ma_mask[idx] = collapse_prediction_value(
            archives,
            metadata_list,
            row_refs,
            forward_pos,
            "predicted_6mA",
            "mask_6mA",
            pred_cache,
            reg_model,
            reg_device,
            config.max_length,
        )
        pred_5mc[idx], pred_5mc_mask[idx] = collapse_prediction_value(
            archives, metadata_list, row_refs, forward_pos, "predicted_5mC", "mask_5mC"
        )

    coverage_fraction = covered / float(n)
    missing_signal = 1.0 - (int(obs_6ma_mask.sum()) / float(max(1, n)))
    mismatch_fraction = mismatches / float(max(1, covered))
    if coverage_fraction < config.min_window_coverage_fraction:
        return None
    if int(obs_6ma_mask.sum()) < config.min_valid_6ma_positions:
        return None
    if int(obs_5mc_mask.sum()) < config.min_valid_5mc_positions:
        return None
    if missing_signal > config.max_missing_signal_fraction:
        return None
    if config.max_mismatch_rate is not None and mismatch_fraction > config.max_mismatch_rate:
        return None

    return ReadProfile(
        locus_id=locus_id,
        read_id=read.query_name,
        sample=config.sample,
        chrom=chrom,
        window_start=window_start,
        window_end=window_end,
        reference_sequence=reference_sequence,
        read_sequence_aligned=seq,
        base_quality=qual,
        observed_6ma=obs_6ma,
        observed_6ma_mask=obs_6ma_mask,
        predicted_6ma=pred_6ma,
        predicted_6ma_mask=pred_6ma_mask,
        observed_5mc=obs_5mc,
        observed_5mc_mask=obs_5mc_mask,
        predicted_5mc=pred_5mc,
        predicted_5mc_mask=pred_5mc_mask,
        mapq=int(read.mapping_quality),
        cigar=read.cigarstring or "",
        is_reverse=bool(read.is_reverse),
        coverage_fraction=coverage_fraction,
        mismatch_to_ref_fraction=mismatch_fraction,
    )


def collect_profiles(config: Config) -> dict[str, list[ReadProfile]]:
    """Collect eligible read profiles around selected high-variance loci."""
    loci = select_high_variance_loci(config)
    archives, metadata_list, rows_by_read = load_tensor_data(config)
    reg_model, reg_device = load_reg_model(config, archives[0])
    pred_cache: dict[tuple[int, int], np.ndarray] = {}
    bam = pysam.AlignmentFile(config.bam, "rb")
    fasta = pysam.FastaFile(str(config.reference_fasta)) if config.reference_fasta else None
    profiles_by_locus: dict[str, list[ReadProfile]] = defaultdict(list)
    half = config.window_size // 2
    for chrom, center in loci:
        start = max(0, center - half)
        end = start + config.window_size
        locus_id = f"{chrom}:{start}-{end}"
        ref_seq = reference_sequence_for_window(fasta, chrom, start, end)
        seen = set()
        for read in bam.fetch(chrom, start, end):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.query_name in seen:
                continue
            if read.query_name not in rows_by_read:
                continue
            seen.add(read.query_name)
            profile = read_profile_for_window(
                read,
                locus_id,
                chrom,
                start,
                end,
                ref_seq,
                rows_by_read[read.query_name],
                archives,
                metadata_list,
                config,
                pred_cache,
                reg_model,
                reg_device,
            )
            if profile is not None:
                profiles_by_locus[locus_id].append(profile)
            if len(profiles_by_locus[locus_id]) >= config.max_reads_per_locus:
                break
    bam.close()
    if fasta is not None:
        fasta.close()
    return {k: v for k, v in profiles_by_locus.items() if len(v) >= 2}


def binned_signal(values: np.ndarray, mask: np.ndarray, bin_size: int, min_valid: int) -> tuple[np.ndarray, np.ndarray]:
    """Masked binned mean; bins with insufficient observations stay missing."""
    n_bins = int(math.ceil(values.size / bin_size))
    out = np.full(n_bins, np.nan, dtype=np.float32)
    out_mask = np.zeros(n_bins, dtype=bool)
    for b in range(n_bins):
        lo = b * bin_size
        hi = min(values.size, (b + 1) * bin_size)
        m = mask[lo:hi]
        if int(m.sum()) >= min_valid:
            out[b] = float(np.nanmean(values[lo:hi][m]))
            out_mask[b] = True
    return out, out_mask


def profile_mae(a: np.ndarray, ma: np.ndarray, b: np.ndarray, mb: np.ndarray, min_shared: int) -> tuple[float, int]:
    """Masked mean absolute profile distance."""
    shared = ma & mb & np.isfinite(a) & np.isfinite(b)
    n = int(shared.sum())
    if n < min_shared:
        return float("nan"), n
    return float(np.mean(np.abs(a[shared] - b[shared]))), n


def correlation_distance(a: np.ndarray, ma: np.ndarray, b: np.ndarray, mb: np.ndarray, min_shared: int) -> tuple[float, int]:
    """Return 1 - Pearson correlation on shared valid bins."""
    shared = ma & mb & np.isfinite(a) & np.isfinite(b)
    n = int(shared.sum())
    if n < max(3, min_shared):
        return float("nan"), n
    x = a[shared].astype(float)
    y = b[shared].astype(float)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan"), n
    return float(1.0 - np.corrcoef(x, y)[0, 1]), n


def dna_distances(a: ReadProfile, b: ReadProfile, config: Config) -> dict[str, float | int]:
    """Calculate normalized sequence distances in reference coordinates."""
    qa = a.base_quality >= config.min_base_quality
    qb = b.base_quality >= config.min_base_quality
    sa = a.read_sequence_aligned
    sb = b.read_sequence_aligned
    covered_a = np.isin(sa, list(BASES | {"-"}))
    covered_b = np.isin(sb, list(BASES | {"-"}))
    comparable = qa & qb & covered_a & covered_b
    n = int(comparable.sum())
    if n == 0:
        return {
            "dna_distance": float("nan"),
            "substitution_distance": float("nan"),
            "indel_distance": float("nan"),
            "comparable_bases": 0,
            "comparable_fraction": 0.0,
            "sequence_mismatches": 0,
            "sequence_indels": 0,
        }
    both_base = comparable & np.isin(sa, list(BASES)) & np.isin(sb, list(BASES))
    substitutions = both_base & (sa != sb)
    indels = comparable & ((sa == "-") ^ (sb == "-"))
    any_diff = comparable & (sa != sb)
    return {
        "dna_distance": float(any_diff.sum() / n),
        "substitution_distance": float(substitutions.sum() / n),
        "indel_distance": float(indels.sum() / n),
        "comparable_bases": n,
        "comparable_fraction": float(n / sa.size),
        "sequence_mismatches": int(substitutions.sum()),
        "sequence_indels": int(indels.sum()),
    }


def pair_metrics(a: ReadProfile, b: ReadProfile, config: Config) -> dict[str, object] | None:
    """Calculate all pairwise distances and quality statistics."""
    d = dna_distances(a, b, config)
    if not math.isfinite(float(d["dna_distance"])):
        return None
    if float(d["comparable_fraction"]) < config.min_pair_comparable_fraction:
        return None
    obs6_a, obs6_ma = binned_signal(a.observed_6ma, a.observed_6ma_mask, config.bin_size, config.min_valid_per_bin)
    obs6_b, obs6_mb = binned_signal(b.observed_6ma, b.observed_6ma_mask, config.bin_size, config.min_valid_per_bin)
    pred6_a, pred6_ma = binned_signal(a.predicted_6ma, a.predicted_6ma_mask, config.bin_size, config.min_valid_per_bin)
    pred6_b, pred6_mb = binned_signal(b.predicted_6ma, b.predicted_6ma_mask, config.bin_size, config.min_valid_per_bin)
    obs5_a, obs5_ma = binned_signal(a.observed_5mc, a.observed_5mc_mask, config.bin_size, config.min_valid_per_bin)
    obs5_b, obs5_mb = binned_signal(b.observed_5mc, b.observed_5mc_mask, config.bin_size, config.min_valid_per_bin)
    pred5_a, pred5_ma = binned_signal(a.predicted_5mc, a.predicted_5mc_mask, config.bin_size, config.min_valid_per_bin)
    pred5_b, pred5_mb = binned_signal(b.predicted_5mc, b.predicted_5mc_mask, config.bin_size, config.min_valid_per_bin)

    obs6_mae, obs6_bins = profile_mae(obs6_a, obs6_ma, obs6_b, obs6_mb, config.min_shared_bins)
    if not math.isfinite(obs6_mae):
        return None
    obs6_corr, obs6_corr_bins = correlation_distance(obs6_a, obs6_ma, obs6_b, obs6_mb, config.min_shared_bins)
    pred6_mae, pred6_bins = profile_mae(pred6_a, pred6_ma, pred6_b, pred6_mb, config.min_shared_bins)
    pred6_corr, pred6_corr_bins = correlation_distance(pred6_a, pred6_ma, pred6_b, pred6_mb, config.min_shared_bins)
    obs5_mae, obs5_bins = profile_mae(obs5_a, obs5_ma, obs5_b, obs5_mb, config.min_shared_bins)
    obs5_corr, obs5_corr_bins = correlation_distance(obs5_a, obs5_ma, obs5_b, obs5_mb, config.min_shared_bins)
    pred5_mae, pred5_bins = profile_mae(pred5_a, pred5_ma, pred5_b, pred5_mb, config.min_shared_bins)
    pred5_corr, pred5_corr_bins = correlation_distance(pred5_a, pred5_ma, pred5_b, pred5_mb, config.min_shared_bins)
    pred_error = abs(pred6_mae - obs6_mae) if math.isfinite(pred6_mae) else float("nan")
    row: dict[str, object] = {
        "locus_id": a.locus_id,
        "chrom": a.chrom,
        "window_start": a.window_start,
        "window_end": a.window_end,
        "read_id_1": a.read_id,
        "read_id_2": b.read_id,
        "mapq_1": a.mapq,
        "mapq_2": b.mapq,
        "coverage_fraction_1": a.coverage_fraction,
        "coverage_fraction_2": b.coverage_fraction,
        "mismatch_to_ref_fraction_1": a.mismatch_to_ref_fraction,
        "mismatch_to_ref_fraction_2": b.mismatch_to_ref_fraction,
        "valid_6ma_positions_1": int(a.observed_6ma_mask.sum()),
        "valid_6ma_positions_2": int(b.observed_6ma_mask.sum()),
        "valid_5mc_positions_1": int(a.observed_5mc_mask.sum()),
        "valid_5mc_positions_2": int(b.observed_5mc_mask.sum()),
        "observed_6ma_mae": obs6_mae,
        "observed_6ma_correlation_distance": obs6_corr,
        "predicted_6ma_mae": pred6_mae,
        "predicted_6ma_correlation_distance": pred6_corr,
        "observed_5mc_mae": obs5_mae,
        "observed_5mc_correlation_distance": obs5_corr,
        "predicted_5mc_mae": pred5_mae,
        "predicted_5mc_correlation_distance": pred5_corr,
        "shared_valid_bins_observed_6ma": obs6_bins,
        "shared_valid_bins_predicted_6ma": pred6_bins,
        "shared_valid_bins_observed_5mc": obs5_bins,
        "shared_valid_bins_predicted_5mc": pred5_bins,
        "pair_prediction_error_6ma": pred_error,
    }
    row.update(d)
    row["flag_high_pairwise_substitution_distance"] = bool(
        math.isfinite(float(row["substitution_distance"]))
        and float(row["substitution_distance"]) > config.flag_substitution_distance
    )
    return row


def all_pair_metrics(profiles_by_locus: dict[str, list[ReadProfile]], config: Config) -> pd.DataFrame:
    """Calculate pair metrics for every eligible same-locus read pair."""
    rows: list[dict[str, object]] = []
    rng = random.Random(config.seed)
    for profiles in profiles_by_locus.values():
        locus_rows: list[dict[str, object]] = []
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                row = pair_metrics(profiles[i], profiles[j], config)
                if row is not None:
                    locus_rows.append(row)
        if len(locus_rows) > config.max_pairs_per_locus:
            rng.shuffle(locus_rows)
            locus_rows = locus_rows[: config.max_pairs_per_locus]
        rows.extend(locus_rows)
    return pd.DataFrame(rows)


def zscore(series: pd.Series) -> pd.Series:
    """Robust z-score helper with NaN preservation."""
    x = series.astype(float)
    sd = x.std(ddof=0)
    if not math.isfinite(float(sd)) or float(sd) == 0.0:
        return pd.Series(np.zeros(len(x)), index=series.index)
    return (x - x.mean()) / sd


def select_case_pairs(pair_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Select top N pairs for cases A-D using quantile thresholds."""
    if pair_df.empty:
        return pair_df
    dna_low = pair_df["dna_distance"].quantile(config.low_quantile)
    dna_high = pair_df["dna_distance"].quantile(config.high_quantile)
    reg_low = pair_df["observed_6ma_mae"].quantile(config.low_quantile)
    reg_high = pair_df["observed_6ma_mae"].quantile(config.high_quantile)
    zdna = zscore(pair_df["dna_distance"])
    zreg = zscore(pair_df["observed_6ma_mae"])
    audit_pass = ~pair_df.get("flag_high_pairwise_substitution_distance", pd.Series(False, index=pair_df.index)).astype(bool)
    specs = {
        "A_lowDNA_high6mA": ((pair_df["dna_distance"] <= dna_low) & (pair_df["observed_6ma_mae"] >= reg_high), zreg - zdna),
        "B_highDNA_low6mA": ((pair_df["dna_distance"] >= dna_high) & (pair_df["observed_6ma_mae"] <= reg_low) & audit_pass, zdna - zreg),
        "C_highDNA_high6mA": ((pair_df["dna_distance"] >= dna_high) & (pair_df["observed_6ma_mae"] >= reg_high), zdna + zreg),
        "D_lowDNA_low6mA": ((pair_df["dna_distance"] <= dna_low) & (pair_df["observed_6ma_mae"] <= reg_low), -zdna - zreg),
    }
    selected = []
    for case, (mask, score) in specs.items():
        subset = pair_df.loc[mask].copy()
        if subset.empty:
            continue
        subset["case"] = case
        subset["case_score"] = score.loc[subset.index]
        subset["dna_low_threshold"] = dna_low
        subset["dna_high_threshold"] = dna_high
        subset["observed_6ma_low_threshold"] = reg_low
        subset["observed_6ma_high_threshold"] = reg_high
        subset = subset.sort_values("case_score", ascending=False).head(config.top_n_per_case)
        selected.append(subset)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def pearson_spearman(x: Iterable[float], y: Iterable[float]) -> dict[str, float | int]:
    """Calculate Pearson/Spearman without scipy."""
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    n = int(xa.size)
    if n < 3 or float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return {"n": n, "pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(xa, ya)[0, 1])
    rx = pd.Series(xa).rank(method="average").to_numpy()
    ry = pd.Series(ya).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"n": n, "pearson": pearson, "spearman": spearman}


def save_global_plots(pair_df: pd.DataFrame, selected_df: pd.DataFrame, config: Config) -> dict[str, str]:
    """Create global scatter plots."""
    outputs: dict[str, str] = {}
    prefix = config.out_prefix
    selected_keys = set(zip(selected_df.get("locus_id", []), selected_df.get("read_id_1", []), selected_df.get("read_id_2", [])))
    is_selected = pair_df.apply(lambda r: (r["locus_id"], r["read_id_1"], r["read_id_2"]) in selected_keys, axis=1) if not pair_df.empty else []
    thresholds = {}
    if not pair_df.empty:
        thresholds = {
            "dna_low": pair_df["dna_distance"].quantile(config.low_quantile),
            "dna_high": pair_df["dna_distance"].quantile(config.high_quantile),
            "reg_low": pair_df["observed_6ma_mae"].quantile(config.low_quantile),
            "reg_high": pair_df["observed_6ma_mae"].quantile(config.high_quantile),
        }

    fig, ax = plt.subplots(figsize=(7, 5))
    loci = sorted(pair_df["locus_id"].unique())
    cmap = plt.get_cmap("tab20", max(1, len(loci)))
    for idx, locus in enumerate(loci):
        sub = pair_df[pair_df["locus_id"] == locus]
        ax.scatter(sub["dna_distance"], sub["observed_6ma_mae"], s=18, alpha=0.45, color=cmap(idx), label=locus if len(loci) <= 12 else None)
    if len(pair_df) and len(selected_df):
        ax.scatter(pair_df.loc[is_selected, "dna_distance"], pair_df.loc[is_selected, "observed_6ma_mae"], s=60, c="#d1495b", edgecolor="black")
    if thresholds:
        ax.axvline(thresholds["dna_low"], color="grey", ls="--", lw=1)
        ax.axvline(thresholds["dna_high"], color="grey", ls="--", lw=1)
        ax.axhline(thresholds["reg_low"], color="grey", ls="--", lw=1)
        ax.axhline(thresholds["reg_high"], color="grey", ls="--", lw=1)
    ax.set_xlabel("DNA distance")
    ax.set_ylabel("Observed 6mA MAE")
    ax.set_title("DNA distance versus observed 6mA distance")
    if len(loci) <= 12:
        ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    for ext in ("png", "svg"):
        path = prefix.with_suffix(f".plot1_dna_vs_observed6ma.{ext}")
        fig.savefig(path, dpi=220)
        outputs[f"plot1_{ext}"] = str(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ok = np.isfinite(pair_df["predicted_6ma_mae"].astype(float))
    if ok.any():
        ax.scatter(pair_df.loc[ok, "observed_6ma_mae"], pair_df.loc[ok, "predicted_6ma_mae"], s=18, alpha=0.4)
        lim = max(float(pair_df.loc[ok, "observed_6ma_mae"].max()), float(pair_df.loc[ok, "predicted_6ma_mae"].max()))
        ax.plot([0, lim], [0, lim], color="black", lw=1)
        corr = pearson_spearman(pair_df.loc[ok, "observed_6ma_mae"], pair_df.loc[ok, "predicted_6ma_mae"])
        mae = float(np.mean(np.abs(pair_df.loc[ok, "observed_6ma_mae"].astype(float) - pair_df.loc[ok, "predicted_6ma_mae"].astype(float))))
        ax.text(
            0.02,
            0.98,
            f"n={corr['n']}\nPearson={corr['pearson']:.3f}\nSpearman={corr['spearman']:.3f}\nMAE={mae:.3f}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )
        ax.set_xlabel("Observed 6mA MAE")
        ax.set_ylabel("Predicted 6mA MAE")
        ax.set_title("Observed versus predicted pairwise 6mA distance")
        fig.tight_layout()
        for ext in ("png", "svg"):
            path = prefix.with_suffix(f".plot2_observed_vs_predicted6ma.{ext}")
            fig.savefig(path, dpi=220)
            outputs[f"plot2_{ext}"] = str(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    color = pair_df["observed_5mc_mae"].astype(float)
    sc = ax.scatter(pair_df["dna_distance"], pair_df["observed_6ma_mae"], c=color, s=18, alpha=0.45, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="Observed 5mC MAE")
    ax.set_xlabel("DNA distance")
    ax.set_ylabel("Observed 6mA MAE")
    ax.set_title("DNA versus 6mA distance coloured by 5mC distance")
    fig.tight_layout()
    for ext in ("png", "svg"):
        path = prefix.with_suffix(f".plot3_dna_vs_6ma_colored_5mc.{ext}")
        fig.savefig(path, dpi=220)
        outputs[f"plot3_{ext}"] = str(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ok = np.isfinite(pair_df["observed_5mc_mae"].astype(float))
    if ok.any():
        sc = ax.scatter(pair_df.loc[ok, "observed_5mc_mae"], pair_df.loc[ok, "observed_6ma_mae"], c=pair_df.loc[ok, "dna_distance"], s=18, alpha=0.45, cmap="plasma")
        fig.colorbar(sc, ax=ax, label="DNA distance")
    else:
        ax.text(0.5, 0.5, "Observed 5mC distances unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Observed 5mC MAE")
    ax.set_ylabel("Observed 6mA MAE")
    ax.set_title("Observed 5mC distance versus observed 6mA distance")
    fig.tight_layout()
    for ext in ("png", "svg"):
        path = prefix.with_suffix(f".plot4_observed5mc_vs_observed6ma.{ext}")
        fig.savefig(path, dpi=220)
        outputs[f"plot4_{ext}"] = str(path)
    plt.close(fig)
    return outputs


def profile_lookup(profiles_by_locus: dict[str, list[ReadProfile]]) -> dict[tuple[str, str], ReadProfile]:
    """Map (locus_id, read_id) to profile."""
    return {(p.locus_id, p.read_id): p for profiles in profiles_by_locus.values() for p in profiles}


def plot_pair_tracks(row: pd.Series, lookup: dict[tuple[str, str], ReadProfile], config: Config, suffix: str) -> str:
    """Create a detailed multipanel read-pair track plot."""
    p1 = lookup[(row["locus_id"], row["read_id_1"])]
    p2 = lookup[(row["locus_id"], row["read_id_2"])]
    x = np.arange(p1.window_start, p1.window_end)
    diff = np.zeros(len(x), dtype=int)
    missing = (p1.read_sequence_aligned == ".") | (p2.read_sequence_aligned == ".")
    indel = (p1.read_sequence_aligned == "-") ^ (p2.read_sequence_aligned == "-")
    sub = np.isin(p1.read_sequence_aligned, list(BASES)) & np.isin(p2.read_sequence_aligned, list(BASES)) & (p1.read_sequence_aligned != p2.read_sequence_aligned)
    diff[missing] = 1
    diff[sub] = 2
    diff[indel] = 3

    tracks = [
        ("sequence difference", diff, np.ones_like(diff, dtype=bool), "categorical"),
        ("observed 6mA read 1", p1.observed_6ma, p1.observed_6ma_mask, "signal"),
        ("observed 6mA read 2", p2.observed_6ma, p2.observed_6ma_mask, "signal"),
        ("observed 5mC read 1", p1.observed_5mc, p1.observed_5mc_mask, "signal"),
        ("observed 5mC read 2", p2.observed_5mc, p2.observed_5mc_mask, "signal"),
    ]
    if p1.predicted_6ma_mask.any() or p2.predicted_6ma_mask.any():
        tracks.insert(3, ("predicted 6mA read 1", p1.predicted_6ma, p1.predicted_6ma_mask, "signal"))
        tracks.insert(4, ("predicted 6mA read 2", p2.predicted_6ma, p2.predicted_6ma_mask, "signal"))
    if p1.predicted_5mc_mask.any() or p2.predicted_5mc_mask.any():
        tracks.extend(
            [
                ("predicted 5mC read 1", p1.predicted_5mc, p1.predicted_5mc_mask, "signal"),
                ("predicted 5mC read 2", p2.predicted_5mc, p2.predicted_5mc_mask, "signal"),
            ]
        )
    fig, axes = plt.subplots(len(tracks), 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [0.55] + [1] * (len(tracks) - 1)})
    fig.suptitle(f"{row.get('case', 'selected')} | {p1.locus_id}", fontsize=12)
    for ax, (label, values, mask, kind) in zip(axes, tracks):
        if kind == "categorical":
            ax.imshow(values[np.newaxis, :], aspect="auto", extent=[p1.window_start, p1.window_end, 0, 1], cmap="tab10", interpolation="nearest", vmin=0, vmax=3)
            ax.set_yticks([])
        else:
            valid = mask & np.isfinite(values)
            if valid.any():
                ax.scatter(x[valid], values[valid], s=8)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(label, fontsize=8)
    axes[-1].set_xlabel("Reference coordinate")
    txt = (
        f"read1={p1.read_id[:8]} read2={p2.read_id[:8]}\n"
        f"DNA distance={row['dna_distance']:.3f}; comparable={int(row['comparable_bases'])}\n"
        f"obs 6mA MAE={row['observed_6ma_mae']:.3f}; pred 6mA MAE={row['predicted_6ma_mae'] if math.isfinite(float(row['predicted_6ma_mae'])) else 'NA'}\n"
        f"obs 5mC MAE={row['observed_5mc_mae'] if math.isfinite(float(row['observed_5mc_mae'])) else 'NA'}"
    )
    axes[0].text(0.995, 1.2, txt, ha="right", va="bottom", transform=axes[0].transAxes, fontsize=8, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = config.out_prefix.with_suffix(f".pair_{suffix}.png")
    fig.savefig(path, dpi=220)
    svg_path = config.out_prefix.with_suffix(f".pair_{suffix}.svg")
    fig.savefig(svg_path)
    plt.close(fig)
    return str(path)


def save_pair_track_plots(selected_df: pd.DataFrame, profiles_by_locus: dict[str, list[ReadProfile]], config: Config) -> dict[str, str]:
    """Save detailed plots for selected pairs and a compact combined figure pointer."""
    outputs: dict[str, str] = {}
    lookup = profile_lookup(profiles_by_locus)
    for idx, row in selected_df.iterrows():
        safe_case = str(row["case"]).replace("/", "_")
        suffix = f"{safe_case}_{idx + 1}"
        outputs[f"pair_{idx + 1}"] = plot_pair_tracks(row, lookup, config, suffix)
    return outputs


def save_selected_pair_dna_audit(
    selected_df: pd.DataFrame, profiles_by_locus: dict[str, list[ReadProfile]], config: Config
) -> str | None:
    """Export per-reference-position DNA comparison audit for selected pairs."""
    if selected_df.empty:
        return None
    lookup = profile_lookup(profiles_by_locus)
    rows: list[dict[str, object]] = []
    for _, row in selected_df.iterrows():
        p1 = lookup[(row["locus_id"], row["read_id_1"])]
        p2 = lookup[(row["locus_id"], row["read_id_2"])]
        for offset, ref_pos in enumerate(range(p1.window_start, p1.window_end)):
            b1 = str(p1.read_sequence_aligned[offset])
            b2 = str(p2.read_sequence_aligned[offset])
            ref = p1.reference_sequence[offset] if offset < len(p1.reference_sequence) else "N"
            comparable = (
                p1.base_quality[offset] >= config.min_base_quality
                and p2.base_quality[offset] >= config.min_base_quality
                and b1 in BASES.union({"-"})
                and b2 in BASES.union({"-"})
            )
            rows.append(
                {
                    "case": row.get("case", ""),
                    "locus_id": row["locus_id"],
                    "chrom": p1.chrom,
                    "position_0based": ref_pos,
                    "reference_base": ref,
                    "read_id_1": p1.read_id,
                    "read_id_2": p2.read_id,
                    "base_1_reference_orientation": b1,
                    "base_2_reference_orientation": b2,
                    "quality_1": int(p1.base_quality[offset]),
                    "quality_2": int(p2.base_quality[offset]),
                    "comparable": bool(comparable),
                    "pairwise_substitution": bool(comparable and b1 in BASES and b2 in BASES and b1 != b2),
                    "pairwise_indel": bool(comparable and ((b1 == "-") ^ (b2 == "-"))),
                    "read1_mismatch_to_reference": bool(comparable and b1 in BASES and ref in BASES and b1 != ref),
                    "read2_mismatch_to_reference": bool(comparable and b2 in BASES and ref in BASES and b2 != ref),
                }
            )
    path = config.out_prefix.with_suffix(".selected_pair_position_audit.tsv")
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return str(path)


def save_locus_associations(pair_df: pd.DataFrame, config: Config) -> str:
    """Save within-locus associations for DNA/5mC versus 6mA distances."""
    rows = []
    for locus, sub in pair_df.groupby("locus_id"):
        dna = pearson_spearman(sub["dna_distance"], sub["observed_6ma_mae"])
        m = pearson_spearman(sub["observed_5mc_mae"], sub["observed_6ma_mae"])
        pred = pearson_spearman(sub["observed_6ma_mae"], sub["predicted_6ma_mae"])
        rows.append(
            {
                "locus_id": locus,
                "n_pairs": int(len(sub)),
                "dna_vs_6ma_pearson": dna["pearson"],
                "dna_vs_6ma_spearman": dna["spearman"],
                "dna_vs_6ma_n": dna["n"],
                "5mc_vs_6ma_pearson": m["pearson"],
                "5mc_vs_6ma_spearman": m["spearman"],
                "5mc_vs_6ma_n": m["n"],
                "observed_vs_predicted_6ma_pearson": pred["pearson"],
                "observed_vs_predicted_6ma_spearman": pred["spearman"],
                "observed_vs_predicted_6ma_n": pred["n"],
            }
        )
    path = config.out_prefix.with_suffix(".within_locus_associations.tsv")
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return str(path)


def methylation_added_value(pair_df: pd.DataFrame, config: Config) -> dict[str, object]:
    """Grouped-by-locus CV comparing DNA-only vs DNA+5mC association models."""
    cols = ["observed_6ma_mae", "dna_distance", "observed_5mc_mae", "locus_id", "read_id_1", "read_id_2"]
    df = pair_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if df["locus_id"].nunique() < 3 or len(df) < 10:
        return {"status": "not_enough_data", "n_pairs": int(len(df)), "n_loci": int(df["locus_id"].nunique())}
    if df["locus_id"].nunique() < 5:
        results = {}
        y = df["observed_6ma_mae"].to_numpy(float)
        for name, predictors in {"dna_only": ["dna_distance"], "dna_plus_5mc": ["dna_distance", "observed_5mc_mae"]}.items():
            x = np.column_stack([np.ones(len(df))] + [df[p].to_numpy(float) for p in predictors])
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            pred = x @ beta
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            results[name] = {
                "descriptive_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "descriptive_mae": float(np.mean(np.abs(y - pred))),
                "coefficients": beta.tolist(),
                "predictors": ["intercept"] + predictors,
            }
        results["delta_descriptive_r2_adding_5mc"] = float(results["dna_plus_5mc"]["descriptive_r2"] - results["dna_only"]["descriptive_r2"])
        results["delta_descriptive_mae_adding_5mc"] = float(results["dna_plus_5mc"]["descriptive_mae"] - results["dna_only"]["descriptive_mae"])
        return {
            "status": "descriptive_not_enough_loci_for_grouped_cv",
            "n_pairs": int(len(df)),
            "n_loci": int(df["locus_id"].nunique()),
            "minimum_loci_for_grouped_cv": 5,
            "models": results,
        }
    rng = random.Random(config.seed)
    loci = list(df["locus_id"].unique())
    rng.shuffle(loci)
    folds = np.array_split(loci, min(5, len(loci)))

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame, predictors: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_train = np.column_stack([np.ones(len(train))] + [train[p].to_numpy(float) for p in predictors])
        y_train = train["observed_6ma_mae"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        x_test = np.column_stack([np.ones(len(test))] + [test[p].to_numpy(float) for p in predictors])
        return test["observed_6ma_mae"].to_numpy(float), x_test @ beta, beta

    results = {}
    for name, predictors in {"dna_only": ["dna_distance"], "dna_plus_5mc": ["dna_distance", "observed_5mc_mae"]}.items():
        y_all = []
        pred_all = []
        betas = []
        for fold in folds:
            test = df[df["locus_id"].isin(fold)]
            train = df[~df["locus_id"].isin(fold)]
            if len(test) == 0 or len(train) < len(predictors) + 2:
                continue
            train_reads = set(train["read_id_1"]).union(set(train["read_id_2"]))
            test_reads = set(test["read_id_1"]).union(set(test["read_id_2"]))
            if train_reads.intersection(test_reads):
                raise ValueError("Grouped CV leakage: at least one read occurs in both train and test folds.")
            y, pred, beta = fit_predict(train, test, predictors)
            y_all.append(y)
            pred_all.append(pred)
            betas.append(beta)
        if not y_all:
            results[name] = {"status": "not_enough_fold_data"}
            continue
        y = np.concatenate(y_all)
        pred = np.concatenate(pred_all)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        results[name] = {
            "cv_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "cv_mae": float(np.mean(np.abs(y - pred))),
            "mean_coefficients": np.mean(np.vstack(betas), axis=0).tolist(),
            "predictors": ["intercept"] + predictors,
        }
    if "cv_r2" in results.get("dna_only", {}) and "cv_r2" in results.get("dna_plus_5mc", {}):
        results["delta_cv_r2_adding_5mc"] = float(results["dna_plus_5mc"]["cv_r2"] - results["dna_only"]["cv_r2"])
        results["delta_cv_mae_adding_5mc"] = float(results["dna_plus_5mc"]["cv_mae"] - results["dna_only"]["cv_mae"])
    return {"status": "ok", "n_pairs": int(len(df)), "n_loci": int(df["locus_id"].nunique()), "models": results}


def write_report(pair_df: pd.DataFrame, selected_df: pd.DataFrame, plots: dict[str, str], added: dict[str, object], config: Config) -> str:
    """Write a short markdown report."""
    corr_dna_6ma = pearson_spearman(pair_df["dna_distance"], pair_df["observed_6ma_mae"])
    corr_5mc_6ma = pearson_spearman(pair_df["observed_5mc_mae"], pair_df["observed_6ma_mae"])
    report = config.out_prefix.with_suffix(".report.md")
    with report.open("wt", encoding="utf-8") as handle:
        handle.write("# Paired-read interpretability analysis\n\n")
        unique_reads = len(set(pair_df["read_id_1"]).union(set(pair_df["read_id_2"]))) if len(pair_df) else 0
        pred_pairs = int(np.isfinite(pair_df["predicted_6ma_mae"].astype(float)).sum()) if "predicted_6ma_mae" in pair_df else 0
        high_sub = int(pair_df["flag_high_pairwise_substitution_distance"].sum()) if "flag_high_pairwise_substitution_distance" in pair_df else 0
        handle.write(f"- Eligible read pairs: {len(pair_df)}\n")
        handle.write(f"- Unique reads represented: {unique_reads}\n")
        handle.write(f"- Loci represented: {pair_df['locus_id'].nunique() if len(pair_df) else 0}\n")
        handle.write(f"- Selected examples: {len(selected_df)}\n\n")
        handle.write(f"- P(Reg|D,M) predicted 6mA distances available for {pred_pairs}/{len(pair_df)} pairs\n")
        handle.write(f"- Pairwise substitution-distance audit flags > {config.flag_substitution_distance}: {high_sub}/{len(pair_df)} pairs\n\n")
        if len(selected_df):
            handle.write("## Selected Cases\n\n")
            for case, count in selected_df["case"].value_counts().items():
                handle.write(f"- {case}: {int(count)}\n")
            handle.write("\n")
        handle.write("## Correlations\n\n")
        handle.write(f"- DNA distance vs observed 6mA MAE: Pearson={corr_dna_6ma['pearson']}, Spearman={corr_dna_6ma['spearman']}, n={corr_dna_6ma['n']}\n")
        handle.write(f"- Observed 5mC MAE vs observed 6mA MAE: Pearson={corr_5mc_6ma['pearson']}, Spearman={corr_5mc_6ma['spearman']}, n={corr_5mc_6ma['n']}\n\n")
        handle.write("## Methylation added-value analysis\n\n")
        handle.write("This is an association analysis, not causal evidence.\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(added, indent=2))
        handle.write("\n```\n\n")
        handle.write("## Plots\n\n")
        for key, value in plots.items():
            handle.write(f"- {key}: `{value}`\n")
    return str(report)


def save_config(config: Config) -> str:
    """Save parameters used for reproducibility."""
    path = config.out_prefix.with_suffix(".params.json")
    payload = {
        k: (str(v) if isinstance(v, Path) else [str(x.npz) + "|" + str(x.metadata) for x in v] if k == "datasets" else v)
        for k, v in config.__dict__.items()
    }
    with path.open("wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(path)


def run_synthetic_test(out_prefix: Path) -> None:
    """Small synthetic check of quantile-based A-D category selection."""
    rng = np.random.default_rng(7)
    rows = []
    for case, dna_center, reg_center in [
        ("A", 0.05, 0.80),
        ("B", 0.80, 0.05),
        ("C", 0.80, 0.80),
        ("D", 0.05, 0.05),
    ]:
        for i in range(20):
            dna_noise = float(rng.normal(0.0, 0.005))
            reg_noise = float(rng.normal(0.0, 0.005))
            if i < 3:
                dna_noise = 0.0
                reg_noise = 0.0
            rows.append(
                {
                    "locus_id": f"synthetic_{case}_{i % 4}",
                    "chrom": "chrSynthetic",
                    "window_start": 0,
                    "window_end": 1000,
                    "read_id_1": f"{case}_{i}_a",
                    "read_id_2": f"{case}_{i}_b",
                    "dna_distance": float(np.clip(dna_center + dna_noise, 0, 1)),
                    "observed_6ma_mae": float(np.clip(reg_center + reg_noise, 0, 1)),
                    "observed_5mc_mae": float(rng.random()),
                    "predicted_6ma_mae": float("nan"),
                    "predicted_5mc_mae": float("nan"),
                    "observed_6ma_correlation_distance": float("nan"),
                    "predicted_6ma_correlation_distance": float("nan"),
                    "observed_5mc_correlation_distance": float("nan"),
                    "predicted_5mc_correlation_distance": float("nan"),
                    "shared_valid_bins_observed_6ma": 5,
                    "shared_valid_bins_observed_5mc": 5,
                    "comparable_bases": 1000,
                    "comparable_fraction": 1.0,
                    "sequence_mismatches": 0,
                    "sequence_indels": 0,
                    "substitution_distance": 0.0,
                    "indel_distance": 0.0,
                    "pair_prediction_error_6ma": float("nan"),
                    "mapq_1": 60,
                    "mapq_2": 60,
                    "coverage_fraction_1": 1.0,
                    "coverage_fraction_2": 1.0,
                    "valid_6ma_positions_1": 10,
                    "valid_6ma_positions_2": 10,
                    "valid_5mc_positions_1": 2,
                    "valid_5mc_positions_2": 2,
                }
            )
    config = Config(
        [],
        Path("dummy.bam"),
        "",
        "",
        Path("dummy.tsv"),
        out_prefix,
        low_quantile=0.25,
        high_quantile=0.75,
    )
    df = pd.DataFrame(rows)
    selected = select_case_pairs(df, config)
    df.to_csv(out_prefix.with_suffix(".synthetic_pairs.tsv"), sep="\t", index=False)
    selected.to_csv(out_prefix.with_suffix(".synthetic_selected.tsv"), sep="\t", index=False)
    print(json.dumps({"synthetic_pairs": len(df), "selected": selected["case"].value_counts().to_dict()}, indent=2))


def config_from_args(args: argparse.Namespace) -> Config:
    """Build typed config from CLI args."""
    if args.synthetic_test:
        return Config([], Path("dummy.bam"), args.sample, args.chrom, Path("dummy.tsv"), Path(args.out_prefix or "outputs/paired_read_interpretability_synthetic"))
    if not args.dataset or not args.bam or not args.out_prefix:
        raise SystemExit("--dataset, --bam and --out-prefix are required unless --synthetic-test is used")
    datasets = [DatasetSpec(Path(npz), Path(meta)) for npz, meta in args.dataset]
    return Config(
        datasets=datasets,
        bam=Path(args.bam),
        sample=args.sample,
        chrom=args.chrom,
        locus_variance=Path(args.locus_variance),
        out_prefix=Path(args.out_prefix),
        reference_fasta=Path(args.reference_fasta) if args.reference_fasta else None,
        top_loci=args.top_loci,
        max_candidate_loci=args.max_candidate_loci,
        max_reads_per_locus=args.max_reads_per_locus,
        window_size=args.window_size,
        bin_size=args.bin_size,
        min_reads_locus=args.min_reads_locus,
        min_window_coverage_fraction=args.min_window_coverage_fraction,
        min_pair_comparable_fraction=args.min_pair_comparable_fraction,
        min_base_quality=args.min_base_quality,
        min_valid_6ma_positions=args.min_valid_6ma_positions,
        min_valid_5mc_positions=args.min_valid_5mc_positions,
        max_missing_signal_fraction=args.max_missing_signal_fraction,
        max_mismatch_rate=args.max_mismatch_rate,
        min_valid_per_bin=args.min_valid_per_bin,
        min_shared_bins=args.min_shared_bins,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        top_n_per_case=args.top_n_per_case,
        seed=args.seed,
        reg_checkpoint=Path(args.reg_checkpoint) if args.reg_checkpoint else None,
        hyena_root=Path(args.hyena_root),
        checkpoint_dir=Path(args.checkpoint_dir),
        model_name=args.model_name,
        max_length=args.max_length,
        device=args.device,
        max_pairs_per_locus=args.max_pairs_per_locus,
        flag_substitution_distance=args.flag_substitution_distance,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    random.seed(config.seed)
    np.random.seed(config.seed)
    config.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.synthetic_test:
        run_synthetic_test(config.out_prefix)
        return

    profiles_by_locus = collect_profiles(config)
    pair_df = all_pair_metrics(profiles_by_locus, config)
    if pair_df.empty:
        raise SystemExit("No eligible read pairs found. Relax filters or increase --top-loci.")
    selected_df = select_case_pairs(pair_df, config)

    pair_path = config.out_prefix.with_suffix(".all_pairs.tsv")
    selected_path = config.out_prefix.with_suffix(".selected_pairs.tsv")
    pair_df.to_csv(pair_path, sep="\t", index=False)
    selected_df.to_csv(selected_path, sep="\t", index=False)
    plots = save_global_plots(pair_df, selected_df, config)
    locus_assoc = save_locus_associations(pair_df, config)
    dna_audit = save_selected_pair_dna_audit(selected_df, profiles_by_locus, config)
    if not selected_df.empty:
        plots.update(save_pair_track_plots(selected_df, profiles_by_locus, config))
    added = methylation_added_value(pair_df, config)
    report = write_report(pair_df, selected_df, plots, added, config)
    params = save_config(config)
    summary = {
        "n_loci_with_profiles": len(profiles_by_locus),
        "n_profiles": int(sum(len(v) for v in profiles_by_locus.values())),
        "n_eligible_pairs": int(len(pair_df)),
        "n_selected_pairs": int(len(selected_df)),
        "outputs": {
            "all_pairs": str(pair_path),
            "selected_pairs": str(selected_path),
            "within_locus_associations": locus_assoc,
            "selected_pair_position_audit": dna_audit,
            "report": report,
            "params": params,
            **plots,
        },
        "methylation_added_value": added,
    }
    summary_path = config.out_prefix.with_suffix(".summary.json")
    with summary_path.open("wt", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
