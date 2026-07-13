#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pysam
import torch
from torch import nn


ID_TO_BASE = {
    7: "A",
    8: "C",
    9: "G",
    10: "T",
    11: "N",
    4: "-",
    6: "N",
}


@dataclass
class ReadObservation:
    sample: str
    read_id: str
    chrom: str
    position_0based: int
    row_indices: list[int]
    read_pos: int
    target_6ma: float
    target_5mc_context_mean: float | None
    target_5mc_context_observed: int
    dna_context: str
    prediction_6ma: float | None
    mapq: int
    cigar: str
    is_reverse: bool


class Hyena6mAMethylConditionedNoSample(nn.Module):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select same-locus read pairs for interpretability: reads with maximally "
            "different 6mA, and reads with similar 6mA but different local DNA."
        )
    )
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NPZ", "METADATA"),
        required=True,
        help="Tensor/metadata pair. Repeat for val/test/train if desired.",
    )
    parser.add_argument("--bam", required=True, help="BAM for the selected sample.")
    parser.add_argument("--sample", default="merged_c1")
    parser.add_argument(
        "--locus-variance",
        default="outputs/full_chr16_c1_locus_variance.per_locus_variance.tsv.gz",
        help="Per-locus variance table from analyze_full_chromosome_locus_variance.py.",
    )
    parser.add_argument("--chrom", default="chr16")
    parser.add_argument("--mark", default="6mA", choices=["6mA"])
    parser.add_argument("--top-loci", type=int, default=500)
    parser.add_argument("--min-reads", type=int, default=8)
    parser.add_argument("--local-window", type=int, default=80)
    parser.add_argument("--similar-6ma-delta", type=float, default=0.05)
    parser.add_argument(
        "--different-min-6ma",
        type=float,
        default=0.0,
        help=(
            "For the maximally different 6mA pair, require both reads to have at "
            "least this observed 6mA value. Use 0.001 to avoid choosing exact-zero "
            "reads in presentation examples."
        ),
    )
    parser.add_argument("--max-candidate-loci", type=int, default=100)
    parser.add_argument("--reg-checkpoint", default=None)
    parser.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    parser.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--out-prefix", required=True)
    return parser.parse_args()


def open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.endswith(".gz") else open(path, "r", encoding="utf-8", newline="")


def read_metadata(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"sample", "read_id", "chrom", "window_start", "window_length"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise SystemExit(f"{path} missing metadata columns: {sorted(missing)}")
    return rows


def select_loci(args: argparse.Namespace) -> list[tuple[str, int]]:
    top_rows = []
    counter = 0
    with open_text(args.locus_variance) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("group") != args.sample:
                continue
            if row.get("mark") != args.mark or row.get("chrom") != args.chrom:
                continue
            if int(float(row["count"] if "count" in row else row["n_reads"])) < args.min_reads:
                continue
            item = (float(row["variance"]), counter, row["chrom"], int(row["position_0based"]))
            counter += 1
            if len(top_rows) < args.top_loci:
                heapq.heappush(top_rows, item)
            elif item[0] > top_rows[0][0]:
                heapq.heapreplace(top_rows, item)
    if not top_rows:
        raise SystemExit("No candidate loci found. Check --sample/--chrom/--locus-variance.")
    top_rows.sort(reverse=True)
    return [(chrom, position) for _, _, chrom, position in top_rows]


def query_to_forward_pos(read: pysam.AlignedSegment, query_pos: int) -> int:
    read_length = int(read.query_length or 0)
    return read_length - 1 - int(query_pos) if read.is_reverse else int(query_pos)


def reference_locus_to_forward_pos(read: pysam.AlignedSegment, pos0: int) -> int | None:
    for query_pos, ref_pos in read.get_aligned_pairs(matches_only=False):
        if query_pos is None or ref_pos is None:
            continue
        if int(ref_pos) == pos0:
            return query_to_forward_pos(read, int(query_pos))
    return None


def collapse_value_at_read_pos(
    archive: np.lib.npyio.NpzFile,
    rows: list[dict[str, str]],
    row_indices: list[int],
    key_target: str,
    key_mask: str,
    read_pos: int,
) -> float | None:
    values = []
    targets = archive[key_target]
    masks = archive[key_mask].astype(bool)
    for row_idx in row_indices:
        row = rows[row_idx]
        local_pos = read_pos - int(row["window_start"])
        if local_pos < 0 or local_pos >= int(row["window_length"]) or local_pos >= targets.shape[1]:
            continue
        if masks[row_idx, local_pos]:
            values.append(float(targets[row_idx, local_pos]))
    return float(np.mean(values)) if values else None


def context_for_read_pos(
    archive: np.lib.npyio.NpzFile,
    rows: list[dict[str, str]],
    row_indices: list[int],
    read_pos: int,
    local_window: int,
) -> tuple[str, float | None, int]:
    input_ids = archive["input_ids"]
    target_5mc = archive["target_5mC"]
    mask_5mc = archive["mask_5mC"].astype(bool)
    best = None
    best_distance = None
    for row_idx in row_indices:
        row = rows[row_idx]
        local_pos = read_pos - int(row["window_start"])
        if local_pos < 0 or local_pos >= int(row["window_length"]) or local_pos >= input_ids.shape[1]:
            continue
        distance_to_edge = min(local_pos, input_ids.shape[1] - 1 - local_pos)
        if best_distance is None or distance_to_edge > best_distance:
            best = (row_idx, local_pos)
            best_distance = distance_to_edge
    if best is None:
        return "", None, 0

    row_idx, local_pos = best
    start = max(0, local_pos - local_window)
    end = min(input_ids.shape[1], local_pos + local_window + 1)
    dna = "".join(ID_TO_BASE.get(int(x), "N") for x in input_ids[row_idx, start:end])
    methyl_mask = mask_5mc[row_idx, start:end]
    methyl_values = target_5mc[row_idx, start:end][methyl_mask]
    methyl_mean = float(np.mean(methyl_values)) if methyl_values.size else None
    return dna, methyl_mean, int(methyl_values.size)


def dna_distance(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(sum(x != y for x, y in zip(a[:n], b[:n])) / n)


def load_reg_model(args: argparse.Namespace, archive: np.lib.npyio.NpzFile):
    if args.reg_checkpoint is None:
        return None, None
    sys.path.insert(0, str(Path(args.hyena_root)))
    from huggingface import HyenaDNAPreTrainedModel

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    backbone = HyenaDNAPreTrainedModel.from_pretrained(
        args.checkpoint_dir,
        args.model_name,
        download=False,
        device=device,
        use_head=False,
    ).to(device)
    backbone.eval()
    probe = torch.as_tensor(archive["input_ids"][0:1, : args.max_length], dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden_dim = int(backbone(probe).shape[-1])
    checkpoint = torch.load(args.reg_checkpoint, map_location=device, weights_only=False)
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


def predict_at_read_pos(
    model,
    device: str | None,
    archive: np.lib.npyio.NpzFile,
    rows: list[dict[str, str]],
    row_indices: list[int],
    read_pos: int,
    max_length: int,
) -> float | None:
    if model is None or device is None:
        return None
    preds = []
    for row_idx in row_indices:
        row = rows[row_idx]
        local_pos = read_pos - int(row["window_start"])
        if local_pos < 0 or local_pos >= int(row["window_length"]) or local_pos >= max_length:
            continue
        input_ids = torch.as_tensor(archive["input_ids"][row_idx : row_idx + 1, :max_length], dtype=torch.long, device=device)
        target_5mc = archive["target_5mC"][row_idx : row_idx + 1, :max_length]
        mask_5mc = archive["mask_5mC"][row_idx : row_idx + 1, :max_length].astype(bool)
        methyl_value = torch.as_tensor(np.where(mask_5mc, target_5mc, 0.0), dtype=torch.float32, device=device)
        methyl_observed = torch.as_tensor(mask_5mc.astype(np.float32), dtype=torch.float32, device=device)
        with torch.inference_mode():
            pred = torch.sigmoid(model(input_ids, methyl_value, methyl_observed)).detach().cpu().numpy()
        preds.append(float(pred[0, local_pos]))
    return float(np.mean(preds)) if preds else None


def collect_observations(args: argparse.Namespace, loci: list[tuple[str, int]]) -> list[ReadObservation]:
    rows_by_dataset: list[list[dict[str, str]]] = []
    archives = []
    row_indices_by_read: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for dataset_idx, (npz_path, meta_path) in enumerate(args.dataset):
        archive = np.load(npz_path)
        rows = read_metadata(meta_path)
        archives.append(archive)
        rows_by_dataset.append(rows)
        for row_idx, row in enumerate(rows):
            if row["sample"] == args.sample and row["chrom"] == args.chrom:
                row_indices_by_read[row["read_id"]].append((dataset_idx, row_idx))

    reg_model, device = load_reg_model(args, archives[0])

    loci_by_chrom = defaultdict(list)
    for chrom, pos in loci:
        loci_by_chrom[chrom].append(pos)

    observations = []
    with pysam.AlignmentFile(args.bam, "rb") as bam:
        for chrom, positions in loci_by_chrom.items():
            for pos0 in positions[: args.max_candidate_loci]:
                for read in bam.fetch(chrom, pos0, pos0 + 1):
                    if read.is_unmapped or read.is_secondary or read.is_supplementary:
                        continue
                    read_rows = row_indices_by_read.get(read.query_name)
                    if not read_rows:
                        continue
                    read_pos = reference_locus_to_forward_pos(read, pos0)
                    if read_pos is None:
                        continue
                    target_values = []
                    first_archive = None
                    first_dataset_idx = None
                    local_indices_for_first_dataset = []
                    for dataset_idx, global_idx in read_rows:
                        archive = archives[dataset_idx]
                        if first_archive is None:
                            first_archive = archive
                            first_dataset_idx = dataset_idx
                        if dataset_idx == first_dataset_idx:
                            local_indices_for_first_dataset.append(global_idx)
                        value = collapse_value_at_read_pos(
                            archive,
                            rows_by_dataset[dataset_idx],
                            [global_idx],
                            "target_6mA",
                            "mask_6mA",
                            read_pos,
                        )
                        if value is not None:
                            target_values.append(value)
                    if not target_values or first_archive is None:
                        continue
                    target_6ma = float(np.mean(target_values))
                    dna, methyl_mean, methyl_n = context_for_read_pos(
                        first_archive,
                        rows_by_dataset[int(first_dataset_idx)],
                        local_indices_for_first_dataset,
                        read_pos,
                        args.local_window,
                    )
                    pred = predict_at_read_pos(
                        reg_model,
                        device,
                        first_archive,
                        rows_by_dataset[int(first_dataset_idx)],
                        local_indices_for_first_dataset,
                        read_pos,
                        args.max_length,
                    )
                    observations.append(
                        ReadObservation(
                            sample=args.sample,
                            read_id=read.query_name,
                            chrom=chrom,
                            position_0based=pos0,
                            row_indices=local_indices_for_first_dataset,
                            read_pos=read_pos,
                            target_6ma=target_6ma,
                            target_5mc_context_mean=methyl_mean,
                            target_5mc_context_observed=methyl_n,
                            dna_context=dna,
                            prediction_6ma=pred,
                            mapq=int(read.mapping_quality),
                            cigar=read.cigarstring or "",
                            is_reverse=bool(read.is_reverse),
                        )
                    )
    return observations


def select_pairs(
    observations: list[ReadObservation],
    similar_delta: float,
    different_min_6ma: float,
) -> dict[str, tuple[ReadObservation, ReadObservation, float, float]]:
    by_locus: dict[tuple[str, int], list[ReadObservation]] = defaultdict(list)
    for obs in observations:
        by_locus[(obs.chrom, obs.position_0based)].append(obs)

    best_different_signal = None
    best_similar_signal_different_dna = None
    for values in by_locus.values():
        if len(values) < 2:
            continue
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                a, b = values[i], values[j]
                signal_delta = abs(a.target_6ma - b.target_6ma)
                dna_delta = dna_distance(a.dna_context, b.dna_context)
                if np.isnan(dna_delta):
                    continue
                score = (signal_delta, dna_delta)
                passes_min_signal = (
                    a.target_6ma >= different_min_6ma
                    and b.target_6ma >= different_min_6ma
                )
                if passes_min_signal and (
                    best_different_signal is None or score > best_different_signal[2:]
                ):
                    best_different_signal = (a, b, signal_delta, dna_delta)
                if signal_delta <= similar_delta:
                    reverse_score = (dna_delta, -signal_delta)
                    if (
                        best_similar_signal_different_dna is None
                        or reverse_score
                        > (best_similar_signal_different_dna[3], -best_similar_signal_different_dna[2])
                    ):
                        best_similar_signal_different_dna = (a, b, signal_delta, dna_delta)
    result = {}
    if best_different_signal is not None:
        result["same_locus_different_6mA"] = best_different_signal
    if best_similar_signal_different_dna is not None:
        result["same_locus_similar_6mA_different_DNA"] = best_similar_signal_different_dna
    return result


def write_outputs(args: argparse.Namespace, pairs: dict[str, tuple[ReadObservation, ReadObservation, float, float]]) -> None:
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pair_path = out_prefix.with_suffix(".selected_pairs.tsv")
    seq_path = out_prefix.with_suffix(".dna_context_alignment.tsv")
    summary_path = out_prefix.with_suffix(".summary.json")
    plot_path = out_prefix.with_suffix(".interpretability_pairs.png")

    pair_fields = [
        "case",
        "read_label",
        "sample",
        "read_id",
        "chrom",
        "position_0based",
        "read_pos",
        "target_6ma",
        "prediction_6ma",
        "target_5mc_context_mean",
        "target_5mc_context_observed",
        "mapq",
        "is_reverse",
        "cigar",
        "pair_signal_delta",
        "pair_dna_distance",
    ]
    with pair_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields, delimiter="\t")
        writer.writeheader()
        for case, (a, b, signal_delta, dna_delta) in pairs.items():
            for label, obs in (("read_A", a), ("read_B", b)):
                writer.writerow(
                    {
                        "case": case,
                        "read_label": label,
                        "sample": obs.sample,
                        "read_id": obs.read_id,
                        "chrom": obs.chrom,
                        "position_0based": obs.position_0based,
                        "read_pos": obs.read_pos,
                        "target_6ma": obs.target_6ma,
                        "prediction_6ma": obs.prediction_6ma,
                        "target_5mc_context_mean": obs.target_5mc_context_mean,
                        "target_5mc_context_observed": obs.target_5mc_context_observed,
                        "mapq": obs.mapq,
                        "is_reverse": obs.is_reverse,
                        "cigar": obs.cigar,
                        "pair_signal_delta": signal_delta,
                        "pair_dna_distance": dna_delta,
                    }
                )

    with seq_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["case", "offset_from_locus", "read_A_base", "read_B_base", "bases_match"])
        for case, (a, b, _signal_delta, _dna_delta) in pairs.items():
            n = min(len(a.dna_context), len(b.dna_context))
            center = n // 2
            for i in range(n):
                writer.writerow([case, i - center, a.dna_context[i], b.dna_context[i], a.dna_context[i] == b.dna_context[i]])

    fig, axes = plt.subplots(max(1, len(pairs)), 1, figsize=(9, 3.2 * max(1, len(pairs))), squeeze=False)
    for ax, (case, (a, b, signal_delta, dna_delta)) in zip(axes[:, 0], pairs.items()):
        labels = ["read A", "read B"]
        observed = [a.target_6ma, b.target_6ma]
        predicted = [a.prediction_6ma, b.prediction_6ma]
        methyl = [
            np.nan if a.target_5mc_context_mean is None else a.target_5mc_context_mean,
            np.nan if b.target_5mc_context_mean is None else b.target_5mc_context_mean,
        ]
        x = np.arange(2)
        ax.bar(x - 0.22, observed, width=0.22, label="observed 6mA")
        if all(v is not None for v in predicted):
            ax.bar(x, predicted, width=0.22, label="predicted 6mA")
        ax.bar(x + 0.22, methyl, width=0.22, label="local 5mC mean")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("probability")
        ax.set_title(f"{case}: Δ6mA={signal_delta:.3f}, DNA distance={dna_delta:.3f}")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    summary = {
        "cases": list(pairs),
        "outputs": {
            "selected_pairs": str(pair_path),
            "dna_context_alignment": str(seq_path),
            "plot": str(plot_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "same_locus_different_6mA tests whether reads mapping to the same locus can "
            "have very different regulatory labels. same_locus_similar_6mA_different_DNA "
            "tests whether local read DNA differences necessarily imply regulatory-label "
            "differences. Adding local 5mC and optional P(Reg|D,M) predictions lets you "
            "ask whether methylation explains variation better than DNA sequence alone."
        ),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    loci = select_loci(args)
    observations = collect_observations(args, loci)
    pairs = select_pairs(observations, args.similar_6ma_delta, args.different_min_6ma)
    if not pairs:
        raise SystemExit("No suitable pairs found. Try larger --top-loci or --similar-6ma-delta.")
    write_outputs(args, pairs)


if __name__ == "__main__":
    main()
