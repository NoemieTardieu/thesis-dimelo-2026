#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import torch
from torch import nn

from benchmark_utils import CHROMS, collapse_windows, load_regions, read_tsv
from build_hyenadna_128bp_tracks import prepare_hyenadna_imports
from reference_tracks import build_reference_track


class HyenaTwoHead(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head_5mc = nn.Linear(hidden_dim, 1)
        self.head_6ma = nn.Linear(hidden_dim, 1)

    def forward_5mc(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head_5mc(self.backbone(input_ids)).squeeze(-1)


def cpg_dyad_reference_position(
    alignment: pysam.AlignedSegment, reference_pos: int
) -> int | None:
    # In the tensorization convention, reverse-strand CpG cytosine calls are
    # represented one base after the forward-strand CpG coordinate.
    if alignment.is_reverse:
        reference_pos -= 1
    return reference_pos if reference_pos >= 0 else None


def load_hyenadna_model(args: argparse.Namespace, device: str) -> HyenaTwoHead:
    sys.path.insert(0, str(args.hyena_root.resolve()))
    prepare_hyenadna_imports()
    from huggingface import HyenaDNAPreTrainedModel

    backbone = HyenaDNAPreTrainedModel.from_pretrained(
        args.checkpoint_dir,
        args.model_name,
        download=False,
        device=device,
        use_head=False,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HyenaTwoHead(backbone, int(checkpoint["hidden_dim"])).to(device)
    model.backbone.load_state_dict(checkpoint.get("trainable_backbone_state_dict", {}), strict=False)
    model.head_5mc.load_state_dict(checkpoint["head_5mC_state_dict"])
    model.head_6ma.load_state_dict(checkpoint["head_6mA_state_dict"])
    model.eval()
    return model


def build_tracks(args: argparse.Namespace, device: str) -> tuple[Path, Path]:
    regions = load_regions(args.regions, split=args.split)
    dimelo_out = args.out_dir / f"dimelo_5mc_{args.split}_{args.bin_size}bp.tsv"
    hyena_out = args.out_dir / f"hyenadna_5mc_{args.split}_{args.bin_size}bp.tsv"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_hyenadna_model(args, device)
    dimelo_parts = []
    hyena_parts = []
    for chrom in CHROMS:
        chrom_regions = [region for region in regions if region.chrom == chrom]
        prefix = (
            args.outputs_dir
            / f"merged_e5b_c1_{chrom}_selected_top100_overlap16k_full5000_region_split.{args.split}"
        )
        metadata_path = Path(f"{prefix}.metadata.tsv")
        npz_path = Path(f"{prefix}.npz")
        metadata = read_tsv(metadata_path)
        archive = np.load(npz_path)
        target_5mc = archive["target_5mC"]
        mask_5mc = archive["mask_5mC"].astype(bool)
        input_ids = archive["input_ids"]

        def target_getter(indices: list[int]) -> tuple[dict[int, float], int]:
            return collapse_windows(target_5mc, mask_5mc, metadata, indices)

        def pred_getter(indices: list[int]) -> tuple[dict[int, float], int]:
            sums: dict[int, float] = {}
            counts: dict[int, int] = {}
            raw = 0
            with torch.inference_mode():
                for idx in indices:
                    length = min(int(metadata[idx]["window_length"]), input_ids.shape[1])
                    tokens = torch.as_tensor(
                        input_ids[idx : idx + 1, :length], dtype=torch.long, device=device
                    )
                    predictions = torch.sigmoid(model.forward_5mc(tokens))[0].cpu().numpy()
                    for local_pos in np.flatnonzero(mask_5mc[idx, :length]):
                        read_pos = int(metadata[idx]["window_start"]) + int(local_pos)
                        sums[read_pos] = sums.get(read_pos, 0.0) + float(predictions[local_pos])
                        counts[read_pos] = counts.get(read_pos, 0) + 1
                        raw += 1
            return ({pos: sums[pos] / counts[pos] for pos in sums}, raw - len(sums))

        for label, getter, out_path, part_paths in (
            ("dimelo", target_getter, dimelo_out, dimelo_parts),
            ("hyenadna", pred_getter, hyena_out, hyena_parts),
        ):
            chrom_out = out_path.with_name(f"{out_path.stem}.{chrom}{out_path.suffix}")
            build_reference_track(
                metadata_path,
                chrom_regions,
                {"merged_c1": args.bam_c1, "merged_e5b": args.bam_e5b},
                getter,
                chrom_out,
                chrom_out.with_suffix(".summary.json"),
                bin_size=args.bin_size,
                reference_transform=cpg_dyad_reference_position,
            )
            part_paths.append(chrom_out)
            print(f"Wrote {label} {chrom_out}")
        archive.close()

    for out_path, part_paths in ((dimelo_out, dimelo_parts), (hyena_out, hyena_parts)):
        with open(out_path, "w", encoding="utf-8") as destination:
            for index, part in enumerate(part_paths):
                with open(part, "r", encoding="utf-8") as source:
                    for line_number, line in enumerate(source):
                        if index and line_number == 0:
                            continue
                        destination.write(line)
        print(f"Wrote {out_path}")
    return dimelo_out, hyena_out


def plot_tracks(dimelo_path: Path, hyena_path: Path, args: argparse.Namespace) -> None:
    dimelo = pd.read_csv(dimelo_path, sep="\t")
    hyena = pd.read_csv(hyena_path, sep="\t")
    plot_dir = args.out_dir / f"plots_5mc_{args.bin_size}bp"
    plot_dir.mkdir(parents=True, exist_ok=True)
    regions = (
        dimelo[["chrom", "region_id", "region_name"]]
        .drop_duplicates()
        .sort_values(["chrom", "region_id"])
        .head(args.max_regions)
    )
    for _, selected in regions.iterrows():
        chrom = selected["chrom"]
        region_id = selected["region_id"]
        region_name = selected["region_name"]
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        for sample, color in (("merged_c1", "tab:blue"), ("merged_e5b", "tab:orange")):
            d = dimelo[
                (dimelo["chrom"] == chrom)
                & (dimelo["region_id"] == region_id)
                & (dimelo["sample"] == sample)
            ]
            h = hyena[
                (hyena["chrom"] == chrom)
                & (hyena["region_id"] == region_id)
                & (hyena["sample"] == sample)
            ]
            axes[0].plot((d["bin_start"] + d["bin_end"]) / 2, d["mean_signal"], label=sample, color=color)
            axes[1].plot((h["bin_start"] + h["bin_end"]) / 2, h["mean_signal"], label=sample, color=color)
        axes[0].set_ylabel("DiMeLo 5mC")
        axes[1].set_ylabel("HyenaDNA 5mC")
        axes[1].set_xlabel(f"{chrom} coordinate (hg38)")
        axes[0].legend(frameon=False)
        axes[1].legend(frameon=False)
        fig.suptitle(f"{chrom} region {region_id}: 5mC target vs HyenaDNA\n{region_name}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{chrom}_region{region_id}_5mc.png", dpi=180)
        plt.close(fig)
    print(f"Wrote 5mC plots to {plot_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and plot DiMeLo-vs-HyenaDNA 5mC reference tracks.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--regions", type=Path, default=Path("outputs/4chrom_test_regions.tsv"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/outputs"))
    parser.add_argument("--hyena-root", type=Path, default=Path("/data/leuven/383/vsc38330/hyena-dna-main"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/data/leuven/383/vsc38330/hyena-dna-main/checkpoints"))
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/outputs/hyenadna_small32k_4chrom_overlap16k_full5000_region_split_nosample_short_2epochs_1000batches.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bin-size", type=int, default=1000)
    parser.add_argument("--max-regions", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/5mc_visual"))
    parser.add_argument("--bam-c1", type=Path, default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam"))
    parser.add_argument("--bam-e5b", type=Path, default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dimelo_path, hyena_path = build_tracks(args, device)
    plot_tracks(dimelo_path, hyena_path, args)


if __name__ == "__main__":
    main()
