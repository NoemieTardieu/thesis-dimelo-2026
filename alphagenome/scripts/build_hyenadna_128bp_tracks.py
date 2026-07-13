#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmark_utils import CHROMS, load_regions, read_tsv
from reference_tracks import build_reference_track


class HyenaTwoHead(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head_5mc = nn.Linear(hidden_dim, 1)
        self.head_6ma = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head_6ma(self.backbone(input_ids)).squeeze(-1)


def prepare_hyenadna_imports() -> None:
    """Avoid importing incompatible torchvision features unused by DNA inference."""
    import transformers.utils.import_utils as import_utils

    import_utils._torchvision_available = False

    class InferenceStochasticDepth(nn.Module):
        def __init__(self, probability: float, mode: str) -> None:
            super().__init__()
            self.probability = probability
            self.mode = mode

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if self.training and self.probability:
                raise RuntimeError(
                    "The torchvision compatibility shim supports inference only."
                )
            return inputs

    torchvision = types.ModuleType("torchvision")
    torchvision.__path__ = []
    ops = types.ModuleType("torchvision.ops")
    ops.StochasticDepth = InferenceStochasticDepth
    torchvision.ops = ops
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.ops"] = ops


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CIGAR-aware 128 bp HyenaDNA prediction tracks.")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--outputs-dir", type=Path, default=Path("../outputs"))
    parser.add_argument("--hyena-root", type=Path, default=Path("../.."))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("../../checkpoints"))
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../outputs/hyenadna_small32k_4chrom_overlap16k_full5000_region_split_nosample_short_2epochs_1000batches.pt"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--grid", type=Path, help="Optional AlphaGenome bin grid TSV for exact test alignment.")
    parser.add_argument(
        "--bam-c1",
        type=Path,
        default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_c1.sorted.bam"),
    )
    parser.add_argument(
        "--bam-e5b",
        type=Path,
        default=Path("/staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam"),
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.hyena_root.resolve()))
    prepare_hyenadna_imports()
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hidden_dim = int(checkpoint["hidden_dim"])
    model = HyenaTwoHead(backbone, hidden_dim).to(device)
    model.backbone.load_state_dict(checkpoint.get("trainable_backbone_state_dict", {}), strict=False)
    model.head_5mc.load_state_dict(checkpoint["head_5mC_state_dict"])
    model.head_6ma.load_state_dict(checkpoint["head_6mA_state_dict"])
    model.eval()

    regions = load_regions(args.regions, split=args.split)
    grid_by_region = None
    if args.grid:
        grid_by_region = {}
        for row in read_tsv(args.grid):
            key = f"{row['chrom']}:{row['region_start']}-{row['region_end']}"
            grid_by_region.setdefault(key, []).append((int(row["bin_start"]), int(row["bin_end"])))
    out = args.out or Path(f"outputs/hyenadna_{args.split}_128bp.tsv")
    per_chrom_outputs = []
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
        input_ids = archive["input_ids"]
        masks = archive["mask_6mA"].astype(bool)

        def getter(indices: list[int]) -> tuple[dict[int, float], int]:
            sums: dict[int, float] = {}
            counts: dict[int, int] = {}
            raw = 0
            with torch.inference_mode():
                for idx in indices:
                    length = min(int(metadata[idx]["window_length"]), input_ids.shape[1])
                    tokens = torch.as_tensor(
                        input_ids[idx : idx + 1, :length], dtype=torch.long, device=device
                    )
                    predictions = torch.sigmoid(model(tokens))[0].cpu().numpy()
                    for local_pos in np.flatnonzero(masks[idx, :length]):
                        read_pos = int(metadata[idx]["window_start"]) + int(local_pos)
                        sums[read_pos] = sums.get(read_pos, 0.0) + float(predictions[local_pos])
                        counts[read_pos] = counts.get(read_pos, 0) + 1
                        raw += 1
            return ({pos: sums[pos] / counts[pos] for pos in sums}, raw - len(sums))

        chrom_out = out.with_name(f"{out.stem}.{chrom}{out.suffix}")
        build_reference_track(
            metadata_path,
            chrom_regions,
            {"merged_c1": args.bam_c1, "merged_e5b": args.bam_e5b},
            getter,
            chrom_out,
            chrom_out.with_suffix(".summary.json"),
            grid_by_region,
        )
        per_chrom_outputs.append(chrom_out)
        archive.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as destination:
        for index, path in enumerate(per_chrom_outputs):
            with open(path, "r", encoding="utf-8") as source:
                for line_number, line in enumerate(source):
                    if index and line_number == 0:
                        continue
                    destination.write(line)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
