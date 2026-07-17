#!/usr/bin/env python3
"""Export methylation-conditioned HyenaDNA predictions as 200 bp genomic tracks.

This is the population-track companion for the final P(Reg | DNA, 5mC) model.
Predictions are made on ONT read windows, overlapping windows are collapsed by
sample/read/read-position, read positions are projected to GRCh38 through the
BAM CIGAR, and values are aggregated into the same 200 bp table format used by
the AlphaGenome population benchmark.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
ALPHAGENOME_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ALPHAGENOME_ROOT.parent
HYENA_ROOT = PROJECT_ROOT / "hyena-dna"

sys.path.insert(0, str(ALPHAGENOME_ROOT / "src"))
from benchmark_utils import CHROMS, load_regions, read_tsv  # noqa: E402
from reference_tracks import build_reference_track  # noqa: E402


def load_eval_module(path: Path):
    """Load the final HyenaDNA evaluation module without requiring package imports."""

    spec = importlib.util.spec_from_file_location("methyl_embedding_eval", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load evaluation module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_hyenadna_imports() -> None:
    """Avoid importing torchvision features unused by DNA inference."""

    import transformers.utils.import_utils as import_utils

    import_utils._torchvision_available = False

    class InferenceStochasticDepth(nn.Module):
        def __init__(self, probability: float, mode: str) -> None:
            super().__init__()
            self.probability = probability
            self.mode = mode

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if self.training and self.probability:
                raise RuntimeError("The torchvision compatibility shim supports inference only.")
            return inputs

    torchvision = types.ModuleType("torchvision")
    torchvision.__path__ = []
    ops = types.ModuleType("torchvision.ops")
    ops.StochasticDepth = InferenceStochasticDepth
    torchvision.ops = ops
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.ops"] = ops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--regions", type=Path, default=ALPHAGENOME_ROOT / "metadata/regions/4chrom_test_regions.tsv")
    parser.add_argument("--outputs-dir", type=Path, default=HYENA_ROOT / "logs/generated_tensors_and_checkpoints")
    parser.add_argument("--hyena-root", type=Path, default=HYENA_ROOT / "upstream_hyena_dna")
    parser.add_argument("--checkpoint-dir", type=Path, default=HYENA_ROOT / "server_artifacts/upstream_checkpoints")
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=HYENA_ROOT / "results/checkpoints/hyenadna_reg_given_m_methyl_embedding_long.best.pt",
    )
    parser.add_argument(
        "--eval-module",
        type=Path,
        default=HYENA_ROOT
        / "evaluation/evaluate_region_split_hyenadna_6ma_methyl_embedding_conditioned_nosample_overlap_aggregated.py",
    )
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--bin-size", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    parser.add_argument(
        "--out",
        type=Path,
        default=ALPHAGENOME_ROOT / "results/model_comparison_inputs/hyenadna_methyl_embedding_test_200bp.tsv",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.hyena_root.resolve()))
    prepare_hyenadna_imports()
    eval_module = load_eval_module(args.eval_module)
    from huggingface import HyenaDNAPreTrainedModel

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    c_token_id = int(checkpoint.get("c_token_id", checkpoint.get("args", {}).get("c_token_id", 8)))
    g_token_id = int(checkpoint.get("g_token_id", checkpoint.get("args", {}).get("g_token_id", 9)))

    backbone = HyenaDNAPreTrainedModel.from_pretrained(
        args.checkpoint_dir,
        args.model_name,
        download=False,
        device=device,
        use_head=False,
    ).to(device)
    backbone.eval()

    probe_ids = torch.zeros((1, min(args.max_length, 16)), dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden_dim = int(backbone(probe_ids).shape[-1])

    if checkpoint.get("methylation_conditioning") != "input_embedding_addition":
        raise SystemExit(f"{args.checkpoint} is not a methylation-embedding checkpoint.")
    model = eval_module.Hyena6mAMethylEmbeddingConditionedNoSample(
        backbone,
        hidden_dim,
        decoder_hidden_dim=int(checkpoint.get("decoder_hidden_dim", 0)),
        decoder_dropout=float(checkpoint.get("decoder_dropout", 0.0)),
        unknown_methylation_value=float(checkpoint.get("unknown_methylation_value", 0.5)),
    ).to(device)
    if checkpoint.get("trainable_backbone_state_dict"):
        model.backbone.load_state_dict(checkpoint["trainable_backbone_state_dict"], strict=False)
    model.head_6ma.load_state_dict(checkpoint["head_6mA_state_dict"])
    embeddings = checkpoint["methylation_embedding_state_dict"]
    model.unmethylated_cpg_embedding.data.copy_(embeddings["unmethylated_cpg_embedding"].to(device))
    model.methylated_cpg_embedding.data.copy_(embeddings["methylated_cpg_embedding"].to(device))
    model.non_cpg_embedding.data.copy_(embeddings["non_cpg_embedding"].to(device))
    model.eval()

    regions = load_regions(args.regions, split=args.split)
    per_chrom_outputs = []
    for chrom in CHROMS:
        chrom_regions = [region for region in regions if region.chrom == chrom]
        if not chrom_regions:
            continue
        prefix = args.outputs_dir / f"merged_e5b_c1_{chrom}_selected_top100_overlap16k_full5000_region_split.{args.split}"
        metadata_path = Path(f"{prefix}.metadata.tsv")
        npz_path = Path(f"{prefix}.npz")
        if not metadata_path.exists() or not npz_path.exists():
            raise SystemExit(f"Missing tensor inputs for {chrom}: {metadata_path}, {npz_path}")
        metadata = read_tsv(metadata_path)
        dataset = eval_module.Dimelo6mAMethylEmbeddingDataset(npz_path, args.max_length, c_token_id, g_token_id)
        if len(metadata) != len(dataset):
            raise SystemExit(f"{chrom}: metadata rows ({len(metadata)}) != tensor rows ({len(dataset)})")

        def getter(indices: list[int]) -> tuple[dict[int, float], int]:
            sums: dict[int, float] = {}
            counts: dict[int, int] = {}
            raw = 0
            with torch.inference_mode():
                for idx in indices:
                    item = dataset[idx]
                    length = min(int(metadata[idx]["window_length"]), args.max_length)
                    input_ids = item["input_ids"][:length].unsqueeze(0).to(device)
                    methyl_value = item["methyl_value"][:length].unsqueeze(0).to(device)
                    methyl_observed = item["methyl_observed"][:length].unsqueeze(0).to(device)
                    is_cpg = item["is_cpg"][:length].unsqueeze(0).to(device)
                    mask = item["mask_6mA"][:length].cpu().numpy().astype(bool)
                    pred = torch.sigmoid(
                        model(input_ids, methyl_value, methyl_observed, is_cpg)
                    )[0].detach().cpu().numpy()
                    for local_pos in np.flatnonzero(mask):
                        read_pos = int(metadata[idx]["window_start"]) + int(local_pos)
                        sums[read_pos] = sums.get(read_pos, 0.0) + float(pred[local_pos])
                        counts[read_pos] = counts.get(read_pos, 0) + 1
                        raw += 1
            return ({pos: sums[pos] / counts[pos] for pos in sums}, raw - len(sums))

        chrom_out = args.out.with_name(f"{args.out.stem}.{chrom}{args.out.suffix}")
        build_reference_track(
            metadata_path,
            chrom_regions,
            {"merged_c1": args.bam_c1, "merged_e5b": args.bam_e5b},
            getter,
            chrom_out,
            chrom_out.with_suffix(".summary.json"),
            grid_by_region=None,
            bin_size=args.bin_size,
        )
        per_chrom_outputs.append(chrom_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as destination:
        for index, path in enumerate(per_chrom_outputs):
            with path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source):
                    if index and line_number == 0:
                        continue
                    destination.write(line)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
