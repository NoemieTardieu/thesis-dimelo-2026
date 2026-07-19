#!/usr/bin/env python3
"""Create HyenaDNA tensors with predicted 5mC as methylation input.

This utility is for chained/factorized evaluation:

    D -> P(M|D) -> M_hat
    D + M_hat -> P(Reg|D,M_hat)

It leaves all original arrays unchanged except the 5mC conditioning arrays:
`target_5mC` is replaced by predicted 5mC probabilities, and `mask_5mC`
is set according to the requested mode. The output NPZ can then be passed to
the existing methylation-conditioned regulatory evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DimeloSequenceDataset(Dataset):
    def __init__(self, npz_path: str | Path, max_length: int) -> None:
        self.path = Path(npz_path)
        self.data = np.load(self.path)
        self.max_length = int(max_length)
        self.n = int(self.data["input_ids"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "row_number": torch.tensor(idx, dtype=torch.long),
            "input_ids": torch.as_tensor(
                self.data["input_ids"][idx, : self.max_length], dtype=torch.long
            ),
        }


class Hyena5mCOnly(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        decoder_hidden_dim: int = 0,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if decoder_hidden_dim > 0:
            self.head_5mc = nn.Sequential(
                nn.Linear(hidden_dim, decoder_hidden_dim),
                nn.GELU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(decoder_hidden_dim, 1),
            )
        else:
            self.head_5mc = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids)
        return self.head_5mc(hidden).squeeze(-1)


def ensure_transformers_stub() -> None:
    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("transformers")
        module.PreTrainedModel = nn.Module
        sys.modules["transformers"] = module


def derive_cpg_mask(input_ids: np.ndarray, c_token_id: int, g_token_id: int) -> np.ndarray:
    is_cpg = np.zeros_like(input_ids, dtype=bool)
    if input_ids.shape[1] > 1:
        is_cpg[:, :-1] = (input_ids[:, :-1] == c_token_id) & (
            input_ids[:, 1:] == g_token_id
        )
    return is_cpg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace target_5mC/mask_5mC in an NPZ with P(5mC|DNA) predictions."
    )
    parser.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/upstream_hyena_dna")
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/server_artifacts/upstream_checkpoints")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--checkpoint-5mc", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--c-token-id", type=int, default=8)
    parser.add_argument("--g-token-id", type=int, default=9)
    parser.add_argument(
        "--predicted-mask-mode",
        choices=["all-cpg", "original-5mc-mask"],
        default="all-cpg",
        help=(
            "all-cpg makes every sequence CpG use M_hat; original-5mc-mask only "
            "replaces positions where observed 5mC was available and leaves other "
            "CpGs as unknown to the regulatory model."
        ),
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.hyena_root)))
    ensure_transformers_stub()

    from huggingface import HyenaDNAPreTrainedModel

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    dataset = DimeloSequenceDataset(args.npz, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    backbone = HyenaDNAPreTrainedModel.from_pretrained(
        args.checkpoint_dir,
        args.model_name,
        download=False,
        device=device,
        use_head=False,
    ).to(device)
    backbone.eval()

    probe = next(iter(loader))["input_ids"].to(device)
    with torch.inference_mode():
        hidden_probe = backbone(probe)
    hidden_dim = int(hidden_probe.shape[-1])

    checkpoint = torch.load(args.checkpoint_5mc, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    decoder_hidden_dim = int(
        checkpoint.get("decoder_hidden_dim", checkpoint_args.get("decoder_hidden_dim", 0))
    )
    decoder_dropout = float(
        checkpoint.get("decoder_dropout", checkpoint_args.get("decoder_dropout", 0.0))
    )
    model = Hyena5mCOnly(
        backbone,
        hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_dropout=decoder_dropout,
    ).to(device)
    if checkpoint.get("trainable_backbone_state_dict"):
        model.backbone.load_state_dict(checkpoint["trainable_backbone_state_dict"], strict=False)
    model.head_5mc.load_state_dict(checkpoint["head_5mC_state_dict"])
    model.eval()

    source = np.load(args.npz)
    arrays = {name: source[name] for name in source.files}
    input_ids = arrays["input_ids"]
    if input_ids.shape[1] < args.max_length:
        raise SystemExit(
            f"Input length {input_ids.shape[1]} is shorter than max_length {args.max_length}."
        )

    predicted_5mc = np.array(arrays["target_5mC"], copy=True)
    with torch.inference_mode():
        for batch in loader:
            row_numbers = batch["row_number"].numpy()
            logits = model(batch["input_ids"].to(device))
            pred = torch.sigmoid(logits).detach().cpu().numpy().astype(predicted_5mc.dtype)
            predicted_5mc[row_numbers, : args.max_length] = pred

    is_cpg = derive_cpg_mask(input_ids[:, : args.max_length], args.c_token_id, args.g_token_id)
    new_mask = np.zeros_like(arrays["mask_5mC"], dtype=bool)
    if args.predicted_mask_mode == "all-cpg":
        new_mask[:, : args.max_length] = is_cpg
    else:
        original = arrays["mask_5mC"][:, : args.max_length].astype(bool)
        new_mask[:, : args.max_length] = original & is_cpg

    arrays["target_5mC"] = predicted_5mc
    arrays["mask_5mC"] = new_mask.astype(arrays["mask_5mC"].dtype, copy=False)
    if "is_cpg" not in arrays:
        full_is_cpg = np.zeros_like(arrays["mask_5mC"], dtype=bool)
        full_is_cpg[:, : args.max_length] = is_cpg
        arrays["is_cpg"] = full_is_cpg

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)

    summary = {
        "source_npz": str(args.npz),
        "out_npz": str(out_npz),
        "checkpoint_5mc": str(args.checkpoint_5mc),
        "model_type": "P(M|D) predicted 5mC conditioning tensor",
        "predicted_mask_mode": args.predicted_mask_mode,
        "rows": len(dataset),
        "max_length": args.max_length,
        "c_token_id": args.c_token_id,
        "g_token_id": args.g_token_id,
        "cpg_positions_with_mhat": int(new_mask[:, : args.max_length].sum()),
        "mean_mhat_at_conditioned_positions": (
            float(predicted_5mc[:, : args.max_length][new_mask[:, : args.max_length]].mean())
            if int(new_mask[:, : args.max_length].sum()) > 0
            else None
        ),
    }
    with open(out_npz.with_suffix(".summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
