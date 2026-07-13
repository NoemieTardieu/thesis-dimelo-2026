#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Minimal HyenaDNA tiny-1k embedding smoke test adapted from the public Colab."
    )
    p.add_argument(
        "--hyena-root",
        default="/data/leuven/383/vsc38330/hyena-dna-main",
        help="Path to the local hyena-dna-main checkout.",
    )
    p.add_argument(
        "--model-name",
        default="hyenadna-tiny-1k-seqlen",
        help="LongSafari model name to load.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints",
        help="Directory containing or receiving LongSafari checkpoints.",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Download the checkpoint from HuggingFace if it is not already present.",
    )
    p.add_argument(
        "--download-method",
        default="huggingface_hub",
        choices=["huggingface_hub", "hyenadna_git_lfs"],
        help=(
            "Download method. huggingface_hub avoids needing the git-lfs executable; "
            "hyenadna_git_lfs uses the original HyenaDNA helper."
        ),
    )
    p.add_argument(
        "--sequence",
        default=("ACGT" * 250),
        help="DNA sequence for the smoke test. Defaults to exactly 1000 nt.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Tokenizer/model maximum length for tiny-1k.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    hyena_root = Path(args.hyena_root)
    if not hyena_root.exists():
        raise SystemExit(f"HyenaDNA root does not exist: {hyena_root}")

    sys.path.insert(0, str(hyena_root))

    from huggingface import HyenaDNAPreTrainedModel
    from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    checkpoint_root = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_root / args.model_name
    use_hyenadna_download = args.download
    if args.download and args.download_method == "huggingface_hub":
        from huggingface_hub import snapshot_download

        checkpoint_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=f"LongSafari/{args.model_name}",
            local_dir=str(checkpoint_path),
            local_dir_use_symlinks=False,
        )
        use_hyenadna_download = False

    model = HyenaDNAPreTrainedModel.from_pretrained(
        args.checkpoint_dir,
        args.model_name,
        download=use_hyenadna_download,
        device=device,
        use_head=False,
    )
    model = model.to(device)
    model.eval()

    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=args.max_length,
        padding_side="left",
    )

    encoded = tokenizer(
        args.sequence.upper(),
        add_special_tokens=False,
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
    )
    input_ids = torch.LongTensor(encoded["input_ids"]).unsqueeze(0).to(device)

    with torch.inference_mode():
        hidden = model(input_ids)

    print(f"model_name\t{args.model_name}")
    print(f"device\t{device}")
    print(f"input_ids_shape\t{tuple(input_ids.shape)}")
    print(f"hidden_shape\t{tuple(hidden.shape)}")
    print(f"hidden_dtype\t{hidden.dtype}")
    print(f"hidden_mean\t{hidden.float().mean().item():.6g}")
    print(f"hidden_std\t{hidden.float().std().item():.6g}")


if __name__ == "__main__":
    main()
