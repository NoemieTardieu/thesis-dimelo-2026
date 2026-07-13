#!/usr/bin/env python3
"""Check imports needed for HyenaDNA training in the active Python env."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def check_import(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
        return {
            "name": name,
            "ok": True,
            "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        return {"name": name, "ok": False, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyena-root", default="/data/leuven/383/vsc38330/hyena-dna-main")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--model-name", default="hyenadna-small-32k-seqlen")
    parser.add_argument("--checkpoint-dir", default="/data/leuven/383/vsc38330/hyena-dna-main/checkpoints")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.hyena_root)))
    print(json.dumps({"python": sys.executable, "sys_path_0": sys.path[0]}), flush=True)

    modules = [
        "numpy",
        "torch",
        "einops",
        "torchvision",
        "torchvision.ops",
        "transformers",
        "transformers.tokenization_utils",
        "standalone_hyenadna",
        "huggingface",
    ]
    results = [check_import(name) for name in modules]
    print(json.dumps({"imports": results}, indent=2), flush=True)

    failed = [item for item in results if not item["ok"]]
    if failed:
        raise SystemExit(1)

    import torch

    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
            },
            indent=2,
        ),
        flush=True,
    )

    if args.load_model:
        from huggingface import HyenaDNAPreTrainedModel

        model = HyenaDNAPreTrainedModel.from_pretrained(
            args.checkpoint_dir,
            args.model_name,
            download=False,
            device="cpu",
            use_head=False,
        )
        print(json.dumps({"model_load": "ok", "type": type(model).__name__}), flush=True)


if __name__ == "__main__":
    main()
