#!/bin/bash
#SBATCH --job-name=check_hyena_b200
#SBATCH --clusters=mindwell
#SBATCH --account=lccbe
#SBATCH --partition=gpu_b200
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-check-hyena-imports-mindwell-b200-%j.out
#SBATCH --error=/data/leuven/383/vsc38330/hyena-dna-main/preprocessing_chr16_merged_e5b/slurm-check-hyena-imports-mindwell-b200-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1

cd /data/leuven/383/vsc38330/hyena-dna-main

/lustre1/scratch/383/vsc38330/.venv/bin/python - <<'PY'
import json
import sys

sys.path.insert(0, "/data/leuven/383/vsc38330/hyena-dna-main")

import torch
import numpy
import einops
import transformers
import torchvision

print(json.dumps({
    "python": sys.executable,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "numpy": numpy.__version__,
    "einops": einops.__version__,
    "transformers": transformers.__version__,
}, indent=2))

from huggingface import HyenaDNAPreTrainedModel

model = HyenaDNAPreTrainedModel.from_pretrained(
    "./checkpoints",
    "hyenadna-small-32k-seqlen",
    download=False,
    config=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
probe = torch.randint(0, 4, (1, 2048), device=device)
with torch.no_grad():
    out = model(probe)
print(json.dumps({"model_load_and_forward": "ok", "output_shape": list(out.shape)}))
PY
