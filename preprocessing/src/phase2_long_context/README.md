# Phase 2 Long-Context Preprocessing

This module builds long-interval methylation labels (`0/1/2`) from DiMeLo modBAM files for HyenaDNA-style long-context training.

## Outputs

1. `intervals_long_context.tsv`
- Columns: `window_id, chrom, start, end, split`
- Typical setup: `window_size=1,000,000` and `stride=1,000,000`

2. `manifest_<mark>_<base>.tsv`
- One row per interval
- Includes `npz_path`, `known_frac`, and `methylated_frac_known`

3. Per-interval NPZ files
- `methyl_ids`: uint8 labels per base position (`0=unmethylated`, `1=methylated`, `2=unknown`)
- `coverage`: number of reads covering target base at each position
- `meth_counts`: confident modified counts at each position

## Step 1: Generate long-context intervals

```bash
python3 generate_long_context_intervals.py \
  --bam /path/to/mark.bam \
  --out-tsv intervals_long_context.tsv \
  --window-size 1000000 \
  --stride 1000000
```

## Step 2: Build methylation backend from BAM + intervals

Example for C-based methylation labels:

```bash
python3 build_methylation_interval_backend.py \
  --bam /path/to/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam \
  --intervals-tsv intervals_long_context.tsv \
  --out-dir ./backend_h3k27ac \
  --mark h3k27ac \
  --target-base C \
  --min-mapq 20 \
  --ml-threshold 128 \
  --min-coverage 3 \
  --methylated-frac-thr 0.7 \
  --unmethylated-frac-thr 0.3 \
  --resume \
  --progress-every 10
```

For quick smoke test, add:

```bash
--max-intervals 20
```

## Step 3: Use loader hook in Hyena dataset

Use `methylation_interval_store.py`:

```python
from methylation_interval_store import make_interval_loader

loader = make_interval_loader(
    manifest_tsv="/path/to/manifest_h3k27ac_C.tsv",
    mark="h3k27ac",
)

# loader(chrom, start, end) -> np.ndarray of shape [L], values {0,1,2}
```

Then pass this callable into the dataset as `methylation_loader`.

## Notes

- This backend is interval-indexed and intended for long-context training.
- Unknown labels (`2`) are expected where coverage is low or ambiguous.
- Keep chromosome split (`train/valid/test`) fixed at interval generation stage.
- If your run is interrupted, restart with `--resume`.
