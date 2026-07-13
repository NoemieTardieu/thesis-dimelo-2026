## Phase 2 Long-Context Backend and Windowed Analysis

### Methods (brief)
After Phase 1 read-level QC, the dataset was reformatted into long genomic context windows. First, the genome was partitioned into fixed 1 Mb intervals with chromosome-based train/validation/test split labels to avoid leakage across homologous regions. Next, for each histone-mark-specific modBAM (`h3k27ac`, `h3k27me3`, `h3k4me3`), a backend was built per interval, storing per-position C-channel labels as discrete states (`0` unmethylated, `1` methylated, `2` unknown) in NPZ files with per-window manifests. For the A-channel, the pipeline did not store discrete per-position labels; instead it aggregated confident A-modification calls into continuous signals, including `reg_a_mod_per_kb` (A-mod signal per kilobase) and, in a later extension, a bin-level A-methylation fraction defined as `(# A calls with ML ≥ 0.8) / (total A calls)` in each bin. Finally, window-level structure was analyzed by sub-binning each 1 Mb interval into smaller bins (10 kb and 1 kb runs), computing Reg and M features per bin, and visualizing observed/expected (O/E) enrichment heatmaps.

### Why this was done
Phase 2 bridges raw read-level signal and model-ready long-context data. The 1 Mb interval backend preserves long-range genomic context required for downstream HyenaDNA-style modeling, while 1 kb/10 kb sub-bin analyses provide interpretable local structure and quality checks before modeling. This separation allows both biological interpretability and computational scalability.

### What was produced
1. Interval table (`intervals_long_context.tsv`): fixed 1 Mb windows with split assignments.
2. Mark-specific backend stores: NPZ label arrays + manifest TSV per mark.
3. Backend QC summaries: known-label fraction, methylated fraction, read-usage stats.
4. Windowed O/E enrichment outputs: per-mark, pooled, and by-mark heatmaps plus summary notes/tables.

### Operational robustness
Long-running commands were configured with:
- `--progress-every 25` for regular status updates.
- `--resume` for safe restart after session interruption.
- `--checkpoint-every 25` (windowed O/E script) for partial-progress recovery.

### Interpretation value for thesis
This phase establishes that long-context preprocessing is technically complete and reproducible, and that local Reg-vs-M structure can be quantified at biologically meaningful resolution before entering Phase 3 modeling. It provides the validated, split-aware data foundation needed for multi-modal sequence+methylation model development.


