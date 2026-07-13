# AlphaGenome Read-Level Failure Diagnostics

AlphaGenome was run on original ONT read sequences and evaluated against DiMeLo 6mA observations at read positions.

## Main Read-Level Result

- Pooled Pearson: 0.111
- Pooled Spearman: -0.051
- Pooled AUROC: 0.546
- Pooled AUPRC: 0.077
- Positive fraction: 0.046

## What The Diagnostics Show

The read-level relationship is weak even for AlphaGenome. The lowest AlphaGenome score decile has a DiMeLo positive fraction of 0.047, while the highest AlphaGenome score decile has a positive fraction of 0.097. This weak separation explains why AUROC/AUPRC are low at read level.

The most likely interpretation is that AlphaGenome predicts a population/reference-like A549 H3K4me3 regulatory propensity, while single-read DiMeLo 6mA observations are sparse and noisy. The biological signal becomes visible after aggregating across reads and genomic bins, but individual read positions are not reliably predictable from sequence-only H3K4me3 propensity.
