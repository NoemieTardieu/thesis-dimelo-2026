# AlphaGenome vs HyenaDNA: Readable Interpretation

This summary compares both models on the same averaged/binned DiMeLo 6mA target.
Positive deltas mean HyenaDNA is higher; negative deltas mean AlphaGenome is higher.

## Main Takeaway

- HyenaDNA is usually better for continuous agreement with the averaged signal (`pearson`, `spearman`).
- AlphaGenome is often competitive or better for peak-like binary retrieval (`auprc`), especially in pooled/e5b summaries.
- AUROC is very similar between models; neither model dominates every metric.
- This means the two models are useful in slightly different ways: HyenaDNA tracks the experimental signal shape better, while AlphaGenome can be strong for identifying high-signal bins.

## Pooled Summary

| sample | metric | AlphaGenome | HyenaDNA | Hyena - Alpha | better |
| --- | ---: | ---: | ---: | ---: | --- |
| merged_c1 | pearson | 0.455 | 0.563 | 0.108 | HyenaDNA |
| merged_c1 | spearman | 0.300 | 0.419 | 0.119 | HyenaDNA |
| merged_c1 | auroc | 0.795 | 0.803 | 0.008 | similar |
| merged_c1 | auprc | 0.484 | 0.473 | -0.012 | AlphaGenome |
| merged_e5b | pearson | 0.454 | 0.558 | 0.104 | HyenaDNA |
| merged_e5b | spearman | 0.293 | 0.468 | 0.175 | HyenaDNA |
| merged_e5b | auroc | 0.749 | 0.791 | 0.042 | HyenaDNA |
| merged_e5b | auprc | 0.491 | 0.474 | -0.017 | AlphaGenome |
| pooled | pearson | 0.477 | 0.593 | 0.116 | HyenaDNA |
| pooled | spearman | 0.326 | 0.491 | 0.164 | HyenaDNA |
| pooled | auroc | 0.809 | 0.823 | 0.014 | HyenaDNA |
| pooled | auprc | 0.517 | 0.495 | -0.023 | AlphaGenome |

## Suggested Text To Send

The averaged-signal comparison shows that HyenaDNA and AlphaGenome capture related but not identical aspects of the DiMeLo-derived regulatory signal. HyenaDNA generally achieves higher Pearson and Spearman correlations, suggesting better agreement with the continuous averaged 6mA signal. AlphaGenome is competitive for AUROC and often stronger for AUPRC, suggesting good enrichment for the highest-signal bins. Overall, this supports that the HyenaDNA model has learned sequence-associated regulatory signal, while AlphaGenome provides a useful SoTA reference and complementary benchmark.

## Files

- `alphagenome_vs_hyenadna_primary_metrics.tsv`: full raw metrics with confidence intervals.
- `alphagenome_vs_hyenadna_readable_summary.deltas.tsv`: compact model-vs-model deltas.
- `alphagenome_vs_hyenadna_readable_summary.colored.html`: color-coded table for easy visual inspection.
