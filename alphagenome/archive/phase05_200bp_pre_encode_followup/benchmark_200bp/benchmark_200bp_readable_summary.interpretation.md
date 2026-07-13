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
| merged_c1 | pearson | 0.568 | 0.673 | 0.106 | HyenaDNA |
| merged_c1 | spearman | 0.362 | 0.520 | 0.158 | HyenaDNA |
| merged_c1 | auroc | 0.867 | 0.856 | -0.011 | AlphaGenome |
| merged_c1 | auprc | 0.622 | 0.557 | -0.066 | AlphaGenome |
| merged_e5b | pearson | 0.585 | 0.685 | 0.101 | HyenaDNA |
| merged_e5b | spearman | 0.347 | 0.570 | 0.223 | HyenaDNA |
| merged_e5b | auroc | 0.837 | 0.853 | 0.017 | HyenaDNA |
| merged_e5b | auprc | 0.645 | 0.571 | -0.074 | AlphaGenome |
| pooled | pearson | 0.581 | 0.693 | 0.112 | HyenaDNA |
| pooled | spearman | 0.375 | 0.573 | 0.198 | HyenaDNA |
| pooled | auroc | 0.873 | 0.865 | -0.008 | similar |
| pooled | auprc | 0.641 | 0.565 | -0.076 | AlphaGenome |

## Suggested Text To Send

The averaged-signal comparison shows that HyenaDNA and AlphaGenome capture related but not identical aspects of the DiMeLo-derived regulatory signal. HyenaDNA generally achieves higher Pearson and Spearman correlations, suggesting better agreement with the continuous averaged 6mA signal. AlphaGenome is competitive for AUROC and often stronger for AUPRC, suggesting good enrichment for the highest-signal bins. Overall, this supports that the HyenaDNA model has learned sequence-associated regulatory signal, while AlphaGenome provides a useful SoTA reference and complementary benchmark.

## Files

- `alphagenome_vs_hyenadna_primary_metrics.tsv`: full raw metrics with confidence intervals.
- `alphagenome_vs_hyenadna_readable_summary.deltas.tsv`: compact model-vs-model deltas.
- `alphagenome_vs_hyenadna_readable_summary.colored.html`: color-coded table for easy visual inspection.
