# ChIP-seq versus DiMeLo Follow-Up Diagnostics

This downstream analysis investigates why the external ENCODE H3K4me3 track and aggregated DiMeLo 6mA show moderate bin-level correlation.

## Main ENCODE-vs-DiMeLo Result

- Minimum DiMeLo coverage: `5` valid observations per bin.
- Shared bins: `20376`.
- Pearson: `0.5336`.
- Spearman: `0.3214`.

## Peak-Proxy Enrichment

Using the top 10% of ENCODE H3K4me3 bins as a peak-like proxy:

- DiMeLo mean in high-ENCODE bins: `0.2274`.
- DiMeLo mean in other bins: `0.0808`.
- DiMeLo fold enrichment: `2.8134`.
- DiMeLo AUROC for high-ENCODE bins: `0.8835`.
- DiMeLo AUPRC for high-ENCODE bins: `0.5886`.
- Random AUPRC baseline: `0.1000`.

## Chromosome Consistency

| Chromosome | n bins | Pearson | Spearman |
| --- | ---: | ---: | ---: |
| chr11 | 5040 | 0.5269 | 0.2848 |
| chr16 | 6086 | 0.5452 | 0.3594 |
| chr17 | 3989 | 0.5572 | 0.3223 |
| chr19 | 5261 | 0.5156 | 0.3308 |

## Adenine Density

Adenine-density stratification was calculated from the supplied reference FASTA.

## Output Tables

- `coverage_sensitivity.tsv`
- `chromosome_metrics.tsv`
- `region_metrics_external_dimelo.tsv`
- `stratified_external_dimelo_metrics.tsv`
- `external_peak_proxy_enrichment.tsv`
- `adenine_density_external_dimelo.tsv`

Interpretation note: these are diagnostics for the moderate ENCODE-vs-DiMeLo agreement. The peak-proxy enrichment/AUROC can be stronger than Pearson correlation because enrichment asks whether DiMeLo is higher in ChIP-seq-like regions, whereas correlation asks whether exact bin-level amplitudes match.
