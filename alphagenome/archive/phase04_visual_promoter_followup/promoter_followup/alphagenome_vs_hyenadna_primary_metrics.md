# AlphaGenome vs HyenaDNA Averaged-Signal Metrics

Primary threshold: validation-derived top 10% pooled DiMeLo 6mA signal. Confidence intervals use region-level bootstrap replicates.

| model | track | experimental_sample | number_of_bins | pearson | spearman | auroc | auprc | positive_fraction | auprc_enrichment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AlphaGenome | A549_H3K4me3_fixed_mean | merged_c1 | 27993 | 0.4553 | 0.3000 | 0.7951 | 0.4843 | 0.1248 | 3.8804 |
| HyenaDNA | HyenaDNA | merged_c1 | 27993 | 0.5633 | 0.4190 | 0.8032 | 0.4728 | 0.1248 | 3.7879 |
| AlphaGenome | A549_H3K4me3_fixed_mean | merged_e5b | 25154 | 0.4540 | 0.2927 | 0.7488 | 0.4913 | 0.1529 | 3.2125 |
| HyenaDNA | HyenaDNA | merged_e5b | 25154 | 0.5576 | 0.4677 | 0.7906 | 0.4739 | 0.1529 | 3.0986 |
| AlphaGenome | A549_H3K4me3_fixed_mean | pooled | 31707 | 0.4767 | 0.3263 | 0.8087 | 0.5172 | 0.1228 | 4.2124 |
| HyenaDNA | HyenaDNA | pooled | 31707 | 0.5929 | 0.4906 | 0.8230 | 0.4946 | 0.1228 | 4.0287 |

