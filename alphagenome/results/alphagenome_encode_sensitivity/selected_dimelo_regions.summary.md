# AlphaGenome-ENCODE Sensitivity Check

This analysis re-checks AlphaGenome A549 H3K4me3 versus ENCODE A549 H3K4me3 outside the main DiMeLo/HyenaDNA population benchmark.

- Regions: `metadata/regions/4chrom_test_regions.tsv`
- Cache directory: `logs/cache/cache`
- External BigWig: `server_artifacts/external_tracks/ENCSR203XPU/ENCFF074PND_ENCSR203XPU_A549_H3K4me3_fold_change_GRCh38.bigWig`

| label | bin size | n bins | Pearson | Spearman | normalized MAE | normalized RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected_dimelo_regions_128bp | 128 | 46800 | 0.9480 | 0.6385 | 0.0199 | 0.0490 |
| selected_dimelo_regions_200bp | 200 | 30000 | 0.9499 | 0.6489 | 0.0196 | 0.0486 |

Interpretation note: if the random-region correlation remains near the selected-region value, the high AlphaGenome-ENCODE agreement is likely driven by matched/in-distribution A549 H3K4me3 signal rather than only DiMeLo-region selection. If it drops substantially, the original 0.95 was likely inflated by selected regions and/or 200 bp smoothing.
