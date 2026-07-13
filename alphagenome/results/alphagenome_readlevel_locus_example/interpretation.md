# Same-Locus Read-Level Example

Selected locus: `chr19:35694000-35694200`.

This locus contains `19` reads and `625` read-position observations.

Across reads:

- AlphaGenome mean score: `69.797`
- AlphaGenome read-to-read SD: `15.755`
- DiMeLo 6mA mean: `0.351`
- DiMeLo 6mA read-to-read SD: `0.193`
- DiMeLo read-level range: `0.000` to `0.707`

Interpretation: this is an example where reads map to the same genomic locus and receive broadly similar AlphaGenome regulatory scores, but their observed DiMeLo 6mA signal varies substantially. This illustrates why read-level prediction is difficult: AlphaGenome captures a locus-level regulatory propensity, while individual read-level DiMeLo observations are variable/sparse. Aggregating across reads recovers the locus-level signal.
