# Read-pair interpretability example: observed signal only

This analysis selected read pairs mapping to the same chr16 locus in merged_c1.

## Case 1: same locus, different 6mA

Two reads at the same reference locus showed very different observed 6mA signal:
one read had almost fully positive 6mA signal, while the other had essentially zero 6mA.
This illustrates strong read-to-read regulatory-signal heterogeneity at a fixed genomic position.

## Case 2: same locus, similar 6mA but different DNA

A second pair of reads at the same locus had very similar observed 6mA values, despite a large local DNA-sequence difference.
This shows that local read-level DNA differences do not necessarily imply different 6mA signal.

## Interpretation

Together, these examples support the variance analysis: read-level 6mA variation is not explained by DNA sequence alone in a simple deterministic way.
The next step is to overlay P(Reg|D,M) predictions from the methylation-conditioned HyenaDNA model to test whether adding 5mC context helps explain the observed regulatory-signal differences.
