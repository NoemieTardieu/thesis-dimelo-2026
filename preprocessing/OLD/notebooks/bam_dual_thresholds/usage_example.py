from bam_processing_dual_thresholds import EXAMPLE_BAM_PATHS, process_bam_dual_thresholds

# Thresholds requested by Noemie:
# C-channel (dataset mod code Z) >= 0.8  -> 204 on the ML 0-255 scale
# A-channel (dataset mod code Y) >= 0.9  -> 230 on the ML 0-255 scale

bam_path = EXAMPLE_BAM_PATHS["h3k27ac"]
chromosomes = ["chr1"]

df = process_bam_dual_thresholds(
    bam_path=bam_path,
    chromosomes=chromosomes,
    n_jobs=1,
    methyl_5mc_tr=204,
    methyl_6ma_tr=230,
    methyl_5mc_code="Z",
    methyl_6ma_code="Y",
    min_cpgs=1,
    min_adenines=0,
)

print(df.head())
print(df[[
    "read_name",
    "total_cpgs",
    "methylated_cpgs",
    "cpg_methylation_rate",
    "total_adenines",
    "methylated_adenines",
    "adenine_methylation_rate",
]].head())
