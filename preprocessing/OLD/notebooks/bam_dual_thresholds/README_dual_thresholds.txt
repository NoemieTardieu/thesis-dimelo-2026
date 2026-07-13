Dual-threshold BAM methylation processing

Files in this folder
- bam_processing_dual_thresholds.py
- usage_example.py
- README_dual_thresholds.txt

Goal
This folder contains a separate adaptation of the original BAM processing code for your ONT BAM files:
- /scratch/leuven/383/vsc38330/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam
- /scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam
- /scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam

The purpose of the adapted script is to extract per-read methylation information while using two separate thresholds:
- C / 5mC threshold = 0.8
- A / 6mA threshold = 0.9

Because BAM ML tags are stored on a 0-255 scale, these thresholds are converted to:
- 5mC threshold = 204
- 6mA threshold = 230

Why this adapted file was created
The original bam_processing.py mainly focuses on CpG methylation from C+m entries in the MM/ML tags and uses one threshold. Your use case is different because you want:
- both C-based methylation and A-based methylation
- different thresholds for each modification type
- your original file left unchanged

So this new file is a parallel version built specifically for your ONT BAM workflow.

High-level idea
The script reads aligned reads from a BAM file and, for each read:
1. checks basic filters such as mapping quality and SAM flags
2. reads the MM and ML tags from the BAM record
3. separates the ML values for C+m and A+a modifications
4. scans the sequence to assign methylated/unmethylated/unknown states
5. stores those calls in a tabular per-read format
6. returns a pandas DataFrame

What each important part does

1. Constants and BAM paths
The script defines:
- DEFAULT_5MC_PROB = 0.8
- DEFAULT_6MA_PROB = 0.9
- DEFAULT_5MC_ML = 204
- DEFAULT_6MA_ML = 230

It also includes EXAMPLE_BAM_PATHS with your three BAM files, so you can quickly switch between marks.

2. probability_to_ml(probability)
This helper converts a probability threshold in [0, 1] into the ML integer scale [0, 255].

Example:
- 0.8 -> 204
- 0.9 -> 230

Why this matters:
The ML tag in BAM stores values as bytes, not as floating-point probabilities.

3. detect_bam_data_type(bam_path)
This checks a sample of reads to see whether the BAM behaves like:
- ONT modified-base BAM: many reads have ML tags
- WGBS BAM: reads usually do not have ML tags

Why this matters:
ONT modified-base BAMs and WGBS BAMs need different decoding logic.

4. parse_mm_tag(mm_tag)
The MM tag describes which kinds of modifications are encoded in the read.
Examples include patterns like:
- C+m for methylated cytosine
- A+a for methylated adenine

This function parses the MM tag and records:
- which modification blocks are present
- the base type
- the modification code
- where each block begins in the ML array
- how many entries belong to that block

Why this matters:
The ML tag is one long array. To interpret it correctly, the code needs to know which segment belongs to C+m and which belongs to A+a.

5. get_modification_ml_values(...)
This function extracts only one modification family from ML values.
For example:
- all C+m values
- all A+a values

Why this matters:
You want different thresholds for C and A, so the two sets must be separated first.

6. scan_cpg_5mc(seq_bytes, ml_values, tr)
This scans the read sequence for CpG sites and assigns a state per CpG:
- 1 = methylated
- 0 = unmethylated
- 2 = unknown / missing

The threshold used here is your C / 5mC threshold.

Why CpG:
For 5mC analysis in this script, the code is targeting CpG positions specifically.

7. scan_adenines_6ma(seq_bytes, ml_values, tr)
This scans the read sequence for adenines and assigns a state per adenine:
- 1 = methylated
- 0 = unmethylated
- 2 = unknown / missing

The threshold used here is your A / 6mA threshold.

Why this is separate:
You wanted 6mA to use a stricter threshold than 5mC.

8. process_single_read_dual_thresholds(...)
This is the core per-read function.
For each read it:
- checks chromosome filters if provided
- checks flag filters and mapping quality
- reads MM and ML tags
- extracts C+m and A+a ML values separately
- scans the sequence for CpGs and adenines
- counts methylated and unmethylated calls
- returns one dictionary for that read

The output dictionary includes fields such as:
- read_name
- chromosome
- read_start / read_end
- seq
- cpg_positions
- cpg_states
- cpg_encoding
- total_cpgs
- methylated_cpgs
- unmethylated_cpgs
- cpg_methylation_rate
- adenine_positions
- adenine_states
- adenine_encoding
- total_adenines
- methylated_adenines
- unmethylated_adenines
- adenine_methylation_rate
- threshold_5mc_ml
- threshold_6ma_ml

Why this is useful:
It gives you a direct read-level table that you can analyze in pandas.

9. process_tabular_chunk_dual_thresholds(...)
This processes one genomic chunk.
It opens the BAM, fetches reads in that region, and applies process_single_read_dual_thresholds to each read.

Why chunking is useful:
Large BAM files are too big to process efficiently in one pass without splitting work.

10. process_bam_dual_thresholds(...)
This is the main entry point for the script.
It:
- detects whether the BAM is ONT or WGBS if needed
- reads chromosome lengths from the BAM header
- divides chromosomes into genomic chunks
- processes chunks in parallel using multiprocessing
- combines all outputs into one pandas DataFrame

This is the function you would usually call directly.

Example usage
See usage_example.py.
The key call is:

process_bam_dual_thresholds(
    bam_path=..., 
    chromosomes=["chr1"],
    n_jobs=1,
    methyl_5mc_tr=204,
    methyl_6ma_tr=230,
    min_cpgs=1,
    min_adenines=0,
)

Why these thresholds were chosen
You requested:
- C (5mC) threshold of 0.8
- A (6mA) threshold of 0.9

That means:
- a C+m call is treated as methylated only if ML >= 204
- an A+a call is treated as methylated only if ML >= 230

So 6mA is handled more conservatively than 5mC.

Important caveats
1. The script is a practical adaptation, not a perfect reimplementation of all MM-tag semantics.
The MM tag format is subtle. The current script separates ML blocks correctly at a high level, but it still uses simplified scanning logic when matching sequence positions to modification values.

2. CpG handling assumes the C+m information can be aligned through sequence scanning.
That is often reasonable for exploratory work, but if you need publication-grade exact modified-base decoding, we should validate carefully against a dedicated tool.

3. 6mA handling scans all adenines in the sequence.
Depending on how the MM tag was produced, the exact intended modified-base coordinate mapping may need stricter decoding.

4. This script gives read-level flexibility.
That is its main strength. But that flexibility comes with more implementation responsibility.

When this script is a good choice
Use this script when you want:
- custom per-read outputs in pandas
- separate thresholds for different modification types
- direct control over filtering and downstream analysis
- an easy way to prototype ideas in notebooks

When modkit may be better
modkit is often better when you want:
- robust, dedicated parsing of MM/ML modified-base tags
- a standard and well-tested workflow for ONT modified-base BAMs
- pileups, summaries, and extraction done with a tool built specifically for modified-base data
- higher confidence that modification coordinates are interpreted correctly

My recommendation
For your use case, modkit is probably the better primary tool if your main goal is correct and reproducible interpretation of ONT modified-base BAMs.

Why:
- it is purpose-built for MM/ML tags
- it reduces the risk of subtle decoding mistakes
- it is usually the safer choice for exact modified-base extraction

Why keep this Python script anyway:
- it is very useful for experimentation
- it is easier to customize than a command-line tool
- it can still be a great downstream analysis layer after modkit output is generated

Practical strategy I would recommend
Best workflow:
1. use modkit for the modified-base decoding step
2. export calls or summaries from modkit
3. use Python/pandas for your custom filtering, read-level summaries, plotting, and integration with the rest of your workflow

If you specifically need a pure-Python pipeline, then this script is a reasonable starting point, but I would strongly recommend validating a subset of reads against modkit before trusting all results.

Bottom line
- This adapted script was made to support your BAMs and your separate 5mC/6mA thresholds.
- It is good for custom exploratory analysis.
- For exact ONT modified-base decoding, modkit is likely the safer and better foundation.


The commands I have run: 

For h3k27ac:
cd /data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds

python3 - <<'PY'
from bam_processing_dual_thresholds import process_bam_dual_thresholds

bam_path = "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam"
chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

df = process_bam_dual_thresholds(
    bam_path=bam_path,
    chromosomes=chromosomes,
    n_jobs=4,
    data_type="ont",
    methyl_5mc_tr=204,
    methyl_6ma_tr=230,
    min_mapq=10,
    require_flags=0,
    exclude_flags=1796,
    min_cpgs=1,
    min_adenines=0,
)

out = "GSM6337767_h3k27ac_dual_thresholds.tsv.gz"
df.to_csv(out, sep="\t", index=False, compression="gzip")
print(f"saved {len(df)} rows to {out}")
PY



For h3k27me3:
cd /data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds

python3 - <<'PY'
from bam_processing_dual_thresholds import process_bam_dual_thresholds

bam_path = "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam"
chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

df = process_bam_dual_thresholds(
    bam_path=bam_path,
    chromosomes=chromosomes,
    n_jobs=4,
    data_type="ont",
    methyl_5mc_tr=204,
    methyl_6ma_tr=230,
    min_mapq=10,
    require_flags=0,
    exclude_flags=1796,
    min_cpgs=1,
    min_adenines=0,
)

out = "GSM6337768_h3k27me3_dual_thresholds.tsv.gz"
df.to_csv(out, sep="\t", index=False, compression="gzip")
print(f"saved {len(df)} rows to {out}")
PY




For h3k4me3:
cd /data/leuven/383/vsc38330/thesis_dimelo/notebooks/bam_dual_thresholds

python3 - <<'PY'
from bam_processing_dual_thresholds import process_bam_dual_thresholds

bam_path = "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam"
chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

df = process_bam_dual_thresholds(
    bam_path=bam_path,
    chromosomes=chromosomes,
    n_jobs=4,
    data_type="ont",
    methyl_5mc_tr=204,
    methyl_6ma_tr=230,
    min_mapq=10,
    require_flags=0,
    exclude_flags=1796,
    min_cpgs=1,
    min_adenines=0,
)

out = "GSM6337769_h3k4me3_dual_thresholds.tsv.gz"
df.to_csv(out, sep="\t", index=False, compression="gzip")
print(f"saved {len(df)} rows to {out}")
PY



The MODKIT path: 

For h3k27ac:
bam=/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam
outdir=/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/h3k27ac

mkdir -p "$outdir"

modkit modbam check-tags "$bam" --head 1000 > "$outdir/check_tags.txt" 2> "$outdir/check_tags.log"
modkit summary "$bam" --threads 4 > "$outdir/summary.tsv" 2> "$outdir/summary.log"
modkit modbam sample-probs "$bam" --threads 4 > "$outdir/sample_probs.tsv" 2> "$outdir/sample_probs.log"
modkit extract calls "$bam" "$outdir/extract_calls.tsv.gz" --threads 4 --filter-threshold C:0.8 --filter-threshold A:0.9 2> "$outdir/extract_calls.log"


For h3k27me3:
export PATH="/data/leuven/383/vsc38330/bin/dist_modkit_v0.6.1_481e3c9:$PATH"

bam=/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam
outdir=/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/h3k27me3

mkdir -p "$outdir"

modkit modbam check-tags "$bam" --head 1000 > "$outdir/check_tags.txt" 2> "$outdir/check_tags.log"
modkit extract calls "$bam" "$outdir/extract_calls.tsv" --threads 2 --filter-threshold C:0.8 --filter-threshold A:0.9 2> "$outdir/extract_calls.log"
gzip "$outdir/extract_calls.tsv"




For h3k4me3:
export PATH="/data/leuven/383/vsc38330/bin/dist_modkit_v0.6.1_481e3c9:$PATH"

bam=/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam
outdir=/data/leuven/383/vsc38330/thesis_dimelo/modkit_preproc/h3k4me3

mkdir -p "$outdir"

modkit modbam check-tags "$bam" --head 1000 > "$outdir/check_tags.txt" 2> "$outdir/check_tags.log"
modkit extract calls "$bam" "$outdir/extract_calls.tsv" --threads 2 --filter-threshold C:0.8 --filter-threshold A:0.9 2> "$outdir/extract_calls.log"
gzip "$outdir/extract_calls.tsv"




Optional next step: pileup
For CpG-focused 5mC bedMethyl:
modkit pileup "$bam" "$outdir/pileup_cpg_5mC.bed" \
  --threads 4 \
  --cpg \
  --filter-threshold C:0.8 \
  2> "$outdir/pileup_cpg_5mC.log"


what we have done:
Methodologically, this is already something you can explain to another bioinformatician:

verified modBAM tags with modkit modbam check-tags
extracted read-level modification calls with modkit extract calls
used explicit thresholds:
C:0.8
A:0.9


