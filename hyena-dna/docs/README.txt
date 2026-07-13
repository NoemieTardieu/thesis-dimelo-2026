chr16 merged_e5b preprocessing workspace
========================================

Purpose
-------
This folder is a small workspace for the first DiMeLo -> HyenaDNA tensorization
prototype.

Chosen sample:

  merged_e5b

Chosen chromosome:

  chr16

Why chr16:

  - It has a single complete chromosome-level extract-full file.
  - It has useful A-mod and CpG-mod heterogeneity in the existing bigWig tracks.
  - It is large enough to be meaningful but still manageable for a first
    prototype.
  - The central centromeric/no-signal gap is expected and should be avoided for
    interpretation. Read-level windows should come from actual covered regions.


Inputs
------
BAM:

  /staging/leuven/stg_00118/ONT_BAM_Noemie/bam_files/merged_e5b.sorted.bam

modkit extract full:

  /staging/leuven/stg_00118/BAM_Noemie/modkit_preproc/merged_e5b/by_chrom/extract_full_chr16.tsv.gz

Signal tracks used for chromosome choice:

  /staging/leuven/stg_00118/BAM_Noemie/pileup_tracks/merged_e5b/merged_e5b_A_a_all_percent.bw
  /staging/leuven/stg_00118/BAM_Noemie/pileup_tracks/merged_e5b/merged_e5b_C_combined_cpg_percent.bw


What preprocessing must do
--------------------------
modkit extract full is read-level, but it is not directly model-ready.

HyenaDNA needs dense fixed-length tensors:

  input_ids       [N, 32768]
  target_5mC      [N, 32768]
  mask_5mC        [N, 32768]
  target_6mA      [N, 32768]
  mask_6mA        [N, 32768]

where N is the number of reads/windows in the output chunk.

The tensorizer reads full DNA sequence from the BAM and stores it in the same forward-read orientation used by modkit. For reverse-strand BAM alignments, this means reverse-complementing the BAM query sequence before overlaying modkit probabilities by:

  read_id + forward_read_position


Prototype command
-----------------
Run from this folder:

  /data/leuven/383/vsc38330/.venv/bin/python make_chr16_dimelo_tensors.py \
    --max-reads 1000 \
    --out-prefix outputs/merged_e5b_chr16_first1000


Output files
------------
For prefix:

  outputs/merged_e5b_chr16_first1000

the script writes:

  outputs/merged_e5b_chr16_first1000.npz
  outputs/merged_e5b_chr16_first1000.metadata.tsv
  outputs/merged_e5b_chr16_first1000.summary.json


Debug Dataset/DataLoader command
--------------------------------
After making the first1000 tensor file, check that PyTorch can load batches and that a tiny two-head debug model can compute masked losses:

  /data/leuven/383/vsc38330/.venv/bin/python debug_chr16_dataset_and_tiny_model.py \
    --npz outputs/merged_e5b_chr16_first1000.npz \
    --batch-size 2 \
    --max-length 2048 \
    --device cpu

This is not the real HyenaDNA model. It only verifies the dataset interface, batch shapes, two output heads, and masked loss calculation.


HyenaDNA tiny two-head debug command
-----------------------------------
After the plain PyTorch Dataset/DataLoader debug works, test the same tensor file through the local HyenaDNA tiny-1k backbone plus two temporary prediction heads:

  /data/leuven/383/vsc38330/.venv/bin/python debug_chr16_hyenadna_tiny_two_head.py \
    --npz outputs/merged_e5b_chr16_first1000.npz \
    --batch-size 2 \
    --max-length 1024 \
    --device cpu

This verifies:

  input_ids -> HyenaDNA tiny hidden states -> 5mC head and 6mA head

It is still a debug forward pass, not real training.


HyenaDNA tiny two-head debug training
-------------------------------------
This is the first tiny training test. It freezes HyenaDNA tiny-1k and trains only two small per-position heads. This is still not the real 32k thesis model.

Quick CPU command:

  /data/leuven/383/vsc38330/.venv/bin/python train_chr16_hyenadna_tiny_two_head.py \
    --npz outputs/merged_e5b_chr16_first1000.npz \
    --batch-size 2 \
    --max-length 1024 \
    --epochs 2 \
    --max-train-batches 10 \
    --max-val-batches 5 \
    --device cpu \
    --out outputs/hyenadna_tiny_two_head_debug.pt

What this checks:

  - the .npz dataset can be split into train/validation batches
  - HyenaDNA tiny can be used as a frozen backbone
  - two methylation heads can be optimized
  - masked 5mC and 6mA losses can be tracked


HyenaDNA tiny checkpoint evaluation
-----------------------------------
After debug training, evaluate the saved heads on validation batches:

  /data/leuven/383/vsc38330/.venv/bin/python evaluate_chr16_hyenadna_tiny_two_head.py \
    --npz outputs/merged_e5b_chr16_first1000.npz \
    --checkpoint outputs/hyenadna_tiny_two_head_debug_3epochs_25batches.pt \
    --batch-size 2 \
    --max-length 1024 \
    --max-val-batches 25 \
    --device cpu

This reports masked BCE, MSE, MAE, mean prediction, and mean target separately for 5mC and 6mA on the validation split.


HyenaDNA small-32k smoke test
----------------------------
This is the first bridge toward the real 32 kb backbone. It is only a forward pass on one batch, not training.

The local checkpoints folder currently has tiny-1k. If small-32k is not already present, first run with --download:

  /data/leuven/383/vsc38330/.venv/bin/python debug_chr16_hyenadna_small32k_forward.py \
    --npz outputs/merged_e5b_chr16_first1000.npz \
    --batch-size 1 \
    --max-length 32768 \
    --device auto \
    --download

After the checkpoint exists locally, run without --download:

  /data/leuven/383/vsc38330/.venv/bin/python debug_chr16_hyenadna_small32k_forward.py \
    --npz outputs/merged_e5b_chr16_first1000.npz \
    --batch-size 1 \
    --max-length 32768 \
    --device auto

This checks:

  input_ids [1, 32768] -> HyenaDNA-small-32k hidden states -> temporary 5mC/6mA heads

Prefer running this on a GPU/compute node if possible. CPU may be slow.

If PyTorch sees an incompatible GPU or OpenMP cannot create enough threads, force CPU and limit thread creation:

  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  /data/leuven/383/vsc38330/.venv/bin/python debug_chr16_hyenadna_small32k_forward.py \
    --npz outputs/merged_e5b_chr16_first1000.npz \
    --batch-size 1 \
    --max-length 32768 \
    --device cpu


Build a better chr16 training dataset
-------------------------------------
The first1000 tensor file uses the first eligible chr16 reads and is mainly for
debugging. The next dataset should use intentionally selected chr16 regions with
coverage and heterogeneous A-mod/CpG signal, while avoiding the central chr16
gap.

Step 1: select candidate 100 kb regions.

  mkdir -p regions outputs

  /data/leuven/383/vsc38330/.venv/bin/python select_chr16_training_regions.py \
    --top-n 50 \
    --min-eligible-reads 20 \
    --out-prefix regions/merged_e5b_chr16_selected_100kb_top50

This writes:

  regions/merged_e5b_chr16_selected_100kb_top50.ranking.tsv
  regions/merged_e5b_chr16_selected_100kb_top50.bed

Step 2: inspect the selected regions.

  sed -n '1,15p' regions/merged_e5b_chr16_selected_100kb_top50.ranking.tsv

  cat regions/merged_e5b_chr16_selected_100kb_top50.bed

Step 3: tensorize reads overlapping those selected regions.

  /data/leuven/383/vsc38330/.venv/bin/python make_chr16_dimelo_tensors.py \
    --regions-bed regions/merged_e5b_chr16_selected_100kb_top50.bed \
    --max-reads 1000 \
    --out-prefix outputs/merged_e5b_chr16_selected_top50_first1000

For a larger selected-region tensor chunk later:

  /data/leuven/383/vsc38330/.venv/bin/python make_chr16_dimelo_tensors.py \
    --regions-bed regions/merged_e5b_chr16_selected_100kb_top50.bed \
    --max-reads 5000 \
    --out-prefix outputs/merged_e5b_chr16_selected_top50_first5000

Step 4: check the selected-region tensor file can still load as a dataset.

  /data/leuven/383/vsc38330/.venv/bin/python debug_chr16_dataset_and_tiny_model.py \
    --npz outputs/merged_e5b_chr16_selected_top50_first1000.npz \
    --batch-size 2 \
    --max-length 2048 \
    --device cpu

Evenly sampled selected-region dataset
--------------------------------------
To avoid filling the dataset from only the first few high-coverage regions, cap
the number of reads collected per selected region.

Option A: 20 reads per region across 50 regions, about 1000 reads total.

  /data/leuven/383/vsc38330/.venv/bin/python make_chr16_dimelo_tensors.py \
    --regions-bed regions/merged_e5b_chr16_selected_100kb_top50.bed \
    --max-reads-per-region 20 \
    --max-reads 1000 \
    --out-prefix outputs/merged_e5b_chr16_selected_top50_even20

Check it:

  /data/leuven/383/vsc38330/.venv/bin/python debug_chr16_dataset_and_tiny_model.py \
    --npz outputs/merged_e5b_chr16_selected_top50_even20.npz \
    --batch-size 2 \
    --max-length 2048 \
    --device cpu

Option B: 50 reads per region across 50 regions, about 2500 reads total.

  /data/leuven/383/vsc38330/.venv/bin/python make_chr16_dimelo_tensors.py \
    --regions-bed regions/merged_e5b_chr16_selected_100kb_top50.bed \
    --max-reads-per-region 50 \
    --max-reads 2500 \
    --out-prefix outputs/merged_e5b_chr16_selected_top50_even50

Check it:

  /data/leuven/383/vsc38330/.venv/bin/python debug_chr16_dataset_and_tiny_model.py \
    --npz outputs/merged_e5b_chr16_selected_top50_even50.npz \
    --batch-size 2 \
    --max-length 2048 \
    --device cpu


Tensor conventions
------------------
input_ids:

  HyenaDNA character-token IDs:

    [PAD] = 4
    [UNK] = 6
    A = 7
    C = 8
    G = 9
    T = 10
    N = 11

target_5mC:

  C:m modkit probability at CpG cytosines.
  Invalid/unobserved positions are set to -100.

mask_5mC:

  1 where target_5mC is valid.
  0 otherwise.

target_6mA:

  A:a modkit probability at adenines.
  Invalid/unobserved positions are set to -100.

mask_6mA:

  1 where target_6mA is valid.
  0 otherwise.


Important limitations of this first prototype
---------------------------------------------
1. It uses C:m as the 5mC target and ignores C:h.

   Later we can decide whether to:

     - keep 5mC and 5hmC separate
     - combine C:m and C:h
     - use only C:m for the main 5mC task

2. It only keeps reads with read_length <= 32768.

   Later, reads longer than 32k can be split/cropped into 32k windows.

3. It starts with the first eligible reads on chr16.

   Later, read/window selection should avoid the centromeric gap and can sample
   across high-heterogeneity regions.

4. This is a tensorization prototype, not the final training dataset.
