"""Helpers for extracting ONT methylation from BAMs with separate thresholds for 5mC and 6mA.

This variant is based on the original notebook copy but keeps it untouched.
It is geared toward your ONT `mod_mappings.sorted.bam` files and lets you use
independent thresholds for C+m (5mC) and A+a (6mA).
"""

import re
from multiprocessing import Pool

import numba as nb
import numpy as np
import pandas as pd
import pysam


CG_ASCII = np.array([ord("C"), ord("G")], dtype=np.uint8)
A_ASCII = ord("A")


DEFAULT_5MC_PROB = 0.8
DEFAULT_6MA_PROB = 0.9
DEFAULT_5MC_ML = int(round(DEFAULT_5MC_PROB * 255))
DEFAULT_6MA_ML = int(round(DEFAULT_6MA_PROB * 255))
DEFAULT_5MC_MOD_CODE = "Z"
DEFAULT_6MA_MOD_CODE = "Y"


EXAMPLE_BAM_PATHS = {
    "h3k27ac": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam",
    "h3k27me3": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam",
    "h3k4me3": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam",
}


def get_tag_safe(read, candidates):
    """Return the first available BAM tag from a list of candidate names."""
    for tag in candidates:
        if read.has_tag(tag):
            return read.get_tag(tag)
    return None


def probability_to_ml(probability):
    """Convert a probability in [0, 1] to the BAM ML scale [0, 255]."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    return int(round(probability * 255))


def detect_bam_data_type(bam_path, sample_size=1000):
    """Detect if BAM file is ONT (has ML tags) or WGBS (no ML tags)."""
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        reads_checked = 0
        has_ml_tag = 0

        for read in bam:
            if read.is_unmapped:
                continue

            if get_tag_safe(read, ["Ml", "ML"]) is not None:
                has_ml_tag += 1

            reads_checked += 1
            if reads_checked >= sample_size:
                break

        if reads_checked == 0:
            raise ValueError("No mapped reads found in BAM file")

        return "ont" if has_ml_tag / reads_checked > 0.5 else "wgbs"


def parse_mm_tag(mm_tag):
    """Parse MM tag and keep offsets/counts for each modification block."""
    if not mm_tag:
        return None

    modifications = []
    cumulative_count = 0

    for idx, group in enumerate(mm_tag.split(";")):
        if not group.strip():
            continue

        parts = group.split(",")
        mod_info = parts[0]
        match = re.match(r"([ACGT])\+([A-Za-z\?]+)", mod_info)
        if not match:
            continue

        base = match.group(1)
        mod_code = match.group(2)
        skip_positions = [int(x) for x in parts[1:] if x.strip()]
        mod_count = len(skip_positions)

        modifications.append(
            {
                "group_index": idx,
                "base": base,
                "mod_code": mod_code,
                "skip_positions": skip_positions,
                "ml_offset": cumulative_count,
                "count": mod_count,
            }
        )
        cumulative_count += mod_count

    return {"modifications": modifications, "total_mods": cumulative_count}


def get_modification_ml_values(ml_array, mm_info, base, mod_code_prefix):
    """Return ML values for one modification family, e.g. C+m or A+a."""
    if mm_info is None:
        return np.array([], dtype=np.uint8)

    for mod in mm_info["modifications"]:
        if mod["base"] == base and mod["mod_code"].startswith(mod_code_prefix):
            start = mod["ml_offset"]
            end = min(start + mod["count"], len(ml_array))
            return np.array(ml_array[start:end], dtype=np.uint8)

    return np.array([], dtype=np.uint8)


@nb.njit(fastmath=True, cache=True)
def scan_cpg_5mc(seq_bytes, ml_values, tr):
    """Return CpG offsets and 5mC states using C+m ML values."""
    n = len(seq_bytes) - 1
    pos_buf = []
    state_buf = []
    cg_idx = 0

    for i in range(n):
        if seq_bytes[i] == CG_ASCII[0] and seq_bytes[i + 1] == CG_ASCII[1]:
            pos_buf.append(i)
            if cg_idx < len(ml_values):
                state_buf.append(1 if ml_values[cg_idx] >= tr else 0)
            else:
                state_buf.append(2)
            cg_idx += 1

    return pos_buf, state_buf


@nb.njit(fastmath=True, cache=True)
def scan_adenines_6ma(seq_bytes, ml_values, tr):
    """Return adenine offsets and 6mA states using A+a ML values."""
    pos_buf = []
    state_buf = []
    a_idx = 0

    for i in range(len(seq_bytes)):
        if seq_bytes[i] == A_ASCII:
            pos_buf.append(i)
            if a_idx < len(ml_values):
                state_buf.append(1 if ml_values[a_idx] >= tr else 0)
            else:
                state_buf.append(2)
            a_idx += 1

    return pos_buf, state_buf


@nb.njit(fastmath=True, cache=True)
def cpg_scan_wgbs_forward(ref_bytes, read_bytes):
    n = len(ref_bytes) - 1
    pos_buf = []
    state_buf = []

    for i in range(n):
        if ref_bytes[i] == ord("C") and ref_bytes[i + 1] == ord("G"):
            pos_buf.append(i)
            if i < len(read_bytes):
                if read_bytes[i] == ord("C"):
                    state_buf.append(1)
                elif read_bytes[i] == ord("T"):
                    state_buf.append(0)
                else:
                    state_buf.append(2)
            else:
                state_buf.append(2)

    return pos_buf, state_buf


@nb.njit(fastmath=True, cache=True)
def cpg_scan_wgbs_reverse(ref_bytes, read_bytes):
    n = len(ref_bytes) - 1
    pos_buf = []
    state_buf = []

    for i in range(n):
        if ref_bytes[i] == ord("C") and ref_bytes[i + 1] == ord("G"):
            pos_buf.append(i)
            if i + 1 < len(read_bytes):
                if read_bytes[i + 1] == ord("G"):
                    state_buf.append(1)
                elif read_bytes[i + 1] == ord("A"):
                    state_buf.append(0)
                else:
                    state_buf.append(2)
            else:
                state_buf.append(2)

    return pos_buf, state_buf


def clean_cigar_sequence(read):
    """Align read sequence to reference coordinates using the CIGAR string."""
    if read.cigartuples is None:
        return read.query_alignment_sequence

    seq = read.query_sequence
    result = []
    seq_pos = 0

    for op, length in read.cigartuples:
        if op == 0:
            result.append(seq[seq_pos : seq_pos + length])
            seq_pos += length
        elif op == 1:
            seq_pos += length
        elif op == 2:
            result.append("N" * length)
        elif op == 3:
            result.append("N" * length)
        elif op == 4:
            seq_pos += length
        elif op == 5:
            pass
        elif op == 7:
            result.append(seq[seq_pos : seq_pos + length])
            seq_pos += length
        elif op == 8:
            result.append(seq[seq_pos : seq_pos + length])
            seq_pos += length

    return "".join(result)


def process_single_read_dual_thresholds(
    read,
    data_type,
    methyl_5mc_tr=DEFAULT_5MC_ML,
    methyl_6ma_tr=DEFAULT_6MA_ML,
    methyl_5mc_code=DEFAULT_5MC_MOD_CODE,
    methyl_6ma_code=DEFAULT_6MA_MOD_CODE,
    ref_fasta=None,
    interesting_chromosomes=None,
    min_mapq=10,
    require_flags=3,
    exclude_flags=1796,
    min_cpgs=1,
    min_adenines=0,
):
    """Process one read and return a single combined dict with 5mC and 6mA calls."""
    if interesting_chromosomes and read.reference_name not in interesting_chromosomes:
        return []
    if require_flags and (read.flag & require_flags) != require_flags:
        return []
    if exclude_flags and (read.flag & exclude_flags):
        return []
    if read.mapping_quality < min_mapq:
        return []

    chrom = read.reference_name
    r_beg, r_end = read.reference_start, read.reference_end

    if data_type == "ont":
        ml_values_full = get_tag_safe(read, ["Ml", "ML"])
        mm_tag = get_tag_safe(read, ["Mm", "MM"])
        if ml_values_full is None or mm_tag is None:
            return []

        mm_info = parse_mm_tag(mm_tag)
        seq = read.get_forward_sequence()
        read_length = len(seq)
        seq_bytes = seq.encode()

        c_ml_values = get_modification_ml_values(
            ml_values_full, mm_info, "C", methyl_5mc_code
        )
        a_ml_values = get_modification_ml_values(
            ml_values_full, mm_info, "A", methyl_6ma_code
        )

        cpg_positions_in_read, cpg_states = scan_cpg_5mc(
            seq_bytes, c_ml_values, methyl_5mc_tr
        )
        adenine_positions_in_read, adenine_states = scan_adenines_6ma(
            seq_bytes, a_ml_values, methyl_6ma_tr
        )

        total_cpgs = len(cpg_positions_in_read)
        total_adenines = len(adenine_positions_in_read)
        if total_cpgs < min_cpgs or total_adenines < min_adenines:
            return []

        cpg_positions = [r_beg + off for off in cpg_positions_in_read]
        adenine_positions = [r_beg + off for off in adenine_positions_in_read]

        cpg_encoding = ["2" for _ in range(read_length)]
        for x, y in zip(cpg_positions_in_read, cpg_states):
            cpg_encoding[x] = str(y)

        adenine_encoding = ["2" for _ in range(read_length)]
        for x, y in zip(adenine_positions_in_read, adenine_states):
            adenine_encoding[x] = str(y)

        methylated_cpgs = sum(1 for s in cpg_states if s == 1)
        unmethylated_cpgs = sum(1 for s in cpg_states if s == 0)
        methylated_adenines = sum(1 for s in adenine_states if s == 1)
        unmethylated_adenines = sum(1 for s in adenine_states if s == 0)

        read_data = {
            "read_name": read.query_name,
            "chromosome": chrom,
            "read_start": r_beg,
            "read_end": r_end,
            "read_length": r_end - r_beg,
            "seq": seq,
            "cpg_positions": cpg_positions,
            "cpg_states": list(cpg_states),
            "cpg_encoding": "".join(cpg_encoding),
            "total_cpgs": total_cpgs,
            "methylated_cpgs": methylated_cpgs,
            "unmethylated_cpgs": unmethylated_cpgs,
            "cpg_methylation_rate": methylated_cpgs / total_cpgs if total_cpgs else 0.0,
            "adenine_positions": adenine_positions,
            "adenine_states": list(adenine_states),
            "adenine_encoding": "".join(adenine_encoding),
            "total_adenines": total_adenines,
            "methylated_adenines": methylated_adenines,
            "unmethylated_adenines": unmethylated_adenines,
            "adenine_methylation_rate": (
                methylated_adenines / total_adenines if total_adenines else 0.0
            ),
            "threshold_5mc_ml": methyl_5mc_tr,
            "threshold_6ma_ml": methyl_6ma_tr,
            "threshold_5mc_prob": methyl_5mc_tr / 255,
            "threshold_6ma_prob": methyl_6ma_tr / 255,
            "mapping_quality": read.mapping_quality,
            "is_reverse": read.is_reverse,
            "data_type": data_type,
        }
        return [read_data]

    if ref_fasta is None:
        raise ValueError("Reference genome (ref_fasta) required for WGBS data")

    try:
        ref_seq = ref_fasta.fetch(chrom, r_beg, r_end + 1).upper()
        if ref_seq[-2:] != "CG":
            ref_seq = ref_seq[:-1]
    except Exception:
        return []

    read_seq = clean_cigar_sequence(read)
    if read_seq is None:
        return []

    if read.is_reverse:
        cpg_positions_in_read, cpg_states = cpg_scan_wgbs_reverse(
            ref_seq.encode(), read_seq.encode()
        )
    else:
        cpg_positions_in_read, cpg_states = cpg_scan_wgbs_forward(
            ref_seq.encode(), read_seq.encode()
        )

    total_cpgs = len(cpg_positions_in_read)
    if total_cpgs < min_cpgs:
        return []

    cpg_positions = [r_beg + off for off in cpg_positions_in_read]
    cpg_encoding = ["2" for _ in range(len(ref_seq))]
    for x, y in zip(cpg_positions_in_read, cpg_states):
        if x < len(cpg_encoding):
            cpg_encoding[x] = str(y)

    methylated_cpgs = sum(1 for s in cpg_states if s == 1)
    unmethylated_cpgs = sum(1 for s in cpg_states if s == 0)

    return [
        {
            "read_name": read.query_name,
            "chromosome": chrom,
            "read_start": r_beg,
            "read_end": r_end,
            "read_length": r_end - r_beg,
            "seq": ref_seq,
            "cpg_positions": cpg_positions,
            "cpg_states": list(cpg_states),
            "cpg_encoding": "".join(cpg_encoding),
            "total_cpgs": total_cpgs,
            "methylated_cpgs": methylated_cpgs,
            "unmethylated_cpgs": unmethylated_cpgs,
            "cpg_methylation_rate": methylated_cpgs / total_cpgs if total_cpgs else 0.0,
            "adenine_positions": [],
            "adenine_states": [],
            "adenine_encoding": "",
            "total_adenines": 0,
            "methylated_adenines": 0,
            "unmethylated_adenines": 0,
            "adenine_methylation_rate": 0.0,
            "threshold_5mc_ml": methyl_5mc_tr,
            "threshold_6ma_ml": methyl_6ma_tr,
            "threshold_5mc_prob": methyl_5mc_tr / 255,
            "threshold_6ma_prob": methyl_6ma_tr / 255,
            "mapping_quality": read.mapping_quality,
            "is_reverse": read.is_reverse,
            "data_type": data_type,
        }
    ]


def process_tabular_chunk_dual_thresholds(args):
    (
        chrom,
        chunk_start,
        chunk_end,
        bam_path,
        interesting_chromosomes,
        methyl_5mc_tr,
        methyl_6ma_tr,
        methyl_5mc_code,
        methyl_6ma_code,
        data_type,
        reference_path,
        min_mapq,
        require_flags,
        exclude_flags,
        min_cpgs,
        min_adenines,
    ) = args

    tabular_data = []
    ref_fasta = None
    if data_type == "wgbs":
        if reference_path is None:
            raise ValueError("Reference genome path required for WGBS data")
        ref_fasta = pysam.FastaFile(reference_path)

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(chrom, chunk_start, chunk_end):
            if read.reference_start < chunk_start or read.reference_start >= chunk_end:
                continue
            tabular_data.extend(
                process_single_read_dual_thresholds(
                    read=read,
                    data_type=data_type,
                    methyl_5mc_tr=methyl_5mc_tr,
                    methyl_6ma_tr=methyl_6ma_tr,
                    methyl_5mc_code=methyl_5mc_code,
                    methyl_6ma_code=methyl_6ma_code,
                    ref_fasta=ref_fasta,
                    interesting_chromosomes=interesting_chromosomes,
                    min_mapq=min_mapq,
                    require_flags=require_flags,
                    exclude_flags=exclude_flags,
                    min_cpgs=min_cpgs,
                    min_adenines=min_adenines,
                )
            )

    if ref_fasta is not None:
        ref_fasta.close()

    return tabular_data


def process_bam_dual_thresholds(
    bam_path,
    chromosomes,
    n_jobs=4,
    chunk_size_genomic=1_000_000,
    methyl_5mc_tr=DEFAULT_5MC_ML,
    methyl_6ma_tr=DEFAULT_6MA_ML,
    methyl_5mc_code=DEFAULT_5MC_MOD_CODE,
    methyl_6ma_code=DEFAULT_6MA_MOD_CODE,
    reference_path=None,
    data_type=None,
    min_mapq=10,
    require_flags=3,
    exclude_flags=1796,
    min_cpgs=1,
    min_adenines=0,
):
    """Process BAM with separate thresholds for 5mC and 6mA calls."""
    if data_type is None:
        print("Auto-detecting data type...")
        data_type = detect_bam_data_type(bam_path)
        print(f"Detected data type: {data_type.upper()}")

    if data_type == "wgbs" and reference_path is None:
        raise ValueError("Reference genome path is required for WGBS data")

    print("Reading BAM file...")
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        chr_lengths = {ref: length for ref, length in zip(bam.references, bam.lengths)}

    tasks = []
    for chrom in chromosomes:
        if chrom not in chr_lengths:
            print(f"Warning: {chrom} not found in BAM file")
            continue
        chr_len = chr_lengths[chrom]
        for start in range(0, chr_len, chunk_size_genomic):
            end = min(start + chunk_size_genomic, chr_len)
            tasks.append(
                (
                    chrom,
                    start,
                    end,
                    bam_path,
                    chromosomes,
                    methyl_5mc_tr,
                    methyl_6ma_tr,
                    methyl_5mc_code,
                    methyl_6ma_code,
                    data_type,
                    reference_path,
                    min_mapq,
                    require_flags,
                    exclude_flags,
                    min_cpgs,
                    min_adenines,
                )
            )

    print(
        f"Processing {len(tasks)} chunks with {n_jobs} workers "
        f"(5mC={methyl_5mc_tr}/255, 6mA={methyl_6ma_tr}/255)..."
    )

    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            results = pool.map(process_tabular_chunk_dual_thresholds, tasks)
    else:
        results = [process_tabular_chunk_dual_thresholds(task) for task in tasks]

    all_data = []
    for result in results:
        all_data.extend(result)

    print(f"Collected {len(all_data)} reads")
    return pd.DataFrame(all_data)
