import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pysam


DEFAULT_BAMS = {
    "h3k27ac": "/scratch/leuven/383/vsc38330/thesis_dimelo/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam",
    "h3k27me3": "/scratch/leuven/383/vsc38330/thesis_dimelo/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam",
    "h3k4me3": "/scratch/leuven/383/vsc38330/thesis_dimelo/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam",
}
DEFAULT_OUT_DIR = "/scratch/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase1_output"


def get_tag_safe(rec: pysam.AlignedSegment, candidates: List[str]):
    for tag in candidates:
        if rec.has_tag(tag):
            return rec.get_tag(tag)
    return None


def parse_mm_tag(mm_value: str) -> List[Tuple[str, str, List[int]]]:
    groups: List[Tuple[str, str, List[int]]] = []
    if not mm_value:
        return groups

    for grp in [g for g in mm_value.strip().split(";") if g]:
        parts = grp.split(",")
        head = parts[0]
        if len(head) < 2:
            continue

        base = head[0]
        mod_code = head[1:]
        deltas: List[int] = []
        for x in parts[1:]:
            x = x.strip()
            if not x:
                continue
            try:
                deltas.append(int(x))
            except ValueError:
                continue

        groups.append((base, mod_code, deltas))

    return groups


def parse_ml_tag(ml_value) -> List[int]:
    if ml_value is None:
        return []
    try:
        return [int(x) for x in ml_value]
    except Exception:
        return []


def decode_delta_positions(seq: str, base: str, deltas: List[int]) -> List[int]:
    base = base.upper()
    occurrences = [i for i, b in enumerate(seq) if b.upper() == base]
    if not occurrences:
        return []

    positions: List[int] = []
    j = -1
    for delta in deltas:
        j += delta + 1
        if j >= len(occurrences):
            break
        positions.append(occurrences[j])

    return positions


def extract_a_c_mods(rec: pysam.AlignedSegment) -> Dict[str, Tuple[List[int], List[int]]]:
    mm_value = get_tag_safe(rec, ["Mm", "MM"])
    ml_value = get_tag_safe(rec, ["Ml", "ML"])

    if mm_value is None:
        return {}

    mm_groups = parse_mm_tag(mm_value)
    ml_list = parse_ml_tag(ml_value)

    lengths = [len(g[2]) for g in mm_groups]
    total_mods = sum(lengths)
    if len(ml_list) < total_mods:
        ml_list = ml_list + [0] * (total_mods - len(ml_list))

    seq = rec.query_sequence
    if seq is None:
        return {}

    out: Dict[str, Tuple[List[int], List[int]]] = {}
    idx = 0

    for (base, _mod_code, deltas), length in zip(mm_groups, lengths):
        probs = ml_list[idx : idx + length]
        idx += length
        pos = decode_delta_positions(seq, base, deltas)

        m = min(len(pos), len(probs))
        pos = pos[:m]
        probs = probs[:m]

        base_upper = base.upper()
        if base_upper not in ("A", "C"):
            continue

        if base_upper not in out:
            out[base_upper] = (pos, probs)
        else:
            out[base_upper] = (out[base_upper][0] + pos, out[base_upper][1] + probs)

    return out


def bin_mods(
    read_len: int,
    positions: List[int],
    probs: List[int],
    bin_size: int,
    ml_threshold: int,
) -> np.ndarray:
    n_bins = int(np.ceil(read_len / bin_size))
    arr = np.zeros(n_bins, dtype=np.int32)

    for p, pr in zip(positions, probs):
        if pr >= ml_threshold:
            b = p // bin_size
            if 0 <= b < n_bins:
                arr[b] += 1

    return arr


def save_hist(data: np.ndarray, title: str, xlabel: str, out_png: str, bins: int = 80) -> None:
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_png: str,
) -> None:
    if len(x) > 50_000:
        idx = np.random.default_rng(0).choice(len(x), size=50_000, replace=False)
        x_plot = x[idx]
        y_plot = y[idx]
    else:
        x_plot = x
        y_plot = y

    plt.figure()
    plt.scatter(x_plot, y_plot, s=3, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def run_phase1_for_mark(
    mark: str,
    bam_path: str,
    out_dir: str,
    max_reads: int,
    min_mapq: int,
    ml_threshold: int,
    bin_size: int,
    save_binned_reads: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, mark)

    print(f"\\n[Phase 1] {mark}: reading BAM: {bam_path}")
    bam = pysam.AlignmentFile(bam_path, "rb")

    rows = []
    a_binned = []
    c_binned = []

    stats = {
        "scanned": 0,
        "kept": 0,
        "filtered_unmapped": 0,
        "filtered_low_mapq": 0,
        "filtered_no_seq": 0,
        "filtered_no_mod_tag": 0,
    }

    for rec in bam.fetch(until_eof=True):
        stats["scanned"] += 1

        if max_reads and stats["kept"] >= max_reads:
            break

        if rec.is_unmapped:
            stats["filtered_unmapped"] += 1
            continue
        if rec.mapping_quality < min_mapq:
            stats["filtered_low_mapq"] += 1
            continue
        if rec.query_sequence is None:
            stats["filtered_no_seq"] += 1
            continue

        mods = extract_a_c_mods(rec)
        if not mods:
            stats["filtered_no_mod_tag"] += 1
            continue

        read_len = rec.query_length if rec.query_length is not None else len(rec.query_sequence)

        a_pos, a_pr = mods.get("A", ([], []))
        c_pos, c_pr = mods.get("C", ([], []))

        a_total = len(a_pos)
        c_total = len(c_pos)
        a_conf = sum(1 for p in a_pr if p >= ml_threshold)
        c_conf = sum(1 for p in c_pr if p >= ml_threshold)

        a_density = (1000.0 * a_conf / read_len) if read_len > 0 else 0.0
        c_density = (1000.0 * c_conf / read_len) if read_len > 0 else 0.0

        rows.append(
            [
                rec.query_name,
                rec.reference_name,
                rec.reference_start,
                rec.reference_end,
                int(rec.is_reverse),
                rec.mapping_quality,
                read_len,
                a_total,
                a_conf,
                a_density,
                c_total,
                c_conf,
                c_density,
            ]
        )

        if len(a_binned) < save_binned_reads:
            a_binned.append(bin_mods(read_len, a_pos, a_pr, bin_size, ml_threshold))
            c_binned.append(bin_mods(read_len, c_pos, c_pr, bin_size, ml_threshold))

        stats["kept"] += 1
        if stats["kept"] % 20_000 == 0:
            print(f"  {mark}: kept {stats['kept']} reads...")

    bam.close()

    print(f"[Phase 1] {mark}: done reading")
    print(f"  scanned reads:       {stats['scanned']}")
    print(f"  kept reads:          {stats['kept']}")
    print(f"  filtered unmapped:   {stats['filtered_unmapped']}")
    print(f"  filtered low MAPQ:   {stats['filtered_low_mapq']}")
    print(f"  filtered no sequence:{stats['filtered_no_seq']}")
    print(f"  filtered no mod tags:{stats['filtered_no_mod_tag']}")

    tsv_path = f"{out_prefix}.per_read.tsv"
    header = [
        "read_id",
        "chrom",
        "ref_start",
        "ref_end",
        "is_reverse",
        "mapq",
        "read_len",
        "A_mod_total",
        "A_mod_conf",
        "A_mod_density_per_kb",
        "C_mod_total",
        "C_mod_conf",
        "C_mod_density_per_kb",
    ]
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")

    npz_path = f"{out_prefix}.binned_tracks.npz"
    np.savez_compressed(
        npz_path,
        bin_size=bin_size,
        ml_threshold=ml_threshold,
        A_tracks=np.array(a_binned, dtype=object),
        C_tracks=np.array(c_binned, dtype=object),
    )

    if not rows:
        print(f"[Phase 1] {mark}: no reads passed filters; wrote TSV/NPZ only.")
        return

    arr = np.array(rows, dtype=object)
    read_len = arr[:, 6].astype(int)
    mapq = arr[:, 5].astype(int)
    a_dens = arr[:, 9].astype(float)
    c_dens = arr[:, 12].astype(float)

    save_hist(
        read_len,
        f"{mark} read length distribution",
        "Read length (bp)",
        f"{out_prefix}.read_length.png",
    )
    save_hist(
        mapq,
        f"{mark} MAPQ distribution",
        "MAPQ",
        f"{out_prefix}.mapq.png",
        bins=50,
    )
    save_hist(
        a_dens,
        f"{mark} directed A-mod density per kb (Ml >= {ml_threshold})",
        "A-mod density (mods/kb)",
        f"{out_prefix}.A_density.png",
    )
    save_hist(
        c_dens,
        f"{mark} endogenous C-mod density per kb (Ml >= {ml_threshold})",
        "C-mod density (mods/kb)",
        f"{out_prefix}.C_density.png",
    )
    save_scatter(
        read_len.astype(float),
        a_dens,
        f"{mark} A-mod density vs read length",
        "Read length (bp)",
        "A-mod density (mods/kb)",
        f"{out_prefix}.A_density_vs_readlen.png",
    )
    save_scatter(
        read_len.astype(float),
        c_dens,
        f"{mark} C-mod density vs read length",
        "Read length (bp)",
        "C-mod density (mods/kb)",
        f"{out_prefix}.C_density_vs_readlen.png",
    )

    print(f"[Phase 1] {mark}: outputs written")
    print(f"  TSV:  {tsv_path}")
    print(f"  NPZ:  {npz_path}")
    print(f"  PNGs: {out_prefix}.*.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 DiMeLo QC/parsing for one or more histone-mark modBAM files."
    )
    parser.add_argument(
        "--marks",
        nargs="+",
        default=list(DEFAULT_BAMS.keys()),
        choices=list(DEFAULT_BAMS.keys()),
        help="Marks to process. Default: all available marks.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-reads", type=int, default=200_000)
    parser.add_argument("--min-mapq", type=int, default=20)
    parser.add_argument("--ml-threshold", type=int, default=128)
    parser.add_argument("--bin-size", type=int, default=50)
    parser.add_argument("--save-binned-reads", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for mark in args.marks:
        run_phase1_for_mark(
            mark=mark,
            bam_path=DEFAULT_BAMS[mark],
            out_dir=args.out_dir,
            max_reads=args.max_reads,
            min_mapq=args.min_mapq,
            ml_threshold=args.ml_threshold,
            bin_size=args.bin_size,
            save_binned_reads=args.save_binned_reads,
        )


if __name__ == "__main__":
    main()
