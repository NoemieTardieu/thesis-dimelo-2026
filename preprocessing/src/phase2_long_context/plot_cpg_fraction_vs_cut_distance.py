import argparse
import csv
import os
import re
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pysam


DEFAULT_BAMS = {
    "h3k27ac": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam",
    "h3k27me3": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam",
    "h3k4me3": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam",
}
DEFAULT_OUT_DIR = (
    "/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/"
    "phase2_long_context/cpg_fraction_vs_cut_distance"
)
MARK_COLORS = {
    "h3k27ac": "#d55e00",
    "h3k27me3": "#cc79a7",
    "h3k4me3": "#009e73",
}
CPG_PATTERN = re.compile("(?=CG)")


def get_tag_safe(rec: pysam.AlignedSegment, candidates: Sequence[str]):
    for tag in candidates:
        if rec.has_tag(tag):
            return rec.get_tag(tag)
    return None


def parse_mm_tag(mm_value: str) -> List[Tuple[str, str, List[int]]]:
    out: List[Tuple[str, str, List[int]]] = []
    if not mm_value:
        return out
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
        out.append((base, mod_code, deltas))
    return out


def parse_ml_tag(ml_value) -> List[int]:
    if ml_value is None:
        return []
    try:
        return [int(x) for x in ml_value]
    except Exception:
        return []


def decode_delta_positions(seq: str, base: str, deltas: List[int]) -> List[int]:
    occ = [i for i, b in enumerate(seq) if b.upper() == base.upper()]
    if not occ:
        return []
    out = []
    j = -1
    for d in deltas:
        j += d + 1
        if j >= len(occ):
            break
        out.append(occ[j])
    return out


def extract_mod_qpos(rec: pysam.AlignedSegment, base_keep: str) -> Tuple[List[int], List[int]]:
    mm_value = get_tag_safe(rec, ["Mm", "MM"])
    ml_value = get_tag_safe(rec, ["Ml", "ML"])
    if mm_value is None or rec.query_sequence is None:
        return [], []

    mm_groups = parse_mm_tag(mm_value)
    ml_list = parse_ml_tag(ml_value)
    lengths = [len(g[2]) for g in mm_groups]
    total_mods = sum(lengths)
    if len(ml_list) < total_mods:
        ml_list = ml_list + [0] * (total_mods - len(ml_list))

    seq = rec.query_sequence
    idx = 0
    out_pos: List[int] = []
    out_prob: List[int] = []

    for (base, _mod_code, deltas), L in zip(mm_groups, lengths):
        probs = ml_list[idx: idx + L]
        idx += L
        pos = decode_delta_positions(seq, base, deltas)
        m = min(len(pos), len(probs))
        if base.upper() == base_keep.upper():
            out_pos.extend(pos[:m])
            out_prob.extend(probs[:m])

    return out_pos, out_prob


def smooth_fraction(numer: np.ndarray, denom: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        out = np.full(numer.shape, np.nan, dtype=np.float64)
        mask = denom > 0
        out[mask] = numer[mask] / denom[mask]
        return out
    kernel = np.ones(window, dtype=np.float64)
    s_num = np.convolve(numer.astype(np.float64), kernel, mode="same")
    s_den = np.convolve(denom.astype(np.float64), kernel, mode="same")
    out = np.full(numer.shape, np.nan, dtype=np.float64)
    mask = s_den > 0
    out[mask] = s_num[mask] / s_den[mask]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot fraction of CpG methylation as a function of distance from the cut-site end."
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--marks", nargs="+", default=list(DEFAULT_BAMS.keys()))
    p.add_argument("--max-reads", type=int, default=200000, help="Reads per mark; 0 means all.")
    p.add_argument("--min-mapq", type=int, default=20)
    p.add_argument("--ml-threshold", type=int, default=128)
    p.add_argument("--max-distance", type=int, default=550)
    p.add_argument("--smooth-window", type=int, default=9)
    p.add_argument("--progress-every", type=int, default=50000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    profiles: Dict[str, Dict[str, np.ndarray | int]] = {}

    for mark in args.marks:
        if mark not in DEFAULT_BAMS:
            raise SystemExit(f"Unknown mark: {mark}")
        bam_path = DEFAULT_BAMS[mark]
        cov = np.zeros(args.max_distance + 1, dtype=np.int64)
        meth = np.zeros(args.max_distance + 1, dtype=np.int64)

        stats = {
            "scanned": 0,
            "kept": 0,
            "filtered_unmapped": 0,
            "filtered_low_mapq": 0,
            "filtered_secondary": 0,
            "filtered_no_seq": 0,
        }

        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for rec in bam.fetch(until_eof=True):
                stats["scanned"] += 1
                if args.max_reads and stats["kept"] >= args.max_reads:
                    break
                if rec.is_unmapped:
                    stats["filtered_unmapped"] += 1
                    continue
                if rec.is_secondary or rec.is_supplementary:
                    stats["filtered_secondary"] += 1
                    continue
                if rec.mapping_quality < args.min_mapq:
                    stats["filtered_low_mapq"] += 1
                    continue
                seq = rec.query_sequence
                if seq is None:
                    stats["filtered_no_seq"] += 1
                    continue

                seq = seq.upper()
                read_len = len(seq)
                cpg_positions = np.fromiter((m.start() for m in CPG_PATTERN.finditer(seq)), dtype=np.int32)
                if cpg_positions.size == 0:
                    stats["kept"] += 1
                    continue
                mod_pos, mod_prob = extract_mod_qpos(rec, "C")
                confident_mod_cpg = np.fromiter((
                    pos for pos, pr in zip(mod_pos, mod_prob)
                    if pr >= args.ml_threshold and pos + 1 < read_len and seq[pos] == "C" and seq[pos + 1] == "G"
                ), dtype=np.int32)

                if rec.is_reverse:
                    dists = read_len - 1 - cpg_positions
                else:
                    dists = cpg_positions
                dists = dists[dists <= args.max_distance]
                if dists.size > 0:
                    cov[: args.max_distance + 1] += np.bincount(dists, minlength=args.max_distance + 1)

                if confident_mod_cpg.size > 0:
                    if rec.is_reverse:
                        mod_dists = read_len - 1 - confident_mod_cpg
                    else:
                        mod_dists = confident_mod_cpg
                    mod_dists = mod_dists[mod_dists <= args.max_distance]
                    if mod_dists.size > 0:
                        meth[: args.max_distance + 1] += np.bincount(mod_dists, minlength=args.max_distance + 1)

                stats["kept"] += 1
                if args.progress_every and stats["kept"] % args.progress_every == 0:
                    print(f"[{mark}] kept {stats['kept']} reads")

        raw_frac = np.full(cov.shape, np.nan, dtype=np.float64)
        ok = cov > 0
        raw_frac[ok] = meth[ok] / cov[ok]
        smooth_frac = smooth_fraction(meth, cov, args.smooth_window)
        profiles[mark] = {
            "cov": cov,
            "meth": meth,
            "raw_frac": raw_frac,
            "smooth_frac": smooth_frac,
            "reads_kept": stats["kept"],
            "reads_scanned": stats["scanned"],
        }
        print(
            f"[{mark}] scanned={stats['scanned']} kept={stats['kept']} "
            f"dist0_cov={int(cov[0])} dist{args.max_distance}_cov={int(cov[-1])}"
        )

    tsv_path = os.path.join(args.out_dir, "cpg_fraction_vs_cut_distance.tsv")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["mark", "distance_bp", "cpg_coverage", "cpg_methylated", "meth_frac_raw", "meth_frac_smooth"])
        for mark in args.marks:
            profile = profiles[mark]
            cov = profile["cov"]
            meth = profile["meth"]
            raw_frac = profile["raw_frac"]
            smooth_frac = profile["smooth_frac"]
            for dist in range(args.max_distance + 1):
                writer.writerow([
                    mark,
                    dist,
                    int(cov[dist]),
                    int(meth[dist]),
                    "" if np.isnan(raw_frac[dist]) else float(raw_frac[dist]),
                    "" if np.isnan(smooth_frac[dist]) else float(smooth_frac[dist]),
                ])

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=170)
    for mark in args.marks:
        profile = profiles[mark]
        ax.plot(
            np.arange(args.max_distance + 1),
            profile["smooth_frac"],
            label=mark,
            color=MARK_COLORS.get(mark, "#666666"),
            linewidth=2.0,
        )
    ax.set_xlabel("Distance from cut site (bp)")
    ax.set_ylabel("Fraction of CpG methylation")
    ax.set_title("CpG methylation relative to cut site")
    ax.set_xlim(0, args.max_distance)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cpg_fraction_vs_cut_distance.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, "cpg_fraction_vs_cut_distance.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
