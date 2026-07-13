import argparse
import csv
import os
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pysam


def get_tag_safe(rec: pysam.AlignedSegment, candidates: Sequence[str]):
    for t in candidates:
        if rec.has_tag(t):
            return rec.get_tag(t)
    return None


def parse_mm_tag(mm_value: str) -> List[Tuple[str, str, List[int]]]:
    groups_out = []
    if not mm_value:
        return groups_out

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
                pass

        groups_out.append((base, mod_code, deltas))

    return groups_out


def parse_ml_tag(ml_value) -> List[int]:
    if ml_value is None:
        return []
    try:
        return [int(x) for x in ml_value]
    except Exception:
        return []


def decode_delta_positions(seq: str, base: str, deltas: List[int]) -> List[int]:
    base = base.upper()
    occ = [i for i, b in enumerate(seq) if b.upper() == base]
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


def extract_base_mods(rec: pysam.AlignedSegment, base_keep: str) -> Tuple[List[int], List[int]]:
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
    pos_out: List[int] = []
    prob_out: List[int] = []

    for (base, _mod_code, deltas), L in zip(mm_groups, lengths):
        probs = ml_list[idx : idx + L]
        idx += L
        pos = decode_delta_positions(seq, base, deltas)

        m = min(len(pos), len(probs))
        pos = pos[:m]
        probs = probs[:m]

        if base.upper() == base_keep.upper():
            pos_out.extend(pos)
            prob_out.extend(probs)

    return pos_out, prob_out


def infer_discrete_labels(
    coverage: np.ndarray,
    meth_counts: np.ndarray,
    min_coverage: int,
    methylated_frac_thr: float,
    unmethylated_frac_thr: float,
) -> np.ndarray:
    labels = np.full(coverage.shape[0], 2, dtype=np.uint8)
    covered = coverage >= min_coverage
    if not np.any(covered):
        return labels

    frac = np.zeros_like(coverage, dtype=np.float32)
    frac[covered] = meth_counts[covered] / coverage[covered]

    labels[np.logical_and(covered, frac >= methylated_frac_thr)] = 1
    labels[np.logical_and(covered, frac <= unmethylated_frac_thr)] = 0
    return labels


def process_interval(
    bam: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
    target_base: str,
    min_mapq: int,
    ml_threshold: int,
    min_coverage: int,
    methylated_frac_thr: float,
    unmethylated_frac_thr: float,
    max_reads_per_interval: int,
) -> Dict[str, np.ndarray]:
    L = end - start
    coverage = np.zeros(L, dtype=np.uint16)
    meth_counts = np.zeros(L, dtype=np.uint16)

    n_reads = 0
    for rec in bam.fetch(chrom, start, end):
        if max_reads_per_interval > 0 and n_reads >= max_reads_per_interval:
            break
        if rec.is_unmapped or rec.mapping_quality < min_mapq or rec.query_sequence is None:
            continue

        q_to_ref = rec.get_reference_positions(full_length=True)
        if q_to_ref is None:
            continue

        seq = rec.query_sequence
        for q_idx, b in enumerate(seq):
            if b.upper() != target_base.upper():
                continue
            if q_idx >= len(q_to_ref):
                continue
            rpos = q_to_ref[q_idx]
            if rpos is None or rpos < start or rpos >= end:
                continue
            coverage[rpos - start] += 1

        mod_qpos, mod_probs = extract_base_mods(rec, target_base)
        for q_idx, pr in zip(mod_qpos, mod_probs):
            if pr < ml_threshold:
                continue
            if q_idx >= len(q_to_ref):
                continue
            rpos = q_to_ref[q_idx]
            if rpos is None or rpos < start or rpos >= end:
                continue
            meth_counts[rpos - start] += 1

        n_reads += 1

    labels = infer_discrete_labels(
        coverage=coverage,
        meth_counts=meth_counts,
        min_coverage=min_coverage,
        methylated_frac_thr=methylated_frac_thr,
        unmethylated_frac_thr=unmethylated_frac_thr,
    )

    return {
        "methyl_ids": labels,
        "coverage": coverage,
        "meth_counts": meth_counts,
        "n_reads_used": np.array([n_reads], dtype=np.int32),
    }


def read_intervals(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def load_existing_window_ids(manifest_path: str) -> Set[str]:
    if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0:
        return set()
    out: Set[str] = set()
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            wid = row.get("window_id")
            if wid:
                out.add(wid)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build long-context methylation label backend from BAM + interval TSV.")
    parser.add_argument("--bam", required=True)
    parser.add_argument("--intervals-tsv", required=True, help="TSV with columns: window_id, chrom, start, end, split")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mark", required=True, help="Label for manifest (e.g., h3k27ac)")
    parser.add_argument("--target-base", default="C", choices=["A", "C"])
    parser.add_argument("--min-mapq", type=int, default=20)
    parser.add_argument("--ml-threshold", type=int, default=128)
    parser.add_argument("--min-coverage", type=int, default=3)
    parser.add_argument("--methylated-frac-thr", type=float, default=0.7)
    parser.add_argument("--unmethylated-frac-thr", type=float, default=0.3)
    parser.add_argument("--max-reads-per-interval", type=int, default=0, help="0 means no cap")
    parser.add_argument("--max-intervals", type=int, default=0, help="0 means process all")
    parser.add_argument("--resume", action="store_true", help="Resume run by appending to existing manifest and skipping finished windows.")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    labels_dir = os.path.join(args.out_dir, f"{args.mark}_{args.target_base}_labels")
    os.makedirs(labels_dir, exist_ok=True)

    manifest_path = os.path.join(args.out_dir, f"manifest_{args.mark}_{args.target_base}.tsv")
    manifest_fields = [
        "window_id",
        "chrom",
        "start",
        "end",
        "split",
        "mark",
        "target_base",
        "npz_path",
        "known_frac",
        "methylated_frac_known",
        "n_reads_used",
    ]

    existing = load_existing_window_ids(manifest_path) if args.resume else set()
    mode = "a" if args.resume and os.path.exists(manifest_path) else "w"
    need_header = (not os.path.exists(manifest_path)) or os.path.getsize(manifest_path) == 0 or mode == "w"

    with pysam.AlignmentFile(args.bam, "rb") as bam, open(
        manifest_path, mode, encoding="utf-8", newline="", buffering=1
    ) as mf:
        writer = csv.DictWriter(mf, fieldnames=manifest_fields, delimiter="\t")
        if need_header:
            writer.writeheader()
            mf.flush()

        processed = 0
        skipped = 0
        for row in read_intervals(args.intervals_tsv):
            if args.max_intervals > 0 and processed >= args.max_intervals:
                break

            chrom = row["chrom"]
            start = int(row["start"])
            end = int(row["end"])
            window_id = row.get("window_id", f"{chrom}:{start}-{end}")
            split = row.get("split", "unknown")
            if window_id in existing:
                skipped += 1
                continue

            out = process_interval(
                bam=bam,
                chrom=chrom,
                start=start,
                end=end,
                target_base=args.target_base,
                min_mapq=args.min_mapq,
                ml_threshold=args.ml_threshold,
                min_coverage=args.min_coverage,
                methylated_frac_thr=args.methylated_frac_thr,
                unmethylated_frac_thr=args.unmethylated_frac_thr,
                max_reads_per_interval=args.max_reads_per_interval,
            )

            safe_id = window_id.replace(":", "_").replace("-", "_")
            npz_name = f"{safe_id}.npz"
            npz_path = os.path.join(labels_dir, npz_name)
            np.savez_compressed(
                npz_path,
                methyl_ids=out["methyl_ids"],
                coverage=out["coverage"],
                meth_counts=out["meth_counts"],
                n_reads_used=out["n_reads_used"],
                chrom=chrom,
                start=start,
                end=end,
                mark=args.mark,
                target_base=args.target_base,
                min_mapq=args.min_mapq,
                ml_threshold=args.ml_threshold,
                min_coverage=args.min_coverage,
                methylated_frac_thr=args.methylated_frac_thr,
                unmethylated_frac_thr=args.unmethylated_frac_thr,
            )

            labels = out["methyl_ids"]
            known = labels != 2
            known_frac = float(np.mean(known)) if labels.size else 0.0
            meth_known_frac = float(np.mean(labels[known] == 1)) if np.any(known) else 0.0

            writer.writerow(
                {
                    "window_id": window_id,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "split": split,
                    "mark": args.mark,
                    "target_base": args.target_base,
                    "npz_path": npz_path,
                    "known_frac": f"{known_frac:.6f}",
                    "methylated_frac_known": f"{meth_known_frac:.6f}",
                    "n_reads_used": int(out["n_reads_used"][0]),
                }
            )
            mf.flush()
            processed += 1
            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(f"Processed {processed} intervals (skipped={skipped})...", flush=True)

    print(f"Wrote manifest: {manifest_path} (processed={processed}, skipped={skipped})", flush=True)


if __name__ == "__main__":
    main()
