import argparse
import csv
import glob
import os
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pysam


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
                pass
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


def get_tag_safe(rec: pysam.AlignedSegment, candidates: Sequence[str]):
    for t in candidates:
        if rec.has_tag(t):
            return rec.get_tag(t)
    return None


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
        pos = pos[:m]
        probs = probs[:m]
        if base.upper() == base_keep.upper():
            out_pos.extend(pos)
            out_prob.extend(probs)

    return out_pos, out_prob


def parse_intervals(intervals_tsv: str) -> List[Dict[str, str]]:
    with open(intervals_tsv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_manifest_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    return edges


def binned_counts_quantile(reg: np.ndarray, meth: np.ndarray, n_bins: int) -> np.ndarray:
    if reg.size == 0:
        return np.zeros((n_bins, n_bins), dtype=np.int64)
    reg_edges = quantile_edges(reg, n_bins)
    meth_edges = quantile_edges(meth, n_bins)
    xi = np.digitize(reg, reg_edges[1:-1], right=False)
    yi = np.digitize(meth, meth_edges[1:-1], right=False)
    mat = np.zeros((n_bins, n_bins), dtype=np.int64)
    for x, y in zip(xi, yi):
        mat[y, x] += 1
    return mat


def enrichment_log2(counts: np.ndarray, pseudocount: float = 1e-9) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    p_ij = counts / total
    p_i = p_ij.sum(axis=0, keepdims=True)
    p_j = p_ij.sum(axis=1, keepdims=True)
    expected = p_j @ p_i
    return np.log2((p_ij + pseudocount) / (expected + pseudocount))


def plot_heatmap(path: str, enr: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    vmax = np.nanpercentile(np.abs(enr), 99)
    vmax = max(vmax, 0.25)

    fig, ax = plt.subplots(figsize=(8.0, 6.2), dpi=150)
    im = ax.imshow(enr, origin="lower", cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(enr.shape[1]))
    ax.set_yticks(np.arange(enr.shape[0]))
    ax.set_xticklabels([f"Q{i+1}" for i in range(enr.shape[1])], rotation=45, ha="right")
    ax.set_yticklabels([f"Q{i+1}" for i in range(enr.shape[0])])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log2(observed / expected)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap_by_mark(path: str, mark_to_enr: Dict[str, np.ndarray], title: str) -> None:
    marks = sorted(mark_to_enr.keys())
    n = len(marks)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), dpi=150, squeeze=False)
    axes = axes[0]

    vmax = max(max(np.nanpercentile(np.abs(mark_to_enr[m]), 99), 0.25) for m in marks)
    im = None
    for ax, mark in zip(axes, marks):
        enr = mark_to_enr[mark]
        im = ax.imshow(enr, origin="lower", cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(mark)
        ax.set_xlabel("Reg quantile")
        ax.set_ylabel("M quantile")
        ax.set_xticks(np.arange(enr.shape[1]))
        ax.set_yticks(np.arange(enr.shape[0]))
        ax.set_xticklabels([f"Q{i+1}" for i in range(enr.shape[1])], rotation=45, ha="right")
        ax.set_yticklabels([f"Q{i+1}" for i in range(enr.shape[0])])

    fig.subplots_adjust(right=0.90, wspace=0.26)
    cax = fig.add_axes([0.915, 0.17, 0.012, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log2(observed / expected)")
    fig.suptitle(title, y=1.03)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_bed(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    chrom_to_intervals: Dict[str, List[Tuple[int, int]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            if end <= start:
                continue
            chrom_to_intervals.setdefault(chrom, []).append((start, end))

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for chrom, ivs in chrom_to_intervals.items():
        ivs.sort()
        starts = np.asarray([s for s, _ in ivs], dtype=np.int64)
        ends = np.sort(np.asarray([e for _, e in ivs], dtype=np.int64))
        out[chrom] = (starts, ends)
    return out


def overlap_mask_bins(chrom: str, starts: np.ndarray, ends: np.ndarray, peaks_idx: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if chrom not in peaks_idx:
        return np.zeros(starts.shape[0], dtype=bool)
    pstarts, pends = peaks_idx[chrom]
    n_started = np.searchsorted(pstarts, ends, side="left")
    n_ended = np.searchsorted(pends, starts, side="right")
    return n_started > n_ended


def sum_by_bin(arr: np.ndarray, bin_size: int) -> np.ndarray:
    idx = np.arange(0, arr.shape[0], bin_size, dtype=np.int64)
    return np.add.reduceat(arr.astype(np.int64), idx)


def compute_a_bin_counts(
    bam: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
    bin_size: int,
    min_mapq: int,
    ml_threshold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = (end - start + bin_size - 1) // bin_size
    a_cov_bins = np.zeros(n_bins, dtype=np.int64)
    a_mod_bins = np.zeros(n_bins, dtype=np.int64)

    for rec in bam.fetch(chrom, start, end):
        if rec.is_unmapped or rec.mapping_quality < min_mapq or rec.query_sequence is None:
            continue

        q_to_ref = rec.get_reference_positions(full_length=True)
        if q_to_ref is None:
            continue

        seq = rec.query_sequence
        for q_idx, base in enumerate(seq):
            if base.upper() != "A":
                continue
            if q_idx >= len(q_to_ref):
                continue
            rpos = q_to_ref[q_idx]
            if rpos is None or rpos < start or rpos >= end:
                continue
            b = (rpos - start) // bin_size
            a_cov_bins[b] += 1

        mod_qpos, mod_probs = extract_mod_qpos(rec, "A")
        for q_idx, pr in zip(mod_qpos, mod_probs):
            if pr < ml_threshold:
                continue
            if q_idx >= len(q_to_ref):
                continue
            rpos = q_to_ref[q_idx]
            if rpos is None or rpos < start or rpos >= end:
                continue
            b = (rpos - start) // bin_size
            a_mod_bins[b] += 1

    return a_cov_bins, a_mod_bins


def parse_mark_path_map(entries: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for x in entries:
        if "=" not in x:
            raise ValueError(f"Expected MARK=PATH, got: {x}")
        mark, path = x.split("=", 1)
        out[mark.strip()] = path.strip()
    return out


def write_bin_table(path: str, rows: Iterable[Dict[str, object]]) -> None:
    fields = [
        "mark", "window_id", "chrom", "bin_start", "bin_end", "split",
        "a_cov_calls", "a_mod_calls", "c_cov_calls", "c_meth_calls",
        "reg_value", "m_value", "coverage_ok", "peak_overlap"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def feature_cache_path(out_dir: str, mark: str, bin_size: int, reg_mode: str, m_mode: str, min_cov: int) -> str:
    return os.path.join(
        out_dir,
        f"feature_cache_{mark}_bin{bin_size}_{reg_mode}_{m_mode}_cov{min_cov}.npz",
    )


def feature_checkpoint_path(out_dir: str, mark: str, bin_size: int, reg_mode: str, m_mode: str, min_cov: int) -> str:
    return os.path.join(
        out_dir,
        f"feature_checkpoint_{mark}_bin{bin_size}_{reg_mode}_{m_mode}_cov{min_cov}.npz",
    )


def save_feature_cache(path: str, payload: Dict[str, np.ndarray]) -> None:
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(
        tmp,
        reg=payload["reg"],
        meth=payload["meth"],
        cov_ok=payload["cov_ok"],
        peak=payload["peak"],
    )
    os.replace(tmp, path)


def load_feature_cache(path: str) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=False)
    return {
        "reg": arr["reg"],
        "meth": arr["meth"],
        "cov_ok": arr["cov_ok"].astype(bool),
        "peak": arr["peak"].astype(bool),
    }


def save_feature_checkpoint(
    path: str,
    reg: np.ndarray,
    meth: np.ndarray,
    cov_ok: np.ndarray,
    peak: np.ndarray,
    processed_window_ids: Sequence[str],
) -> None:
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(
        tmp,
        reg=reg,
        meth=meth,
        cov_ok=cov_ok,
        peak=peak,
        processed_window_ids=np.asarray(list(processed_window_ids), dtype="U"),
    )
    os.replace(tmp, path)


def load_feature_checkpoint(path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    arr = np.load(path, allow_pickle=False)
    payload = {
        "reg": arr["reg"],
        "meth": arr["meth"],
        "cov_ok": arr["cov_ok"].astype(bool),
        "peak": arr["peak"].astype(bool),
    }
    processed = [str(x) for x in arr["processed_window_ids"].tolist()]
    return payload, processed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 windowed O/E enrichment heatmaps from 1Mb backend windows.")
    p.add_argument(
        "--intervals-tsv",
        default="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase2_long_context/intervals_long_context.tsv",
    )
    p.add_argument(
        "--manifest-glob",
        default="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase2_long_context/backends_all_marks/backend_*_C/manifest_*_C.tsv",
    )
    p.add_argument(
        "--bam-map",
        action="append",
        default=[],
        help="Mark BAM mapping as MARK=/path/to.bam (repeatable).",
    )
    p.add_argument(
        "--out-dir",
        default="/data/leuven/383/vsc38330/thesis_dimelo/src/preprocessing/phase2_long_context/windowed_enrichment",
    )
    p.add_argument("--bin-size", type=int, default=10000, help="Sub-bin size inside each 1Mb window (e.g., 1000 or 10000).")
    p.add_argument("--n-quantiles", type=int, default=10)
    p.add_argument("--min-mapq", type=int, default=20)
    p.add_argument("--ml-threshold", type=int, default=128)
    p.add_argument("--min-bin-c-coverage", type=int, default=30, help="Coverage filter for robustness stratification.")
    p.add_argument("--reg-mode", choices=["a_mod_per_kb", "a_mod_frac"], default="a_mod_per_kb")
    p.add_argument("--m-mode", choices=["c_meth_frac", "c_mod_per_kb"], default="c_meth_frac")
    p.add_argument(
        "--peaks-bed-map",
        action="append",
        default=[],
        help="Optional mark-specific BED mapping as MARK=/path/to/peaks.bed (repeatable).",
    )
    p.add_argument("--write-bin-table", action="store_true", help="Write per-bin feature TSV (large).")
    p.add_argument("--max-intervals-per-mark", type=int, default=0, help="For testing only. 0 = all.")
    p.add_argument("--progress-every", type=int, default=50, help="Print progress every N windows per mark (0 disables).")
    p.add_argument("--resume", action="store_true", help="Reuse per-mark cached features in --out-dir when available.")
    p.add_argument("--checkpoint-every", type=int, default=25, help="Save partial per-mark checkpoint every N new windows (0 disables).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    default_bams = {
        "h3k27ac": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam",
        "h3k27me3": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam",
        "h3k4me3": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam",
    }
    bam_map = dict(default_bams)
    bam_map.update(parse_mark_path_map(args.bam_map))

    peaks_map = parse_mark_path_map(args.peaks_bed_map)
    peaks_idx_by_mark: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for mark, bed_path in peaks_map.items():
        peaks_idx_by_mark[mark] = parse_bed(bed_path)

    intervals = parse_intervals(args.intervals_tsv)
    interval_order = {row["window_id"]: i for i, row in enumerate(intervals)}

    manifests = sorted(glob.glob(args.manifest_glob))
    if not manifests:
        raise SystemExit(f"No manifests matched: {args.manifest_glob}")

    manifest_rows_by_mark: Dict[str, List[Dict[str, str]]] = {}
    for path in manifests:
        rows = load_manifest_rows(path)
        if not rows:
            continue
        mark = rows[0]["mark"]
        rows.sort(key=lambda r: interval_order.get(r["window_id"], 10**12))
        manifest_rows_by_mark[mark] = rows

    marks = sorted(manifest_rows_by_mark.keys())
    missing_bam = [m for m in marks if m not in bam_map]
    if missing_bam:
        raise SystemExit(f"Missing BAM mapping for marks: {missing_bam}. Use --bam-map MARK=/path/to.bam")

    feature_by_mark: Dict[str, Dict[str, np.ndarray]] = {}
    bin_table_rows: List[Dict[str, object]] = []

    for mark in marks:
        rows = manifest_rows_by_mark[mark]
        if args.max_intervals_per_mark > 0:
            rows = rows[: args.max_intervals_per_mark]
        total_rows = len(rows)
        cache_path = feature_cache_path(
            out_dir=args.out_dir,
            mark=mark,
            bin_size=args.bin_size,
            reg_mode=args.reg_mode,
            m_mode=args.m_mode,
            min_cov=args.min_bin_c_coverage,
        )
        checkpoint_path = feature_checkpoint_path(
            out_dir=args.out_dir,
            mark=mark,
            bin_size=args.bin_size,
            reg_mode=args.reg_mode,
            m_mode=args.m_mode,
            min_cov=args.min_bin_c_coverage,
        )

        if args.resume and os.path.exists(cache_path):
            print(f"[{mark}] loading cached features: {cache_path}")
            feature_by_mark[mark] = load_feature_cache(cache_path)
            continue

        resumed_payload: Dict[str, np.ndarray] | None = None
        processed_window_ids: set[str] = set()
        if args.resume and os.path.exists(checkpoint_path):
            if args.write_bin_table:
                raise SystemExit(
                    "Cannot use --resume with --write-bin-table when resuming from partial checkpoint. "
                    "Re-run without --write-bin-table or do a fresh full run for complete bin table."
                )
            resumed_payload, resumed_ids = load_feature_checkpoint(checkpoint_path)
            processed_window_ids = set(resumed_ids)
            print(
                f"[{mark}] resuming from checkpoint: {checkpoint_path} "
                f"({len(processed_window_ids)}/{total_rows} windows already processed)"
            )

        reg_chunks: List[np.ndarray] = []
        m_chunks: List[np.ndarray] = []
        cov_ok_chunks: List[np.ndarray] = []
        peak_chunks: List[np.ndarray] = []
        new_windows_processed = 0

        peak_idx = peaks_idx_by_mark.get(mark)
        if resumed_payload is not None:
            reg_chunks.append(resumed_payload["reg"].astype(np.float32))
            m_chunks.append(resumed_payload["meth"].astype(np.float32))
            cov_ok_chunks.append(resumed_payload["cov_ok"].astype(bool))
            peak_chunks.append(resumed_payload["peak"].astype(bool))

        t0 = time.time()
        with pysam.AlignmentFile(bam_map[mark], "rb") as bam:
            for i, row in enumerate(rows, start=1):
                window_id = row["window_id"]
                if window_id in processed_window_ids:
                    continue

                npz_path = row["npz_path"]
                payload = np.load(npz_path, allow_pickle=False)
                c_cov = payload["coverage"].astype(np.int64)
                c_meth = payload["meth_counts"].astype(np.int64)

                chrom = row["chrom"]
                start = int(row["start"])
                end = int(row["end"])
                split = row.get("split", "")

                # A-channel from BAM (Reg proxy)
                a_cov_bins, a_mod_bins = compute_a_bin_counts(
                    bam=bam,
                    chrom=chrom,
                    start=start,
                    end=end,
                    bin_size=args.bin_size,
                    min_mapq=args.min_mapq,
                    ml_threshold=args.ml_threshold,
                )

                # C-channel from backend payload (M proxy)
                c_cov_bins = sum_by_bin(c_cov, args.bin_size)
                c_meth_bins = sum_by_bin(c_meth, args.bin_size)

                n_bins = c_cov_bins.shape[0]
                if args.reg_mode == "a_mod_per_kb":
                    reg_vals = a_mod_bins.astype(np.float64) / (args.bin_size / 1000.0)
                else:
                    reg_vals = np.divide(
                        a_mod_bins.astype(np.float64),
                        np.maximum(a_cov_bins.astype(np.float64), 1.0),
                    )

                if args.m_mode == "c_meth_frac":
                    m_vals = np.divide(
                        c_meth_bins.astype(np.float64),
                        np.maximum(c_cov_bins.astype(np.float64), 1.0),
                    )
                else:
                    m_vals = c_meth_bins.astype(np.float64) / (args.bin_size / 1000.0)

                cov_ok = c_cov_bins >= args.min_bin_c_coverage

                bin_starts = start + np.arange(n_bins, dtype=np.int64) * args.bin_size
                bin_ends = np.minimum(bin_starts + args.bin_size, end)

                if peak_idx is not None:
                    peak_overlap = overlap_mask_bins(chrom, bin_starts, bin_ends, peak_idx)
                else:
                    peak_overlap = np.zeros(n_bins, dtype=bool)

                reg_chunks.append(reg_vals.astype(np.float32))
                m_chunks.append(m_vals.astype(np.float32))
                cov_ok_chunks.append(cov_ok)
                peak_chunks.append(peak_overlap)
                processed_window_ids.add(window_id)
                new_windows_processed += 1

                if args.write_bin_table:
                    for b in range(n_bins):
                        bin_table_rows.append(
                            {
                                "mark": mark,
                                "window_id": window_id,
                                "chrom": chrom,
                                "bin_start": int(bin_starts[b]),
                                "bin_end": int(bin_ends[b]),
                                "split": split,
                                "a_cov_calls": int(a_cov_bins[b]),
                                "a_mod_calls": int(a_mod_bins[b]),
                                "c_cov_calls": int(c_cov_bins[b]),
                                "c_meth_calls": int(c_meth_bins[b]),
                                "reg_value": float(reg_vals[b]),
                                "m_value": float(m_vals[b]),
                                "coverage_ok": int(cov_ok[b]),
                                "peak_overlap": int(peak_overlap[b]),
                            }
                        )

                done_windows = len(processed_window_ids)
                if args.progress_every > 0 and (done_windows % args.progress_every == 0 or done_windows == total_rows):
                    elapsed = max(time.time() - t0, 1e-6)
                    rate = max(new_windows_processed, 1e-6) / elapsed
                    remaining = total_rows - done_windows
                    eta = remaining / max(rate, 1e-6)
                    print(
                        f"[{mark}] processed {done_windows}/{total_rows} windows "
                        f"({100.0 * done_windows / max(total_rows, 1):.1f}%), "
                        f"rate={rate:.2f} win/s, eta={eta/60.0:.1f} min"
                    )
                if (
                    args.checkpoint_every > 0
                    and new_windows_processed > 0
                    and (new_windows_processed % args.checkpoint_every == 0)
                ):
                    cur_payload = {
                        "reg": np.concatenate(reg_chunks) if reg_chunks else np.array([], dtype=np.float32),
                        "meth": np.concatenate(m_chunks) if m_chunks else np.array([], dtype=np.float32),
                        "cov_ok": np.concatenate(cov_ok_chunks) if cov_ok_chunks else np.array([], dtype=bool),
                        "peak": np.concatenate(peak_chunks) if peak_chunks else np.array([], dtype=bool),
                    }
                    save_feature_checkpoint(
                        path=checkpoint_path,
                        reg=cur_payload["reg"],
                        meth=cur_payload["meth"],
                        cov_ok=cur_payload["cov_ok"],
                        peak=cur_payload["peak"],
                        processed_window_ids=sorted(processed_window_ids),
                    )
                    print(
                        f"[{mark}] checkpoint saved: {len(processed_window_ids)}/{total_rows} windows -> {checkpoint_path}"
                    )

        feature_by_mark[mark] = {
            "reg": np.concatenate(reg_chunks) if reg_chunks else np.array([], dtype=np.float32),
            "meth": np.concatenate(m_chunks) if m_chunks else np.array([], dtype=np.float32),
            "cov_ok": np.concatenate(cov_ok_chunks) if cov_ok_chunks else np.array([], dtype=bool),
            "peak": np.concatenate(peak_chunks) if peak_chunks else np.array([], dtype=bool),
        }
        save_feature_cache(cache_path, feature_by_mark[mark])
        print(f"[{mark}] wrote feature cache: {cache_path}")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"[{mark}] removed checkpoint after successful completion: {checkpoint_path}")

    # Core heatmaps (coverage-filtered bins)
    mark_to_enr: Dict[str, np.ndarray] = {}
    summary_rows: List[Dict[str, object]] = []

    pooled_reg = []
    pooled_meth = []

    for mark in marks:
        d = feature_by_mark[mark]
        mask = d["cov_ok"]
        reg = d["reg"][mask]
        meth = d["meth"][mask]

        counts = binned_counts_quantile(reg, meth, args.n_quantiles)
        enr = enrichment_log2(counts)
        mark_to_enr[mark] = enr

        pooled_reg.append(reg)
        pooled_meth.append(meth)

        plot_heatmap(
            path=os.path.join(args.out_dir, f"oe_heatmap_{mark}.svg"),
            enr=enr,
            title=f"Windowed Reg vs M O/E ({mark})",
            xlabel=f"Reg quantile ({args.reg_mode})",
            ylabel=f"M quantile ({args.m_mode})",
        )

        summary_rows.append(
            {
                "scope": "mark",
                "mark": mark,
                "subset": f"coverage_ge_{args.min_bin_c_coverage}",
                "n_bins": int(mask.sum()),
            }
        )

    pooled_reg_arr = np.concatenate(pooled_reg) if pooled_reg else np.array([], dtype=np.float32)
    pooled_meth_arr = np.concatenate(pooled_meth) if pooled_meth else np.array([], dtype=np.float32)
    pooled_counts = binned_counts_quantile(pooled_reg_arr, pooled_meth_arr, args.n_quantiles)
    pooled_enr = enrichment_log2(pooled_counts)

    plot_heatmap(
        path=os.path.join(args.out_dir, "oe_heatmap_pooled.svg"),
        enr=pooled_enr,
        title="Windowed Reg vs M O/E (Pooled)",
        xlabel=f"Reg quantile ({args.reg_mode})",
        ylabel=f"M quantile ({args.m_mode})",
    )
    plot_heatmap_by_mark(
        path=os.path.join(args.out_dir, "oe_heatmap_by_mark.svg"),
        mark_to_enr=mark_to_enr,
        title="Windowed Reg vs M O/E by Mark",
    )

    summary_rows.append(
        {
            "scope": "pooled",
            "mark": "all",
            "subset": f"coverage_ge_{args.min_bin_c_coverage}",
            "n_bins": int(pooled_reg_arr.shape[0]),
        }
    )

    # Optional peak vs non-peak stratification
    if peaks_idx_by_mark:
        peak_mark_to_enr: Dict[str, np.ndarray] = {}
        nonpeak_mark_to_enr: Dict[str, np.ndarray] = {}
        pooled_peak_reg = []
        pooled_peak_meth = []
        pooled_nonpeak_reg = []
        pooled_nonpeak_meth = []

        for mark in marks:
            if mark not in peaks_idx_by_mark:
                continue
            d = feature_by_mark[mark]
            base_mask = d["cov_ok"]
            in_peak = np.logical_and(base_mask, d["peak"])
            out_peak = np.logical_and(base_mask, np.logical_not(d["peak"]))

            reg_in = d["reg"][in_peak]
            meth_in = d["meth"][in_peak]
            reg_out = d["reg"][out_peak]
            meth_out = d["meth"][out_peak]

            if reg_in.size > 0:
                peak_enr = enrichment_log2(binned_counts_quantile(reg_in, meth_in, args.n_quantiles))
                peak_mark_to_enr[mark] = peak_enr
                plot_heatmap(
                    path=os.path.join(args.out_dir, f"oe_heatmap_{mark}_in_peak.svg"),
                    enr=peak_enr,
                    title=f"Windowed Reg vs M O/E ({mark}, peak bins)",
                    xlabel=f"Reg quantile ({args.reg_mode})",
                    ylabel=f"M quantile ({args.m_mode})",
                )

            if reg_out.size > 0:
                nonpeak_enr = enrichment_log2(binned_counts_quantile(reg_out, meth_out, args.n_quantiles))
                nonpeak_mark_to_enr[mark] = nonpeak_enr
                plot_heatmap(
                    path=os.path.join(args.out_dir, f"oe_heatmap_{mark}_nonpeak.svg"),
                    enr=nonpeak_enr,
                    title=f"Windowed Reg vs M O/E ({mark}, non-peak bins)",
                    xlabel=f"Reg quantile ({args.reg_mode})",
                    ylabel=f"M quantile ({args.m_mode})",
                )

            pooled_peak_reg.append(reg_in)
            pooled_peak_meth.append(meth_in)
            pooled_nonpeak_reg.append(reg_out)
            pooled_nonpeak_meth.append(meth_out)

            summary_rows.append(
                {"scope": "mark", "mark": mark, "subset": "in_peak", "n_bins": int(reg_in.shape[0])}
            )
            summary_rows.append(
                {"scope": "mark", "mark": mark, "subset": "non_peak", "n_bins": int(reg_out.shape[0])}
            )

        if peak_mark_to_enr:
            plot_heatmap_by_mark(
                path=os.path.join(args.out_dir, "oe_heatmap_by_mark_in_peak.svg"),
                mark_to_enr=peak_mark_to_enr,
                title="Windowed Reg vs M O/E by Mark (peak bins)",
            )
        if nonpeak_mark_to_enr:
            plot_heatmap_by_mark(
                path=os.path.join(args.out_dir, "oe_heatmap_by_mark_nonpeak.svg"),
                mark_to_enr=nonpeak_mark_to_enr,
                title="Windowed Reg vs M O/E by Mark (non-peak bins)",
            )

        all_peak_reg = np.concatenate([x for x in pooled_peak_reg if x.size > 0]) if any(x.size > 0 for x in pooled_peak_reg) else np.array([], dtype=np.float32)
        all_peak_meth = np.concatenate([x for x in pooled_peak_meth if x.size > 0]) if any(x.size > 0 for x in pooled_peak_meth) else np.array([], dtype=np.float32)
        all_nonpeak_reg = np.concatenate([x for x in pooled_nonpeak_reg if x.size > 0]) if any(x.size > 0 for x in pooled_nonpeak_reg) else np.array([], dtype=np.float32)
        all_nonpeak_meth = np.concatenate([x for x in pooled_nonpeak_meth if x.size > 0]) if any(x.size > 0 for x in pooled_nonpeak_meth) else np.array([], dtype=np.float32)

        if all_peak_reg.size > 0:
            plot_heatmap(
                path=os.path.join(args.out_dir, "oe_heatmap_pooled_in_peak.svg"),
                enr=enrichment_log2(binned_counts_quantile(all_peak_reg, all_peak_meth, args.n_quantiles)),
                title="Windowed Reg vs M O/E (Pooled peak bins)",
                xlabel=f"Reg quantile ({args.reg_mode})",
                ylabel=f"M quantile ({args.m_mode})",
            )
            summary_rows.append({"scope": "pooled", "mark": "all", "subset": "in_peak", "n_bins": int(all_peak_reg.shape[0])})

        if all_nonpeak_reg.size > 0:
            plot_heatmap(
                path=os.path.join(args.out_dir, "oe_heatmap_pooled_nonpeak.svg"),
                enr=enrichment_log2(binned_counts_quantile(all_nonpeak_reg, all_nonpeak_meth, args.n_quantiles)),
                title="Windowed Reg vs M O/E (Pooled non-peak bins)",
                xlabel=f"Reg quantile ({args.reg_mode})",
                ylabel=f"M quantile ({args.m_mode})",
            )
            summary_rows.append({"scope": "pooled", "mark": "all", "subset": "non_peak", "n_bins": int(all_nonpeak_reg.shape[0])})

    # Write summary and optional long bin table
    summary_path = os.path.join(args.out_dir, "windowed_oe_summary.tsv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "mark", "subset", "n_bins"], delimiter="\t")
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    notes_path = os.path.join(args.out_dir, "windowed_oe_notes.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2 Windowed O/E Enrichment\n\n")
        f.write(f"- bin_size: {args.bin_size} bp\n")
        f.write(f"- reg_mode: {args.reg_mode}\n")
        f.write(f"- m_mode: {args.m_mode}\n")
        f.write(f"- min_bin_c_coverage: {args.min_bin_c_coverage}\n")
        f.write(f"- n_quantiles: {args.n_quantiles}\n")
        f.write(f"- marks: {', '.join(marks)}\n")
        if peaks_idx_by_mark:
            f.write(f"- peak stratification: enabled for {', '.join(sorted(peaks_idx_by_mark.keys()))}\n")
        else:
            f.write("- peak stratification: disabled\n")

    if args.write_bin_table:
        write_bin_table(os.path.join(args.out_dir, "windowed_bin_features.tsv"), bin_table_rows)

    print(f"Wrote windowed O/E outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
