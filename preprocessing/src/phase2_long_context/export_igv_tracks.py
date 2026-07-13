import argparse
import csv
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pysam


DEFAULT_BAMS = {
    "h3k27ac": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337767_gm12878_h3k27ac.mod_mappings.sorted.bam",
    "h3k27me3": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337768_gm12878_h3k27me3.mod_mappings.sorted.bam",
    "h3k4me3": "/scratch/leuven/383/vsc38330/data/raw_bam/GSM6337769_gm12878_h3k4me3.mod_mappings.sorted.bam",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export IGV-compatible tracks from long-context methylation backends. "
            "Produces a CpG methylation fraction bedGraph from manifest NPZs and an optional "
            "cut-site density bedGraph from BAM read ends."
        )
    )
    p.add_argument("--manifest", required=True, help="Manifest TSV produced by build_methylation_interval_backend.py")
    p.add_argument("--out-prefix", required=True, help="Output prefix for generated track files")
    p.add_argument("--bam", help="Input BAM for cut-site track; optional if only exporting methylation")
    p.add_argument("--mark", help="Mark name used to infer default BAM if --bam is omitted")
    p.add_argument("--min-coverage", type=int, default=3, help="Minimum per-base coverage for methylation track")
    p.add_argument(
        "--bin-size",
        type=int,
        default=200,
        help="Bin size for exported tracks. Use 1 for per-base output; 200 or 1000 are more practical for IGV.",
    )
    p.add_argument(
        "--emit-discrete-labels",
        action="store_true",
        help="Export methylation labels 0/1/2 instead of fractions (0=unmeth,1=meth,2=unknown)",
    )
    p.add_argument(
        "--region",
        help="Optional region chrom:start-end to restrict export for testing or focused IGV snapshots",
    )
    p.add_argument(
        "--cut-end",
        choices=["5prime", "both"],
        default="5prime",
        help="Count only read 5' ends or both aligned ends for the cut-site track",
    )
    p.add_argument("--min-mapq", type=int, default=20)
    p.add_argument(
        "--cut-threshold",
        type=int,
        default=1,
        help="Minimum count to emit a bin in the cut-site bedGraph",
    )
    return p.parse_args()


def parse_region(region: Optional[str]) -> Optional[Tuple[str, int, int]]:
    if not region:
        return None
    chrom, rest = region.split(":", 1)
    start_s, end_s = rest.replace(",", "").split("-", 1)
    start = int(start_s)
    end = int(end_s)
    if end <= start:
        raise ValueError(f"Invalid region: {region}")
    return chrom, start, end


def read_manifest_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def filter_rows_to_region(
    rows: Sequence[Dict[str, str]], region: Optional[Tuple[str, int, int]]
) -> List[Dict[str, str]]:
    if region is None:
        return list(rows)
    rchrom, rstart, rend = region
    out: List[Dict[str, str]] = []
    for row in rows:
        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])
        if chrom != rchrom:
            continue
        if end <= rstart or start >= rend:
            continue
        out.append(row)
    return out


def write_bedgraph_header(path: str, name: str, description: str, color: str) -> None:
    with open(path, "w", encoding="utf-8") as out:
        out.write(
            f'track type=bedGraph name="{name}" description="{description}" visibility=full color={color} altColor={color}\n'
        )


def iter_manifest_arrays(
    rows: Sequence[Dict[str, str]],
    region: Optional[Tuple[str, int, int]],
) -> Iterator[Tuple[str, int, np.ndarray, np.ndarray, np.ndarray]]:
    for row in rows:
        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])
        payload = np.load(row["npz_path"], allow_pickle=False)
        labels = payload["methyl_ids"].astype(np.uint8)
        coverage = payload["coverage"].astype(np.int32)
        meth_counts = payload["meth_counts"].astype(np.int32)

        if region is not None and chrom == region[0]:
            rstart = max(start, region[1])
            rend = min(end, region[2])
            if rend <= rstart:
                continue
            lo = rstart - start
            hi = rend - start
            yield chrom, rstart, labels[lo:hi], coverage[lo:hi], meth_counts[lo:hi]
        else:
            yield chrom, start, labels, coverage, meth_counts


def bin_edges(n: int, bin_size: int) -> range:
    return range(0, n, bin_size)


def write_methylation_track(
    path: str,
    rows: Sequence[Dict[str, str]],
    min_coverage: int,
    bin_size: int,
    emit_discrete_labels: bool,
    region: Optional[Tuple[str, int, int]],
) -> None:
    color = "213,94,0" if not emit_discrete_labels else "0,114,178"
    desc = (
        f"CpG methylation fraction, {bin_size} bp bins"
        if not emit_discrete_labels
        else f"Methylation labels, {bin_size} bp bins"
    )
    write_bedgraph_header(path, os.path.basename(path), desc, color)

    with open(path, "a", encoding="utf-8") as out:
        for chrom, start, labels, coverage, meth_counts in iter_manifest_arrays(rows, region):
            n = coverage.shape[0]
            for lo in bin_edges(n, bin_size):
                hi = min(lo + bin_size, n)
                cov = coverage[lo:hi]
                meth = meth_counts[lo:hi]
                mask = cov >= min_coverage
                if not np.any(mask):
                    continue
                if emit_discrete_labels:
                    value = float(labels[lo:hi][mask].mean())
                else:
                    value = float(meth[mask].sum() / cov[mask].sum())
                out.write(f"{chrom}\t{start + lo}\t{start + hi}\t{value:.6f}\n")


def resolve_bam(mark: Optional[str], bam: Optional[str]) -> str:
    if bam:
        return bam
    if mark and mark in DEFAULT_BAMS:
        return DEFAULT_BAMS[mark]
    raise SystemExit("Provide --bam or --mark with a known default BAM")


def write_cut_track(
    path: str,
    bam_path: str,
    rows: Sequence[Dict[str, str]],
    region: Optional[Tuple[str, int, int]],
    cut_end: str,
    min_mapq: int,
    threshold: int,
    bin_size: int,
) -> None:
    write_bedgraph_header(path, os.path.basename(path), f"Read-end cut-site density, {bin_size} bp bins", "0,0,0")
    with pysam.AlignmentFile(bam_path, "rb") as bam, open(path, "a", encoding="utf-8") as out:
        for row in rows:
            chrom = row["chrom"]
            start = int(row["start"])
            end = int(row["end"])
            if region is not None:
                if chrom != region[0]:
                    continue
                start = max(start, region[1])
                end = min(end, region[2])
                if end <= start:
                    continue

            counts = np.zeros(end - start, dtype=np.int32)
            for rec in bam.fetch(chrom, start, end):
                if rec.is_unmapped or rec.is_secondary or rec.is_supplementary:
                    continue
                if rec.mapping_quality < min_mapq:
                    continue
                if rec.reference_start is None or rec.reference_end is None:
                    continue

                five_prime = rec.reference_end - 1 if rec.is_reverse else rec.reference_start
                positions = [five_prime]
                if cut_end == "both":
                    other_end = rec.reference_start if rec.is_reverse else rec.reference_end - 1
                    positions.append(other_end)

                for pos in positions:
                    if pos < start or pos >= end:
                        continue
                    counts[pos - start] += 1

            for lo in bin_edges(counts.shape[0], bin_size):
                hi = min(lo + bin_size, counts.shape[0])
                value = int(counts[lo:hi].sum())
                if value < threshold:
                    continue
                out.write(f"{chrom}\t{start + lo}\t{start + hi}\t{value}\n")


def main() -> None:
    args = parse_args()
    if args.bin_size < 1:
        raise SystemExit("--bin-size must be >= 1")

    region = parse_region(args.region)
    rows = filter_rows_to_region(read_manifest_rows(args.manifest), region)
    if not rows:
        raise SystemExit("No manifest rows overlap the requested region")

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    methyl_path = f"{args.out_prefix}.methyl_frac.bedGraph"
    write_methylation_track(
        methyl_path,
        rows,
        min_coverage=args.min_coverage,
        bin_size=args.bin_size,
        emit_discrete_labels=args.emit_discrete_labels,
        region=region,
    )
    print(f"Wrote {methyl_path}")

    if args.bam or args.mark:
        bam_path = resolve_bam(args.mark, args.bam)
        cut_path = f"{args.out_prefix}.cut_sites.bedGraph"
        write_cut_track(
            cut_path,
            bam_path=bam_path,
            rows=rows,
            region=region,
            cut_end=args.cut_end,
            min_mapq=args.min_mapq,
            threshold=args.cut_threshold,
            bin_size=args.bin_size,
        )
        print(f"Wrote {cut_path}")


if __name__ == "__main__":
    main()
