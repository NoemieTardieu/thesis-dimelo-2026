import argparse
import csv
import glob
import os
import statistics
from typing import Dict, List


def mean(xs: List[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def p50(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[len(ys) // 2]


def summarize_manifest(path: str) -> List[Dict[str, str]]:
    by_split: Dict[str, Dict[str, List[float]]] = {}
    mark = "unknown"
    target_base = "unknown"

    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            mark = row.get("mark", mark)
            target_base = row.get("target_base", target_base)
            split = row.get("split", "unknown")
            by_split.setdefault(
                split,
                {
                    "known_frac": [],
                    "meth_known": [],
                    "reads": [],
                },
            )
            by_split[split]["known_frac"].append(float(row.get("known_frac", 0.0)))
            by_split[split]["meth_known"].append(float(row.get("methylated_frac_known", 0.0)))
            by_split[split]["reads"].append(float(row.get("n_reads_used", 0.0)))

    out: List[Dict[str, str]] = []
    for split, d in sorted(by_split.items()):
        out.append(
            {
                "manifest": path,
                "mark": mark,
                "target_base": target_base,
                "split": split,
                "n_intervals": str(len(d["known_frac"])),
                "known_frac_mean": f"{mean(d['known_frac']):.6f}",
                "known_frac_p50": f"{p50(d['known_frac']):.6f}",
                "methylated_frac_known_mean": f"{mean(d['meth_known']):.6f}",
                "methylated_frac_known_p50": f"{p50(d['meth_known']):.6f}",
                "reads_used_mean": f"{mean(d['reads']):.2f}",
                "reads_used_p50": f"{p50(d['reads']):.2f}",
            }
        )
    return out


def write_tsv(path: str, rows: List[Dict[str, str]], fields: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_md(path: str, rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Long-context backend QC summary\n\n")
        f.write("| Mark | Base | Split | Intervals | known_frac mean | known_frac p50 | methylated_frac_known mean | reads_used mean |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['mark']} | {r['target_base']} | {r['split']} | {r['n_intervals']} | {r['known_frac_mean']} | {r['known_frac_p50']} | {r['methylated_frac_known_mean']} | {r['reads_used_mean']} |\n"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize backend manifest coverage/label balance.")
    p.add_argument("--manifest-glob", required=True, help="Glob for manifest TSV files.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifests = sorted(glob.glob(args.manifest_glob))
    if not manifests:
        raise SystemExit(f"No manifest files matched: {args.manifest_glob}")

    rows: List[Dict[str, str]] = []
    for m in manifests:
        rows.extend(summarize_manifest(m))

    fields = [
        "manifest",
        "mark",
        "target_base",
        "split",
        "n_intervals",
        "known_frac_mean",
        "known_frac_p50",
        "methylated_frac_known_mean",
        "methylated_frac_known_p50",
        "reads_used_mean",
        "reads_used_p50",
    ]
    out_tsv = os.path.join(args.out_dir, "backend_manifest_qc_summary.tsv")
    out_md = os.path.join(args.out_dir, "backend_manifest_qc_summary.md")
    write_tsv(out_tsv, rows, fields)
    write_md(out_md, rows)
    print(f"Wrote summary: {out_tsv}")
    print(f"Wrote summary: {out_md}")


if __name__ == "__main__":
    main()
