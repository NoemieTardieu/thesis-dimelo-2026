import argparse
import csv
import glob
import os
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a Phase 3 training index by merging mark-specific Phase 2 manifests."
    )
    p.add_argument(
        "--manifest-glob",
        required=True,
        help='Glob for mark manifests, e.g. ".../backend_*_C/manifest_*_C.tsv"',
    )
    p.add_argument(
        "--out-tsv",
        required=True,
        help="Output merged index TSV (one row per window_id).",
    )
    p.add_argument(
        "--marks",
        default="h3k27ac,h3k27me3,h3k4me3",
        help="Comma-separated mark names expected in manifests.",
    )
    p.add_argument(
        "--min-known-frac",
        type=float,
        default=0.0,
        help="Optional filter: keep mark entry only if known_frac >= threshold.",
    )
    return p.parse_args()


def load_tsv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    args = parse_args()
    marks = [m.strip() for m in args.marks.split(",") if m.strip()]

    manifest_paths = sorted(glob.glob(args.manifest_glob))
    if not manifest_paths:
        raise SystemExit(f"No manifests matched: {args.manifest_glob}")

    by_window: Dict[str, Dict[str, str]] = {}
    for manifest in manifest_paths:
        rows = load_tsv(manifest)
        if not rows:
            continue
        for r in rows:
            mark = r["mark"]
            if mark not in marks:
                continue

            window_id = r["window_id"]
            known_frac = float(r.get("known_frac", "0") or 0.0)
            if known_frac < args.min_known_frac:
                continue

            base = by_window.setdefault(
                window_id,
                {
                    "window_id": r["window_id"],
                    "chrom": r["chrom"],
                    "start": r["start"],
                    "end": r["end"],
                    "split": r["split"],
                },
            )

            base[f"{mark}_npz_path"] = r["npz_path"]
            base[f"{mark}_known_frac"] = r.get("known_frac", "")
            base[f"{mark}_methylated_frac_known"] = r.get("methylated_frac_known", "")
            base[f"{mark}_n_reads_used"] = r.get("n_reads_used", "")

    # Keep deterministic order (chrom/start lexical from window_id sorting is acceptable here)
    out_rows = [by_window[k] for k in sorted(by_window.keys())]
    if not out_rows:
        raise SystemExit("No rows remained after filtering.")

    fields = ["window_id", "chrom", "start", "end", "split"]
    for m in marks:
        fields.extend(
            [
                f"{m}_npz_path",
                f"{m}_known_frac",
                f"{m}_methylated_frac_known",
                f"{m}_n_reads_used",
            ]
        )

    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    with open(args.out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"Wrote Phase 3 index: {args.out_tsv}")
    print(f"Rows: {len(out_rows)}")


if __name__ == "__main__":
    main()
