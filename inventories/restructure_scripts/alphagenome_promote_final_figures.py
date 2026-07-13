#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path


ALPHA = Path("/data/leuven/383/vsc38330/thesis_project_clean/alphagenome")
SRC_DIR = ALPHA / "archive" / "phase10_encode_benchmark_dev_runs" / "population_track_benchmark_ENCSR203XPU_200bp_final_remaining_duplicate" / "figures"
DST_DIR = ALPHA / "results" / "population_track_benchmark_ENCSR203XPU_200bp_final" / "figures"
MANIFEST = ALPHA.parent / "inventories" / "alphagenome_old_reorg_manifest.tsv"
FIELDS = ["source", "destination", "phase", "status", "reason"]


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    i = 2
    while True:
        candidate = dest.with_name(f"{dest.stem}.moved{i}{dest.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def main() -> None:
    rows: list[dict[str, str]] = []
    for src in sorted(SRC_DIR.glob("thesis_correlation_heatmap_panel.*")):
        dest = unique_dest(DST_DIR / src.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        old = str(src)
        shutil.move(str(src), str(dest))
        rows.append(
            {
                "source": old,
                "destination": str(dest),
                "phase": "phase11_final_thesis_figures",
                "status": "active_results",
                "reason": "thesis-ready final heatmap panel promoted out of archive",
            }
        )
    with MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writerows(rows)
    print(f"Promoted {len(rows)} final thesis figure files")


if __name__ == "__main__":
    main()
