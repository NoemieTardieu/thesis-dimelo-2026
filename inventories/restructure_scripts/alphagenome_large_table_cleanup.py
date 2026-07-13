#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path


ALPHA = Path("/data/leuven/383/vsc38330/thesis_project_clean/alphagenome")
SERVER = ALPHA / "server_artifacts" / "alphagenome_archive_large_tables"
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
    archive = ALPHA / "archive"
    for src in sorted(archive.rglob("canonical_raw_bins.tsv.gz")) + sorted(archive.rglob("canonical_normalized_bins.tsv.gz")):
        rel = src.relative_to(archive)
        dest = unique_dest(SERVER / "canonical_bins_from_archive" / rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        old = str(src)
        shutil.move(str(src), str(dest))
        rows.append(
            {
                "source": old,
                "destination": str(dest),
                "phase": "archive_large_table_cleanup",
                "status": "server_artifact",
                "reason": "canonical raw/normalized bin table moved out of Git-facing archive",
            }
        )
    with MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writerows(rows)
    print(f"Moved {len(rows)} canonical bin tables")


if __name__ == "__main__":
    main()
