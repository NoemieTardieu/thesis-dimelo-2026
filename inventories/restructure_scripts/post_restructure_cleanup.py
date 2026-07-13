#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path


CLEAN = Path("/data/leuven/383/vsc38330/thesis_project_clean")


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    i = 2
    while True:
        candidate = dest.with_name(f"{dest.stem}.moved{i}{dest.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def append_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "destination", "category", "reason"], delimiter="\t")
        writer.writerows(rows)


def move_file(src: Path, dest: Path, reason: str) -> dict[str, str]:
    dest = unique_dest(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = str(src)
    shutil.move(str(src), str(dest))
    return {"source": old, "destination": str(dest), "category": "to_delete", "reason": reason}


def main() -> None:
    rows: list[dict[str, str]] = []

    # Enforce the rule that all cluster job scripts/logs live in pillar-local to_delete/slurm.
    for pillar in ["preprocessing", "hyena-dna", "alphagenome"]:
        base = CLEAN / pillar
        for src in sorted(base.rglob("*")):
            if not src.is_file() or "to_delete" in src.parts:
                continue
            is_slurm = src.suffix == ".sbatch" or src.name.startswith("slurm-") or src.suffix in {".out", ".err"}
            if is_slurm:
                rel = src.relative_to(base)
                rows.append(
                    move_file(
                        src,
                        base / "to_delete" / "slurm" / rel,
                        "post-check cleanup: all SLURM/job logs belong in to_delete/slurm",
                    )
                )

    # Enforce the rule that CNN-related work is quarantined, not just archived.
    pre = CLEAN / "preprocessing"
    for src in sorted(pre.rglob("*")):
        if not src.is_file() or "to_delete" in src.parts:
            continue
        path_l = "/".join(src.parts).lower()
        name_l = src.name.lower()
        if "cnn" in name_l or "cnn" in path_l:
            rel = src.relative_to(pre)
            rows.append(
                move_file(
                    src,
                    pre / "to_delete" / "cnn_legacy" / rel,
                    "post-check cleanup: CNN-related file not used in final thesis",
                )
            )

    # Python caches should not remain in active or OLD trees.
    for pillar in ["preprocessing", "hyena-dna", "alphagenome"]:
        base = CLEAN / pillar
        for src in sorted(base.rglob("__pycache__")):
            if not src.exists() or "to_delete" in src.parts:
                continue
            rel = src.relative_to(base)
            rows.append(
                move_file(
                    src,
                    base / "to_delete" / "python_cache" / rel,
                    "post-check cleanup: Python cache excluded from GitHub",
                )
            )

    append_manifest(CLEAN / "inventories" / "to_delete_manifest.tsv", rows)
    print(f"Post-cleanup moved {len(rows)} files/directories")


if __name__ == "__main__":
    main()
