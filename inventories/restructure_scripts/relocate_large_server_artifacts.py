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
    return {"source": old, "destination": str(dest), "category": "server_artifact", "reason": reason}


def main() -> None:
    rows: list[dict[str, str]] = []

    alpha = CLEAN / "alphagenome"
    ext = alpha / "metadata" / "external_tracks"
    for src in sorted(ext.rglob("*.bigWig")) + sorted(ext.rglob("*.bw")):
        rel = src.relative_to(ext)
        rows.append(
            move_file(
                src,
                alpha / "server_artifacts" / "external_tracks" / rel,
                "large external BigWig retained on server but excluded from GitHub",
            )
        )

    hyena = CLEAN / "hyena-dna"
    ckpt = hyena / "results" / "checkpoints"
    for src in sorted(ckpt.glob("*.pt")):
        rows.append(
            move_file(
                src,
                hyena / "server_artifacts" / "checkpoints" / src.name,
                "large final checkpoint retained on server but excluded from GitHub",
            )
        )

    gitignore = CLEAN / ".gitignore"
    text = gitignore.read_text(encoding="utf-8")
    if "server_artifacts/" not in text:
        gitignore.write_text(text + "\n# server-resident large artifacts\nserver_artifacts/\n**/server_artifacts/\n", encoding="utf-8")

    registry = CLEAN / "metadata" / "external_data_registry.tsv"
    with registry.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"server_artifact\tlarge retained file\t{row['destination']}\t{row['reason']}\n")

    append_manifest(CLEAN / "inventories" / "moved_files.tsv", rows)
    print(f"Relocated {len(rows)} large server artifacts")


if __name__ == "__main__":
    main()
