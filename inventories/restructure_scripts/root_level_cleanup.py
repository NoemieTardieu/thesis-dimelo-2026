#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path("/data/leuven/383/vsc38330")
CLEAN = ROOT / "thesis_project_clean"

FIELDS = ["source", "destination", "category", "reason"]
root_rows: list[dict[str, str]] = []
moved_rows: list[dict[str, str]] = []
old_rows: list[dict[str, str]] = []
delete_rows: list[dict[str, str]] = []


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    i = 2
    while True:
        candidate = dest.with_name(f"{dest.name}.moved{i}")
        if not candidate.exists():
            return candidate
        i += 1


def append_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def record(row: dict[str, str]) -> None:
    root_rows.append(row)
    dest = Path(row["destination"])
    parts = set(dest.parts)
    if "to_delete" in parts:
        delete_rows.append(row)
    elif "OLD" in parts:
        old_rows.append(row)
    else:
        moved_rows.append(row)


def move_path(src: Path, dest: Path, category: str, reason: str) -> bool:
    if not src.exists() and not src.is_symlink():
        return False
    dest = unique_dest(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = str(src)
    shutil.move(str(src), str(dest))
    record({"source": old, "destination": str(dest), "category": category, "reason": reason})
    return True


def move_file_to(src: Path, dest_dir: Path, category: str, reason: str) -> bool:
    return move_path(src, dest_dir / src.name, category, reason)


def write_text_append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def update_gitignore() -> None:
    gitignore = CLEAN / ".gitignore"
    text = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    additions = """

# root cleanup / local runtime state
root_environment/
**/root_environment/
upstream_hyena_dna/.git/
upstream_hyena_dna/outputs/
upstream_hyena_dna/**/__pycache__/
*.sock
coder.json
serve-web-key-half
CachedProfilesData/
CachedExtensionVSIXs/
User/
Machine/
logs/
venvs/
bin/
modkit/
"""
    missing = [line for line in additions.splitlines() if line and line not in text]
    if missing:
        gitignore.write_text(text.rstrip() + "\n" + "\n".join(missing) + "\n", encoding="utf-8")


def main() -> None:
    if not CLEAN.exists():
        raise SystemExit(f"Missing clean project root: {CLEAN}")

    # Residual original preprocessing tree: generated data/reference artifacts only after first cleanup.
    move_path(
        ROOT / "thesis_dimelo",
        CLEAN / "preprocessing" / "server_artifacts" / "original_thesis_dimelo_residual",
        "server_artifact",
        "remaining original preprocessing tree; contains reference/backend/generated residuals ignored by Git",
    )

    # Move original upstream HyenaDNA clone under the clean project, then pull thesis-specific residuals out.
    upstream_dest = CLEAN / "hyena-dna" / "upstream_hyena_dna"
    move_path(
        ROOT / "hyena-dna-main",
        upstream_dest,
        "upstream_dependency",
        "upstream HyenaDNA repository preserved as project dependency/backbone source",
    )
    for name in ["DIMELO_HYENADNA_WORK_LOG.txt", "ALPHAGENOME_FOUR_CHROMOSOME_HANDOFF.md"]:
        move_file_to(
            upstream_dest / name,
            CLEAN / "hyena-dna" / "docs",
            "active_docs",
            "residual thesis documentation extracted from upstream HyenaDNA tree",
        )
    residual_work = upstream_dest / "preprocessing_chr16_merged_e5b"
    residual_outputs = residual_work / "outputs"
    for pattern in [
        "reg_given_m_crosschrom_roc_pr_curves.*",
    ]:
        if residual_outputs.exists():
            for src in sorted(residual_outputs.glob(pattern)):
                move_file_to(
                    src,
                    CLEAN / "hyena-dna" / "results",
                    "active_results",
                    "remaining final HyenaDNA ROC/PR curve output",
                )
    move_path(
        residual_work,
        CLEAN / "hyena-dna" / "to_delete" / "residual_preprocessing_chr16_merged_e5b",
        "to_delete",
        "residual generated/debug/cache files from original HyenaDNA thesis work folder",
    )
    move_path(
        upstream_dest / "outputs",
        CLEAN / "hyena-dna" / "to_delete" / "upstream_hydra_outputs",
        "to_delete",
        "upstream HyenaDNA hydra/run outputs excluded from clean active tree",
    )
    move_path(
        upstream_dest / "checkpoints",
        CLEAN / "hyena-dna" / "server_artifacts" / "upstream_checkpoints",
        "server_artifact",
        "large upstream pretrained HyenaDNA checkpoints retained on server and ignored by Git",
    )

    # Remaining AlphaGenome original tree was reduced to residual scaffolding by the first pass.
    move_path(
        ROOT / "alphagenome",
        CLEAN / "alphagenome" / "to_delete" / "residual_original_alphagenome",
        "to_delete",
        "residual original AlphaGenome tree after active code/results were moved",
    )

    # Root-level project notes and debris.
    move_file_to(
        ROOT / "Pipeline thesis.docx",
        CLEAN / "docs" / "original_notes",
        "active_docs",
        "root-level thesis note document",
    )
    move_file_to(
        ROOT / "HyenaDNA_training_&_inference_example_(Public).ipynb",
        CLEAN / "hyena-dna" / "OLD" / "notebooks",
        "OLD",
        "public HyenaDNA example notebook kept for reference",
    )
    for name in [
        "resume_modkit_pileup.sh",
        "run_modkit_merged_c1.sh",
        "--bgzf",
        "--modified-bases",
        "--reference",
        "--filter-threshold",
        "--log-filepath",
    ]:
        move_file_to(
            ROOT / name,
            CLEAN / "preprocessing" / "to_delete" / "root_modkit_command_debris",
            "to_delete",
            "root-level modkit command script/debris removed from visible workspace root",
        )
    move_path(
        ROOT / "checkpoints",
        CLEAN / "hyena-dna" / "server_artifacts" / "root_checkpoints",
        "server_artifact",
        "root checkpoint folder retained under ignored server artifacts",
    )
    move_path(
        ROOT / "modkit",
        CLEAN / "preprocessing" / "server_artifacts" / "modkit_tool_install",
        "server_artifact",
        "local modkit tool install retained under ignored server artifacts",
    )

    # Visible runtime/environment folders. Hidden runtime folders are intentionally left at root.
    for name in [
        "venvs",
        "bin",
        "logs",
        "CachedProfilesData",
        "CachedExtensionVSIXs",
        "User",
        "Machine",
        "coder.json",
        "serve-web-key-half",
    ]:
        move_path(
            ROOT / name,
            CLEAN / "server_artifacts" / "root_environment" / name,
            "runtime_environment",
            "visible root runtime/tooling state moved under ignored server artifacts for a cleaner workspace root",
        )

    update_gitignore()
    write_text_append(
        CLEAN / "metadata" / "external_data_registry.tsv",
        "\n".join(
            [
                "preprocessing\tresidual original preprocessing tree\t/data/leuven/383/vsc38330/thesis_project_clean/preprocessing/server_artifacts/original_thesis_dimelo_residual\tlarge generated/reference artifacts retained on server",
                "hyena-dna\tupstream HyenaDNA source dependency\t/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/upstream_hyena_dna\tpreserved upstream repository/backbone source",
                "hyena-dna\tupstream HyenaDNA checkpoints\t/data/leuven/383/vsc38330/thesis_project_clean/hyena-dna/server_artifacts/upstream_checkpoints\tlarge pretrained checkpoints ignored by Git",
                "preprocessing\tlocal modkit install\t/data/leuven/383/vsc38330/thesis_project_clean/preprocessing/server_artifacts/modkit_tool_install\tserver-resident tool install ignored by Git",
                "root\truntime environment archive\t/data/leuven/383/vsc38330/thesis_project_clean/server_artifacts/root_environment\tvisible root tooling state ignored by Git",
            ]
        )
        + "\n",
    )
    root_notes = """# Root Cleanup Notes

The visible workspace root was consolidated into `thesis_project_clean/`.

Moved into the clean project:

- Original residual project roots: `thesis_dimelo/`, `hyena-dna-main/`, and `alphagenome/`.
- Root-level thesis notes, notebooks, command debris, checkpoints, modkit install, logs, local tools, and editor/runtime visible folders.

Intentionally left at `/data/leuven/383/vsc38330`:

- `thesis_project_clean/`: the consolidated project.
- `code-server-ipc.sock`: active runtime socket.
- Hidden/runtime state such as `.codex`, `.agents`, `.git`, `.config`, `.cache`, and `.ondemand`, because moving these during an active Codex/code-server session could break the environment.

No files were permanently deleted. Discard candidates are quarantined under pillar-local `to_delete/` folders.
"""
    (CLEAN / "docs" / "root_cleanup_notes.md").write_text(root_notes, encoding="utf-8")

    append_manifest(CLEAN / "inventories" / "root_cleanup_manifest.tsv", root_rows)
    append_manifest(CLEAN / "inventories" / "moved_files.tsv", moved_rows)
    append_manifest(CLEAN / "inventories" / "old_manifest.tsv", old_rows)
    append_manifest(CLEAN / "inventories" / "to_delete_manifest.tsv", delete_rows)
    print(f"Root cleanup moves: {len(root_rows)}")


if __name__ == "__main__":
    main()
