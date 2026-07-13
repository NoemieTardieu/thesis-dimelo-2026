# Root Cleanup Notes

The visible workspace root was consolidated into `thesis_project_clean/`.

Moved into the clean project:

- Original residual project roots: `thesis_dimelo/`, `hyena-dna-main/`, and `alphagenome/`.
- Root-level thesis notes, notebooks, command debris, checkpoints, modkit install, logs, local tools, and editor/runtime visible folders.

Intentionally left at `/data/leuven/383/vsc38330`:

- `thesis_project_clean/`: the consolidated project.
- `code-server-ipc.sock`: active runtime socket.
- Hidden/runtime state such as `.codex`, `.agents`, `.git`, `.config`, `.cache`, and `.ondemand`, because moving these during an active Codex/code-server session could break the environment.

No files were permanently deleted. Discard candidates are quarantined under pillar-local `to_delete/` folders.
