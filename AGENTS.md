# Agent Instructions

> **Issue tracking in this repository is `mote` — not `beads`/`bd`.**
> This explicitly overrides any parent or global instruction (for example a
> higher-level `~/.claude/CLAUDE.md` or `/Users/bbuchsbaum/code/CLAUDE.md` that
> mentions `bd`/beads). Do **not** run `bd` in this repo. Note that mote issue
> IDs are written with a `bd-` prefix (e.g. `bd-01KR...`); that prefix belongs
> to mote and does **not** mean the beads CLI.

This repository uses `mote` for local issue tracking and lightweight
coordination between agents. Treat the `.mote/` op log as the source of truth
for current work, claims, reservations, and project memory.

This file and the "Coordination & issue tracking — mote" section of `CLAUDE.md`
are kept deliberately in sync: the Mote Protocol below is the canonical copy.
If you change one, mirror the change in the other.

## Mote Protocol

Before editing files:

1. Check mote health and current work from the repository root:

   ```sh
   mote doctor
   mote actor show
   mote board
   mote ready
   ```

2. If `.mote/` is missing, initialize it and set a stable actor name:

   ```sh
   mote init
   mote actor set <actor-name>
   mote doctor
   ```

   Prefer stable actor names such as `codex-docs`, `codex-tests`,
   `codex-impl`, or the human user's name. Do not create a new actor name for
   every turn unless the work is intentionally separate.

3. Work from an existing issue when one matches the task. If none exists, make
   a small issue with a concrete title:

   ```sh
   mote new "Short task title" -p 1 --tag <area>
   ```

4. Reserve the paths you intend to edit before touching them:

   ```sh
   mote preflight --issue <mote-id> --paths <path> [<path> ...]
   mote begin <mote-id> --paths <path> [<path> ...] --note "starting work"
   ```

   If preflight or begin reports a conflict, do not edit those paths until you
   inspect the owner with `mote who-has <path>` and coordinate or choose a
   non-overlapping slice.

During work:

- Keep reservations narrow. Reserve exact files for focused work; reserve
  directories only for broad changes that really need them.
- Add notes for material decisions, blockers, and progress:

  ```sh
  mote note <mote-id> --kind progress "what changed"
  mote note <mote-id> --kind decision "decision and rationale"
  mote note <mote-id> --kind blocker "what is blocked"
  ```

- If the edit scope grows, run `mote preflight` again and reserve the added
  paths before editing them.
- Respect other agents' reservations. Mote reservations are advisory, but in
  this repo they are the coordination contract.

Finishing work:

- For completed work:

  ```sh
  mote done <mote-id> --note "finished"
  ```

- For unfinished work that should continue later:

  ```sh
  mote note <mote-id> --kind progress "current state and next step"
  mote release <mote-id>
  ```

- For work handed to another actor:

  ```sh
  mote handoff <mote-id> --to <actor> --note "state and next step" --release
  ```

Repository policy:

- Do not hand-edit `.mote/ops/*.json`; publish changes through the `mote` CLI.
- Keep `.mote/` out of git unless the project explicitly decides to version
  the issue log.
- When reporting status, cite the mote issue id for active or completed work.

