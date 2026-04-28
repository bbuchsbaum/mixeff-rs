# Agent Instructions

This repository uses `mote` for local issue tracking and lightweight
coordination between agents. Treat the `.mote/` op log as the source of truth
for current work, claims, reservations, and project memory.

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
   mote preflight --issue <bd-id> --paths <path> [<path> ...]
   mote begin <bd-id> --paths <path> [<path> ...] --note "starting work"
   ```

   If preflight or begin reports a conflict, do not edit those paths until you
   inspect the owner with `mote who-has <path>` and coordinate or choose a
   non-overlapping slice.

During work:

- Keep reservations narrow. Reserve exact files for focused work; reserve
  directories only for broad changes that really need them.
- Add notes for material decisions, blockers, and progress:

  ```sh
  mote note <bd-id> --kind progress "what changed"
  mote note <bd-id> --kind decision "decision and rationale"
  mote note <bd-id> --kind blocker "what is blocked"
  ```

- If the edit scope grows, run `mote preflight` again and reserve the added
  paths before editing them.
- Respect other agents' reservations. Mote reservations are advisory, but in
  this repo they are the coordination contract.

Finishing work:

- For completed work:

  ```sh
  mote done <bd-id> --note "finished"
  ```

- For unfinished work that should continue later:

  ```sh
  mote note <bd-id> --kind progress "current state and next step"
  mote release <bd-id>
  ```

- For work handed to another actor:

  ```sh
  mote handoff <bd-id> --to <actor> --note "state and next step" --release
  ```

Repository policy:

- Do not hand-edit `.mote/ops/*.json`; publish changes through the `mote` CLI.
- Keep `.mote/` out of git unless the project explicitly decides to version
  the issue log.
- When reporting status, cite the mote issue id for active or completed work.

