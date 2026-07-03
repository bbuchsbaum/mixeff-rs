# RELEASE_CHECKLIST.md — mixeff-rs

Runbook for cutting a release. `<VER>` = e.g. `1.0.0-rc.1` or `1.0.0`.
Trunk-based: release off a clean `main`. Tags are signed, annotated `v<VER>`.
See `VERSIONING.md` for what bump a change requires.

## 0. Preconditions
- [ ] On `main`, `git pull`, working tree clean (`git status` empty).
- [ ] You can `git tag -s` (GPG key configured).
- [ ] `CARGO_REGISTRY_TOKEN` secret set in GitHub Actions (publish is CI-driven).
- [ ] Julia available locally with `MixedModels`, `DataFrames`, `JSON3`
      (for the local parity gate; CI also runs it on the tag).
- [ ] PRIMA C library available if running the PRIMA feature gate locally:
      `PRIMA_DIR` points to the install prefix containing `lib/libprimac`.
      If the local machine lacks PRIMA, use the Linux CI `prima` leg as the
      release evidence; `.github/workflows/ci.yml` installs PRIMA before
      running the same Cargo command.
- [ ] Supply-chain tools available: `cargo-deny` and `cargo-audit` installed
      as Cargo subcommands, or available on `PATH` as `cargo-deny` and
      `cargo-audit`.

## 1. Branch
- [ ] `git switch -c release-prep-v<VER>`

## 2. Pre-release gates (all must pass; BLOCK)
- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo clippy --all-targets --features unstable-internals -- -D warnings`
- [ ] `cargo test`
- [ ] `cargo test --features nlopt`
- [ ] `cargo test --no-default-features`
- [ ] `cargo test --features unstable-internals`
- [ ] `cargo test --no-default-features --features prima`  # requires `libprimac`
- [ ] `cargo test --release`
- [ ] `cargo +1.85 test --no-default-features`   # MSRV must RUN, not just check
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
- [ ] `cargo deny check && cargo audit`
- [ ] `cargo test --test boundary_lrt_contract --test profile_likelihood_json \
        --test public_api --test glmm_artifact_contract`
- [ ] `bash scripts/check_julia_parity_fixtures.sh`   # BLOCK on release; exits !=0 on drift
      # If drift is an intentional upstream MixedModels.jl correction:
      #   document it in CHANGELOG as a PATCH-level numeric change, then
      #   `bash scripts/check_julia_parity_fixtures.sh --accept` and commit fixtures.

## 3. Version & changelog bump
- [ ] Set `version = "<VER>"` in Cargo.toml.
- [ ] `cargo update -p mixeff-rs`   # refresh own Cargo.lock entry only
- [ ] CHANGELOG.md: rename `## [Unreleased]` -> `## [<VER>] - <YYYY-MM-DD>`,
      add fresh empty `## [Unreleased]`.
- [ ] Add/refresh compare links at bottom of CHANGELOG.md.
- [ ] Wire-contract / numeric-parity callouts added if any schema_version or
      fitted value moved (see VERSIONING.md §2.B / §2.D).
- [ ] (FINAL 1.0.0 only) README.md install snippet -> `mixeff-rs = "1.0"`.
      (RC: leave README at the prior stable string.)

## 4. Package verification
- [ ] `cargo publish --dry-run`
- [ ] `cargo package --list | sort`  -> confirm NO `MixedModels.jl/`,
      top-level `docs/*.md`, `scripts/`, `tests/`, `.github/`, `audit/`,
      `=`, or `halving_bound`; confirm `docs/guide/` is included.
- [ ] `.crate` size sanity (KBs–low MB, not the Julia tree).

## 5. Commit, PR, merge
- [ ] `git commit -am "release: v<VER>"`
- [ ] Push branch, open PR, confirm full CI green.
- [ ] Squash-merge to `main`. `git switch main && git pull`.

## 6. Tag
- [ ] `git tag -s v<VER> -m "mixeff-rs v<VER>"`
- [ ] `git push origin v<VER>`
      # The `v*` tag triggers .github/workflows/release.yml:
      #   gates + REQUIRED julia-parity job -> publish job (cargo publish).

## 7. Verify publication
- [ ] crates.io shows `<VER>` (RC will not be picked by `"1.0"` reqs — expected).
- [ ] docs.rs build for `<VER>` succeeds.
- [ ] GitHub Release created from tag:
      - body = CHANGELOG section for <VER>
      - RC: mark "pre-release", add 2-week soak feedback ask.
      - FINAL: not pre-release.

## 8. RC soak (RC only) — min 2 weeks
- [ ] Weekly scheduled julia-parity job green at least once on this tree.
- [ ] mixeff R package builds against `mixeff-rs = "=<VER>"`.
- [ ] No API-breaking feedback. Any breaking fix -> new -rc.(N+1), reset clock.
- [ ] To finalize: rerun this checklist with `<VER>` = `1.0.0`.

## 9. Post-release
- [ ] Announce in mixeff R / Python repos; bump their pins
      (RC: `=<VER>`; after 1.0.0: `"1.0"`).
- [ ] If a shipped version is harmful: publish the fix version FIRST,
      then `cargo yank --version <bad>`, add `### Yanked` CHANGELOG note,
      edit the GitHub Release explaining why.
- [ ] Close the mote release issue; note the tag in the issue.
