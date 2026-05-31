# Audit 07/07 — Optimizers, Error Model, Public API & Release Readiness

**Crate:** `mixeff-rs` @ 0.1.0  **Auditor:** 7/7 (RC hardening)  **Date:** 2026-05-18
**Scope:** `src/optimizer/*`, `src/error.rs`, `src/lib.rs`, `src/main.rs`, `Cargo.toml`,
`examples/`, plus a cross-cutting sweep of all of `src/` for `unwrap/expect/panic/
unreachable/todo/unimplemented`, `unsafe`, integer-overflow and truncating `as` casts.

**Build evidence collected**
- `cargo clippy --all-targets --all-features` → exit 0, **2 warnings** (one real
  lint, duplicated lib + lib-test). No errors.
- `cargo build --examples` (default features) → exit 0. Documented parity/bench
  examples compile against the current API.
- `unsafe` blocks: 12, all in `src/optimizer/prima.rs` (PRIMA C FFI). No `unsafe`
  elsewhere in `src/`.
- Reachable `todo!`/`unimplemented!`: **none**. `unreachable!` in library paths:
  all guard-dominated (verified, see HIGH/LOW below).

---

## Severity summary

| Severity | Count | Items |
|---|---|---|
| CRITICAL | 0 | — |
| HIGH | 2 | H1 nlopt `maxeval as u32` unguarded overflow; H2 `LinearMixedModel` raw public mutable fields |
| MEDIUM | 4 | M1 clippy lint shipped; M2 `optsum` public field / convergence not gated by default; M3 `approx` dep duplicated; M4 prima `MaybeUninit::zeroed().assume_init()` zero-validity reliance |
| LOW | 5 | L1 `nlopt` heavy default feature; L2 lossy error stringification at FFI boundaries; L3 prima `ftarget`/`npt` zeroed; L4 trust_bq NaN-on-first-eval; L5 minor naming drift |

No CRITICAL or HIGH issue is HIGH-confidence-blocking on correctness of a *default*
fit; the HIGH items are release-hygiene / robustness gaps. **Recommendation:
REQUEST CHANGES** for H1 + H2 before tagging 0.1.0 (both are cheap), everything
else is COMMENT-level.

---

## HIGH

### H1 — `nlopt` `maxeval as u32` is an unguarded narrowing cast (premature-stop risk)
**File:** `src/model/linear.rs:5492-5493`
**Confidence:** HIGH (defect is real) / MEDIUM (real-world trigger is unlikely)

```rust
if maxeval > 0 {
    Self::nlopt_ok(opt.set_maxeval(maxeval as u32), "set_maxeval")?;
}
```

`maxeval: usize` derives from `self.optsum.max_feval` (an `i64`, public/caller
settable) via `max_feval as usize`. The PRIMA path correctly guards this:
`src/optimizer/prima.rs:159` rejects `options.maxfun > c_int::MAX as usize`. The
NLopt path has **no equivalent guard**. A caller setting `max_feval` above
`u32::MAX` (~4.29e9) wraps silently: e.g. `4_294_967_300` → `4`, so NLopt stops
after 4 evaluations and `fit()` returns `Ok` with a near-initial θ and
`return_value = "MAXEVAL_REACHED"`. A non-convergent fit masquerading as a
completed fit is exactly the failure mode this audit targets.

**Repro:** `optsum.max_feval = (u32::MAX as i64) + 4;` then fit a model routed to
an NLopt optimizer.
**Fix:** Mirror the PRIMA guard — if `maxeval > u32::MAX as usize` return
`MixedModelError::Optimization("NLopt max_feval exceeds backend limit")`, or
saturate with `maxeval.min(u32::MAX as usize) as u32` (saturation is acceptable
here since u32::MAX evals is effectively "unbounded").

### H2 — `LinearMixedModel` exposes fitted state as raw public mutable fields
**File:** `src/model/linear.rs:15-26` (`pub formula`, `pub reterms`, `pub y`,
`pub dims`, `pub optsum`), confirmed `pub optsum: OptSummary` at `:95`
**Confidence:** HIGH

`pub formula`, `pub reterms: Vec<ReMat>`, `pub y: DVector<f64>`, `pub dims`, and
`pub optsum: OptSummary` are directly mutable on a fitted model. A consumer can
do `model.y[0] = …` or `model.optsum.fmin = 0.0` *after* `fit()` and every
derived quantity (`vcov`, `coef`, `ranef`, the Cholesky blocks) is now silently
inconsistent — no invariant re-checks. For a 1.0-track public type this is a
SemVer trap: these fields can never change shape without a major bump, and they
defeat the crate's own "no hidden model surgery / explicit refusal" inference
contract (`docs/mixed_model_compiler_inference_contract.md`) because callers can
perform *visible* model surgery the type cannot detect.

**Fix:** Demote to `pub(crate)` and add read-only accessors
(`fn formula(&self) -> &Formula`, `fn opt_summary(&self) -> &OptSummary`,
`fn response(&self) -> &DVector<f64>`). `optsum` already has `#[non_exhaustive]`
on the *struct*, which is good, but the *field* being `pub` undoes the benefit.
This is the single largest 0.1.0 API-hygiene liability in scope.

---

## MEDIUM

### M1 — Clippy lint shipped in library code
**File:** `src/optimizer/prima.rs:235`
**Confidence:** HIGH
```
warning: manual `!Range::contains` implementation
235 | if minimize_rc >= 100 || minimize_rc < 0 {
    help: use: `!(0..100).contains(&minimize_rc)`
```
Only outstanding clippy warning (counted twice: lib + lib-test). A 0.1.0 RC
should ship clippy-clean. **Fix:** apply the suggestion, or add a scoped
`#[allow(clippy::manual_range_contains)]` with a one-line rationale consistent
with the crate's documented allowlist policy in `lib.rs:56-79` (the existing
policy explicitly governs *optimizer boundary logic*, so a local allow here is
defensible — but it should be explicit, not a raw warning).

### M2 — Convergence status is not gated by default; only discoverable via public field or opt-in API
**File:** `src/model/linear.rs:4465`, `:5541`, `:5197`, `verify_convergence` at `:1206`
**Confidence:** MEDIUM

`fit()` returns `Ok` when an optimizer hits its evaluation budget. The true
status is faithfully preserved (`optsum.return_value` ∈ {`FTOL_REACHED`,
`MAXEVAL_REACHED`, `XTOL_REACHED`, …} — confirmed no `MAXEVAL_REACHED` is
rewritten to `SUCCESS`; `finalize_fit_result` only defaults to `"SUCCESS"` when
the driver passed `None`, and the NLopt/COBYLA/TrustBQ drivers always pass a
concrete label). This matches lme4/MixedModels.jl semantics (warn, don't error)
and there is a solid opt-in `verify_convergence_with_options` (restart / jitter /
optimizer-consensus). **However**: the only non-opt-in way a caller learns the
fit hit max-eval is by reading the *public field* `optsum.return_value: String`
and string-matching `"MAXEVAL_REACHED"`. There is no
`model.converged() -> bool` / typed status accessor on the public surface.
Combined with H2, downstream R bindings will end up parsing a `String`.
**Fix (release-blocking-adjacent, but COMMENT given the contract):** add a
typed `pub fn optimizer_return(&self) -> OptimizerReturn` (or
`converged(&self) -> bool`) so the stable contract is an enum, not a string
the docstring at `error.rs:75` already warns against parsing.

### M3 — `approx` declared in both `[dependencies]` and `[dev-dependencies]`
**File:** `Cargo.toml:41` and `Cargo.toml:68`
**Confidence:** HIGH

`approx = "0.5"` is a normal dependency *and* a dev-dependency (identical
version). The dev-dep line is dead — Cargo already makes normal deps available
to tests/examples. Harmless functionally but it is exactly the "dev-deps leaking
/ duplicated dependency metadata" smell a 0.1.0 RC review flags. Verify `approx`
is genuinely used in non-test library code (a numeric-parity crate plausibly
uses it in `#[cfg(test)]` only — in which case it should be *dev-only*, not a
normal dep, to keep the published dependency graph minimal). **Fix:** remove the
redundant `[dev-dependencies]` entry; if `approx` is test-only, move it there and
drop the normal-deps entry.

### M4 — PRIMA FFI structs built via `MaybeUninit::zeroed().assume_init()`
**File:** `src/optimizer/prima.rs:169, 189, 203`
**Confidence:** MEDIUM (sound *in practice*, fragile *in principle*)

```rust
let mut problem = unsafe { std::mem::MaybeUninit::<PrimaProblem>::zeroed().assume_init() };
```
`PrimaProblem`/`PrimaOptions`/`PrimaResult` contain raw pointers (null on zero —
valid), `Option<extern "C" fn>` (None on zero — valid), `c_double` (0.0 — valid),
and `c_int` (0 — valid). The one genuinely sharp edge: `PrimaOptions.iprint:
PrimaMessage` and `PrimaProblem` has no enum that lacks a 0 discriminant in the
problem struct, but `PrimaAlgorithm` (discriminant 2, no 0 variant) is **not** a
field of either zeroed struct — it is passed by value at `:205` after explicit
construction, so no invalid-enum UB. `PrimaMessage::None = 0` so the zeroed
`iprint` is a valid variant. Net: currently sound, but it relies on every field
being zero-valid and is re-validated only by the subsequent `prima_init_problem`
/ `prima_init_options` calls (return codes are checked — good). This is the
documented PRIMA C-API initialization handshake, so it is acceptable, but each
struct should carry a `// SAFETY:` comment enumerating the zero-validity argument
(all-zero = null ptrs + None fn + 0.0 + PrimaMessage::None, then C-side init
populates required fields, rc checked). **Fix:** add the SAFETY comments; no code
change required. (`prima` is off by default and the gate-feature is documented as
requiring a system C lib, which lowers release exposure.)

---

## LOW

### L1 — `default = ["nlopt"]` makes a C/Fortran dependency the default build
**File:** `Cargo.toml:50, 44`
`nlopt 0.8.1` (optional) is enabled by default, pulling a C/Fortran NLopt build
into the default `cargo add mixeff-rs` experience. This is *documented*
(`Cargo.toml:51-55`, with `--no-default-features` escape hatch using TrustBQ),
and is a deliberate parity/performance choice — acceptable, but for a 0.1.0 first
release consider whether the lighter native `trust_bq` path should be the
default to keep first-build friction low. Confidence HIGH on the fact,
LOW that it is wrong (design call). No action required if documented in README.

### L2 — Lossy stringification at FFI / optimizer error boundaries
**File:** `src/error.rs:16` (`Optimization(String)`), `linear.rs:4351, 5510, 5168`
The internal error model is good: `MixedModelError` is `#[non_exhaustive]`,
`#[from] FormulaError`/`LinAlgError` preserve source via `thiserror`, and
`code()`/`LinAlgError::code()` give a stable machine contract with a test
(`error.rs:118`). Refusal/identifiability errors are distinct & catchable
(`RankSaturatedFixedEffects`, `NoRandomEffects`, `UnsupportedFamilyLink`,
`Singular`, `PosDefException`, `ConstantResponse` — all separate variants with
stable codes). The only lossiness is *at the boundary*: NLopt/COBYLA/PRIMA
failure states are folded into `Optimization(String)` via
`format!("…{status:?}")` (`linear.rs:4351-4352`, `prima.rs:240`). The underlying
typed backend status is discarded. Low impact because the failure is still
catchable as `Optimization` with the textual detail, but a downstream client
cannot branch on "backend ran out of memory" vs "roundoff limited" without
substring matching — the same anti-pattern `error.rs:75` explicitly warns
against, applied to optimizer status. **Fix (optional):** an
`Optimization { backend: &'static str, status: OptimizerFailKind }` structured
variant. Defer past 0.1.0.

### L3 — PRIMA `ftarget` / `npt` / `ctol` left at zeroed defaults
**File:** `src/optimizer/prima.rs:188-200`
`prima_init_options` is called (return checked) then only `rhobeg`, `rhoend`,
`maxfun`, `iprint`, `data` are overwritten. `ftarget`, `npt`, `ctol`,
`callback` keep whatever `prima_init_options` set (the C side initializes them —
acceptable since the rc is checked) — but `ftarget` after a *successful* C init
is PRIMA's documented sentinel (`-Inf`), not the zeroed `0.0`; this is fine
because `prima_init_options` overwrites the zeroed struct. Flagging only so a
maintainer knows the zeroing at `:189` is immediately superseded by the C init —
no defect, document the ordering.

### L4 — TrustBQ errors out if the *first* objective evaluation is non-finite
**File:** `src/optimizer/trust_bq.rs:224, 789-797`
`minimize_with_progress` evaluates the start point and `evaluate()` returns
`Err(Optimization("…non-finite value"))` if it is NaN/Inf. NLopt/COBYLA paths
instead substitute a sentinel (`unwrap_or(invalid_objective)` /
`unwrap_or(f64::INFINITY)`, `linear.rs:5083, 5422`) and keep searching. So an
identical model that fails the very first θ probe errors hard under TrustBQ
(`--no-default-features`) but is gracefully recovered under the default NLopt
build. This is a *determinism / cross-backend-consistency* gap: the same data
can fit on the default build and error on the dependency-light build. Confidence
MEDIUM. **Fix:** make TrustBQ's first-point handling match the other backends
(treat non-finite as `+Inf` and let the trust region move away), or document
that `--no-default-features` has stricter start-point requirements.

### L5 — Minor public-surface naming drift
**File:** `src/model/mod.rs:44-55`, `src/lib.rs:188-194`
The `linear` re-export list mixes `snake_case` free fn `parametricbootstrap`
(no underscores) with `Builder`/`Options` PascalCase types and SCREAMING consts
(`BOOTSTRAP_RUN_SCHEMA`). `parametricbootstrap` reads as a port-ism of Julia's
`parametricbootstrap`; for a Rust 0.1.0 public fn, `parametric_bootstrap` is the
idiomatic spelling and renaming is free now, breaking later. The `prelude`
(`lib.rs:187`) is tight and correct. Confidence MEDIUM, cosmetic.

---

## Cross-cutting sweep results (clean / commendations)

- **`todo!`/`unimplemented!`:** none reachable anywhere in `src/`.
- **`unreachable!` in library paths:** `formula/parser.rs:544,724` are dominated
  by an immediately-preceding `peek()` match on the same token variant — sound.
  `model/linear.rs:8224` (`"column name came from this frame"`) and the
  `generalized.rs:646` (`"dispersion families refused above"`) are
  invariant-documented and refusal-guarded. No reachable panic path found in a
  public happy/refusal flow. The remaining `panic!`/`unreachable!` hits are all
  inside `#[cfg(test)]` modules.
- **`unwrap()`/`expect()`:** ~42 pre-test-module occurrences; spot-checked the
  hot ones — `linalg/*` entirely test-module; `linear.rs:6033/6050/9799/13343`
  are `Normal::new(0.0, 1.0).unwrap()` / `Normal::new(0.0, sigma)` with σ proven
  positive, and `partial_cmp(...).unwrap()` on values filtered to finite first
  (`linear.rs:9530` is `finite.sort_by`). No reachable `unwrap` on
  attacker/caller-controlled non-finite input found in scope. `linear.rs:6008`
  `.expect("fitted beta should match active fixed-effect design")` is an
  internal post-fit invariant.
- **`unsafe`:** 12 blocks, 100% in `prima.rs` FFI, all behind the off-by-default
  `prima` feature. Panic safety is correct: `objective_trampoline`
  (`prima.rs:116-136`) wraps the user closure in
  `catch_unwind(AssertUnwindSafe(...))` as the *first* operation and converts a
  panic to `NAN` + a `state.panicked` flag that is re-checked at `:230` — no
  unwinding across `extern "C"`. Null pointers checked before deref
  (`:120`, `:207`, `:216`). Result freed via `prima_free_result` (`:227`). This
  is careful FFI; only M4/L3 documentation gaps remain.
- **Integer/index casts:** only H1 (`maxeval as u32`) is an unguarded
  size-math narrowing reachable from caller input. `data.rs:243`
  `levels.len() as u32` and `linear.rs:8740 idx as u32` for categorical refs
  would only wrap at >4.29e9 factor levels — not a realistic RC concern (LOW,
  folded into commentary). PRIMA `n as c_int` / `maxfun as c_int` are guarded
  (`prima.rs:159`).
- **Error model:** `#[non_exhaustive]` on both `MixedModelError` and
  `LinAlgError`; `From` impls match the documented `?`-propagation contract;
  identifiability/refusal errors are distinct, catchable variants with a tested
  stable `code()` string contract. This is a strong, well-designed error layer.
- **Public enum hygiene:** `Family`, `LinkFunction`, `Optimizer`,
  `OptimizerBackend`, `OptSummary` all carry `#[non_exhaustive]` — correct for
  types expected to grow. The crate has a deliberate, documented
  `unstable-internals`-gated split (`compiler`/`datasets`/`pathology`) and
  demotes `linalg` to `pub(crate)` — mature SemVer thinking.
- **No `dbg!` in library paths.** `eprintln!` occurrences are either in
  `#[cfg(test)]`, behind a debug-print path in `generalized.rs:833` (PIRLS
  iteration trace — gated by a verbose flag, verify it is not unconditional),
  or in `datasets` iteration (unstable-internals only). Worth a 1-line check
  that `generalized.rs:833` is flag-gated.

## Commendations

1. **Error layer is exemplary** — non-exhaustive, source-preserving `#[from]`,
   stable machine `code()` with a regression test, and a docstring that
   explicitly tells clients not to parse `Display`. This is the right contract.
2. **TrustBQ termination model is honest** — six distinct stop reasons,
   `is_acceptable_convergence()` cleanly separates `BudgetExhaustion` from real
   convergence, the stagnation early-stop is well-documented and unit-tested
   including the exact-memoization "must not perturb the path" property.
3. **`MAXEVAL_REACHED` is never laundered into `SUCCESS`** — verified across all
   four drivers; budget exhaustion is faithfully recorded.
4. **PRIMA FFI panic safety is done correctly** — `catch_unwind` first,
   `state.panicked` re-checked, null guards, result freed.
5. **Deliberate SemVer engineering** — `unstable-internals` feature gate,
   `pub(crate)` demotion of `linalg`, `#[non_exhaustive]` on all growable enums,
   re-exported `nalgebra`, documented clippy-allow policy.
6. Examples and benches compile against the current API; the unstable examples
   are correctly gated behind `required-features`.

## Verdict

**REQUEST CHANGES** (low-cost, pre-tag): fix **H1** (guard `maxeval as u32`),
**H2** (demote `LinearMixedModel` public fields + add accessors), and **M1**
(clear the last clippy warning). **M2/M3/M4** are strongly recommended but not
strictly release-blocking. Everything else is COMMENT-level / post-0.1.0.
No CRITICAL issues; the optimizer correctness core (termination criteria,
max-iter honesty, NaN handling, bound projection, determinism) is sound.
