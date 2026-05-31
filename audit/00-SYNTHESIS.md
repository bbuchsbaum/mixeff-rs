# mixeff-rs — Release-Candidate Hardening Synthesis

- **Commit audited:** `d516a1c` (working tree has the in-progress KR change: `src/model/linear.rs`, `docs/kenward_roger_contract.md`, `tests/fixtures/compiler_contract/kenward_roger_pbkrtest_parity_v1.json`)
- **Scope:** 7 parallel opus auditors over all of `src/` + docs/contracts + Cargo/examples
- **Baseline:** `cargo build --tests` ✅ · `cargo clippy --all-targets` ✅ (2 trivial lints) · 821 tests, `stats::` 101/101, KR 13/13
- **Per-area reports:** `audit/01..07-*.md`

## Verdict: **DO NOT TAG THE RC YET.**

The crate is structurally strong (faithful Julia ports, honest error layer, real refusal paths, exemplary SemVer engineering in `error.rs`/linalg demotion). But the audit found **2 release-blocking defects** and a **systemic honesty gap** that directly contradicts the project's own "no fake numbers / no hidden surgery" inference contract. None are large fixes.

---

## Cross-cutting theme (the important finding)

Three independent auditors (02 LMM, 03 GLMM, 07 optimizer) found the **same class of bug**: a non-converged / non-finite / max-eval-truncated fit is finalized and returned as `Ok` with the failure recorded only in a *string* (`optsum.return_value`), with **no typed `converged()` accessor** at the trait boundary. For a parity-focused crate whose contract forbids fake certainty, "silently ship a bad fit" is the highest-leverage thing to fix. Treat 02·HIGH-1/2, 03·H1, 07·H1, 07·M2 as one workstream.

---

## BLOCKERS (must fix before RC)

| ID | Sev | Area | Issue | Fix |
|----|-----|------|-------|-----|
| **B1** | CRITICAL | 03 GLMM | `loglikelihood() = -objective()/2` uses the **dropped-constant** Laplace deviance on the default fast path. Every default-path Poisson/Binomial GLMM reports **AIC/BIC/logLik offset** by `2·Σln(yᵢ!)` / `2·Σln C(nᵢ,kᵢ)` vs lme4/Julia. A correct full-likelihood helper (`minus_two_loglik_observation`, `generalized.rs:984`) already exists but is only wired to the nlopt joint path. | Route `loglikelihood()` through the full normalized-density helper + RE penalty on all paths; add a Poisson/Binomial AIC parity test vs lme4. |
| **B2** | CRITICAL | 06 formula | A valid-but-pathological formula (`y ~ I((((…)))) + (1\|g)`, deep `sqrt(sqrt(…))`) **overflows the stack and aborts the process** — uncatchable by `catch_unwind`, so a host wrapper passing untrusted/typo'd formulas cannot defend. Unbounded recursion in `transform.rs` `parse_expr/parse_primary/canonical_label/eval_row`. | Enforce a paren-depth / expression-size budget in `TParser`; return `FormulaError` instead of recursing. |

## HIGH (fix before RC, or consciously accept + document/track)

| ID | Area | Issue |
|----|------|-------|
| **H-conv** (02·HIGH-1, 03·H1, 07·H1, 07·M2) | LMM/GLMM/opt | Non-convergence returns `Ok`; PIRLS accepts non-converged modes silently (Julia's `iter<2→throw` not ported); `nlopt maxeval as u32` narrows silently → optimizer stops early but `fit()`=`Ok`. No typed `converged()`. **Fix together:** add typed convergence status to `MixedModelFit`; port PIRLS divergence guard; saturate the `maxeval` cast (mirror `prima.rs:159`). |
| **02·HIGH-2** | LMM | Non-finite initial objective not rescued/checked → loop finalizes at bad initial θ, returns `Ok` (Julia rescales, `linearmixedmodel.jl:480-491`). |
| **03·H2** | GLMM | Gamma deviance can go NaN/negative from valid inverse-linked input; NaN passes the step-halving guard and is accepted. Bound `mu`; reject non-finite deviance. |
| **03·H3** | GLMM | AGQ contract guard is `debug_assert!` — a **no-op in release**. Promote to a hard runtime check. |
| **05·H1** (= 03·M3 dup) | stats | `stats::bootstrap::shortest_cov_int` (public, re-exported) `partial_cmp().unwrap()` **panics on NaN** — and its documented inputs (`boot.objectives()`/`sigmas()`) deliberately contain NaN for failed refits. Trim non-finite like Julia `shortestcovint`. |
| **06·H1/H2** | pathology | `certify()` (documented "total, pure linear algebra") **panics** via `assert!`/index-out-of-bounds on shape-mismatched-but-valid specs. Validate spec shape up front; return errors. |
| **04·HIGH-1** | compiler | `kenward_roger_contract.md` claims crossed/nested KR "certified", but on the non-default native build the new scalar Penicillin/Pastes rows are skipped (`linear.rs:18650-18653`). Scope the doc claim to the NLopt path, or remove the native filter. |
| **07·H2** | API | `LinearMixedModel` fitted state is raw `pub` mutable (`formula/reterms/y/dims/optsum`) — post-fit mutation desyncs all derived quantities; permanent 0.1.0 SemVer trap + violates "no hidden surgery". Demote to `pub(crate)` + read-only accessors **(do this before the version is frozen).** |

## Notable MEDIUM (recommended pre-RC, cheap)

- **01·H1/M1** `pivot.rs:257-295` `quick_full_rank_identity_pivot`: Rust-only n≤2 rank shortcut with 100× inflated tolerance + `sqrt(eps)` floor on the **live** `compiler::audit`/`lrt`/`FeTerm` route → can misreport rank, changing LRT df & identifiability gating. Either prove it never looser than `compute_rank_from_r` or delete it; add fuzz parity test.
- **07·M1** clippy `manual_range_contains` `prima.rs:235`; **07·M3** duplicate `approx` dep (normal + dev).
- **04·MEDIUM-1 / 02-note**: working-tree KR parity-band loosening is legitimate (Auditor 04 independently regenerated the fixture in R and it matches to full precision — **fixture is regenerated, not hand-edited**), but the single-row branch lacks the drift-justifying comment its multi-df sibling has. Add the comment before committing.
- **05·M1/M2** profile-CI silent spline extrapolation past the computed grid; plain `LikelihoodRatioTest` has no boundary caveat (sound variants exist).

## Confirmed-clean / commendations (do not "fix")

Estimability enforced before any p-value; explicit KR never silently degrades; θ-map round-trips deterministically; VarCorr σ-cancellation **exact** vs Julia; boundary/parametric-bootstrap LRT statistically honest; GLMM fast-vs-joint divergence **is** surfaced (`uncertified_joint_fallback` + `documented_divergence`) — the old project-memory concern is addressed; `error.rs` `#[non_exhaustive]`+`#[from]` exemplary; PRIMA FFI panic-safety correct; no reachable `todo!`/`unimplemented!`; no `unsafe` in the numerical core.

## Recommended sequencing

1. **RC-blocking:** B1, B2, H-conv bundle, 07·H2 (API freeze), 06·H1/H2.
2. **Same PR-set, cheap:** 02·HIGH-2, 03·H2, 03·H3, 05·H1, 04·HIGH-1 doc scope, 01·H1, 07·M1/M3, KR comment.
3. **Track post-0.1.0:** remaining MEDIUM/LOW across reports.
