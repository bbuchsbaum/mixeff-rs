# Audit 05/07 — Post-fit Statistics (`src/stats/*.rs`)

**Auditor:** 5/7 (release-candidate hardening pass)
**Scope:** `src/stats/{varcorr,coeftable,model_summary,block_description,lrt,bootstrap,profile,spline,mod}.rs`
**Method:** real call-path tracing; formulas cross-checked against `MixedModels.jl/src/{varcorr.jl,mixedmodel.jl,utilities.jl,likelihoodratiotest.jl,bootstrap.jl}`. `cargo test --lib stats::` → 101 passed / 0 failed.
**Read-only:** no source modified.

---

## Severity summary

| Sev | Count | Items |
|-----|-------|-------|
| CRITICAL | 0 | — |
| HIGH | 1 | H1 public `shortest_cov_int` panics on NaN replicate input |
| MEDIUM | 3 | M1 profile-CI silent spline extrapolation past computed ζ grid; M2 plain `LikelihoodRatioTest` carries no boundary caveat for variance-component tests; M3 `shortest_cov_int` window math diverges from Julia (no finite-trim) → wrong interval if non-finite present |
| LOW | 2 | L1 `quantile_sorted` clamps probability semantics differ at exact 0/1 endpoints vs Julia type-7; L2 `confint` `swap` can mask a non-monotone reverse spline |
| COMMEND | 4 | see end |

No CRITICAL/HIGH issue at HIGH confidence blocks the *internal* fitted-model→summary path; the HIGH item is a public-API footgun reachable with documented inputs. Verdict: **RELEASE-ACCEPTABLE with H1 fixed or documented**.

---

## HIGH

### H1 — `bootstrap::shortest_cov_int` panics on NaN (failed-refit replicate values)
**File:** `src/stats/bootstrap.rs:137-155` (exported at `src/stats/mod.rs:30`)
**Confidence:** HIGH (reproduced).

`shortest_cov_int(v, level)` sorts via `v.sort_by(|a,b| a.partial_cmp(b).unwrap())`. `partial_cmp` returns `None` for any NaN, so `.unwrap()` panics. This is a *public* exported helper documented as "Mirrors the `shortestcovint` summary helper used by the Julia bootstrap surface."

The parametric-bootstrap surfaces deliberately record `f64::NAN` for refits that fail numerically (`src/model/linear.rs:13304-13309`, `:6916-6921`; `stats/bootstrap.rs:119-124`). The natural call is `shortest_cov_int(&mut boot.objectives(), 0.95)` / `&mut boot.sigmas()` etc. — those vectors contain NaN whenever any replicate failed (the common case for hard models), so the documented usage panics.

Reproduced standalone: `vec![1.0, NaN, 2.0, 3.0, 4.0]`, level 0.9 → thread panic `called Option::unwrap() on a None value`.

Contrast: Julia `shortestcovint` (`MixedModels.jl/src/bootstrap.jl:471-484`) explicitly trims non-finite ends via `findfirst(isfinite, vv)` / `findlast(isfinite, vv)`. The crate's *internal* `MixedModelBootstrap::{percentile,shortest}_intervals` path is safe because `parameter_series()` (`linear.rs:13006-13045`) filters `is_finite()` first — but the standalone exported helper does not, and it is the one named in the public re-export and rustdoc.

**Fix:** trim/skip non-finite values before sorting (match Julia's `findfirst/findlast(isfinite,…)`), or sort with `total_cmp` and then exclude the NaN tail before window scanning. Returning `(NaN, NaN)` or an `Err` on all-non-finite is acceptable; a panic from a documented public helper on documented inputs is not for a release candidate.

---

## MEDIUM

### M1 — Profile-likelihood CI silently extrapolates past the computed ζ grid → fabricated bounds
**File:** `src/stats/profile.rs:140-176` (`MixedModelProfile::confint`), interacting with `spline.rs:119-154` linear-tail extrapolation and the profile walkers `profile_sigma:418-460`, `profile_theta:591-626`, `profile_beta:893-914`.
**Confidence:** MEDIUM.

`confint` reads the interval as `spline.eval(±cutoff)` on the reverse (ζ→param) spline, `cutoff = Φ⁻¹(0.5+level/2)` (up to ≈3.29 at level 0.999, ≈2.58 at 0.99). `NaturalCubicSpline::eval` extrapolates **linearly with no error/flag** when the query is outside the knot range (`spline.rs:124-135`).

The profile walkers stop at `|ζ| ≥ threshold` (default 4.0 via `profile()`), *or earlier* on a boundary/iteration cap:
- `profile_sigma`: negative side breaks when `sigma_v <= sigma_min = 1e-6·σ̂` (σ→0) — ζ can stop far short of −4.
- `profile_theta`/`profile_beta`: negative side breaks at the θ lower bound (variance component → 0), often at |ζ|≈1–2.

When the requested `cutoff` exceeds the achieved |ζ| range on a side, the returned CI endpoint is produced by **linear extrapolation of the spline tail**, i.e. a confidence bound that no conditional refit ever supported. The only guards in `confint` are (a) `lower>upper` swap, (b) nonnegative-boundary clamp to 0, (c) an `Err` if the interval fails to bracket the estimate. None of these detects "cutoff outside computed ζ span". This violates the project's no-fake-statistics stance for the boundary case.

Mitigations already present: natural-spline tails are *linear* (bounded, not divergent); `profile()` always uses threshold 4.0 which covers 95–99% for well-behaved parameters; the exact-zero clamp + `boundary_clamped_lower` flag covers the most common variance-component-at-zero case. Hence MEDIUM, not HIGH.

**Fix:** record per-parameter achieved `[ζ_min, ζ_max]`; if `±cutoff` falls outside, either return a typed refusal ("profile did not reach the requested level; increase threshold") or mark the row's `regularity` as `extrapolated_beyond_profile_grid` and set a non-finite/`None` bound rather than emitting an interpolated-looking number.

### M2 — Plain `LikelihoodRatioTest` reports a naive χ²(Δdf) p-value with no boundary caveat for variance-component comparisons
**File:** `src/stats/lrt.rs:758-797` (`test_with_formulas`), struct `LikelihoodRatioTest:205-225`.
**Confidence:** MEDIUM (statistical-correctness caveat, matches Julia behavior).

For a `NestedRandomEffects` / `SameFixedEffectsCovarianceDifference` comparison the added parameter is a variance/covariance term on the boundary of its space. The plain path computes `pval = 1 - ChiSquared(ddof).cdf(chi)` (`lrt.rs:778-783`) — the textbook anti-conservative result; the true reference is a χ̄² mixture (Self & Liang). The `LikelihoodRatioTest` struct has **no** boundary note/flag (confirmed: no `notes`/`caveat` field; only the separate `BoundaryLikelihoodRatioTest` carries the mixture note at `:558-559`).

The crate *does* provide the correct routes — `BoundaryLikelihoodRatioTest` (50:50 χ²₀:χ²₁, `self_liang_one_parameter_pvalue` `:1490-1497`, correctly refuses >1 boundary param `:536-538`) and the boundary-robust `parametric_bootstrap_lrt` (`:1643-1718`, sound: separate failure counting, `(1+extreme)/(completed+1)`, refuses all-failed). But a user calling the basic `LikelihoodRatioTest::test` on `(1|g)` vs `(1+x|g)` gets an unflagged anti-conservative p-value. This mirrors `MixedModels.jl/likelihoodratiotest.jl` (Julia also emits plain χ² there), so it is a domain caveat rather than a regression — but for an RC the absent caveat is worth a doc/field note steering users to the boundary route.

**Fix (non-breaking):** add an advisory note when an adjacent comparison is classified `NestedRandomEffects`/`SameFixedEffectsCovarianceDifference`, pointing at `BoundaryLikelihoodRatioTest`/`parametric_bootstrap_lrt`. No numeric change required.

### M3 — `shortest_cov_int` window count diverges from Julia even on finite input
**File:** `src/stats/bootstrap.rs:141-154` vs `MixedModels.jl/src/bootstrap.jl:474-483`.
**Confidence:** MEDIUM.

Julia computes the window over `start:(stop+1-ilen)` where `start/stop` are the finite-value bounds and has the early-out `if stop < start+ilen-1 return (vv[1],vv[end])`. The Rust port scans `0..=(n-ilen)` over the *raw* sorted slice with no finite trimming. When all values are finite the *interval-length* logic agrees, but: (a) any non-finite value first triggers H1 (panic); (b) `-0.0`/`+0.0` ordering and ties are handled by `partial_cmp` whereas Julia sorts then trims — edge tie behavior can pick a different equally-short window than the reference dump. Low practical impact on the headline number but a parity-divergence in a function explicitly advertised as mirroring Julia.

**Fix:** port the finite-trim + `stop < start+ilen-1` early-out faithfully; this also resolves H1.

---

## LOW

### L1 — `quantile_sorted` endpoint semantics
**File:** `src/model/linear.rs:13104-13117`. Type-7 interpolation is correct in the interior; at `probability` exactly 0 or 1 it returns `values[0]`/`values[n-1]` which matches R type-7, but `validate_probability` admits the closed `[0,1]` while `validate_level` uses open `(0,1)` — inconsistent domain contracts across the two summary entry points. Cosmetic; document or unify. Confidence: HIGH (behavior), LOW (impact).

### L2 — `confint` `swap` can mask a non-monotone reverse spline
**File:** `src/stats/profile.rs:155-159`. `add_profile_splines`/`profile_sigma` already refuse non-strictly-increasing ζ before building the reverse spline (`profile.rs:1244-1250`, `:495-501`), so a non-monotone reverse map should be unreachable from the supported entry points; the unconditional `lower>upper` swap is then dead defensive code that would *hide* (rather than surface) a future regression if the upstream guard were weakened. Prefer asserting monotonicity over silently swapping. Confidence: LOW.

---

## Verified-correct (cross-checked vs Julia) — commendations

- **C1 — VarCorr variance/correlation reconstruction is exact.** `varcorr.rs:42-91`: `std_dev[i] = σ·√Σλ[i,k]² = √(σ²(λλ')[i,i])` and `corr[i,j] = (λλ')[i,j]/√((λλ')[i,i](λλ')[j,j])` reproduce Julia `sdcorr` (`utilities.jl:159-171`) / `σρs` (`mixedmodel.jl:207-211`) precisely; σ cancels in the correlation exactly as in Julia. Scalar-RE no-correlation and residual-source gating are correctly handled. Strong test parity with `pls.jl`.
- **C2 — LRT REML/ML guard is enforced, not cosmetic.** `assess_model_pair` (`lrt.rs:1104,1121-1122,1163-1169`) routes any REML/ML mismatch to `MixedFitCriterion` → `lrt_available=false`, and REML + differing fixed effects → `ml_refit_required` with a clear reason; `LinearModelFit` is forced `optsum.reml=false` (`lrt.rs:97`) so lm-vs-LMM rejects REML LMMs. `dof()` correctly counts `rank + n_theta + (σ estimated)` (`linear.rs:8010-8012`) so Δdf includes variance components. The statistic uses `loglik` differencing (not deviance), and `loglik_within_optimizer_tol` cleanly handles tiny negative diffs.
- **C3 — Boundary + bootstrap LRT routes are statistically honest.** `BoundaryLikelihoodRatioTest` uses the correct 50:50 χ²₀:χ²₁ mixture and *refuses* >1 added boundary parameter (`:536-538`), explicitly pointing at the boundary-robust bootstrap. `parametric_bootstrap_lrt` counts refit failures separately, refuses when all fail, and uses the `+1` plug-in estimator; the caller owns RNG seeding (deterministic, verified by `test_parametric_bootstrap_lrt_runs_and_is_seed_deterministic`).
- **C4 — Internal bootstrap summaries correctly exclude failed refits.** `parameter_series()` (`linear.rs:13006-13045`) filters `is_finite()` per-parameter before quantiles/intervals, so the CI is computed on converged replicates only and reports `n` per parameter — no silent NaN contamination of `percentile_intervals`/`shortest_intervals`. GLMM PB refuses InverseGaussian/Normal until a certified simulator exists (`bootstrap.rs:92-99`). spline.rs natural-BC math and strictly-increasing-x guard are correct.

---

## Recommendation

**REQUEST CHANGES (one HIGH) — otherwise release-acceptable.**

- Fix **H1** before tagging the RC (panic on documented public-API input) — the same fix resolves **M3**.
- **M1** and **M2** are statistical-honesty gaps that should at minimum be documented in the RC notes and tracked; M1 ideally gets a typed "profile did not reach level" refusal.
- L1/L2 are polish.

The internal fitted-model → VarCorr/CoefTable/ModelSummary/LRT/profile/bootstrap pipeline is numerically faithful to MixedModels.jl; the risk is concentrated at the public utility boundary (H1) and in profile-CI extrapolation honesty (M1).
