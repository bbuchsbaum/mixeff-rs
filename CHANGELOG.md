# Changelog

All notable changes to `mixeff-rs` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once 1.0.0 ships. See [VERSIONING.md](VERSIONING.md) for the authoritative
versioning contract (breaking-change rules across the Rust API, numerical output,
formula DSL, JSON schemas, and Julia parity) and
[docs/semver_policy.md](docs/semver_policy.md) for the module-by-module stable
vs. `unstable-internals` surface inventory.

## [Unreleased]

### Changed (certification labels)

- Joint Laplace/AGQ GLMM stationarity is judged by a Newton-decrement
  estimate of the objective gap (`λ²/2`, deviance units) instead of the raw
  finite-difference gradient norm against 2e-2 whenever the estimate is
  assessed; tolerance 1e-6. The estimate uses the certificate's existing
  probes (central second differences) plus the fixed-effect block of the
  working PLS Hessian, so it costs no extra objective evaluations; a stop it
  would reject first has its single-probe covariance-parameter curvatures
  confirmed at the escalated steps (four evaluations each). The
  pure-diagonal fallback (no usable working block) is recorded but does not
  decide. Gradient
  magnitudes depend on parameter scaling; the gap does not. On a calibration
  set of premature stops (contraception, cbpp, culcita; Laplace and AGQ) the
  estimate tracks the true gap within a factor of 1.5, while the raw
  gradient certified stops up to 1.6e-3 above the optimum and rejected
  stops 5e-7 above it. Fitted values are unchanged; only `fit_status` and
  the labels that follow it can move:
  - `converged_interior` → `not_optimized` for stops whose estimated gap
    exceeds 1e-6 although the raw gradient passed (for example the
    no-`nlopt` TrustBQ joint AGQ fit of the rare-event Bernoulli set, 1e-6
    above the optimum). Such a fit is then returned as the labelled
    uncertified joint candidate, or as the labelled fast-PIRLS fallback when
    the joint fit did not improve on its start.
  - `not_optimized` / `not_assessed` → `converged_interior` for stops whose
    raw gradient exceeded 2e-2 only along stiff directions (grouseticks joint
    Laplace, whose `cHEIGHT` curvature is 1.7e5; the no-`nlopt` contraception
    random-slope fit).
  - `not_assessed` → `not_optimized` when the decrement assesses a gap the
    noise-aware gradient could not (no-`nlopt` contraception joint fits,
    1.05e-4 above the optimum).
  When the decrement cannot be assessed (a coordinate without a determined
  gradient or positive curvature), the previous gradient rule decides.
- `OptimizerCertificate` gains an optional `stationarity_decrement` field
  (`NewtonDecrementEvidence`: gap tolerance, the gradient readings used,
  excluded coordinates with reason codes, the eager estimate and verdict,
  and a `full` estimate from the finite-difference joint Hessian when
  joint-Laplace inference is computed). The field is omitted when absent, so
  existing artifacts serialize unchanged; the certificate lives in the
  `unstable-internals` compiled-artifact schema.

## [1.0.0-rc.3] - 2026-09-23

Documentation-only release: no code, API, or numerical changes from
1.0.0-rc.2. It exists so crates.io and docs.rs carry the corrected README and
crate description. The RC soak clock is not reset.

### Documentation

- README rewritten for clarity. It describes mixeff-rs as a native Rust
  implementation of the penalized-least-squares/PIRLS formulation, tested
  against lme4 and MixedModels.jl. It names GLMM options by the Rust API, says
  precisely what each parity gate checks, and its quick start runs as a
  doctest.
- Crate description (crates.io) updated to match.
- VERSIONING.md §3.1 separates the two guarantees that were conflated: Rust
  output is "within band" when every Rust parity test passes at its committed
  tolerance; the abs 1e-7 / rel 1e-8 drift-gate band applies to the
  MixedModels.jl reference fixtures themselves. §2.E, §3.2 and §3.4 follow.
- Supported-features guide lists `NegativeBinomial` (NB2, `Log` link) and its
  parametric bootstrap support.

## [1.0.0-rc.2] - 2026-09-22

Second 1.0 release candidate. Corrects the Type III term hypothesis, the
profiled-GLMM fixed-effect covariance scale, and R's coding of non-marginal
interactions; adds warm-started refits, host progress/interrupt callbacks, and
GLMM convergence verification; and cuts construction, optimizer, and post-fit
cost substantially. Default-path results stay inside the documented parity band
on the fixture corpus. Because this RC changes the meaning of existing outputs
(see **Fixed (numerical)**), the soak clock restarts (VERSIONING.md §5).

### Upgrading from 1.0.0-rc.1

These would be MAJOR changes after 1.0.0; they land in the RC line as
correctness fixes or deliberate convention changes. Check downstream code
and stored outputs for:

- Type III term tests now test the marginal hypothesis (different F/p for
  models with interactions); the old quantity is `coefficient_block`.
- Type I term tests follow R `terms()` order.
- Profiled (fast-PIRLS) GLMM `vcov` / `stderror` drop a spurious σ² factor.
- Non-marginal interactions use R's full dummy expansion (β and fitted values
  change for such formulas).
- Profiled GLMM covariance payloads report
  `status = "available_noninferential"` instead of `"available"`.
- With `nlopt`, fits with more than six θ parameters default to
  gradient-driven TrustBQ instead of NEWUOA.

### Added

- `RefitStart::{Initial, Fitted, From(theta)}` with
  `LinearMixedModel::refit_with_start` and
  `GeneralizedLinearMixedModel::refit_with_start`. `refit` keeps `Initial`
  (Julia `refit!` semantics); warm starts contract the optimizer's first step
  to `model::linear::WARM_REFIT_INITIAL_STEP` (0.75 / 8). GLMM warm refits also
  reuse the current conditional modes and fixed effects as the first PIRLS
  start.
- `FitProgress`, `FitProgressPhase`, and `FitProgressCallback` (throttled via
  `with_interval`), attached with `FitOptions::with_progress_callback` /
  `GlmmFitOptions::with_progress_callback`. A callback error stops the LMM
  optimizer, joint GLMM optimizer, PIRLS, and bootstrap loops and is returned
  as the new `MixedModelError::Interrupted` (code `"interrupted"`).
- `try_parametricbootstrap`: fallible LMM bootstrap that reports progress and
  propagates interrupts. `parametricbootstrap` is unchanged and ignores
  callbacks.
- `GeneralizedLinearMixedModel::verify_convergence` /
  `verify_convergence_with_options`, mirroring the LMM contract
  (restart-from-optimum and deterministic jitter refits of the estimator that
  produced the fit, attached to `certificate.verification`).
- `TrustBqGradientOracle` and `OptimizerControl::with_trust_bq_gradient_oracle`
  to override the analytic-gradient policy of the native TrustBQ path
  (A/B and diagnostic use; the default `FamilyPolicy` is recommended).
- `LinearMixedModel::fit_phase_timings()` returning
  `model::linear::FitPhaseTimings` (optimizer vs post-fit wall time).
  Diagnostic only; not part of any parity or serialized contract.
- `Formula::materialize_cow`: borrows the input `DataFrame` when the formula
  has no derived columns. `materialize` keeps its owned signature.
- Fixed-effect term tests gain a `coefficient_block` test type: the former
  Type III identity-block hypothesis (simple effects at the other factors'
  reference levels) under an honest name.
- TrustBQ stop code `GTOL_REACHED` (projected gradient below tolerance),
  classified as converged by `OptSummary`.
- Advanced surfaces outside the stable barrel (`model::batch`, documented
  unstable in `docs/semver_policy.md`): `BatchWarmStart::Chained { order }`
  chains per-column θ starts in a caller-given order.
  Behind `unstable-internals`: `EstimatorSubstitution` on the optimizer
  certificate and typed `requested_method` / `effective_method` on GLMM fit
  metadata.

### Changed

- Fixed-effect term Type I tests sequence terms by R `terms()` order
  (interaction order, then formula order), so `y ~ A*B + x` tests
  `A, B, x, A:B` rather than `A, B, A:B, x`. This matches R/lmerTest; results
  for formulas whose terms were already in that order are unchanged.
- `mixedmodels.fixed_effect_inference_table` schema `1.0.0` → `1.1.0`
  (`coefficient_block` value; Type III / Type I semantics as above and under
  **Fixed (numerical)**). Fixed-effect covariance-matrix schema `1.0.0` →
  `1.1.0`: working-Hessian payloads (profiled fast-PIRLS GLMM) now report
  `status = "available_noninferential"` with a reason instead of `"available"`;
  certified joint-Laplace payloads keep `"available"`. Consumers that test for
  `"available"` should also accept the new value.
- Certificate derivative evidence, the fast-PIRLS profiled-optimum
  certificate, and the joint-Laplace fixed-effect Hessian are computed on
  first inspection (artifact, certificate, audit report, `print_summary`,
  `vcov`/`stderror`, `coeftable`, prediction variance, `verify_convergence`)
  instead of eagerly at the end of `fit`. Inspected values are identical to
  the eager path; the cost moves to the first inspection.
- Certificate gradients come from the analytic profiled-deviance gradient
  (reported as `exact`); the Hessian is a central difference of that gradient
  and is reported separately as `finite_difference`, so certification
  quality stays `Approximate`. The objective-difference path remains the
  fallback when the blocked gradient cannot be formed.
- Default optimizer dispatch: with the `nlopt` feature, `n_theta > 6` fits
  now use gradient-driven TrustBQ instead of NEWUOA (a `max_time` budget
  keeps NEWUOA; `n_theta <= 6` keeps BOBYQA). The native (no-`nlopt`) TrustBQ
  path uses the analytic gradient for every family and reuses its quadratic
  model across rejected and well-predicted steps. Native builds also start
  crossed full-covariance vector-term fits with `n_theta >= 7` on the
  diagonal-first ladder with a 2000-evaluation budget.
- TrustBQ in the profiled/joint drivers accepts FTOL and stagnation stops
  only once the trust region has contracted by four halvings, and (on the
  gradient path) only when the projected gradient is within 1e3× of the
  gradient tolerance.
- A GLMM joint fit that falls back to fast PIRLS records the substitution on
  the optimizer certificate and in the audit report; `OptSummary`
  classifies such fits by the returned fast-PIRLS stop code.
- Audit and summary wording: recommendations phrased as options, the
  boundary skip reason stated once, `verify_convergence` pointer made
  host-agnostic, and plain-language random-effect explanations. Render-only;
  stored certificates unchanged.

### Changed (numerical)

All within the documented parity band (VERSIONING.md §3.1) on the fixture
corpus; the Rust parity tests against the Julia fixtures pass unchanged.

- Optimizer trajectories and evaluation counts change with the dispatch and
  TrustBQ changes above (for example, gradient-driven crossed benchmark rows
  52/110 evaluations instead of ~270/410). On flat large-θ objectives outside
  the fixture corpus the new default can stop at a lower objective than
  NEWUOA did (benchmark crossed rows: 1.9e-5 and 5.1e-5 lower).
- Sparse block products in the blocked Cholesky change summation order:
  arabidopsis fevals 122 → 143 and verbagg 41 → 43, objectives within 3.2e-10.
- Kenward-Roger's variance-parameter Hessian is built from the analytic
  gradient; KR denominator df can shift at the finite-difference noise level.

### Fixed (numerical)

- **Type III term tests now test the marginal hypothesis.** Previously each
  term used the identity block on its own coefficients, which under
  treatment coding with an interaction tests the simple effect at the other
  factor's reference level and depends on the reference level. Type III now
  uses SAS/`car`/`lmerTest` containment averaging (coding-independent,
  reference-invariant). Example (unbalanced A(2)×B(3) + covariate LMM): the old
  A-row F was 11.94 against lmerTest's 56.51; on a worked fixture the old F
  moved 18.01 → 14.94 when B's reference level was relabeled, while the new F
  is 9.985 under either.
  The old quantity is available as `coefficient_block`.
- **Profiled (fast-PIRLS) GLMM `vcov` / `stderror`** were on the inner
  working-LMM residual scale (a spurious σ² factor). They are now rescaled to
  the GLMM dispersion scale (exactly 1 for Bernoulli/Poisson; scaled families
  use the ML `sqrt(pwrss / n)` convention).
- **Non-marginal interactions** (e.g. `y ~ temperature + recipe:temperature`,
  no `recipe` main effect) now use R's full dummy expansion for the factor
  without a marginal term. Previously that factor was treatment-coded,
  dropping its reference-level column and constraining the model; fitted
  values and β for such formulas change to match R/lme4.
- TrustBQ no longer accepts FTOL stops at a coarse trust radius: a
  two-variance ML fit had stopped at radius 8.4e-2 with |gradient| 17.7.
- **Certified joint GLMM (NLopt BOBYQA) premature stops.** NLopt's BOBYQA
  reports `FTOL_REACHED` on the first small improving step regardless of the
  trust radius, so the joint (θ, β) fit could stop early at a point decided by
  platform rounding (contraception `(1 | dist)`: 9.2e-5 above the optimum on
  Linux/Windows, at the optimum on macOS). The driver now uses MixedModels.jl's
  tolerances (ftol_rel 1e-12, ftol_abs 1e-8), scales the β initial step by the
  profiled standard errors, confirms every FTOL stop with a restart at 0.1×
  the initial step, and raises the default budget from 200 to 500 + 80·n.
  Across perturbed starts (1e-8 to 1e-5 relative), every cbpp, culcita, and
  contraception Laplace/AGQ fit now lands within 1e-7 of the best objective
  (previously up to 2.5e-4 above). Joint objectives move down by up to
  ~1.7e-6 on macOS (contraception 2413.61646229 → 2413.61646071, MixedModels.jl
  `fast=false` 2413.6164609).

### Fixed

- `verify_convergence` on reduced-rank fits reports each run's effective
  ranks; the spurious "ranks changed" verdict is gone.
- A joint GLMM fit that followed a profiled stage no longer carries a stale
  deferral flag that could overwrite the joint certificate with profiled
  probes on inspection.
- Interaction/cell groupings key on level references rather than joined
  label strings, so two level combinations whose labels contain the
  separator no longer merge.

### Performance

Release profile, same host; objectives, θ, and evaluation counts
bit-identical unless noted above.

- Model construction 1.4–1.9× faster (borrowed data frame, single-pass audit,
  slice-based cross-product kernels, one pivoted QR, column-major Z).
- Nested scalar random effects keep sparse L blocks: grouseticks GLMM
  220 → 28 ms (lme4 246 ms).
- Joint Laplace: contraception `(1 | dist)` 265 → 129 ms (242 evaluations
  after the convergence fix above; 1.4× lme4), cbpp 4.2 → 2.1 ms (deferred
  Hessian, fixed-β PIRLS skips fixed-effect blocks, cached response
  log-constants). Joint AGQ(7): contraception 211 → 82 ms; n = 20k Bernoulli
  4.6 → 0.35 s.
- Profiled GLMM: verbagg 560 → 336 ms from weighted A-block rebuilds; deferred
  certificates add 1.25–1.74× on grouseticks, verbagg, and contraception.
- LMM: deferred certificate evidence 1.4–1.6× on crossed fits; gradient-driven
  TrustBQ cuts native-profile evaluations 81% (vector rows ~150–200 → ~20).
- Warm-started refits: LMM bootstrap 1.45–1.73× per replicate, GLMM
  1.2–2.4×. Batch per-column θ search 2.6× with 10× fewer allocations.

### Parity

- The Julia reference is pinned to **MixedModels.jl 5.9.0** (previously 5.3.0)
  through a committed `scripts/julia` Project/Manifest used by CI and the
  release workflow (Julia 1.12.4). The eight parity fixtures were regenerated
  under the pin: θ/β move by at most ~4e-9 and objectives by ~1e-10 relative,
  inside the abs 1e-7 / rel 1e-8 band; provenance strings record 5.9.0.
- The Julia drift gate now passes on Linux CI (it had never reached most
  fixtures there) and reports every drifting fixture instead of stopping at
  the first. Three references were only as precise as Float64 rounding
  allowed, which let Linux and macOS Julia land at different points:
  `kb07_ranef` and the `easy_full_rank` pathology fixture are now
  Newton-polished in 256-bit arithmetic from the MixedModels optimum (θ
  moves by up to 3.3e-4 / 6e-5), and `gamma_glmm_engines`, a boundary fit,
  is pinned to θ = 0 exactly with β from the Gamma GLM (β moves up to
  1.5e-5, now matching the Rust reference to ~1e-16). Each polish verifies
  that the new point is no worse than MixedModels' own optimum.
- `reduced_rank_unit_correlation` (pathology corpus) has no REML minimum, so
  MixedModels stops at an arbitrary θ. Its Julia fixture is now checked for
  its recorded behaviour rather than its digits (a named exception in
  VERSIONING.md §3.1). Every other fixture stays on the strict band.
- `comparison/lme4_results.json` regenerated at full precision
  (`digits = 17`); every value rounds to the previous record.
- Contraception `(1 | dist)` Binomial/Laplace promoted from
  `documented_divergence` to `release_blocking_parity` on the certified joint
  path: 9.1e-5 below lme4 (MixedModels.jl `fast=false` agrees with the Rust
  side within the parity band, 8e-11 relative); the gap had been an artefact
  of the 4-decimal reference. The random-slope row stays
  `documented_divergence`.

### Internal

- Fit-phase benchmark harness with same-host Julia GLMM references,
  re-pinned speed gates, and a Julia GLMM section in `comparison/REPORT.md`.
- Numerical-loop Clippy exceptions localized; tree rustfmt-clean.
- Cross-platform pinned tests hold objectives to the parity band and
  coordinates/SEs to commented looser tolerances (VERSIONING.md §3.1); the
  gradient finite-difference test uses a Richardson-extrapolated reference.

## [1.0.0-rc.1] - 2026-07-03

Work toward the 1.0.0 release. The numerics (PLS/PIRLS, blocked Cholesky,
profiled (RE)ML) are structurally stable; the remaining churn is in the public
API framing, the inference surface, and release infrastructure.

### Added

- `VERSIONING.md` — authoritative versioning policy covering the five versioned
  surfaces (Rust API, numerical output, formula DSL, JSON schemas, Julia parity),
  SemVer interpretation, numerical-tolerance band, MSRV policy, deprecation
  process, and downstream R/Python compatibility matrix.
- `RELEASE_CHECKLIST.md` — step-by-step release runbook (pre-release gates,
  version/changelog bump, package verification, tag, publish, soak, post-release).
- `LinearMixedModelBuilder` + `GeneralizedLinearMixedModelBuilder` +
  `FitOptions` / `ModelCriterion` — fluent construction that collapses the
  `fit(reml: bool)` boolean and the GLMM `new_with_*` constructor set into one
  chained surface. Additive; the existing constructors still work.
- `mixeff_rs::prelude` — glob-import module bundling `DataFrame`,
  `LinearMixedModel`, `GeneralizedLinearMixedModel`, `MixedModelFit`, `Family`,
  `LinkFunction`, `parse_formula`, `Result`, and `MixedModelError`.
- `pub use nalgebra;` — downstream code can name the exact `nalgebra` this
  crate builds against (it appears in public signatures) without pinning its
  own dependency to our minor version.
- `wald_confint` on `CoefTable` and `MixedModelFit`; `CoefTable` surfaces
  Satterthwaite / Kenward-Roger degrees-of-freedom rows.
- GLMM parametric bootstrap for Bernoulli / Binomial / Poisson families;
  parametric-bootstrap LRT route for one added variance component.
- Inference simulation harness (`examples/inference_route_simulation.rs`) with
  a stable JSON output schema.
- `mixedmodels.fit_summary` `1.0.0` — versioned JSON envelope for downstream
  wrappers that need objective values, optimizer metadata, coefficient tables,
  variance components, and summary-table rows in one stable payload.
- Stable `MixedModelError::code()` and `LinAlgError::code()` machine strings
  for downstream bindings that must branch without parsing display text.
- Default-feature compiler-contract coverage
  (`tests/compiler_contract_structure.rs`) so wire-serialization regressions
  are caught on every CI run, not only the NLopt leg.
- Expanded CI matrix (Linux/macOS/Windows, `--no-default-features`,
  `--features nlopt`, `--features prima`, MSRV-pinned leg), scheduled Julia
  parity gate, and a `cargo deny` / `cargo audit` supply-chain job.

### Changed

- `docs/semver_policy.md` reduced to the module-inventory appendix of
  `VERSIONING.md`; the broader policy prose that duplicated `VERSIONING.md` has
  been removed and replaced with a header note cross-linking to
  `VERSIONING.md` as the authoritative source.
- `#[non_exhaustive]` added to public enums and model/result structs where
  downstream construction should stay builder/accessor based; adding fields or
  variants is no longer a SemVer-major change.
- Solver internals on `LinearMixedModel` and `GeneralizedLinearMixedModel`
  sealed behind accessors / trait methods so 1.0 does not freeze the PLS,
  Cholesky, compiler-artifact, or PIRLS working-state layout.
- Numerical primitives (`linalg`) demoted from `pub` to `pub(crate)` — they are
  internal to the fit path, not part of the stable API.
- `compiler`, `datasets`, and `pathology` are no longer part of the default
  public API. They are `pub` only under the new opt-in `unstable-internals`
  Cargo feature (and `pub(crate)` otherwise), so the in-flux compiler/IR is not
  frozen into the 1.0 SemVer contract. Internal crate code is unaffected;
  downstream code that needs them must enable `unstable-internals`. CI runs an
  `unstable-internals` leg on every push so that surface stays tested.
- MSRV declared honestly as Rust 1.85, matching the current dependency graph's
  Rust 2024 edition requirement.
- Documented Clippy policy: numerical-loop exceptions are locally scoped with
  an explicit algebra or accumulation-order reason, while genuinely
  cross-cutting API and boundary exceptions remain crate-wide (see
  `src/lib.rs`).

### Notes

- This is a 1.0 release candidate. The stable API and wire-contract surface are
  intended to soak before the final 1.0.0 tag; any breaking RC feedback will
  require a new `-rc.N` release and restart the soak clock.
- Multivariate response (`cbind(y1, y2) ~ ...`), Gamma GLMM bootstrap,
  Kenward-Roger beyond the current scalar-test scope, full `I()` /
  formula-level transformations, first-class `polars`/`arrow` ingestion, and
  GLMM profile likelihood are explicitly **out of scope for 1.0** and tracked
  as post-1.0 work.

[Unreleased]: https://github.com/bbuchsbaum/mixeff-rs/compare/v1.0.0-rc.3...HEAD
[1.0.0-rc.3]: https://github.com/bbuchsbaum/mixeff-rs/compare/v1.0.0-rc.2...v1.0.0-rc.3
[1.0.0-rc.2]: https://github.com/bbuchsbaum/mixeff-rs/compare/3332f3e2bd06a21d67bb519860475cdcec0ac9c1...v1.0.0-rc.2
[1.0.0-rc.1]: https://github.com/bbuchsbaum/mixeff-rs/tree/3332f3e2bd06a21d67bb519860475cdcec0ac9c1
