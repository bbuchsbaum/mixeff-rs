# PRD: Mixed Model Compiler Contract v0

## State

Status: Implementation in progress  
Owner: mixedmodels Rust crate  
Last updated: 2026-04-27  
Related design note: `docs/mixed_model_compiler_inference_contract.md`  
Formula-layer slice: `docs/random_effects_formulas.md`  
Focused p-value support plan: `docs/fixed_effect_p_values_plan.md`
Implementation state: Started; initial additive compiler-contract skeleton is
in `src/compiler/`; grouping canonicalization for `a:b`, `a/b`, and `a*b`
is started in the formula parser; fixed-effect design audits now report rank,
aliased columns, missing fixed-effect columns, and empty categorical interaction
cells; `LinearMixedModel` construction now carries a read-only compiled artifact
with attached design audit, and artifact theta maps are rebuilt in optimizer
random-term order after the fit engine's random-term sorting step; fitted LMMs
now attach a structured optimizer certificate that records optimizer stop
evidence, parameter-space/boundary context, sample-size context, and explicit
not-available derivative/KKT/Hessian evidence; random
term audits now include deterministic v0 information-budget thresholds and
statuses for variance/covariance support; a typed compiler policy now emits
advisory maximal-feasible recommendations without rewriting the fitted model;
compiled artifacts and fitted LMMs now expose a stable model-audit report for
requested model, design, policy, optimizer, and inference state, including a
dedicated random-effect information-budget/effective-n section that reports
rows, grouping levels, observations per level, covariance-parameter counts,
levels per parameter, misleading-total-row warnings, and concrete
recommendations; versioned JSON wire fixtures now snapshot the compiled
artifact and audit-report schemas for external clients, including the first
ordinary worked-example fixture for the `sleepstudy` random intercept/slope
LMM and the crossed random-intercept `penicillin` example; fixed/random
intercept redundancy is now emitted as a structured design-audit diagnostic
and snapshotted as a worked example; GLMM
artifacts now record family/link, objective approximation, certificate scope,
derivative availability, and unsupported LMM finite-sample inference; effective
covariance rank summaries now record supported and unsupported random-effect
directions with user-scale loadings, and fitted LMMs now generate those
summaries automatically from `sigma^2 * Lambda Lambda'`, including
certificate-time reduced-rank records and covariance-family transitions;
LMMs now expose `verify_convergence()` for bounded restart, jittered-start, and
alternate-optimizer checks, with the result attached to the optimizer
certificate and audit report as a structured convergence-verification record;
a pathology-corpus foundation now provides a pure linear-algebra generator
certificate, four TOML fixture strata, and status-set membership tests for
easy, boundary, reduced-rank, and refusal cases;
few-level scalar random-intercept policy is now implemented in code, separating
fit eligibility from low-reliability warnings while keeping stricter gates for
non-intercept variance directions and correlations;
LMMs and GLMMs now expose public compiler-policy setters and constructors so
policy is attached before fitting without mutating internal artifacts directly;
`CompilerPolicy::design_compiled()` now provides an executable
confirmatory/design-compiled path that can apply deterministic design-time
full-to-diagonal covariance reductions before building the optimizer problem,
records the effective formula/semantic model, and refuses unsupported
random-effect distributions before optimization;
the formula-layer v0 open questions are now resolved into explicit decisions
for within-group variation thresholds, collision-free composite keys,
lexicographic composite-level ordering, R6/R7 behavior, `||` centering
reference, and random interaction coding;
the initial five worked-example fixtures are complete; the `singular` dataset
now pins the Bates-style reduced-rank/too-rich covariance story; a local mote
store has been initialized for implementation tracking  
Scope: Binding near-term contract for the Rust engine, before R bindings and
advanced finite-sample inference.
Revision focus: harden the compiler contract by making every transformation
round-trippable, typed, deterministic, serializable, and traceable.

Mote tracking is local to this repository under `.mote/`. The umbrella
reconciliation issue is `bd-01KQ7WW46RJ6B71TAPZ0CK6KVJ`.

This PRD covers the first shippable version of the mixed-model compiler idea:
semantic model representation, design audit, structured diagnostics,
requested-versus-effective models, and a parameterization map that makes the
optimizer-facing `theta` layout explicit.

This PRD does not commit the project to the full future architecture described
in the design note. Kenward-Roger, broad residual covariance structures,
regularized covariance selection, full model-lattice search, and R bindings are
future work unless explicitly called out here as interface constraints.

## Description

The current project is a Rust mixed-model crate with lme4/MixedModels.jl-style
formula parsing and model fitting. The long-term product direction is a mixed
model compiler: formula syntax is treated as a surface language, compiled into
a semantic model object, checked against the observed design, and fitted only
when the model is coherent enough to support the requested output.

The v0 product goal is narrower:

> Before changing the numerical engine deeply, build the compiler contract that
> lets the crate explain what a model means, identify obvious design problems,
> record requested versus effective model structure, and expose structured
> diagnostics that downstream clients, including a future R layer, can consume.

The revised thesis is:

> Mixed-model pain is caused by hidden transformations between user intent,
> formula syntax, random-effect basis, covariance parameterization, optimizer
> state, and inference output. The product must make those transformations
> explicit.

The most important v0 primitive is a round-trippable compiled model artifact:

```text
requested model
  -> semantic model
  -> compiled basis
  -> covariance/theta map
  -> optimizer objective
  -> active fitted model
  -> inference object
  -> audit/explanation
```

Every arrow must be recorded. Requested-versus-effective model state is still
central, but it is not enough by itself. Every fitted covariance parameter must
be traceable back to the requested formula term, semantic random-effect term,
compiled basis column, `theta` slot, `Lambda` slot, and user-facing VarCorr
summary.

If only one user-visible behavior ships from this PRD, it should be:

```text
explain_model() can say what the random-effects formula means, what assumptions
it makes, and which parts are suspicious before the optimizer runs.
```

## Problem

Mixed-model failures are often reported as numerical failures even when the
root cause is a model-specification problem:

- a random intercept is redundant with fixed effects
- a repeated unit is treated as independent
- a random slope is requested without within-group variation
- a full covariance matrix asks for more parameters than the design can support
- formula syntax such as `(1 | a:b)`, `(1 | a/b)`, and `(1 | a*b)` implies
  different dependence structures that users frequently confuse
- diagonal intercept/slope covariance depends on predictor centering
- optimizer convergence does not imply the fitted covariance structure is
  statistically interpretable

The crate needs a compiler layer that can distinguish these issues before
fitting and carry the resulting meaning through fitting, diagnostics, and
eventual client APIs.

## Goals

- Define stable Rust data contracts for diagnostics, audits, semantic random
  effects, requested/effective model state, and optimizer certificates.
- Define an explicit owner for the map between user-facing basis/covariance
  structure and optimizer-facing `theta` parameters.
- Make the parameterization map a sum type over covariance families rather than
  one Cholesky shape with implicit active zeros.
- Compile supported formula syntax into semantic IR without changing numerical
  fitting behavior initially.
- Provide `explain_model()` or equivalent prefit explanation for random-effect
  assumptions and covariance stories.
- Add v0 design-audit checks for rank, grouping levels, repeated units,
  random-slope support, empty cells, and obvious fixed/random redundancy.
- Implement a deterministic `maximal_feasible` v0 selection rule before
  implementing a general candidate lattice.
- Define reduction categories: design-time, certificate-time boundary, and
  selection-time.
- Define the GLMM boundary contract so non-Gaussian models do not inherit LMM
  inference guarantees by accident.
- Choose a v0 wire format for contract objects.
- Keep future R exposure in mind by making core results serializable from the
  start, while leaving R package implementation out of v0 scope.
- Preserve existing fitting behavior and parity tests while the compiler layer
  is introduced.

## Non-Goals

- No Kenward-Roger implementation in v0.
- No broad residual correlation structures such as AR(1), spatial exponential,
  or Matern in v0.
- No automatic regularized covariance selection in v0.
- No ordinary p-values from regularized or post-selection workflows in v0.
- No full R package or R-native syntax implementation in v0.
- No dense influence diagnostics requiring repeated refits in v0.
- No full automatic model-lattice search beyond the deterministic
  `maximal_feasible` v0 rule.
- No claim that GLMM KKT, derivative, or inference semantics are equivalent to
  LMM semantics.

## Users

- Rust crate users who need programmatic model diagnostics.
- Future R package users who expect lme4-compatible syntax but clearer
  explanations.
- Developers of this crate who need stable internal contracts before touching
  optimizer, inference, and residual-model machinery.

## Functional Requirements

### 1. Structured Diagnostics

The crate must define a common diagnostic representation.

Required fields:

- stable diagnostic code
- severity
- compiler/fitting stage
- human-readable message
- affected terms or variables
- suggested action when available
- optional machine-readable payload

The first implementation may emit a small set of diagnostics, but all new
compiler/audit components must use the shared type.

### 2. Requested, Supported, and Fitted Model State

Fit and prefit objects must be able to expose:

- original source formula or model spec
- canonical semantic model
- requested random-effect structures
- design-supported and fitted effective random-effect structures
- reason for every reduction, rejection, or recommendation
- whether a model was fit as specified or changed by compiler policy

The supported/fitted model story must not be hidden inside warnings or print
methods. In v0 this is exposed as a computed `ModelStateSummary` plus
`changes()`; later design-compiled fitting can reuse the same transition
schema when it applies model rewrites.

Every model change must be classified into one of three categories:

- design-time reduction: pre-response (`y`-independent) reduction based on
  formula, roles, design rank, grouping levels, within-group variation, or
  unsupported covariance structure
- certificate-time boundary: response-dependent optimizer result where a
  variance/covariance parameter lands on a valid boundary without comparing
  alternative fitted models
- selection-time reduction: response-dependent model search, regularization, or
  candidate selection across covariance structures

Certificate-time boundary reductions are compatible with confirmatory inference
when the inference method is boundary-aware or explicitly conditional on the
active set. They are not automatically post-selection events. Selection-time
reductions are exploratory unless selective-inference or sample-splitting
machinery is explicitly implemented.

Fit intent must use two confirmatory paths and two non-confirmatory paths:

```text
confirmatory/as_specified:
  fit exactly the requested structure or refuse with a structured diagnostic

confirmatory/design_compiled:
  allow pre-y design-time reductions, report every change, then fit

exploratory:
  allow regularization or response-dependent structure discovery; do not print
  ordinary confirmatory p-values

predictive:
  optimize predictive performance or validation target; do not print
  confirmatory p-values
```

`as_specified` must never silently reduce. `design_compiled` may reduce only
before seeing the response values.

### 3. Semantic Random-Effects IR

The compiler must represent random effects as semantic terms, not as formula
strings.

Minimum v0 IR:

- grouping factor
- random-coefficient basis
- intercept included or omitted
- covariance form: full, diagonal, scalar, or unsupported
- source syntax span/string
- grouping role: declared, inferred, or unknown
- generated covariance story

Supported formula canonicalization in v0 (rules and basis-construction owned
by `docs/random_effects_formulas.md` §3 and §4):

- `(x | g)`
- `(x || g)`
- `(0 + x | g)`
- `(1 | a:b)`
- `(1 | a/b)`
- `(1 | a*b)`
- `(1 | a) + (1 | b)`

The IR must preserve both what the user wrote and what the compiler understands
it to mean. The diagnostic codes emitted at parse time and at canonicalization
time are inventoried in `random_effects_formulas.md` §8.

### 4. ThetaMap / CovarianceMap

The crate must define a first-class owner of the mapping:

```text
user basis/covariance structure <-> canonical basis <-> theta layout
```

This contract is required in v0 even if only part of it is populated initially.

The map must record:

- user-scale basis columns
- canonical optimizer basis columns
- basis transformation from user scale to optimizer scale
- covariance form
- free optimizer parameters
- `theta` parameter names and order
- `Lambda` slots populated by each `theta`
- bounds/constraints
- active and inactive parameter status
- relation to existing `parmap`
- Jacobian `d theta / d phi` for free optimizer parameters when available
- second derivative hooks when available
- back-transform rules where known
- derivative mapping placeholder for later Satterthwaite/KR work

Any basis rewrite, scaling, diagonalization, or future reduced-rank operation
must pass through this map. Downstream gradient, Hessian, KKT, and inference
APIs must not infer this mapping ad hoc.

The map must be a sum type over covariance families. Full, diagonal, scalar,
structured, and reduced-rank covariance families are not the same optimization
manifold with different zero entries.

Sketch:

```rust
pub enum ThetaMap {
    Scalar(ScalarThetaMap),
    Diagonal(DiagonalThetaMap),
    FullCholesky(FullCholeskyThetaMap),
    Structured(StructuredThetaMap),
    ReducedRank(ReducedRankThetaMap),
}
```

Family transitions must be explicit records:

```rust
pub struct CovarianceFamilyTransition {
    pub from: CovarianceFamily,
    pub to: CovarianceFamily,
    pub trigger: ReductionTrigger,
    pub affected_term: String,
    pub dropped_or_reparameterized_slots: Vec<ThetaSlot>,
    pub inference_consequence: InferenceConsequence,
}
```

Reduced or unsupported parameters should be dropped from the optimization
manifold for the effective model, not silently retained as active zeros in a
larger full-Cholesky parameterization. Certificate-time boundary parameters are
different: they remain part of the requested/effective family but are marked as
boundary solutions by the optimizer certificate.

### 5. Prefit Explanation

The crate must expose a prefit explanation path.

The explanation must cover:

- fixed-effect terms at a basic level
- random-effect grouping factors
- random intercept presence or omission
- random slopes
- covariance form
- canonical expansion for nesting/crossing/cell expressions
- covariance consequence in plain language
- warnings for obvious suspicious assumptions

Example distinction:

```text
(1 | subject) + (1 | item)
  observations sharing subject are correlated
  observations sharing item are correlated

(1 | subject:item)
  only observations sharing the same subject-item cell are correlated
```

### 6. Design Audit v0

The v0 design audit must compute:

- fixed-effect design rank
- aliased fixed-effect columns where available
- empty factor combinations for terms represented in the model matrix
- grouping factor level counts
- observations per grouping level
- within-group variation for requested random slopes
- repeated-unit detection for grouping-like variables
- basic crossing/nesting/cell-only classification
- redundant random intercepts when fixed effects already span the same grouping
  indicators

The audit must return structured diagnostics, not only formatted text.

### 7. Estimability Module

Fixed-effect and random-effect estimability diagnostics must live in one module,
but they must not be represented as an unconstrained product of
`Estimand x Status`. Some status combinations are nonsensical. For example, a
fixed contrast cannot be basis-dependent in the same sense as a random-slope
covariance parameter.

The module should use typed variants that encode valid combinations:

```rust
pub enum EstimabilityAssessment {
    FixedContrast(FixedContrastEstimability),
    FixedTerm(FixedTermEstimability),
    RandomVarianceDirection(RandomVarianceEstimability),
    RandomCovarianceParameter(RandomCovarianceEstimability),
    KernelPath(KernelPathEstimability),
}
```

Each variant may use a context-specific status vocabulary, but shared concepts
must have shared names and diagnostic payload shapes:

- estimable
- partially estimable
- weakly estimable
- not estimable
- basis dependent
- not assessed

Invalid combinations should be impossible or awkward to construct at the type
level.

### 8. Deterministic `maximal_feasible` v0 Rule

The project must define a deterministic first-pass rule before implementing a
general scoring lattice.

The v0 rule must be documented in code and tests. Soft predicates are not
allowed in the binding rule. Every predicate must have a named threshold, a
default value, and an override path in policy options.

Default v0 thresholds:

```text
min_levels_random_intercept_fit = 2
min_levels_random_intercept_reliability = 5
min_levels_variance_direction(d_basis) = max(5, 2 * d_basis + 1)
min_levels_full_cov(n_cov_params) = max(10, 5 * n_cov_params)
max_condition_number = 1e10
min_within_group_sd = 1e-8 on canonical scale
max_basis_pairwise_abs_corr = 0.999
min_observations_per_supported_level = 2
effective_rank_relative_tolerance = 1e-6
effective_rank_absolute_tolerance = 1e-10
convergence_derivative_nparmax = 10
```

These numbers are engineering defaults, not universal statistical truths. They
must be configurable, but the default compiler behavior must be deterministic.
The few-level policy intentionally separates fit eligibility from reliability:
a scalar random intercept with at least two observed grouping levels may be fit
when it is not algebraically redundant, but levels below
`min_levels_random_intercept_reliability` degrade the reliability assessment
rather than forcing a fixed-effect conversion or design-time refusal. Stronger
level-count gates apply to non-intercept variance directions and especially to
correlation parameters.

The v0 rule follows this order:

1. Preserve the fixed-effect estimand.
2. Add required random intercepts for repeated sampled units when detectable.
3. Add random slopes for within-unit fixed effects when declared roles or design
   imply they are needed for confirmatory inference.
4. Fit scalar random intercepts when they meet the fit threshold and are not
   structurally redundant; mark their reliability weak when they fail the
   reliability threshold.
5. Estimate non-intercept variance directions only when within-group variation
   and grouping level counts pass the configured thresholds.
6. Prefer diagonal covariance over full covariance until both variance
   directions are supported and the information budget permits correlations.
7. Do not estimate correlations attached to unsupported or boundary variance
   directions.
8. If the supported direction is a mixture of named basis columns, report it as
   a supported covariance direction with user-scale loadings rather than hiding
   it behind a generic reduced-rank formula.
9. Refuse or mark inference unavailable when the required dependence path is
   not estimable.

The v0 rule must be deterministic for the same data, formula, options, and
seed.

Repeated-observation and within-unit checks must use explicit marginalization:

- A grouping factor `g` is repeated if any level of `g` has at least two rows in
  the model frame.
- A cell factor `a:b` is repeated if any observed `(a, b)` cell has at least two
  rows.
- In crossed designs, `a`, `b`, and `a:b` are assessed separately; repetition
  of `a:b` does not substitute for repetition of `a` or `b`.
- A fixed effect varies within `g` if its compiled contrast/basis columns have
  nonzero within-`g` variation above `min_within_group_sd`.
- A between-unit effect for `g` must not trigger a random slope by `g` unless
  explicitly requested.

Full covariance is allowed by the v0 compiler only when:

- all involved variance directions are supported
- `n_levels >= min_levels_full_cov(n_cov_params)`
- the relevant basis condition number is below `max_condition_number`
- pairwise absolute basis correlations are below
  `max_basis_pairwise_abs_corr`
- the predictor zero point and basis interpretation are recorded

Otherwise the v0 compiler must choose diagonal covariance, a scalar variance
direction, a design-time refusal, or a structured diagnostic, depending on the
fit intent.

Few grouping levels by themselves are not a sufficient reason to refuse a
simple random-intercept model. The audit should instead report weak precision,
imbalance, boundary risk, and inference reliability. Design-time refusal is
reserved for algebraic non-identifiability, unsupported required dependence
paths, invalid data, or policy modes that explicitly require refusal.

### 9. KKT Certificate Interface

The full KKT implementation may come later, but v0 must define the certificate
shape and integrate current optimizer status into it.

The interface must distinguish:

- optimizer exit status
- statistical fit status
- active boundary parameters
- sample-size and parameter-count context
- raw/scaled/free/projected gradient evidence
- Hessian method, quality, eigenvalue/rank/condition evidence
- overall certification quality
- unavailable checks
- failed checks

The certificate must be part of the fit object, even if many fields are `None`
or `NotAssessed` in v0.

The optimizer's convergence stop is authoritative. Post-hoc finite-difference
gradient/KKT/Hessian checks are inspection metadata: they may add caveats,
numeric evidence, or next actions, but they must not reclassify an optimizer
accepted fit as non-converged. If a stricter derivative criterion is intended
to define convergence, `fit()` must honor it directly instead of letting the
print or report layer overrule the optimizer after the fact.

Derivative inspection is gated by regime. Interior-theta fits may carry
free-gradient and active-subspace Hessian evidence. Boundary or singular fits
skip interior KKT checks and surface boundary/reduced-rank status instead.
Large-theta fits skip finite-difference derivative inspection above
`convergence_derivative_nparmax` and rely on optimizer-stop evidence plus
optional `verify_convergence()` agreement.

### 10. Serialization Boundary

The v0 wire format is versioned JSON using `serde`.

Rationale: JSON is inspectable, easy to snapshot in tests, easy to expose to a
future R layer, and sufficient for diagnostics/audit artifacts. Arrow or a
binary format can be added later for large numeric arrays, but the v0 audit and
contract schema must round-trip through JSON.

All v0 public contract objects should be serializable where practical:

- diagnostics
- semantic model
- requested/effective model state
- design audit
- parameterization map
- optimizer certificate
- fit status
- reproducibility record

Every serialized artifact must include:

- schema name
- schema version
- crate version when available
- deterministic ordering for lists/maps
- explicit `null`/missing handling for not-yet-assessed fields

This is not an R binding implementation, but it prevents the Rust engine from
evolving internal result shapes that are impossible to expose later.

### 11. Reproducibility

Any automatic reduction, future bootstrap, or randomized diagnostic path must
be reproducible.

In v0:

- deterministic compiler decisions must not depend on hash-map iteration order
- result ordering must be stable
- option structs must include a seed field or an explicit "not used in v0"
  placeholder where random behavior will later exist
- tests must not rely on nondeterministic diagnostic ordering
- every fit/audit artifact must include a reproducibility record with options,
  thresholds, selected fit intent, schema version, and seed/random-state status

### 12. Performance Budget

The compiler/audit layer must not make ordinary fitting feel slow.

Initial targets:

- formula-to-IR and prefit explanation for `sleepstudy`-scale examples:
  less than 10 ms target
- parse + compile + fit for `sleepstudy`-scale LMM: less than 50 ms target
- 10k rows with crossed random intercepts: fit less than 1 s target
- 100k rows with sparse crossed intercept/slope structure: fit within
  interactive range, initially 10-30 s depending on structure
- design-time refusal for unsupported `sleepstudy`-scale models: less than
  20 ms target
- design-time refusal for 10k-row grouping/rank failures: less than 250 ms
  target
- design audit should be linear or near-linear in rows for grouping summaries
- v0 diagnostics must avoid dense `n by n` covariance construction
- any operation that can become superlinear in rows or groups must be marked
  and tested separately
- bootstrap, when implemented later, must avoid an R formula loop and target
  near-linear parallel scaling

These are engineering targets and may be revised after baseline benchmarking,
but any revision must update this PRD. v0 implementation must include benchmark
hooks for formula-to-IR, explanation, design audit, successful fit, and
failure-path diagnosis.

### 13. GLMM Boundary Contract

The semantic compiler, diagnostics, requested/effective model state, ThetaMap,
and serialization contracts must be designed so they can represent both LMMs
and GLMMs.

v0 does not need to implement new GLMM inference. It must, however, prevent
LMM-only guarantees from leaking into GLMM output.

For GLMMs, fit/audit artifacts must be able to record:

- response distribution and link
- objective approximation: PIRLS, Laplace, adaptive Gauss-Hermite, or other
- optimizer certificate scope: exact objective, approximated objective, or not
  assessed
- whether covariance derivatives are available for the approximation used
- inference availability and method

No Satterthwaite/Kenward-Roger promise applies to GLMMs in v0. GLMM output must
mark those methods as unsupported unless a later PRD defines the required
derivative and approximation certificates.

### 14. Derivative Strategy Contract

The derivative backend is not implemented in v0, but the contract must choose a
direction so ThetaMap and optimizer interfaces do not drift.

Default strategy:

- analytic derivatives where the current linear algebra exposes them naturally
- forward-mode automatic differentiation for covariance-parameter derivatives
  once ThetaMap variants are stable
- finite differences for tests, validation, and fallback diagnostics, not as
  the primary production path for Satterthwaite/KR

No Satterthwaite or Kenward-Roger p-values may be printed unless a derivative
certificate exists for the relevant objective, ThetaMap variant, and fitted
active manifold.

This finite-sample restriction does not prohibit the row-level fallback
contract in `docs/fixed_effect_p_values_plan.md`: coefficient and scalar
contrast rows may expose labeled `asymptotic_wald_z` p-values with low
reliability, null finite-sample degrees of freedom, and explicit notes that the
result is not a Satterthwaite or Kenward-Roger correction.

### 15. Print-Layer Default

The internal artifact may contain requested, semantic, supported, fitted, and
changed model views. The default print output must not dump all of them.

Default output should show one canonical summary:

- fit status
- convergence verdict and stable documentation anchor
- requested formula only when useful
- effective formula/structure when it differs
- top diagnostics
- inference availability
- pointer to drilldowns

Detailed views should be opt-in:

- `explain_model()`: semantic and covariance story
- `audit()`: structured health report
- `parameterization()`: source syntax, semantic term, expanded basis,
  ThetaMap, `Lambda`, `parmap`, and VarCorr trace
- `changes()`: requested-to-supported-to-fitted recommendations, reductions,
  and consequences

The audit must reduce noise, not become another wall of warnings.
Convergence verdicts use the shared taxonomy in
`docs/compiler_verdicts.md`: optimizer, boundary/singular, structural
identifiability, verification, and not-assessed. These states are orthogonal;
compact print must not collapse singular, rank-deficient, and optimizer-failed
models into one generic convergence warning.

### 16. Compiled Model Artifact

The central v0 output of the compiler is a compiled model artifact. Fitting may
consume it, but it must also be inspectable before fitting.

The artifact must contain:

- requested model
- semantic model
- compiled basis
- ThetaMap/CovarianceMap
- covariance parameter traces from source syntax through `theta`, `Lambda`,
  `parmap`, and VarCorr entries
- design audit
- reduction records
- reproducibility record
- JSON schema metadata

After fitting, the same artifact is extended with:

- optimizer objective metadata
- optimizer certificate
- active fitted model
- inference availability/status
- audit/explanation

No optimizer, summary, or inference code should reconstruct these mappings from
formula strings.

## Acceptance Criteria

Legend:

- `[x]` implemented and covered by tests/fixtures.
- `[~]` partially implemented; follow-up work is tracked below.
- `[ ]` not implemented or intentionally deferred.

### Contract Acceptance

- [x] The crate defines shared diagnostic, audit, fit status, and optimizer
      certificate types.
- [x] Requested, semantic, supported, and fitted model structures can be
      represented separately through `ModelStateSummary`, exposed by
      `model_state_summary()` and `changes()`, and pinned by JSON fixtures.
- [~] Every model reduction or refusal can carry a structured reason.
      Certificate-time reductions, design-time covariance reductions, and
      policy refusals/recommendations are visible in `changes()`; broader
      basis-dropping and selection-time records remain open.
- [x] The `theta`/basis/covariance parameterization map has a named type and
      owner.
- [x] The parameterization map is a sum type over covariance families.
- [x] The parameterization map records the relationship to existing `parmap`.
- [x] Every fitted covariance parameter is traceable to source term, semantic
      term, basis column, `theta` slot, `Lambda` slot, and VarCorr summary.
      `covariance_parameter_traces` are serialized in the compiled artifact,
      refreshed from fitted `Lambda`/`parmap`, and summarized in the audit
      report.
- [x] Core v0 contract objects are serializable or explicitly documented as
      not yet serializable with a reason.
- [x] Versioned JSON round-trip tests exist for diagnostics, semantic model,
      design audit, ThetaMap, and optimizer certificate.
- [x] A compiled model artifact can be produced and inspected before fitting.
- [x] Fitting extends the compiled model artifact rather than reconstructing
      mappings from formula strings.
- [x] Fit intent distinguishes `confirmatory/as_specified`,
      `confirmatory/design_compiled`, `exploratory`, and `predictive`.
      `as_specified` and `design_compiled` are operational for the LMM
      constructor path, and row-level fixed-effect p-value gating suppresses
      ordinary confirmatory p-values for exploratory and predictive fits.
- [~] Model changes are classified as design-time, certificate-time boundary,
      or selection-time. Certificate-time boundary records exist; design-time
      covariance reductions exist; selection-time records are open.

### Semantic IR Acceptance

- [x] Supported lme4-style random-effect syntax compiles into semantic IR.
- [x] The IR separates grouping factor, random-coefficient basis, intercept
      policy, and covariance form.
- [~] Source syntax is preserved for diagnostics. Current `SourceSyntax`
      records canonical term text; written-vs-canonical spans/strings are
      still open (`bd-01KQ7WYW472SRF7NPG9P1HDMSM`).
- [x] `(1 | a/b)` canonicalizes to the nested semantic equivalent.
- [x] `(1 | a*b)` canonicalizes to main effects plus cell effect.
- [x] `(1 | a:b)` remains cell-only and is explained as such.
- [x] `(x || g)` is represented as diagonal covariance, not as a string-level
      special case.
- [x] `(0 + x | g)` records random-intercept omission explicitly.

### Explanation Acceptance

- [x] A public or crate-public `explain_model()` path exists.
- [x] Explanation can run without fitting.
- [x] Explanation distinguishes crossed, nested, and cell-only random
      structures.
- [x] Explanation states when a random slope is present without a random
      intercept.
- [x] Explanation reports covariance assumptions in user-facing language.
- [x] Explanation output is backed by structured data, not only string
      formatting.

### Design Audit Acceptance

- [x] Fixed-effect rank is computed or reported as not assessed with a reason.
- [x] Grouping level counts and observations per level are reported.
- [x] Random-effect information-budget/effective-n reports expose grouping
      levels, total rows, observations per level, random-coefficient basis
      dimension, covariance-parameter count, levels per covariance parameter,
      and whether total rows are misleading for covariance support.
- [x] Within-group variation for random slopes is checked over the expanded
      optimizer basis, including numeric slopes, categorical dummy/cell
      columns, and interaction columns.
- [x] Empty cells are detected for supported factor terms.
- [x] Subject fixed effects plus subject random intercept is diagnosed as
      redundant when column-space overlap is detected.
- [x] Repeated grouping-like variables without modeled dependence paths can be
      diagnosed (`bd-01KQ7WZVTTHMF5VWK355NMPR6G`). The design audit now records
      a compact covariance-kernel/dependence-path graph and emits
      `RepeatedUnitUnmodeled` when repeated marginal or cell paths lack a
      random-intercept kernel; cell paths do not substitute for crossed
      marginal paths.
- [x] Diagnostics are deterministic in ordering and content for fixed inputs.

### `maximal_feasible` v0 Acceptance

- [x] The v0 deterministic selection rule is written in docs and code comments.
      Threshold policy exists and the initial design-compiled path can apply
      deterministic full-to-diagonal covariance reductions; few-level scalar
      random intercepts are fit-eligible at the fit threshold and warn below
      the reliability threshold instead of forcing design-time refusal.
- [x] The v0 rule uses named numeric thresholds for levels, condition number,
      within-group variation, and basis correlation.
- [x] Thresholds are configurable through policy options.
- [x] The rule prefers variance directions before correlation parameters.
- [x] The rule can choose diagonal covariance when full covariance is too rich.
      Advisory mode reports the recommendation; `design_compiled` rewrites the
      optimizer-facing formula before fitting.
- [~] The rule refuses or marks inference unavailable when required random
      slopes are not supported. `design_compiled` refuses unsupported
      random-effect distributions; missing required-slope inference gating
      remains open.
- [ ] Repeated-observation checks use explicit marginalization for `a`, `b`,
      and `a:b` (`bd-01KQ7WZVTTHMF5VWK355NMPR6G`).
- [x] Mixture directions are reported with user-scale loadings when detected,
      not only with opaque spectral labels.
- [~] Tests cover too-minimal, too-rich, and coherent ordinary models.
      Coherent and too-rich are covered; too-minimal/missing-dependence tests
      remain open.

### Integration Acceptance

- [x] Existing ordinary LMM fits still pass baseline/parity tests.
- [x] Existing formula parsing behavior remains available.
- [x] New compiler/audit layers can run before fitting.
- [x] Fit objects can carry diagnostics and requested/supported/fitted model
      metadata through the compiler artifact and `ModelStateSummary`.
- [x] The LMM constructor can run an explicit `design_compiled` policy path
      that records an effective formula/semantic model and applied design-time
      reductions before optimizer construction.
- [x] No ordinary p-value table is added without method, status, and
      reliability placeholders.
- [x] Coefficient-level p-values have a concrete row-level support plan:
      labeled asymptotic Wald fallback first, Satterthwaite/KR only with
      derivative and finite-sample certificates
      (`docs/fixed_effect_p_values_plan.md`).
- [x] Regularized, exploratory, predictive, and selection-time-reduced LMM
      output refuses ordinary confirmatory fixed-effect p-values through the
      row-level inference table. Ordinary finite-sample p-values remain
      deferred until the Satterthwaite/KR/bootstrap prerequisites are
      certified.
- [x] Certificate-time boundary fits are not automatically labeled as
      post-selection.

### Optimizer Certificate Acceptance

- [x] Optimizer certificates separate optimizer stop evidence from statistical
      fit status.
- [x] Optimizer certificates record parameter-space evidence: `theta` count,
      free count, boundary count, and boundary indices.
- [x] Optimizer certificates record sample-size context: observations,
      `theta` count, and observations per `theta`.
- [x] Optimizer certificates include explicit gradient and Hessian evidence
      records, with unavailable derivative paths represented as structured
      `NotAvailable` evidence rather than omitted fields.
- [x] Audit reports surface optimizer stop evidence, parameter-space context,
      sample-size context, gradient evidence, Hessian evidence, and overall
      certification quality.
- [x] LMMs expose a bounded `verify_convergence()` workflow that records
      restart, jittered-start, and alternate-optimizer agreement checks as
      structured optimizer-certificate evidence.
- [ ] Real derivative-backed KKT/Hessian checks remain open
      (`bd-01KQ7X05J0PXWDCAF808479XAP`).

### Reproducibility and Performance Acceptance

- [x] Compiler decisions are deterministic for fixed inputs.
- [x] Diagnostic ordering is stable.
- [x] Benchmark hooks exist for formula-to-IR, explanation, and design audit
      (`examples/bench_compiler_contract.rs`,
      `bd-01KQ7X0MS57DRCGKDTQY0B5EVJ`).
- [x] Benchmark hooks exist for failure-path diagnosis
      (`examples/bench_compiler_contract.rs`,
      `bd-01KQ7X0MS57DRCGKDTQY0B5EVJ`).
- [x] No v0 diagnostic requires dense row-by-row covariance matrix materializing
      by default.
- [x] A reproducibility record is serialized with fit/audit artifacts.
- [x] A pathology-corpus foundation asserts fitted status membership against
      certificate-derived status sets rather than single optimizer outcomes.

### GLMM Boundary Acceptance

- [x] GLMM artifacts can record family/link and objective approximation.
- [x] GLMM optimizer certificates state whether they apply to an exact or
      approximated objective.
- [x] LMM-only finite-sample inference methods are marked unsupported for GLMMs
      unless a derivative/approximation certificate exists.

### Print-Layer Acceptance

- [x] Default print output shows one canonical summary rather than all internal
      model views — `LinearMixedModel`/`GeneralizedLinearMixedModel` `Display`
      delegates to `print_summary()` returning [`compiler::ModelPrint`], which
      shows fit status, formula(s), top diagnostics, inference availability,
      and a one-line drilldowns pointer (`bd-01KQ7X0RPMTTQ88MTRYNFC60YP`).
- [x] Drilldowns exist for explanation (`explain_model()` →
      `ModelExplanation`), audit (`audit_report()` → `ModelAuditReport`),
      parameterization (`parameterization()` → `compiler::ParameterizationDrilldown`,
      grouping the artifact's `covariance_parameter_traces` per random term
      with source syntax, basis, θ/Λ/parmap/VarCorr slots), and
      requested/effective changes (`changes()` →
      `Vec<ModelStateChange>`) (`bd-01KQ7X0RPMTTQ88MTRYNFC60YP`).

### Worked Example Acceptance

The contract must be validated against five worked examples before v0 is
considered complete:

- [x] `sleepstudy`-scale random intercept/slope LMM.
- [x] crossed subject/item design.
- [x] supported-rank mixture case with user-scale loadings, e.g. a 0.7/0.3
      intercept/slope direction.
- [x] confounded fixed/random structure, e.g. subject fixed effects plus
      subject random intercept.
- [x] glmer-style logistic model with a small grouping factor.

Each worked example must produce a deterministic artifact containing requested
model, semantic model, compiled basis, ThetaMap, design audit, effective/fitted
model state, diagnostics, and explanation. This is now implemented for the
initial five v0 examples; the additional `singular` fixture pins the
too-rich/reduced-rank covariance story.

## Mote Issue Index

Local mote tracking has been initialized for the remaining work. The issue IDs
below are intentionally grouped by implementable slices rather than one issue
per checklist line.

| ID | Priority | Area | Summary |
|---|---:|---|---|
| `bd-01KQ7WYW472SRF7NPG9P1HDMSM` | 1 | formula/diagnostics | Add formula canonicalization diagnostics and written source-syntax preservation. |
| `bd-01KQ7WZ5ZTVQETY5PN3F5KHF02` | 1 | formula/basis | Completed categorical/cell-means/random-interaction basis manager, expanded-basis audit, and ThetaMap optimizer-basis traceability. |
| `bd-01KQ7WZF56ASNYE240MMG0GWWF` | 1 | formula/decisions | Completed v0 decisions on thresholds, separators, level ordering, covariance conflicts, R7 behavior, `||` centering, and random interaction coding. |
| `bd-01KQ7WZK8G0EME3K6ZX539883D` | 1 | artifact/reductions | Completed computed `ModelStateSummary` and `changes()` view for requested, semantic, supported, and fitted model state. |
| `bd-01KQ815MNJ76WJT109WEJGZKRT` | 1 | audit/reporting | Completed random-effect information-budget/effective-n audit report with grouping-level support ratios and action-oriented recommendations. |
| `bd-01KQ7WZQFWZQW1VVARWF6Y9ZYS` | 1 | policy | Completed initial executable `design_compiled` path for deterministic full-to-diagonal covariance reduction and refusal before optimization. |
| `bd-01KQ8J33DB4HET0F56836TDX2K` | 1 | policy | Completed few-level random-effect policy refinement: permissive scalar random intercepts, reliability warnings, and stricter slope/correlation gates. |
| `bd-01KQ7WZVTTHMF5VWK355NMPR6G` | 2 | audit/kernels | Completed initial covariance-kernel/dependence-path graph, repeated-unit under-modeling diagnostics, and audit-report surfacing. |
| `bd-01KQ7X00EAAK9F6MN9EEWFTC9P` | 1 | theta-map | Completed covariance parameter traces from source syntax through semantic/optimizer basis, `theta`, `Lambda`, `parmap`, and VarCorr entries. |
| `bd-01KQ7XZW4QY5R9Q5X7QE35XMCE` | 1 | optimizer | Completed structured convergence evidence fields on optimizer certificates and audit reporting for stop, parameter-space, sample-size, gradient, Hessian, and certification-quality evidence. |
| `bd-01KQ7Y01ZMJB6BES26FF4K21WG` | 1 | optimizer | Completed bounded `verify_convergence()` restarts, jittered starts, alternate-optimizer consensus checks, and audit-report surfacing. |
| `bd-01KQ8FRGFQEQT8J217YB02D7CB` | 1 | testing | Completed pathology-corpus foundation with pure certificate, four fixture strata, one near-singular transform, and status-set membership test. |
| `bd-01KQ7X05J0PXWDCAF808479XAP` | 2 | optimizer | Implement real KKT and derivative certificate checks. |
| `bd-01KQ7X0F5808VDKGCAM88Z3P95` | 2 | inference | Build estimability and contrast-first inference scaffold. |
| `bd-01KQ7X0MS57DRCGKDTQY0B5EVJ` | 2 | performance | Add compiler/audit and failure-path benchmarks. |
| `bd-01KQ7X0RPMTTQ88MTRYNFC60YP` | 2 | UX/reporting | Design default print and drilldown API for compiler artifacts. |
| `bd-01KQ7X0YPQ4TWA0P5J35SY5ZDJ` | 2 | R/wire | Define future R client wire schema expectations. |
| `bd-01KQ7X12J1TGDA6MFM3E5KQDJE` | 3 | vNext | Track multivariate shared-theta design. |
| `bd-01KQ7X17P91CW715ZRY16H7CTX` | 3 | vNext | Track deferred residual, comparison, prediction, bootstrap, and KR architecture. |

## Notes

### Architectural Notes

- The parameterization map is a Phase-0 contract, not an optimization detail.
  If the basis changes, the `theta` layout changes. Gradients, Hessians, KKT
  checks, derivative APIs, `parmap`, and output back-transforms all depend on
  this mapping.
- ThetaMap/CovarianceMap variants must represent distinct manifolds. Do not
  encode diagonal or reduced-rank covariance merely as full covariance with
  active zeros.
- `maximal_feasible` must start as a deterministic rule. A general model lattice
  can come later after the rule has been exercised on real designs.
- `as_specified` means fit exactly the requested structure or refuse. It does
  not mean "fit something close and warn."
- Boundary fits are certificate-time events. They are response-dependent, but
  they are not model-search events unless the engine compares or selects among
  alternative structures.
- Regularized mode is exploratory in this contract. It must not print ordinary
  p-values. Confirmatory p-values require an unpenalized, predeclared or
  transparently reduced structure and a valid inference status.
- Residual structures are intentionally out of v0 except for acknowledging the
  future split between structures compatible with the current sparse engine and
  structures requiring a different solver.
- R implementation is out of scope, but versioned JSON serialization is in
  scope. R should be a client of Rust-owned diagnostics later.
- GLMMs share the compiler contract but not the LMM inference contract.

### Implementation Notes

- Start by adding types and tests without changing the numerical fitting path.
- Prefer crate-public APIs first if public API naming is not stable.
- Use stable ordering for maps and diagnostics, likely via existing ordered
  collections where appropriate.
- Keep diagnostic messages short; put detailed machine-readable context in
  payloads.
- Avoid adding new abstractions that cannot be exercised by a test in v0.
- Treat future architecture sections in
  `docs/mixed_model_compiler_inference_contract.md` as non-binding until this
  PRD is implemented.

### Suggested First Issues

1. Add diagnostic/audit skeleton types with serde derives where feasible.
2. Add semantic random-effect IR types.
3. Add sum-typed ThetaMap/CovarianceMap types that wrap or reference current
   `parmap`.
4. Compile current random-effect formula terms into semantic IR.
5. Add canonicalization tests for crossing, nesting, cell-only, diagonal, and
   slope-only syntax.
6. Add prefit `explain_model()` backed by semantic IR.
7. Add grouping-level and within-group variation audit.
8. Add fixed-effect rank and empty-cell audit.
9. Add information-budget reporting for random-effect covariance parameters.
10. Add deterministic `maximal_feasible` v0 thresholds and policy options.
11. Add JSON schema-versioned serialization tests for contract artifacts.
12. Add the five worked examples as expected artifacts.
13. Wire diagnostics and requested/effective metadata into the fit object
    without changing numerical results.

### Explicitly Deferred

- Kenward-Roger
- derivative implementation beyond the v0 strategy contract
- full reduced-rank optimizer parameterization
- adaptive bootstrap implementation
- residual AR(1), spatial, and Matern structures
- R package interface
- influence diagnostics requiring many refits
- automatic regularized covariance search
