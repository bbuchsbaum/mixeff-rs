# Mixed Model Compiler and Inference Contract

Working notes from design discussion, part 1.

Implementation note: the binding near-term implementation contract lives in
`docs/compiler_contract_v0_prd.md`. The formula-layer slice of that contract
is single-sourced in `docs/random_effects_formulas.md` (random-effects
parsing, canonicalization, basis construction, grouping factor materialization,
diagnostics inventory). The focused row-level p-value support plan lives in
`docs/fixed_effect_p_values_plan.md`. This document is the broader architecture
and product vision; later-phase details here should be treated as non-binding
until they are promoted into a PRD or implemented contract. Sections below covering
canonical nesting/crossing semantics, intercept omission mechanics, and basis
centering specifics are now superseded by the formulas doc; the discussion is
preserved for design rationale only.

Current implementation status, 2026-04-27: the project is in Compiler Contract
v0 implementation, not the full inference product. The crate now has structured
diagnostics, semantic random-effect IR, design audit, sum-typed `ThetaMap`,
compiled artifacts, model-audit reports, optimizer certificates, GLMM boundary
metadata, effective covariance/rePCA-style summaries, and JSON fixtures for the
initial worked examples plus the `singular` too-rich covariance fixture. The
remaining work is tracked locally in mote; the umbrella issue is
`bd-01KQ7WW46RJ6B71TAPZ0CK6KVJ`, and the detailed issue index is in
`docs/compiler_contract_v0_prd.md`.

## Contract Version Log

Pathology-corpus TOML fixtures carry a `contract_version` field. The current
pathology corpus contract is `v0.3`, single-sourced as
`PATHOLOGY_CORPUS_CONTRACT_VERSION` in `src/pathology/certificate.rs`.

| Version | Date | Scope | Migration rule |
| ------- | ---- | ----- | -------------- |
| `v0.3` | 2026-04-29 | Pathology corpus expected `FitStatus` sets after separation and penalised-convergence support. | Baseline stamp for every existing `tests/fixtures/pathology_corpus/*.toml` fixture; future bumps must re-evaluate each expected-status set by hand and update this log with the review rationale. |

The long-term goal is a Rust mixed-model engine that can stand alone as a
self-contained crate while also serving as the computational and diagnostic
core for an R layer. The R interface should be a client of the Rust engine, not
the place where statistical coherence is patched in after fitting.

The product is not "a quieter lmer." It is a mixed-model compiler with an
explicit inference contract:

> Given a formula and data, the system either returns a statistically coherent
> model with certified convergence and explicit finite-sample inference, or it
> refuses or simplifies the model with a specific design-level reason.

The Bates/lme4 architecture remains the right computational starting point:
formula parsing, construction of fixed and random effect design structures,
profiled ML/REML objective over covariance parameters, sparse penalized
least-squares solves, optimizer, and output layer. The new requirement is that
this pipeline expose stronger contracts at every boundary.

## Core Principles

1. No fake certainty.
   If a p-value is not defensible, return `NA`/missing with a reason rather
   than printing a plausible number.

2. No raw optimizer folklore.
   Users should see statistical diagnostics and identifiability explanations,
   not raw messages such as "degenerate Hessian" or "unable to evaluate scaled
   gradient."

3. No hidden model surgery.
   If a covariance structure is reduced, zeroed, diagonalized, or refit, the
   fitted object records the requested model, the effective model, and the
   reason for the change.

4. Fast path for normal cases, safe path for hard cases.
   Ordinary LMMs should fit quickly and get Satterthwaite/KR-style inference
   when feasible. Pathological designs should trigger preflight refusal,
   transparent reduction, regularized exploration, bootstrap validation, or no
   inference.

## Crate Boundary and R Layer

The Rust crate should own:

- formula representation after parsing
- semantic model IR
- data schema and model frame validation
- declared/inferred variable roles
- design graph and covariance-kernel graph
- design-matrix construction
- fixed-effect and random-effect term metadata
- basis management and formula canonicalization
- design preflight/audit
- internal scaling and canonicalization
- information-budget and model-lattice construction
- covariance-structure compilation
- profiled ML/REML objective
- optimization and KKT certification
- derivative and information APIs
- fixed-effect contrast inference
- bootstrap simulation machinery
- structured diagnostics
- serialization of fit/audit/inference results

The R layer should own:

- R formula/data capture
- R factor/contrast policy translation into the Rust model frame
- idiomatic R object classes and print methods
- compatibility helpers for lme4-like syntax
- user-facing convenience functions such as `summary()`, `anova()`, `drop1()`,
  `VarCorr()`, `ranef()`, `fixef()`, `predict()`, `audit()`

The R layer should not decide whether a model converged, whether a random-effect
term is identifiable, or whether a p-value is defensible. Those decisions belong
inside the Rust engine so the crate remains coherent and testable on its own.

## High-Level Architecture

The desired engine is a sequence of explicit compiler stages:

```text
formula + data
  -> model frame/schema
  -> formula AST
  -> semantic model IR
  -> design graph / covariance-kernel graph
  -> design compiler
  -> information budget and model lattice
  -> scaling/canonicalization plan
  -> covariance structure compiler
  -> profiled ML/REML engine
  -> constrained optimizer
  -> KKT/convergence certificate
  -> active/effective model
  -> finite-sample inference layer
  -> structured fit object + audit report
```

Each stage should emit machine-readable diagnostics. Later stages should not
silently repair invalid earlier stages.

## Inference Contract

Every completed fit should have an explicit top-level status:

```rust
pub enum FitStatus {
    ConvergedInterior,
    ConvergedBoundary,
    ConvergedReducedRank,
    ConvergedPenalised,
    NotIdentifiable,
    NotOptimized,
}
```

Suggested meanings:

- `ConvergedInterior`: all active covariance parameters are interior, KKT and
  Hessian checks pass on the full active parameter space.
- `ConvergedBoundary`: one or more variance components are on a valid boundary;
  KKT signs and active-subspace curvature pass.
- `ConvergedReducedRank`: requested covariance structure was singular or
  over-rich, but an effective lower-rank structure is identifiable and fitted.
- `ConvergedPenalised`: the **maximum-likelihood** estimate does not exist
  (likelihood unbounded — typically fixed-effect or conditional separation in
  a logistic GLMM), but the user opted into a penalised path (Firth, ridge,
  weakly-informative prior) and the *penalised* objective has a unique
  optimum. The fit is honest about being a penalised estimate rather than an
  MLE; consumers must inspect the artifact's penalty method and ML-non-
  existence reason before treating point estimates as MLE-valid for things
  like profile likelihood or Wald statistics.
- `NotIdentifiable`: design or fitted information cannot support the requested
  fixed/random structure and no allowed reduction resolves it.
- `NotOptimized`: numerical optimization failed before a certificate could be
  issued.

This status should be supported by lower-level certificates rather than by
optimizer exit codes alone.

### Refusal vs `ConvergedPenalised` decision tree

Whenever the design or response makes the MLE non-existent, the engine has
exactly two honest answers — `NotIdentifiable` (refuse) or
`ConvergedPenalised` (penalise). Choose between them with this tree:

1. **Does the MLE exist?**
   - **Yes** — return the appropriate `Converged*` status. Penalised paths
     do **not** apply here; reporting `ConvergedPenalised` for a fit whose
     MLE exists would mislabel a regular maximum likelihood estimate.
   - **No** — proceed.
2. **Did the caller opt into a penalty (e.g. `fit(..., penalty = firth())`),
   and does the engine support that penalty for this family/link?**
   - **No** — return `NotIdentifiable` and surface the structural diagnostic
     (separation kind, reduced-rank direction, etc.). Refusal is the default
     for non-existent MLEs; quietly substituting a penalty would violate
     the contract's no-hidden-model-surgery principle.
   - **Yes** — proceed.
3. **Does the *penalised* objective have a unique optimum that satisfies
   the optimizer KKT certificate?**
   - **No** — return `NotIdentifiable`. A penalty alone does not rescue an
     irreducibly degenerate design (e.g. perfect FE collinearity); refusal
     stays the right answer.
   - **Yes** — return `ConvergedPenalised`. The artifact must record the
     penalty method, the penalty strength, and the ML-non-existence reason
     so consumers can decide whether to trust the point estimate for their
     downstream task.

`NotOptimized` is orthogonal to this tree: it covers numerical failure of
the optimizer (line search collapse, NaN propagation) regardless of whether
the underlying problem is identifiable. A penalised fit whose optimizer
failed remains `NotOptimized`, not `ConvergedPenalised`.

The pathology corpus exercises the contract end-to-end: separation-stratum
fixtures admit `{NotIdentifiable, NotOptimized, ConvergedPenalised}` in
their certificate's expected status set
(`src/pathology/certificate.rs::expected_statuses`), and a fit that lands
outside that set is a contract regression. Proper LP-based separation
detection lands under `bd-01KQ8FS7HK6TX2TMX0J0XFGYFD`; until then the
certificate uses the placeholder Bernoulli + extreme-intercept-shift
heuristic to gate the branch.

## KKT-Certified Optimization

Variance and covariance parameters are constrained optimization parameters.
Boundary solutions are normal, not automatically failures.

For a final covariance parameter vector, check:

- free parameters: gradient approximately zero
- boundary parameters: projected/KKT gradient has the correct sign
- free Hessian or observed information: positive semidefinite on the active
  subspace
- identified rank: full rank on the active parameter subspace
- inactive or unsupported directions: clearly marked as boundary, aliased,
  reduced, or non-estimable

Proposed certificate structure:

```rust
pub struct OptimizerCertificate {
    pub status: FitStatus,
    pub optimizer_name: String,
    pub objective_value: f64,
    pub iterations: usize,
    pub free_gradient_norm: f64,
    pub projected_gradient_norm: f64,
    pub active_set: ActiveSet,
    pub hessian_eigen_min: Option<f64>,
    pub hessian_rank: Option<usize>,
    pub information_rank: Option<usize>,
    pub checks: Vec<CertificateCheck>,
}

pub enum CertificateCheck {
    FreeGradientOk { tolerance: f64, value: f64 },
    BoundaryGradientOk { tolerance: f64, value: f64 },
    HessianPsdOnActiveSubspace { min_eigenvalue: f64 },
    RankOk { rank: usize, expected: usize },
    Failed { code: DiagnosticCode, message: String },
}
```

Important user-facing policy:

- A zero variance component is an active boundary, not a convergence failure.
- A correlation involving a zero variance component is not estimated.
- A negative Hessian eigenvalue in an unidentifiable direction should not be
  shown as "degenerate Hessian"; the covariance block is reduced or the model is
  rejected.

## Design Compiler and Preflight Audit

Most unusable mixed models are visible before optimization. The engine should
inspect the fixed-effect design, random-effect structure, grouping factors, and
requested covariance parameters before fitting.

For each random-effects term, compute at least:

- grouping factor name
- number of grouping levels
- number of observations per level
- balance/imbalance summaries
- number of random coefficients per group
- number of covariance parameters requested
- within-group rank of the random-slope design
- global rank of `X`
- global rank/structural rank of `Z`
- crossing or nesting relationships
- confounding between fixed and random effects
- whether the grouping factor appears to encode experimental conditions
- expected information rank for variance/covariance parameters
- whether random slopes have within-group variation
- whether random-slope correlations are estimable

Term-level classification:

```rust
pub enum RandomTermEstimability {
    Estimable,
    WeaklyEstimable,
    CorrelationNotEstimable,
    SlopeNotEstimable,
    GroupingFactorNotRandomEnough,
    RankDeficient,
}
```

Model-level design output:

```rust
pub struct DesignAudit {
    pub fixed_effects: FixedEffectAudit,
    pub random_terms: Vec<RandomTermAudit>,
    pub nesting: NestingAudit,
    pub crossing: CrossingAudit,
    pub confounding: Vec<ConfoundingDiagnostic>,
    pub recommendations: Vec<ModelRecommendation>,
    pub blocking_diagnostics: Vec<Diagnostic>,
}
```

A maximal model should mean:

> Include scientifically relevant random-effect directions when the design
> contains information for them.

It should not mean:

> Estimate a full unstructured covariance matrix for a small number of groups
> and hope the optimizer succeeds.

The compiler must distinguish:

- unsupported grouping structure
- supported grouping with unsupported slopes
- supported slopes with unsupported correlations
- weak but possible covariance estimation
- valid boundary solution after fitting

## Example Diagnostic Policy

For a model like:

```r
effect ~ duration + (1 + duration | sites) + (1 + duration | season)
```

where `season` has three levels and represents conditions of interest, and
`sites` also has very few levels, the compiler should not merely try alternate
optimizers. It should diagnose the design:

- `(1 + duration | season)` is not a defensible random-effect distribution if
  `season` has only three levels and its levels are the conditions being
  compared.
- `season` should usually be treated as fixed.
- `(1 + duration | sites)` requests an intercept variance, slope variance, and
  covariance from too few site levels.
- the random slope/correlation structure is weakly identified or not
  identifiable.
- if sites are the actual sampled population units and there are many of them
  in the real study, a model like `duration * season + (1 | site)` may be
  reasonable.
- if there are only a few named sites and they are themselves conditions of
  interest, sites should probably be fixed.

The diagnostic should be design-level and actionable. It should not surface raw
optimizer internals as the main explanation.

## Scaling and Canonicalization

Users should not have to manually discover that poor predictor scaling caused
optimization instability. The engine should separate three scales:

- user scale: what the user wrote and what output is reported on
- canonical scale: what the optimizer and sparse linear algebra use
- inference scale: where contrasts, predictions, and hypothesis tests are
  evaluated

Internal preprocessing should support:

- centering and scaling continuous fixed-effect columns
- centering and scaling random-slope columns
- preserving factor/contrast semantics
- orthogonalizing random-slope bases where possible
- detecting near-collinearity
- fitting on the stable internal scale
- back-transforming coefficients, standard errors, contrasts, predictions, and
  variance/covariance output

Suggested representation:

```rust
pub struct ScalingPlan {
    pub fixed_columns: Vec<ColumnTransform>,
    pub random_columns: Vec<RandomSlopeTransform>,
    pub contrast_transforms: Vec<ContrastTransform>,
    pub canonical_to_user: LinearMap,
    pub user_to_canonical: LinearMap,
}
```

Special attention is needed for random slopes. Correlated random intercept/slope
models are invariant to additive shifts of the predictor in a way that
zero-correlation models are not. Therefore, the covariance compiler must know
which transformations preserve the requested model and which change its
meaning.

## Covariance Structure Compiler

Replace informal "keep it maximal" advice with a compiler strategy:

1. Fixed-effect estimand is sacred.
2. Random-effect grouping structure must match the design.
3. Random slopes are included only when the design supports them.
4. Covariance/correlation parameters are estimated only when identifiable.
5. Unsupported variance directions are zeroed, reduced-rank, or rejected.

The user may request a covariance model, but the engine needs an effective
covariance model:

```rust
pub enum CovarianceStrategy {
    AsSpecified,
    MaximalFeasible,
    Regularized,
}

pub enum EffectiveCovarianceStructure {
    Unstructured,
    Diagonal,
    LowRank { rank: usize },
    ZeroComponent { component: String },
    Rejected { reason: Diagnostic },
}
```

A spectral representation is the most informative way to report reduced-rank
fits:

```text
G = Q diag(lambda_1, ..., lambda_r, 0, ..., 0) Q'
```

The fitted rank becomes part of the model result. This lets the engine say:

> Groups vary in intercept and in the intercept+duration direction, but the
> independent duration-slope direction is not supported.

That is more useful than "singular fit."

## Confirmatory and Regularized Modes

There should be two explicitly different modes.

### Confirmatory Mode

```r
mode = "confirmatory"
```

Fit the unpenalized ML/REML model after preflight simplification. If a variance
is zero, report a boundary fit. If the model is not identifiable, refuse or
apply only explicitly allowed reductions. This is the mode for traditional
p-values.

### Regularized Exploration Mode

```r
mode = "regularized"
penalty = "variance_shrinkage" | "correlation_shrinkage" | "reduced_rank"
```

Use penalized ML/REML to stabilize or discover covariance structure. This is
exploratory unless followed by an unpenalized refit. Do not silently regularize
and print ordinary confirmatory p-values.

Preferred workflow:

1. use regularization to discover a stable covariance structure
2. refit the selected active structure with unpenalized REML
3. compute inference conditional on that structure
4. label the inference as post-selection/exploratory

## Contrast-First Fixed-Effect Inference

The inference layer should not ask which p-value belongs in a coefficient row.
It should ask what hypothesis is being tested:

```text
H0: L beta = c
```

Every fixed-effect test should reduce to a contrast or contrast family:

```rust
pub struct FixedEffectHypothesis {
    pub label: String,
    pub l: ContrastMatrix,
    pub rhs: ContrastRhs,
}

pub struct FixedEffectTest {
    pub estimate: EstimateValue,
    pub standard_error: Option<f64>,
    pub statistic: Option<f64>,
    pub numerator_df: Option<f64>,
    pub denominator_df: Option<f64>,
    pub p_value: Option<f64>,
    pub method: InferenceMethod,
    pub reliability: ReliabilityGrade,
    pub reliability_reason: FixedEffectReliabilityReasonCode,
    pub status: InferenceStatus,
}
```

`reliability` is the coarse grade for consumers that only need a traffic-light
signal. `reliability_reason` is the stable vocabulary for why that grade was
assigned:

- `interior_converged_well_specified`: asymptotic Wald z used on an interior,
  well-specified fit.
- `asymptotic_wald_z_at_boundary`: asymptotic Wald z used despite a boundary or
  reduced-covariance fit.
- `degrees_of_freedom_unavailable_so_z_substituted`: finite-sample degrees of
  freedom were requested but unavailable, so Wald z was used as the labeled
  fallback.
- `satterthwaite_finite_difference_approximation`: Satterthwaite inference was
  computed from finite-difference variance-parameter derivatives.
- `kenward_roger_approximation`: Kenward-Roger inference was used.
- `bootstrap_monte_carlo_replicates`: parametric bootstrap inference was used.
- `inference_unavailable_by_policy`: p-values were withheld because the fit
  intent or reduction policy makes ordinary fixed-effect p-values invalid.
- `contrast_not_estimable`: the requested contrast touches aliased or otherwise
  non-estimable coefficient directions.
- `standard_error_unavailable`: the statistic or p-value is unavailable because
  the fixed-effect standard error is unavailable.

Low grouping-level support remains a model/audit diagnostic in schema `1.0.0`,
not a fixed-effect row `reliability_reason`. It can explain why covariance
support is weak or why ordinary p-values are withheld by policy, but the row
reason should name the inference rule that actually produced or withheld the
row value.

For single-df contrasts:

```text
t = (L beta_hat - c) / sqrt(L Var(beta_hat) L')
```

For multi-df terms, use an F-test.

Default inference stack:

- Satterthwaite t/F as the fast default
- Kenward-Roger F for small-sample/high-accuracy mode when feasible
- parametric bootstrap as calibration or fallback
- asymptotic z/chi-square only as a large-sample fallback

Policy:

- `summary()` can default to Satterthwaite for fixed-effect contrasts.
- `anova()` can use Kenward-Roger when feasible, otherwise Satterthwaite.
- `drop1()` should use Kenward-Roger or parametric bootstrap for serious model
  comparison.
- random effects should not get naive ordinary p-values; use boundary-aware
  tests, restricted likelihood ratio tests in simple cases, or bootstrap.

Every p-value row should carry its method and status:

```text
duration:season  F = 4.92, ndf = 2, ddf = 14.7, p = 0.023
method: Kenward-Roger
status: OK
```

If inference is not defensible:

```text
p = NA
reason: denominator df below threshold after covariance reduction
```

## Derivative API for Satterthwaite and Kenward-Roger

The Rust engine can improve on R-level wrappers by exposing exact or
high-quality derivatives from the core objective and linear algebra.

Needed engine methods:

```rust
pub trait DifferentiableMixedModelObjective {
    fn objective(&self, theta: &[f64]) -> f64;
    fn gradient(&self, theta: &[f64]) -> Vec<f64>;
    fn hessian(&self, theta: &[f64]) -> Matrix;
    fn beta_at(&self, theta: &[f64]) -> Vector;
    fn vcov_beta_at(&self, theta: &[f64]) -> Matrix;
    fn d_vcov_beta_d_theta(&self, theta: &[f64]) -> Vec<Matrix>;
    fn d2_vcov_beta_d_theta2(&self, theta: &[f64]) -> Vec<Vec<Matrix>>;
    fn information_theta(&self, theta: &[f64]) -> Matrix;
}
```

For Satterthwaite, the essential quantity is the derivative of contrast
variance with respect to covariance parameters:

```text
v(theta) = L Var(beta_hat | theta) L'

df ~= 2 v(theta_hat)^2 /
      (grad v(theta_hat)' Var(theta_hat) grad v(theta_hat))
```

For Kenward-Roger, the engine needs first and second derivative information for
the covariance of `beta_hat`, plus adjusted F statistics and denominator df.

Design implication:

- derivative support should not be an afterthought bolted onto summaries
- objective evaluation, factorization, and derivative calculation should share
  cached symbolic structure where possible
- the fit object should retain enough information to compute derivatives for
  requested contrasts after fitting

The row-level fallback contract in `docs/fixed_effect_p_values_plan.md` is the
near-term exception to finite-sample deferral: coefficient and scalar contrast
rows may expose labeled `asymptotic_wald_z` p-values with low reliability,
null finite-sample degrees of freedom, and notes that they are not
Satterthwaite or Kenward-Roger corrections.

## Adaptive Parametric Bootstrap

Bootstrap should validate difficult inference, not become the universal default.

Trigger bootstrap when:

- small number of groups
- tested effect is near a boundary fit
- severe imbalance
- Satterthwaite/KR denominator df is extremely low
- p-value is near a user-specified alpha threshold
- covariance simplification or post-selection occurred
- derivative-based approximations fail reliability checks

Adaptive policy:

1. run 199 simulations
2. stop if the p-value is clearly far from the decision threshold
3. continue to 999, 1999, or 4999 when near the threshold
4. parallelize simulations
5. reuse symbolic factorizations
6. reuse warm starts
7. report Monte Carlo uncertainty

Example output:

```text
p = 0.041, MC SE = 0.006, B = 1999
```

The bootstrap result should include the data-generating fitted model, refit
policy, convergence failure count, boundary count, and whether failed simulated
fits were excluded or counted conservatively.

## Structured Diagnostics

Warnings should be structured objects with stable codes, severity, affected
terms, and suggested actions.

```rust
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub severity: DiagnosticSeverity,
    pub stage: DiagnosticStage,
    pub message: String,
    pub affected_terms: Vec<String>,
    pub suggested_actions: Vec<SuggestedAction>,
}
```

Stages:

- formula parsing
- model frame construction
- fixed-effect design compilation
- random-effect design compilation
- covariance compilation
- scaling/canonicalization
- optimization
- convergence certification
- inference
- bootstrap

Example user-facing diagnostic:

```text
Fit status: converged_reduced_rank

The requested covariance structure for (1 + duration | sites) has rank 2, but
the data support rank 1. The duration random-slope variance was estimated as 0.
The effective model uses a random intercept for sites.

Fixed-effect inference was computed using Satterthwaite df. The p-values are
conditional on the reduced random-effects structure.
```

Example refusal:

```text
Fit status: not_identifiable

The term (1 + duration | season) cannot be estimated as a random effect:
season has 3 levels and duration has 2 levels. Use season as a fixed effect, or
collect more season-like levels.

No p-values were computed.
```

## R API Around Intent

Keep familiar lme4-like syntax, but expose intent-aware controls.

Basic use:

```r
fit <- lmm(
  log(effect) ~ duration * season * sites,
  data = dat1,
  inference = "auto"
)
```

Mixed model:

```r
fit <- lmm(
  y ~ duration * season + (duration | site),
  data = dat,
  random_strategy = "maximal_feasible",
  inference = "auto"
)
```

Important controls:

```r
random_strategy =
  "as_specified"        # lme4-like
  "maximal_feasible"    # conservative default
  "regularized"         # exploratory shrinkage

inference =
  "satterthwaite"
  "kenward-roger"
  "bootstrap"
  "auto"
  "none"

on_nonidentifiable =
  "error"
  "reduce"
  "regularize"
```

Default policy:

- if the model is coherent, fit it
- if the covariance model is too rich but reducible, reduce it transparently
- if the fixed-effect test is not defensible, withhold the p-value

The R print layer should display both requested and effective formulas when
they differ.

## Fit Object Contents

A robust fit object should retain:

- original call/formula
- normalized formula AST
- semantic random-effects IR
- native model spec when supplied
- model frame schema
- declared or inferred variable roles
- contrast coding
- fixed-effect and random-effect basis manager
- design graph
- covariance-kernel graph
- requested fixed-effect and random-effect structures
- effective fixed-effect and random-effect structures
- information budget
- candidate model lattice summary
- scaling/canonicalization plan
- covariance strategy and effective covariance structure
- profiled objective type: ML or REML
- optimizer trace summary
- KKT/convergence certificate
- fixed-effect estimates
- random-effect conditional modes
- variance/covariance estimates on user scale
- random-effect PCA/reduced-rank summary
- derivative/information cache or recomputation handles
- inference table with method/status/reliability
- fixed-effect sensitivity audit
- explanation/covariance story
- diagnostics
- audit report

## `audit(fit)` / `validate_fit()`

The package should ship an audit function that returns a concise, structured
health report:

```text
design rank: OK
fixed effects: estimable
random effects: reduced-rank
optimizer: KKT OK
inference: Satterthwaite, grade B
recommendation: confirm with bootstrap for duration:season
```

Programmatic representation:

```rust
pub struct FitAudit {
    pub design: DesignAudit,
    pub optimizer: OptimizerCertificate,
    pub covariance: CovarianceAudit,
    pub inference: InferenceAudit,
    pub recommendation: Vec<ModelRecommendation>,
}
```

## Additional lme4 Pain Points and Product Requirements

Part 2 of the design discussion adds five important pain points beyond
p-values, convergence messages, and maximal random-effects folklore:

```text
lme4 pain point               Engine/product answer
---------------------------------------------------------------
REML/ML confusion             intent-aware comparison
rank deficiency               estimability compiler
residual structure limits     residual covariance module
prediction ambiguity          explicit prediction targets
scattered diagnostics         first-class model audit report
```

The broader pattern is that the engine should understand the user's
statistical intent before producing polished numerical output.

## Intent-Aware Model Comparison

Users should not need to remember informal rules such as "REML for estimation,
ML for fixed-effect comparison." The comparison API should ask what the user is
trying to compare:

```r
compare(fit1, fit2, target = "fixed_effects")
compare(fit1, fit2, target = "random_effects")
compare(fit1, fit2, target = "prediction")
```

The engine should select the appropriate comparison criterion:

- fixed-effect comparison: ML likelihood comparison where appropriate, or
  KR/Satterthwaite contrast tests when the comparison is expressible as fixed
  effect hypotheses
- variance-structure comparison: REML-compatible comparison where valid, with
  boundary-aware tests or bootstrap when required
- predictive comparison: cross-validation, marginal likelihood, expected
  log-predictive density, or a clearly labeled AIC-style criterion

The comparison result must record whether models were refit and why:

```text
These models differ in fixed effects and were fitted by REML.
They were refit by ML for this comparison.
Fixed-effect estimates shown in the original summaries remain REML estimates.
```

Invalid comparisons should refuse or return a structured diagnostic rather than
silently emitting a likelihood-ratio table.

Proposed types:

```rust
pub enum ComparisonTarget {
    FixedEffects,
    RandomEffects,
    Prediction,
}

pub enum ComparisonCriterion {
    MlLikelihoodRatio,
    RemlLikelihoodRatio,
    KenwardRogerContrast,
    SatterthwaiteContrast,
    ParametricBootstrap,
    CrossValidation,
    InformationCriterion,
}

pub struct ModelComparison {
    pub target: ComparisonTarget,
    pub criterion: ComparisonCriterion,
    pub refit_actions: Vec<RefitAction>,
    pub rows: Vec<ModelComparisonRow>,
    pub diagnostics: Vec<Diagnostic>,
}
```

Design requirements:

- retain both original fit criterion and comparison criterion
- retain original summaries separately from refit summaries
- detect fixed-effect differences, random-effect differences, residual-model
  differences, and response/data incompatibilities
- explain when a comparison is conditional on a selected/reduced covariance
  structure
- use bootstrap or refuse when boundary null distributions make ordinary
  chi-square approximations unreliable

## Estimability Compiler for Fixed Effects

Rank deficiency should be diagnosed as a design property, not hidden in a
dropped-column coefficient table. Before fitting, inspect:

- `rank(X)`
- aliased fixed-effect columns
- empty factor combinations
- unsupported high-order interactions
- contrast coding
- requested coefficient-level and term-level hypothesis matrices
- whether each requested contrast is estimable under the observed design

Term and contrast outputs should carry explicit estimability status:

```rust
pub enum EstimabilityStatus {
    Estimable,
    PartiallyEstimable,
    NotEstimable,
}

pub struct EstimabilityDiagnostic {
    pub status: EstimabilityStatus,
    pub term: String,
    pub aliasing_columns: Vec<String>,
    pub empty_cells: Vec<CellDescriptor>,
    pub estimable_basis_rank: usize,
    pub requested_rank: usize,
    pub suggested_actions: Vec<SuggestedAction>,
}
```

User-facing tests should be contrast-first:

```r
test(fit, "season")
test(fit, "duration:season")
test(fit, contrast = c("season[post] - season[pre]"))
```

Example diagnostic:

```text
The coefficient seasonpost:sites3 is not separately estimable.
Reason: no observations exist for site s3 in post season under duration 7d.
Use marginal contrasts over observed cells, or remove the unsupported
interaction.
```

Product policy:

- coefficient tables should not pretend aliased coefficients are ordinary
  estimates
- term-level tests must say whether they are fully estimable, partially
  estimable, or impossible
- `summary()`, `anova()`, and `test()` should all query the same estimability
  compiler
- empty cells and aliasing should be reported with design-level reasons
- contrast coding should be stored in the fit object and shown when relevant

This turns "why is my coefficient missing?" into a concrete design diagnosis.

## Residual Model Layer

Random effects and residual structure should be separate model components:

```text
y = X beta + Z b + e
b ~ N(0, G(theta))
e ~ N(0, R(psi))
```

Do not overload random effects to mimic residual heteroskedasticity or
autocorrelation. Add a residual model layer with explicit syntax:

```r
fit <- lmm(
  y ~ treatment * time + (time | subject),
  data = dat,
  residual = ar1(time, subject)
)

fit <- lmm(
  y ~ treatment + (1 | site),
  data = dat,
  residual = var_by(group)
)
```

Candidate residual structures:

- iid residual variance
- group-specific residual variance
- power/exponential variance functions
- AR(1) within subject/site
- continuous-time AR(1)
- spatial exponential covariance when coordinates are supplied
- Matern covariance when coordinates are supplied
- known covariance matrix

Proposed representation:

```rust
pub enum ResidualStructure {
    Iid,
    VarBy { factor: String },
    VarPower { covariate: String },
    VarExp { covariate: String },
    Ar1 { time: String, group: String },
    ContinuousAr1 { time: String, group: String },
    SpatialExp { x: String, y: String, group: Option<String> },
    SpatialMatern { x: String, y: String, group: Option<String> },
    KnownCovariance { name: String },
}

pub struct ResidualModel {
    pub requested: ResidualStructure,
    pub effective: ResidualStructure,
    pub parameters: Vec<ResidualParameter>,
    pub diagnostics: Vec<Diagnostic>,
}
```

The compiler should detect confounding between random effects and residual
correlation structures:

```text
You requested both (1 | subject) and compound-symmetric residual correlation
within subject. These describe similar dependence. Keeping the random intercept
and dropping residual CS.
```

Design requirements:

- keep `G(theta)` and `R(psi)` separate in model representation
- validate residual grouping/time variables before fitting
- decide which residual structures preserve sparse computation and which need
  special solvers
- report when residual structure changes the interpretation of random effects,
  prediction uncertainty, or finite-sample inference
- make residual diagnostics part of `audit(fit)`

## Explicit Prediction Targets

Prediction must not be a single opaque function with hidden conditioning rules.
The user should choose the estimand:

```r
predict(fit, target = "conditional")       # existing groups, use fitted REs
predict(fit, target = "population")        # random effects set to zero
predict(fit, target = "new_group_mean")    # new group, no residual noise
predict(fit, target = "new_observation")   # new group + residual variation
predict(fit, target = "partial", condition_on = "site")
```

Proposed target enum:

```rust
pub enum PredictionTarget {
    Conditional,
    Population,
    NewGroupMean,
    NewObservation,
    Partial { condition_on: Vec<String> },
}
```

Every prediction result should include an uncertainty decomposition:

```rust
pub struct PredictionUncertainty {
    pub se_fixed: Option<f64>,
    pub sd_random_effect: Option<f64>,
    pub sd_residual: Option<f64>,
    pub se_mean: Option<f64>,
    pub sd_total_prediction: Option<f64>,
}
```

Example user-facing output:

```text
Prediction interval for new observation:
mean = 12.4
SE_fixed = 0.8
SD_new_site = 1.7
SD_residual = 2.3
95% PI = [7.0, 17.8]
```

Prediction design requirements:

- distinguish fitted groups from new groups
- define behavior for unknown grouping levels explicitly
- allow conditioning on all, none, or a named subset of random effects
- distinguish mean intervals from observation/prediction intervals
- include residual model uncertainty where relevant
- preserve user-scale transformations in prediction output
- return structured errors when requested prediction target is impossible

This should replace ambiguous `re.form`-style behavior with explicit
statistical targets.

## Diagnostics as a First-Class Model Product

Diagnostics should be structured, reproducible, attached to the model object,
and available without relying on optional plots or external expert judgment.

The existing `audit(fit)` concept should expand to include:

```text
Design:
  fixed-effect rank: OK
  empty cells: none
  random-effect support: weak for slope by site

Fit:
  optimizer certificate: KKT OK
  boundary parameters: site slope variance = 0
  Hessian on active subspace: positive definite

Distribution:
  residual normality: mild tail issue
  residual variance: differs by season
  random-effect normality: insufficient groups to assess

Influence:
  high-impact groups: site_12, subject_05
  high-leverage observations: 3 flagged

Inference:
  fixed-effect p-values: Satterthwaite, grade B
  recommend bootstrap for treatment:time
```

Additional audit components:

```rust
pub struct DistributionAudit {
    pub residual_normality: DiagnosticResult,
    pub residual_variance: DiagnosticResult,
    pub residual_autocorrelation: DiagnosticResult,
    pub random_effect_normality: DiagnosticResult,
    pub posterior_predictive_checks: Vec<DiagnosticResult>,
}

pub struct InfluenceAudit {
    pub high_impact_groups: Vec<InfluenceFlag>,
    pub high_leverage_observations: Vec<InfluenceFlag>,
    pub deletion_diagnostics_available: bool,
}
```

Diagnostic families to support over time:

- residual plots and numerical residual checks
- residual heteroskedasticity checks
- residual autocorrelation checks
- random-effect normality checks, with "insufficient groups" as a valid result
- influence diagnostics by observation and grouping unit
- profile likelihood interval diagnostics
- posterior predictive or parametric simulation checks
- leverage-like measures for fixed-effect design
- checks that the fitted model explains the experimental design implied by the
  formula

The audit report should be the single place that consolidates design,
optimization, covariance, residual, influence, and inference health.

## Semantic Formula IR and Covariance Stories

Part 3 sharpens the main architectural point:

> lme4 formula syntax is a surface language, not the model.

Internally, a formula should compile to a semantic model object that understands
grouping, nesting, crossing, random-coefficient bases, covariance structures,
estimability, missing dependence paths, and the scientific role of each factor.

The formula parser remains important for compatibility, but it should not be
the authority on model meaning. The authority is the compiled semantic IR plus
the design graph and covariance-kernel graph.

## Random-Effects Semantic IR

The core internal representation should make random effects explicit:

```rust
pub struct RandomTermIr {
    pub group: GroupingFactor,
    pub basis: RandomCoefficientBasis,
    pub covariance: CovarianceForm,
    pub intercept: InterceptPolicy,
    pub role: GroupingRole,
    pub source_syntax: SourceSyntax,
    pub interpretation: CovarianceStory,
}

pub struct VarianceComponentIr {
    pub grouping_relation: GroupingRelation,
    pub basis: RandomCoefficientBasis,
    pub source_syntax: SourceSyntax,
    pub interpretation: CovarianceStory,
}

pub enum CovarianceForm {
    Full,
    Diagonal,
    LowRank { rank: Option<usize> },
    Scalar,
    Structured { kind: StructuredCovarianceKind },
}

pub enum InterceptPolicy {
    Included,
    Omitted,
}

pub enum GroupingRole {
    SampledUnit,
    Item,
    Site,
    Batch,
    Block,
    Treatment,
    RepeatedUnit,
    Unknown,
}
```

For canonical examples of `(x | g)`, `(x || g)`, `(0 + x | g)` and the
print-layer translation of these forms, see `random_effects_formulas.md`
§3 (R3 zerocorr, R4 intercept policy) and §6 (requested → canonical →
effective reporting). The struct definitions above remain authoritative for
the IR shape; the formula-mechanics examples are single-sourced in the
formulas doc.

## Intercept Omission as a First-Class Concept

> Mechanics in `random_effects_formulas.md` §3 R4 (intercept policy) and §3
> R8 (fixed/random redundancy). The discussion below is design rationale.

The syntax `(0 + x | subject)` overloads zero to mean "remove the intercept from
the random-coefficient basis." The native API should expose this directly:

```r
re(subject, slope = x, intercept = FALSE)

re(subject, basis = x, cov = "scalar")
```

A slope-only term says:

> Subjects may differ in the effect of x, but subjects do not differ in baseline
> level.

That assumption is coherent in some designs:

- the response has already been centered or differenced within subject
- subject fixed effects are already included
- the scientific model concerns only subject-specific deviations in slope
- the random effect is a cell-means basis where the intercept is represented
  elsewhere

It is suspicious in ordinary repeated-measures settings. The compiler should
produce a design-level diagnostic:

```text
You specified a subject-level random slope without a subject-level random
intercept. This assumes subjects have no baseline heterogeneity after fixed
effects.

Because subject has repeated observations and no subject fixed effect is
present, the more standard model is:
  y ~ x + re(subject, 1 + x, cov = "full")

or, if slope-intercept correlation is unsupported:
  y ~ x + re(subject, 1 + x, cov = "diagonal")
```

The reverse case should be rejected or reduced when it is algebraically
redundant:

```r
y ~ subject + x + (1 | subject)
```

Diagnostic:

```text
The subject random intercept is redundant with subject fixed effects.
The fixed-effect design already contains one intercept per subject.
The subject random-intercept variance is not separately identifiable.

Suggested model:
  y ~ subject + x + re(subject, slope = x, intercept = FALSE)
```

This follows from column-space overlap between fixed-effect subject indicators
and the random-intercept design. It is not merely a warning heuristic.

## Covariance Choice Is Not Formula Magic

> Mechanics for `||` numeric centering and basis-stable summaries in
> `random_effects_formulas.md` §4.5. The discussion below is design rationale.

The semantic split is:

- basis: which coefficients vary by group?
- covariance: how are those coefficients allowed to covary?

For subject-specific intercepts and slopes:

```text
b_s = [b0_s, b1_s]'
b_s ~ N(0, G)

G =
  [ sigma0^2    sigma01  ]
  [ sigma01     sigma1^2 ]
```

The user should see:

```text
subject:
  varying coefficients: intercept, x
  covariance: full
  parameters: sd(intercept), sd(x), corr(intercept, x)
```

For diagonal covariance:

```text
G =
  [ sigma0^2    0        ]
  [ 0           sigma1^2 ]
```

The user should see:

```text
subject:
  varying coefficients: intercept, x
  covariance: diagonal
  assumption: subject baseline and subject x effect are independent
```

For continuous predictors, the compiler must understand the invariance issue.
A full intercept/slope covariance model is invariant to additive shifts of `x`.
A diagonal intercept/slope model is not. The independence assumption depends on
where `x = 0` is placed.

Possible diagnostic:

```text
You requested an uncorrelated random intercept/slope model for x by subject.
x is continuous and its zero point appears arbitrary.
The independence assumption depends on where x = 0 is placed.
I centered x internally at its design-relevant reference value.
Reported coefficients are back-transformed.
```

Or, when centering cannot make the assumption defensible:

```text
The diagonal covariance model is not recommended unless x has a meaningful zero
point or the independence assumption is intended at the chosen reference value.
```

## Canonical Nesting, Crossing, and Interaction Semantics

> **Superseded by `random_effects_formulas.md`.** The deterministic rules for
> canonical expansion of `(b | a*c)`, `(b | a/c)`, `(b | a:c)`, and the
> grouping-factor materialization for the resulting interaction levels live
> in §3 R1, R2 and §5 of the formulas doc. The covariance consequence of
> each form, the `FormulaCanonicalized` Info diagnostic emitted on every
> expansion, and the `CrossingLikelyUnintended` recommendation that fires
> when `(b | a*c)` is likely a confusion for `(b|a)+(b|c)` or `(b|a:c)`,
> are also specified there.

The remainder of this contract assumes the formulas doc holds: every
random-effects grouping expression has a deterministic canonical form,
recorded as `requested_formula → canonical_formula` in the compiled artifact.

## Covariance Kernels Over Rows

Every random-effect term induces a covariance pattern among observations. This
is the unifying abstraction for diagnostics.

For a random intercept:

```r
(1 | subject)
```

the induced kernel is:

```text
K_subject[i, j] = 1[subject_i = subject_j]
```

For crossed subject and item intercepts:

```r
(1 | subject) + (1 | item)
```

the marginal covariance is:

```text
V = sigma_subject^2 K_subject
  + sigma_item^2 K_item
  + sigma_residual^2 I
```

For cell effects:

```r
(1 | subject:item)
```

the kernel is:

```text
K_subject_item[i, j] =
  1[subject_i = subject_j and item_i = item_j]
```

For random slopes:

```r
(x | subject)
```

the covariance contribution is:

```text
Cov(y_i, y_j) = z_i' G z_j
when subject_i = subject_j
where z_i = [1, x_i]'
```

Once terms are represented as covariance kernels, diagnostics become formal:

- does this term add a new covariance path?
- is this covariance path already represented by another term?
- is the kernel rank deficient?
- is the covariance parameter estimable?
- are repeated observations left independent?
- is a grouping factor treated as random even though it is a treatment of
  direct interest?

This kernel graph should be the backbone of the model-audit system.

## Detect Under-Modeling, Not Only Over-Modeling

Current mixed-model tools focus heavily on "too complex." The compiler should
also detect models that are too minimal, because ignored dependence can make
fixed-effect inference anti-conservative.

Example:

```r
y ~ condition
```

with repeated observations per subject should trigger:

```text
subject appears in multiple rows, but no subject-level term is present.
The current model treats repeated observations from the same subject as
independent.

Suggested minimum model:
  y ~ condition + (1 | subject)
```

If `condition` varies within subject:

```text
condition varies within subject.
The model with only (1 | subject) assumes the condition effect is identical
across subjects.

For confirmatory inference on condition, consider:
  y ~ condition + (condition | subject)

If the slope variance/correlation is not estimable, I will reduce it to:
  y ~ condition + (1 | subject) + (0 + condition | subject)
or report that the slope cannot be supported by the design.
```

Principle:

- random intercepts protect dependence in baselines
- random slopes protect dependence in effects

The compiler should ask:

- which fixed effects vary within which grouping units?
- which grouping units are repeatedly observed?
- which effect heterogeneities matter for the target inference?
- which missing random slopes would make inference anti-conservative?

## Model Lattice and Information Budget

"Keep it maximal" is a user-level proxy for a better compiler rule:

> Include the random-effect structure needed to make fixed-effect inference
> valid, but only to the extent that the design and data can support it.

For each fixed effect being tested, the compiler should ask:

- which grouping units could vary in this effect?
- is the effect within-unit, between-unit, or partially within-unit?
- is a random slope required for defensible inference?
- is that slope estimable?
- is its covariance with other random coefficients estimable?

The compiler should show an information budget before fitting. A full
unstructured covariance matrix for `d` random coefficients requires:

```text
d * (d + 1) / 2
```

covariance parameters. This grows quickly:

```text
8 random coefficients -> 36 covariance parameters per grouping factor
8 random coefficients for subject and item -> 72 covariance parameters
```

Example audit:

```text
Random-effect information budget
subject:
  levels: 56
  requested random coefficients: 8
  requested covariance parameters: 36
  status: too rich; full covariance unlikely to be estimable

item:
  levels: 32
  requested random coefficients: 8
  requested covariance parameters: 36
  status: too rich; full covariance unlikely to be estimable

Recommended starting covariance:
  random coefficients: keep scientifically relevant slopes
  covariance structure: diagonal or reduced-rank
  correlation parameters: estimate only after variance directions are supported
```

Suggestions should come from a formal candidate lattice rather than ad hoc
rules:

```text
M0: fixed effects only
M1: add required random intercepts
M2: add scientifically relevant random slopes
M3: add diagonal covariance among random coefficients
M4: add full covariance
M5: add interaction/cell variance components
M6: reduced-rank covariance model
```

Each candidate should be classified:

- coherent
- scientifically incomplete
- estimable
- weakly estimable
- non-identifiable
- overparameterized
- singular/reduced-rank
- fixed-effect inference stable or fragile

Selection policy:

```text
policy = "confirmatory":
  prefer required random slopes
  avoid unsupported correlations
  use unpenalized REML/ML
  bootstrap fragile tests

policy = "exploratory":
  allow regularized covariance shrinkage
  report selected structure
  mark p-values as exploratory/post-selection
```

The target is not the largest or smallest model. It is the maximal
scientifically justified structure subject to design estimability, with
transparent covariance reduction.

## rePCA and Reduced-Rank Covariance as Core Output

Convergence is weaker than interpretability. A model can converge and still
have fitted covariance matrices that are lower-rank than requested. Therefore,
random-effect PCA should be a core engine diagnostic, not an expert-only
afterthought.

Example:

```text
Requested subject random-effect basis:
  intercept, A, B, A:B

Supported random-effect directions:
  PC1: mostly intercept
  PC2: A contrast
  PC3: weak B/A:B mixture
  PC4: unsupported

subject covariance:
  requested rank: 4
  supported rank: 2
  recommended refit:
    reduced_rank(subject, basis = intercept + A + B + A:B, rank = 2)
```

This is preferable to blindly dropping named terms one by one, because named
terms depend on coding choices. The supported directions are properties of the
fitted covariance geometry.

Post-fit semantic certificate:

```text
optimization: converged
KKT conditions: satisfied
random-effect rank: reduced
requested covariance dimension: 8
supported covariance dimension: 4
correlation parameters: not fully interpretable
fixed-effect inference: stable across reduced structures
statistical coherence: fragile
```

## Correlations Are Optional Luxuries

Correlation parameters among random coefficients are expensive, fragile, and
often basis-dependent. They should be gated behind support for the associated
variance directions.

Estimate `corr(intercept, slope)` only if:

- intercept variance is supported
- slope variance is supported
- predictor basis is declared and interpretable
- enough grouping levels exist
- likelihood/profile behavior is stable
- KKT and information checks pass on the active subspace

Otherwise report:

```text
The data support subject-specific intercepts and subject-specific A effects,
but not their correlation.

Using diagonal covariance:
  re(subject, intercept = TRUE, slopes = A, cov = "diagonal")
```

For continuous predictors:

```text
The uncorrelated intercept/slope assumption depends on the zero point of time.
I centered time at baseline before fitting and back-transformed reported
estimates.
```

## "Unsupported" Is Better Than "Zero"

Removing or zeroing a variance component does not prove the true variance is
zero. It means this data set does not support estimating it reliably.

Bad message:

```text
Dropped random slope for A.
```

Better message:

```text
The random slope for A by subject was not supported by this data set.
This does not prove subject-to-subject variation in A is zero.
It means this study does not contain enough information to estimate it
reliably.
```

Inference consequence:

```text
Fixed-effect p-values are conditional on the selected random-effect structure.
Because a theoretically relevant random slope was unsupported, I recommend a
bootstrap sensitivity check for the A effect.
```

This language should be part of the diagnostic style guide.

## Basis Manager and Coding Dependence

Random-effect covariance parameters depend on factor coding and basis choice.
The engine should make basis choices explicit instead of inheriting hidden R
contrast settings.

Example output:

```text
Fixed-effect basis:
  season: treatment coding, reference = pre

Random-effect basis:
  season slopes by subject: orthonormal sum-to-zero basis

Covariance interpretation:
  correlations are among basis coefficients, not raw season labels.
```

For user-facing inference, prefer basis-stable quantities:

- variance of subject-specific season effects
- rank of subject season-effect covariance
- predicted subject-to-subject heterogeneity in post vs pre
- covariance-kernel contribution to marginal dependence

Avoid encouraging users to overinterpret basis-dependent parameters such as:

```text
corr(seasonpost, seasonmonsoon) = -0.84
```

when that value can change materially under recoding.

## Roles and Generalization Targets

Users often use "random factor" to mean several different things:

- sampled unit to generalize over
- nuisance blocking factor
- treatment condition
- repeated-measures unit
- batch or process artifact
- spatial or temporal cluster

The API should allow declared roles:

```r
lmm(
  y ~ condition,
  data = dat,
  roles = list(
    subject = sampled_unit(),
    item = sampled_unit(),
    condition = treatment(),
    lab = batch()
  )
)
```

Then the compiler can say:

```text
condition is a treatment with 3 designed levels.
It should usually be fixed, not random.

subject is a sampled unit with repeated observations.
A subject random intercept is required.

condition varies within subject.
A subject random slope for condition is recommended for confirmatory inference.
```

Roles connect random-effect syntax to the actual generalization target.

## Fixed-Effect Sensitivity to Random-Effect Structure

The audit should report whether fixed-effect conclusions are stable across the
supported part of the model lattice.

Example:

```text
Fixed-effect sensitivity audit
effect              estimate range    SE range       p range      status
condition           stable            stable         stable       OK
condition:time      stable            unstable       0.03-0.09    fragile
```

Interpretation:

```text
The estimated treatment effect is robust, but its p-value depends on whether
the subject-specific treatment slope is included.
Use bootstrap or report this as fragile.
```

This is more informative than a single polished p-value.

## Native Random-Effects API

Accept lme4 syntax for compatibility:

```r
lmm(y ~ x + (x | subject), data = dat)
```

Expose a clearer native syntax:

```r
lmm(
  y ~ x,
  data = dat,
  random = re(subject, intercept = TRUE, slopes = x, cov = "full")
)
```

Crossed factors:

```r
random = vc(subject) + vc(item)
```

Nested factors:

```r
random = vc(school) + vc(class %in% school)
```

Cell effects:

```r
random = vc(subject:item)
```

Main-plus-cell crossed structure:

```r
random = vc(subject) + vc(item) + vc(subject:item)
```

The formula layer remains familiar. The semantic layer is explicit.

## `explain_model()` Before Fit

The engine should expose explanation before fitting:

```r
explain_model(
  y ~ condition + (1 | subject:item),
  data = dat
)
```

Possible output:

```text
Fixed effects:
  condition: population-level contrast

Random effects:
  subject:item:
    random intercept for each subject-item cell

Detected design:
  subject appears in multiple rows
  item appears in multiple rows
  subject and item appear crossed

Potential under-modeling:
  no subject main random effect
  no item main random effect

The specified model only correlates observations within the same subject-item
cell. It does not model subject-wide or item-wide dependence.

Suggested model:
  y ~ condition + (1 | subject) + (1 | item)

Optional, if repeated observations exist within subject-item cells:
  y ~ condition + (1 | subject) + (1 | item) + (1 | subject:item)
```

`fit()` should call this machinery internally and attach the resulting
explanation/audit to the fitted object.

## Final Semantic Architecture

The architecture becomes:

```text
formula parser / native model spec
  -> semantic random-effects IR
  -> design graph and covariance-kernel graph
  -> estimability, role, basis, and invariance checks
  -> information budget
  -> candidate model lattice
  -> maximal feasible coherent model
  -> constrained REML/ML engine
  -> KKT-certified optimization
  -> finite-sample inference
  -> model audit and explanation
```

When a model is bad, the package should not say only:

```text
Model failed to converge.
Try another optimizer.
```

It should classify the failure:

```text
This model is incoherent:
  subject fixed effects make the subject random intercept unidentified.

This model is under-specified:
  repeated subject observations are treated as independent.

This model is over-specified:
  the subject intercept/slope correlation is not estimable from this design.

This model is ambiguous:
  (1 | a*b) expands to vc(a) + vc(b) + vc(a:b).
  If you intended cells only, use vc(a:b).

This model is coherent but fragile:
  random slope variance is weakly identified; confirm with bootstrap.
```

The deeper product lesson is:

> Mixed-model pain is not mostly a numerical problem. It is a
> model-specification language problem plus an information-allocation problem.

The system should make hidden assumptions visible, prevent both underfitting
and overfitting, and only print inferential quantities whose interpretation it
can defend.

## Validation Is the Product

Claims about sensible p-values require a public simulation and parity suite.

Design grid:

- balanced nested designs with known exact df
- split-plot designs
- crossed subject/item designs
- unbalanced designs
- few-group designs
- many-group designs
- true zero variance components
- near-zero variance components
- random-slope correlation equal to zero
- random-slope correlation near +/- 1
- rank-deficient `X`
- rank-deficient `Z`
- badly scaled predictors
- confounded fixed/random structures
- unsupported random slopes
- supported slopes but unsupported correlations
- empty cells and partially estimable fixed-effect terms
- residual heteroskedasticity
- residual autocorrelation
- conditional, population, and new-group prediction targets
- high-impact observations and high-impact groups
- slope-only random effects with and without fixed effects for the same unit
- crossed, nested, cell-only, and main-plus-cell random structures
- repeated-unit designs missing random intercepts
- within-unit fixed effects missing random slopes
- high-dimensional random-coefficient bases with limited grouping levels
- basis/coding changes that should preserve basis-stable conclusions
- treatment factors incorrectly specified as random sampled units
- reduced-rank random-effect covariance with known supported directions

Comparators:

- lme4 point estimates
- MixedModels.jl point estimates
- lmerTest Satterthwaite
- pbkrtest Kenward-Roger
- SAS PROC MIXED where possible
- parametric bootstrap ground truth

Acceptance criteria:

- under H0, p-values are approximately uniform in simulation
- nominal 95% intervals cover approximately 95%
- known boundary cases do not crash
- true non-identifiable models are rejected before inference
- optimizer certificate agrees with numerical profiling
- effective covariance reductions are reproducible and recorded
- badly scaled predictors fit equivalently after back-transformation
- estimability diagnostics match known design aliasing
- prediction uncertainty decomposition matches simulation targets
- canonicalization of nesting/crossing formula syntax is stable and explained
- under-modeling diagnostics identify missing dependence paths
- information-budget warnings trigger before optimization
- rePCA/reduced-rank summaries recover known supported covariance dimensions
- fixed-effect sensitivity audits flag deliberately fragile cases
- basis-stable quantities remain stable under contrast recoding

## Suggested Rust Modules

The current crate has formula, model, linalg, stats, and types modules. The
following additions or internal splits would make the compiler contract explicit:

```text
src/ir/
  random_term.rs
  variance_component.rs
  semantic_model.rs
  source_syntax.rs

src/design/
  audit.rs
  compiler.rs
  graph.rs
  estimability.rs
  rank.rs
  nesting.rs
  confounding.rs
  roles.rs

src/basis/
  manager.rs
  fixed.rs
  random.rs
  invariance.rs

src/kernel/
  graph.rs
  random_effect.rs
  residual.rs
  overlap.rs

src/scale/
  plan.rs
  transform.rs
  backtransform.rs

src/covariance/
  structure.rs
  strategy.rs
  spectral.rs
  reduction.rs
  repca.rs
  information_budget.rs

src/lattice/
  candidate.rs
  classifier.rs
  policy.rs
  sensitivity.rs

src/residual/
  structure.rs
  compiler.rs
  covariance.rs
  diagnostics.rs

src/optimize/
  objective.rs
  constrained.rs
  certificate.rs
  active_set.rs

src/inference/
  contrast.rs
  satterthwaite.rs
  kenward_roger.rs
  bootstrap.rs
  reliability.rs

src/compare/
  target.rs
  criterion.rs
  refit.rs

src/predict/
  target.rs
  uncertainty.rs
  new_levels.rs

src/explain/
  covariance_story.rs
  model_explanation.rs
  suggestions.rs

src/diagnostics/
  codes.rs
  diagnostic.rs
  audit.rs
  distribution.rs
  influence.rs
```

This does not require an immediate large refactor. It is a target shape that
can be reached incrementally by extracting responsibilities from the current
model/statistics code.

## Implementation Phases

### Phase 0: Write Down Contracts

- define `FitStatus`
- define `Diagnostic`
- define `DesignAudit`
- define `OptimizerCertificate`
- define `InferenceStatus`
- define `ComparisonTarget`
- define `PredictionTarget`
- define `ResidualStructure`
- define `RandomTermIr`
- define `VarianceComponentIr`
- define `GroupingRole`
- define `CovarianceStory`
- define `InformationBudget`
- define `CandidateModel`
- define requested-vs-effective model representation

The goal is to make future behavior explicit before implementing every
algorithm.

### Phase 1: Semantic IR and Formula Canonicalization

- compile lme4-style random-effects syntax into `RandomTermIr` and
  `VarianceComponentIr`
- canonicalize `(1 | a/b)`, `(1 | a*b)`, `(1 | a:b)`, crossed terms, and
  slope-only terms
- preserve source syntax so diagnostics can say what the user wrote and what it
  means
- represent basis and covariance as separate concepts
- expose random-intercept omission as `InterceptPolicy::Omitted`
- generate initial covariance stories from the semantic IR

### Phase 2: Design Graph, Roles, and Covariance Kernels

- build a design graph for repeated, crossed, nested, and cell-level factors
- allow declared roles such as sampled unit, treatment, batch, block, item, and
  site
- infer repeated units and obvious treatment-like small factors cautiously
- compile random-effect terms into covariance kernels over rows
- detect kernel overlap, redundancy, and missing dependence paths
- distinguish under-modeling from over-modeling in diagnostics

### Phase 3: Preflight Design and Estimability Audit

- compute fixed-effect rank
- detect aliased fixed-effect columns and empty cells
- classify fixed-effect terms and requested contrasts as estimable, partially
  estimable, or not estimable
- compute grouping levels and observations per level
- compute random-term coefficient counts and covariance parameter counts
- detect random slopes with no within-group variation
- flag too-few-level grouping factors
- classify terms as estimable, weakly estimable, or unsupported
- return an audit without fitting

This is the first major user-facing win because many failures can be explained
before optimization.

### Phase 4: Basis Manager and Scaling Plan

- make fixed-effect and random-effect bases explicit
- report coding choices and covariance interpretation
- prefer basis-stable summaries for user-facing random-effect interpretation
- detect invariance problems in diagonal intercept/slope covariance models
- add internal centering/scaling for continuous columns
- preserve user-scale output
- test back-transformed fixed effects, SEs, predictions, and contrasts
- detect near-collinearity and report it through diagnostics

### Phase 5: Information Budget and Model Lattice

- compute grouping levels, random coefficient counts, and covariance parameter
  counts before fitting
- classify covariance structures as supported, weak, too rich, or impossible
- construct candidate models from minimal valid through maximal feasible
- score candidates for scientific sufficiency, estimability, fragility, and
  fixed-effect inference stability
- implement confirmatory and exploratory lattice policies

### Phase 6: Constrained Optimizer Certificate

- introduce active-set or box-constrained covariance parameter handling
- compute projected gradients
- separate optimizer exit status from statistical convergence status
- classify interior, boundary, reduced-rank, and failed fits

### Phase 7: Covariance Strategy Compiler and rePCA

- implement `as_specified`
- implement `maximal_feasible`
- record requested/effective covariance structures
- support diagonalization or zeroing of unsupported components
- add reduced-rank representation where practical
- compute random-effect PCA for fitted covariance blocks
- report requested versus supported covariance rank
- treat unsupported variance directions as information limits, not proof of
  true zero variance

### Phase 8: Contrast Inference

- represent fixed-effect hypotheses as contrast matrices
- compute Satterthwaite df for scalar and multi-df tests
- add reliability grading
- withhold p-values when required inputs are invalid
- condition inference labels on the selected/effective random-effect structure

### Phase 9: Intent-Aware Model Comparison

- represent comparison targets explicitly
- detect when fixed-effect comparison requires ML refits
- keep original summaries separate from comparison refits
- refuse invalid likelihood comparisons with structured diagnostics
- add boundary-aware or bootstrap paths for random-effect comparisons

### Phase 10: Kenward-Roger and Derivatives

- expose derivative API for objective and `vcov(beta)`
- implement KR adjustments where feasible
- share factorization caches with derivative calculations

### Phase 11: Adaptive Bootstrap

- implement simulation/refit loop
- track convergence outcomes of simulated fits
- add adaptive stopping and MC SE reporting
- trigger bootstrap from inference reliability rules

### Phase 12: Residual Model Layer

- add iid and group-specific residual variance structures
- add AR(1) residual structure for grouped repeated measures
- validate residual time/group variables in the compiler
- detect confounding between residual correlation and random effects
- expose residual diagnostics through `audit(fit)`

### Phase 13: Explicit Prediction Targets

- implement conditional, population, new-group mean, new-observation, and
  partial prediction targets
- define behavior for unknown grouping levels
- return fixed, random-effect, residual, and total uncertainty components
- preserve user-scale transformations in prediction output

### Phase 14: Explain Model and Audit Expansion

- expose `explain_model()` without fitting
- print covariance stories for fixed/random/residual assumptions
- explain canonical formula expansion and covariance consequences
- report under-modeling, over-modeling, basis dependence, information budget,
  rePCA rank, and fixed-effect sensitivity
- consolidate design, optimizer, covariance, residual, distribution, influence,
  and inference diagnostics
- add high-impact group and observation flags
- add residual variance/autocorrelation checks
- make "insufficient information to assess" a first-class diagnostic result

### Phase 15: R Binding

- expose Rust fit, audit, and inference objects through a stable FFI layer
- keep R syntax lme4-like
- expose the clearer native R random-effect API with `re()` and `vc()`
- map R formulas and contrasts into Rust model specs
- print structured diagnostics and effective formulas

## Execution Roadmap

The phase list above is the conceptual dependency graph. The execution roadmap
below is the pragmatic build order. The central rule is:

> Build the compiler and audit contract before building advanced inference.

The first usable product should be a Rust crate that can explain a model before
fitting it, fit ordinary LMMs at parity with the current engine, and attach a
structured audit. Satterthwaite, Kenward-Roger, residual covariance models, and
the R layer should build on that contract rather than define it.

### Roadmap Principles

- Keep the Rust crate self-contained at every milestone.
- Preserve current fitting behavior while adding compiler layers around it.
- Make every new diagnostic structured before writing polished prose output.
- Prefer prefit refusal/explanation over postfit warning cleanup.
- Add one public capability per milestone, with tests and examples.
- Do not implement the R layer until the Rust fit/audit/inference objects are
  stable enough to serialize.
- Treat finite-sample p-values as late-stage output that depends on the design,
  covariance, optimizer, derivative, and audit contracts.

### Milestone 0: Baseline and Safety Rails

Goal: establish the current crate as a stable baseline before refactoring.

Deliverables:

- inventory current formula, model, optimizer, and summary pathways
- add smoke tests for existing LMM fits
- add golden/parity tests for simple models against lme4 or MixedModels.jl
- record current limitations as expected failures
- define a small set of canonical demo data sets for future compiler tests

Exit criteria:

- current tests pass
- simple random-intercept and random-slope models have locked expected output
- future semantic/audit layers can be tested without changing numerical fits

Why first: the project needs a regression harness before the design compiler
starts changing how models are represented.

### Milestone 1: Contracts and Structured Diagnostics

Goal: create the vocabulary that all later components will use.

Deliverables:

- `FitStatus`
- `Diagnostic`, `DiagnosticCode`, `DiagnosticSeverity`, `DiagnosticStage`
- `FitAudit`
- `DesignAudit`
- `OptimizerCertificate`
- `InferenceStatus`
- requested-versus-effective model containers
- serde-friendly result structures where practical

Public capability:

```rust
let diagnostics: Vec<Diagnostic> = fit.diagnostics();
let audit: FitAudit = fit.audit();
```

Exit criteria:

- existing fit objects can carry empty or basic diagnostics
- diagnostics can be formatted for humans and inspected programmatically
- no statistical behavior changes are required yet

Why now: without shared diagnostic types, every later module invents its own
warning language.

### Milestone 2: Semantic Formula IR

Goal: stop treating formula syntax as the internal model.

Deliverables:

- `RandomTermIr`
- `VarianceComponentIr`
- `SemanticModel`
- `GroupingRole`
- `CovarianceForm`
- `InterceptPolicy`
- `SourceSyntax`
- canonicalization for `(x | g)`, `(x || g)`, `(0 + x | g)`, `(1 | a:b)`,
  `(1 | a/b)`, `(1 | a*b)`, and crossed random intercepts

Public capability:

```rust
let spec = compile_formula("y ~ x + (x | subject)", &schema)?;
println!("{}", spec.explain());
```

Exit criteria:

- formula-to-IR tests cover all lme4 random-effect syntax currently supported
- source syntax and canonical semantic meaning are both preserved
- random-effect basis and covariance form are separate in the IR
- existing model fitting can still be driven from the old path while IR matures

Why now: semantic IR is the foundation for every design, covariance, and audit
decision.

### Milestone 3: `explain_model()` Without Fitting

Goal: deliver the first visible value from the compiler.

Deliverables:

- initial `CovarianceStory`
- human-readable explanation for fixed effects and random effects
- canonical expansion output for nesting, crossing, and interaction grouping
- explicit intercept omission explanation
- basic native API sketch for `re()` and `vc()` represented internally, even if
  R syntax does not exist yet

Public capability:

```rust
let explanation = explain_model("y ~ condition + (1 | subject:item)", &data)?;
```

Exit criteria:

- users can see what covariance assumptions a formula implies before fitting
- `(1 | a) + (1 | b)`, `(1 | a:b)`, `(1 | a/b)`, and `(1 | a*b)` produce
  distinct explanations
- slope-only terms say whether baseline dependence is modeled

Why now: a prefit explanation is low numerical risk and validates the semantic
direction early.

### Milestone 4: Design Graph and Estimability Audit

Goal: inspect the data design before optimization.

Deliverables:

- model-frame schema inspection
- fixed-effect rank checks
- empty-cell detection for factor combinations
- grouping-level counts and observations-per-level summaries
- within-group variation checks for random slopes
- crossing/nesting/cell-only classification
- fixed-effect estimability statuses:
  `Estimable`, `PartiallyEstimable`, `NotEstimable`

Public capability:

```rust
let audit = audit_design(&semantic_model, &data)?;
```

Exit criteria:

- rank-deficient fixed-effect designs are diagnosed before fitting
- non-estimable contrasts can return `NA`/missing with a reason
- repeated grouping variables and absent random intercepts are detectable
- subject fixed effects plus subject random intercept is flagged as redundant

Why now: many important failures are design failures, not optimizer failures.

### Milestone 5: Covariance Kernels and Under-Modeling Guard

Goal: reason about dependence paths directly.

Deliverables:

- compact covariance-kernel descriptors for random intercepts, random slopes,
  crossed factors, nested factors, and cell effects
- kernel overlap/redundancy checks
- missing-dependence-path diagnostics
- role-aware warnings for treatment-like random factors and repeated sampled
  units without random effects

Public capability:

```rust
let graph = CovarianceKernelGraph::from_model(&semantic_model, &data)?;
let missing = graph.missing_dependence_paths();
```

Exit criteria:

- `y ~ condition` with repeated subject rows recommends at least `(1 | subject)`
- within-subject fixed effects without subject slopes are flagged for
  confirmatory inference
- `(1 | subject:item)` with crossed repeated subject/item data reports missing
  subject-wide and item-wide dependence
- redundant kernels are detected without constructing a dense `n by n` matrix

Why now: this is the main guard against anti-conservative under-modeling.

### Milestone 6: Information Budget and Maximal-Feasible Lattice

Goal: replace ad hoc random-effect simplification with explicit candidate
classification.

Deliverables:

- `InformationBudget` per grouping factor
- covariance parameter counts for full, diagonal, scalar, and reduced-rank
  forms
- candidate model lattice
- policy enum for `as_specified`, `maximal_feasible`, and `regularized`
- candidate classifications: coherent, scientifically incomplete, weakly
  estimable, non-identifiable, overparameterized, reduced-rank

Public capability:

```rust
let lattice = build_model_lattice(&semantic_model, &design_audit)?;
let recommendation = lattice.recommend(RandomStrategy::MaximalFeasible);
```

Exit criteria:

- high-dimensional random-coefficient bases show covariance-parameter budgets
  before fitting
- unsupported correlations can be removed while retaining supported slopes
- selected candidate records why it was chosen over larger/smaller models

Why now: the system needs a formal path from user formula to effective model
before optimizer certificates can be interpreted.

### Milestone 7: Fit Integration With Requested and Effective Models

Goal: make the existing numerical engine consume the effective semantic model.

Deliverables:

- bridge from `SemanticModel`/effective random structure into current `X`, `Z`,
  and covariance parameter construction
- fit object stores requested model, effective model, and reduction diagnostics
- existing optimizer output wrapped in `OptimizerCertificate`
- top-level `FitStatus` populated at least for ordinary success, boundary, and
  not-optimized cases

Public capability:

```rust
let fit = lmm("y ~ x + (x | subject)", &data, FitOptions::default())?;
println!("{}", fit.audit());
```

Exit criteria:

- ordinary existing fits still match baseline parity tests
- reduced effective models are visible in the fit object
- diagnostics explain model surgery before coefficient summaries are printed

Why now: this is where the compiler stops being advisory and becomes the entry
point to fitting.

### Milestone 8: KKT Certificate and Boundary Semantics

Goal: make convergence a certificate rather than an optimizer exit code.

Deliverables:

- active-set representation for covariance parameters
- projected-gradient checks
- boundary-gradient sign checks
- active-subspace Hessian/information checks where available
- status classification:
  `ConvergedInterior`, `ConvergedBoundary`, `ConvergedReducedRank`,
  `NotIdentifiable`, `NotOptimized`

Public capability:

```rust
let cert = fit.optimizer_certificate();
```

Exit criteria:

- zero variance components are reported as active boundaries when valid
- correlations attached to zero variance components are marked unestimated
- raw optimizer errors are mapped to structured diagnostics
- boundary cases have tests that do not rely on string matching warnings

Why now: p-values and model comparison should not run on uncertified fits.

### Milestone 9: Basis Manager, Scaling, and rePCA

Goal: make covariance interpretation stable enough for diagnostics and
inference.

Deliverables:

- fixed-effect and random-effect basis metadata
- internal centering/scaling plan for continuous predictors
- invariance diagnostics for diagonal continuous intercept/slope models
- back-transformation hooks for coefficients, SEs, contrasts, and predictions
- random-effect PCA summaries for fitted covariance blocks
- reduced-rank covariance audit

Public capability:

```rust
fit.basis_report();
fit.random_effect_pca();
```

Exit criteria:

- badly scaled predictors fit on a canonical scale and report on user scale
- diagonal intercept/slope models report zero-point dependence
- reduced-rank covariance blocks report supported directions rather than only
  named dropped terms

Why now: finite-sample inference depends on the basis and covariance geometry.

### Milestone 10: Contrast-First Inference MVP

Goal: provide defensible fixed-effect tests only when the prerequisites are met.

Deliverables:

- `FixedEffectHypothesis`
- contrast estimability checks reused from design audit
- asymptotic tests as a clearly labeled fallback
- Satterthwaite MVP for supported LMM cases
- inference reliability grades
- missing p-values with explicit reasons

Public capability:

```rust
fit.test("condition");
fit.test_contrast(L);
```

Exit criteria:

- every p-value has method, df where applicable, reliability, and status
- non-estimable contrasts return no p-value and include a design reason
- inference labels state whether covariance reduction or post-selection
  occurred

Why now: this is the first point at which coefficient summaries can become
statistically honest rather than merely numerical.

### Milestone 11: Comparison and Prediction Semantics

Goal: remove ambiguity from two common user workflows.

Deliverables:

- `compare(..., target = fixed_effects | random_effects | prediction)`
- explicit ML refit diagnostics for fixed-effect comparisons from REML fits
- `PredictionTarget`
- conditional, population, new-group mean, new-observation, and partial
  predictions
- uncertainty decomposition for predictions

Public capability:

```rust
compare(&fit1, &fit2, ComparisonTarget::FixedEffects);
predict(&fit, newdata, PredictionTarget::NewObservation);
```

Exit criteria:

- invalid comparisons are refused or refit with explicit diagnostics
- prediction output distinguishes fixed, random-effect, residual, and total
  uncertainty
- unknown grouping levels have explicit behavior

Why now: these APIs depend on fit/audit state but can precede full
Kenward-Roger and residual covariance support.

### Milestone 12: Bootstrap and Sensitivity Audits

Goal: validate fragile inference rather than hiding it.

Deliverables:

- parametric bootstrap refit loop
- adaptive stopping rules and Monte Carlo SE
- convergence/boundary accounting across bootstrap samples
- fixed-effect sensitivity across supported model-lattice candidates
- bootstrap recommendations wired into `audit(fit)`

Public capability:

```rust
fit.bootstrap_test("condition:time", BootstrapOptions::adaptive());
fit.sensitivity_audit();
```

Exit criteria:

- bootstrap reports `B`, p-value, MC SE, failed-refit policy, and boundary count
- fragile fixed-effect tests are flagged when conclusions depend on random
  structure
- audit can recommend bootstrap without running it by default

Why now: bootstrap is the safety valve for cases where approximations are weak.

### Milestone 13: Kenward-Roger and Derivative Engine

Goal: add high-quality finite-sample inference after the objective and
covariance contracts are stable.

Deliverables:

- derivative API for objective, `beta(theta)`, `vcov_beta(theta)`, and
  information in covariance parameters
- Satterthwaite improvements from exact/high-quality derivatives
- Kenward-Roger implementation for supported LMM classes
- derivative fallback policy and reliability diagnostics

Public capability:

```rust
fit.test("condition", InferenceMethod::KenwardRoger);
```

Exit criteria:

- derivative calculations are tested against finite differences on small models
- KR/Satterthwaite results have parity tests against lmerTest/pbkrtest where
  feasible
- unsupported cases fall back or return no p-value with a reason

Why now: derivative work is expensive and should target a stable model object.

### Milestone 14: Residual Model Layer

Goal: extend beyond iid residuals without confusing residual dependence with
random effects.

Deliverables:

- `ResidualStructure::Iid`
- group-specific residual variance
- AR(1) for grouped repeated measures
- residual model diagnostics
- confounding checks between residual correlation and random effects

Public capability:

```rust
lmm_with_residual("y ~ treatment * time + (time | subject)", residual_ar1);
```

Exit criteria:

- residual structures are represented separately from `G(theta)`
- unsupported residual structures fail before fitting
- prediction and inference report when residual structure affects uncertainty

Why late: residual covariance changes the linear algebra and inference contract;
it should not be mixed into the first compiler build.

### Milestone 15: R Layer

Goal: expose the Rust engine as an intent-aware R interface.

Deliverables:

- stable FFI/serialization boundary for model specs, fit results, audits, and
  inference tables
- lme4-compatible formula entry point
- native R syntax for `re()`, `vc()`, roles, comparison, prediction, and
  `explain_model()`
- R print methods that show requested/effective formulas and diagnostics

Public capability:

```r
explain_model(y ~ condition + (1 | subject:item), data = dat)
fit <- lmm(y ~ condition, data = dat,
           random_strategy = "maximal_feasible",
           inference = "auto")
audit(fit)
```

Exit criteria:

- R is a client of Rust-owned diagnostics and inference status
- no R-only statistical decisions are required for convergence, estimability,
  covariance reduction, or p-value defensibility
- R examples match Rust examples semantically

Why last: the R package should expose the engine contract, not compensate for
missing Rust semantics.

### Near-Term Issue Backlog

The original first development slice has mostly landed. Status:

1. Diagnostic and audit skeleton types: done.
2. Semantic IR types for random terms and variance components: random-term IR
   done; variance-component/residual extensions deferred.
3. Existing random-effect formula terms compile into IR: done for the v0
   lme4-style surface.
4. Canonicalization tests for `(x | g)`, `(x || g)`, `(0 + x | g)`,
   `(1 | a:b)`, `(1 | a/b)`, and `(1 | a*b)`: done.
5. `explain_model()` for random-effect covariance stories: done as a
   prefit structured explanation path.
6. Design audit for grouping level counts and repeated-unit detection:
   grouping counts done; missing dependence-path/under-modeling detection is
   tracked in `bd-01KQ7WZVTTHMF5VWK355NMPR6G`.
7. Fixed-effect rank and empty-cell diagnostics: done.
8. Information-budget reporting for random-effect covariance parameters: done.
9. Row-level fixed-effect inference table: Rust-side schema, fitted artifact
   field, bridge accessor, Wald fallback rows, p-value suppression reasons,
   and fixtures are done under `bd-01KQASCG9KZH36RNTPAHHH2NA9`. External R
   consumption remains a client task. Satterthwaite, Kenward-Roger, and
   bootstrap rows remain deferred until the prerequisites in
   `docs/fixed_effect_p_values_plan.md` are certified.

This backlog should produce the first visible user-facing improvement:

```text
explain_model() can say what the model means and what is suspicious before
running an optimizer.
```

### Work Not To Start Yet

Delay these until the compiler/audit contract exists:

- Kenward-Roger beyond the row-level table contract, until second-derivative
  and adjusted-covariance certificates exist
- broad residual covariance structures
- broad R bindings beyond consuming Rust-owned artifact/table payloads
- automatic regularized covariance selection
- dense influence diagnostics requiring many refits
- finite-sample p-value rows that do not yet have method prerequisites and
  reliability grades

Starting with these would create impressive outputs before the system can defend
their assumptions.

Deferred architecture is still tracked so it is not lost:
`bd-01KQ7X17P91CW715ZRY16H7CTX` covers residual structures, comparison and
prediction semantics, bootstrap, sensitivity audits, and Kenward-Roger/
Satterthwaite derivative work; `bd-01KQ7X0YPQ4TWA0P5J35SY5ZDJ` covers future
R client wire-schema expectations.

## Open Design Questions

- What is the minimum number of grouping levels below which a grouping factor is
  automatically treated as non-random versus weakly estimable?
- Should thresholds be hard-coded, configurable, or learned from simulation
  reliability grades?
- How should the engine distinguish a scientific condition factor from a
  sampled grouping factor when both are categorical? Candidate answer: require
  explicit user intent or infer cautiously from small named condition sets.
- What reduced-rank covariance parameterization is best for optimization:
  Cholesky with active zero constraints, spectral post-processing, or direct
  low-rank factors?
- How much automatic simplification should `maximal_feasible` perform by
  default before requiring explicit user approval?
- What is the exact reliability grading rubric for Satterthwaite and
  Kenward-Roger output?
- How should post-selection inference be labeled in summaries and downstream
  R methods?
- Which derivative backend should be preferred first: analytic derivatives,
  forward-mode AD, finite-difference fallback, or a hybrid?
- What should the default comparison criterion be when `target = "prediction"`:
  cross-validation, marginal likelihood, conditional AIC, or a user-selected
  policy?
- Which residual covariance structures can be supported without giving up the
  sparse engine, and which require separate dense or structured solvers?
- How should residual covariance parameters participate in Satterthwaite and
  Kenward-Roger derivative calculations?
- What should the default prediction interval method be for new groups:
  analytic decomposition, parametric bootstrap, or simulation from the fitted
  hierarchical model?
- Which influence diagnostics can be computed cheaply enough for `audit(fit)`,
  and which should be opt-in because they require many refits?
- How should roles be inferred safely when the user does not declare them,
  especially for small treatment-like factors versus sampled units?
- What exact graph algorithm should classify crossed, nested, partially nested,
  and cell-only dependence structures under missing cells?
- How should covariance kernels be stored so diagnostics can reason about them
  without forcing dense `n by n` covariance construction?
- What threshold or profile criterion should determine the supported rank in
  rePCA/reduced-rank covariance summaries?
- Which random-effect sensitivity candidates should be fit automatically, and
  which should be reported as suggested follow-up work because they are too
  expensive?
- What is the native R syntax for roles, `re()`, `vc()`, reduced-rank terms,
  and cell/nesting expressions that remains readable but not too magical?
- How should the system phrase "unsupported" variance directions in a way that
  is statistically honest but not overly conservative?
- How should basis-stable random-effect summaries be selected for categorical
  slopes with several contrast codings?

## Design North Star

The target system is:

```text
lmm =
  formula compiler
  + semantic random-effects IR
  + design graph and covariance-kernel graph
  + information budget and model lattice
  + REML/ML sparse engine
  + constrained optimizer with KKT certificate
  + covariance-structure compiler
  + finite-sample inference layer
  + explicit prediction/comparison targets
  + simulation-backed diagnostics
  + covariance-story explanation layer
```

The user experience should be boring in the best sense: either the model is fit
with explicit, defensible inference, or the system explains why that model is
not estimable and what design-level change would make the analysis coherent.
