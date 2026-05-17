# Proposal: R Layer for the Rust Mixed Model Engine

Status: proposal  
Owner: future R package layer  
Related issue: `bd-01KQ89Y655GVMPBQB7R584882Y`  
Earlier tracking issues: `bd-01KQ88CMA8636JV714B6ET5G49`,
`bd-01KQ7X0YPQ4TWA0P5J35SY5ZDJ`  
Related docs:

- `docs/compiler_contract_v0_prd.md`
- `docs/random_effects_formulas.md`
- `docs/mixed_model_compiler_inference_contract.md`
- `docs/multivariate_shared_theta.md`
- `docs/fixed_effect_p_values_plan.md`

## Purpose

The R layer should make the Rust engine feel familiar to users who know
`lme4`, while exposing the compiler contract that this project is building:
explicit model intent, prefit explanation, design audit, requested versus
effective model state, optimizer certificates, and defensible inference status.

The target is not a drop-in `lme4` clone. The target is an R interface where a
user can start with ordinary lme4-style formulas and extractors, then ask
better questions:

```r
fit <- lmm(
  Reaction ~ Days + (Days | Subject),
  data = sleepstudy,
  mode = "confirmatory",
  inference = "auto"
)

summary(fit)
audit(fit)
changes(fit)
parameterization(fit)
explain_model(fit)
```

The key product promise is:

> R captures user intent and presents results idiomatically. Rust owns model
> semantics, diagnostics, convergence evidence, covariance reductions, and
> inference availability.

The first R-layer deliverable is therefore not a fitter. It is a faithful
client of the compiler contract: model specs go to Rust, versioned artifacts
come back, and R formats those artifacts without reconstructing hidden
statistical decisions.

## Design Principles

1. **lme4 surface, stronger contract.** Accept familiar random-effect syntax:
   `(1 | g)`, `(1 + x | g)`, `(1 + x || g)`, `(0 + x | g)`,
   `(1 | a:b)`, `(1 | a/b)`, `(1 | a*b)`, and crossed terms such as
   `(1 | subject) + (1 | item)`.

2. **No hidden model surgery.** If the Rust compiler canonicalizes, recommends,
   refuses, reduces, or fits a boundary model, the R object stores that state
   and shows it through `changes()`, `audit()`, and compact print output.

3. **Mode is the public intent.** Users choose one public `mode`:
   `confirmatory`, `strict`, `exploratory`, or `predictive`. The internal
   Rust artifact can still record fit intent and random strategy separately,
   but ordinary users should not have to reason about their Cartesian product.

4. **Diagnostics are data, not warning strings.** R can format diagnostics, but
   it should not parse warning text or reconstruct reason codes. The stable
   source is the Rust artifact/report JSON and typed result tables.

5. **Explain before fit.** `explain_model()` and `audit_design()` should work
   without optimization. This is the first user-facing win over a traditional
   fit-then-debug workflow.

6. **Compatibility where honest.** Provide familiar generics and extractors,
   but do not subclass `lme4::merMod` or claim full drop-in compatibility when
   the semantics differ.

7. **Round-trip before convenience.** Every printed claim in R must trace back
   to a versioned Rust artifact field. Convenience wrappers may cache tables,
   but the cache is derived data, not the source of truth.

## Layer Boundaries

### Rust Engine Owns

- formula AST after parsing and canonicalization
- semantic random-effect IR
- model-frame schema validation after R translation
- design-matrix construction
- fixed/random design audit
- information-budget checks
- `ThetaMap` and covariance-family transitions
- optimizer objective, fit status, and optimizer certificate
- requested, semantic, supported, and fitted model state
- finite-sample inference availability and reliability status
- versioned JSON contract objects

### R Layer Owns

- R formula capture and environment evaluation
- `model.frame` construction details that are inherently R-specific
- R factor levels, ordered factors, contrasts, missing-data policy, and row names
- translation of R data columns into the Rust model-frame schema
- R object classes, generics, print methods, and help pages
- user-facing convenience wrappers: `summary()`, `anova()`, `drop1()`,
  `predict()`, `VarCorr()`, `ranef()`, `fixef()`, `coef()`, `sigma()`,
  `audit()`, `changes()`, `parameterization()`, `explain_model()`

### R Layer Must Not Own

- convergence decisions
- singularity or reduced-rank decisions
- whether a random-effect term is identifiable
- whether a covariance parameter was dropped, diagonalized, or boundary-active
- whether a p-value is defensible
- the mapping from user basis to optimizer `theta`

## Mode Mapping

The public R API exposes one mode. Rust records the lower-level intent and
random strategy in the artifact:

| Public mode | Internal behavior | Confirmatory p-values |
|---|---|---|
| `confirmatory` | deterministic design-compiled / maximal feasible; pre-response reductions or refusals only | only if Rust certifies them |
| `strict` | fit exactly as specified or refuse | only if the exact requested model works |
| `exploratory` | regularized or response-dependent structure discovery | no ordinary p-values |
| `predictive` | optimize prediction or validation target | no confirmatory p-values |

R must not quietly convert an exploratory or predictive object into an ordinary
confirmatory summary. If a user asks for a p-value and Rust says the method is
unavailable, R prints `NA` with the Rust reason.

## User-Facing API

### Primary Constructors

Use short names for the new interface rather than masking `lme4::lmer()` and
`lme4::glmer()` by default:

```r
lmm(
  formula,
  data,
  random = NULL,
  roles = NULL,
  REML = TRUE,
  weights = NULL,
  subset = NULL,
  na.action = na.omit,
  contrasts = NULL,
  mode = c("confirmatory", "strict", "exploratory", "predictive"),
  inference = c("auto", "none", "satterthwaite", "kenward_roger",
                "bootstrap", "asymptotic"),
  control = mm_control(),
  ...
)

glmm(
  formula,
  data,
  family,
  random = NULL,
  roles = NULL,
  weights = NULL,
  subset = NULL,
  na.action = na.omit,
  contrasts = NULL,
  mode = c("confirmatory", "strict", "exploratory", "predictive"),
  approximation = c("laplace", "agq"),
  nAGQ = 1,
  inference = c("auto", "none", "asymptotic", "bootstrap"),
  control = mm_control(),
  ...
)
```

For GLMMs, `approximation = "laplace"` is the user-facing default and
corresponds to `nAGQ = 1`. `approximation = "agq"` requires `nAGQ > 1`.
PIRLS is an internal fitting engine detail, not a public approximation choice.

Optional compatibility aliases can be considered later:

```r
lmer_mm(...)
glmer_mm(...)
```

An exact `lmer()` alias should be opt-in because masking `lme4::lmer()` creates
more confusion than it solves.

### Prefit API

The prefit path should be available before any numerical optimization:

```r
spec <- compile_model(
  y ~ condition + (1 | subject:item),
  data = dat,
  mode = "confirmatory"
)

explain_model(spec)
audit_design(spec)
parameterization(spec)
as_json(spec)
```

Convenience form:

```r
explain_model(y ~ condition + (1 | subject:item), data = dat)
audit_design(y ~ condition + (1 | subject:item), data = dat)
```

### Random Specification API

Keep lme4 formula syntax as the on-ramp:

```r
lmm(
  y ~ condition + (1 + condition | subject) + (1 | item),
  data = dat
)
```

For v1, do not put native `re()` / `vc()` calls inside the fixed-effect
formula. R's formula and model-frame machinery may otherwise try to evaluate
them as ordinary terms. Use a separate `random` argument for explicit native
random-effect constructors:

```r
lmm(
  y ~ condition,
  random = re(subject, basis = ~ 1 + condition, cov = "full") +
           vc(item),
  data = dat,
  roles = roles(
    subject = "sampled_unit",
    item = "sampled_unit",
    condition = "fixed_condition"
  )
)
```

Proposed extension helpers:

- `re(group, basis = ~ 1, cov = c("full", "diag", "scalar"))`
- `vc(group)` as sugar for scalar variance components
- `roles(...)` for sampled units, fixed conditions, repeated units, time, item,
  cluster, and blocking variables
- `mm_control()` for optimizer, thresholds, reproducibility, and verbosity
- `mm_thresholds()` for compiler-policy thresholds

The lme4 formula path and the `random = ...` path should compile to the same
Rust semantic IR. The native helpers must not create an R-only model language.

## Formula/Data Manifest Handshake

R evaluates formulas and data. Rust owns formula semantics after translation.
The bridge should make that boundary explicit with a manifest call before
full model compilation:

```text
mm_formula_manifest(formula_string, dialect, schema_version) ->
  variables_needed
  transformations_needed
  random_terms_detected
  unsupported_syntax_diagnostics
```

The R-side sequence should be:

1. Capture the formula, `random` expression, roles, and formula environment.
2. Ask Rust for a formula manifest using the declared dialect.
3. Evaluate R-specific transformations requested by the manifest.
4. Apply `subset`, `na.action`, offsets, and weights policy.
5. Freeze factor levels, contrasts, row order, and row names.
6. Send a deterministic data schema and payload to Rust.
7. Let Rust compile, audit, explain, and fit the model.

This avoids R quietly becoming a formula compiler. R may evaluate expressions,
but Rust decides which variables are part of the model, which syntax is random
effect syntax, which transformations are unsupported, and how the requested
model maps to semantic IR.

## Fit Object

Use S3 classes initially, backed by a Rust external pointer plus stable R-side
metadata:

```r
class(fit)
# c("mm_lmm", "mm_fit")
```

The external pointer is a cache, not the source of truth. Every fitted object
must be reconstructible after `saveRDS()` / `loadRDS()`, process fork, or
knitr cache restore from durable R-side state:

- model spec JSON
- compiled artifact JSON
- model-state and audit-report JSON
- final parameter vectors and compact numeric summaries needed to revive the
  Rust handle
- optional raw serialized engine bytes if a later Rust API supports them

The package should provide:

```r
fit_handle_alive(fit)
revive(fit)
```

Extractor methods may auto-revive a missing handle. If revival is impossible
because the Rust crate or schema version is incompatible, the method must fail
with a structured diagnostic rather than touching a dangling pointer.

The object should contain:

- original call
- original formula and terms object
- model-frame row mapping and missing-data policy
- factor level and contrast declarations sent to Rust
- external pointer to the Rust fit, when available
- compact cached JSON artifacts:
  - compiled model artifact
  - model audit report
  - model state summary
  - reproducibility record
- cached R tables for common print/extractor paths
- R-side reproducibility metadata: R version, locale, `options("contrasts")`,
  package versions, and bridge/schema versions

Do not store large dense matrices in the R object by default. Provide lazy
extractors for `X`, `Z`, `Lambda`, `theta`, and related internals.

## Printing and Summaries

Default `print(fit)` should be short:

```text
Linear mixed model fit by REML
Formula: Reaction ~ Days + (Days | Subject)
Mode: confirmatory / maximal feasible
Fit status: converged_boundary
Effective random effects: unchanged
Diagnostics: 1 boundary variance component
Inference: Satterthwaite available; KR not assessed
Use audit(), changes(), or optimizer_certificate() for details.
```

`summary(fit)` should include familiar sections:

- formula and fitting method
- fit status and optimizer certificate summary
- fixed effects table
- random effects / `VarCorr` table
- residual scale
- diagnostics summary
- coefficient-level contrast tests when requested
- inference method/status/reliability per inferential row

`summary(fit, tests = "coefficients", method = "auto")` should test
coefficient-level contrasts. Every inferential row should carry `estimate`,
`std.error`, `numerator.df`, `denominator.df`, `statistic`, `p.value`,
`method`, `status`, `reliability`, and `estimability`. If the Rust artifact
says inference is unavailable, R prints `NA` with a reason instead of
manufacturing a plausible p-value.

Rendering should use `cli` for severity styling, actionable hints, and
terminal-safe links to drilldowns such as `audit(fit)` and
`parameterization(fit)`. Tables should use stable, cached R objects so repeated
`summary(fit)` calls do not re-query the Rust engine.

Hard failures should use typed R conditions whose payloads wrap Rust
diagnostics:

- `mm_not_identifiable`
- `mm_design_refusal`
- `mm_fit_not_optimized`
- `mm_inference_unavailable`
- `mm_schema_error`
- `mm_formula_error`

Successful fits with boundary, reduced-rank, or other diagnostic information
should not emit warning spam by default. Store diagnostics in the fit object,
show compact status in `print()`, and expose details through `audit()`,
`changes()`, `diagnostics()`, and `optimizer_certificate()`.

## Familiar Extractors

The R layer should implement these generics with lme4-like names and shapes
where the semantics match:

- `fixef(fit)`
- `ranef(fit, condVar = FALSE)`
- `coef(fit)`
- `VarCorr(fit, sigma = 1)`
- `sigma(fit)`
- `vcov(fit, type = c("fixed", "theta"))`
- `residuals(fit, type = c("response", "pearson", "working"))`
- `fitted(fit)`
- `logLik(fit)`
- `deviance(fit)`
- `AIC(fit)`, `BIC(fit)`
- `model.frame(fit)`
- `model.matrix(fit, type = c("fixed", "random"))`
- `getME(fit, name)`
- `simulate(fit, nsim = 1, seed = NULL, ...)`
- `refit(fit, newresp, ...)`
- `is_singular(fit, tol = NULL)`
- `recover_data.mm_fit()` and `emm_basis.mm_fit()` for future `emmeans`
  integration

`getME()` should be read-only and explicit about support. The initial support
set should include common lme4 names only when Rust can return them without
reconstructing internal mappings in R:

- `"X"`
- `"Z"`
- `"theta"`
- `"Lambda"`
- `"cnms"`
- `"flist"`
- `"Gp"`
- `"lower"`
- `"devcomp"`
- `"optinfo"`

Where a name has no exact equivalent, return a structured error that points to
`parameterization(fit)` or `audit(fit)`.

`getME()` is a compatibility shim, not the preferred interface. It may emit a
soft lifecycle hint pointing to the corresponding `parameterization(fit)` or
`audit(fit)` field.

`model.matrix(fit, type = "random")` should return the full sparse `Z` as a
`Matrix::sparseMatrix`, matching the most common lme4 expectation. A separate
`random_blocks(fit)` helper can expose per-term blocks.

`ranef(fit, condVar = TRUE)` should be supported as an explicit, potentially
expensive path. The bridge should return random effects plus per-group
conditional covariance blocks without materializing an avoidable dense
`n_groups x q x q` array.

`is_singular()` should map boundary and reduced-rank fit statuses to `TRUE`,
but its message should point users to `audit(fit)` for the full reduced-rank
or boundary explanation. If lme4 is attached, an `isSingular()` method can
delegate to the same implementation.

### emmeans Integration

Once Rust exposes fixed-effect covariance and denominator-df machinery, support
`emmeans` as a first-class downstream client:

```r
recover_data.mm_fit(...)
emm_basis.mm_fit(...)
```

The R layer should make these Rust-owned quantities available:

- `fixef(fit)`
- `vcov(fit, type = "fixed", method = ...)`
- `df_for_contrast(fit, L, method = ...)`
- `estimability(fit, L)`

This lets marginal means, adjusted pairwise contrasts, and custom contrast
workflows reuse the same inference status and estimability contract as
`summary()`, `contrast()`, and `anova()`.

## New Drilldown API

These are first-class, not secondary diagnostics:

```r
audit(fit)
audit_design(fit)
changes(fit)
explain_model(fit)
parameterization(fit)
diagnostics(fit, severity = NULL, stage = NULL)
fit_status(fit)
optimizer_certificate(fit)
reproducibility(fit)
inference_table(fit)
estimability(fit, L = NULL)
df_for_contrast(fit, L, method = "auto")
fit_handle_alive(fit)
revive(fit)
```

`parameterization(fit)` should be the R face of `ThetaMap`: source term,
semantic basis, optimizer basis, covariance family, `theta` slots, `Lambda`
slots, active/boundary status, and user-scale back-transforms.

`changes(fit)` should show requested to semantic to supported to fitted
transitions with the same categories as the Rust contract:

- design-time diagnostic or reduction
- certificate-time boundary
- selection-time reduction

## Prediction API

Keep lme4 compatibility for common usage, then add explicit uncertainty
targets:

```r
predict(
  fit,
  newdata = NULL,
  re.form = NULL,
  allow.new.levels = FALSE,
  type = c("link", "response"),
  se.fit = FALSE,
  target = NULL,
  condition_on = NULL,
  interval = c("none", "confidence", "prediction"),
  level = 0.95,
  ...
)
```

Rules:

- `target` is the explicit Rust prediction target. Supported values should
  align with the Rust enum: `"conditional"`, `"population"`,
  `"new_group_mean"`, `"new_observation"`, and `"partial"`.
- `re.form` is lme4-compatible sugar that compiles to `target` when `target`
  is `NULL`.
- `re.form = NULL` maps to `target = "conditional"`.
- `re.form = NA` maps to `target = "population"`, matching lme4 convention.
- A partial `re.form` maps to `target = "partial"` with `condition_on`
  identifying the random-effect terms to condition on.
- If both `target` and a non-default `re.form` are supplied, they must agree;
  conflicting requests are errors.
- New grouping levels require `allow.new.levels = TRUE`.
- Prediction intervals must identify which components are included: fixed
  effects, random effects, residual variation, and total uncertainty.
- R should refuse impossible targets using Rust-owned diagnostics.

## Model Comparison and Inference

`contrast()` is the primitive inferential operation. Other user-facing tests
compile to one or more fixed-effect contrast matrices or to explicitly
supported model comparisons.

Recommended front doors:

```r
contrast(fit, L, rhs = 0, method = "auto")
test_effect(fit, "duration:season", method = "auto")
summary(fit, tests = "coefficients", method = "auto")

anova(fit, type = c("III", "II", "I"),
      method = c("auto", "satterthwaite", "kenward_roger", "bootstrap"),
      refit_for_comparison = c("auto", "error", "ml"))

anova(fit1, fit2, ...,
      refit_for_comparison = c("auto", "error", "ml"))

compare(fit1, fit2, target = c("fixed_effects", "random_effects", "prediction"),
        method = c("auto", "lrt", "bootstrap", "cross_validation"),
        refit_for_comparison = c("auto", "error", "ml"))

drop1(fit, method = c("auto", "kenward_roger", "bootstrap"),
      refit_for_comparison = c("auto", "error", "ml"))
```

Policy:

- `summary()` may show coefficient-level contrast tests only when Rust marks
  them available.
- Coefficient-level Wald fallback p-values are allowed when Rust supplies the
  row-level method/status/reliability contract described in
  `fixed_effect_p_values_plan.md`; model-level finite-sample inference may
  still be marked not assessed.
- The durable payload lives at
  `artifact.fixed_effect_inference_table`; live Rust handles may expose the
  same JSON through the `fixed_effect_inference` bridge table. R must prefer
  that table over `beta`/`std_errors` reconstruction.
- `contrast()` returns one row per contrast with `estimate`, `std.error`,
  `numerator.df`, `denominator.df`, `statistic`, `p.value`, `method`,
  `status`, `reliability`, and `estimability`.
- `test_effect()` and single-model `anova()` test scientific terms, not just
  printed coefficient rows.
- Single-model `anova()` should use Roman-numeral test types in the public API
  and document the mapping to sequential and marginal term tests.
- Multi-model `anova(fit1, fit2, ...)` should route to `compare()` and return
  the same status/method/reliability columns.
- `AIC(fit1, fit2, ...)` may support multi-model ranking. `logLik(fit)` stays
  per model.
- `anova()`, `compare()`, and `drop1()` must handle REML versus ML refits
  explicitly through `refit_for_comparison`.
- `refit_for_comparison = "error"` refuses invalid REML comparisons.
- `refit_for_comparison = "ml"` performs an ML refit for comparison and records
  the refit in `changes()`.
- `refit_for_comparison = "auto"` may refit when statistically required, but
  the original REML fit summary remains unchanged and the comparison table
  must state which fitted objects were compared.
- Random-effect tests should not use naive ordinary p-values.
- Exploratory or regularized fits should label ordinary confirmatory inference
  as unavailable unless an explicit unpenalized refit or selective-inference
  contract exists.

## FFI and Wire Boundary

v0 chooses JSON as the stable metadata wire format. Large numeric arrays can
use native vectors, sparse triplets, or a future Arrow/native path, but all
model-state, diagnostic, certificate, and audit semantics cross the boundary
as versioned JSON.

The JSON contract already exists in the Rust design through:

- `mixedmodels.compiled_model_artifact`
- `mixedmodels.model_state_summary`
- `mixedmodels.model_audit_report`
- diagnostics
- optimizer certificate
- reproducibility record

### Distribution, Bridge, and Namespace Policy

Default recommendation:

- target CRAN-compatible packaging, with R-universe and GitHub as development
  distribution channels
- use `mixeff-rs` with `default-features = false` for the first CRAN
  submission; reserve the default NLopt-backed Rust profile for r-universe,
  GitHub, local, and later CRAN-performance builds
- keep `lme4` in `Suggests`, not `Imports`; use it for parity tests and
  optional generic compatibility
- do not mask `lme4::lmer()` or `lme4::glmer()` on attach
- prototype with `extendr` if it accelerates early work, but keep a documented
  C ABI escape hatch; the wire contract must not depend on the bridge choice
- use stable Rust only and declare an MSRV once the R bridge crate is chosen

For S3 generics:

- use base/stats generics directly where they already exist
- provide package functions for `fixef()`, `ranef()`, `VarCorr()`, `getME()`,
  and `is_singular()`
- conditionally register methods for lme4-owned generics when lme4 is loaded
- document the semantic divergence from `merMod` in `?lmm`
- run a CRAN namespace-collision scan before finalizing broad verbs such as
  `compare()` and `audit()`, and keep namespace-qualified examples where a
  collision is plausible

Fit handles are process-local. They must not be shared across R sessions or
treated as durable state. Multiple R sessions may load the same Rust dynamic
library, but the bridge must avoid mutable global state and synchronize access
to each live handle.

Recommended low-level Rust calls exposed to R:

```text
mm_formula_manifest(formula_string, dialect, schema_version) -> manifest_json
mm_compile(spec, data) -> artifact_json
mm_explain(spec, data) -> explanation_json
mm_fit(spec, data) -> fit_handle + artifact_json + summary_json
mm_artifact(fit_handle) -> artifact_json
mm_audit(fit_handle) -> audit_json
mm_table(fit_handle, table_name) -> data.frame-compatible payload
mm_ranef(fit_handle, condvar) -> random-effects payload
mm_predict(fit_handle, newdata_spec) -> prediction payload
mm_contrast(fit_handle, contrast_spec) -> inference-table payload
mm_estimability(fit_handle, contrast_spec) -> estimability payload
mm_df_for_contrast(fit_handle, contrast_spec) -> df payload
mm_simulate(fit_handle, simulation_spec) -> simulated-response payload
mm_refit(spec, response_payload) -> fit_handle + artifact_json + summary_json
mm_release(fit_handle)
```

Every long-running call must accept an interrupt-poll callback or equivalent
handle so R's Ctrl+C can cancel safely. Cancellation must release temporary
native resources and return an interrupted status without corrupting the saved
R-side artifact state. Bootstrap and GLMM fitting are the critical cases.

R sends a model spec with:

- formula string and captured call metadata
- formula dialect and optional separate native `random` specification
- response/family/link
- optional `responses` or `n_responses` field for future multivariate
  shared-theta models
- REML/ML choice
- public mode plus artifact-level fit intent and random strategy derived from
  that mode
- roles declaration
- compiler thresholds and optimizer control
- contrast declarations
- factor levels and ordering
- missing-data row map
- weights, when supported
- schema version expected by the R package

Rust returns:

- schema name/version and crate version
- requested, semantic, supported, and fitted model state
- diagnostics with stable codes
- design audit and model audit report
- `ThetaMap`/parameterization payload
- covariance-kernel or dependence graph payloads when the Rust engine exposes
  them
- optimizer certificate
- inference availability/status
- compact tables for R display

For large matrices, avoid JSON. Use R numeric/integer vectors, sparse matrix
triplets, or a future Arrow/native array path. JSON remains the contract for
metadata, diagnostics, audit state, model-state changes, `ThetaMap`, optimizer
certificates, inference availability, and reproducibility.

### Schema Version Negotiation

Every wire object must carry schema name, schema version, and crate version.
Use semantic schema versions once the R wire layer is introduced. The current
Rust v0 integer schema version can be treated as `major` until the wire
metadata grows explicit `major`, `minor`, and `patch` fields.

- major mismatch: hard refusal before fitting or extraction
- minor mismatch where R is ahead: R may request an older shape if Rust exposes
  a schema downgrade; otherwise refuse
- minor mismatch where Rust is ahead: R may continue only when required fields
  are present and unknown additive fields can be ignored
- patch mismatch: permissive; unknown additive fields are ignored

Schema negotiation belongs in the first native call, not in individual
extractors. A revived object must repeat the negotiation before recreating a
native handle.

### Bridge Invariants

- R never parses Rust warning strings.
- R never recomputes convergence, singularity, estimability, covariance
  reductions, or inference availability.
- Every R-side diagnostic table row has a Rust diagnostic code, stage,
  severity, affected term, and schema version.
- Every displayed requested/supported/fitted formula comes from Rust artifact
  state, not from R-side formula surgery.
- Every covariance parameter displayed by R is traceable through Rust
  `ThetaMap`/`CovarianceMap` metadata.
- `saveRDS()` / `loadRDS()` must preserve enough JSON state to reproduce
  `audit()`, `changes()`, `parameterization()`, and default print output
  without a live Rust pointer.
- If the Rust crate or schema version cannot revive a handle, R returns a
  structured compatibility diagnostic.

## R Data Translation

The R layer should construct a deterministic data schema before calling Rust:

- use `mm_formula_manifest()` to decide which variables and transformations
  must be evaluated before data are frozen
- accept `data.frame` inputs directly and coerce `tibble` / `data.table`
  inputs to base `data.frame` before schema construction
- treat Arrow tables and other lazy data sources as unsupported in v1 unless
  explicitly collected by the caller
- freeze row order after `subset` and `na.action`
- record original row names and model-frame row indices
- preserve factor level order exactly as R sees it
- record contrasts as named matrices or contrast-policy identifiers
- distinguish numeric, integer, logical, factor, ordered factor, and character
- reject or explicitly convert `integer64`, `Date`, and `POSIXct` columns
  according to documented policy rather than silently coercing through
  `model.matrix()`
- reject unsupported list columns or lossy conversions before calling Rust
- pre-evaluate `offset()` terms in R and pass offsets as named numeric vectors
  until Rust owns offset syntax directly
- pass weights as a separate typed vector with an explicit weight kind
- record all transformations that R performs before Rust receives the data

The Rust engine should still validate that variables named in the formula
exist, that declared types match required roles, and that factor/contrast
payloads are coherent.

## Expected Design Refusal

The product should explain designs before recommending optimizer changes. For
a forum-style model such as:

```r
fit <- lmm(
  effect ~ duration + (1 + duration | sites) + (1 + duration | season),
  data = dat1,
  mode = "confirmatory"
)
```

Expected behavior:

```text
Mixed model not fit: design_refused
Problems:
  1. season has 3 levels and appears to be a condition being compared.
     Treat season as fixed.
  2. sites has 3 levels. The term (1 + duration | sites) requests a full
     2x2 covariance matrix with 3 covariance parameters from 3 grouping levels.
  3. The formula does not include the fixed effects and interactions requested
     by the scientific question.
Suggested starting model:
  lm(log(effect) ~ duration * season * sites, data = dat1)
If the real study has many sampled sites:
  lmm(log(effect) ~ duration * season + (1 | site), data = dat)
```

That is the intended user experience: a statistical explanation of the model
and design, not raw optimizer folklore.

## Package Layout

Proposed repository or package layout:

```text
r-package/
  DESCRIPTION
  NAMESPACE
  R/
    constructors.R
    control.R
    extractors.R
    print.R
    audit.R
    predict.R
    inference.R
    diagnostics.R
    json.R
  src/
    init.c
    rust_bridge.c or extendr-generated bindings
  inst/
    schemas/
    fixtures/
  tests/testthat/
    test-manifest.R
    test-compile.R
    test-fit-lmm.R
    test-diagnostics.R
    test-serialization.R
    test-schema-versioning.R
    test-interrupts.R
    test-extractors.R
    test-predict.R
    test-emmeans.R
```

Expected dependency posture:

- `Imports`: `Matrix`, `cli`, `rlang`
- optional `Imports` or `Suggests` depending on return-shape policy:
  `tibble`, `pillar`
- `Suggests`: `lme4`, `emmeans`, `testthat`, `withr`, `knitr`, `rmarkdown`
- `SystemRequirements`: `Cargo, rustc` for the initial CRAN profile; add
  CMake only if a future CRAN build enables NLopt

The Rust bridge can be implemented with `extendr` or a small C ABI. The
contract should not depend on that choice. The stable boundary is the model
spec, artifact/report schemas, and typed table payloads.

## Implementation Phases

### Phase 0: Packaging and Runtime Decisions

Settle the operational contracts before R code grows around them:

- CRAN-compatible build target, with R-universe/GitHub for development builds
- first CRAN build compiles Rust with `--no-default-features`; NLopt is a
  performance profile outside the initial CRAN submission
- bridge prototype choice and C ABI escape hatch
- Rust MSRV and stable-only build policy
- lme4 as `Suggests`, not `Imports`
- no default masking of `lmer()` / `glmer()`
- schema-version negotiation rules
- interrupt/cancellation callback shape
- saveRDS/loadRDS revival contract
- process-local handle and threading policy
- public mode to internal intent/random-strategy mapping

### Phase 1: Compile, Explain, Audit

Deliver an R package that can:

- capture formula/data
- run the formula/data manifest handshake
- send data schema to Rust
- call compile/explain/audit without fitting
- print `explain_model()` and `audit_design()`
- expose `changes()` and `parameterization()` for compiled specs
- snapshot JSON artifacts from the existing Rust fixtures
- round-trip compiled specs and artifacts through `saveRDS()` / `loadRDS()`
- reject schema mismatches deterministically

The first public demo should show the package explaining:

- `(1 | subject:item)` versus `(1 | subject) + (1 | item)`
- fixed/random redundancy
- too-few grouping levels for requested covariance
- `||` as diagonal random-effect covariance
- requested -> semantic -> supported model state before optimization

This phase proves the R layer is a client of the compiler contract.

### Phase 2: LMM Fit and Core Extractors

Add:

- `lmm()` as a thin wrapper over the same spec/artifact path used by Phase 1
- `print()`, `summary()`
- `fixef()`, `ranef()`, `coef()`, `VarCorr()`, `sigma()`
- `logLik()`, `deviance()`, `AIC()`, `BIC()`
- basic `predict()`
- `simulate()` and `refit()` for cached-spec workflows
- lme4 parity tests for common datasets

No advanced finite-sample p-values are required in this phase. The Phase 2
summary may still show labeled asymptotic Wald coefficient rows when they come
from Rust's versioned fixed-effect inference table; R must not reconstruct
those rows from estimates and standard errors.

### Phase 3: Diagnostics and Parameterization UX

Add:

- `changes()`
- `parameterization()`
- `diagnostics()`
- `optimizer_certificate()`
- `inference_table()`
- `getME()` compatibility subset
- compact default print that highlights only top diagnostics

This phase is where the product should clearly feel more explainable than
`lme4`.

### Phase 4: GLMM Boundary

Add:

- `glmm()`
- family/link translation
- objective approximation metadata
- GLMM-specific inference-unavailable messages
- parity tests for simple binomial and Poisson examples

Do not expose LMM-only Satterthwaite/KR promises for GLMMs.

### Phase 5: Contrast-First Inference

After the Rust engine exposes derivative and inference certificates:

- `test_effect()`
- `contrast()`
- `anova()`
- `drop1()`
- `recover_data.mm_fit()` and `emm_basis.mm_fit()` for `emmeans`
- Satterthwaite/KR/bootstrap result tables with method/status/reliability

### Phase 6: Advanced Workflows

Deferred:

- regularized exploration workflow
- adaptive bootstrap
- residual covariance structures
- profile likelihood intervals
- multivariate shared-theta response API
- high-volume data transfer optimization

## Testing Strategy

1. **Schema tests.** R can read Rust JSON fixtures for compiled artifacts,
   audit reports, diagnostics, model state, and optimizer certificates.

2. **Manifest tests.** `mm_formula_manifest()` returns stable variables,
   transformations, detected random terms, and unsupported-syntax diagnostics
   before R evaluates and freezes data.

3. **lme4 parity tests.** For ordinary models, compare fixed effects,
   `theta`, `sigma`, log-likelihood, `VarCorr`, fitted values, and predictions
   against `lme4` within documented tolerances.

4. **Diagnostic tests.** Known problematic formulas should produce stable
   diagnostic codes, stages, severities, affected terms, and suggested actions.
   Design-refusal examples should fail before optimization with typed R
   conditions and Rust diagnostic payloads.

5. **Factor/contrast tests.** Verify R factor levels and contrast matrices
   round-trip into Rust and affect fixed/random basis construction
   deterministically.

6. **Mode tests.** The same formula/data under `confirmatory`, `strict`,
   `exploratory`, and `predictive` should produce distinct status and
   inference-availability behavior where the Rust contract says so.

7. **No R-only decisions.** Unit tests should assert that R summaries are
   driven by Rust status fields, not by ad hoc R-side convergence heuristics.

8. **Serialization tests.** `saveRDS(fit)` followed by `loadRDS()` in a fresh
   R session must preserve print, `audit()`, `changes()`,
   `parameterization()`, and prediction results within tolerance after handle
   revival.

9. **Schema mismatch tests.** Exercise R package schema N against Rust schema
   N, N+1, and N-1 fixtures, including unknown additive fields.

10. **Interrupt tests.** Cancel a long-running fit or simulated long call and
   verify that no native handles leak and the R object remains usable or
   clearly failed.

11. **Namespace tests.** Verify behavior with and without lme4 attached,
    including optional method registration and no default masking of
    `lmer()` / `glmer()`.

12. **emmeans tests.** When inference support is available, verify
    `recover_data.mm_fit()` and `emm_basis.mm_fit()` expose Rust-owned fixed
    effects, covariance, df, and estimability.

13. **Performance tests.** Keep R overhead visible: JSON parse for a
    sleepstudy-scale artifact should be under 5 ms, cached `summary(fit)`
    should be under 1 ms, and ordinary `lmm(sleepstudy)` should stay within
    a documented factor of `lme4::lmer()` once fitting is exposed.

14. **Static policy tests.** Scan R sources for construction of convergence,
    singularity, covariance-reduction, or inference-availability statuses in R.
    Those statuses must be read from Rust-owned fields.

## Open Decisions

- R package name.
- S3 only versus an S4 shell for closer `lme4` method compatibility.
- Whether to provide opt-in `lmer()`/`glmer()` aliases outside the default
  attach path.
- ~~How much of R's formula language to pre-evaluate before Rust parsing,
  especially `I()`, `poly()`, and spline helpers.~~ **Decided** — see
  [`formula_transform_seam.md`](formula_transform_seam.md): the engine owns the
  stateless pointwise subset; stateful transforms (`poly`/`scale`/splines) are
  the wrapper's responsibility (ownership model (a), `predvars` above the seam).
- Whether contrast handling should initially mirror R's `model.matrix()` or
  use explicit contrast payloads exclusively.
- Matrix transfer format for large sparse internals.
- Whether v1 must guarantee multi-tenant Shiny-style reentrancy beyond the
  process-local handle and no-global-mutable-state contract above.

## Recommended First Commit Scope

Do not start by fitting from R. Start by making the R package a faithful client
of the Rust compiler contract:

1. define the R model-spec object
2. implement `mm_formula_manifest()` and the formula/data handshake
3. translate R data frames to a deterministic Rust data schema
4. call Rust compile/explain/audit
5. parse and print diagnostics from versioned JSON
6. implement schema negotiation and artifact serialization
7. test against the existing compiler-contract fixtures
8. prove `saveRDS()` / `loadRDS()` works for compiled specs before fitting
9. prove Ctrl+C cancellation for one simulated long-running native call

Once that works, `lmm()` can be a thin wrapper over the same spec/artifact path
rather than a separate interface.
