# Fixed-Effect P-Value Support Plan

Status: Wald, Satterthwaite, explicit Kenward-Roger, and explicit bootstrap payload rows implemented in Rust
Owner: Rust inference and R wire contract
Mote issue: `bd-01KQASCG9KZH36RNTPAHHH2NA9`
Related docs:

- `docs/compiler_contract_v0_prd.md`
- `docs/mixed_model_compiler_inference_contract.md`
- `docs/r_layer_proposal.md`
- `docs/satterthwaite_scalar_contract.md`
- `docs/kenward_roger_contract.md`
- `docs/bootstrap_fixed_effect_contract.md`
- `docs/fixed_effect_p_value_validation.md`
- `/Users/bbuchsbaum/code/mixeff/planning/vision.md`
- `/Users/bbuchsbaum/code/mixeff/planning/mission.md`

## Purpose

The R wrapper needs p-values when the Rust engine can defend them, and explicit
missingness when it cannot. The contract is not "never report p-values." The
contract is:

> Every p-value is a row-level inference result with a named method, status,
> reliability grade, estimability status, and reason/notes where needed.

This plan turns that rule into an implementation path for coefficient-level
tests, user-supplied contrasts, and later finite-sample LMM inference.

The plan is aligned with the `mixeff` vision and mission:

- every printed claim has provenance
- Rust owns model semantics and inference availability
- R returns `NA` with Rust-owned reasons when a method is unavailable
- reviewers can ask which method produced a p-value and inspect
  `inference_table(fit)`

## Current State

Implemented Rust pieces:

- `LinearMixedModel::coeftable()` computes estimates, standard errors, Wald z
  statistics, and asymptotic normal p-values where allowed by fit intent.
- `LinearMixedModel::test_contrast()` returns structured `FixedEffectTest`
  with method, status, reliability, estimability, notes, and optional p-values.
- Exploratory, predictive, regularized, and selection-time-reduced fits can
  suppress ordinary fixed-effect p-values with explicit reasons.
- Fitted `CompiledModelArtifact` values carry
  `fixed_effect_inference_table` with schema name
  `mixedmodels.fixed_effect_inference_table` and schema version `1.0.0`.
- `LinearMixedModel::fixed_effect_inference_table()` builds ordered
  coefficient rows from the Rust contrast-testing path.
- `LinearMixedModel::fixed_effect_term_hypotheses()` and
  `fixed_effect_term_inference_table(method)` expose Rust-owned term tests for
  R `test_effect()` / single-model `anova()` callers.
- `LinearMixedModel::fixed_effect_null_bootstrap_inference_table()` exposes a
  certified fixed-effect null bootstrap path that returns
  `mixedmodels.fixed_effect_inference_table` rows.
- The bridge table accessor exposes the same payload as
  `fixed_effect_inference`.
- Contract fixtures cover confirmatory Wald rows, reduced-rank and boundary
  states, rank-deficient fixed effects, regularized/exploratory/predictive
  suppression, selection-time suppression, and unavailable standard errors.

Remaining gaps:

- The external R wrapper must consume `fixed_effect_inference_table` or the
  `fixed_effect_inference` bridge table in `summary()`, `contrast()`, and
  `inference_table()` rather than reconstructing p-values from estimates and
  standard errors.
- Bootstrap rows remain unavailable until their method-specific null-target,
  run-accounting, Monte Carlo uncertainty, and reproducibility prerequisites
  are certified.

## Contract Split

The wire contract must distinguish two related but different concepts.

### Model Boundary Inference Availability

`model_boundary.inference_availability` remains a model-level statement about
finite-sample or certificate-dependent inference methods. For LMM v0 this may
continue to say:

```text
not_assessed: finite-sample inference is not implemented in compiler v0
```

That statement is not the row-level inference source of truth. It must not be
interpreted as "no coefficient-level fallback or finite-sample row can be
labeled." The row-level table owns the actual method/status/reliability for
each fixed-effect result.

### Row-Level Fixed-Effect Inference

A separate fixed-effect inference payload reports coefficient and contrast
rows. It may contain asymptotic Wald, Satterthwaite, explicit Kenward-Roger,
and later bootstrap results when each row's prerequisites are satisfied.

## Wire Location and Schema Negotiation

The durable source of truth is a top-level optional field on fitted compiler
artifacts:

```text
artifact.fixed_effect_inference_table
```

The field is `null` on prefit-only compiled artifacts and populated after a fit
when fixed-effect estimates are available. Bridge endpoints such as
`mm_table(fit_handle, "fixed_effect_inference")` may return the same payload,
but they are accessors, not the canonical storage location. This preserves the
R-layer rule that `saveRDS()` revival can inspect the artifact without a live
Rust handle.

Schema name:

```text
mixedmodels.fixed_effect_inference_table
```

Version:

```text
1.0.0
```

The version participates in the same major/minor/patch negotiation policy as
the other R-facing JSON schemas. Patch changes are non-breaking; minor changes
may add optional fields; major changes may alter required fields or row
semantics.

Minimum table fields:

| Field | Meaning |
|---|---|
| `schema_name` | `mixedmodels.fixed_effect_inference_table` |
| `schema_version` | Semantic schema version, initially `1.0.0` |
| `crate_version` | Optional Rust crate version |
| `rows` | Ordered inference rows |

Minimum row fields:

| Field | Meaning |
|---|---|
| `label` | Coefficient term or contrast label |
| `kind` | `coefficient`, `contrast`, or `term` |
| `estimate` | Estimate on user scale |
| `std_error` | Optional standard error |
| `numerator_df` | Optional numerator df; `null` for Wald z coefficient rows |
| `denominator_df` | Optional denominator df; `null` for Wald z |
| `statistic` | Optional z, t, F, or chi-square statistic |
| `statistic_name` | `z`, `t`, `f`, or `chi_square` |
| `p_value` | Optional p-value |
| `method` | `asymptotic_wald_z`, `satterthwaite`, `kenward_roger`, `bootstrap`, or `not_computed` |
| `status` | `available`, `p_value_unavailable`, `not_estimable`, `not_assessed`, or `unsupported` |
| `reliability` | `low`, `moderate`, `high`, or `not_available` |
| `estimability` | Structured estimability status from Rust |
| `reason` | Optional reason for unavailable or low-reliability output |
| `details` | Optional structured method/family metadata |
| `notes` | Optional row-level notes |

Rows are scalar by default. Coefficient rows and scalar contrast rows carry one
estimate, one standard error, one statistic, and one p-value. Multi-df term
tests are represented as `kind = "term"` rows with a scalar test statistic
such as F or chi-square, `numerator_df > 1` where applicable, and optional
method-specific detail fields added under the schema-version rules. The
underlying Rust implementation may continue to use vector-valued
`FixedEffectTest`; the table flattens scalar coefficient/contrast tests into
rows and summarizes joint tests as rows rather than exposing vector columns to
R's default table surface.

Row order is deterministic:

1. coefficient rows in fitted fixed-effect coefficient order as returned by
   Rust's coefficient-name API
2. user-supplied scalar contrasts in request order
3. term or joint-hypothesis rows in request order

Labels must be stable under repeated serialization of the same fitted artifact.

## Method Policy

### Method Auto Precedence

`method = "auto"` is a versioned policy, not a synonym for whichever method was
implemented most recently.

Current v1 table behavior:

```text
auto -> asymptotic_wald_z when Wald prerequisites are met; otherwise not_computed
```

Future finite-sample behavior:

```text
auto -> satterthwaite -> asymptotic_wald_z -> not_computed
```

Kenward-Roger is opt-in unless a later major schema version changes the public
default. Bootstrap is opt-in or triggerable by explicit calibration policy; it
is not part of the default coefficient-table auto ladder.

Finite-sample support is a commitment, gated by method-specific prerequisites.
Once those prerequisites are certified, the Rust engine should expose
Satterthwaite, Kenward-Roger, and bootstrap rows through this same table rather
than through a parallel R-only path.

| Method | Support commitment | Required prerequisites before availability |
|---|---|---|
| `satterthwaite` | Supported for eligible LMM fixed-effect scalar contrasts after derivative/covariance certificates exist. | `docs/satterthwaite_scalar_contract.md`: `varpar = c(theta, sigma)`, `deviance_varpar`, `vcov_beta(varpar)`, `jac_vcov_beta`, `vcov_varpar`, finite-difference validation, reliability diagnostics, `lmerTestR` parity fixtures. |
| `kenward_roger` | Supported for eligible explicit LMM scalar and multi-df fixed-effect hypotheses. KR is opt-in for schema `1.0.0`. | `docs/kenward_roger_contract.md`: Sigma/G decomposition, adjusted fixed-effect covariance, denominator-df/F-statistic implementation, explicit no-fallback behavior, `pbkrtest` parity fixtures, singular-adjustment fallback policy. |
| `bootstrap` | Supported as explicit calibration for fixed-effect contrasts and model-comparison rows after certified bootstrap result payloads exist. | `docs/bootstrap_fixed_effect_contract.md`: stable simulation/refit path, null-constrained fixed-effect simulation target, replicate accounting, failed-refit policy, Monte Carlo error reporting, boundary-rate summary, reproducibility/seed record. |

### Lane 1: Labeled Asymptotic Wald Fallback

For confirmatory LMM fits with valid estimates and standard errors, Rust emits
coefficient-level rows using:

```text
method = asymptotic_wald_z
numerator_df = null
denominator_df = null
statistic_name = z
reliability = low
```

This is a large-sample fallback, not a finite-sample correction. The row notes
must say so. R may print the p-value because Rust supplied the method label and
status.

Lane 1 is implemented by reusing the
`CoefTablePValuePolicy`/`test_contrast()` behavior rather than adding an
independent p-value gate. The row-level gate suppresses ordinary p-values for
exploratory, predictive, regularized, and selection-time-reduced fits unless a
later selective-inference or explicit unpenalized-refit contract makes the row
available.

Rows must suppress p-values when:

- the contrast is not estimable
- the standard error is unavailable, non-finite, or non-positive
- fit intent is exploratory, predictive, regularized, or post-selection without
  an explicit valid refit or selective-inference contract
- the requested method is unsupported for the model family
- the optimizer/certificate state invalidates the required covariance inputs

Boundary or reduced-rank covariance state does not automatically suppress Wald
coefficient rows. It should lower reliability or add notes unless it invalidates
the specific row's estimate, standard error, estimability, or requested method.

### Lane 2: Satterthwaite Support

Satterthwaite p-values are in scope for supported LMM classes after the required
derivative and covariance-parameter uncertainty inputs are certified. The MVP
should target scalar fixed-effect contrasts first.

The binding scalar-contrast implementation contract is
`docs/satterthwaite_scalar_contract.md`, derived from the vendored
`vendor/lmerTestR` reference.

Prerequisites:

- `deviance_varpar(varpar, reml)` for `varpar = c(theta, sigma)`
- derivative API for `vcov_beta(varpar)` with respect to variance parameters
- covariance-parameter information or covariance matrix for `varpar` on the
  active fitted manifold
- finite-difference validation tests on small models
- row-level reliability diagnostics for low denominator df, boundary-active
  parameters, and unstable derivatives
- parity tests against `lmerTestR` for selected fixtures

Initial supported scope:

- Gaussian LMMs
- scalar fixed-effect contrasts
- interior or certificate-compatible active manifold
- no residual covariance structures beyond the current iid Gaussian residual
  model

Output:

```text
method = satterthwaite
statistic_name = t
numerator_df = null
denominator_df = <denominator df>
reliability = moderate or low
```

Unsupported rows fall back to `asymptotic_wald_z` only when the user requested
`method = "auto"` and the Wald prerequisites are met. Explicit
`method = "satterthwaite"` requests return `p_value = null` with a reason if
Satterthwaite prerequisites fail.

Satterthwaite is considered available only when the derivative path,
covariance-parameter uncertainty, active-manifold certificate, and row-level
reliability checks all pass for the requested contrast.

### Lane 3: Kenward-Roger Support

Kenward-Roger p-values are in scope for supported LMM classes after the
second-derivative, adjusted-covariance, and denominator-df machinery is
certified. KR is not a v0 requirement and should not be printed until the
method-specific certificates exist.

The binding implementation contract is `docs/kenward_roger_contract.md`.

Prerequisites:

- first and second derivatives for the fixed-effect covariance
- adjusted covariance of fixed effects
- denominator df calculation for scalar and multi-df hypotheses
- parity tests against `pbkrtest`
- clear fallback policy when covariance adjustment is singular or boundary
  active

Output:

```text
method = kenward_roger
statistic_name = t or f
numerator_df = <numerator df for F rows, null for scalar t rows>
denominator_df = <denominator df>
reliability = high, moderate, or low
```

Explicit `kenward_roger` requests must not silently degrade to another method.
Automatic mode does not choose KR in schema `1.0.0`; a later major schema may
change that default only after the public policy is updated.

Kenward-Roger is considered available only when the Satterthwaite-level inputs,
second-derivative inputs, adjusted fixed-effect covariance, denominator-df
calculation, and singular-adjustment fallback checks all pass for the requested
hypothesis.

### Lane 4: Bootstrap Calibration Support

Parametric bootstrap is in scope as the calibration path for fragile cases after
certified simulation/refit and result-accounting payloads exist. It is not the
default coefficient table method.

Output rows should include:

- number of requested and successful replicates
- Monte Carlo standard error when available
- failed-refit policy
- boundary count or boundary-rate summary
- seed/reproducibility record

Bootstrap rows may be used for contrast tests and model comparisons, but random
effect tests must remain boundary-aware and must not use naive ordinary
p-values.

Bootstrap is considered available only when the simulation target, refit
procedure, failed-refit policy, replicate accounting, Monte Carlo uncertainty,
boundary-rate summary, and reproducibility record are present in the payload.

## R Behavior

R is a formatter and cache for Rust-owned inference results.

Required behavior:

- `summary(fit)` and `inference_table(fit)` read the fixed-effect inference
  payload instead of reconstructing p-values from `beta` and `std_errors`.
- `summary(fit, tests = "coefficients")` prints p-values only when row
  `status = "available"` and `p_value` is non-null.
- When `status = "available"` and `reliability = "low"`, R prints the p-value
  and surfaces the reason or notes inline; low reliability is not converted to
  `NA`.
- Rows with unavailable p-values print `NA` plus the Rust reason.
- R does not turn `model_boundary.inference_availability = not_assessed` into
  row-level unavailability when a row-level Wald fallback is present.
- R does not invent Satterthwaite, Kenward-Roger, or bootstrap labels.

For the current sleepstudy random-intercept example, the expected near-term
coefficient row is:

```text
label = (Intercept)
method = asymptotic_wald_z
status = available
reliability = low
numerator_df = null
denominator_df = null
reason = null
notes = ["asymptotic Wald z is a labeled fallback, not a finite-sample correction"]
```

## Implementation Steps

1. [x] Add Rust structs for `FixedEffectInferenceTable`, row metadata, schema
   name, and schema version.
2. [x] Add `LinearMixedModel::fixed_effect_inference_table()` that builds
   coefficient rows by calling `coefficient_hypotheses()` and `test_contrast()`.
3. [x] Ensure row labels preserve coefficient names and user-supplied contrast
   labels.
4. [x] Store `fixed_effect_inference_table` on fitted artifacts and serialize
   it to JSON with schema-version round-trip tests.
5. [x] Add deterministic row-order tests.
6. [x] Add/update snapshot fixtures for `sleepstudy`, `penicillin`, singular
   or reduced-rank models, rank-deficient fixed effects, exploratory or
   regularized fit intent, predictive intent, selection-time reduction,
   unavailable-SE, and boundary/reduced-rank cases.
7. [x] Expose the payload through the Rust bridge table endpoint as
   `fixed_effect_inference`.
8. [ ] Update the external R `summary()` and `inference_table()` surfaces to
   consume the payload. This is outside the Rust crate and should not
   reconstruct p-values from `beta` and `std_errors`.
9. [x] Add Satterthwaite request scaffolding and derivative support. Eligible
   scalar Gaussian LMM fixed-effect rows now use certified Satterthwaite
   inference by default through `auto`, with labeled Wald fallback and explicit
   unavailable reasons when Satterthwaite prerequisites fail.
10. [x] Add KR rows only after adjusted-covariance and denominator-df
    certificates exist. Explicit KR scalar and multi-df rows are now wired,
    KR-labeled unavailable rows do not fall back silently, and `pbkrtest`
    parity fixtures cover supported sleepstudy cases.
11. [x] Add bootstrap rows only after certified parametric-bootstrap result
    payloads exist, including null-target construction, replicate counts,
    failed-refit policy, Monte Carlo error where available, boundary-rate
    summary, and reproducibility state. Explicit bootstrap rows are wired from
    certified `fixed_effect_null` payloads. `auto` does not select bootstrap in
    schema `1.0.0`; general scalar contrasts require payload-supplied
    replicate statistics unless the row is a single coefficient.
12. [x] Add bridgeable Rust-owned fixed-effect null bootstrap table APIs for R
    callers. The table rows include structured `details.bootstrap` metadata so
    R does not parse prose notes for MCSE, replicate accounting, failed-refit
    policy, seed state, or null-target summaries.
13. [x] Add Rust-owned term hypothesis and term-row APIs for R
    `test_effect()` / single-model `anova()` callers.
14. [x] Add optional `details.contrast_family` and
    `details.kenward_roger` row metadata for stable restriction-family rank and
    numerator-df semantics, including current KR multi-df F scaling status.

## Mote Work Breakdown

Umbrella issue: `bd-01KQASCG9KZH36RNTPAHHH2NA9`.

| Issue | Status | Scope |
|---|---|---|
| `bd-01KQATANW75B18YZ9FR21M2WCC` | Done | Add fixed-effect inference table schema and `artifact.fixed_effect_inference_table`. |
| `bd-01KQATAW1PGCBD3E6XEX36RWBV` | Done | Build LMM coefficient rows from `coefficient_hypotheses()` and `test_contrast()`. |
| `bd-01KQATB0WPMAQP3H2VQ1G4S7V6` | Done | Finish p-value gating for fit intent and post-selection states. |
| `bd-01KQATB5MZ4652N6YHWD5484EP` | Done | Add fixed-effect inference table fixtures and row-order tests. |
| `bd-01KQATBAEN6GKB145J6QSYBNGZ` | Done for Rust bridge | Expose the table through the Rust bridge endpoint; external R summary/inference-table consumption remains outside this crate. |
| `bd-01KQATBFPGJ956QAVT26EPJZG3` | Done | Certify Satterthwaite fixed-effect scalar-contrast support after derivative prerequisites and lmerTest parity fixtures; `auto` now prefers Satterthwaite for eligible rows. |
| `bd-01KQB2DEERRCGJ66ZK8CX856BS` | Done | Specify the scalar Satterthwaite contract from `vendor/lmerTestR`. |
| `bd-01KQB2DEF65WZDK2MQRJ55AFJ2` | Done | Implement LMM deviance over `varpar = c(theta, sigma)`. |
| `bd-01KQB2DEJ7Y9W2NT15R3VZD2KF` | Done | Implement `vcov_beta(varpar)` and its Jacobian. |
| `bd-01KQB2DWF3X8PAQMEP3J58ZPAD` | Done | Estimate `vcov_varpar` from the `deviance_varpar` Hessian. |
| `bd-01KQB2DWEDTD9E00KVKWQ3W2MA` | Done for explicit requests | Wire scalar Satterthwaite contrast rows once artifacts are certified; `auto` remains Wald-first until parity fixtures certify the default switch. |
| `bd-01KQB2DWFK0JKH890BA2T1ZEB0` | Done | Add `lmerTestR` parity fixtures. |
| `bd-01KQATBN53CKDTMA5VN5A1YMW3` | Done | Certify Kenward-Roger fixed-effect hypothesis support after adjusted-covariance, denominator-df, row-wiring, and `pbkrtest` parity child issues close. |
| `bd-01KQB8C5KJXX2H0D1K5CPQ5R22` | Done | Specify and implement KR `Sigma/G` component artifacts and diagnostics. |
| `bd-01KQB8C8TA6DAGNCFC1E3R8NGY` | Done | Implement KR adjusted fixed-effect covariance. |
| `bd-01KQB8CD1HAQPAT21JS7DA0V1P` | Done | Implement KR `Lb_ddf` denominator df. |
| `bd-01KQB8CG5TBS4AES1AXGRHNRZA` | Done | Wire explicit KR scalar and multi-df fixed-effect rows. |
| `bd-01KQB8CK9GFQ6VMN49M8B8Y2GW` | Done | Add `pbkrtest` KR parity fixtures. |
| `bd-01KQBDHNVJFZJHSBVB8S15GXEM` | Done | Fix KR full-rank user-order contrast mapping through the active fixed-effect pivot. |
| `bd-01KQATBW8DNAD98P76T667BQCE` | Ready for closeout | Add bootstrap fixed-effect inference payloads after certified simulation/refit payloads. |
| `bd-01KQBDWBXM80J8NYX8GJNHT6Z8` | Done | Specify bootstrap fixed-effect inference contract. |
| `bd-01KQBDWN5HHKX4SF8FPJTPD7YV` | Done | Add certified bootstrap run metadata payload. |
| `bd-01KQBDWN5Q6Z8RRQVXNEJVY1M9` | Done | Implement null-constrained fixed-effect bootstrap target. |
| `bd-01KQBDWN66AR1JZB0MDNPWQZRE` | Done | Wire bootstrap fixed-effect rows from certified payloads. |
| `bd-01KQATC0Y1SFMQTXB09C16DEK3` | Blocked on simulation child motes | Validate fixed-effect p-value methods against internal, `lmerTest`, `pbkrtest`, and simulation references. |
| `bd-01KQBF0ZMP9NK20G0EDJGBW53Q` | Done | Add bounded H0 simulation smoke tests for Wald/Satterthwaite/KR type-I behavior. |
| `bd-01KQBF0ZNDDVSJWZX2R810ND54` | Done | Add a bootstrap fixed-effect calibration fixture using `fixed_effect_null` simulation and certified payload rows. |

## Acceptance Criteria

Immediate Rust-side:

- The Rust bridge exposes coefficient-level tests after fit so external R code
  can request/default them without recomputing p-values or inventing method
  labels.
- Confirmatory LMM coefficient rows with valid SEs expose labeled
  `asymptotic_wald_z` p-values.
- Rank-deficient, exploratory, predictive, regularized,
  selection-time-reduced, unavailable-SE, and unsupported-method cases return
  explicit reasons rather than numeric p-values.
- `model_boundary.inference_availability` remains finite-sample-specific and
  does not suppress row-level fallback inference.
- JSON schema name/version, wire location, row ordering, and artifact
  round-trip behavior are tested.

External R wrapper:

- `summary()`, `contrast()`, and `inference_table()` should consume the Rust
  table and print low-reliability available p-values with their notes.

Finite-sample:

- Satterthwaite rows are emitted only with derivative/covariance certificates.
- KR rows are emitted only with adjusted covariance and denominator-df
  certificates.
- Supported Satterthwaite/KR fixtures have parity tests against
  `lmerTest`/`pbkrtest` where feasible.
- Unsupported finite-sample rows fall back only under `method = "auto"` and
  otherwise return `p_value = null` with a method-specific reason.

## Non-Goals

- No GLMM Satterthwaite/KR promise.
- No naive random-effect p-values.
- No ordinary confirmatory p-values for exploratory, predictive, regularized,
  or post-selection workflows unless a later selective-inference or explicit
  unpenalized-refit contract is implemented.
- No R-side reconstruction of p-values from estimates and standard errors.
