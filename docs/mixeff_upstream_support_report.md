# mixeff Upstream Support Report

Date: 2026-04-29
Audience: `mixeff` R wrapper maintainers
Upstream: `/Users/bbuchsbaum/code/rust/mixedmodels`
Downstream: `/Users/bbuchsbaum/code/mixeff`

## Executive Summary

Recent `mixedmodels` work materially changes what `mixeff` can expose. The R
wrapper no longer has to treat p-values and random-effect explanation as future
or unavailable surfaces in all cases. Rust now owns a row-level fixed-effect
inference table, populated on fitted LMM artifacts, plus a structured
random-term-card payload in the audit report. These are the two largest changes
`mixeff` should consume.

The important design rule remains unchanged:

> R formats; Rust authors model semantics, inference availability, and reasons.

The immediate `mixeff` upgrade should therefore avoid reconstructing p-values,
degrees of freedom, random-term explanations, or unavailable reasons in R.
Instead, it should parse and format the Rust payloads that now exist.

## What Changed Upstream

### 1. Fitted Artifacts Now Carry Fixed-Effect Inference Rows

Rust fitted `CompiledModelArtifact` values now have:

```text
artifact.fixed_effect_inference_table
```

This field is `null` on compile-only artifacts and populated after fitting when
fixed-effect estimates exist.

Schema:

```text
schema_name    = mixedmodels.fixed_effect_inference_table
schema_version = 1.0.0
```

Each row includes:

- `label`
- `kind`: `coefficient`, `contrast`, or `term`
- `estimate`
- `std_error`
- `numerator_df`
- `denominator_df`
- `statistic`
- `statistic_name`: `z`, `t`, `f`, or `chi_square`
- `p_value`
- `method`: `asymptotic_wald_z`, `satterthwaite`, `kenward_roger`,
  `bootstrap`, or `not_computed`
- `status`: `available`, `p_value_unavailable`, `not_estimable`,
  `not_assessed`, or `unsupported`
- `reliability`: `low`, `moderate`, `high`, or `not_available`
- `estimability`
- `reason`
- `notes`

The Rust source of this contract is
`src/compiler/artifact.rs::FixedEffectInferenceTable` and
`FixedEffectInferenceRow`. The planning contract is
`docs/fixed_effect_p_values_plan.md`.

### 2. `auto` Now Has Real Method Semantics

The upstream auto policy for eligible Gaussian LMM coefficient rows is now:

```text
auto -> satterthwaite -> asymptotic_wald_z -> not_computed
```

This is a change from the earlier plan that described Wald-only auto behavior.
The switch happened after derivative artifacts and `lmerTestR` parity fixtures
landed.

Implication for `mixeff`: do not describe `"auto"` as unavailable or Wald-only.
If Rust emits `method = "satterthwaite"` and `status = "available"`, R may print
the df, t statistic, and p-value because Rust owns the label and reason trail.

### 3. Satterthwaite Rows Are Implemented for Eligible Scalar LMM Rows

Rust now implements the Satterthwaite scalar-contrast path for Gaussian LMMs:

- `deviance_varpar(varpar, reml)`
- `vcov_beta_varpar(varpar)`
- `jac_vcov_beta_varpar(varpar)`
- `vcov_varpar(varpar, reml)`
- scalar row construction through `test_contrast_with_method(...,
  Satterthwaite)`
- fitted coefficient table rows through `fixed_effect_inference_table()`

The row-level contract is validated against `lmerTestR` fixtures in:

```text
tests/fixtures/compiler_contract/satterthwaite_lmer_test_parity_v1.json
```

Supported examples include sleepstudy random-intercept and random-slope
contrasts, crossed Penicillin intercept, an unbalanced sleepstudy variant, and
unavailable paths for boundary/rank-deficient cases.

`mixeff` should consume the row payload. It should not implement the
Satterthwaite formulas in R.

### 4. Explicit Kenward-Roger Rows Are Implemented, Not Auto

Rust now has explicit Kenward-Roger support for supported Gaussian REML LMM
hypotheses:

- dense Sigma/G decomposition
- adjusted fixed-effect covariance
- `Lb_ddf` denominator-df calculation
- scalar KR t rows
- multi-df KR F rows
- no silent fallback for explicit KR requests

KR remains opt-in in schema `1.0.0`. It is not part of the default auto ladder.

Parity is pinned in:

```text
tests/fixtures/compiler_contract/kenward_roger_pbkrtest_parity_v1.json
```

Important caveat: the current multi-df row parity tracks the unscaled
`pbkrtest::KRmodcomp()` F statistic/p-value. Fixtures also store scaled F
values and `F.scaling`; rows with non-unit scaling remain explicitly documented
as not yet using pbkrtest's scaled F output in the row payload.

`mixeff` should expose KR through explicit method requests once the bridge can
ask Rust for contrast or term rows. It should not let explicit KR degrade to
Satterthwaite or Wald on the R side.

### 5. Bootstrap Fixed-Effect Rows Have a Certified Payload Path

Rust now distinguishes bootstrap distributions from bootstrap hypothesis tests.
The key rule:

> A bootstrap p-value requires a certified `fixed_effect_null` target.

Implemented pieces include:

- bootstrap run metadata schema `mixedmodels.bootstrap_run`, version `1.0.0`
- fixed-effect null target construction
- null simulation/refit support
- payload validation
- continuity-corrected p-values
- Monte Carlo standard error notes
- failed-refit and too-few-replicate unavailable reasons

Bootstrap is explicit only. It is not selected by `auto` in schema `1.0.0`.

`mixeff` should not compute bootstrap p-values from full-model bootstrap
replicates. It should only print a bootstrap p-value if Rust supplied a
`bootstrap` row with `status = "available"`.

### 6. Unsupported and Fragile Cases Are Now Row-Level, Not Global

Rust inference rows can suppress p-values with stable status and reason fields
for:

- rank-deficient fixed effects
- non-estimable contrasts
- missing or non-positive standard errors
- predictive, exploratory, regularized, or post-selection fit intent
- derivative prerequisites unavailable
- boundary or reduced-rank cases where a requested method is not defensible

Boundary and reduced-rank covariance states do not automatically suppress all
coefficient rows. Rust decides row by row. `mixeff` should not apply broad
R-side bans like "singular fit means no p-values"; it should format the row
status Rust provides.

### 7. Model Audit Reports Now Include Random Term Cards

`ModelAuditReport` schema version `2` now includes:

```text
random_term_cards
cross_card_constraints
```

Each card has schema:

```text
schema_name    = mixedmodels.random_term_card
schema_version = 1
```

The card fields are:

- `term_id`
- `original_fragment`
- `canonical_fragment`
- `group`
- `blocks`
- `implied_constraints`
- `design_support`
- `role_origin`

Each `blocks[]` entry carries:

- `basis`
- `intercept`
- `slopes`
- `covariance`
- `theta_parameters`
- `english`

`english` is upstream-authored. R should render it, not rewrite it.

`design_support` carries:

- `group_levels`
- `min_rows_per_group`
- `median_rows_per_group`
- `within_group_variation`
- `status`

`role_origin` currently records observed roles:

```text
declared_by_user = false
observed_from_data = true
role = <resolved GroupingRole>
```

This is ready for future R-side `roles()` declarations without changing the
consumer shape.

### 8. Cross-Card Constraints Explain Fixed Zero Covariances

For split-block or `||` forms, audit reports can include
`cross_card_constraints` with:

- `type = zero_covariance`
- `between_cards`
- `between_basis`
- `reason`

This lets `mixeff` explain that an intercept-slope covariance is fixed at zero
because of syntax, without deriving that fact from formula strings.

The upstream implementation chose a multi-IR-entry decomposition for `||`.
That means `(1 + x || g)` can surface as separate cards tied by a cross-card
constraint, rather than one card with multiple blocks. `mixeff` should format
cards plus constraints together.

### 9. Pedagogical Diagnostics Are Additive

The random-term-card PRD requested informational diagnostic taxonomy variants:

- `scope_note`
- `support_note`
- `syntax_expansion`
- `covariance_assumption`
- `structural_refusal`

`mixeff` should treat these as informational categories used to route tone and
placement. They are not optimizer failures by themselves. Existing optimizer or
fit-status diagnostics remain the source for hard failures.

### 10. Julia Parity Fixture Drift Has a Gate

A new explicit parity gate exists for Julia-backed reference fixtures:

```sh
scripts/check_julia_parity_fixtures.sh
```

This regenerates Julia/MixedModels.jl references into a temporary directory and
compares them against checked-in fixtures with tight tolerances. It currently
covers AGQ, ranef, parmap, rank-deficient metrics, Gamma GLMM, and MMJL
pathology references.

This is not a `mixeff` runtime feature, but it improves trust in the upstream
fixtures that support downstream claims.

## Current mixeff Mismatch

The current `mixeff` R code still reflects an earlier contract in a few places.

### `inference_table.mm_lmm`

Current behavior in `R/revive.R` builds an unavailable table from `fit$beta` and
`fit$std_errors`, with:

```r
method = "unavailable"
status = "unavailable"
reason = "not_certified_by_rust_inference_contract"
```

This is now stale for fitted LMM artifacts. The fitted artifact should be the
source:

```r
fit$artifact$fixed_effect_inference_table
```

If the field is present and schema-valid, `inference_table(fit)` should return
those rows. Only fall back to the old unavailable table when the artifact field
is absent, which should mainly mean a pre-upgrade fit object or a compile-only
artifact.

### `summary.mm_lmm`

Current summary builds coefficient p-value columns as all `NA`. It should
instead join the fixed-effect inference rows to the coefficient table by
coefficient label.

Suggested display mapping:

| Rust row field | R summary column |
|---|---|
| `estimate` | `Estimate` |
| `std_error` | `Std. Error` |
| `denominator_df` | `df` |
| `statistic` | `z value`, `t value`, or `F value`, based on `statistic_name` |
| `p_value` | `Pr(>|z|)`, `Pr(>|t|)`, or method-neutral `p.value` |
| `method` | keep as a visible or footer field |
| `status` / `reason` / `notes` | print in inference-status section |

Do not compute p-values in `summary.mm_lmm`. If `p_value` is `NULL`, print
`NA` and the Rust reason.

### `contrast.mm_lmm` and `test_effect.mm_lmm`

These functions currently validate dimensions in R and return unavailable
tables. Upstream Rust has lower-level support for explicit Satterthwaite,
Kenward-Roger, Wald, and bootstrap rows, but the current `mixeff` bridge does
not yet expose a contrast/table endpoint.

Recommended next bridge endpoint:

```text
mm_fixed_effect_contrast_json(
  artifact_or_fit_handle,
  L,
  rhs,
  method
) -> FixedEffectInferenceTable
```

If the live handle is unavailable after `saveRDS()`, `mixeff` can either revive
the fit or return an unavailable row with reason
`rust_fit_handle_required_for_new_contrast`. It should not reconstruct
Satterthwaite/KR/bootstrap rows from the saved artifact unless Rust adds a
side-effect-safe serialized-artifact evaluator for those methods.

### Schema Negotiation

`mixeff` currently knows:

- `formula`, `v0`
- `mixedmodels.compiled_model_artifact`, `1`
- `mixedmodels.model_audit_report`, `2`
- `mixedmodels.random_term_card`, `1`

Add:

```text
mixedmodels.fixed_effect_inference_table = 1.0.0
```

If and when `mixeff` consumes bootstrap run payloads directly, also add:

```text
mixedmodels.bootstrap_run = 1.0.0
```

Because `fixed_effect_inference_table` is nested inside the artifact, R can
validate it in its parser without requiring a separate FFI function. The
important thing is to fail cleanly on schema mismatches.

## Recommended mixeff Implementation Plan

### Phase A: Parse and Format Existing Fitted Inference Tables

Files likely involved in `mixeff`:

- `R/json.R`
- `R/schema.R`
- `R/revive.R`
- `R/methods-summary.R`
- `src/rust/src/lib.rs`
- `tests/testthat/test-inference.R`
- `vignettes/inference.Rmd`

Tasks:

1. Add `mixedmodels.fixed_effect_inference_table` `1.0.0` to the known schema
   list in the Rust bridge manifest.
2. Add an internal parser such as `mm_json_parse_fixed_effect_inference_table()`
   or a list parser for `artifact$fixed_effect_inference_table`.
3. Update `inference_table.mm_lmm()` to prefer the artifact table.
4. Keep the old unavailable fallback only for legacy objects or missing tables.
5. Update `summary.mm_lmm()` to display available row values.
6. Add tests using current fitted LMMs where Rust emits Satterthwaite rows.
7. Add tests for rank-deficient/not-estimable rows, making sure R preserves
   `status`, `reason`, and `notes`.

Acceptance criteria:

- `inference_table(lmm(...))` shows at least one `method = "satterthwaite"` row
  on a supported LMM.
- `summary(fit)` prints numeric p-values only when Rust row status is
  `available`.
- Unavailable rows preserve Rust reasons without R-side rewrites.
- Saved and revived fits still expose the same inference table from the stored
  artifact.

### Phase B: Improve Random Term Card Consumers

Files likely involved:

- `R/audit.R`
- `R/explain.R`
- `R/random-options.R`
- `R/diagnostics.R`
- `vignettes/demystifying-formulas.Rmd`
- `vignettes/lmm-basics.Rmd`

Tasks:

1. Treat `audit_design(spec)$random_term_cards` as the primary structured
   source for random-effect explanations.
2. Render `card$blocks[[i]]$english` verbatim.
3. Render `cross_card_constraints` after the relevant cards.
4. Use `design_support` for compact factual summaries: levels, min/median rows,
   within-group variation, information-budget status.
5. Use `original_fragment` and `canonical_fragment` to explain syntax
   expansion without parsing formula text.
6. Use `role_origin` to distinguish observed roles now and user-declared roles
   later.

Acceptance criteria:

- `(1 + x || g)` and `(1 | g) + (0 + x | g)` show equivalent covariance
  constraints through cards plus cross-card constraints.
- R does not generate new explanatory prose for blocks; it formats Rust
  `english`.
- Snapshot tests assert no R-side advice phrases are introduced.

### Phase C: Add Explicit Contrast and Term-Test Bridge

After Phase A proves table parsing, add bridge endpoints for new rows not
already present in the fitted coefficient table.

Recommended endpoint surface:

```r
contrast(fit, L, rhs = 0,
         method = c("auto", "satterthwaite", "kenward_roger",
                    "bootstrap", "asymptotic", "none"))
```

Implementation rule:

- R validates matrix shape and labels.
- Rust evaluates estimability, method prerequisites, statistics, df, p-values,
  reliability, and reasons.
- R formats the returned `FixedEffectInferenceTable`.

Do not implement Satterthwaite/KR math in R.

### Phase D: Bootstrap Payload Surfacing

Bootstrap rows are useful but should remain explicit. `mixeff` should expose
bootstrap only after it can pass a Rust-certified bootstrap run payload or ask
Rust to produce one.

Minimum R surface:

```r
bootstrap_contrast(fit, L, rhs = 0, B = 999, seed = NULL, ...)
```

or an explicit `contrast(..., method = "bootstrap", B = ...)` only if the
payload and runtime are clear.

R must distinguish:

- full-model bootstrap intervals or diagnostics
- fixed-effect-null bootstrap hypothesis tests

Only the second can produce a bootstrap p-value.

## User-Facing Guidance for mixeff Documentation

Recommended wording direction:

- "p-values are printed when the Rust artifact contains an available row with a
  named method."
- "Unavailable p-values are unavailable by row, not by blanket model class."
- "The method column is part of the result, not a footnote."
- "Satterthwaite is the default finite-sample attempt for eligible LMM
  coefficient rows; Wald is a labeled fallback."
- "Kenward-Roger is explicit in this schema version."
- "Bootstrap p-values require a fixed-effect-null bootstrap target."
- "Random-effect explanations come from Rust-authored random term cards."

Avoid:

- "mixeff computes p-values"
- "singular models have no p-values"
- "Kenward-Roger is the default"
- "bootstrap p-values come from the fitted-model bootstrap distribution"
- R-side explanation of what a random term "means" unless the wording is copied
  from Rust card fields.

## Concrete Migration Checklist

1. Re-vendor or path-update `mixeff` against the current upstream
   `mixedmodels`.
2. Add schema negotiation for `mixedmodels.fixed_effect_inference_table`
   `1.0.0`.
3. Update `mm_formula_manifest()` capabilities to distinguish:
   - `inference = TRUE`
   - `fixed_effect_inference_table = TRUE`
   - `satterthwaite = TRUE`
   - `kenward_roger_explicit = TRUE`
   - `bootstrap_fixed_effect_payload = TRUE` only when R can invoke it.
4. Update `inference_table.mm_lmm()` to use
   `fit$artifact$fixed_effect_inference_table`.
5. Update `summary.mm_lmm()` to use inference rows.
6. Keep p-value columns `NA` when Rust row `p_value` is `null`.
7. Preserve Rust `method`, `status`, `reliability`, `estimability`, `reason`,
   and `notes` in returned objects.
8. Add `contrast()` bridge only after Rust endpoint plumbing exists.
9. Update `audit_design()`, `explain_model()`, and `random_options()` to treat
   `random_term_cards` and `cross_card_constraints` as first-class payloads.
10. Add R snapshot tests for:
    - Satterthwaite available rows
    - Wald fallback rows
    - not-estimable rank-deficient rows
    - explicit KR unavailable-on-ML reason
    - random-term cards and cross-card constraints.

## Priority Recommendation

The highest-value next `mixeff` change is Phase A: consume
`fixed_effect_inference_table` from fitted artifacts. It unlocks real
Satterthwaite/Wald row display without adding any new Rust FFI endpoint and
without violating the no-fabricated-inference contract.

The second highest-value change is Phase B: format `random_term_cards` and
`cross_card_constraints` from `audit_design()`. The R parser already retrieves
these fields; the remaining work is to make the user-facing verbs prefer them
over older placeholder explanations.

Explicit contrasts, KR term tests, and bootstrap calibration should follow
after the fitted artifact table is correctly consumed and tested.
