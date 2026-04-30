# Random-Effects Formula Handling — v0 Contract

This document is the formula-layer slice required by §3 (Semantic Random-Effects IR) of [`compiler_contract_v0_prd.md`](compiler_contract_v0_prd.md). It pins down the deterministic rules for parsing, canonicalizing, and materializing random-effects formulas, and it states what the compiler reports back to the user.

Companion documents:

- [`compiler_contract_v0_prd.md`](compiler_contract_v0_prd.md) — the binding v0 contract. Owns ThetaMap, the deterministic `maximal_feasible` v0 rule, design audit, estimability typing, the KKT certificate, the serialization boundary, the GLMM boundary, reproducibility, and performance budgets. Where this doc and the v0 PRD overlap on what v0 must do, the PRD wins.
- [`mixed_model_compiler_inference_contract.md`](mixed_model_compiler_inference_contract.md) — broader product vision. Non-binding except where its content has been lifted into the v0 PRD.
- [`multivariate_shared_theta.md`](multivariate_shared_theta.md) — vNext design for multivariate Y; out of v0 scope.

The v0 PRD owns *what gets done* with a fitted model and how transformations are recorded. This document owns *what model is actually being fit* once a user types a formula. Where the two could disagree, this document defers and notes the cross-reference.

This is a v0 contract. §10 records the resolved deterministic decisions that
make the formula layer implementable; non-conforming code paths are listed in
Appendix A rather than left as product ambiguity.

---

## 1. Scope and non-goals

In scope:

- Surface syntax for random-effect blocks: `(re | g)`, `(re || g)`, with grouping forms `g`, `g1 & g2`, `g1 : g2`, `g1 / g2`, `g1 * g2`.
- Canonicalization of the parsed AST into a deterministic IR before materialization.
- Random-effects basis construction (which columns go into `Z_g` for a term).
- Grouping factor materialization (composite levels, empty cells, level ordering).
- The requested → canonical → effective formula reporting model.
- Diagnostics emitted at parse time and at canonicalization time.

Out of scope (deferred to other documents or vNext):

- Fixed-effect contrast coding policy beyond what's needed to mirror it on the random side. The fixed side gets its own treatment elsewhere.
- GAM-style smooths: `s()`, `te()`, `bs()`, `ns()`. Not parsed in v0.
- Offsets, `I(...)`, `poly(...)`. Not parsed in v0.
- Multivariate response: see `multivariate_shared_theta.md`.
- Residual-structure formula syntax (e.g. `residual = ar1(time, subject)`): see the residual-model section of the inference contract.
- Post-fit reductions (rePCA, KKT-driven boundary reduction): see the inference contract.

---

## 2. Surface syntax accepted (v0)

The parser accepts the following constructs. Every accepted construct lands in the AST defined at `src/formula/terms.rs:21–68`.

| Construct | Example | AST | Notes |
|---|---|---|---|
| Random block, correlated | `(1 + x \| g)` | `RandomTerm { terms, grouping: Single("g"), zerocorr: false }` | |
| Random block, zero-correlation | `(1 + x \|\| g)` | `RandomTerm { ..., zerocorr: true }` | `\|\|` is a single token, see `parser.rs:527–529` |
| Single grouping | `(... \| g)` | `GroupingFactor::Single("g")` | |
| Legacy interaction | `(... \| g1 & g2)` | `GroupingFactor::Interaction(...)` | `&` syntax preserved verbatim, `parser.rs:607–613` |
| Cell grouping | `(... \| g1:g2)` | `GroupingFactor::Cell(...)` | `parser.rs:614–620` |
| Nested grouping | `(... \| g1/g2/g3)` | expanded — see R1 | parser performs parse-time expansion |
| Crossed grouping | `(... \| g1*g2)` | expanded — see R2 | parser performs parse-time expansion |
| Implicit intercept (RE) | `(x \| g)` | terms = `[Column("x")]` | Note: this gets a random intercept added at materialization time, see §4 below |
| Explicit intercept (RE) | `(1 + x \| g)` | terms = `[Intercept, Column("x")]` | |
| Suppressed intercept (RE) | `(0 + x \| g)` | terms = `[NoIntercept, Column("x")]` | See R4 |
| Numeric basis column | `(x \| g)` | `Column("x")` | x must be a numeric column at materialization |
| Categorical basis column | `(cond \| g)` | `Column("cond")` | treatment-coded and cell-means materialization implemented; expanded-basis audit and ThetaMap traceability implemented |
| Interaction basis | `(x:y \| g)` | `Interaction(["x","y"])` | numeric/categorical treatment-coded and cell-means materialization implemented; expanded-basis audit and ThetaMap traceability implemented |

Rejected at parse time (v0):

- Empty grouping: `(x | )` → `FormulaError::EmptyGrouping`.
- Missing `|` in a parenthesized RE block → `FormulaError::MissingBar`.
- Unmatched parens, unknown tokens → `FormulaError::UnmatchedParen`, `FormulaError::UnexpectedToken`.

Accepted and supported at materialization/audit:

- Categorical basis columns — treatment-coded expansion, explicit
  frontend-supplied contrast bases, and the cell-means `0 + factor`
  parameterization materialize on the random side; design audit inspects the
  expanded optimizer columns and emits `FormulaCanonicalized` diagnostics when
  the semantic basis expands.
- Interaction basis columns — numeric/categorical treatment-coded and
  cell-means interaction expansions materialize on the random side; design
  audit and ThetaMap use the expanded optimizer columns.
- Fixed-effect interaction expansion now uses the same treatment-coded
  basis machinery in the common path.

---

## 3. Canonicalization rules (the deterministic IR)

After `parse_formula` (`src/formula/parser.rs:704–748`), the formula is normalized by the rules below before materialization. Each rule is deterministic, idempotent, and traceable: every canonicalization step records the input form so diagnostics can quote what the user wrote.

The canonical form is the AST that downstream stages (basis manager, ThetaMap, optimizer) consume. The semantic IR layer (`src/compiler/ir.rs`) is the natural home for the rules; today R1/R2 are applied at parse time, R3/R4 are represented in IR/materialization, and R8 is diagnosed at design-audit time. The remaining rules are defined here for v0 even where the code does not yet enforce them.

### R1 — Nesting expansion

`(b | a/c)` → `(b | a) + (b | a:c)`. Generalizes to arbitrary depth: `(b | a/c/d)` → `(b | a) + (b | a:c) + (b | a:c:d)`.

Status: applied at parse time by the grouping expansion helpers. The
`Nested(...)` AST variant exists for representation, but grouping factors are
normalized before downstream materialization.

Diagnostic: a structural (data-free) check at semantic-IR time should warn when the data do not in fact nest (i.e. when some level of `c` appears under more than one level of `a`). The semantic IR now emits `FormulaCanonicalized` (Info) on nesting expansion. A separate data-level `Unsupported` (Warning) for violated nesting assumptions remains future audit work.

### R2 — Crossing expansion

`(b | a*c)` → `(b | a) + (b | c) + (b | a:c)`. Generalizes to all subsets of size 1..n.

Status: applied at parse time by the grouping expansion helpers.

Diagnostic: this is rarely what users want. The semantic IR emits a
`CrossingLikelyUnintended` Info diagnostic listing the canonical expansion and
the two likely-intended alternatives (`(b|a)+(b|c)` for crossed mains;
`(b|a:c)` for cells only).

### R3 — Zero-correlation

`(b1 + b2 | g)` with the `||` separator stays as a single `RandomTerm { zerocorr: true, ... }` through canonicalization. The decomposition into independent blocks happens at materialization time, where `ReMat::zerocorr()` (`src/types/re_mat.rs:313–325`) zeroes off-diagonal Λ entries and reduces `inds` to the diagonal positions.

Reason: keeping `||` as a flag rather than rewriting to `(b1|g) + (0+b2|g)` preserves source syntax for diagnostics (R9) and keeps the term identifiable to ThetaMap as a single covariance block.

Status: conforms.

### R4 — Intercept policy is first-class

`InterceptPolicy ∈ { Included, Omitted }` is a property of every random term, set deterministically from the AST:

- `terms = [Intercept, ...]` → `Included`.
- `terms = [NoIntercept, ...]` (i.e. the user wrote `0 +` or `-1`) → `Omitted`.
- `terms = [Column(_), ...]` with no explicit `Intercept` or `NoIntercept` → `Included` (implicit). This matches the current materialization rule where a random term with slope columns and no explicit `0 +` still receives an intercept basis column.

`(0 + x | g)` is **not** rewritten to `(x | g)`. The omission is recorded.

Diagnostic: `RandomSlopeWithoutIntercept` (Info, already emitted at `src/compiler/ir.rs:222–232`) fires when `InterceptPolicy::Omitted` co-occurs with a non-empty random-slope basis on a grouping factor that has repeated observations and no other random term covers the intercept for that group.

Aligned with `mixed_model_compiler_inference_contract.md` §"Random-Effects Semantic IR" which defines `InterceptPolicy` as a first-class concept.

### R5 — Duplicate term detection

Exact-duplicate random terms `(b | g) + (b | g)` (same basis, same grouping, same `zerocorr`) are diagnosed with a `DuplicateRandomTerm` Warning listing both source spans. The contract target is to merge them into a single canonical/effective term once requested→effective model rewriting is implemented.

Status: diagnostic emitted by the semantic IR. Materialization still consumes
the parsed formula terms, so actual canonical/effective merging is tracked
under `bd-01KQ7WZQFWZQW1VVARWF6Y9ZYS`.

Equality test for "exact duplicate":

- Same `GroupingFactor` (variant and contents).
- Same `zerocorr` flag.
- Same multiset of `FixedTerm` entries in the basis (order does not matter; intercept/no-intercept tokens normalize to the canonical `InterceptPolicy`).

### R6 — Conflicting covariance on same basis

`(b | g)` and `(b || g)` with identical `b` and identical `g` is refused with `Unsupported` (Error) and the message that the user has requested both correlated and uncorrelated covariance for the same basis on the same grouping factor.

Status: diagnostic emitted by the semantic IR as an error. Turning that error
into a hard fit refusal belongs to the requested/effective model path
(`bd-01KQ7WZQFWZQW1VVARWF6Y9ZYS`).

### R7 — Same-grouping different-basis is preserved, not merged

`(1 + x | g) + (1 + y | g)` is **not** auto-merged into `(1 + x + y | g)`. The two terms are independent covariance blocks: variance components for `x` and `y` are estimated, and intercepts are correlated *within* each block but not *across* blocks. Merging would impose a joint 3x3 covariance the user did not request.

Diagnostic policy: no warning or suggestion is emitted by default. This form
is legal and common enough that a suggestion would be noisy. `explain_model()`,
`parameterization()`, and `audit()` should make the independent-block
covariance story visible; if the user wants joint covariance they must write
the joint form. This is also the correct behavior of lme4 and MixedModels.jl.

Status: conforms (parser preserves them; materialization builds two `ReMat`s).

### R8 — Fixed/random redundancy

`g + (1 | g)` (a fixed-effect categorical `g` with full treatment indicators *plus* a random intercept on the same `g`) is structurally redundant: the column space of the fixed-effect indicators contains the random-intercept column space, so the random-intercept variance is not separately identifiable.

Contract target behavior depends on mode (per the inference contract's mode taxonomy):

- `as_specified` → refuse with `NotIdentifiable` (Error) and message that fixed `g` indicators absorb the random intercept.
- `design_compiled` → drop the random intercept term, emit `FixedRandomRedundant` (Warning, code already declared at `src/compiler/diagnostics.rs:52`), record the change in the effective formula (§6).
- `exploratory`, `predictive` → same as `design_compiled` but without the implication that confirmatory inference is unaffected.

Status: emitted by the design audit as `FixedRandomRedundant` when fixed-effect
columns span a matching random-intercept term. The current implementation
diagnoses and records the problem; it does not yet apply the `design_compiled`
drop/refit behavior (`bd-01KQ7WZQFWZQW1VVARWF6Y9ZYS`).

### R9 — Source-syntax preservation

Every canonicalization records source syntax on the resulting term. The current
semantic IR stores canonical text plus `written` source text when the parser
rewrites the term, such as implicit random intercepts or grouping expansion.
Diagnostics quote the written form. The fitted artifact carries both forms.
This is the mechanism by which the print layer can show the user "you wrote X,
the canonical form is Y" (§6).

The semantic IR's `RandomTermIr` (per the inference contract) is the natural home for this field.

---

## 4. Random-effects basis (Z columns)

For a canonical random term `(b1 + b2 + … | g)` with `InterceptPolicy = P` and basis vars `b1..bk`, the Z block for grouping `g` has the following columns. Column count `s` (= `vsize` in `ReMat`) is determined entirely from the AST and the data schema, not from y.

### 4.1 Numeric basis column

A numeric basis variable contributes one Z column equal to the variable's vector. Current implementation: `src/model/linear.rs:2900–2908`.

### 4.2 Categorical basis column

A categorical basis variable contributes a column block. Two parameterizations, picked by intercept policy:

- **Treatment coding (default when `InterceptPolicy::Included` is in effect for this term)**: one Z column per non-reference level (`L − 1` columns). Reference level = first observed level unless the frontend supplied an explicit categorical contrast basis. Each default column is the indicator for that level.
- **Explicit contrast coding (when `CategoricalColumn.contrast` is present and treatment coding is in effect)**: one Z column per supplied contrast column. The row values are looked up from the supplied `levels x k` numeric matrix, column names come from `contrast_column_names`, and the same basis is used by fixed effects, random slopes, and interactions.
- **Cell-means coding (when the basis is `0 + factor` and `InterceptPolicy::Omitted`)**: one Z column per level (`L` columns). This is the "one variance per condition" parameterization commonly written `(0 + cond | g)` in lme4. This no-intercept formula semantics takes precedence even when an explicit contrast basis is available for the factor.

The design audit records categorical bases under `fixed_effects.contrast_bases`.
Each entry includes `variable`, `levels`, `contrast_matrix`, `column_names`,
`source`, `ordered`, and `explicit`. `source` uses the stable labels
`treatment`, `sum`, `helmert`, `polynomial`, `custom`, and `unknown`. When no
explicit contrast basis is supplied, Rust records the default treatment basis
with `explicit = false`.

The covariance interpretation:

- Treatment + intercept = baseline variance plus condition-specific deviations correlated with baseline (via the corresponding off-diagonals of Λ). The user is reading off "subjects vary in baseline; subjects also differ in the contrast for condition vs reference."
- Cell-means with no intercept = condition-specific variances and pairwise covariances. The user is reading off "each condition has its own subject-to-subject variance."

These are not interchangeable. The basis-manager doc (under inference contract) is responsible for warning that `corr(condA, condB)` from the cell-means parameterization is contrast-stable while `corr(intercept, condA)` from treatment coding is not.

Status: conforms for v0 basis construction. Treatment-coded categorical
expansion, explicit frontend-supplied contrast bases, and cell-means
`0 + factor` expansion materialize in the random-effect basis. Design audit
reports the expanded basis columns, records categorical contrast/defaulting
metadata, emits a `FormulaCanonicalized` diagnostic with semantic and expanded
basis payloads, and marks no-intercept categorical terms that used cell-means
coding despite an available explicit contrast basis. ThetaMap records both
`user_basis` and optimizer-facing `optimizer_basis` for round-tripping.

### 4.3 Interaction basis

For a basis `Interaction(vars)`:

- numeric × numeric → element-wise product, one Z column.
- numeric × factor (`L` levels) → the factor basis columns selected above
  (default treatment, explicit contrast, or cell-means under
  `InterceptPolicy::Omitted`), each multiplied by the numeric column.
- factor × factor (`L1 × L2` levels) → Cartesian products of the selected
  basis columns for each factor.
- arity > 2 → recursive: build the interaction of the first two and then interact with the next, dropping cells with no observations.

Status: partial. Numeric and categorical interaction basis columns now
materialize using treatment-coded expansion when an intercept is present and
cell-means expansion when the random term omits the intercept. Design audit
and ThetaMap operate over those expanded optimizer columns, with explicit
expansion diagnostics. Random-side empty-cell diagnostics remain open.
Composite key collision policy is decided in §5.2; implementation remains
listed in Appendix A.

### 4.4 Within-group variation requirement

A basis column with constant value within every group of `g` is not a usable random slope direction: the variance component for that direction is structurally unsupported by the design. The compiler emits `RandomSlopeUnsupported` at design-time. The contract target is to reduce the term's effective basis by one column before fitting; the current implementation diagnoses and reports the condition, while the requested→effective model rewrite remains tracked in `bd-01KQ7WZQFWZQW1VVARWF6Y9ZYS`.

Detection rule: every materialized random-basis column is grouped by `g`; the
maximum within-group standard deviation must exceed `min_within_group_sd =
1e-8` on the canonical/audit scale. The threshold is absolute after canonical
scaling, configurable through compiler policy, and recorded in the
reproducibility record. The earlier relative `1e-12 * max|b|` proposal is not
the v0 rule.

Status: current implementation uses the same `1e-8` threshold over expanded
optimizer columns, covering numeric slopes, categorical dummy/cell columns,
and interaction columns.

### 4.5 Numeric centering for `||`

The zero-correlation parameterization is not invariant to additive shifts of a continuous basis column: under `(1 + x | g)` (full covariance), recoding `x → x + c` is absorbed into the intercept-slope correlation; under `(1 + x || g)` (diagonal), the same recoding changes the model.

v0 policy: when a `RandomTerm` has `zerocorr = true` and a numeric basis
column, the independence assumption is interpreted at an explicit reference
value:

1. Use a declared reference from roles/control metadata if supplied.
2. Otherwise use the weighted model-frame mean after `subset`, `na.action`,
   and weights have been applied.

The centering transformation must be recorded and emitted as
`FormulaCanonicalized` (Info), including the reference value and whether it was
declared or compiler-chosen. Reported coefficients, predictions, contrasts,
and user-facing basis labels must be back-transformed to the user scale. The
engine must not silently use a hidden zero point for `||`.

Status: not currently performed.

---

## 5. Grouping factor materialization

### 5.1 Single grouping

`Single(g)` reuses the data's categorical encoding for `g` directly: `refs` and `levels` come from the `CategoricalColumn` (`src/model/linear.rs:2842–2849`). Level order is first-appearance, controlled by `CategoricalColumn::new` at `src/model/data.rs:37–58`.

### 5.2 Composite grouping

`Interaction([g1, g2, ...])` and `Cell([g1, g2, ...])` form composite levels
from the level labels of each component in declared order. The internal key
must be collision-free: v0 uses the ASCII record-separator byte `\x1E`
between escaped labels, while display strings continue to use user-facing
formula syntax (`a:b` or `a & b`). Both AST variants share the same
materialization today; the only difference is the `Display` form (`&` vs `:`).

Composite-level keys:

```
key(obs) = escape(labels(g1)[obs]) + "\x1E" + escape(labels(g2)[obs]) + ...
```

If a level label contains `\x1E`, it must be escaped before joining. This is
an implementation detail and should not appear in printed formulas, audit
labels, or R output.

Status: current Rust materialization still uses `_`, so collision-free
composite keys remain non-conforming until implementation catches up.

### 5.3 Empty cells

A composite cell that does not appear in the data does not get a level entry. `(1 | a:b)` over an unbalanced design materializes exactly the observed cells, no more. This is correct behavior — fitting a variance for a cell with zero observations is meaningless — but the audit must report it.

Diagnostic: `FixedEffectEmptyCell` (already declared at `src/compiler/diagnostics.rs:50`; reuse for the random side) at design-audit time, listing the missing combinations.

### 5.4 Level ordering

Today: first-appearance (`indexmap::IndexMap` at `src/model/linear.rs:2866`). This is permutation-unstable: two runs over the same data with rows in different order produce different level numbering, which leaks into `parmap`, into the optimizer's parameter ordering, and into output tables.

v0 contract: switch to **lexicographic** ordering of composite level labels at
canonicalization time, and record `level_order_source in { FirstAppearance,
Lexicographic, Declared }` on the materialized `ReMat`. `Lexicographic` is the
default for composite grouping keys; `Declared` is for users who supply
explicit level orderings via the data layer; `FirstAppearance` is retained for
compatibility only when explicitly requested.

Status: **non-conforming**. The change is small but visible in test fixtures
(level numbering would change). It should land with a fixture update and a
reproducibility-record entry, not behind another open policy question.

### 5.5 Levels in formula but not in data

A grouping variable named in the formula whose column does not exist in the data is already an error (`MixedModelError::InvalidArgument` at `src/model/linear.rs:2843–2848`). A grouping variable whose column exists but has zero observations is a degenerate case and is also an error in v0.

---

## 6. Requested → semantic → supported → fitted model state

Every fitted artifact can produce a computed `ModelStateSummary`. This keeps
the mutable artifact small while still giving R and other clients a stable
wire object for requested, semantic, supported, fitted, and changed model
views.

| Form | Source | Reproducible from |
|---|---|---|
| `requested_formula` | The string the user passed to `parse_formula`, plus its parsed AST. | The input string alone. |
| `semantic` | Formula compiled into semantic IR after rules R1–R7 of §3. No data-dependent reductions. | `requested_formula`. Pure function. |
| `supported` | Design-audited state plus policy recommendations/refusals. This may be `supported`, `advisory_changes`, or `refused`; v0 does not silently apply the rewrite. | `requested_formula` + data + compiler policy. |
| `fitted` | Fit/certificate state, including certificate-time boundary or reduced-rank reductions. | `requested_formula` + data + fit certificate. |

`changes()` returns the transition list. Each entry records `status`
(`diagnostic`, `recommended`, or `applied`), `trigger`, source/destination
stage, affected term, reason, replacement when available, inference
consequence, and diagnostics. The list is empty when nothing changed or was
recommended.

Print policy:

- Default: show `requested_formula` only.
- Show semantic/canonical state if it differs from `requested_formula`
  non-trivially (more than mere whitespace/parenthesization).
- Show supported/fitted model-state changes whenever `changes()` is non-empty.

Wire schema: the JSON shape is owned by the inference contract's R-as-client section, but at minimum each form serializes as `{ string: <display>, ast: <RandomTermIr[]> }`.

---

## 7. Interaction with the inference contract

This document binds the formula layer. The inference contract binds the rest. Boundaries:

**Estimability.** This doc owns *structural* estimability checks (rank-deficient X via pivoted QR at `src/types/fe_term.rs:73–136`, within-group variation §4.4, redundancy R8, empty cells §5.3). The inference contract owns *fitted* estimability — what the certificate says after optimization, including reduced-rank covariance, boundary parameters, and aliased contrast directions. The split is: structural means "decidable from (X, Z, formula) alone, no y, no fit"; fitted means "requires the optimizer's output."

**Under-modeling.** This doc emits structural under-modeling candidates:
`RepeatedUnitUnmodeled` when a grouping-like categorical variable has repeated
rows but no random-intercept dependence path covers it. The design audit records
the marginal-vs-cell distinction explicitly: `(1 | subject:item)` covers the
cell path only and does not cover subject-wide or item-wide dependence. The
missing-random-slope case for fixed effects that vary within a grouping unit is
still governed by the inference contract. The inference contract decides what
to *do* about these — refuse, suggest, regularize — per the mode taxonomy.

**Reduced-rank reporting.** This doc only flags pre-fit support (rank reductions from §4.4). Post-fit rePCA on the fitted Λ is owned by the inference contract.

**Information budget.** Formula canonicalization and basis expansion determine
the random-coefficient dimension `d`; covariance form determines the requested
covariance-parameter count. The design audit now reports the grouping-level
effective-n budget for each expanded random term: rows, grouping levels,
observations per level, levels per covariance parameter, and whether total
rows are misleading for covariance support. This is structural, pre-fit
information; the inference contract decides whether `design_compiled` reduces,
refuses, or fits with a warning.

**ThetaMap ownership.** The basis chosen here is the basis ThetaMap parameterizes. Any later basis rewrite (centering, reduced-rank reparameterization, family transition) must go through ThetaMap's round-trip contract; this doc forbids any later stage from modifying `cnames`, basis-column order, or `vsize` without recording the change as a `Reduction` per §6.

**Mode taxonomy.** Where R8 (redundancy) and similar rules behave differently per mode, this doc defers to the inference contract's modes (`as_specified`, `design_compiled`, `exploratory`, `predictive`). The default is `design_compiled`.

---

## 8. Diagnostics inventory

Codes used or proposed by the formula layer. Stage = `FormulaParsing` for surface-syntax errors and `SemanticIr` for canonicalization. Existing codes live at `src/compiler/diagnostics.rs:45–63`.

| Code | Severity | Stage | Status | Trigger |
|---|---|---|---|---|
| `FormulaCanonicalized` | Info | SemanticIr | declared | Emitted on representation-changing canonicalization steps (R1, R2, §4.5 centering, basis expansion). R7 does not emit by default. |
| `FormulaCanonicalizationUnsupported` | Warning | SemanticIr | emitted at `ir.rs:188` | A construct parses but cannot be canonicalized in v0 (e.g. `FixedTerm::Nested` inside a random basis). |
| `FixedEffectColumnMissing` | Error | FormulaParsing | declared | Variable named in the formula is not in the data. |
| `FixedEffectRankDeficient` | Warning | DesignAudit | declared | X loses columns under pivoted QR. |
| `FixedEffectEmptyCell` | Warning | DesignAudit | declared | Reuse for §5.3 missing composite cells on the random side. |
| `RandomSlopeWithoutIntercept` | Info | SemanticIr | emitted at `ir.rs:223` | R4 trigger when an `Omitted` policy plus a non-empty slope basis appears on a repeated grouping with no covering intercept. |
| `FixedRandomRedundant` | Warning | DesignAudit | emitted | R8 trigger; `g + (1 \| g)` style fixed/random column-space overlap. |
| `RepeatedUnitUnmodeled` | Warning | DesignAudit | emitted | Under-modeling: a repeated marginal or cell dependence path has no covering random-intercept kernel. |
| `RandomSlopeUnsupported` | Warning | DesignAudit | emitted | §4.4 numeric no-within-group-variation checks; expanded-basis support is still partial. |
| `CovarianceTooRich` | Warning | DesignAudit | emitted | Information-budget exceeded for the requested covariance. |
| `CovarianceReduced` | Info | Certification | emitted | Emitted for certificate-time reduced effective covariance rank. |
| `BoundaryParameter` | Info | Certification | emitted | Boundary/certificate-time covariance result. |
| `NotIdentifiable` | Error | SemanticIr | emitted at `ir.rs:207` | R8 in `as_specified` mode; or any other structural non-identifiability. |
| `Unsupported` | Warning/Error | varies | declared | Catch-all for v0 limitations (`||`-needs-centering with no reference, label collision in §5.2, etc.). |
| `DuplicateRandomTerm` | Warning | SemanticIr | emitted | R5 trigger. |
| `ConflictingCovariance` | Error | SemanticIr | emitted | R6 trigger. |
| `CrossingLikelyUnintended` | Info | SemanticIr | emitted | R2 fires; recommend `(b\|a)+(b\|c)` or `(b\|a:c)` instead. |

Two-step new-code policy: when a new code is added to `diagnostics.rs`, it must (a) appear in this table with a defined trigger and (b) have at least one test fixture in the audit module that exercises the trigger.

---

## 9. v0 vs vNext

v0 binds:

- The 9 canonicalization rules in §3.
- The basis-construction rules in §4 (including the categorical and interaction expansions that fix the silent-drop sites).
- The grouping-factor materialization rules in §5 including the lexicographic level-order default.
- The three-form formula reporting in §6.
- The diagnostic inventory in §8 including duplicate, conflicting-covariance,
  and crossing diagnostics.
- Grouping-level effective-n and information-budget reporting for expanded
  random-effect bases.

vNext (deferred):

- Smooth terms `s()`, `te()`, `bs()`, `ns()`. Requires a basis library and a separate IR variant.
- Offsets and `I(...)` literal-protection. Tokenizer extension; semantic interpretation.
- Roles declaration as part of the formula syntax (e.g. `re(subject, slope = x, intercept = FALSE)`). The inference contract floats `re()` and `vc()` constructors; v0 accepts only the lme4-style surface and infers role via the role-inference module.
- Cell-coded factor × factor random-effect bases beyond what §4.3 specifies. v0 supports them in principle but the test matrix is large.
- Multivariate response shared-θ formulas: see `multivariate_shared_theta.md`.
- Residual-structure formula syntax (`residual = ar1(time, subject)` etc.): out of scope here; cross-reference the inference contract.

---

## 10. Resolved v0 Decisions

These decisions close the former v0 open questions. Implementation gaps remain
tracked in Appendix A, but the product contract is no longer ambiguous.

1. **Within-group variation threshold.** Use `min_within_group_sd = 1e-8` on
   the canonical/audit scale after internal scaling. This is configurable and
   serialized in the reproducibility record. Do not use the older relative
   `1e-12 * max|b|` rule.
2. **Composite-level separator.** Use collision-free internal composite keys
   with escaped labels and `\x1E` as the join byte. Display labels remain
   formula-like and human-readable.
3. **Lexicographic ordering rollout.** Lexicographic ordering is the v0 default
   for composite grouping keys. First-appearance ordering is compatibility
   mode only and must be recorded when used.
4. **R6 strictness.** Keep `(b | g) + (b || g)` with identical basis/grouping
   as an error. Do not auto-rename or silently split the covariance story.
5. **R7 suggestion behavior.** Preserve same-grouping different-basis terms as
   independent blocks and stay silent by default. Explanation/parameterization
   views show the covariance consequence; no default warning is emitted.
6. **`||` centering reference.** Use a declared reference when supplied;
   otherwise use the weighted model-frame mean after filtering. Record and
   report the reference value and back-transform user-facing quantities.
7. **Random interaction coding.** Confirm treatment-coded interaction columns
   when the random term includes an intercept and cell-means interaction
   columns only when the random intercept is omitted. This keeps fixed and
   random bases aligned for default formulas.

These decisions are tracked in `bd-01KQ7WZF56ASNYE240MMG0GWWF`.

---

## Appendix A — Current code vs rule

Snapshot at the time of writing. Each row maps a v0 rule to where the code currently sits.

| Rule | Citation | Status |
|---|---|---|
| R1 nesting expansion | parser grouping expansion | conforms |
| R2 crossing expansion | parser grouping expansion | conforms; `CrossingLikelyUnintended` emitted |
| R3 zerocorr as flag | `src/types/re_mat.rs:313–325` | conforms |
| R4 InterceptPolicy first-class | semantic IR + materialization | partial — enum exists and common cases work; audit/print/source distinction needs hardening |
| R5 duplicate detection | semantic IR | diagnostic emitted; canonical/effective merge still pending |
| R6 conflicting covariance | semantic IR | diagnostic emitted; fit refusal still pending |
| R7 same-grouping different-basis preserved | materialization builds two `ReMat`s | conforms; intentionally no default diagnostic |
| R8 fixed/random redundancy | design audit | conforms for diagnosis; design-compiled reduction missing |
| R9 source-syntax preservation | `RandomTermIr::source_syntax` + `ModelStateSummary` | conforms for v0 reporting; parser-level written-vs-canonical syntax and requested/semantic/supported/fitted state are preserved |
| §4.1 numeric basis column | materialization | conforms |
| §4.2 categorical basis column | materialization + audit + ThetaMap | conforms for v0 basis construction; treatment-coded/cell-means materialization, expanded-basis audit, expansion diagnostics, and optimizer-basis ThetaMap traceability implemented |
| §4.3 interaction basis | materialization + audit + ThetaMap | partial — treatment-coded and cell-means numeric/categorical products materialize with expanded-basis audit/ThetaMap; random-side empty-cell diagnostics remain open |
| §4.4 within-group variation | design audit | partial — shared expanded-basis check exists; requested→effective reductions for unsupported basis directions remain open |
| §4.5 `||` centering | nowhere | **missing**: declared/weighted-mean reference rule decided; implementation remains |
| §5.1 single grouping | materialization | conforms |
| §5.2 collision-free composite grouping keys | materialization | **non-conforming**: implementation still joins with `_`; switch to escaped `\x1E` keys remains |
| §5.3 empty cells | implicit (cells absent from `level_map`) | conforms; **`FixedEffectEmptyCell` reuse for random side missing** |
| §5.4 lexicographic ordering | first-appearance order | **non-conforming** by v0 default; implementation remains |
| §6 requested→semantic→supported→fitted | `ModelStateSummary` + `changes()` | partial — state reporting, change records, and initial design-compiled covariance rewriting are implemented; broader effective-basis rewriting remains open |
| §8 new codes | diagnostics enum | duplicate/conflict/crossing diagnostics emitted; future codes still follow the table policy |

Each non-conforming or missing row is a candidate ticket. None are blocking the survey; they are blocking the contract.

Fixed-effect and random-effect basis expansion must stay aligned. When the
fixed-side interaction machinery changes, §4.3 should be checked so random
slopes and fixed contrasts still use compatible expanded columns.

---

## Appendix B — Mote Tracking

Formula-layer implementation work is tracked in local mote issues:

- `bd-01KQ7WYW472SRF7NPG9P1HDMSM` — formula canonicalization diagnostics,
  duplicate/conflicting covariance detection, crossing warnings, and written
  source-syntax preservation. The diagnostic/source-syntax slice is implemented;
  full canonical/effective rewriting is tracked separately.
- `bd-01KQ7WZ5ZTVQETY5PN3F5KHF02` — random-effect basis manager completion for
  categorical/cell-means bases, interactions, and expanded-basis audit. The
  materialization, expanded-basis audit, expansion diagnostics, and ThetaMap
  optimizer-basis traceability slices have landed.
- `bd-01KQ7WZK8G0EME3K6ZX539883D` — requested, semantic, supported, and fitted
  model-state reporting. The computed `ModelStateSummary` and `changes()` view
  have landed; initial design-compiled covariance rewriting has landed, while
  broader effective-basis rewriting remains separate.
- `bd-01KQ7WZF56ASNYE240MMG0GWWF` — completed v0 formula decisions:
  thresholds, separators, level ordering, R6/R7 behavior, `||` centering
  reference, and random interaction coding. Implementation gaps remain listed
  in Appendix A.

These issues intentionally cover the non-conforming/missing rows above
at implementation-slice granularity.

---

## Appendix C — Worked examples

Five formulas through the v0 pipeline. Each shows requested → canonical → effective, with the diagnostics that fire.

### C.1 Vanilla random intercept and slope

Input: `y ~ 1 + x + (1 + x | subj)`, with `subj` categorical, `x` numeric.

- `requested`: AST per `parser.rs`. `random_terms = [{ terms: [Intercept, Column("x")], grouping: Single("subj"), zerocorr: false }]`.
- `canonical`: identical (no rule fires).
- `effective`: identical (assuming `x` has within-`subj` variation).
- Z columns for `subj`: `[(Intercept), x]`, `vsize = 2`.
- Diagnostics: none.

### C.2 Categorical random slope

Input: `y ~ 1 + cond + (1 + cond | subj)`, with `cond` categorical with levels `{A, B, C}` (A first observed).

- `requested`: `random_terms = [{ terms: [Intercept, Column("cond")], grouping: Single("subj"), zerocorr: false }]`.
- `canonical`: same shape; basis is `[Intercept, Column("cond")]`.
- `effective`: under §4.2 treatment coding (because `InterceptPolicy::Included`), Z columns are `[(Intercept), cond:B, cond:C]`, `vsize = 3`. `FormulaCanonicalized` Info fires listing the column expansion.
- Diagnostics: `FormulaCanonicalized` (Info), with `semantic_basis` and
  `expanded_basis` payloads. The design audit and ThetaMap use the expanded
  optimizer basis while preserving the semantic user basis for explanation.

### C.3 Two-level nesting

Input: `y ~ 1 + (1 | a/b)`.

- `requested`: parser expands at parse time (R1) to `random_terms = [{ ..., grouping: Single("a") }, { ..., grouping: Cell(["a", "b"]) }]`.
- `canonical`: same as requested (the expansion is the canonical form). `FormulaCanonicalized` Info fires.
- `effective`: depends on the data. If some `b` level appears under more than one `a` level (the data don't actually nest), an `Unsupported` (Warning) fires. Otherwise effective = canonical.
- Z columns: two terms, each `vsize = 1` (intercept-only).

### C.4 Crossing expansion with the unintended-form warning

Input: `y ~ 1 + (1 | a*b)`.

- `requested`: parser expands at parse time (R2) to `random_terms = [{ ..., Single("a") }, { ..., Single("b") }, { ..., Cell(["a","b"]) }]`.
- `canonical`: same. `FormulaCanonicalized` Info fires.
- `effective`: same.
- Diagnostics: `CrossingLikelyUnintended` (Info) listing the canonical expansion and noting that the user may have meant `(1|a)+(1|b)` (crossed mains) or `(1|a:b)` (cells only).

### C.5 Fixed/random redundancy

Input: `y ~ subj + x + (1 | subj)`, with `subj` categorical (5 levels).

- `requested`: `fixed_terms = [Intercept, Column("subj"), Column("x")]`, `random_terms = [{ terms: [Intercept], grouping: Single("subj"), zerocorr: false }]`.
- `canonical`: identical.
- `effective`:
  - `as_specified` mode → refused with `NotIdentifiable` (Error). No fit produced.
  - `design_compiled` mode (default) → random term dropped, `effective_formula = y ~ subj + x`. `FixedRandomRedundant` (Warning) fires; the `Reduction` list contains `{ phase: DesignAudit, reason: FixedRandomRedundant, affected_term: "(1 | subj)", replacement: ∅ }`.
- Diagnostics: see above.
