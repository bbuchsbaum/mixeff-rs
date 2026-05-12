# PRD: Random Term Card and Pedagogical Diagnostic Codes

## State

Status: Proposed by downstream R wrapper (`mixeff`)
Owner: `mixeff-rs` Rust crate maintainers
Last updated: 2026-04-28
Related design notes:
- `docs/compiler_contract_v0_prd.md` (the v0 contract this PRD extends)
- `docs/r_layer_proposal.md` (the R-layer interface this contract serves)
- `docs/random_effects_formulas.md` (the formula syntax these cards explain)

Requesting document: `mixeff` PRD §9.5 — §9.7 (random-effects guidance
contract; §9.7 pedagogical diagnostic taxonomy; §9.6 random term card
schema). The mixeff repository lives at `/Users/bbuchsbaum/code/mixeff`
and tracks this work as sub-bead
`bd-01KQ9VFN9XM6J4TE4YC40WF576` ("Phase 1.B: upstream RandomTermCard +
DiagnosticCode pedagogical variants").

Mote tracking (upstream): umbrella `bd-01KQ9ZZXJPRKKP3SYWQ0RJWZXX`,
with seven sub-issues in dependency order — see
`Notes/Suggested First Issues` below. The umbrella records the FR3
Option B decision (§FR3 below). The upstream reconciliation issue
referenced from `compiler_contract_v0_prd.md`
(`bd-01KQ7WW46RJ6B71TAPZ0CK6KVJ`) remains the parent context.

This PRD asks for two co-required additive changes inside the existing
v0 compiler contract. Both changes are small in surface area, both are
purely additive, and both serve a single product invariant the R
wrapper needs to uphold:

> **R formats; Rust authors wording.** Single source of truth for
> per-block English and per-constraint reasoning across language
> bindings.

Without these changes, the R layer cannot satisfy its own §9.5 contract
without re-deriving semantics from formula text — which is precisely
the failure mode `mixeff` PRD §3 calls out as a non-goal ("no advice
creep"). With them, the R layer becomes a faithful formatter of
upstream-authored data.

## Description

`mixeff-rs` already authors per-term English in `CovarianceStory`
(`src/compiler/ir.rs:138`) and exposes an unstructured audit-report
text rendering through `ModelAuditReport::Display` (`src/compiler/report.rs:79`).
That is the right shape for human-readable summary text but is the
wrong shape for the R wrapper's three downstream verbs:

- `explain_model()` (mixeff PRD §9.5.2) — must paraphrase each
  random-effect term *block-by-block*, with an explicit "what is
  modeled / what is not modeled / nearby spellings" structure.
- `random_options(spec, group=)` (mixeff PRD §9.5.3) — must surface
  the requested-vs-nearby-syntax map with stable column meaning.
- `audit_design()` (mixeff PRD §9.5.4) — must distinguish three
  *kinds of help* (structural impossibility, low information budget,
  unmodeled-but-possible) so each is rendered in the appropriate
  tone register.

All three need structured per-block payloads with English authored in
Rust, and all three need a stable taxonomy for informational diagnostics
that today are conflated under `DiagnosticSeverity::Info`.

This PRD specifies:

1. Five additive `DiagnosticCode` variants for the §9.7 pedagogical
   taxonomy (`ScopeNote`, `SupportNote`, `SyntaxExpansion`,
   `CovarianceAssumption`, `StructuralRefusal`).
2. A new `RandomTermCard` struct that aggregates existing per-term
   compiler/audit data plus net-new per-block English and
   implied-constraint reasoning, exposed through `ModelAuditReport`
   (`src/compiler/report.rs`).

Both changes preserve the existing v0 compiler contract verbatim. No
existing public type changes shape; no fitting behavior changes.

## Problem

Three concrete drivers from the R wrapper's perspective:

- **Per-block English does not exist anywhere structured.** The two
  blocks in `(1 + x || g)` and `(1 | g) + (0 + x | g)` ("subjects may
  differ in average outcome" / "subjects may differ in their x slope")
  must produce identical structured output with `written` differing —
  the §9.5.2 "same model, different font" contract. `CovarianceStory`
  is single-string-per-term and does not capture this.

- **Pedagogical info diagnostics share a severity but lack a taxonomy.**
  When the R wrapper sees `DiagnosticSeverity::Info`, it cannot tell
  whether the message is "you wrote `(1 | a/b)` and we expanded it"
  (`SyntaxExpansion`) or "you wrote a within-group fixed effect with
  no random slope" (`ScopeNote`) without parsing the free-text
  message. The §9.5.4 three-kinds-of-help register depends on this
  distinction.

- **Implied constraints are not surfaced as data.** When `||` or split
  blocks fix a covariance to zero by syntax, the R wrapper must say
  *that* a covariance was fixed and *why*. Today this is implicit in
  `CovarianceForm::Diagonal` plus formula syntax; making it explicit
  removes a class of round-trip ambiguity.

## Goals

- Extend `DiagnosticCode` (`src/compiler/diagnostics.rs:60`) with five
  additive informational variants per the §9.7 taxonomy.
- Define `RandomTermCard` and supporting structs that aggregate
  existing compiler/audit per-term data into a single round-trippable
  artifact with per-block English authored upstream.
- Expose `random_term_cards: Vec<RandomTermCard>` on
  `ModelAuditReport` so downstream clients (R, future Python /
  JavaScript bindings) read cards through the same channel they read
  the existing audit-report sections.
- Version the new card schema independently
  (`mixedmodels.random_term_card`, v1) so future additions can
  evolve without breaking the audit-report schema.

## Non-Goals

- No behavioral change to the optimizer, fit pipeline, or
  `CompiledModelArtifact` numerical content.
- No new fit refusals beyond what the existing `RandomSlopeUnsupported`
  already rejects. `StructuralRefusal` is the same condition under a
  pedagogical name; in non-strict mode it reports the same fact at
  `Info` severity.
- No interactive `re_builder()` API. The R wrapper defers that to v2;
  upstream has no v0/v1 obligation here.
- No R-side concerns. Wrapper formatting, print methods, snapshot
  tests, and the R9 forbidden-string assertions all live in mixeff
  PRD §9.5–§11.
- No change to `attach_design_audit` or the cards' relationship to a
  fitted model. Cards are populated from already-attached audit data.

## Users

- The R wrapper `mixeff` (immediate consumer; this PRD's primary
  driver).
- Future non-R bindings (Python, JavaScript) that follow the same
  R9 wording-authority contract.
- Crate users programmatically inspecting random-term semantics
  pre-fit.

## Functional Requirements

### 1. DiagnosticCode pedagogical variants

Insertion point: `src/compiler/diagnostics.rs:60`. The current closed
enum ends at `Unsupported` (line 82). Append the following five
variants in this exact order; do not interleave with existing
variants (the discriminant is informally load-bearing in some
downstream cases that whitelist by string form):

```rust
pub enum DiagnosticCode {
    // ... existing variants unchanged ...
    Unsupported,
    // Pedagogical taxonomy added per random-term-card PRD.
    ScopeNote,
    SupportNote,
    SyntaxExpansion,
    CovarianceAssumption,
    StructuralRefusal,
}
```

The `#[serde(rename_all = "snake_case")]` derive at `diagnostics.rs:59`
produces stable string forms `scope_note`, `support_note`,
`syntax_expansion`, `covariance_assumption`, `structural_refusal`.

Add a comment immediately above the appended variants instructing
future contributors not to reorder them — the discriminant index is
informally load-bearing in some downstream code paths that whitelist
by string form. A pragmatic comment is enough; e.g.,
`// Pedagogical taxonomy — append-only ordering. Do not alphabetize.`

For each variant, the contract is:

#### `ScopeNote`

- **Triggers when**: a fixed effect varies within a grouping factor
  but no corresponding random slope was requested. e.g.,
  `y ~ time + (1 | subject)` where `time` varies within `subject`.
- **Severity**: `Info`. Never escalates. Per `mixeff` PRD §9.5.4
  ("unmodeled-but-possible") this is a quiet single-line scope note,
  not a warning.
- **Stage**: `DesignAudit` (the within-group variation check is what
  decides this).
- **Payload schema** (`payload: BTreeMap<String, serde_json::Value>`):
  ```json
  {
    "group": "subject",
    "fixed_effect": "time",
    "varies_within_group": true
  }
  ```
- **Suggested-action template**: `"`time` varies within `subject`, so a `subject`-level slope is structurally possible."`
- **Affected terms**: the relevant random-effect `term_id` (e.g.
  `"r0"`).

#### `SupportNote`

- **Triggers when**: a requested random-effect term is below the
  reliability floor for its covariance family but is not refused.
  Signalled today by `InformationBudgetStatus::WeaklySupported` on
  `RandomEffectInformationBudget` (`src/compiler/audit.rs:206`).
- **Severity**: `Info`. Factual, non-moralizing — must not say "you
  should" or "we recommend." Per mixeff PRD §9.5.4 ("low information
  budget"). The strings carried by the diagnostic must be parameter
  counts and grouping levels, not normative language.
- **Stage**: `DesignAudit`.
- **Payload schema**:
  ```json
  {
    "group": "subject",
    "covariance_family": "full",
    "requested_covariance_parameters": 3,
    "n_levels": 8,
    "policy_threshold": 15
  }
  ```
- **Suggested-action template**: `"The requested covariance structure is information-hungry relative to the observed grouping levels."`

#### `SyntaxExpansion`

- **Triggers when**: surface syntax expands to a longer canonical
  form. The two existing v0 cases:
  - `(1 | a/b)` → `(1 | a) + (1 | a:b)` (nested)
  - `(1 | a*b)` → `(1 | a) + (1 | b) + (1 | a:b)` (crossed-with-cell)
- **Severity**: `Info`.
- **Stage**: `SemanticIr`.
- **Payload schema**:
  ```json
  {
    "written": "(1 | a/b)",
    "canonical": "(1 | a) + (1 | a:b)",
    "expansion_kind": "nested"
  }
  ```
- **Suggested-action template**: `"`(1 | a/b)` expands to `(1 | a) + (1 | a:b)` — the canonical form."`

#### `CovarianceAssumption`

- **Triggers when**: a formula choice fixes a covariance to zero by
  syntax. Two existing v0 cases:
  - `||`: `(1 + x || g)` fixes cov(intercept, x slope) = 0.
  - Split blocks: `(1 | g) + (0 + x | g)` fixes the same covariance
    structurally (separate blocks).
- **Severity**: `Info`.
- **Stage**: `SemanticIr`.
- **Payload schema**:
  ```json
  {
    "group": "subject",
    "between": ["Intercept", "x"],
    "reason": "double_bar_syntax"
  }
  ```
  `reason` ∈ {`"double_bar_syntax"`, `"separate_random_effect_blocks"`}
- **Suggested-action template**: `"The intercept-slope covariance is fixed at zero by `||` syntax."`

#### `StructuralRefusal`

- **Triggers when**: a requested random slope cannot be estimated
  because the slope variable does not vary within group. This is the
  same condition as the existing `RandomSlopeUnsupported`
  (`diagnostics.rs:72`) but emitted *before* policy reduction so the
  R wrapper can route it to the §9.5.4 "structural impossibility"
  register independently.
- **Severity**: `Info` by default; under
  `CompilerPolicy::strict_mode = true` (a future flag, not in v0)
  upgrades to `Error` and aborts compilation. For v0 this PRD asks
  only for `Info` emission alongside the existing
  `RandomSlopeUnsupported`. Strict-mode escalation may land later.
- **Stage**: `DesignAudit`.
- **Payload schema**:
  ```json
  {
    "group": "subject",
    "slope": "condition",
    "reason": "slope_variable_does_not_vary_within_group"
  }
  ```
- **Suggested-action template**: `"`condition` does not vary within `subject`, so a `subject`-level `condition` slope cannot be estimated from this design."`

#### Co-existence with existing codes

`StructuralRefusal` and `RandomSlopeUnsupported` may both fire for
the same situation. That is intentional: the existing code is the
optimizer-facing category (the term is dropped); the new code is
the pedagogical category (the wrapper renders it as a "structural
impossibility"). The R wrapper deduplicates on `(code, affected_terms)`
when it formats; upstream emits both.

### 2. RandomTermCard schema

A new struct, defined in a new module file
`src/compiler/random_term_card.rs` and re-exported through
`src/compiler/mod.rs`. Fields:

```rust
pub const RANDOM_TERM_CARD_SCHEMA: &str = "mixedmodels.random_term_card";
pub const RANDOM_TERM_CARD_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomTermCard {
    pub schema_name: String,         // = RANDOM_TERM_CARD_SCHEMA
    pub schema_version: u32,         // = RANDOM_TERM_CARD_SCHEMA_VERSION
    pub term_id: String,
    pub original_fragment: String,
    pub canonical_fragment: String,
    pub group: GroupingFactorIr,
    pub blocks: Vec<RandomTermBlock>,
    pub implied_constraints: Vec<ImpliedConstraint>,
    pub design_support: DesignSupport,
    pub role_origin: RoleOrigin,
}
```

Field-by-field provenance and intent:

#### `term_id`

Source: `RandomTermIr.id` (`src/compiler/ir.rs:26`). Stable across
the artifact; lets clients join cards back to
`covariance_parameter_traces`, `theta_maps`, and per-term audit
sections by `term_id`.

#### `original_fragment` and `canonical_fragment`

Source: `SourceSyntax` (`src/compiler/ir.rs:113`). Direct mapping:

- `canonical_fragment = source_syntax.text` (always set).
- `original_fragment = source_syntax.written.unwrap_or(source_syntax.text)`.

Use `SourceSyntax::user_text()` (already defined at line 131) for the
read-side accessor.

The §9.5.7 "same model, different font" contract requires that two
formulas like `(1 + x || g)` and `(1 | g) + (0 + x | g)` produce
*identical* card output except for `original_fragment`. Specify this
as a contract clause: "Two structurally equivalent formulas produce
cards differing only in `original_fragment`; `canonical_fragment`,
`blocks`, `implied_constraints`, and `design_support` are byte-for-byte
identical."

#### `group`

Source: `RandomTermIr.group` (`src/compiler/ir.rs:27`). The existing
`GroupingFactorIr` enum (`ir.rs:39`) is sufficient; it already encodes
`Single { name }`, `Interaction { names }`, and `Cell { names }`.

#### `blocks: Vec<RandomTermBlock>`

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomTermBlock {
    pub basis: Vec<String>,                  // basis-column names
    pub intercept: bool,
    pub slopes: Vec<String>,                 // basis names with kind=Slope
    pub covariance: CovarianceForm,
    pub theta_parameters: usize,
    pub english: String,                     // single sentence; authored upstream
}
```

A *block* is a maximal correlated sub-group of a random-effect term.
For a single `(1 + x | g)` term there is one block (intercept + x
slope, full 2x2). For `(1 + x || g)` there are two blocks (one for
the intercept, one for `x`), each scalar covariance. For
`(1 | g) + (0 + x | g)` there are two `RandomTermIr` entries today,
each producing its own card with a single block; **see FR3 below**
for whether `||` should also produce two `RandomTermIr` entries or
keep its current single-term Diagonal representation.

`english` is one sentence describing what the block says about the
group. **Authored upstream** (FR4). The R wrapper renders this
verbatim with `cat()`; it never parses or splits it.

#### `implied_constraints: Vec<ImpliedConstraint>`

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImpliedConstraint {
    #[serde(rename = "type")]
    pub kind: ImpliedConstraintKind,
    pub between: Vec<String>,                // basis names
    pub reason: String,                      // English; authored upstream
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImpliedConstraintKind {
    ZeroCovariance,
    // future variants: ZeroVariance, EqualVariance, etc.
}
```

For the `(1 + x || g)` case, the card carries one
`ImpliedConstraint { kind: ZeroCovariance, between: ["Intercept", "x"], reason: "double-bar syntax fixes the intercept-slope covariance to zero" }`.
For the split-block case the same constraint with
`reason: "separate random-effect blocks fix the intercept-slope covariance to zero"`.

The two `reason` strings differ deliberately: that's how mixeff's
`explain_model()` distinguishes the two presentations of the same
underlying model when it generates the §9.5.2 split-block
explanation.

#### `design_support: DesignSupport`

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignSupport {
    pub group_levels: Option<usize>,
    pub min_rows_per_group: Option<usize>,
    pub median_rows_per_group: Option<usize>,
    pub within_group_variation: BTreeMap<String, WithinGroupVariation>,
    pub status: InformationBudgetStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WithinGroupVariation {
    Present,
    Absent,
    Constant,
    NotAssessed,
}
```

Provenance:

- `group_levels` = `RandomTermAudit.group.n_levels`
  (`src/compiler/audit.rs:194` → `:119`).
- `min_rows_per_group` = `RandomTermAudit.group.min_obs_per_level`.
- `median_rows_per_group`: **net-new computation**. `GroupingAudit`
  (`audit.rs:116`) exposes `min_obs_per_level` and
  `max_obs_per_level` but not median. **Recommended: add
  `median_obs_per_level: Option<usize>` to `GroupingAudit`** so the
  computation lives in one place and ships with the existing
  audit-construction pass. Falling back to on-the-fly card-time
  computation works but duplicates the iteration; the field-level
  approach is cleaner. The median is the most useful single number
  when groups are unbalanced; min and max alone bracket but do not
  summarize. This `GroupingAudit` extension is its own small step
  in the implementation order (see §Suggested First Issues).
- `within_group_variation`: aggregated from `BasisAudit.{name, kind, min_within_group_sd, max_within_group_sd, supported}`
  (`audit.rs:251`). Map basis name → variation classification:
  - `min_within_group_sd > min_within_group_sd_threshold` → `Present`
  - `max_within_group_sd ≤ min_within_group_sd_threshold` → `Absent`
  - both equal and finite → `Constant`
  - any field unavailable → `NotAssessed`
  Use the existing within-group-sd thresholds from
  `CompilerThresholds`.
- `status` = `RandomEffectInformationBudget.status`
  (`audit.rs:206` → `:214`). Direct passthrough of the existing
  `Sufficient | WeaklySupported | TooRich | NotAssessable` enum.

#### `role_origin: RoleOrigin`

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoleOrigin {
    pub declared_by_user: bool,
    pub observed_from_data: bool,
    pub role: GroupingRole,
}
```

**Highest-risk new wiring.** Today `RandomTermIr.role`
(`src/compiler/ir.rs:31`) is a single `GroupingRole` enum
(`ir.rs:100`) without provenance. The mixeff PRD §9.8 distinguishes
"role declared by the user via `roles()`" from "role inferred from
the data" (e.g., a variable that does not vary within group is
inferred as a between-group covariate).

Two implementation options for upstream to choose:

1. **Annotate on `RandomTermIr`**: add a sibling
   `role_origin: RoleOriginEnum` field. Touches every `RandomTermIr`
   construction site but keeps the data on the IR.
2. **Side-table on `SemanticModel`**: add a
   `role_origins: BTreeMap<String, RoleOrigin>` keyed by
   `RandomTermIr.id`. Purely additive; existing IR shape unchanged.

Implemented choice: Option 2. `SemanticModel.role_origins` stores
one `RoleOrigin` per `RandomTermIr.id`; v1 populates each entry as
`declared_by_user = false, observed_from_data = true` and mirrors
the resolved `RandomTermIr.role`.

mixeff's PRD §9.8 expects both `declared_by_user` and
`observed_from_data` booleans plus the resolved `GroupingRole`. The
v1 R wrapper will only set `declared_by_user = false,
observed_from_data = true` (since `roles()` is string-form-only and
not yet wired through to the FFI in mixeff Phase 1.F). Upstream may
ship the field as `declared_by_user = false` until the R wrapper
provides a way to declare; making the boolean available is what
matters now.

### 3. Block decomposition for `||` and split blocks

`(1 + x || g)` and `(1 | g) + (0 + x | g)` are the two surface forms
of the same model. The wrapper must render both as "two scalar
blocks with a fixed zero covariance," with the two surface forms
differing only in `original_fragment`.

Today:

- `(1 | g) + (0 + x | g)` produces **two** `RandomTermIr` entries.
- `(1 + x || g)` produces **one** `RandomTermIr` with
  `CovarianceForm::Diagonal`.

The R wrapper needs both surfaces to produce the same card output
(modulo `original_fragment`). Two implementation options for upstream:

#### Option A — Multi-card emission for single-term `Diagonal`

Construct the card layer above the IR. When the card layer sees a
`RandomTermIr` with `CovarianceForm::Diagonal` and `basis.len() > 1`,
it emits multiple `RandomTermBlock` entries inside a *single* card,
with one `ImpliedConstraint::ZeroCovariance` per off-diagonal pair.

Pros: purely additive on the IR side. Card constructor owns the
"split a Diagonal into independent blocks" logic. No existing IR
layout changes.

Cons: the card layer must know about `Diagonal`-as-multi-block
semantics; the IR's `covariance` field becomes per-term rather than
per-block, with a hidden contract.

#### Option B — Multi-IR-entry decomposition

Change `compile_formula_ir` so `(1 + x || g)` produces two
`RandomTermIr` entries (one for the intercept, one for the slope),
each with `CovarianceForm::Scalar`, plus a synthetic shared
`group` link.

Pros: every `RandomTermIr` corresponds to exactly one block. Card
construction is mechanical. The R wrapper sees two cards for `||`
and two cards for split-blocks — symmetric.

Cons: changes the IR layout for the `||` case; existing snapshot
fixtures must be updated. Adds a `block_group_id` or similar to
correlate IR entries that came from one written term.

#### Recommendation

**Option A is recommended** as additive and minimal-blast-radius.
The card-construction layer is new code; it can absorb the
"`Diagonal` term with multiple basis columns expands into multiple
blocks" rule without touching the IR. Snapshot fixtures for the
existing IR remain stable; the new fixtures live alongside the
new card.

If upstream prefers Option B for cleanliness, that is fine — the R
wrapper consumes cards regardless. The decision belongs to the
upstream maintainers.

### 4. Per-block English authorship

The `english` field on each `RandomTermBlock` and the `reason`
field on each `ImpliedConstraint` carry the user-visible wording
for the R wrapper's `explain_model()`.

The R9 contract requires Rust to author this wording. Reviewing
drift in one place (this crate) is the entire point.

The following table is a **non-binding starting set**. Upstream
maintainers should refine the wording to match the existing tone
of `CovarianceStory` (`src/compiler/ir.rs:138`) and the project's
documentation register. The canonical templates the wrapper expects
to see in mixeff Phase 1.C snapshot fixtures are these or close
descendants:

| Block shape | Suggested `english` |
| --- | --- |
| Intercept-only on `g` (`(1 \| g)`) | "`g` units may differ in average outcome." |
| Slope-only on `x` with `g` (`(0 + x \| g)`) | "`g` units may differ in their `x` slope." |
| Correlated full block on `g` (`(1 + x \| g)`) | "`g` units differ in baseline and `x` slope; the model estimates whether these are associated." |
| Diagonal block (one of two from `\|\|` or split) | "`g` units may differ in their `x` slope." (same string as slope-only — block-equivalence) |
| Cell grouping (`(1 \| a:b)`) | "Each combination of `a` and `b` may differ in average outcome." |
| Nested grouping (`(1 \| a/b)` after expansion) | One card per expanded term: "Each `a` may differ in average outcome." and "Each `b` within `a` may differ in average outcome." |
| Crossed (`(1 \| g) + (1 \| h)`) | One card each, identical wording form. |

| Implied constraint | Suggested `reason` |
| --- | --- |
| `ZeroCovariance` from `\|\|` syntax | "The double-bar syntax `\|\|` fixes the covariance between `Intercept` and `x` to zero." |
| `ZeroCovariance` from split blocks | "Separate random-effect blocks fix the covariance between `Intercept` and `x` to zero." |

A consequence to flag: the `english` for the `||` block and the
matching split-block card are intentionally **the same string**
because they describe the same model. The two presentations differ
only in `original_fragment` and the `reason` of the
`ImpliedConstraint`. The R wrapper relies on this for its
"same model, different font" rendering.

### 5. Schema versioning

Add to `src/compiler/random_term_card.rs`:

```rust
pub const RANDOM_TERM_CARD_SCHEMA: &str = "mixedmodels.random_term_card";
pub const RANDOM_TERM_CARD_SCHEMA_VERSION: u32 = 1;
```

Mirror the pattern at `src/compiler/ir.rs:9-10`:
- `pub const SEMANTIC_MODEL_SCHEMA = "mixedmodels.semantic_model";`
- `pub const SEMANTIC_MODEL_SCHEMA_VERSION: u32 = 1;`

JSON round-trip test required, mirroring
`semantic_model_round_trips_json` (or whichever test the existing
`SemanticModel` uses). One test that:

1. Constructs a `RandomTermCard` programmatically.
2. `serde_json::to_string` then `serde_json::from_str`.
3. `assert_eq!(decoded, original)`.

### 6. Integration points

#### `attach_design_audit` — no changes

`src/compiler/artifact.rs:563`. The audit data the cards aggregate is
already populated; no new audit pass is needed.

#### `ModelAuditReport::from_artifact` — additive

`src/compiler/report.rs:50`. Add a new section *and* a new top-level
field:

```rust
pub struct ModelAuditReport {
    pub schema_name: String,
    pub schema_version: u32,
    pub requested_formula: String,
    pub sections: Vec<AuditReportSection>,
    pub random_term_cards: Vec<RandomTermCard>,    // NEW
    pub diagnostics: Vec<Diagnostic>,
}
```

Bumping `MODEL_AUDIT_REPORT_SCHEMA_VERSION` from `1` to `2` (a
breaking-but-additive change for clients that strict-validate)
is the upstream's call. **Recommendation: bump to v2.** Better to
keep the schema honest than to add a field silently.

Downstream commitment: the `mixeff` wrapper's R-side schema
negotiator currently accepts only v1 (registered in
`src/rust/src/lib.rs`'s `KNOWN_SCHEMAS` table as
`("mixedmodels.compiled_model_artifact", "1")`). When this PRD
lands upstream and the audit-report schema bumps to v2, mixeff
will follow with a one-line registry update plus a regenerated
test snapshot. The wrapper does not yet hard-validate the
audit-report schema (it consumes the rendered text via
`mm_audit_report_text`), so the negotiator change is purely
proactive — not a coordination blocker. mixeff Phase 1.B will
land the matching wrapper update in the same change set as the
upstream pickup.

#### `CompiledModelArtifact` — optional convenience

The artifact already carries everything needed to construct cards.
A thin wrapper method on `CompiledModelArtifact`:

```rust
impl CompiledModelArtifact {
    pub fn random_term_cards(&self) -> Vec<RandomTermCard> { ... }
}
```

would let clients access cards without going through the audit
report. Optional; not required by the R wrapper, which reads the
audit report.

#### Snapshot fixtures will perturb

`tests/golden/` (or wherever the worked-example fixtures live) will
diff once `random_term_cards` is added to `ModelAuditReport`. This
is the largest cross-cutting change in this PRD, but it is purely
additive — no field is removed or renamed. Update fixtures
alongside the implementation.

## Acceptance Criteria

Legend (matches `compiler_contract_v0_prd.md`):

- `[x]` implemented and covered by tests/fixtures.
- `[~]` partially implemented; follow-up tracked below.
- `[ ]` not implemented or intentionally deferred.

### DiagnosticCode Acceptance

- [x] `DiagnosticCode` enum carries the five new variants in the
      order specified in §FR1.
- [x] Each variant serializes to its `snake_case` form via
      `serde(rename_all = "snake_case")`.
- [x] Each variant has at least one fixture emitting it under
      conditions that match the §FR1 trigger description.
- [x] `ScopeNote` fires for `y ~ time + (1 | subject)` when `time`
      varies within `subject`.
- [x] `SupportNote` fires when
      `RandomEffectInformationBudget.status == WeaklySupported`.
- [x] `SyntaxExpansion` fires for `(1 | a/b)` and `(1 | a*b)`,
      with `payload.expansion_kind` set to `"nested"` and
      `"crossed_with_cell"` respectively.
- [x] `CovarianceAssumption` fires for `(1 + x || g)` and
      `(1 | g) + (0 + x | g)` with distinct `payload.reason` values
      (`"double_bar_syntax"` and `"separate_random_effect_blocks"`).
- [x] `StructuralRefusal` fires alongside the existing
      `RandomSlopeUnsupported` when a slope variable does not vary
      within group; both diagnostics carry the same `affected_terms`.

### RandomTermCard Acceptance

- [x] `RandomTermCard` is defined in
      `src/compiler/random_term_card.rs` and re-exported via
      `src/compiler/mod.rs`.
- [x] `RANDOM_TERM_CARD_SCHEMA = "mixedmodels.random_term_card"`,
      `RANDOM_TERM_CARD_SCHEMA_VERSION = 1`.
- [x] Cards round-trip cleanly through `serde_json` (one
      programmatic test).
- [x] `ModelAuditReport.random_term_cards` is populated for every
      compiled artifact post-`attach_design_audit`. Empty `Vec` is
      acceptable for fixed-effects-only formulas.
- [x] `MODEL_AUDIT_REPORT_SCHEMA_VERSION` bumped to `2` and a JSON
      snapshot test pins the new shape.
- [x] `term_id` matches `RandomTermIr.id` for joinability.
- [x] `original_fragment == canonical_fragment` whenever
      `SourceSyntax.written` is `None`; otherwise
      `original_fragment == source_syntax.written.unwrap()`.
- [x] `(1 + x || g)` and `(1 | g) + (0 + x | g)` produce cards with
      identical `canonical_fragment`, `blocks`, `implied_constraints`,
      and `design_support` — they differ **only** in
      `original_fragment` and the constraint's `reason` string.
      This is the §9.5.2 "same model, different font" contract.
- [x] `design_support.status` equals
      `RandomEffectInformationBudget.status` verbatim for the same
      term.
- [x] Each `RandomTermBlock.english` is non-empty, non-NA, and a
      single-sentence string.
- [x] Each report-level `CrossCardConstraint.reason` is non-empty
      and distinguishes `||` from split-block sources under the
      accepted Option B design.

### Block Decomposition Acceptance (§FR3)

- [x] The implementing maintainer records the chosen option (A or B)
      and its rationale in this PRD's `Notes/Architectural Notes`
      section as part of the implementation PR. Option B is locked
      in the upstream mote: `||` emits per-basis `RandomTermIr`
      entries with shared `block_group`, and cross-card constraints
      live on `ModelAuditReport.cross_card_constraints`.
- [x] `(1 + x | g)` produces one card with one block (full
      covariance).
- [x] `(1 + x || g)` produces two cards (one per split
      `RandomTermIr`), each with one scalar block, plus a
      report-level `CrossCardConstraint::ZeroCovariance`.
- [x] `(1 | g) + (0 + x | g)` produces two cards (one per
      `RandomTermIr`), each with one scalar block, plus a
      report-level `CrossCardConstraint::ZeroCovariance`.
- [x] `(1 | a/b)` produces two cards (one per expanded term) and a
      `SyntaxExpansion` diagnostic with the original written form.

### Worked-Example Acceptance

- [x] *(existing)* The five worked-example fixtures pinned in
      `compiler_contract_v0_prd.md` lines 913–922 continue to pass.
- [x] A new sleepstudy fixture pins the full `RandomTermCard` JSON
      for `Reaction ~ Days + (Days | Subject)`. Field-by-field
      identity check; English wording locked.
- [x] A new `(1 + x || g)`/split-block fixture pins the
      "same model, different font" structural identity contract.

### Wording Acceptance

- [x] Every `english` and `reason` string is authored upstream and
      survives review. Wording table in §FR4 is treated as
      starting-point only.
- [x] No `english` or `reason` string contains the §9.5.5
      forbidden phrases — `"suggested starting model"`,
      `"we recommend"`, `"you should"`, `"try ... instead"`,
      `"drop the random slope"`. The R wrapper enforces this with
      assertions on its side too, but upstream's wording is the
      first line of defense.

### Schema Versioning Acceptance

- [x] The `RandomTermCardSchema` round-trip test mirrors
      `SemanticModel`'s round-trip test pattern.
- [x] `MODEL_AUDIT_REPORT_SCHEMA_VERSION` bump is reflected in the
      audit-report round-trip fixture.

## Worked Example

The fixture sleepstudy: `Reaction ~ Days + (Days | Subject)`. Single
random-effect term with full 2×2 covariance, 18 subjects, 10
observations each, complete balance.

Expected `RandomTermCard` JSON (English wording is *suggested*; the
maintainer's wording wins):

```json
{
  "schema_name": "mixedmodels.random_term_card",
  "schema_version": 1,
  "term_id": "r0",
  "original_fragment": "(Days | Subject)",
  "canonical_fragment": "(1 + Days | Subject)",
  "group": { "single": { "name": "Subject" } },
  "blocks": [
    {
      "basis": ["Intercept", "Days"],
      "intercept": true,
      "slopes": ["Days"],
      "covariance": "full",
      "theta_parameters": 3,
      "english": "Subject units differ in baseline and Days slope; the model estimates whether these are associated."
    }
  ],
  "implied_constraints": [],
  "design_support": {
    "group_levels": 18,
    "min_rows_per_group": 10,
    "median_rows_per_group": 10,
    "within_group_variation": {
      "Intercept": "constant",
      "Days": "present"
    },
    "status": "sufficient"
  },
  "role_origin": {
    "declared_by_user": false,
    "observed_from_data": true,
    "role": "sampled_unit"
  }
}
```

For comparison, the same data fit with `(1 + Days || Subject)`:

```json
{
  "schema_name": "mixedmodels.random_term_card",
  "schema_version": 1,
  "term_id": "r0",
  "original_fragment": "(1 + Days || Subject)",
  "canonical_fragment": "(1 + Days || Subject)",
  "group": { "single": { "name": "Subject" } },
  "blocks": [
    {
      "basis": ["Intercept"],
      "intercept": true,
      "slopes": [],
      "covariance": "scalar",
      "theta_parameters": 1,
      "english": "Subject units may differ in average outcome."
    },
    {
      "basis": ["Days"],
      "intercept": false,
      "slopes": ["Days"],
      "covariance": "scalar",
      "theta_parameters": 1,
      "english": "Subject units may differ in their Days slope."
    }
  ],
  "implied_constraints": [
    {
      "type": "zero_covariance",
      "between": ["Intercept", "Days"],
      "reason": "The double-bar syntax `||` fixes the covariance between `Intercept` and `Days` to zero."
    }
  ],
  "design_support": { /* identical to the previous example */ },
  "role_origin": { /* identical */ }
}
```

And for `(1 | Subject) + (0 + Days | Subject)` — the split-block
form — the fixture must produce **two** cards, the first matching
the `||` form's first block (modulo `original_fragment`), the
second matching the second. The implied-zero-covariance constraint
is conveyed by the *pair* of cards; if `ImpliedConstraint` is to
be carried inside a single card, the upstream choice between
Option A and Option B (§FR3) determines whether one card carries
both blocks or two cards each carry one.

## Notes

### Architectural Notes

- The `english` field is single-source-of-truth wording. When the R
  wrapper renders an audit report, it calls
  `mm_audit_report_text(artifact_json)` which delegates to
  `ModelAuditReport::Display::fmt`. That `Display` impl, after this
  PRD lands, must include the `random_term_cards` section. The R
  layer never reformats `english`.
- Block decomposition (FR3) uses Option B. `||` forms produce
  per-basis `RandomTermIr` entries with a shared `block_group`,
  and report-level `cross_card_constraints` record the zero
  covariance. This gives `||` and split-block forms structurally
  identical card lists modulo `original_fragment` and the
  constraint `reason` string.
- `RoleOrigin` is the highest-risk new wiring. v1 needs only
  `observed_from_data = true, declared_by_user = false` because
  mixeff's Phase 1.F (`roles()` v1 string-form) does not flow back
  into the FFI. The `SemanticModel.role_origins` side-table is in
  place so that future declarations can flow without reshaping
  `RandomTermIr`.

### Suggested First Issues

In dependency order:

1. Extend `DiagnosticCode` with the five new variants (additive,
   low-risk; one PR). Independent of everything else here.
2. Add `median_obs_per_level: Option<usize>` to `GroupingAudit`
   (`audit.rs:116`) and populate it in
   `audit_design`/`attach_design_audit`. Trivial, self-contained,
   prerequisite to FR2's `design_support.median_rows_per_group`.
3. Resolve §FR3 Option A vs Option B and record the decision in
   this PRD's `Notes/Architectural Notes` section. The decision
   determines the cross-card-constraint shape (single-card
   `implied_constraints` only, vs an `ModelAuditReport`-level
   `cross_card_constraints` field).
4. Add `RandomTermCard` struct + serde + schema constants in a new
   `src/compiler/random_term_card.rs`.
5. Wire `random_term_cards: Vec<RandomTermCard>` into
   `ModelAuditReport::from_artifact` and bump
   `MODEL_AUDIT_REPORT_SCHEMA_VERSION` to 2.
6. Author per-block `english` strings and per-constraint `reason`
   strings (FR4 wording table is starting-point only).
7. Pin one new worked-example fixture (sleepstudy) and update the
   audit-report round-trip JSON snapshot.

Each step is independently shippable. Steps 1–2 are unblocked now.
Step 3 unblocks 4–7.

### Implementation Notes

- Stable-ordering invariants: `blocks` ordered by first-basis-column
  appearance; `implied_constraints` ordered lexicographically by
  `between` (matching the rest of the project's diagnostic ordering
  convention).
- `SemanticModel.role_origins` is keyed by `RandomTermIr.id`.
  `RandomTermCard.role_origin` is read from that side-table and
  falls back to `RandomTermIr.role` only when deserializing older
  artifacts that lack the additive field.
- The `english` strings should not interpolate variable names
  through `format!` directly; use a small helper that escapes
  identifiers consistently with the existing
  `CovarianceStory::summary` formatting at `src/compiler/ir.rs:138+`.
  This keeps backtick conventions in step.

### Explicitly Deferred

- Strict-mode escalation of `StructuralRefusal` to `Error`. The v0
  contract today emits `RandomSlopeUnsupported` at `Warning` and
  the optimizer drops the term; a future strict-mode flag may
  abort compilation. Not in v1 scope.
- An interactive `re_builder()` API. mixeff defers this to v2 per
  PRD §9.5.7; upstream has no v0 obligation.
- Cross-card `ImpliedConstraint` representation for the
  split-block case. If Option A is chosen, all constraints live
  inside a single card; if Option B is chosen, the upstream may
  need a `cross_card_constraints: Vec<ImpliedConstraint>` field
  on `ModelAuditReport`. Decide alongside FR3.
- Localization. All `english` and `reason` strings are in English.
  i18n is a future concern; the schema is forward-compatible
  (strings can be replaced with localized variants without
  changing wire shape).

## Out of Scope

- Implementation details of how the R wrapper renders cards. See
  mixeff PRD §9.5.2 (`explain_model()`), §9.5.3 (`random_options()`),
  §9.5.4 ("three kinds of help"), §9.6 (this card schema's R-side
  consumer), §9.7 (this PRD's diagnostic taxonomy consumer).
- The `nlopt` feature-gate PR, tracked separately as mixeff bead
  `bd-01KQ906S43Q5T2WD7GRZDAK7VZ` with a draft at
  `mixeff/planning/upstream-nlopt-issue.md`.
- mixeff's other Phase 1 sub-beads (1.D — `random_options()` /
  `compare_covariance()`; 1.E — `lmm()` + extractors; 1.F —
  `audit/changes/diagnostics/parameterization/roles/as_json`; 1.G —
  vignettes). These all consume the artifacts this PRD specifies
  but live in the R repository.

## Cross-references

- mixeff PRD: `/Users/bbuchsbaum/code/mixeff/planning/PRD.md`
  - §3 (non-goals — including the no-recommendation stance)
  - §5.2 item 4 (extended `DiagnosticCode` enum — this PRD §FR1)
  - §5.2 item 5 (random term card per random-effect term — this
    PRD §FR2)
  - §9.5 (random-effects guidance contract — the R-layer use of
    these cards)
  - §9.5.2 (split-block explanation — this PRD §FR3)
  - §9.5.4 (three kinds of help — uses §FR1's diagnostic taxonomy)
  - §9.5.5 (forbidden phrases — wording acceptance)
  - §9.5.6 (singularity rendering — covered by `design_support`
    and the Reduced-Rank case)
  - §9.5.7 (eight-pattern syntax coverage list — upstream test
    corpus)
  - §9.6 (random term card schema — this PRD §FR2)
  - §9.7 (pedagogical diagnostic taxonomy — this PRD §FR1)
  - §9.8 (`roles()` — declared and observed origin — this PRD
    §FR2 `role_origin`)
  - §11 (test strategy — wording assertions and round-trip tests)
  - §12 R9 ("advice creep" mitigation — wording authored upstream)
- This crate: `compiler_contract_v0_prd.md` (the v0 contract this
  PRD extends additively), `r_layer_proposal.md` (the R-layer the
  cards serve), `random_effects_formulas.md` (the surface syntax
  the cards explain).
