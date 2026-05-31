# Audit 06 — Formula Parsing, Transforms, Pathology Detection, Datasets

Scope: src/formula/{parser,terms,transform,mod}.rs, src/pathology/{certificate,separation,spec,transforms,mod}.rs, src/datasets/mod.rs
Method: source trace + standalone probe harness linking the crate with `unstable-internals`. READ-ONLY (no source files modified; probe lived in /tmp).

## Verdict: REQUEST CHANGES

One CRITICAL reachable process abort and two HIGH `certify()` panics make this unwise to ship as-is. The bulk of the parser, the separation detectors, and the dataset loader are well hardened.

---

## CRITICAL

### C1. Unbounded recursion → stack overflow / process abort from a valid formula string
- Files: `src/formula/transform.rs` `TParser::parse_expr`/`parse_unary`/`parse_primary` (lines 577-679); `canonical_label`/`write_expr`/`inner_arg_label` (187-283); `eval_row` (308-332). Also `src/formula/parser.rs` `parse_term_expr` (669) and the `tokenize` transform branch (369-402).
- Trigger (confirmed, release build):
  - `y ~ I(((((( ... x ... )))))) + (1|g)` with ~5,000–20,000 nested parens.
  - `y ~ sqrt(sqrt(sqrt(... x ...))) + (1|g)` with ~20,000–100,000 nested calls.
- Behavior: `thread 'main' has overflowed its stack; fatal runtime error: stack overflow, aborting`. This is an **abort, not an unwind** — `std::panic::catch_unwind` does NOT catch it, so a host wrapper (the R client) that passes an attacker- or typo-supplied formula cannot defend against it; the whole process dies. Note `matching_paren` (parser.rs:135) is correctly iterative, so the lexer survives, but the recursive-descent transform parser and the recursive label/eval walkers do not.
- Why it matters: `parse_formula` is the primary untrusted-input boundary. The contract documents an actionable refusal for unsupported constructs; an unrecoverable abort on a deeply nested but otherwise legal expression violates that promise.
- Confidence: HIGH (reproduced deterministically).
- Fix: enforce an explicit nesting/expression-size budget. Cheapest: cap input length and/or paren-depth in `tokenize` before invoking the transform sub-parser (e.g. reject when `depth > 256` in the `'(' => depth += 1` loop of `matching_paren`, returning `FormulaError::Other`). Additionally thread a depth counter through `TParser::parse_expr`/`parse_primary` and bail with a `FormulaError` past a fixed limit, and convert `canonical_label`/`write_expr`/`eval_row` to bounded/iterative or guard them with the same budget. A depth limit in the low hundreds preserves every realistic formula.

---

## HIGH

### H1. `certify()` panics on a malformed Bernoulli spec instead of returning a classification
- File: `src/pathology/certificate.rs:501-517` (the `Family::Bernoulli` branch calling `detect_separation`) → `src/pathology/separation.rs:130` → `src/pathology/spec.rs:376-382` `assert!(spec.n_re_slopes <= n_predictors, ...)`.
- Trigger: `GeneratorSpec` with `family = Bernoulli`, `n_re_slopes = 2`, `fe_truth = [0.0]` (0 predictors). `certify(&spec)` panics: `spec 'bad' requests 2 random slopes but only 0 fixed-effect predictors exist`.
- Why it matters: `certify` is documented as "**Pure linear algebra. This function must not call any fitting engine, must not draw data, and must not depend on `seed`.**" (certificate.rs:241-245). For Bernoulli it *does* draw data via `detect_separation`/`generate`, and therefore inherits `generate`'s `assert!`s as panics in what callers are told is a pure, total classifier. A pathology-corpus harness that builds specs programmatically will abort instead of getting a `Certificate`.
- Confidence: HIGH (reproduced).
- Fix: make `detect_separation` defensive — it already early-returns `SeparationReport::empty()` when `generate` returns `Err` (separation.rs:130), but the failure here is a `panic!`, not an `Err`. Change `generate`'s spec-shape checks (spec.rs:365-382) from `assert!`/`assert_eq!` to returned `MixedModelError`, or have `detect_separation` validate `spec.n_re_slopes <= spec.n_fe_predictors()` and `re_cov_truth` dims up front and return an empty report on violation.

### H2. `certify()` panics "Matrix index out of bounds" on a re_cov/re_dim mismatch
- File: `src/pathology/certificate.rs:255` `if sigma[(i, i)].abs() < ZERO_VARIANCE_TOL` where `i` ranges over `q = spec.re_dim()` but `sigma = &spec.re_cov_truth` may be smaller.
- Trigger: `re_intercept = true`, `n_re_slopes = 1` (so `re_dim() = 2`) but `re_cov_truth` is 1×1. `certify(&spec)` panics before any data draw.
- Why it matters: same contract violation as H1 — the documented-total classifier panics on a malformed spec. The boundary-direction loop indexes `sigma[(i,i)]`/`sigma[(i,j)]` without checking `sigma.nrows() == q`.
- Confidence: HIGH (reproduced).
- Fix: validate `spec.re_cov_truth.nrows() == q && .ncols() == q` at the top of `certify` and return a `Certificate` flagged with a structural issue (or document `certify` as requiring a well-formed spec and provide a checked constructor). At minimum, clamp the boundary loops to `min(q, sigma.nrows())`.

---

## MEDIUM

### M1. Numeric literal accepted as a response (LHS) though rejected as a term (RHS)
- File: `src/formula/parser.rs:536-548` `parse_response` returns any `Ident` verbatim; the `name.parse::<f64>()` rejection in `parse_atom` (715-720) is RHS-only.
- Trigger: `2.5 ~ x + (1|g)` parses successfully with `response = "2.5"`. `1 ~ x` is rejected only incidentally (the lexer makes bare `1` a `One` token, not an `Ident`, so it hits `MissingResponse`), but `2.5`, `3e2`, `0.0` all sail through as a "column named 2.5".
- Why it matters: a numeric response is never meaningful; it surfaces much later as a confusing "column `2.5` not present" error (or a transform/materialize error) far from the real cause. Inconsistent with the deliberate `NumericLiteralTerm` guard on the RHS.
- Confidence: HIGH.
- Fix: apply the same `name.parse::<f64>().is_ok()` rejection in `parse_response` (emit `NumericLiteralTerm` or `MissingResponse`).

### M2. Structurally-empty random-effects block accepted: `(0 | g)`
- File: `src/formula/parser.rs:778-875` `parse_random_term`. With inner terms `[NoIntercept]`, the "no explicit intercept" guard at 848-853 does not fire (NoIntercept *is* an intercept directive), so the term is emitted with `terms = [NoIntercept]`, `grouping = Single("g")` — a random-effect block that contributes no random effect at all.
- Trigger: `y ~ (0 | g)` and `y ~ x + (0 | g)`.
- Why it matters: lme4 treats `(0 | g)` as an error / empty. Here it produces a `RandomTerm` with zero effective columns; downstream Λ_θ / ReMat construction (out of this audit's scope, model/linear.rs) must then special-case or will build a zero-dimensional block. Best rejected at the parser with `EmptyRandomTerms`.
- Confidence: MEDIUM (parser-confirmed; downstream impact not traced — out of scope).
- Fix: after the intercept-normalization step, if the random term has no `Column`/`Interaction`/`Intercept` (only `NoIntercept`), return `FormulaError::EmptyRandomTerms`.

---

## LOW

### L1. Mixed same-level `:` and `*` is refused, not supported
- File: `src/formula/parser.rs:669-704` `parse_term_expr` only loops on a single operator kind.
- Trigger: `y ~ a:b*c`, `y ~ a*b:c` → `UnexpectedToken("Star"/"Colon", _)`. lme4 accepts these. This is a clean refusal (no panic), so it is a documented-grammar limitation, not a safety bug. Recommend documenting the unsupported mixed-operator case in the module-level supported-syntax table, or implementing the lme4 expansion.
- Confidence: HIGH.

### L2. Duplicate CSV header silently selects the first matching column
- File: `src/datasets/mod.rs:408` `headers.iter().position(|h| h == &c.name)` returns the first match.
- Trigger: a `data.csv` with header `y,y,g` loads the first `y` and silently discards the second. Only reachable with a malformed bundled/external dataset; all shipped datasets are fine. A duplicate-header check in `read_csv_with_schema` would make this an explicit `DatasetError::Schema`.
- Confidence: HIGH (reproduced).

### L3. `fmt_lit` `v as i64` for canonical labels
- File: `src/formula/transform.rs:226-232`. Guarded by `v.abs() < 1e15`, so the `as i64` cast cannot saturate in the integral branch (1e15 < i64::MAX); values ≥ 1e15 fall to the float branch. `5e18` correctly renders via the integer path only because... actually it renders as `I(x+5000000000000000000)` — wait, 5e18 ≥ 1e15 so it takes the float `{v}` branch which prints `5000000000000000000`. No correctness bug found; noting that the 1e15 threshold is the load-bearing guard and is correct. No action required; documented here to record that the cast was checked and is safe.
- Confidence: HIGH (verified safe).

---

## Commendations

- `matching_paren` (parser.rs:135) is iterative and correctly skips backtick-quoted spans — the lexer does not blow the stack even though the transform sub-parser does.
- Transform refusals are genuinely actionable and whitelist-by-construct, not by surface syntax (transform.rs:363-421); stateful constructs (`poly`, `scale`, `ns`, `bs`, `factor`, multi-arg `log(x, base)`) are firmly rejected with precompute guidance. The stateless contract (pure pointwise, no fitting-time state) holds — verified `materialize` re-evaluates the closed form and the collision policy recomputes + verifies rather than trusting a pre-supplied column (terms.rs:144-201).
- Non-finite transform results (`I(1/x)` at x=0, `sqrt` of negatives) are caught as actionable `MixedModelError`s at `materialize_column`, not silent NaN propagation into Cholesky (transform.rs:337-349).
- Separation detection is solid: huge-margin designs (`x ∈ {±1000}`), intercept-only separation (all-ones response), and per-group all-zero/all-one scans all classify correctly; `detect_conditional_separation` safely handles empty and length-mismatched inputs (separation.rs:282-306). Extreme-prevalence Bernoulli specs correctly flip `expected_statuses` to `{NotIdentifiable, NotOptimized, ConvergedPenalised}` — no garbage fit reported as fine.
- `generate` is deterministic (same spec ⇒ identical draw), verified.
- Dataset loader is robust: NaN/Inf rejected via DataFrame construction, n_rows mismatch caught, BOM-prefixed headers tolerated, categorical values outside declared canonical levels rejected. `iter()` is sorted/deterministic and skips malformed `meta.toml` with a warning rather than panicking.
- Parser rejects (does not panic on) a large adversarial set: `(1|g) ~ x`, `log(x)` (no tilde), `y ~ )`, `y ~ x1 : : x2`, `y ~ (1 | )`, `y ~ (|g)`, 20-way `*` expansion, trailing operators, missing separators, removed-then-referenced transforms.

## Recommended gate before RC
- Must-fix: C1 (depth budget), H1 + H2 (`certify` totality / spec validation).
- Should-fix: M1, M2.
- Nice: L1 doc, L2 duplicate-header check.
