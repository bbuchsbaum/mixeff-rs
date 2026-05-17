# Formula-transform seam contract

Status: **decided** (supersedes the "How much of R's formula language to
pre-evaluate" open decision in `r_layer_proposal.md`).

## The question

R/lme4 formulas routinely contain in-formula transformations:
`log(reaction) ~ days + I(days^2) + scale(age) + poly(dose, 2) + (1 | subj)`.
Where should these be evaluated — in the Rust engine, or in the host-language
wrapper (R/Python)?

## The principle: stateless lives below the seam, stateful lives above it

Transformations split on exactly one axis, and that axis decides ownership:

| Class | Examples | Fitting-time state | Owner |
|---|---|---|---|
| **Stateless / pointwise** | `I(x^2)`, `I(x*z)`, `log(y)`, `exp`, `sqrt`, `abs`, `1/x` | none — closed form | **Rust engine** |
| **Stateful / basis** | `poly(x, 2)`, `scale(x)`, `ns(x, df=4)`, `bs(x)`, `cut(x, …)`, `factor(x)` | centering means, QR basis, knots, level set | **host wrapper** |

A stateless transform is a pure pointwise function `R -> R`. "The recipe" *is*
the closed-form expression, so applying it to new data is automatically
correct: prediction just re-evaluates the same expression. There is no
fitting-time parameter to capture, therefore no `predvars`, therefore no
second source of truth.

A stateful transform carries parameters learned from the training column
(`poly`'s orthonormal QR basis, `scale`'s centre/spread, spline knots,
`factor`'s level set). Prediction on new data **must** reuse the *training*
parameters. R captures this as `predvars` on the `terms` object. If the Rust
engine grew its own evaluator for these, there would be two implementations of
the recipe — the wrapper's and Rust's — that must agree bit-for-bit or
`predict()` silently diverges. That is the
[`mixed_model_compiler_inference_contract.md`](mixed_model_compiler_inference_contract.md)
"hidden model surgery" failure, instantiated. **One owner of the model frame =
one `predvars` = no divergence.** Therefore stateful transforms are *forbidden*
in the engine, not merely unimplemented.

This is also lme4's own architecture: lme4 implements no transforms; base R's
`model.frame`/`model.matrix`/`terms` expand everything before lme4 sees a
matrix. Pushing stateful transforms to the wrapper is not a compromise — it is
the design the mature ecosystems converged on.

## Engine obligations (the part below the seam)

The engine evaluates the **stateless subset** into synthetic numeric columns:

- `I(<arith>)` where `<arith>` is `+ - * / ^`, unary `-`, parentheses, numeric
  literals, and column references.
- Bare pointwise calls outside `I()`: `log`, `log2`, `log10`, `exp`, `sqrt`,
  `abs` (composable, e.g. `log(1 + x)` via `I(...)` inside, or `sqrt(x)`).
- Allowed on both sides: `log(reaction) ~ days + I(days^2) + (1 | subj)`.

Mechanics (see `formula/transform.rs`):

1. The parser recognises the subset and stores, per derived column, an
   expression AST plus a **canonical R-style label** (`I(days^2)`,
   `log(reaction)`). The label is the column name and the coefficient name —
   identical text to what R would print, so wrapper round-tripping is exact.
2. Transforms are **lowered into synthetic `DataFrame` columns at the data
   boundary, before design construction.** They are *not* threaded through
   `FixedTerm`/`RandomTerm`/`FixedDesign`/`predict`/`coef_names`. Every layer
   above the data boundary keeps seeing "a column by name"; the layered tower
   is unchanged.
3. `predict_new` re-runs the same stateless evaluator on `newdata`. Correct by
   construction — no recipe is stored because none exists.
4. Fitted values and residuals stay on the **transformed scale** (no automatic
   back-transform), matching lme4.

Anything outside the subset — `poly`, `scale`, `ns`, `bs`, `cut`, `factor`,
`center`, unknown functions, `log(x, base)` with a second argument — keeps the
existing **actionable refusal** in `parser.rs`: name the construct, say it is
out of scope for the engine, and point at precompute / the host wrapper.

## Host-wrapper obligations (the part above the seam)

A wrapper (e.g. the independent R package at arm's length from this crate;
likewise a future Python wrapper) that wants full R formula fidelity **must**:

1. Own the model frame for stateful transforms: evaluate `poly`/`scale`/`ns`/
   `bs`/`factor`/etc. itself (R: `model.matrix`; Python: formulaic/patsy).
2. Own prediction for those terms: when predicting on new data, re-expand with
   the **training** basis (R: `predvars`) and hand the engine plain numeric
   columns. The engine never learns a transform happened — **ownership
   model (a)**. The engine will not accept a "recipe" object; do not rely on
   ownership model (b).
3. Use the engine's stateless subset *or* pre-evaluate it — both produce
   identical results (that is the defining property of a clean seam). The
   engine will not know or care which the wrapper chose.

The engine does **not** cater to any specific wrapper and is not aware of one.
This contract is the only coupling; a wrapper relies on the stateless/stateful
line, not on engine internals.

## Why the line is drawn at stateless, not at `I()`

The naive split ("`I()` is fine, functions are not", or vice-versa) is wrong.
`I(poly_basis_lookup(x))` would be stateful; `sqrt(x)` is stateless. The only
property that makes a transform safe to own below the seam is being *pointwise
and parameter-free*. The contract is defined by that property, and the parser
enforces it by whitelist, not by surface syntax.

## Out of scope (unchanged)

Full `I()` / formula-level transformations *beyond this stateless subset*
remain post-v1, tracked under the umbrella mote issue. This document fixes the
boundary so the subset can ship without re-litigating ownership and so any
future wrapper is built against a frozen contract.
