# Multivariate Y With Shared Design and Shared Theta

> **Status: vNext (deferred).** Multivariate response is explicitly out of
> scope for v0 per the [v0 PRD](compiler_contract_v0_prd.md) non-goals.
> Formula handling for multivariate response inherits the v0 random-effects
> formula contract in
> [`random_effects_formulas.md`](random_effects_formulas.md); the only
> addition is the response-side split described below. ThetaMap, design
> audit, KKT certificate, and reproducibility requirements from the v0 PRD
> apply unchanged when this work is promoted from vNext to a binding
> contract. Local mote tracking: `bd-01KQ7X12J1TGDA6MFM3E5KQDJE`.

This note describes the simplest multivariate extension that fits the current
Rust architecture:

- `Y` is an `n x q` response matrix
- every response column shares the same fixed-effects design `X`
- every response column shares the same random-effects structure `Z`
- all response columns share the same relative covariance parameters `theta`
- each response column has its own fixed effects `beta_j`
- each response column has its own profiled residual scale `sigma_j`

This is not yet a full multivariate mixed model with cross-outcome residual
covariance. It is a batched shared-structure LMM.

## Why the current code cannot express this directly

The current scalar LMM pipeline bakes a single response vector into the
structural matrix:

- `Formula.response` is one `String`
- `LinearMixedModel::new` extracts one numeric column into a `DVector`
- `FeMat` stores `[X | y]`
- the blocked system stores `[Z X y]' [Z X y]`
- `MixedModelFit` exposes response, fitted values, and residuals as vectors

That architecture is efficient for one outcome, but it couples the shared
system matrix with the response-specific right-hand side. A multivariate
extension should split those two concerns.

## Phase-1 statistical model

For response columns `y_1, ..., y_q`:

`y_j = X beta_j + Z b_j + e_j`

with

- `b_j ~ N(0, sigma_j^2 * Lambda_theta * Lambda_theta')`
- `e_j ~ N(0, sigma_j^2 * I_n)`

and conditional independence across `j` given the shared `theta`.

This gives a shared profiled objective:

`f(theta) = sum_j f_j(theta)`

where each `f_j(theta)` is the ordinary scalar-response profiled deviance or
REML criterion.

## Shared factorization

For a fixed `theta`, define the shared penalized least-squares system

`A(theta) = [[Lambda' Z' Z Lambda + I, Lambda' Z' X], [X' Z Lambda, X' X]]`

This matrix depends on `X`, `Z`, and `theta`, but not on any particular
response column.

Let `L(theta)` be the lower Cholesky factor of `A(theta)`.

For each response column `y_j`, define the right-hand side

`b_j(theta) = [Lambda' Z' y_j, X' y_j]`

and the scalar

`c_j = y_j' y_j`

Then

`pwrss_j(theta) = c_j - b_j(theta)' A(theta)^(-1) b_j(theta)`

Equivalently, if `W = L^(-1) B` with `B = [b_1 ... b_q]`, then

`diag(B' A^(-1) B) = diag(W' W)`

so each column's profiled residual sum of squares is

`pwrss_j(theta) = c_j - sum_i W[i, j]^2`

This is the key batching opportunity:

- factor `A(theta)` once
- solve one triangular system with `q` right-hand sides
- get all `pwrss_j(theta)` values together

If fixed effects are needed, solve `L' U = W`; the lower `p` rows of `U`
contain the batched fixed-effects solutions.

## Objective assembly

Let

- `logdet_zz(theta)` be the current shared random-effects determinant term
- `logdet_xx(theta)` be the current shared REML fixed-effects determinant term

Then for independent response columns with separate profiled `sigma_j`:

ML:

`f(theta) = q * logdet_zz(theta) + sum_j n * (1 + log(2*pi*pwrss_j(theta)/n))`

REML:

`f(theta) = q * logdet_zz(theta) + q * 2*logdet_xx(theta) + sum_j (n-p) * (1 + log(2*pi*pwrss_j(theta)/(n-p)))`

The determinant terms are shared. Only the `pwrss_j` terms vary across
response columns.

## Recommended data structures

Do not extend `LinearMixedModel` in place first. Introduce a new internal split:

1. `SharedLmmSystem`

- fixed-effects structure
- random-effects structure
- `parmap`
- `a_blocks` and `l_blocks` for `[Z X]' [Z X]`
- all `theta`-dependent factorization logic

2. `ResponseBatch`

- `Y: DMatrix<f64>` with shape `n x q`
- response names
- `xty: DMatrix<f64>` with shape `p x q`
- `yty_diag: DVector<f64>` with length `q`
- per-random-term cross-products `zty` before `Lambda` scaling

3. `MultivariateLinearMixedModel`

- one `SharedLmmSystem`
- one `ResponseBatch`
- fitted `beta: DMatrix<f64>` with shape `p x q`
- profiled `sigma: DVector<f64>` with length `q`
- shared `theta`

## Crucial refactor boundary

The current `[X | y]` representation should not be the basis of the
multivariate implementation.

The scalar-response stepping stone is
[`lazy_fixed_design_materialization.md`](lazy_fixed_design_materialization.md):
split fixed-design structure from response right-hand sides before adding a
matrix-valued response.

Instead, the scalar and multivariate paths should eventually share:

- the structural factorization of `[Z X]' [Z X]`
- response-specific right-hand sides handled separately

That split is the real enabling refactor.

## API recommendation

Do not try to overload the current formula parser with a matrix-valued left-hand
side first.

Prefer one of:

- `MultivariateLinearMixedModel::new(formula, response_cols, data, weights)`
- `MultivariateLinearMixedModel::new_from_response_matrix(formula, y, data, weights)`

where `formula` still describes the shared RHS structure.

## Implementation order

1. Extract a shared `[Z X]` system from the current scalar code.
2. Add batched right-hand-side solves for `Y`.
3. Implement the summed shared-theta objective.
4. Recover batched `beta` and `sigma`.
5. Add differential tests against fitting each response column separately.
6. Only after that consider a true multivariate covariance model.

## Test strategy

The first correctness target is differential parity:

- fit each column of `Y` separately with the current scalar model
- fit the shared-theta batched model
- when `theta` is fixed, verify identical `pwrss_j`, `beta_j`, and objective sums

Then add permutation invariance and performance benchmarks with increasing `q`.
