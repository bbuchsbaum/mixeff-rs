# Gradient of the profiled (RE)ML deviance with respect to θ

Status: Phase 5 S5.1 of `docs/plans/2026-09-05-performance-lead-plan.md`.
The formulas below are implemented as a dense reference in
`src/model/linear/gradient.rs` and validated against the finite-difference
optimizer derivatives on interior, boundary, weighted, and crossed fits.
The blocked (production) implementation is S5.2.

## Setup

For observation weights `W = diag(w)` (all ones when unweighted) write
`√W` for `diag(√w)` and

- `Z` (n × q): the random-effects model matrix, term blocks side by side,
  each term `j` contributing `s_j` columns per level (`ReMat::z` stores the
  transpose, `s_j × n`, and `ReMat::refs` the level of each row);
- `Λ(θ)` (q × q): block-diagonal, one lower-triangular `s_j × s_j` factor
  `Λ_j` repeated for every level of term `j`; `θ` writes into `Λ_j` at the
  column-major positions `ReMat::inds` (the parameter order is the term
  order of `reterms`, then `inds` order within a term);
- `X` (n × p): the full-rank fixed-effects design; `y` the response.

Let `Z_w = √W Z`, `X_w = √W X`, `y_w = √W y`, `Z_t = Z_w Λ` and

    M(θ) = Z_tᵀ Z_t + I_q .

The blocked Cholesky the crate forms is exactly the Cholesky of the
augmented system `[Z_t X_w y_w]ᵀ[Z_t X_w y_w] + diag(I_q, 0, 0)`; its
leading `q × q` factor is `chol(M)`.

The penalized least squares problem for fixed `θ`

    min_{u, β} ‖y_w − X_w β − Z_t u‖² + ‖u‖²

has solution `(û, β̂)` (conditional modes and profiled fixed effects), the
weighted residual `r̂ = y_w − X_w β̂ − Z_t û`, and

    pwrss(θ) = r̂ᵀ r̂ + ûᵀ û = L_yy² ,

the square of the last diagonal entry of the blocked factor. Finally

    C(θ) = X_wᵀ X_w − X_wᵀ Z_t M⁻¹ Z_tᵀ X_w = L_xx L_xxᵀ

is the profiled fixed-effects cross product (the `X` diagonal block of the
factor).

## Objective

With `d = n` (ML) or `d = n − p` (REML), `ℓ(θ) = log det M` plus, for REML,
`log det C`:

- profiled σ:  `f(θ) = ℓ(θ) + d · (1 + log(2π · pwrss / d))`
- fixed σ:     `f(θ) = ℓ(θ) + d · (2 log σ + log 2π) + pwrss / σ²`

Observation weights add the constant `−2 Σ log √w_i`, which does not depend
on `θ` (`weight_logdet_correction`).

## Gradient

Let `E_k = ∂Λ/∂θ_k`: a block-diagonal matrix with the unit entry `e_a e_bᵀ`
(`(a, b)` the row/column of `inds[m]` in `Λ_j`) repeated in every level
block of term `j`, and zero elsewhere. Then `∂Z_t/∂θ_k = Z_w E_k` and, with
`A_ZZ = Z_wᵀ Z_w`,

    ∂M/∂θ_k = E_kᵀ A_ZZ Λ + Λᵀ A_ZZ E_k .

1. Log-determinant of `M`:

       ∂ log det M / ∂θ_k = tr(M⁻¹ ∂M/∂θ_k) = 2 tr(M⁻¹ Λᵀ A_ZZ E_k) .

2. Penalized residual sum of squares. `(û, β̂)` minimise the penalized
   objective for fixed `θ`, so by the envelope theorem only the explicit
   dependence through `Z_t` survives:

       ∂pwrss / ∂θ_k = −2 r̂ᵀ Z_w E_k û .

   Writing `G_j = reshape(Z_{w,j}ᵀ r̂)` and `U_j = û_j` as `s_j × levels_j`
   matrices, this is `−2 (G_j U_jᵀ)[a, b]`: one small `s_j × s_j` product
   per term gives every component of that term's pwrss gradient.

3. REML profiled fixed-effects term. With `R = Z_tᵀ X_w` (q × p),

       ∂C/∂θ_k = −(E_kᵀ Z_wᵀ X_w)ᵀ M⁻¹ R − Rᵀ M⁻¹ (E_kᵀ Z_wᵀ X_w)
                 + Rᵀ M⁻¹ (∂M/∂θ_k) M⁻¹ R ,
       ∂ log det C / ∂θ_k = tr(C⁻¹ ∂C/∂θ_k) .

4. Assemble:

   - profiled σ:  `∂f/∂θ_k = ∂ℓ/∂θ_k + d · (∂pwrss/∂θ_k) / pwrss`
   - fixed σ:     `∂f/∂θ_k = ∂ℓ/∂θ_k + (∂pwrss/∂θ_k) / σ²`

## Boundary parameters

The objective is a smooth function of `θ` on the closed orthant (Λ may have
zero diagonal entries; nothing above is divided by θ), so the gradient at a
boundary point is the ordinary one-sided limit and the KKT check compares
its sign against the bound. The dense reference is validated against the
one-sided finite differences the certificate already uses there.

## Blocked implementation (S5.2)

`profiled_gradient_at_current_theta` (`src/model/linear/gradient.rs`)
computes everything above from the θ-invariant `A` blocks and the current
factor `L`, never from the observations:

- `β̂` by back-substitution on the trailing block; `C⁻¹ = L_xx⁻ᵀ L_xx⁻¹`.
- `e_j = Z_{w,j}ᵀ (y_w − X_w β̂)` is the last row of the `[X|y]ᵀ Z_j` block
  minus `β̂ᵀ` times its first `p` rows; `T_j = Z_{w,j}ᵀ X_w` is those `p`
  rows transposed.
- `[Q | û] = M⁻¹ Λᵀ [T | e]`: one blocked forward/backward solve on the
  random-effects part of `L` with `p + 1` right-hand sides (`û` only for
  ML).
- `Z_{w,j}ᵀ r̂ = e_j − Σ_i S_ji Λ_i û_i`, with `S_ji` the stored `Z_iᵀ Z_j`
  blocks (transposed above the diagonal), and `V_j = Σ_i S_ji Λ_i Q_i`
  from the same products. Hence `∂pwrss/∂θ_k = −2 Σ_ℓ g_{j,ℓ}[a] û_{j,ℓ}[b]`.
- REML: `∂ logdet C/∂θ_k = 2 Σ_ℓ ((V_j − T_j) C⁻¹ Q_jᵀ)_ℓ[a, b]`; the two
  trace terms of the derivation share `Q C⁻¹` and fold into one product.
- `∂ logdet M/∂θ_k = 2 Σ_ℓ (M⁻¹ Λᵀ A_ZZ)_{(j,ℓ),(j,ℓ)}[b, a]` needs `M⁻¹`
  only on the block pattern of `L`. One term: `(L_ℓ L_ℓᵀ)⁻¹` per level.
  Several terms: the block Takahashi recursion
  `Z_ij = ([i = j] L_jjᵀ⁻¹ − Σ_{m>j} Z_im L_mj) L_jj⁻¹`, last term first.
  The leading term's `L_00` is (block-)diagonal, so only the
  level-diagonal blocks of `Z_00` are formed; a dense diagonal block costs
  one `q_j³` product `(L_jjᵀ⁻¹ − W) L_jj⁻¹`. The level blocks of
  `(M⁻¹ Λᵀ A_ZZ)_jj` are then `Z_jj,ℓℓ Λ_jᵀ A_jj,ℓ` plus, per other term
  `i`, `Z_ji[ℓ rows, ·] (Λ_iᵀ S_ij)[·, ℓ cols]`.

Single-term models run one allocation-free pass over the levels (scalar
terms, an unrolled `s = 2` kernel, and a generic `s ≥ 3` kernel on flat
scratch). `Λ_jᵀ g_j = û_j` holds by the normal equations, but `g_j` is
computed directly because `Λ` is singular at θ = 0.

Validation (`gradient::tests`): the blocked gradient matches the dense
reference to 1e-9 relative on single-term fits with `s = 1, 2, 3`, weighted
fits, crossed and nested vector terms, ML and REML, at the optimum, away
from it, and with θ slots at 0.

Cost (release build, `gradient::cost` probes, ratio of gradient time to one
full `update_l` evaluation; host load average ≈ 40 during the run):

| model | d | ML | REML |
|---|---|---|---|
| scalar term, n = 10 000, 1 000 levels | 1 | 0.89 | 1.24 |
| vector term (`1 + days`), n = 10 000, 1 000 levels | 3 | 0.93 | 1.47 |
| crossed `(1 + days | subj) + (1 + days | item) + (1 | site)`, 60 × 40 × 12 | 7 | – | 1.93 |

The optimizer's single-term fast objective path is 2–2.5× cheaper than
`update_l`, so an objective-plus-gradient pair costs 2.7–4.9× one fast
evaluation on single-term rows and ≈ 2.9× on the crossed row; a gradient
optimizer has to cut evaluations by more than that factor to win wall
time (S5.3).

## Consumers (S5.3)

`minimize_with_gradient_and_progress` (`src/optimizer/trust_bq.rs`) drives
the native TrustBQ trust-region loop with this gradient: one oracle call
per iteration, a secant Hessian seeded by forward differences of the
gradient, and an exact projected-gradient stop. The LMM driver reaches it
through `profiled_objective_and_gradient_from_parts`
(`src/model/linear/optimizer.rs`), which factorizes on the optimizer's work
blocks and hands them to `ProfiledGradientInputs::profiled_gradient`. See
the S5.3 entry in `docs/plans/2026-09-05-performance-lead-plan.md` for the
evidence.

The certificate's derivative evidence (`analytic_optimizer_derivatives`)
and the Kenward-Roger varpar Hessian (`hessian_deviance_varpar`) use the
same gradient with a central-difference-of-gradient Hessian (S5.4).
