# `mixeff-rs` documentation index

This directory holds design and contract documents. They fall into three
groups: **stability contracts** that constrain user-facing behavior (read these
before relying on an inference/diagnostic output), **internal design** that is
forward-looking and still in flux (do not treat as a stable API), and
**parity & audit** evidence backing the cross-language port.

For the public API stability guarantee itself, see
[`semver_policy.md`](semver_policy.md) and the top-level
[`../CHANGELOG.md`](../CHANGELOG.md).

## Stability contracts

User-facing behavior these documents describe is covered by the SemVer policy.
They define what an output means, when it is refused, and the identifiability
rules — no fake p-values, no hidden model surgery.

- [`semver_policy.md`](semver_policy.md) — the API stability contract; stable
  vs unstable surface, breaking-change definition, MSRV policy.
- [`mixed_model_compiler_inference_contract.md`](mixed_model_compiler_inference_contract.md)
  — the project's stance on inference: explicit identifiability / refusal
  paths, diagnostics live in the crate, the R layer is a client.
- [`satterthwaite_scalar_contract.md`](satterthwaite_scalar_contract.md) —
  Satterthwaite degrees-of-freedom contract (scalar scope).
- [`kenward_roger_contract.md`](kenward_roger_contract.md) — Kenward-Roger
  adjusted vcov / denominator-df contract.
- [`boundary_lrt_variance_component_contract.md`](boundary_lrt_variance_component_contract.md)
  — boundary-LRT behavior for an added variance component.
- [`profile_likelihood_json_contract.md`](profile_likelihood_json_contract.md)
  — profile-likelihood CI JSON contract.
- [`bootstrap_fixed_effect_contract.md`](bootstrap_fixed_effect_contract.md) —
  parametric-bootstrap fixed-effect output contract.
- [`model_comparison_policy.md`](model_comparison_policy.md) — when model
  comparison (LRT/AIC/BIC) is admissible vs refused.
- [`glmm_support_contract.md`](glmm_support_contract.md) — which GLMM
  families/links are supported and the refusal boundaries.
- [`summary_estimates_meta_analysis.md`](summary_estimates_meta_analysis.md) —
  the summary-estimate (meta-analysis) front door contract.

## Internal design (in flux — not a stable contract)

Forward-looking plans and PRDs. These constrain how new features should land
but are explicitly **not** part of the 1.0 stability surface.

- [`v1_0_release_roadmap.md`](v1_0_release_roadmap.md) — the phased plan to
  1.0 (Phases A–E) and explicit out-of-scope list.
- [`compiler_contract_v0_prd.md`](compiler_contract_v0_prd.md) — the v0
  compiler/IR PRD (the `compiler` module is unstable surface).
- [`random_term_card_prd.md`](random_term_card_prd.md) — random-term card PRD.
- [`multivariate_shared_theta.md`](multivariate_shared_theta.md) — planned
  split between shared `[Z X]` factorization and per-response RHS (post-1.0).
- [`response_matrix_batch_lmm.md`](response_matrix_batch_lmm.md) — batch
  response-matrix API design.
- [`fixed_effect_p_values_plan.md`](fixed_effect_p_values_plan.md) — design
  plan behind the fixed-effect p-value surface.
- [`random_effects_formulas.md`](random_effects_formulas.md) — random-effects
  formula syntax reference.
- [`r_layer_proposal.md`](r_layer_proposal.md) — proposed R client layer.

## Parity & audit evidence

Cross-language verification backing the port. Useful when chasing numerical
discrepancies against the Julia reference.

- [`cross_engine_parity_scoreboard.md`](cross_engine_parity_scoreboard.md) —
  cross-engine parity scoreboard.
- [`julia_parity_fixture_drift.md`](julia_parity_fixture_drift.md) — tracked
  drift between Rust and Julia parity fixtures.
- [`prima_backend_parity.md`](prima_backend_parity.md) — PRIMA optimizer
  backend parity.
- [`cobyla_default_optimizer_audit.md`](cobyla_default_optimizer_audit.md) —
  audit of COBYLA as the default optimizer.
- [`fixed_effect_p_value_validation.md`](fixed_effect_p_value_validation.md) —
  validation evidence for the fixed-effect p-value contract.
- [`inference_simulation_harness.md`](inference_simulation_harness.md) — the
  inference-route simulation harness and its output schema.
- [`compiler_verdicts.md`](compiler_verdicts.md) — compiler audit verdicts.
- [`mixeff_upstream_support_report.md`](mixeff_upstream_support_report.md) —
  upstream MixedModels.jl feature support report.
