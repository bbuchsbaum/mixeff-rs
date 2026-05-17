# Difficult Model Release Contract

Status: release-facing wording and label contract.

`mixeff-rs` should not claim blanket superiority over `lme4` or
MixedModels.jl. The release-safe claim is narrower:

`mixeff-rs` aims to return either a certified fit or a precise diagnostic on
difficult mixed models.

That claim covers boundary-heavy, singular, weakly identified, badly scaled,
crossed, maximal, and selected GLMM pathology cases. It does not mean every
hard row is parity-clean, faster, or inference-equivalent to another engine.

## Vocabulary Source

Difficult-model wording must be derived from existing crate surfaces:

| Concept | Existing surface |
| --- | --- |
| clean convergence, boundary convergence, reduced-rank convergence, weak/non-identifiable fits, optimizer exhaustion | `FitStatus`, `OptimizerCertificate`, and `ConvergenceVerdict` |
| valid zero variance, valid rank-deficient covariance, invalid boundary stop, weak identification | scalar and 2x2 covariance KKT certificates |
| recovered convergence after an invalid boundary stop | `KKT_BOUNDARY_RESTART(...)`, `OptimizerRecovery`, and recovered-convergence verdict wording |
| GLMM approximation gaps and non-parity rows | `comparison/parity_scorecard.toml`, `comparison/difficult_model_scoreboard.toml`, and GLMM support docs |

Do not add a second hard-model status object, R-side convergence taxonomy, or
private downstream message vocabulary for these concepts.

## Stable Versus Unstable Surfaces

Downstream clients should treat the public fitted-model outputs, printed
verdicts, scorecard labels, and documented status meanings as the stable
contract. In particular, release-facing code may rely on whether a result is
reported as a clean fit, certified boundary, reduced-rank fit, weak
identification, invalid boundary stop, recovered convergence, approximation
gap, or optimizer exhaustion.

The `unstable-internals` feature may expose richer compiler, dataset, and
pathology structs for testing and development. Those internal structs are not
the downstream contract. If an R/Python client needs a hard-model field that is
only available behind `unstable-internals`, add a small stable API field rather
than making the downstream client depend on internals.

## Release Labels

The public scorecard/release labels are:

| Label | Meaning |
| --- | --- |
| `release_blocking_parity` | Release gate: the row is expected to match the declared reference within tolerance. |
| `documented_divergence` | The row is useful evidence but is explicitly not a parity claim. |
| `diagnostic_contract` | Rust returns a precise diagnostic or certificate-backed classification where ordinary parity is not the point. |
| `performance_known_slow` | Numeric behavior is separately classified, but speed remains a known issue. |
| `experimental_recovery` | Recovery behavior is covered by tests/scoreboard evidence but is not a blanket default-success claim. |

Other labels such as `stress_opt_in` and `unsupported_with_contract` can appear
in the broader parity scorecard, but the five labels above are the difficult
model release vocabulary.

## Examples And Claims

Accepted release wording:

- "zero random-effect variance certified as a valid boundary optimum"
- "rank-deficient covariance reported as a reduced-rank fit"
- "invalid boundary stop recovered by KKT-guided restart"
- "GLMM fast-PIRLS row documented as an approximation gap"
- "optimizer exhausted its budget; row remains documented divergence"

Rejected release wording:

- "more robust than lme4"
- "converges difficult models better than MixedModels.jl"
- "GLMM parity" for rows marked `documented_divergence`
- "warning-free" for a boundary fit unless the certificate actually supports
  the boundary interpretation

## Example Interpretation

`dyestuff2_reml_zero_variance` in
`comparison/difficult_model_scoreboard.toml` is the canonical scalar LMM
boundary example:

- the fitted random-effect variance is zero;
- the row remains `release_blocking_parity` because the fitted quantities match
  the declared reference tolerance;
- the hard-model certification status is `certified_boundary`;
- downstream output should describe this as a valid boundary fit, not as a
  generic convergence warning.

That example is the model for future valid rank-deficient covariance rows:
promote only when the certificate supports the boundary or lower-rank face, and
keep the scorecard/test update in the same change.

## Evidence Links

Evidence must be traceable to:

- `comparison/parity_scorecard.toml` for ordinary release parity and
  documented divergence classes;
- `comparison/difficult_model_scoreboard.toml` for hard-model rows,
  certification claims, required metrics, and time-to-certified-fit inputs;
- `tests/difficult_model_scoreboard.rs` for manifest invariants;
- covariance KKT and KKT-guided recovery tests in `src/model/linear.rs`;
- GLMM diagnostic taxonomy and support-contract tests for GLMM rows.

Changing a row from divergence/diagnostic/recovery into release parity requires
updating both the manifest and the corresponding executable test.
