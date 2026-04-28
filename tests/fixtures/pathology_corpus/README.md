# Pathology corpus

Foundation slice for `bd-01KQ8FRGFQEQT8J217YB02D7CB`.

The corpus is contract-driven: each generator spec admits an *analytical*
identifiability certificate, computed in `src/pathology/certificate.rs`
from linear algebra alone (no engine call, no draw of `seed`-dependent
data). The certificate maps to the set of `FitStatus` values that any
conformant fit engine must produce. Tests assert the engine's actual
status is a member of that set, never equality.

## Strata

| Stratum         | Truth condition                              | Acceptable `FitStatus`                                            |
| --------------- | -------------------------------------------- | ----------------------------------------------------------------- |
| `easy`          | full-rank Σ, far from boundary               | `ConvergedInterior`                                               |
| `boundary`      | true σ² = 0 (or \|ρ\| = 1)                   | `ConvergedBoundary`, `ConvergedInterior`                          |
| `reduced_rank`  | rank(Σ_truth) < requested                    | `ConvergedReducedRank`, `ConvergedBoundary`                       |
| `refusal`       | structural unidentifiability (e.g. singletons) | `NotIdentifiable`, `NotOptimized`, `ConvergedBoundary`           |
| `separation`    | logistic separation (FE / conditional / both) | `NotIdentifiable`, `NotOptimized`, `ConvergedPenalised`         |

The `separation` stratum is sub-classified by the
[`SeparationKind`](../../../src/pathology/certificate.rs) carried in
`StructuralIssue::Separation`:

- `FixedEffect(FeSeparationKind)` — Konis (2007) trichotomy detected
  a separating hyperplane in the FE design alone. Fixture:
  `separation_fe.toml`.
- `Conditional { n_groups }` — at least one grouping level has all-
  zero or all-one outcomes; the per-group random intercept's MLE
  drifts to ±∞. Fixture: `separation_conditional.toml`.
- `Both { fe_kind, n_groups }` — both tiers fired, the most
  pathological combination. Fixture: `separation.toml`.

The acceptable set is intentionally larger than one element for boundary,
reduced-rank, and refusal strata. Truth on a contract boundary can
legitimately surface as more than one status depending on optimizer
landing point. Asserting set membership keeps the harness robust to
optimizer noise near boundaries while still catching real regressions:
a `Refusal` that becomes a `Converged` (or vice versa) is always a
regression.

## Where the fixtures live

The foundation slice keeps fixtures *in Rust* under
`tests/pathology_corpus.rs::fixtures`. Each fixture is a
`fn name() -> GeneratorSpec`. This avoids a TOML schema for nalgebra
matrices in the foundation slice; converting to a TOML/YAML loader is
deferred to a follow-up issue once the spec shape is stable. The mote
issue body originally called for TOML fixtures — that requirement is
deliberately deferred and tracked in the same issue's progress notes.

## Composable transforms (`bd-01KQ8FRYQ851X6Q9M33QCTD6PA`)

Transforms in `src/pathology/transforms.rs` mutate a `GeneratorSpec`
in place so callers can stack pathologies on top of a base spec.
Available transforms after the DSL extension:

| Transform                                              | Effect                                                                                  |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `near_singular_re(rho)`                                | Set the (0, 1) off-diagonal of `re_cov_truth` to a target Pearson correlation           |
| `set_group_sizes(sizes)`                               | Replace `group_sizes` wholesale (e.g. with the output of `pareto_sizes`)                |
| `singletons_with_slope(n_groups)`                      | Force one observation per group across `n_groups` groups                                |
| `extreme_prevalence(shift)`                            | Promote the spec to `(Bernoulli, Logit)` and add `shift` to η at sample time            |
| `scale_mismatch(scales)`                               | Set per-predictor scale factors (e.g. `[1.0, 1e6]` for a 6-OOM mismatch)                |
| `collinear_fe(i, j, rho)`                              | Set Pearson correlation between predictors `i` and `j` in `fe_corr_matrix`              |
| `pareto_sizes(seed, n, α, μ)`                          | Free function returning a right-skewed `Vec<usize>` for use with `set_group_sizes`      |
| `empty_crossings(name, n, var, density, seed)`         | Attach a crossed secondary RE; randomly drop cells with `1 - density` probability       |
| `block_diagonal_crossings(name, block, n_blocks, var)` | Attach a crossed secondary RE with a block-diagonal cell pattern (disconnected graph)   |

### Composability rules

Transforms touching disjoint fields commute. Transforms touching the
same field are last-writer-wins; in particular:

- `set_group_sizes`, `singletons_with_slope`, and feeding `pareto_sizes`
  into `set_group_sizes` are mutually exclusive in practice — call once.
- Combinations of pathologies (e.g. imbalance × separation) are
  deliberately deferred until the single-axis suite is stable. The
  policy follows GLMM stratum hygiene
  (`bd-01KQ8FVHD7WCN88RYJX1Y81NEP`) — combining axes obscures which
  subsystem (PIRLS, AGQ, link, dispersion) is at fault.

### Crossed REs and the secondary grouping factor

`GeneratorSpec` carries an optional `crossed: Option<CrossedSpec>` field
that attaches a *secondary* grouping factor with a scalar intercept-only
random effect. When set, observations are emitted from the explicit cell
list `crossed.cells` rather than from `group_sizes`, and the formula
becomes `y ~ ... + (re | g) + (1 | h)`.

`empty_crossings(spec, name, n, var, density, seed)` attaches a crossed
factor with cells drawn independently at the requested density.
`block_diagonal_crossings(spec, name, block_size, n_blocks, var)`
produces the canonical "structurally empty crossings" pathology — a
block-diagonal cell pattern whose bipartite graph has `n_blocks`
disconnected components. The certificate flags this as
`StructuralIssue::DisconnectedCrossings` from the cell list alone, with
no engine call.

`Certificate::crossed_summary` is populated for crossed designs and
records `(n_primary, n_secondary, n_cells, n_components, primary_orphans,
secondary_orphans)` so downstream tooling can introspect cell-graph
topology without reparsing the spec.

## Adding a new fixture

1. Add a `pub fn name() -> GeneratorSpec` under
   `tests/pathology_corpus.rs::fixtures`.
2. Run `certify` on it inside a `#[test]`; assert any structural
   properties you want to lock (rank, boundary directions).
3. Add a `#[test]` that calls `assert_status_in_set` to drive the engine
   and assert the contract status set.

## Weak-identification index (`bd-01KQ8FT90WXSG9VSZQH30HZY9P`)

Each `Certificate` carries a dimensionless weak-identification index
`weak_id_score = n * lambda_min(I) / trace(I)`, where `I` is the
expected Fisher information at truth in correlation form. The index is
invariant to (i) uniform rescaling of the response, (ii) per-axis
rescaling of any fixed-effect predictor — multiplying any column of
`X` by `1e3` leaves the score unchanged.

Designs scoring below `WEAK_ID_THRESHOLD` (default `10`) flip
`Certificate::weak_identification = true`, and `expected_statuses`
widens to admit `ConvergedReducedRank` alongside the usual converged
set in recognition that weakly-identified directions can plausibly
collapse during fitting. See `calibration.md` for the per-fixture
sweep that informs the threshold.

## Out of scope (handled by sibling motes)

- Composable transform DSL beyond `near_singular_re` —
  `bd-01KQ8FRYQ851X6Q9M33QCTD6PA`
- Spectral interpretability (nearest-submodel suggestion) —
  `bd-01KQ8FSZPCBTWWS2Q11WWMQ2VY`
- Cross-engine parity scoreboard (lme4 / MixedModels.jl) —
  `bd-01KQ8FTM1GYAQHG9EKK1V9SNDS`
- Corpus versioning with `contract_version` —
  `bd-01KQ8FV0FYKVT3CHZWXPYW1NPY`
- GLMM stratum hygiene — `bd-01KQ8FVHD7WCN88RYJX1Y81NEP`
