# Weak-identification index calibration

This file documents the empirical calibration of the dimensionless
weak-identification index `weak_id_score` against the existing pathology
corpus, per `bd-01KQ8FT90WXSG9VSZQH30HZY9P`.

## Index definition

Let `I` be the expected Fisher information at truth, reduced to
**correlation form** so the spectrum is invariant to per-axis predictor
rescaling:

```
I_corr = D^{-1/2} · (D · C · D) · D^{-1/2} = C
```

where `D = diag(predictor scales)` and `C = spec.fe_corr_matrix`.
We append a unit eigenvalue for the intercept slot (uncorrelated with
the slopes by construction). The index is then

```
weak_id_score = n_total · lambda_min(I_corr) / trace(I_corr)
```

with `n_total` = total observations across the spec.

A design is flagged `weak_identification = true` when
`weak_id_score < WEAK_ID_THRESHOLD`. The threshold default is **10**,
defined in `src/pathology/certificate.rs::WEAK_ID_THRESHOLD`.

## Why correlation form

The acceptance criterion is *"the same dataset rescaled by 1e3 produces
identical `weak_id_score`"*. Reducing the population predictor
covariance `S_X = D · C · D` to its correlation form cancels `D` from
both `lambda_min` and `trace`, giving an index that does not move under

- uniform rescaling of all predictors,
- per-axis rescaling of any single predictor,
- uniform rescaling of the response.

## Calibration sweep

Every value below was produced by `cargo run --example probe_weak_id`
on commit-time pathology fixtures. The probe materialises each fixture
by name, calls `certify`, and reports `weak_id_score`, `n_total`, and
the resulting `weak_identification` flag.

| Fixture                            | n    | score   | weak_id | notes                                     |
| ---------------------------------- | ---- | ------- | ------- | ----------------------------------------- |
| `easy`                             | 180  |  90.000 | false   | balanced 30×6, single predictor, C = I    |
| `boundary_zero_slope`              | 180  |  90.000 | false   | same FE as easy, RE slope variance = 0    |
| `reduced_rank`                     | 180  |  90.000 | false   | RE rank-1 (ρ=1 in Σ_truth), C = I         |
| `refusal_singletons`               |   6  |   3.000 | **true**| 6 singleton groups; structural refusal     |
| `imbalance_pareto`                 | 174  |  87.000 | false   | 30 groups, pareto-sized cells             |
| `scale_mismatch_1e3`               | 180  |  90.000 | false   | FE predictor scale ×1000; C = I           |
| `collinear_fe_rho_one`             | 180  |   0.000 | **true**| two predictors at ρ = 1 (structural)      |
| `extreme_prevalence_negative_5`    | 600  | 300.000 | false   | Bernoulli/logit, 600 obs, C = I           |
| `singletons_via_transform`         |   8  |   4.000 | **true**| transform-built singletons; structural    |
| `random_slope_singletons`          |  12  |   6.000 | **true**| 12 singletons + slope; structural         |
| `crossed_block_diagonal_4x4x4`     |  64  |  64.000 | false   | intercept-only, 4 disjoint 4×4 blocks     |
| `crossed_sparse_connected`         |  67  |  67.000 | false   | connected sparse 12×12 crossing           |
| `weakly_identified_near_collinear` |  12  |   0.040 | **true**| small n + ρ = 0.99 (weak-id only)         |

## Threshold rationale

Threshold `= 10` separates the two regimes cleanly on this corpus:

- **Above 10 (well-identified):** `easy`, `boundary`, `reduced_rank`,
  `imbalance_pareto`, `scale_mismatch_1e3`, `extreme_prevalence_*`,
  both crossed-RE fixtures. None of these designs has a near-collinear
  predictor pair *and* a small `n_total` simultaneously.
- **Below 10 (weakly identified):** the structural-refusal fixtures
  (`refusal_singletons`, `collinear_fe_rho_one`,
  `singletons_via_transform`, `random_slope_singletons`) trigger weak
  identification on top of their structural flag — both signals coexist
  and the structural one takes precedence in `expected_statuses`. The
  one *purely* weak-id fixture, `weakly_identified_near_collinear`,
  scores `0.04` and exercises the weak-id widening branch directly.

Per-fixture monotonicity sanity checks are baked into the unit-test
suite (`weak_id_score_drops_with_collinear_predictors`,
`weak_id_score_is_invariant_under_uniform_predictor_rescale`).

## Re-running the sweep

```bash
cargo run --example probe_weak_id
```

If the threshold needs to move, regenerate the table above and update
`WEAK_ID_THRESHOLD`. The calibration is meant to evolve as the corpus
grows; the unit tests pin behaviour rather than specific numbers, so a
threshold tweak only requires the doc above to stay in sync with the
fresh probe output.
