# Bootstrap Fixed-Effect Inference Contract

Status: implemented for explicit scalar LMM contrast rows; bridge callable added
Owner: Rust fixed-effect inference
Parent issue: `bd-01KQATBW8DNAD98P76T667BQCE`

## Purpose

This note pins the Rust contract for `method = bootstrap` fixed-effect
inference rows before numeric bootstrap p-values are enabled.

Bootstrap is an explicit calibration method for fragile fixed-effect contrasts
and model-comparison rows. It is not part of the schema `1.0.0` coefficient
table `auto` ladder.

The central rule is:

> A bootstrap p-value requires a certified null simulation target. A bootstrap
> sample from the fitted full model is useful for coefficient distributions and
> intervals, but it is not by itself a valid hypothesis-test p-value for
> `L beta = rhs`.

## Scope

Initial scope:

- Gaussian LMM fixed-effect scalar contrasts
- explicit `method = bootstrap` requests
- parametric simulation from a Rust-owned fitted model state
- refit of each simulated response through the same Rust LMM engine

Later scope:

- adaptive replicate escalation
- parallel execution
- GLMM bootstrap calibration

Out of scope:

- naive random-effect p-values
- R-side reconstruction of p-values from bootstrap replicate files
- treating full-model bootstrap distributions as null hypothesis tests
- treating cluster-resample estimator distributions as null hypothesis tests

## Bootstrap Targets

Rust must distinguish bootstrap simulation/resampling targets.

| Target | Meaning | May produce fixed-effect p-value? |
|---|---|---|
| `full_model_distribution` | Simulate from the fitted model as estimated. Useful for replicate distributions, percentile intervals, diagnostics, and smoke tests. | No |
| `fixed_effect_null` | Simulate from a constrained model satisfying `L beta = rhs`, with variance parameters and residual scale estimated under the declared null policy. | Yes |
| `cluster_resample` | Resample observed clusters with replacement, refit the full model, and summarize estimator distributions/intervals. | No |

For `fixed_effect_null`, the payload must record:

- contrast label
- `L` matrix and `rhs`
- coefficient names/order used by `L`
- null-estimation policy
- whether the null model was fit by ML or REML
- fitted null beta, theta, and sigma
- whether covariance structure was reused, refit, simplified, or unavailable
- reason if the null target cannot be constructed

Explicit bootstrap fixed-effect p-values must come from a certified
`fixed_effect_null` payload. Full-model and cluster-resample payloads may carry
replicate statistics and intervals, but they do not certify null-hypothesis
p-values.

Model-comparison bootstrap LRT is a separate model-comparison surface:
`stats::parametric_bootstrap_lrt` simulates from the smaller/null fitted model
and refits both nested models. It is intentionally not a fixed-effect
inference-row payload.

## Run Payload

Bootstrap result payloads must be durable JSON and independent of live R state.
The minimum run metadata is:

| Field | Meaning |
|---|---|
| `schema_name` | e.g. `mixedmodels.bootstrap_run` |
| `schema_version` | semantic version, initially `1.0.0` |
| `target` | structured bootstrap target |
| `requested_replicates` | requested replicate count |
| `completed_replicates` | attempted simulations/refits |
| `successful_replicates` | replicates with finite requested statistic |
| `failed_refits` | count and optional reasons/classes |
| `failed_refit_policy` | `exclude`, `count_extreme`, or `abort` |
| `boundary_count` | number of successful refits ending at a boundary |
| `boundary_rate` | `boundary_count / successful_replicates` |
| `seed_record` | seed, RNG family, and reproducibility note |
| `refit_options` | optimizer/refit settings used for simulated responses |
| `statistic` | statistic definition and observed value |
| `replicate_statistics` | finite/non-finite bootstrap statistic values or a durable reference to them |
| `intervals` | optional bootstrap intervals for estimator-distribution targets |
| `mcse` | Monte Carlo standard error for p-value when available |
| `notes` | method caveats and reliability notes |

The existing `MixedModelBootstrap` replicate collection is a useful input, but
it is not yet this certified run payload because it does not declare a null
target, failed-refit policy, MCSE, boundary summary, or seed record.

Implementation note: `bd-01KQBDWN5HHKX4SF8FPJTPD7YV` adds a durable
`BootstrapRunPayload` wrapper with `BootstrapRunMetadata` while preserving the
older replicate-only `MixedModelBootstrap` JSON used by `savereplicates()`.
The metadata records the target, requested/completed/successful replicate
counts, failed-refit policy, boundary count/rate, seed record, refit options,
finite statistic count, MCSE when a p-value is supplied, and notes warning that
`full_model_distribution` and `cluster_resample` runs do not certify
fixed-effect hypothesis-test p-values. The payload may also carry
`replicate_statistics`; this is required for non-coefficient scalar contrasts
because the basic replicate collection stores coefficient standard errors but
not replicate covariance matrices. Estimator-distribution targets may also
carry percentile intervals.

### Stable Wire Labels

Bootstrap option and detail labels are part of the bridge contract for schema
version `1.0.0`.

`failed_refit_policy` accepts and reports exactly:

| Rust variant | Wire label |
|---|---|
| `BootstrapFailedRefitPolicy::Exclude` | `exclude` |
| `BootstrapFailedRefitPolicy::CountExtreme` | `count_extreme` |
| `BootstrapFailedRefitPolicy::Abort` | `abort` |

Seed handling is represented by `FixedEffectBootstrapOptions.seed` on input
and by `details.bootstrap.seed_rng` plus `details.bootstrap.seed` on output.
When a caller supplies a seed, Rust uses `StdRng` and reports
`seed_rng = "StdRng"` with the same integer seed. When no seed is supplied,
Rust draws entropy internally and reports `seed_rng = "unknown"` with
`seed = null`; that run is intentionally not exactly reproducible from the
wire payload alone.

The fixed-effect-null target labels emitted in `details.bootstrap` are also
stable snake-case strings:

| Target field | Wire label |
|---|---|
| `target_kind` | `fixed_effect_null` or `full_model_distribution` |
| `null_target.covariance_policy` | `reuse_fitted_covariance` |

Implementation note: `bd-01KQBDWN5Q6Z8RRQVXNEJVY1M9` adds
`LinearMixedModel::fixed_effect_null_bootstrap_target()` and
`simulate_fixed_effect_null()`. The initial target policy is
`reuse_fitted_covariance`: Rust projects the fitted fixed-effect vector onto
`L beta = rhs` using the fitted fixed-effect covariance and then simulates from
the original fitted theta/sigma with the constrained beta. This is a declared
null target for bootstrap testing, but the payload notes that variance
re-estimation under the null is not yet implemented.

Implementation note: `bd-01KQBDWN66AR1JZB0MDNPWQZRE` adds
`LinearMixedModel::test_contrast_with_bootstrap_payload()` and
`fixed_effect_bootstrap_inference_row()`. These APIs validate a
`fixed_effect_null` payload, check replicate accounting, compute the continuity
corrected bootstrap p-value, attach MCSE/accounting notes, and return
bootstrap-labeled unavailable rows for schema, target, policy, statistic, and
replicate-count failures. `auto` does not select bootstrap in schema `1.0.0`.

Implementation note: `bd-01KQDBF2MKD9WYE3YMH11SCVC3` adds
`LinearMixedModel::fixed_effect_null_bootstrap_inference_table()` and
`fixed_effect_null_bootstrap_inference_row()` as the bridgeable Rust-owned
entry points for R. They construct a certified `fixed_effect_null` target,
simulate/refit through the Rust LMM engine, build a durable
`BootstrapRunPayload`, and return `mixedmodels.fixed_effect_inference_table`
rows. R should call this surface rather than deriving fixed-effect bootstrap
p-values from replicate files.

## P-Value Rule

For a scalar fixed-effect contrast `L beta = rhs`, the initial bootstrap
statistic is the absolute studentized statistic:

```text
t_obs = abs((L beta_hat - rhs) / se_hat)
t_b   = abs((L beta_b   - rhs) / se_b)
```

where the bootstrap samples are generated from the certified
`fixed_effect_null` target.

For a multi-df fixed-effect term or contrast with effective rank `q > 1`, the
bootstrap statistic is a joint Wald/F statistic:

```text
Q = (L beta - rhs)' [L V_beta L']^+ (L beta - rhs)
F = Q / q
```

The same statistic is computed for each bootstrap refit. Rows are available
only when Rust can compute a finite observed statistic and at least the minimum
number of finite replicate statistics. Multi-df rows report
`statistic_name = f`, `numerator_df = q`, and `denominator_df = null`.

The p-value is:

```text
p = (r + c) / (B + c)
```

where:

- `B` is the number of successful finite replicate statistics
- `r = count(t_b >= t_obs)`
- `c` is the continuity correction, initially `1`

Monte Carlo standard error is:

```text
mcse = sqrt(p * (1 - p) / B)
```

Rows are unavailable when `B` is below the method's minimum successful
replicates, when the observed statistic is non-finite, when null target
construction failed, or when the failed-refit policy marks the run unusable.
The initial Rust wiring requires at least 30 finite replicate statistics for an
available row; rows with fewer finite statistics report
`bootstrap_successful_replicates_too_few`.

## Fixed-Effect Row Shape

Available scalar rows use:

```text
method = bootstrap
kind = contrast or coefficient
statistic_name = t
numerator_df = null
denominator_df = null
status = available
reliability = low or moderate
```

Available multi-df rows use:

```text
method = bootstrap
kind = term or contrast
statistic_name = f
numerator_df = effective restriction rank
denominator_df = null
status = available
reliability = low or moderate
```

The row notes must include:

- requested and successful replicate counts
- failed-refit policy
- MCSE
- boundary rate
- null target label

Rows now carry optional structured `details.bootstrap` metadata in addition to
notes. The structured fields include MCSE, requested/completed/successful
replicate counts, failed-refit policy/count, boundary count/rate, seed record,
and a null-target summary. Notes remain user-facing method caveats; R should
prefer structured fields for programmatic decisions.

## Reliability

Initial reliability grades:

| Grade | Rule |
|---|---|
| `moderate` | Null target certified, at least 999 successful finite replicates, low failed-refit rate, finite MCSE, and no severe boundary-rate warning. |
| `low` | Null target certified but replicate count is small, MCSE is large, failed-refit rate is nonzero, or boundary rate is notable. |
| `not_available` | Null target, statistic, refit policy, or replicate accounting is unavailable. |

Bootstrap rows are never `high` in the initial contract.

## Failure Reasons

Bootstrap-specific reasons should distinguish at least:

- `bootstrap_null_target_unavailable`
- `bootstrap_null_fit_failed`
- `bootstrap_replicate_accounting_unavailable`
- `bootstrap_successful_replicates_too_few`
- `bootstrap_observed_statistic_nonfinite`
- `bootstrap_replicate_statistic_nonfinite`
- `bootstrap_failed_refit_policy_unavailable`
- `bootstrap_mcse_unavailable`
- `bootstrap_boundary_rate_too_high`

## Work Breakdown

| Issue | Scope |
|---|---|
| `bd-01KQBDWBXM80J8NYX8GJNHT6Z8` | Specify this bootstrap fixed-effect inference contract. |
| `bd-01KQBDWN5HHKX4SF8FPJTPD7YV` | Add certified bootstrap run metadata payload. |
| `bd-01KQBDWN5Q6Z8RRQVXNEJVY1M9` | Implement `fixed_effect_null` bootstrap target construction. |
| `bd-01KQBDWN66AR1JZB0MDNPWQZRE` | Wire `method = bootstrap` fixed-effect rows from certified payloads. |

Parent issue `bd-01KQATBW8DNAD98P76T667BQCE` closes after the implementation
issues pass tests and explicit bootstrap rows can either return numeric
p-values from a certified null target or unavailable rows with method-specific
reasons.
