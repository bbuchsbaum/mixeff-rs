# Historical COBYLA No-Default-Features Optimizer Audit

This is a historical audit of the former no-default-features LMM path. The
current dependency-light `--no-default-features` LMM path uses native TrustBQ
for multi-parameter theta fits; COBYLA remains available explicitly and is
still used by native GLMM fallback paths.

Context at the time of the audit: the release default optimizer path was NLopt
because BOBYQA/NEWUOA was substantially more iteration-efficient on vector and
crossed LMMs. The dependency-light `--no-default-features` build used native
COBYLA for multi-parameter theta fits, so this audit recorded the expected
COBYLA drift against the NLopt reference path used by the Julia/R fixtures.

## Commands

Measured with:

```sh
cargo test --features nlopt --lib test_cobyla_nlopt_delta_audit_stays_within_documented_envelope -- --nocapture
cargo test --no-default-features --lib cobyla_default -- --nocapture
```

The first command compares forced COBYLA against the NLopt-enabled reference
path in one test binary. The second command exercised the former
no-default-features COBYLA-owned contracts; current no-default LMM tests assert
the TrustBQ native path instead.

## Delta Table

| Case | Quantity | COBYLA - NLopt |
| --- | --- | ---: |
| pastes ML varcorr | objective | 1.098270e-6 |
| pastes ML varcorr | max theta abs delta | 7.686167e-4 |
| pastes ML varcorr | max beta abs delta | 2.003731e-12 |
| pastes ML varcorr | max VarCorr sd abs delta | 3.847998e-4 |
| pastes ML varcorr | sigma^2 delta | 1.158696e-4 |
| sleepstudy KR scalar days | SE delta | 1.601118e-4 |
| sleepstudy KR scalar days | denominator df delta | 2.131628e-13 |
| sleepstudy KR scalar days | statistic delta | -7.013127e-4 |
| sleepstudy KR joint intercept+days | unscaled F delta | -6.494661e-1 |
| sleepstudy KR joint intercept+days | denominator df delta | 1.492140e-13 |
| sleepstudy KR joint intercept+days | p-value delta | 0.000000e0 |
| sleepstudy Satterthwaite days | beta delta | 9.414691e-14 |
| sleepstudy Satterthwaite days | SE delta | 1.601118e-4 |
| sleepstudy Satterthwaite days | denominator df delta | -4.714760e-3 |
| sleepstudy Satterthwaite days | statistic delta | -7.013127e-4 |
| singular ZCP fit | objective delta | 2.662178e-1 |
| singular ZCP fit | NLopt certificate | ConvergedReducedRank, effective_covariance=1 |
| singular ZCP fit | COBYLA certificate | NotOptimized, effective_covariance=0 |

The ordinary-fit objective deltas are small, but KR/Satterthwaite amplify the
theta drift enough that the exact parity tests should remain NLopt-gated. The
singular ZCP fit is a different class: COBYLA does not reach the reduced-rank
certificate region and correctly leaves no effective-covariance summary.

## Control Sweep

For `reaction ~ days + (days | subj)` under REML:

| COBYLA control | objective delta vs NLopt | max theta delta | feval | return |
| --- | ---: | ---: | ---: | --- |
| default | 9.039912e-6 | 6.975222e-4 | 155 | FTOL_REACHED |
| max_feval=50000 | 9.039912e-6 | 6.975222e-4 | 155 | FTOL_REACHED |
| initial_step=0.25 | 5.877828e-6 | 5.725523e-4 | 189 | FTOL_REACHED |
| xtol_abs=1e-6 | 9.039912e-6 | 6.975222e-4 | 155 | FTOL_REACHED |
| start at NLopt theta | 0.000000e0 | 0.000000e0 | 42 | FTOL_REACHED |

Increasing the evaluation budget does not help because COBYLA stops by `ftol`
before using the default budget. A smaller initial step improves the drift only
modestly. Starting from the NLopt optimum stays there, which points to the
optimizer trajectory/termination rather than an objective mismatch.

The native `cobyla` crate exposes `RhoBeg` and `StopTols`, not a direct
`rhoend` argument. `xtol_abs` was swept as the closest native end-tolerance
control. `OptSummary.rhoend` remains PRIMA-specific.

## Resulting Contract

- NLopt parity tests stay behind `#[cfg(feature = "nlopt")]` and are covered
  by the default release build.
- No-default-features builds now own TrustBQ-native tests for finite
  KR/Satterthwaite inference, finite pastes variance components, explicit
  singular-fit certificate state, and realistic drift envelopes.
- Native COBYLA now validates and honors `OptSummary.initial_step` instead of
  silently using a hard-coded scalar step.
- CI must run both `cargo test` and `cargo test --no-default-features`.
