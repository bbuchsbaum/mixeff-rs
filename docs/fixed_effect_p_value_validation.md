# Fixed-Effect P-Value Validation Matrix

Status: reference parity is covered for the implemented analytic methods; simulation calibration remains split into follow-up motes.

Parent mote: `bd-01KQATC0Y1SFMQTXB09C16DEK3`

## Scope

This note records which fixed-effect p-value claims are currently validated,
which reference owns the claim, and which gaps should remain explicit rather
than implied by passing unit tests.

The validation target is the row-level Rust contract consumed by R:

- `test_contrast_with_method(..., AsymptoticWaldZ)`
- `test_contrast_with_method(..., Satterthwaite)`
- `test_contrast_with_method(..., KenwardRoger)`
- `test_contrast_with_bootstrap_payload(...)`
- `fixed_effect_inference_table()` and bridge table `fixed_effect_inference`

## Current Validation Coverage

| Method | Current coverage | Reference or oracle | Residual gap |
|---|---|---|---|
| `asymptotic_wald_z` | Explicit scalar contrast tests, coefficient-table consistency checks, unavailable p-value gating, and a bounded H0 simulation smoke test. | Internal `coeftable()`/`test_contrast()` identity, standard normal formula, and fixed-seed null simulation. | Add a compact row-level table test that asserts Wald rows match `coeftable()` exactly when requested explicitly. |
| `satterthwaite` | Scalar LMM rows have parity fixtures, boundary/rank-deficient unavailable-reason tests, and a bounded H0 simulation smoke test. `auto` prefers Satterthwaite only after derivative and parity prerequisites. | `vendor/lmerTestR` fixtures in `tests/fixtures/compiler_contract/satterthwaite_lmer_test_parity_v1.json` plus fixed-seed null simulation. | Broaden simulation cases only if runtime remains acceptable. |
| `kenward_roger` | Explicit scalar and multi-df rows have fixture coverage, no-fallback tests, denominator-df tests, active-basis contrast mapping coverage, and a bounded H0 simulation smoke test for a scalar row. | `pbkrtest`/`lmerTest` fixture in `tests/fixtures/compiler_contract/kenward_roger_pbkrtest_parity_v1.json` plus fixed-seed null simulation. | Multi-df rows still document unscaled F parity. |
| `bootstrap` | Explicit payload rows validate `fixed_effect_null` target shape, null simulate/refit/payload row construction, replicate accounting, continuity-corrected p-value, MCSE notes, coefficient-row fallback, contrast-row supplied statistics, and too-few-replicate unavailable reasons. | Rust-owned null payload contract in `docs/bootstrap_fixed_effect_contract.md` plus fixed-seed null simulation. | Broaden to larger/adaptive bootstrap calibration only if runtime and statistical-power rationale are documented. |
| unsupported cases | Rank-deficient, predictive, regularized, post-selection, missing-SE, boundary, and method-prerequisite failures return labeled unavailable rows. | Rust row status/reason tests and compiler-contract fixtures. | Keep new unsupported reasons covered by focused tests whenever the method surface grows. |

## Simulation Follow-Ups

The existing reference fixtures check numerical parity for known examples. They
do not by themselves establish calibration. The calibration checks should be
small, deterministic smoke tests rather than slow Monte Carlo studies.

Open child motes:

| Issue | Scope |
|---|---|
| `bd-01KQBF0ZMP9NK20G0EDJGBW53Q` | Done: add bounded H0 simulation smoke tests for Wald/Satterthwaite/KR type-I behavior. |
| `bd-01KQBF0ZNDDVSJWZX2R810ND54` | Done: add a bootstrap fixed-effect calibration fixture using `fixed_effect_null` simulation and certified payload rows. |

Suggested simulation rules:

- Use fixed RNG seeds and small fixture data to keep CI runtime stable.
- Assert broad type-I or p-value-bin sanity, not exact uniformity.
- Record method, reliability, replicate count, and unavailable reasons in test
  names or assertion messages.
- Do not make bootstrap part of `auto`; it remains explicit in schema `1.0.0`.

## Closeout Rule

The parent validation mote can close when:

- analytic reference parity remains green for Wald/Satterthwaite/KR;
- bootstrap payload rows have at least one full null simulate/refit validation;
- simulation smoke tests are either landed or explicitly deferred with a
  documented runtime/statistical-power rationale;
- unsupported-case reasons remain covered by tests or contract fixtures.
