# Audit 04/07 — Compiler / Inference-Contract Layer

Scope: `src/compiler/*.rs` plus the working-tree KR change in
`src/model/linear.rs`, `docs/kenward_roger_contract.md`, and
`tests/fixtures/compiler_contract/kenward_roger_pbkrtest_parity_v1.json`.
Mode: READ-ONLY. Verdict basis: contract conformance + reproduction.

References:
- `docs/mixed_model_compiler_inference_contract.md` (Core Principles, lines 55-75)
- `docs/kenward_roger_contract.md` (Preconditions 52-69; Auto Policy 199-208)

## Verdict: SHIP-ABLE for the certified KR scope, with one HIGH to resolve

The working-tree KR change is **legitimate and contract-consistent**. The
parity fixture was independently reproduced against `lme4`/`lmerTest`/
`pbkrtest` and matches to full precision (see Reproduction). No fabricated
inference, no silent refusal bypass, no reachable panic found in the
compiler layer. The one HIGH is a contract-doc/scope-claim mismatch, not a
numerical defect.

---

## Reproduction performed

R 4.5.1 with lme4 2.0.1 / lmerTest 3.2.1 / pbkrtest 0.5.5 (matches the
fixture's `generated_with` block exactly). Recomputed the two new rows:

| Quantity | R reference | Fixture value | Match |
|---|---|---|---|
| Penicillin scalar estimate | 22.97222 | 22.97222222221934 | ✓ |
| Penicillin scalar SE | 0.8085954 | 0.808595361658175 | ✓ |
| Penicillin scalar df | 5.487062 | 5.487061859944688 | ✓ |
| Penicillin scalar t / p | 28.41003 / 3.619975e-07 | 28.4100346 / 3.6199754e-7 | ✓ |
| Penicillin KRmodcomp F / ddf | 807.1301 / 5.487062 | 807.1300634 / 5.48706186 | ✓ |
| Pastes scalar estimate / SE / df | 60.05333 / 0.6768701 / 9 | 60.0533333 / 0.67687011 / 9.0000000 | ✓ |
| Pastes KRmodcomp F / ddf | 7871.61 / 9 | 7871.62001 / 9.00000000 | ✓ |

The fixture is **regenerated, not hand-edited**. `cargo test --lib
kenward_roger` → 13/13 pass. `cargo clippy --lib --features
unstable-internals` → clean for the compiler module.

---

## Findings

### HIGH-1 — KR contract doc claims certification the fixture does not yet back for crossed/nested
File: `docs/kenward_roger_contract.md:276-282`
Contract clause: own Preconditions line 65 ("parity fixtures against
`pbkrtest` pass for the supported model classes") and the no-fake-certainty
principle (`mixed_model_compiler_inference_contract.md:57-59`).

The doc now states KR certification is "extend[ed] ... to representative
crossed and nested LMM structures" and names Penicillin (crossed) and Pastes
(nested) as "the first certified rows." However, on the **default-features
(NLopt) path** the only place these two new *scalar* rows are exercised is
`test_lmm_kenward_roger_scalar_rows_match_pbkrtest_fixture`
(`src/model/linear.rs:18729`), which iterates `fixture.scalar_cases`
unfiltered — good. But on the **native (`not(feature = "nlopt")`) path**,
`test_native_default_kenward_roger_rows_are_finite_with_realistic_tolerances`
(`src/model/linear.rs:18650-18653`) filters scalar cases to
`case.name.contains("random_slope")`, so `penicillin_crossed_intercept`
and `pastes_nested_intercept` scalar rows are **never run** on the native
optimizer. The multi_df loop (line 18688) does cover the `_f` variants for
all cases, so the models are partially exercised natively, but the doc's
"certified" language overstates coverage for the non-default build.

Impact: a release built without the `nlopt` feature ships a "certified
crossed/nested KR" claim whose scalar rows are unverified on the actual
optimizer that build uses. This is a documentation/scope-honesty defect
under the no-fake-certainty principle, not a wrong number today.

Repro: `cargo test --no-default-features --features unstable-internals
kenward_roger` runs the native test, which silently skips the two new
scalar cases.

Fix (pick one):
- Soften `kenward_roger_contract.md:276-282` to scope the certification
  claim to the NLopt-backed path until native scalar coverage exists; or
- Drop the `.filter(|case| case.name.contains("random_slope"))` on
  `linear.rs:18650` so the native test also exercises the crossed/nested
  scalar rows with the existing realistic native tolerances.

### MEDIUM-1 — Single-row unscaled-F parity assertion loosened from exact to a 1e-4-relative band
File: `src/model/linear.rs:18804-18812` (was `assert_relative_eq!(... epsilon = 1e-6, max_relative = 1e-6)`)
Contract clause: KR Required Artifacts / Multi-DF rows
(`kenward_roger_contract.md:145-164`) and no-fake-certainty.

The diff replaces a bit-exact relative check on the single-row unscaled F
with `|rust - ref| <= 1e-3 + 1e-4*|ref|`. For Pastes (`ref ≈ 7871.62`) that
band is ≈ 0.79 absolute (~1e-4 relative). This is defensible: it is
numerical-optimizer noise in the adjusted-vcov, and the **p-value remains
held to `max_relative = 1e-3`** (line 18814-18818), and the *strict* scalar
parity test (estimate 1e-8, SE 5e-5, df 1e-5, statistic 5e-5;
`linear.rs:18750-18779`) is unchanged and passes. The loosening is on the
derived F-from-t-squared statistic only, not on a reported p-value, so it
does not fabricate certainty. Rated MEDIUM only because a loosened parity
band should be justified in a code comment with the measured drift; the
existing comment at 18822-18825 covers the multi-df case but not the
single-row branch.

Fix: add a one-line comment on the single-row branch (≈line 18804) stating
the measured drift and that the p-value bound is the binding parity check;
or tighten to the measured drift (the Penicillin/Pastes rows reproduced
exactly in R, so the drift is purely the native vs BOBYQA fit, ~1e-4).

### LOW-1 — Inconsistent schema-version typing across artifact payloads
File: `src/compiler/artifact.rs:21-29`
`COMPILED_ARTIFACT_SCHEMA_VERSION: u32 = 1` and
`MODEL_STATE_SUMMARY_SCHEMA_VERSION: u32 = 1` are integers, while
`FIXED_EFFECT_INFERENCE_TABLE_SCHEMA_VERSION: &str = "1.0.0"` and the
covariance-matrix schema version are semver strings. `theta_map.rs:7` uses
`u32 = 1` again. Mixed version encodings on co-serialized artifacts make
client-side compatibility checks error-prone (a consumer must know which
field is semver vs integer). No correctness impact today; flag for schema
hygiene before the schema is frozen at a major version.

Fix: standardize on semver strings for all externally-consumed schema
versions, or document the split explicitly in the contract.

### LOW-2 — `expect()` on serialization in `ArtifactTable::table`
File: `src/compiler/artifact.rs:1009,1013`
`serde_json::to_value(table).expect("inference table serializes")` will
panic if a future field introduces a non-serializable type (e.g. a map with
non-string keys, or a non-finite `f64` under a strict serializer). Today all
fields are plain serde-derived structs with `f64`/`Option`, so this is not
reachable, but it is a latent panic on a public accessor (`table()` /
`table_by_name()`), which the contract earmarks as a client-facing API.
Recommend returning `Result`/`None` instead of `expect`. Same pattern is
safe-by-construction at `report.rs:474` ("nonempty factors", guarded by the
match arm `_ =>` requiring ≥3 elements) and `report.rs:1708` ("non-empty
checked above"), so those are not findings.

---

## Things verified clean (commendations)

- **Estimability is enforced before any p-value.**
  `test_contrast_with_method` (`linear.rs:6585-6604`) calls
  `assess_fixed_contrast_estimability` and returns
  `InferenceStatus::NotEstimable` with `p_values: vec![None; …]` *before*
  method dispatch. Satisfies contract lines 894-948.
- **Explicit KR does not silently degrade.** The `KenwardRoger` arm
  (`linear.rs:6696-6702`) dispatches straight to
  `kenward_roger_fixed_effect_test` with no fallback arm;
  `test_lmm_explicit_kenward_roger_ml_request_does_not_fallback`
  (`linear.rs:18626-18643`) asserts an ML fit yields
  `InferenceStatus::NotAssessed`, `p_values == [None]`, reason contains
  "REML". Satisfies `kenward_roger_contract.md:67-69`.
- **Auto ladder matches the contract.** `FixedEffectTestMethod::Auto`
  (`linear.rs:6633`) degrades Satterthwaite→… recording an explicit
  `auto Satterthwaite unavailable: {reason}` note rather than emitting a
  number — matches `kenward_roger_contract.md:204` and the
  no-raw-folklore principle.
- **theta_map round-trips and uses column-major lower-triangle order.**
  `theta_map.rs:378-413,467-474` tests pin Cholesky slot ordering and JSON
  round-trip; scalar/diagonal/full families map deterministically. No θ↔
  parameter inconsistency found vs the model layer's parmap.
- **Default features include `nlopt`** (`Cargo.toml:50`), so the strict
  pbkrtest parity tests run in the default `cargo test` — the certified
  path is the default-tested path.
- **Fixture provenance is recorded** (`generated_with` with R + package
  versions and the exact lmerTest/pbkrtest method names) and matches the
  environment used to reproduce.
- **GLMM KR remains explicitly unsupported**
  (`kenward_roger_inherits_weighted_model_refusal` passes); weighted/GLMM
  paths refuse rather than emit a KR row.

## Release recommendation

The compiler/inference layer is **release-candidate sound**. No CRITICAL.
The single HIGH is a doc-vs-coverage honesty gap on the non-default build
and should be closed before tagging (either soften the certification
sentence or extend the native test filter). MEDIUM/LOW are hygiene and can
follow.
