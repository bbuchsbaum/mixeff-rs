# Aphantasia parity fixture (anonymized)

Delivered from the downstream **mixeff** R package (`inst/extdata/aphantasia`)
for native joint-GLMM parity and performance testing in mixeff-rs. Converted to
language-neutral CSV/JSON so the Rust crate can ingest it without R.

## Provenance & anonymization

Trial-level fixture from the *Loo aphantasia revision 3* manuscript analysis.
Participant IDs are anonymized: fixed salt label `mixeff-aphantasia-revision3-v1`,
MD5 of `salt:id`, prefix `p_`, first 16 hex chars. Raw identifiers are NOT
present. See `ORIGINAL_README.md` for the full upstream description.

## Publishing note

This fixture lives under `tests/fixtures/`, which is already listed in the
crate `exclude` in `Cargo.toml` — so it ships in the git repo for tests/CI but
is **NOT** included in the published crate tarball. Keep it that way (do not
move these files into `src/`, `datasets/`, or anywhere outside an excluded path)
unless you intend to publish manuscript-derived data on crates.io.

## Files

- `trials.csv` — raw trial-level data, 25,916 x 14. R `NA` is written as an
  empty field. This is the master table; the prepared frames below are derived
  subsets/transforms of it.
- `metadata.csv` — participant metadata keyed by the same hashed IDs.
- `reference.json` — frozen **lme4** reference fits + tolerances (verbatim copy
  of the R fixture). This is the parity target. Top-level keys:
  `models` (per case: `formula`, `model_type`, `fixef`, `logLik`, `AIC`,
  `varcorr`/`theta`, ...), `tolerances` (`lmm`/`glmm` abs/rel bounds),
  `counts`, and `inference` (cached manuscript inference summaries).
- `prepared/<case>.csv` — model-ready data frames (exact design inputs used by
  the mixeff reproduction tests), one per case. Factors are written as literal
  level strings. Use these to avoid re-deriving the subsetting / `soa_s`
  standardization / `stimtype` logic.

## Key parity targets (binomial / Bernoulli, `correct` in {0,1})

The slow-convergence issue (see cross-ref) is on `intact` and `combined`. lme4
joint-Laplace references (from `reference.json`):

- `intact`:  logLik = -1297.886
- profiled fast-PIRLS reaches max|Delta fixef| ~ 0.466 vs lme4; the native joint
  path reaches ~0.0075 but only at the full evaluation budget (~22 min).

## Derived-column notes

- `group` = aphant/control (from `aphantasia`); `mask` from `back_masked`;
  `soa_log` = log(SOA), `soa_s` = standardized `soa_log`; `item` = `trial_image`;
  `block` = factor(`block_num`); `stimtype` (combined only) = bubbled/intact.
- The RT case (`rt`) is an LMM on `log_rt` (Gaussian), included for completeness.

## Schema (prepared frames)

#### `prepared/primary.csv` — n = 17280
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 | item)`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric)

#### `prepared/sensitivity.csv` — n = 18240
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 | item)`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric)

#### `prepared/intact.csv` — n = 5760
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 | item)`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric)

#### `prepared/combined.csv` — n = 23040
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s * stimtype + block + (1 + mask + soa_s || participant) + (1 | item)`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric), stimtype (factor)

#### `prepared/rt.csv` — n = 9971
- family / type: **lmm**
- formula: `log_rt ~ group * mask * soa_s + block + (1 | participant) + (1 | item)`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric), log_rt (numeric)

#### `prepared/S1_intercept_only.csv` — n = 17280
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s + block + (1 | participant) + (1 | item)`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric)

#### `prepared/S1_current_uncorrelated_slopes.csv` — n = 17280
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 | item)`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric)

#### `prepared/S1_correlated_slopes.csv` — n = 17280
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s + block + (1 + mask + soa_s | participant) + (1 | item)`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric)

#### `prepared/S1_item_mask_slope.csv` — n = 17280
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 + mask | item)`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric)

#### `prepared/S1_maximal.csv` — n = 17280
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s + block + (1 + mask * soa_s | participant) + (1 + group | item)`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric)

#### `prepared/S7_age_covariate.csv` — n = 16800
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 | item) + age_z`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric), age_z (numeric)

#### `prepared/S9_age_matched_subset.csv` — n = 12480
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 | item)`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric)

#### `prepared/S9_age_matched_subset_age_covariate.csv` — n = 12480
- family / type: **binomial**
- formula: `correct ~ group * mask * soa_s + block + (1 + mask + soa_s || participant) + (1 | item) + age_z`
- columns: participant (factor), bubbled (character), back_masked (character), SOA (numeric), block_num (character), trial_image (character), category (character), correct (integer), rt (numeric), aphantasia (character), age (integer), vviq_standard (numeric), source (character), source_folder (character), item (factor), group (factor), mask (factor), block (factor), soa_log (numeric), soa_s (numeric), age_z (numeric)


## Cross-reference

- Upstream perf issue: `bd-01KT1EAD69WXHX2WNSYV3JK3DQ` (native joint GLMM optimizer ~0 parity progress in first ~150 evals).
- Downstream tracking: mixeff bead `bd-01KSZWJ9KC2Q1BQ5JV2SB0RAER`.
- Delivery bead: `bd-01KT1FFJ78KMF3RABJX9X36BDR`.
