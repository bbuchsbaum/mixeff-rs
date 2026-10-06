# Downstream adoption regression fixtures

These model frames are copied unchanged from the standalone Rust reproducers
attached to Motes `bd-01M491AR35FPJN02FVNN17YA3K` (LMM) and
`bd-01M491ARP1DS899H0M6GPMAQ3P` (GLMM), reported against engine `897dd25`.
They preserve explicit R factor levels and column order.

- `pw2_nested.json`: the public `mixeff/tests/fixtures/pw2_crossed_nested.csv`,
  60 observations. Formula: `player_value ~ 1 + (1 | team) + (1 | team:player_id)`.
- `osf_centered.json`: the public `mixeff/inst/extdata/osf-statcheck-t2.csv`,
  5279 observations and 426 Source groups. `Source = factor(gid)`,
  `OpenPractice = OpenData == 1 | OpenMaterials == 1 | Preregistration == 1`,
  and `cYear = Year - 2015`. Formula: `Error ~ OpenPractice * cYear + (1 | Source)`.

Only variables used in each model are retained. The tests exercise initial fits,
with no snapshot/restoration calls. Numerical acceptance uses the original
downstream bounds and independent objective/derivative checks.

`generate_lme4.R` regenerates `lme4_reference.json` from these exact model frames
using default `lme4::lmer` (REML) and `lme4::glmer` (Bernoulli/logit, nAGQ=1)
controls. The reference records the R and lme4 versions. LMM checks retain
the downstream fixed-effect, sigma/log-likelihood/AIC, fitted-value and
variance-component bounds; GLMM checks retain coefficient and log-likelihood
bounds and independently verify the conditional objective and stationarity.
