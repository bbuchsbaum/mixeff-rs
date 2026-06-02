# OSF statcheck parity / convergence-honesty fixture

Reproducer for **bd-01KT3Z64AY45NHA5144G2ZBMSY** (joint_laplace reports
`ConvergedInterior` with a non-finite objective + a sub-optimal coefficient on a
badly-scaled GLMM design). Also exercises the GLMM SE/inference gap
(**bd-01KT3Z64YE5QN7626PQRJSJJVA**).

## Provenance

`osf_statcheck_t2.csv` is the Period-2 (2014–2016) slice of the merged statcheck
dataset from OSF node **538bc**, "Journal Data Sharing Policies and Statistical
Reporting Inconsistencies in Psychology", Study 3
(https://osf.io/538bc/ ; data component https://osf.io/st2ex/). The original
analysis script (`171020MultilevelAnalysis.R`, https://osf.io/jz9r6/) fits these
models with `lme4::glmer`. The OSF-hosted merged file is a lossy UTF-16 re-save
of the authors' `write.csv` output (article titles containing commas lost their
field quoting); this slice was reconstructed with a right-anchored parser that
preserves the trailing columns and the deterministic `Source` grouping. The
reconstruction is faithful: `glmer(nAGQ=0)` reproduces the paper's reported
`OpenPractice:Year` = 0.7958, Z = 1.825, p = .0679 to 4 decimals.

`Source` titles are replaced by a stable integer `gid` (1..426).

Columns: `gid, Error, DecisionError, Year, OpenData, OpenMaterials, Preregistration`
(5279 rows, 426 groups, 380 `Error` events). Derived: `OpenPractice = OpenData |
OpenMaterials | Preregistration`.

## The model

```
Error ~ OpenPractice * Year + (1 | Source)     # binomial / logit
```

`Year` ∈ {2014, 2015, 2016} carries a large additive offset, which is what
triggers the pathology (cf. lme4's "predictor variables are on very different
scales" warning).

## Expected (lme4 / centered reference)

The `OpenPractice:Year` interaction is **invariant to centering `Year`** (centering
shifts only main effects). Reference values:

| quantity                         | value     |
| -------------------------------- | --------- |
| `OpenPractice:Year` (glmer, centered) | 0.85439   |
| `OpenPractice:Year` (mixeff joint_laplace, centered) | 0.85288 |
| logLik (centered)                | −1270.188 |

## Observed bug (mixeff joint_laplace, **raw** `Year`)

| quantity            | value                                   |
| ------------------- | --------------------------------------- |
| `OpenPractice:Year` | **0.79586** (≠ invariant MLE 0.853)     |
| `(Intercept)`       | +448.9995                               |
| `fit_status`        | **`converged_interior`** (FTOL_REACHED) |
| logLik / deviance   | **non-finite**                          |

A `Converged*` status must not accompany a non-finite objective, and the
gradient/KKT gate accepted a non-stationary point because of the scaling. lme4
fails *loud* here (scaling warning; harder cases hard-error in PIRLS); the engine
fails *quiet*.

## Notes

- The silent-wrong variant needs the full-signal slice; small synthetic
  large-offset designs hard-error instead, so this CSV (not a toy `GeneratorSpec`)
  is the reliable reproducer.
- Suggested contract: a model that is `easy` after centering must not surface
  `ConvergedInterior` when fit raw with a non-finite objective — either downgrade
  the `FitStatus`, autoscale internally, or refuse with a structured diagnostic.
