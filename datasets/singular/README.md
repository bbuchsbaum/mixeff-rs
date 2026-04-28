# `singular`

This fixture is the compact version of the "singular fit with no obvious
VarCorr symptom" problem.

The requested maximal model is:

```text
y ~ A * B * C + (A * B * C | group)
```

On the user scale, that random-effect term means an 8-dimensional
group-specific coefficient vector:

```text
intercept, A, B, C, A:B, A:C, B:C, A:B:C
```

A full covariance for that basis has 36 covariance parameters. The data have
only 10 groups. Under the compiler-contract v0 budget, the requested full
covariance is therefore design-too-rich before optimization starts.

The important point is not that lme4 prints a singular-fit warning. The point
is that the usual printed covariance table can look plausible: no visibly
zero main variances are required, and correlations need not sit at exactly
`+1` or `-1`. The rank problem lives in the covariance matrix as a whole.

The intended compiler story is:

1. Preserve the user's requested model.
2. Compile `(A * B * C | group)` into the eight named random-coefficient
   directions above.
3. Record the information budget: 10 grouping levels, 8 basis directions, 36
   full-covariance parameters, threshold 180 levels.
4. If the zero-correlation diagnostic is fitted,
   `(A * B * C || group)`, report reduced effective rank and unsupported
   interaction directions.
5. Say "unsupported by this data set", not "the true population variance is
   zero".
6. Treat p-values after any data-dependent simplification as conditional or
   exploratory unless a confirmatory structure was declared before fitting.

The reduced diagnostic model keeps fixed interactions but removes random
interaction slopes:

```text
y ~ A * B * C + (A + B + C || group)
```

The final simulation-truth model described in the source answer is:

```text
y ~ A + B + C + (A + B + C || group)
```

Sources:

- Cross Validated discussion:
  <https://stats.stackexchange.com/questions/449095/how-to-simplify-a-singular-random-structure-when-reported-correlations-are-not-n>
- CSV mirror used here:
  <https://raw.githubusercontent.com/rnorouzian/e/master/sng.csv>
- Original answer reference:
  Bates, Kliegl, Vasishth, and Baayen (2015), "Parsimonious mixed models",
  <https://arxiv.org/abs/1506.04967>
