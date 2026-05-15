# Boundary LRT Variance-Component Contract

Schema: `mixedmodels.boundary_lrt` version `1.0.0`.

Purpose: make `boundary_lrt` a variance-component comparison route, not another
fixed-effect p-value method.

## Certified v1 Route

`BoundaryLikelihoodRatioTest::variance_component(smaller, larger)` is available
only when all of the following hold:

- the two models are nested by random-effect structure or covariance-parameter
  count;
- fixed-effect column spaces are identical;
- the ordinary likelihood comparison itself is available;
- exactly one variance/covariance parameter is added.

The reference distribution is the Self-Liang one-parameter boundary mixture:

- weight `0.5`: point mass at zero;
- weight `0.5`: chi-square with one degree of freedom.

For positive LRT statistic `x`, the p-value is
`0.5 * Pr[ChiSq(1) >= x]`; at `x = 0`, the p-value is `1`.

## Refusals

Fixed-effect comparisons return
`boundary_lrt_requires_variance_component_comparison` or
`boundary_lrt_not_fixed_effect_method`. Multi-parameter boundary comparisons
return `boundary_lrt_mixture_weights_not_certified` and should be routed to a
bootstrap or a separately calibrated simulation.

## References

- Self, S. G., and Liang, K.-Y. (1987). Asymptotic properties of maximum
  likelihood estimators and likelihood ratio tests under nonstandard
  conditions. Journal of the American Statistical Association, 82, 605-610.
- Stram, D. O., and Lee, J. W. (1994). Variance components testing in the
  longitudinal mixed effects model. Biometrics, 50, 1171-1177.
