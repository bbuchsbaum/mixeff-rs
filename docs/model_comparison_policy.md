# Model Comparison Policy

Mixed model comparison separates likelihood-ratio tests from information
criteria. AIC and BIC are ranking criteria. They are not hypothesis tests, and
the comparison table must not fill chi-square, chi-square degrees of freedom, or
p-value fields unless an ordinary adjacent likelihood-ratio test is valid.

## Likelihood-ratio tests

`LikelihoodRatioTest::test` and the LRT columns of `ModelComparisonTable` are
available only when every adjacent pair satisfies all of the following:

- the fitted response values are identical;
- the conditional response family and link are identical;
- both fits use the same likelihood criterion, ML or REML;
- the smaller fixed-effect column space is nested in the larger one;
- the smaller random-effect term structure is nested in the larger one;
- model degrees of freedom increase in the supplied order; and
- if the fits use REML, the fixed-effect column spaces are identical.

Nested ML fixed-effect comparisons may use an ordinary LRT. REML fixed-effect
comparisons require ML refits; the Rust API reports `requires_ml_refit` and does
not perform the refit.

Nested random-effect comparisons with identical fixed effects may use an LRT
under a common ML or REML criterion. Boundary-sensitive random-effect
comparisons may still need a bootstrap or restricted LRT for inferential use;
the ordinary LRT table is only the mechanical likelihood-ratio calculation.

## Information criteria

For non-nested but otherwise comparable models, `ModelComparisonTable` reports
the model label, observation count, degrees of freedom, log-likelihood,
deviance, AIC, BIC, delta AIC, and delta BIC. The LRT fields remain null and the
row records `reason_code = "non_nested_models_lrt_invalid"`.

When the user explicitly requests information criteria,
`ModelComparisonMethod::InformationCriteria` suppresses LRT columns even if the
models are nested. This makes the output a ranking table rather than a
hypothesis-test table.

Information criteria are not comparable across different responses, different
families, different links, or mixed ML/REML criteria. Rows for those cases carry
`information_criteria_available = false` and a stable reason code; strict
callers can reject them before display.

