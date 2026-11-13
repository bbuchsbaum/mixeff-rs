# Optional finite-sample inference at a variance boundary

`LinearMixedModel::test_contrast_with_boundary_policy` can retain
Satterthwaite or Kenward–Roger inference when an independent random-effect
variance is estimated as zero. The policy is explicit; existing contrast
methods and their defaults are unchanged.

```rust
use mixeff_rs::model::linear::{
    BoundaryInferencePolicy, FixedEffectTestMethod, LinearMixedModel,
};

fn slope_test(model: &LinearMixedModel) {
    let slope = model.coefficient_hypotheses().into_iter()
        .find(|hypothesis| hypothesis.label == "x")
        .expect("model contains a coefficient named x");
    let result = model.test_contrast_with_boundary_policy(
        slope,
        FixedEffectTestMethod::Satterthwaite, // or KenwardRoger
        BoundaryInferencePolicy::ActiveSet,
    );

    if result.test.status.p_value_available() {
        println!("df={:?}, p={:?}", result.test.denominator_df, result.test.p_values);
    }
    println!("conditioned: {}", result.conditioning_applied);
    println!("fixed at zero: {:?}", result.fixed_components);
}
```

Use `BoundaryInferencePolicy::Default` to obtain the ordinary result through
the same wrapper. `ActiveSet` requires an explicit `Satterthwaite` or
`KenwardRoger` method. Scalar contrasts use t tests; multirow hypotheses use
the existing joint F-test machinery. An interior fit passes through to the
ordinary method without conditioning.

At a boundary, `ActiveSet` is a strict request for conditional inference.
An unsupported covariance structure, lack of a certified zero component,
or failed snapping check returns a refusal **even if the ordinary method
would return a p-value**. It does not silently substitute ordinary inference.
Call `Default` explicitly when that is the desired calculation. Contrast and
result types are intentionally available through `model::linear`, without
requiring the `unstable-internals` feature.

## Interpretation and limits

The conditional calculation holds the selected zero variances fixed. It
retains residual sigma and every free covariance parameter, restricts the
information calculation **before inversion**, and applies the selected
finite-sample method to that reduced parameter space. A successful
conditional result has `conditioning_applied = true`, identifies the fixed
components, and receives `ReliabilityGrade::Low`.

This treats the selected covariance structure as given. It does not account
for uncertainty from selecting that structure using the same data. Agreement
with an explicitly reduced model validates the calculation, not its
unconditional Type I error. Prefer a prespecified covariance structure when
possible; use this override when the conditional interpretation is appropriate.

Boundary conditioning currently supports fitted, unweighted REML LMMs with
estimated residual sigma and independent scalar or diagonal covariance
parameters, such as `(1 | subject) + (0 + x | subject)` or
`(1 + x || subject)`. It does not condition general correlated singular blocks,
ML fits, fixed-sampling-variance models, or weighted models. The Rust contrast
API exposes this option; the R bridge does not yet expose it.

The override cannot turn a failed optimizer stop into valid inference. It
requires inward covariance scores consistent with the selected zeros,
stationarity of retained parameters, negligible objective change when snapping
near-zero estimates to zero, and positive-definite restricted curvature.
It refuses generalized-inverse repairs and existing model-level p-value
prohibitions. Refusals carry a reason and no p-value. The caller's model is
unchanged.

Numerical checks use fixed dimensionless parameter coordinates: theta is
scaled by its random-effect design, and residual sigma by its fitted value.
Stationarity is checked against `1e-5` per residual degree of freedom; snapping
may change average restricted deviance by at most `1e-7`. The latter uses
differences of determinant terms and log pwrss ratios, so additive constants
from changing response units do not affect the decision. Finite-difference
steps, curvature inversion and KR covariance-component calculations use
matching scaled coordinates.
An ordinary optimizer stop can still leave more score error than this stricter
inference check permits. Such a fit is refused; tighten its optimizer controls
before requesting conditional inference rather than treating unit changes as
a way to obtain a p-value.

## Boundary optimizer repair

[Issue 19](https://github.com/bbuchsbaum/mixeff-rs/issues/19) exposed a native
optimizer stop with both independent variances zero despite a descent
direction for the slope variance. A zero derivative with respect to theta
does not establish a variance optimum, because variance is theta squared.

Covariance-space checks now cover diagonal terms as well as separate scalar
terms. Probes, classification and recovery coordinates are normalized by the
random-effect design scale. Recovery searches smaller steps, refines the
improving coordinate, preserves the best evaluated point, and allows bounded
successive recovery when another component remains at an invalid zero.
Unresolved inward descent is reported as a nonoptimized fit.
The additional post-fit score certification follows the existing deferred
certificate path: uninspected bootstrap/batch fits avoid repeated probes;
the first certificate, artifact or inference inspection completes and caches
the evidence. Recovery probes needed to find the optimum still run during
fitting. Failed recovery probes are recorded as optimizer-recovery diagnostics.

The supplied 816-row fixture now reaches REML criterion **2311.4456015819**,
matching lme4, rather than the erroneous all-zero criterion
2313.004551862766. Regression checks cover both formula representations,
reversed term order, predictor/response rescaling, and feasible objective
witnesses for two weak variance components.
Inference rescaling checks select the same tightly controlled native optimizer
in both feature builds to separate the calculation from backend termination
and near-zero rounding differences.

## Fit and inspection cost

A focused `bench_rust` adaptation measured valid balanced all-zero random
intercept fits on macOS 27.2, arm64 T6031, with no concurrent test load. The
release build used `--no-default-features --features unstable-internals`.
Two alternating batches per version each used five warmups and 101 fresh
REML fits. These are medians of batch medians, not confidence intervals.

| Rows | Baseline `33c8639` fit | Eager-check version | Deferred-check fit | First inspection |
| --- | ---: | ---: | ---: | ---: |
| 816 | 0.114 ms | 0.124 ms | 0.114 ms | 0.014 ms |
| 8,160 | 0.936 ms | 1.016 ms | 0.958 ms | 0.057 ms |

All three versions reached identical objectives on these valid workloads.
Cached inspections were below the reported 0.0005 ms precision. This supports
deferring the extra certification work in bootstrap and batch use; it does not
establish performance for every covariance structure.

Issue 19 is unsuitable for a baseline speed comparison: the old code is faster
because it returns the erroneous collapsed solution. Correct recovery still
costs work during fitting, independently of deferred certification.

## Numerical evidence

The fixture's conditional slope results agree with an explicitly reduced
model and independent R calculations:

| Method | Denominator df | Standard error | Two-sided p |
| --- | ---: | ---: | ---: |
| Satterthwaite (lmerTest 3.2.1) | 37.96719309 | 0.03930555189 | 0.01787728420 |
| Kenward–Roger (pbkrtest 0.5.5) | 31.34118800 | 0.03941000993 | 0.01919515610 |

Tests also cover joint contrasts, recovery of ordinary linear-model df when
all random variances are zero, unchanged default results and caller state,
invalid boundary stops, malformed contrasts, unavailable p-value policies,
and singular or indefinite restricted curvature.

A separate 192-row crossed-model parity dump agrees with MixedModels.jl to
about 1.2e-6 in the REML criterion. This is a broader optimizer check; the
conditional inference reference above comes from R.

The ignored `active_set_null_calibration` experiment in
`tests/lmm_boundary_active_set.rs` simulates 34 subjects with 24 observations
each, null fixed slope, random slope SD 0.15, residual SD 1, and random
intercept SD either 0 or 0.1 (200 replicates per setting, seed 20261113).
It reports available tests, conditional tests, refusals and p-values below
0.05 separately. It is an exploratory diagnostic, not a nominal-size gate:

| True intercept SD | Available / 200 | Conditional subset | p < 0.05, all available | p < 0.05, conditional subset |
| --- | ---: | ---: | ---: | ---: |
| 0 | 198 | 120 | 13 / 198 | 7 / 120 |
| 0.1 | 198 | 58 | 13 / 198 | 6 / 58 |

These native-backend results include ordinary interior inference among the
available tests. One fit was refused because its variance estimate was not
close enough to zero for the stricter snapping rule; three failed the
unit-invariant free-covariance stationarity check. There were no fit errors.
The small experiment does not establish 5% error control, especially within
the data-selected boundary subset.

In the true-intercept-SD 0.1 setting, the conditional subset rejected 6/58
nulls: **10.3%, about twice the nominal 5%**. Here a truly nonzero variance was estimated
as zero and then treated as known zero. The observed excess is consistent with
anticonservative inference from that selection and is a concrete reason to
retain the Low grade. This small selected sample cannot establish the exact rejection
rate or isolate selection as its sole cause.

```sh
cargo test --no-default-features --test lmm_boundary_active_set \
  active_set_null_calibration -- --ignored --nocapture
```

This implements the opt-in approach discussed in
[issue 20](https://github.com/bbuchsbaum/mixeff-rs/issues/20); broader calibration
remains necessary before assigning a higher reliability grade.
