# Inference Route Simulation Harness

The harness compares inference routes by scenario, not by a single golden
number. Each scenario records which routes are available, which refuse with a
stable reason, and which produce finite calibrated output.

## Scenario Strata

- `interior`: full-rank, interior covariance estimate.
- `boundary`: one variance component near or at zero.
- `reduced_rank`: random-effect covariance has unsupported directions.
- `small_group`: few grouping levels, where finite-sample approximations and
  bootstrap Monte Carlo error need explicit labels.

## Routes

- Wald z fixed-effect row.
- Satterthwaite fixed-effect row.
- Kenward-Roger fixed-effect row.
- Parametric fixed-effect bootstrap row.
- Profile-likelihood confidence interval payload.
- Bootstrap LRT row.
- Boundary LRT variance-component row.

## Required Checks

- Every route returns finite output or a stable `reason_code`.
- Boundary LRT is available only for variance-component comparisons; it refuses
  fixed-effect comparisons.
- Profile CI payloads round-trip through JSON and label REML beta omission.
- Bootstrap rows record replicate count, finite statistic count, failed-refit
  policy, boundary rate, and Monte Carlo SE when available.
- Repeated runs with a fixed seed preserve route status and reason codes.

## CI Tiers

- Fast PR tests: route availability/refusal contracts and JSON round trips.
- Optional slow tests: simulation replicates across all scenario strata.
- Nightly/performance: compare runtime and calibration summaries against the
  previous stored baseline.
