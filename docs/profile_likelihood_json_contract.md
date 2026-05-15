# Profile-Likelihood CI JSON Contract

Schema: `mixedmodels.profile_likelihood_ci` version `1.0.0`.

Purpose: expose the existing Rust profile-likelihood confidence interval
machinery to R bindings without making spline internals part of the wire
contract.

## Scope

- `profile_confint_payload(model, level)` profiles a fitted LMM and returns a
  serializable payload.
- ML fits include `sigma`, `theta`, and active fixed-effect `beta` intervals.
- REML fits include `sigma` and `theta`; fixed-effect beta profile intervals are
  explicitly omitted until a REML beta-profile contract is certified.
- Profile rows are serialized for diagnostics. Forward/reverse spline
  coefficients remain Rust internals.

## Payload Fields

- `schema_name`, `schema_version`: stable wire identity.
- `level`: confidence level used for all interval rows.
- `fit_criterion`: `ML` or `REML`.
- `intervals`: one row per profiled parameter with estimate, lower, upper,
  method, regularity label, and boundary-clamp flag.
- `profile_rows`: raw signed-root deviance rows.
- `notes`: contract notes for downstream display.

## R Binding Rule

The R wrapper should call the Rust payload and deserialize JSON directly. R may
format column names and print methods, but it must not recompute profile
intervals or reinterpret spline diagnostics.
