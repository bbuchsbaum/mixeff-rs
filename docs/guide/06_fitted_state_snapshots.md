# Fitted-state snapshots

`LinearMixedModel` and `GeneralizedLinearMixedModel` expose two methods:

```rust,ignore
let json = fitted.snapshot_json()?;
// Persist `json`; the original model and its native handle may be dropped.
let restored = LinearMixedModel::restore_json(&json)?;
```

Use `GeneralizedLinearMixedModel::restore_json` for a GLMM. The returned model
supports the ordinary post-fit query APIs. Restoration reconstructs the design
and matrix factorizations at the recorded fitted parameters. It does not run
an optimizer or PIRLS, and does not replace the original fit's certificate.
Inference methods can still perform their documented fixed-parameter derivative
calculations. Explicit refitting, profiling, bootstrap and convergence
verification remain operations that intentionally optimize.

The snapshot preserves the original construction formula and compiler policy,
training columns, categorical level order and contrast coding, current response,
weights, covariance parameters, optimizer controls and evidence. GLMM snapshots
also retain the observed response, offsets, family/link, dispersion, fixed
coefficients, random modes, quadrature setting, effective estimator and the last
PIRLS working response and weights. Construction-time policy rebuilds the design;
the final recorded policy governs reconstruction of the fitted factors.

The JSON is an opaque engine-owned persistence record with schema
`mixeff-rs.fitted-state` version `1.0.0`. Consumers must store the entire string
without editing its contents. This is a new contract, independent of the
existing reporting schemas. It is not a cross-version model exchange format.

Restoration requires matching crate version, engine source fingerprint, Cargo
features, target, profile, compiler identity and Rust flags. These conservative
checks can refuse an otherwise compatible build. They do not prove identical
resolved dependency or linked native-library binaries. Exact design witnesses
and numerical fitted-state witnesses provide additional reconstruction checks.
Schema, configuration, missing-state and contradictory-state failures return a
normal model error rather than initiating a replacement fit.

A SHA-256 checksum covers the original serialized payload bytes and envelope
metadata. This detects accidental corruption; it does not authenticate the
producer or make deliberately rewritten snapshots trustworthy. Persist only
snapshots produced by the engine. Serialization refuses nonfinite fitted state,
controls and diagnostic values rather than allowing JSON to silently turn them
into missing values. Initial/trial optimizer objectives can legitimately be
infinite; those history fields use an explicit IEEE-754 bit encoding. Finite
floating-point values are round-tripped without decimal parsing loss.

A successful Rust round trip is provider evidence. Downstream wrappers must
separately store this payload, restore it when a live handle is unavailable, and
test fresh-process save/load/query behavior. Until that consumer work passes,
the existing R bridge's implicit-refit lifecycle limitations remain in effect.
