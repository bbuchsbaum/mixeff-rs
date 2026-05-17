# mixeff-rs

Mixed-effects models in Rust — fitting linear and generalized linear
mixed-effects models. The implementation is developed against Julia's
[MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl) as its
reference and parity target: it follows the same PLS/PIRLS formulation and is
cross-checked numerically against it, but it is an **independent
implementation** that diverges where pragmatic (notably optimizer selection
and the fallback strategy, which have no direct Julia analogue) rather than a
line-for-line port. This crate is the numerical engine behind the `mixeff` R
package.

## Features

- **Linear mixed models** (`LinearMixedModel`): profiled (RE)ML via a blocked
  Cholesky PLS step, with automatic optimizer selection.
- **Generalized linear mixed models** (`GeneralizedLinearMixedModel`): PIRLS
  for the conditional modes with optional adaptive Gauss-Hermite quadrature.
- **lme4-style formulas**: `y ~ 1 + x + (1 + x | g)`, including `||`
  zero-correlation, nested, and interaction grouping terms.
- **Post-fit inference**: variance components, coefficient tables, likelihood
  ratio tests, profile-likelihood and bootstrap confidence intervals — with
  explicit, typed refusals rather than fabricated statistics.
- **Difficult-model diagnostics**: boundary, reduced-rank, weak-identification,
  optimizer-exhaustion, and GLMM approximation-gap cases are reported as
  certified fits or precise diagnostics, not blanket claims of superiority over
  `lme4` or MixedModels.jl.
- **Ergonomic API**: a `prelude`, fluent `LinearMixedModelBuilder` /
  `GeneralizedLinearMixedModelBuilder` with `FitOptions`, built-in contrast
  constructors, and a re-exported `nalgebra` so callers don't pin our version.

## Installation

```toml
[dependencies]
mixeff-rs = "0.1"
```

The default build enables the NLopt optimizer backend for fast BOBYQA/NEWUOA
fits. Use `default-features = false` for a dependency-light native build that
uses TrustBQ for multi-theta LMMs. See [Cargo features](#cargo-features) for
details.

The native TrustBQ profile is useful for downstream packages, binary
distribution, embedded use, and build systems that prefer to avoid additional C
dependencies. NLopt-backed builds remain the default performance path.

## Quick start

The `prelude` pulls in the common types; the builder collapses construction and
the ML/REML choice into one chain:

```rust
use mixeff_rs::prelude::*;
use mixeff_rs::model::{FitOptions, LinearMixedModelBuilder};

fn main() -> Result<()> {
    // Balanced toy data: 8 groups, clear fixed slope + group intercepts.
    let group_offsets = [-3.0, -1.5, 0.5, 2.0, -2.0, 1.0, 3.0, -0.5];
    let jitter = [0.12, -0.20, 0.05, 0.17, -0.09, 0.22];

    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut g = Vec::new();
    for (gi, off) in group_offsets.iter().enumerate() {
        for (k, j) in jitter.iter().enumerate() {
            let xv = k as f64;
            x.push(xv);
            y.push(2.0 + 1.5 * xv + off + j);
            g.push(format!("g{gi}"));
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("y", y)?;
    df.add_numeric("x", x)?;
    df.add_categorical("g", g)?;

    let model = LinearMixedModelBuilder::new(parse_formula("y ~ 1 + x + (1 | g)")?, &df)
        .fit(FitOptions::reml())?; // or FitOptions::ml()

    println!("fixed effects: {:?}", model.coef()); // ~[2.0, 1.5]
    Ok(())
}
```

The lower-level form (`LinearMixedModel::new(formula, &df, None)?` then
`model.fit(false)`) remains available; the builder is purely additive.

## Cargo features

- `default`: enables `nlopt`, the release optimizer path for fast BOBYQA /
  NEWUOA LMM fits and optional GLMM optimizer parity. Requires CMake plus a
  C/C++ toolchain at build time.
- `nlopt`: enables NLopt explicitly. Downstream packagers that cannot carry
  NLopt can use `default-features = false` to keep the native TrustBQ LMM path
  and native GLMM fallbacks.
- `prima`: routes bounded LMM θ optimization through the PRIMA C library
  (BOBYQA). Expects a system PRIMA library visible to the linker.
- `unstable-internals`: exposes the in-flux internal surface (`compiler`,
  `datasets`, `pathology`) as public modules. **Not** covered by the SemVer
  guarantee — opt in only if you need it; it may change in any release.

## Status

Early release (`0.x`). The numerical core — PLS/PIRLS, the blocked Cholesky
update, and the profiled (RE)ML objective — is stable and parity-tested
against MixedModels.jl; the public API framing is still settling. Per Cargo's
`0.x` semantics **any release may contain breaking changes**, so pin an exact
version if you need stability. The stable vs. explicitly-unstable surface and
the precise breaking-change definition live in the
[SemVer policy](https://github.com/bbuchsbaum/mixeff-rs/blob/main/docs/semver_policy.md);
[`CHANGELOG.md`](CHANGELOG.md) records release notes, and the API reference is
on [docs.rs](https://docs.rs/mixeff-rs).

**Scope.** Single-response models only. Multivariate response
(`cbind(y1, y2) ~ …`), Gamma GLMM bootstrap, GLMM profile likelihood, and the
full `I()` / formula-transformation surface are out of scope for the current
line and tracked as later work.

**Difficult models.** The release claim is "certified fit or precise
diagnostic", not "always faster" or "always more convergent" than other
engines. Boundary and reduced-rank LMMs are interpreted through optimizer
certificates and covariance KKT checks; GLMM rows marked as documented
divergence remain non-parity claims until their scorecard row and tests are
promoted together. See the
[difficult-model release contract](docs/difficult_model_release_contract.md)
and [certification note](docs/difficult_model_certification.md).

**GLMM estimation semantics.** The default GLMM path is `fast=true`: profiled
fast-PIRLS estimation with Laplace/AGQ approximation metadata carried in the
fit summary and compiler artifact. It is intentionally not the same
statistical approximation as `lme4::glmer`'s joint Laplace fit, and it can be
less accurate for inference on overdispersed or observation-level-random-effect
models. With the NLopt backend, `fast=false` selects a labelled joint path:
Laplace for `n_agq <= 1`, and AGQ for valid single-scalar random-effect GLMMs
with `n_agq > 1`. Without NLopt it remains an explicit unsupported request.
Any joint attempt or fast-PIRLS fallback is labelled in optimizer status and
diagnostics rather than silently presented as ordinary `lme4` parity.

## License

MIT — see [LICENSE](LICENSE).
