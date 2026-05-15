# mixeff-rs

Mixed-effects models in Rust — a port of Julia's
[MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl) for fitting
linear and generalized linear mixed-effects models. This crate is the
numerical engine behind the `mixeff` R package.

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

## Quick start

```rust
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel, MixedModelFit};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let formula = parse_formula("y ~ 1 + x + (1 | g)")?;
    let mut model = LinearMixedModel::new(formula, &df, None)?;
    model.fit(false)?; // false = ML, true = REML

    let coef = model.coef();
    println!("fixed effects: {coef:?}"); // ~[2.0, 1.5]
    Ok(())
}
```

## Cargo features

- `default`: COBYLA-based native fits for LMMs and GLMMs. No system
  dependencies.
- `nlopt`: enables NLopt (BOBYQA / large-θ paths and the optional GLMM
  optimizer parity path). Requires CMake plus a C/C++ toolchain at build time.
- `prima`: routes bounded LMM θ optimization through the PRIMA C library
  (BOBYQA). Expects a system PRIMA library visible to the linker.

## Status

Pre-1.0. The public API is still being shaped; see
`docs/v1_0_release_roadmap.md` for the staged path to a stable 1.0 release.

## License

MIT — see [LICENSE](LICENSE).
