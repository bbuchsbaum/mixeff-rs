# mixeff-rs

[API docs](https://docs.rs/mixeff-rs) · [Guide](docs/guide/) · [Changelog](CHANGELOG.md) · [crates.io](https://crates.io/crates/mixeff-rs)

mixeff-rs is a Rust library for fitting linear and generalized linear
mixed-effects models from lme4-style formulas. Give it a data frame and a
formula such as `y ~ 1 + x + (1 + x | subject)`, and it returns fixed effects,
variance components, standard errors, and Wald and likelihood-ratio tests. When
a model is on a boundary, rank-deficient, or poorly identified, it tells you so
instead of returning numbers that look ordinary.

It is a native Rust implementation of the penalized-least-squares formulation
(with PIRLS for GLMMs) that lme4 and MixedModels.jl also use, and its results
are tested against both on shared reference problems. It is also the numerical engine
behind the [mixeff](https://github.com/bbuchsbaum/mixeff) R package.

> **Status:** release candidate `1.0.0-rc.2`. The numerical core is stable; the
> public API is in its final soak before 1.0.0, so pin the exact version.

## Installation

```toml
[dependencies]
mixeff-rs = "=1.0.0-rc.2"
```

The default build includes the NLopt optimizers and needs CMake and a C/C++
toolchain. For a pure-Rust build with no C dependencies, use
`default-features = false`. See [Cargo features](#cargo-features).

## Quick start

Fit a random-intercept model to eight groups of simulated data:

```rust
use mixeff_rs::model::{FitOptions, LinearMixedModelBuilder};
use mixeff_rs::prelude::*;

fn main() -> Result<()> {
    // 8 groups x 6 observations: y = 2 + 1.5x + group offset + noise.
    let offsets = [-3.0, -1.5, 0.5, 2.0, -2.0, 1.0, 3.0, -0.5];
    let noise = [0.12, -0.20, 0.05, 0.17, -0.09, 0.22];
    let (mut y, mut x, mut g) = (vec![], vec![], vec![]);
    for (i, off) in offsets.iter().enumerate() {
        for (k, e) in noise.iter().enumerate() {
            x.push(k as f64);
            y.push(2.0 + 1.5 * k as f64 + off + e);
            g.push(format!("g{i}"));
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("y", y)?;
    df.add_numeric("x", x)?;
    df.add_categorical("g", g)?;

    let formula = parse_formula("y ~ 1 + x + (1 | g)")?;
    let model = LinearMixedModelBuilder::new(formula, &df).fit(FitOptions::reml())?;

    println!("{}", model.summary_markdown());
    Ok(())
}
```

```text
|             |   Est. |     SE |      z |      p |    σ_g |
|:----------- | ------:| ------:| ------:| ------:| ------:|
| (Intercept) | 1.9146 | 0.7292 |   2.63 | 0.0086 | 2.0595 |
| x           | 1.5271 | 0.0131 | 116.51 | <1e-99 |        |
| Residual    | 0.1551 |        |        |        |        |
```

The fit recovers the simulated intercept (2) and slope (1.5), a between-group
standard deviation of about 2, and the small residual noise. Use
`FitOptions::ml()` for maximum likelihood. The
[getting-started guide](docs/guide/01_getting_started.md) walks through the
same model and how to read each part of the result.

## What it covers

- **Linear mixed models:** REML or ML fits for crossed, nested, and correlated
  random effects. The optimizer is selected automatically.
- **Generalized linear mixed models:** Bernoulli, binomial, Poisson, negative
  binomial, Gamma, and inverse Gaussian families, with Laplace or adaptive
  Gauss–Hermite quadrature.
- **lme4 formula syntax:** `*`, `:`, `/`, `(x | g)`, zero-correlation
  `(x || g)`, and interaction groupings `(1 | g1 & g2)`.
- **Inference:** coefficient tables, variance components, likelihood-ratio
  tests, and profile-likelihood and bootstrap confidence intervals. Unsupported
  inference returns a typed refusal, not a made-up statistic.
- **Diagnostics for difficult models:** boundary fits, reduced-rank covariance,
  weak identification, and an exhausted optimizer budget are detected and
  reported with a specific diagnostic rather than returned silently.

The [supported-features page](docs/guide/05_what_is_supported.md) has the full
matrix of families, links, formula features, and inference methods.

## Scope and limitations

mixeff-rs fits models with a single response. Multivariate responses
(`cbind(y1, y2) ~ …`), GLMM profile likelihood, and general formula
transformations (`I()`, …) are not covered yet.

Its guarantee is not speed or convergence rate on every problem, but that every
fit either passes its convergence checks or comes with a specific diagnostic.
The [guide](docs/guide/) explains how to read those results and refusals.

**GLMM estimation.** The default, `GlmmFitOptions::fast_laplace()`, profiles out
the fixed effects for speed. It is a different approximation from
`lme4::glmer`'s joint Laplace fit and can be less accurate for inference on
overdispersed models or models with an observation-level random effect.
`GlmmFitOptions::joint_laplace()` optimizes all parameters jointly; adding
`.with_n_agq(n)` switches it to adaptive quadrature for models with a single
scalar random effect. Each fit records which method actually ran, including any
fallback. See the [GLMM guide](docs/guide/03_glmms.md).

## How results are checked

Results are tested against two reference implementations:

- **MixedModels.jl** (Julia): Rust fits are compared with checked-in reference
  fixtures, using per-test tolerances (typically 1e-6 to 1e-10). CI regenerates
  the fixtures with a pinned Julia environment to catch changes in the
  reference itself.
- **lme4** (R): a comparison suite of benchmark datasets. Each row is
  classified in a scorecard as parity or a documented divergence, and the
  release gate re-runs lme4 and checks those classifications. The current
  results are in [comparison/REPORT.md](comparison/REPORT.md).

[VERSIONING.md](VERSIONING.md) defines what counts as a breaking change to
numerical output.

## Cargo features

| Feature | Default | What it does |
| --- | :---: | --- |
| `nlopt` | ✓ | NLopt BOBYQA/NEWUOA optimizers, the fastest path. Needs CMake and a C/C++ toolchain. |
| `prima` | | Routes bounded LMM optimization through the PRIMA C library. Set `PRIMA_DIR` if it is installed under a custom prefix. |
| `faer-backend` | | Experimental: faster blocked-Cholesky updates via [faer](https://crates.io/crates/faer) on crossed designs. Results differ at rounding level; benchmark before adopting. |
| `rayon` | | Opt-in parallel execution for batch fitting of many responses. |
| `unstable-internals` | | Exposes internal modules (`compiler`, `datasets`, `pathology`). Not covered by SemVer. |

With `default-features = false`, the crate builds in pure Rust and uses its
own TrustBQ optimizer. This suits restricted build systems and binary
distribution. Wrappers should pin their feature set explicitly rather than
inherit the defaults:

| Consumer | Recommended features |
| --- | --- |
| Rust application | default (`nlopt`) |
| Restricted or pure-Rust builds | `default-features = false` |
| R package, CRAN build | `default-features = false` |
| R package, performance build | explicit `nlopt`, adding `faer-backend` only after wrapper CI evidence |

## Documentation

- [Guide](docs/guide/): getting started, reading results, GLMMs, typed
  refusals, and the supported-features matrix. Rendered as the `guide` module
  on [docs.rs](https://docs.rs/mixeff-rs/latest/mixeff_rs/guide/).
- [API reference](https://docs.rs/mixeff-rs): the stable surface is `prelude`,
  `formula`, `model`, `stats`, `error`, and `types`.
- [CHANGELOG.md](CHANGELOG.md): release notes, including upgrade notes between
  release candidates.
- [VERSIONING.md](VERSIONING.md): SemVer policy for the API, numerical output,
  formula syntax, and JSON schemas.

## License

MIT. See [LICENSE](LICENSE).
