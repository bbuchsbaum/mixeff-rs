//! Mixed-effects models in Rust.
//!
//! This crate provides types and algorithms for fitting linear and
//! generalized linear mixed-effects models. It is developed against Julia's
//! [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl) as the
//! reference and parity target — the same PLS/PIRLS formulation, cross-checked
//! numerically — but is an independent implementation with deliberate
//! divergences (notably optimizer selection and the fallback strategy), not a
//! line-for-line port.
//!
//! # Tutorial
//!
//! New to the crate? Read the [`guide`] module — five short, doctested
//! pages:
//!
//! 1. [Getting started](crate::guide::getting_started) — build a frame,
//!    parse a formula, fit an LMM.
//! 2. [Reading results](crate::guide::reading_results) — coefficients,
//!    variance components, summaries, Wald vs. profile vs. bootstrap CIs.
//! 3. [GLMMs](crate::guide::glmms) — families, links, and the GLMM
//!    estimation semantics.
//! 4. [When the crate refuses](crate::guide::when_the_crate_refuses) —
//!    typed errors and typed inference refusals (the no-fake-statistics
//!    contract from a caller's POV).
//! 5. [What is supported](crate::guide::what_is_supported) — a concrete
//!    inventory: model classes, families/links, formula syntax, inference
//!    paths, and what is deliberately out of scope.
//!
//! # Quick start
//!
//! Fit a linear mixed model `y ~ 1 + x + (1 | g)` on a small balanced dataset.
//! The [`prelude`] pulls in the common types and the
//! [`LinearMixedModelBuilder`](crate::model::LinearMixedModelBuilder)
//! collapses construction and the ML/REML choice into one chain:
//!
//! ```
//! use mixeff_rs::prelude::*;
//! use mixeff_rs::model::{FitOptions, LinearMixedModelBuilder};
//!
//! # fn main() -> Result<()> {
//! let group_offsets = [-3.0, -1.5, 0.5, 2.0, -2.0, 1.0, 3.0, -0.5];
//! let jitter = [0.12, -0.20, 0.05, 0.17, -0.09, 0.22];
//!
//! let mut y = Vec::new();
//! let mut x = Vec::new();
//! let mut g = Vec::new();
//! for (gi, off) in group_offsets.iter().enumerate() {
//!     for (k, j) in jitter.iter().enumerate() {
//!         let xv = k as f64;
//!         x.push(xv);
//!         y.push(2.0 + 1.5 * xv + off + j);
//!         g.push(format!("g{gi}"));
//!     }
//! }
//!
//! let mut df = DataFrame::new();
//! df.add_numeric("y", y)?;
//! df.add_numeric("x", x)?;
//! df.add_categorical("g", g)?;
//!
//! let model = LinearMixedModelBuilder::new(parse_formula("y ~ 1 + x + (1 | g)")?, &df)
//!     .fit(FitOptions::reml())?; // or FitOptions::ml()
//!
//! let coef = model.coef(); // fixed effects, ~[2.0, 1.5]
//! assert_eq!(coef.len(), 2);
//! assert!(coef.iter().all(|v| v.is_finite()));
//! # Ok(())
//! # }
//! ```
//!
//! The lower-level form (`LinearMixedModel::new(formula, &df, None)?` then
//! `model.fit(false)`) remains available; the builder is purely additive.

#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::bare_urls)]
// Clippy allowlist policy (v1.0 — see docs/v1_0_release_roadmap.md Phase D #18).
//
// This crate is a numerically-faithful implementation of the MixedModels.jl
// algorithms. A small, deliberately narrow set of style lints is allowed
// crate-wide because fixing them would obscure the reference algorithms or
// change numeric semantics:
//
// - `needless_range_loop`: blocked-Cholesky / Z'Z / Λθ kernels index several
//   parallel arrays by the same counter to mirror the Julia/BLAS loop algebra;
//   iterator rewrites hide the index relationships and invite parity drift.
// - `too_many_arguments`: solver/optimizer entry points mirror the reference
//   API surface (θ, β, weights, blocks, …) and the inference contract docs;
//   bundling into ad-hoc structs would diverge from that contract.
// - `type_complexity`: blocked-factor return tuples (`Vec<MatrixBlock>`, …) are
//   intrinsic; aliasing them adds indirection without improving clarity.
// - `float_equality_without_abs` / `neg_cmp_op_on_partial_ord`: optimizer and
//   boundary logic compare against exact values (e.g. θ exactly at the lower
//   bound `0.0`) and use partial-order comparisons deliberately; an `abs()`
//   tolerance or total-order wrapper would change boundary/sentinel semantics.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::float_equality_without_abs)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]

// Re-exported so downstream code can name the exact `nalgebra` this crate
// builds against (it appears in public signatures, e.g. `DMatrix`/`DVector`)
// without pinning its own dependency to our minor version.
pub use nalgebra;

/// Declare one or more methods that belong to the **unstable internal**
/// surface.
///
/// Each wrapped method is emitted as `pub` only when the opt-in
/// `unstable-internals` Cargo feature is enabled; otherwise it is emitted as
/// `pub(crate)`. This is the per-item analogue of the module-level
/// `#[cfg(feature = "unstable-internals")] pub mod / pub(crate) mod` pattern
/// used for `compiler` / `datasets` / `pathology`. Raw numerical solver entry
/// points (`update_l`, `objective_at`, the Kenward-Roger builders, the KKT
/// certificate, …) leak PLS internals and so must not be part of the stable
/// 1.0 surface, but in-tree benches/examples and the `unstable-internals`
/// consumers still need them. See docs/semver_policy.md.
///
/// Apply `unstable-internals`-gated visibility to a method.
///
/// Each invocation re-emits a single method with `pub` when the
/// `unstable-internals` feature is on and `pub(crate)` otherwise, by token
/// substitution on the leading visibility keyword. Write the method with a
/// placeholder `unstable_vis` where the visibility would go:
///
/// ```ignore
/// unstable_internal_method! {
///     /// docs
///     unstable_vis fn foo(&self) -> u32 { 1 }
/// }
/// ```
macro_rules! unstable_internal_method {
    ( $(#[$attr:meta])* unstable_vis $($rest:tt)* ) => {
        #[cfg(feature = "unstable-internals")]
        $(#[$attr])*
        pub $($rest)*

        #[cfg(not(feature = "unstable-internals"))]
        $(#[$attr])*
        pub(crate) $($rest)*
    };
}

// `compiler`, `datasets`, and `pathology` are NOT part of the stable 1.0
// public API (the compiler/IR is still in flux; dataset provenance metadata
// churns). They are public only under the opt-in `unstable-internals`
// feature so that every IR refinement is not a SemVer-major break. Internal
// crate code always reaches them via `pub(crate)`. See
// docs/semver_policy.md and docs/v1_0_release_roadmap.md Phase A.
#[cfg(feature = "unstable-internals")]
pub mod compiler;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod compiler;

#[cfg(feature = "unstable-internals")]
pub mod datasets;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod datasets;

pub mod error;
pub mod formula;
/// Narrative tutorial pages (rendered on docs.rs, doctested in CI).
pub mod guide;
// Numerical primitives (blocked Cholesky, pivoted QR, rank updates). Internal
// to the crate's fit path; not part of the stable public API. Demoted from
// `pub` for v1.0 — see docs/v1_0_release_roadmap.md Phase A.
pub(crate) mod linalg;
pub mod model;
mod optimizer;

#[cfg(feature = "unstable-internals")]
pub mod pathology;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod pathology;

pub mod stats;
pub mod types;

/// Common imports for fitting and inspecting mixed models.
///
/// Glob-importing the prelude pulls in the handful of types needed to
/// build a [`DataFrame`](crate::model::DataFrame), parse a formula, fit a
/// model, and read its estimates:
///
/// ```
/// use mixeff_rs::prelude::*;
///
/// # fn main() -> Result<()> {
/// let mut df = DataFrame::new();
/// df.add_numeric("y", vec![1.0, 2.1, 3.0, 4.2, 5.1, 6.0])?;
/// df.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
/// df.add_categorical(
///     "g",
///     vec!["a", "a", "b", "b", "c", "c"]
///         .into_iter()
///         .map(str::to_string)
///         .collect(),
/// )?;
///
/// let formula = parse_formula("y ~ 1 + x + (1 | g)")?;
/// let mut model = LinearMixedModel::new(formula, &df, None)?;
/// model.fit(false)?;
/// assert_eq!(model.coef().len(), 2);
/// # Ok(())
/// # }
/// ```
pub mod prelude {
    pub use crate::error::{MixedModelError, Result};
    pub use crate::formula::parse_formula;
    pub use crate::model::{
        DataFrame, Family, GeneralizedLinearMixedModel, LinearMixedModel, LinkFunction,
        MixedModelFit,
    };
}
