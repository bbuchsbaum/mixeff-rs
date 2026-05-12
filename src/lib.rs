//! Mixed-effects models in Rust — a port of Julia's MixedModels.jl.
//!
//! This crate provides types and algorithms for fitting linear and
//! generalized linear mixed-effects models, ported from the Julia
//! package [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl).
//!
//! # Quick start
//!
//! ```no_run
//! use mixeff_rs::formula::parse_formula;
//! use mixeff_rs::model::{DataFrame, LinearMixedModel};
//!
//! let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
//! // ... build DataFrame, construct model, fit ...
//! ```

pub mod compiler;
pub mod datasets;
pub mod error;
pub mod formula;
pub mod linalg;
pub mod model;
mod optimizer;
pub mod pathology;
pub mod stats;
pub mod types;
