//! Narrative tutorial for newcomers, in reading order.
//!
//! These pages walk from a first fit through reading results, GLMMs, and the
//! crate's refusal contract. Every code block is compiled and run as a
//! doctest, so the tutorial cannot drift from the API. For the reference
//! surface, see [`crate::model`], [`crate::stats`], and [`crate::formula`].
//!
//! 1. [`getting_started`] — build a frame, parse a formula, fit an LMM.
//! 2. [`reading_results`] — coefficients, variance components, summaries, CIs.
//! 3. [`glmms`] — families, links, and the GLMM estimation semantics.
//! 4. [`when_the_crate_refuses`] — typed errors and typed inference refusals.

#[doc = include_str!("../../docs/guide/01_getting_started.md")]
pub mod getting_started {}

#[doc = include_str!("../../docs/guide/02_reading_results.md")]
pub mod reading_results {}

#[doc = include_str!("../../docs/guide/03_glmms.md")]
pub mod glmms {}

#[doc = include_str!("../../docs/guide/04_when_the_crate_refuses.md")]
pub mod when_the_crate_refuses {}
