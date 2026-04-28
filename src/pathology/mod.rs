//! Pathology corpus: synthetic mixed-model designs with analytically-derived
//! identifiability certificates.
//!
//! The pathology suite is contract-driven. For each [`GeneratorSpec`] we can
//! derive a [`Certificate`] from linear algebra alone — no engine call —
//! that classifies the design's identifiability and the truth's relation to
//! contract boundaries (zero variance components, unit correlation, reduced
//! rank). [`expected_statuses`] maps the certificate to the *set* of
//! [`crate::compiler::FitStatus`] values any conformant fit engine must produce.
//!
//! Tests assert that the engine's actual status is a member of that set,
//! never an equality against a single value. This is the project's neutral
//! referee for "did the model fit correctly given the design?", independent
//! of any optimizer's idiosyncrasies (lme4 warnings, MixedModels.jl notes).
//!
//! Strata covered by the foundation corpus:
//! - **easy**: fully identified, well-conditioned, far from boundary
//! - **boundary**: truth has σ² = 0 or |ρ| = 1 (contract `ConvergedBoundary`)
//! - **reduced_rank**: rank(Σ_truth) < requested (contract `ConvergedReducedRank`)
//! - **refusal**: design is structurally unidentifiable (contract `NotIdentifiable`)

pub mod certificate;
pub mod separation;
pub mod spec;
pub mod transforms;

pub use certificate::{
    certify, effective_status, effective_status_from_artifact, expected_statuses,
    fisher_correlation_eigvals, map_error_to_status, BoundaryKind, Certificate, CrossedSummary,
    ExpectedStatusSet, SeparationKind, StructuralIssue, WEAK_ID_THRESHOLD,
};
pub use separation::{
    detect_conditional_separation, detect_fe_separation, detect_separation, FeSeparationKind,
    SeparationReport,
};
pub use spec::{generate, CrossedSpec, GeneratorOutput, GeneratorSpec};
pub use transforms::{
    block_diagonal_crossings, collinear_fe, empty_crossings, extreme_prevalence, near_singular_re,
    pareto_sizes, scale_mismatch, set_group_sizes, singletons_with_slope,
};
