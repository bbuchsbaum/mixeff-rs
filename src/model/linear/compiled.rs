//! Compile-once model construction.
//!
//! [`CompiledModelSpec`] is the result of the model compiler's front half —
//! data-boundary transform lowering, `||` factor expansion, the semantic IR
//! and the design audit — held together with the data it was compiled
//! against. A host that must inspect the compiled artifact *before* fitting
//! (to validate the structure, explain the model, or refuse it) compiles
//! once, inspects [`CompiledModelSpec::artifact`], then hands the same spec
//! to [`LinearMixedModel::from_compiled`] or
//! [`GeneralizedLinearMixedModelBuilder::from_compiled`](crate::model::GeneralizedLinearMixedModelBuilder::from_compiled):
//! the model is built from it without a second compile or audit.
//!
//! [`LinearMixedModel::new`] and the GLMM constructors are themselves
//! implemented as "compile, then build from the spec", so the two routes
//! produce the same model by construction.

use std::borrow::Cow;

use nalgebra::DMatrix;

use crate::compiler::{compile_formula_ir, CompiledModelArtifact, CompilerPolicy};
use crate::error::{MixedModelError, Result};
use crate::formula::Formula;
use crate::model::data::DataFrame;

use super::{expand_zerocorr_factor_terms, LinearMixedModel};

/// A model formula compiled and audited against its data, ready to build a
/// model without recompiling.
///
/// Created by [`CompiledModelSpec::compile`] (or
/// [`compile_with_policy`](Self::compile_with_policy)); consumed by
/// [`LinearMixedModel::from_compiled`] and
/// [`GeneralizedLinearMixedModelBuilder::from_compiled`](crate::model::GeneralizedLinearMixedModelBuilder::from_compiled).
///
/// The spec borrows the data it was compiled from; [`into_owned`](Self::into_owned)
/// detaches it (e.g. to keep it alive across host calls). It also retains
/// the dense fixed-effect matrix the design audit factorized (`n × p`), so
/// the model constructor can reuse that factorization exactly as the
/// one-step constructors do.
///
/// ```
/// use mixeff_rs::formula::parse_formula;
/// use mixeff_rs::model::{CompiledModelSpec, DataFrame, FitOptions, LinearMixedModel, MixedModelFit};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = DataFrame::new();
/// df.add_numeric("y", vec![1.0, 2.1, 3.0, 4.2, 5.1, 6.0, 6.8, 8.1])?;
/// df.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
/// df.add_categorical(
///     "g",
///     ["a", "a", "b", "b", "c", "c", "d", "d"].iter().map(|s| s.to_string()).collect(),
/// )?;
///
/// let spec = CompiledModelSpec::compile(parse_formula("y ~ 1 + x + (1 | g)")?, &df)?;
/// // Inspect the pre-fit artifact (design audit, diagnostics) here ...
/// assert!(spec.artifact().design_audit.is_some());
/// // ... then build and fit from the same compilation.
/// let mut model = LinearMixedModel::from_compiled(spec, None)?;
/// model.fit_with_options(FitOptions::reml())?;
/// assert_eq!(model.coef().len(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CompiledModelSpec<'a> {
    /// The formula exactly as supplied (the training recipe's formula).
    pub(crate) source_formula: Formula,
    /// The data exactly as supplied (the training recipe's frame).
    pub(crate) source: Cow<'a, DataFrame>,
    /// `source` plus the lowered in-formula transform columns; `None` when
    /// the formula has no transforms (the source already is the design data).
    pub(crate) materialized: Option<DataFrame>,
    /// The formula after `||` factor expansion, as compiled.
    pub(crate) formula: Formula,
    pub(crate) artifact: CompiledModelArtifact,
    pub(crate) audit_fixed_matrix: DMatrix<f64>,
    pub(crate) audit_fixed_pivot: Vec<usize>,
}

impl<'a> CompiledModelSpec<'a> {
    /// Compile and audit `formula` against `data` under the default
    /// [`CompilerPolicy`].
    ///
    /// Errors when the formula has no random-effects term or an in-formula
    /// transform cannot be evaluated on `data` — the same errors
    /// [`LinearMixedModel::new`] raises before construction.
    pub fn compile(formula: Formula, data: &'a DataFrame) -> Result<Self> {
        Self::compile_with_policy(formula, data, CompilerPolicy::default())
    }

    /// [`compile`](Self::compile) under an explicit compiler policy. The
    /// policy is fixed from here on: it shapes the audit's recommendations
    /// and any design-time reductions the model build applies.
    pub fn compile_with_policy(
        formula: Formula,
        data: &'a DataFrame,
        compiler_policy: CompilerPolicy,
    ) -> Result<Self> {
        if formula.random_terms.is_empty() {
            return Err(MixedModelError::NoRandomEffects);
        }
        // Data-boundary seam: lower the stateless in-formula transforms
        // (`I(days^2)`, `log(reaction)`, …) into synthetic numeric columns
        // before any design construction. See `docs/formula_transform_seam.md`.
        let materialized = match formula.materialize_cow(data)? {
            Cow::Borrowed(_) => None,
            Cow::Owned(frame) => Some(frame),
        };
        let design_data: &DataFrame = materialized.as_ref().unwrap_or(data);

        // lme4-style `||` expansion for terms that contain a factor; needs
        // the data to know which variables are factors.
        let mut expanded = formula.clone();
        expand_zerocorr_factor_terms(&mut expanded, design_data);

        let semantic_model = compile_formula_ir(&expanded);
        let mut artifact = CompiledModelArtifact::new_with_policy(
            expanded.to_string(),
            semantic_model,
            compiler_policy,
        );
        // The audit factorizes the fixed-effect design it builds; keep that
        // matrix so the model's own FeTerm can reuse the factorization when
        // the two designs are the same matrix.
        let (audit_fixed_matrix, audit_fixed_pivot) =
            artifact.attach_design_audit_with_matrix(design_data);

        Ok(Self {
            source_formula: formula,
            source: Cow::Borrowed(data),
            materialized,
            formula: expanded,
            artifact,
            audit_fixed_matrix,
            audit_fixed_pivot,
        })
    }

    /// The compiled, audited artifact (pre-fit). The model built from this
    /// spec starts from exactly this artifact.
    pub fn artifact(&self) -> &CompiledModelArtifact {
        &self.artifact
    }

    /// The formula as compiled (after `||` factor expansion).
    pub fn formula(&self) -> &Formula {
        &self.formula
    }

    /// The design data: the supplied frame plus any lowered in-formula
    /// transform columns.
    pub fn data(&self) -> &DataFrame {
        self.materialized.as_ref().unwrap_or(&self.source)
    }

    /// Number of observations.
    pub fn nrow(&self) -> usize {
        self.data().nrow()
    }

    /// Detach the spec from the borrowed data (clones the frame if it was
    /// borrowed), so it can outlive the caller's `DataFrame`.
    pub fn into_owned(self) -> CompiledModelSpec<'static> {
        CompiledModelSpec {
            source_formula: self.source_formula,
            source: Cow::Owned(self.source.into_owned()),
            materialized: self.materialized,
            formula: self.formula,
            artifact: self.artifact,
            audit_fixed_matrix: self.audit_fixed_matrix,
            audit_fixed_pivot: self.audit_fixed_pivot,
        }
    }
}

/// Which frame a model built from a spec records as its training recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecipeFrame {
    /// The frame as supplied (standalone LMMs: the recipe re-lowers the
    /// transforms on restore).
    Source,
    /// The design data (the GLMM's inner working LMM, which has always been
    /// built from the already-lowered frame).
    Design,
}

impl LinearMixedModel {
    /// Build an (unfitted) LMM from a [`CompiledModelSpec`] without
    /// recompiling or re-auditing.
    ///
    /// The result is the same model [`LinearMixedModel::new`] (or
    /// [`new_with_compiler_policy`](Self::new_with_compiler_policy) with the
    /// spec's policy) builds from the same formula, data and weights: that
    /// constructor is this function applied to a fresh compilation.
    pub fn from_compiled(spec: CompiledModelSpec<'_>, weights: Option<&[f64]>) -> Result<Self> {
        Self::from_compiled_internal(
            spec,
            weights,
            crate::model::fixed_design::FixedDesignBuildPolicy::default(),
            RecipeFrame::Source,
        )
    }
}
