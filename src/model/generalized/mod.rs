//! Generalized linear mixed-effects model (GLMM).
//!
//! Fits models of the form:
//!   `g(E[y]) = Xβ + Zb`
//! where g is a link function, b ~ N(0, σ²Λ'Λ), and y|b follows
//! an exponential family distribution.
//!
//! Uses Penalized Iteratively Reweighted Least Squares (PIRLS) for
//! the conditional modes, with optional adaptive Gauss-Hermite quadrature.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use statrs::distribution::{
    ContinuousCDF, DiscreteCDF, Gamma as GammaDist, NegativeBinomial as NegativeBinomialDist,
    Normal, Poisson as PoissonDist,
};
use statrs::function::gamma::ln_gamma;
use std::cell::Cell;
use std::cell::RefCell;
use std::rc::Rc;

use crate::compiler::{
    CompiledModelArtifact, CompilerPolicy, Diagnostic, DiagnosticCode, DiagnosticSeverity,
    DiagnosticStage, EstimabilityAssessment, EvidenceMethod, EvidenceQuality,
    FixedContrastEstimability, FixedEffectCovarianceDetails, FixedEffectCovarianceMatrix,
    FixedEffectCovarianceMethod, FixedEffectCovarianceStatus, FixedEffectInferenceMethod,
    FixedEffectInferenceRow, FixedEffectInferenceRowKind, FixedEffectInferenceStatus,
    FixedEffectInferenceTable, FixedEffectReliabilityReason, GlmmFitMetadata,
    InferenceAvailability, ModelAuditReport, ModelBoundary, ObjectiveApproximation,
    OptimizerCertificate, OptimizerDerivativeEvidence, ReliabilityGrade,
};
use crate::error::{MixedModelError, Result};
use crate::formula::Formula;
use crate::model::data::DataFrame;
use crate::model::linear::{
    prediction_interval_cutoff, CovarianceKktClassification, FitProgressCallback, FitProgressPhase,
    LinearMixedModel, NewReLevels, OptimizerControl, PredictionVarianceMethod,
    PredictionVariancePayload, PredictionVarianceStatus,
};
use crate::model::traits::{Family, LinkFunction, MixedModelFit, RandomEffectTermInfo};
use crate::optimizer::trust_bq::{
    minimize_with_progress as minimize_trust_bq_with_progress, TrustBqOptions, TrustBqProgress,
    TrustBqStopReason,
};
use crate::stats::{BlockDescription, MixedModelProfile, ModelSummary, VarCorr};
use crate::types::{gh_norm, FitLogEntry, MatrixBlock, OptSummary, Optimizer, ReMat};
mod certify;
mod joint;
mod metadata;
mod optimizer;
mod pirls;
mod predictive;
pub(crate) use certify::*;
pub(crate) use joint::*;
pub(crate) use optimizer::*;
pub(crate) use pirls::*;
// The predictive free helpers are only called from within `predictive` itself
// in the library build; `tests.rs` still reaches them through `use super::*`.
#[cfg(test)]
pub(crate) use predictive::*;

/// A generalized linear mixed-effects model.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[allow(dead_code)] // beta0/u0 reserved for step-halving; devc/devc0/sd/mult reserved for AGQ
pub struct GeneralizedLinearMixedModel {
    /// Internal linear mixed model (local Laplace approximation).
    pub(crate) lmm: LinearMixedModel,

    /// Fixed-effects coefficients (pivoted).
    pub beta: DVector<f64>,
    /// Previous β for step-halving.
    beta0: DVector<f64>,

    /// Covariance parameters.
    pub(crate) theta: Vec<f64>,

    /// Random effects on the b-scale: vec(Λ * u) per term.
    pub(crate) b: Vec<DMatrix<f64>>,
    /// Random effects on the u-scale (orthogonal).
    pub(crate) u: Vec<DMatrix<f64>>,
    /// Previous u for step-halving.
    u0: Vec<DMatrix<f64>>,

    /// Linear predictor η = Xβ + Zb.
    pub(crate) eta: DVector<f64>,
    /// Conditional mean μ = g⁻¹(η).
    pub(crate) mu: DVector<f64>,
    /// Response vector.
    pub y: DVector<f64>,
    /// Fixed linear predictor offset, one value per observation.
    pub offset: DVector<f64>,
    /// Prior weights.
    pub wt: Vec<f64>,
    /// Estimated residual scale for dispersion families.
    ///
    /// Stored as sigma, so `dispersion(false)` returns this value and
    /// `dispersion(true)` returns sigma squared. Non-dispersion GLMM families
    /// keep this at 1.
    dispersion: f64,
    /// NB2 size/dispersion parameter for negative-binomial GLMMs.
    ///
    /// This is the `theta` in `Var(Y | b) = mu + mu^2 / theta`. The first
    /// supported modes either fix it from the caller or update it in an outer
    /// glmer.nb-style iteration around the conditional GLMM fit.
    negative_binomial_theta: Option<f64>,
    /// Whether the NB2 theta parameter was estimated by the engine rather
    /// than supplied as a fixed family parameter.
    negative_binomial_estimate_theta: bool,

    /// Distribution family.
    pub family: Family,
    /// Link function.
    pub link: LinkFunction,

    /// Deviance components for AGQ.
    devc: Vec<f64>,
    devc0: Vec<f64>,
    /// Approximate SDs for AGQ.
    sd: Vec<f64>,
    /// Multipliers for AGQ.
    mult: Vec<f64>,

    /// Post-fit certificate that the fast-PIRLS profiled optimum passed
    /// stationarity and curvature quality gates, or the reason it did not.
    /// `None` until a profiled fast-PIRLS fit records its metadata; joint
    /// fits leave this `None` and certify through the joint Hessian instead.
    pirls_profiled_optimum_certificate:
        Option<std::result::Result<PirlsProfiledOptimumCertificate, String>>,

    /// Callback failure captured inside an optimizer API whose objective
    /// callback cannot return `Result`. The driver takes and returns it as soon
    /// as the external optimizer yields control.
    pending_progress_error: Option<String>,
}

/// Options controlling how a GLMM is fit.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GlmmFitOptions {
    /// `true` for the profiled fast-PIRLS path; `false` for labelled joint
    /// Laplace/AGQ.
    pub fast: bool,
    /// Number of adaptive Gauss-Hermite quadrature points. `1` means Laplace.
    pub n_agq: usize,
    /// Whether to emit verbose progress from paths that support it.
    pub verbose: bool,
    /// Optional audit-recorded optimizer controls.
    pub optimizer_control: OptimizerControl,
    /// Optional throttled host progress/interrupt callback.
    pub progress_callback: Option<FitProgressCallback>,
}

/// Scale for GLMM new-data predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GlmmPredictionScale {
    /// Return the linear predictor η = Xβ + Zb (+ offset when supplied).
    Link,
    /// Return the fitted conditional mean μ = g⁻¹(η).
    Response,
}

impl Default for GlmmFitOptions {
    fn default() -> Self {
        Self {
            fast: true,
            n_agq: 1,
            verbose: false,
            optimizer_control: OptimizerControl::default(),
            progress_callback: None,
        }
    }
}

impl GlmmFitOptions {
    /// Default fast-PIRLS Laplace options.
    pub fn fast_laplace() -> Self {
        Self::default()
    }

    /// Labelled joint-Laplace options.
    pub fn joint_laplace() -> Self {
        Self {
            fast: false,
            ..Self::default()
        }
    }

    /// Set the AGQ point count.
    pub fn with_n_agq(mut self, n_agq: usize) -> Self {
        self.n_agq = n_agq;
        self
    }

    /// Set verbose progress reporting.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Attach optimizer controls to these GLMM fit options.
    pub fn with_optimizer_control(mut self, control: OptimizerControl) -> Self {
        self.optimizer_control = control;
        self
    }

    /// Request a specific optimizer while leaving other controls at default.
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer_control = self.optimizer_control.with_optimizer(optimizer);
        self
    }

    /// Attach a host progress/interrupt callback to this fit and later refits.
    pub fn with_progress_callback(mut self, callback: FitProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }
}

/// Fluent builder for [`GeneralizedLinearMixedModel`].
///
/// Collapses the `new` / `new_with_weights` / `new_with_offset` /
/// `new_with_weights_and_offset` / `new_with_compiler_policy` constructor set
/// into one chained surface. Unset options default to the same behavior as
/// plain [`GeneralizedLinearMixedModel::new`].
///
/// ```
/// use mixeff_rs::formula::parse_formula;
/// use mixeff_rs::model::{
///     DataFrame, Family, GeneralizedLinearMixedModelBuilder, MixedModelFit,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut y = Vec::new();
/// let mut x = Vec::new();
/// let mut g = Vec::new();
/// for grp in 0..5 {
///     for obs in 0..8 {
///         let xv = obs as f64 - 3.5;
///         let eta = 0.8 + 0.1 * xv + [-0.2, 0.1, 0.0, 0.15, -0.05][grp];
///         y.push(eta.exp().round().max(1.0));
///         x.push(xv);
///         g.push(format!("g{}", grp + 1));
///     }
/// }
/// let mut df = DataFrame::new();
/// df.add_numeric("y", y)?;
/// df.add_numeric("x", x)?;
/// df.add_categorical("g", g)?;
///
/// let model = GeneralizedLinearMixedModelBuilder::new(
///     parse_formula("y ~ 1 + x + (1 | g)")?,
///     &df,
///     Family::Poisson,
/// )
/// .fit()?;
/// assert_eq!(model.coef().len(), 2);
/// # Ok(())
/// # }
/// ```
pub struct GeneralizedLinearMixedModelBuilder<'a> {
    formula: Formula,
    data: &'a DataFrame,
    family: Family,
    link: Option<LinkFunction>,
    negative_binomial_theta: Option<f64>,
    negative_binomial_estimate_theta: bool,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    compiler_policy: Option<CompilerPolicy>,
}

impl<'a> GeneralizedLinearMixedModelBuilder<'a> {
    /// Start a builder for `formula` over `data` with the given `family`.
    pub fn new(formula: Formula, data: &'a DataFrame, family: Family) -> Self {
        Self {
            formula,
            data,
            family,
            link: None,
            negative_binomial_theta: None,
            negative_binomial_estimate_theta: false,
            weights: None,
            offset: None,
            compiler_policy: None,
        }
    }

    /// Override the link (defaults to the family's canonical link).
    pub fn link(mut self, link: LinkFunction) -> Self {
        self.link = Some(link);
        self
    }

    /// Supply a fixed NB2 size/dispersion parameter for
    /// [`Family::NegativeBinomial`].
    ///
    /// This enables the smaller `MASS::negative.binomial(theta)`-style mode:
    /// the model is estimated conditional on `theta`, and no outer theta
    /// profiling is attempted.
    pub fn negative_binomial_theta(mut self, theta: f64) -> Self {
        self.negative_binomial_theta = Some(theta);
        self.negative_binomial_estimate_theta = false;
        self
    }

    /// Request glmer.nb-style estimation of the NB2 size parameter.
    ///
    /// `start_theta` is optional; when omitted, the constructor derives a
    /// method-of-moments start from the response.
    pub fn estimate_negative_binomial_theta(mut self, start_theta: Option<f64>) -> Self {
        self.negative_binomial_theta = start_theta;
        self.negative_binomial_estimate_theta = true;
        self
    }

    /// Attach per-observation case weights (e.g. binomial trial counts).
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Attach a fixed per-observation linear-predictor offset.
    pub fn offset(mut self, offset: Vec<f64>) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Attach a compiler policy applied to the internal compiled artifact.
    pub fn compiler_policy(mut self, compiler_policy: CompilerPolicy) -> Self {
        self.compiler_policy = Some(compiler_policy);
        self
    }

    /// Construct the (unfitted) model.
    pub fn build(self) -> Result<GeneralizedLinearMixedModel> {
        let policy = self.compiler_policy.unwrap_or_default();
        let mut model = GeneralizedLinearMixedModel::new_with_policy_internal(
            self.formula,
            self.data,
            self.family,
            self.link,
            self.negative_binomial_theta,
            self.negative_binomial_estimate_theta,
            policy,
        )?;
        if let Some(offset) = self.offset {
            model.set_offset(offset)?;
        }
        if let Some(weights) = self.weights {
            validate_case_weights(&weights, model.y.len())?;
            model.wt = weights;
            model.initialize_beta_from_response();
        }
        Ok(model)
    }

    /// Construct and fit the model in one step.
    pub fn fit(self) -> Result<GeneralizedLinearMixedModel> {
        let mut model = self.build()?;
        model.fit()?;
        Ok(model)
    }

    /// Construct and fit the model in one step with explicit GLMM options.
    pub fn fit_with_glmm_options(
        self,
        options: GlmmFitOptions,
    ) -> Result<GeneralizedLinearMixedModel> {
        let mut model = self.build()?;
        model.fit_with_glmm_options(options)?;
        Ok(model)
    }
}

impl GeneralizedLinearMixedModel {
    fn fixed_effect_inference_standard_errors(&self) -> Option<DVector<f64>> {
        let table = self
            .lmm
            .compiler_artifact
            .fixed_effect_inference_table
            .as_ref()?;
        let names = self.coef_names();
        let rows = table
            .rows
            .iter()
            .filter(|row| row.kind == FixedEffectInferenceRowKind::Coefficient)
            .collect::<Vec<_>>();
        if rows.len() != names.len() {
            return None;
        }

        let mut standard_errors = Vec::with_capacity(names.len());
        for name in names {
            let row = rows.iter().find(|row| row.label == name)?;
            standard_errors.push(
                row.std_error
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .unwrap_or(f64::NAN),
            );
        }
        Some(DVector::from_vec(standard_errors))
    }

    /// Standard errors recorded on a parametric-bootstrap replicate refit.
    ///
    /// Replicate SEs are descriptive resampling payloads, not certified Wald
    /// inference: downstream summaries use their finiteness to separate
    /// successful refits from failed ones. When the refit carries a fully
    /// certified Wald table its SEs are recorded; otherwise (fast-PIRLS
    /// fits, refused or partially refused tables) the working-covariance
    /// standard errors are recorded instead of NaN refusals.
    pub(crate) fn bootstrap_replicate_standard_errors(&self) -> DVector<f64> {
        match self.fixed_effect_inference_standard_errors() {
            Some(se) if se.iter().all(|value| value.is_finite()) => se,
            _ => self.lmm.stderror(),
        }
    }

    unstable_internal_method! {
    /// Inner [`LinearMixedModel`] holding the local Laplace approximation
    /// (raw PIRLS solver state).
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Not part of the stable 1.0 API; the
    /// θ vector is available through [`MixedModelFit::theta`].
    #[allow(dead_code)]
    unstable_vis fn lmm(&self) -> &LinearMixedModel {
        &self.lmm
    }
    }

    unstable_internal_method! {
    /// Mutable inner [`LinearMixedModel`] (raw PIRLS solver state).
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Not part of the stable 1.0 API.
    #[allow(dead_code)]
    unstable_vis fn lmm_mut(&mut self) -> &mut LinearMixedModel {
        &mut self.lmm
    }
    }

    unstable_internal_method! {
    /// Profiled (beta-varying PIRLS) deviance with response constants at an
    /// arbitrary `theta`, for optimizer diagnostics: lets probes evaluate the
    /// fast-PIRLS objective at an externally supplied covariance candidate
    /// (e.g. a reference fit's theta) without running an optimizer.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Not part of the stable 1.0 API.
    #[allow(dead_code)]
    unstable_vis fn profiled_deviance_at_theta(
        &mut self,
        theta: &[f64],
        n_agq: usize,
    ) -> Result<f64> {
        self.update_pirls_at_theta(theta, true)?;
        Ok(self.deviance_with_response_constants(n_agq))
    }
    }

    unstable_internal_method! {
    /// [`profiled_deviance_at_theta`](Self::profiled_deviance_at_theta) with
    /// an explicit PIRLS iteration budget, for diagnosing objective
    /// distortion from the default per-evaluation PIRLS cap.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Not part of the stable 1.0 API.
    #[allow(dead_code)]
    unstable_vis fn profiled_deviance_at_theta_with_pirls_budget(
        &mut self,
        theta: &[f64],
        n_agq: usize,
        max_iter: usize,
    ) -> Result<f64> {
        self.update_pirls_at_theta_with_options(theta, true, max_iter, true)?;
        Ok(self.deviance_with_response_constants(n_agq))
    }
    }

    unstable_internal_method! {
    /// Joint-Laplace deviance at `theta` with the fast-PIRLS profiled beta:
    /// first solves beta jointly inside PIRLS at `theta`, then re-evaluates
    /// the u-only Laplace criterion at that (beta, theta) — the objective the
    /// joint optimizer and glmer's nAGQ=1 deviance use. The beta-varying
    /// PIRLS criterion folds the fixed-effects block into the log-determinant
    /// and is NOT comparable to glmer's reported logLik.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Not part of the stable 1.0 API.
    #[allow(dead_code)]
    unstable_vis fn joint_deviance_at_theta_with_profiled_beta(
        &mut self,
        theta: &[f64],
        n_agq: usize,
    ) -> Result<f64> {
        self.update_pirls_at_theta(theta, true)?;
        let n_beta = self.beta.len();
        let mut params = self.beta.as_slice().to_vec();
        params.extend_from_slice(theta);
        Ok(self.joint_glmm_deviance_at_params(&params, n_beta, n_agq))
    }
    }

    unstable_internal_method! {
    /// Joint-Laplace fit seeded at an arbitrary `start_theta` (beta seeded at
    /// the fast-PIRLS profiled solution for that theta), bypassing the usual
    /// profiled-fit start. Optimizer-diagnostics surface: lets probes test
    /// whether a reference optimum is reachable when the search starts beside
    /// it instead of at the profiled fit.
    ///
    /// Unstable internal surface: `pub` only with the `unstable-internals`
    /// feature; otherwise `pub(crate)`. Not part of the stable 1.0 API.
    #[allow(dead_code)]
    unstable_vis fn fit_joint_glmm_from_custom_theta(
        &mut self,
        start_theta: &[f64],
        n_agq: usize,
    ) -> Result<&mut Self> {
        if self.lmm.optsum.feval > 0 {
            return Err(MixedModelError::AlreadyFitted);
        }
        self.validate_agq(n_agq)?;
        self.update_pirls_at_theta(start_theta, true)?;
        let start_beta = self.beta.as_slice().to_vec();
        let n_beta = start_beta.len();
        let mut params = start_beta.clone();
        params.extend_from_slice(start_theta);
        let start_objective = self.joint_glmm_deviance_at_params(&params, n_beta, n_agq);
        let maxeval = joint_glmm_configured_maxeval_for(
            &self.lmm.optsum,
            params.len(),
            Optimizer::TrustBq,
        );
        self.lmm.optsum.optimizer = Optimizer::TrustBq;
        self.lmm.optsum.backend = Optimizer::TrustBq.canonical_backend();
        self.fit_joint_glmm_from_start(
            start_beta,
            start_theta.to_vec(),
            start_objective,
            n_agq,
            maxeval,
            None,
        )
    }
    }

    /// Construct a GLMM from formula, data, distribution, and link.
    pub fn new(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
    ) -> Result<Self> {
        Self::new_with_policy_internal(
            formula,
            data,
            family,
            link,
            None,
            false,
            CompilerPolicy::default(),
        )
    }

    /// Construct a negative-binomial NB2 GLMM with a fixed size parameter.
    ///
    /// `theta` must be positive and finite. It is not estimated by this
    /// constructor; the fit is conditional on the supplied value, matching the
    /// smaller `MASS::negative.binomial(theta)` route. The default link is log.
    pub fn new_negative_binomial(
        formula: Formula,
        data: &DataFrame,
        theta: f64,
        link: Option<LinkFunction>,
    ) -> Result<Self> {
        Self::new_with_policy_internal(
            formula,
            data,
            Family::NegativeBinomial,
            link,
            Some(theta),
            false,
            CompilerPolicy::default(),
        )
    }

    /// Construct a negative-binomial NB2 GLMM that estimates its size parameter.
    ///
    /// This is the engine-side analogue of `glmer.nb`: `start_theta`, when
    /// supplied, seeds the outer theta iteration; otherwise a response-moment
    /// start is used. The default link is log.
    pub fn new_negative_binomial_estimated(
        formula: Formula,
        data: &DataFrame,
        start_theta: Option<f64>,
        link: Option<LinkFunction>,
    ) -> Result<Self> {
        Self::new_with_policy_internal(
            formula,
            data,
            Family::NegativeBinomial,
            link,
            start_theta,
            true,
            CompilerPolicy::default(),
        )
    }

    /// Construct a GLMM with per-observation case weights.
    ///
    /// Used for binomial-with-trials data where the response is a proportion
    /// `y / n` and `weights[i] = n_i` is the trial count. Mirrors Julia's
    /// `fit(MixedModel, formula, data, Binomial(); wts = ...)`.
    ///
    /// `weights.len()` must equal the number of observations. The vector is
    /// stored as `self.wt` and incorporated into IRLS weights and deviance
    /// residuals during PIRLS.
    pub fn new_with_weights(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
        weights: Vec<f64>,
    ) -> Result<Self> {
        let mut model = Self::new_with_policy_internal(
            formula,
            data,
            family,
            link,
            None,
            false,
            CompilerPolicy::default(),
        )?;
        validate_case_weights(&weights, model.y.len())?;
        model.wt = weights;
        model.initialize_beta_from_response();
        Ok(model)
    }

    /// Construct a GLMM with a fixed per-observation linear predictor offset.
    ///
    /// The offset is not estimated. It enters the linear predictor as
    /// `eta = offset + X beta + Z b` and is subtracted from the PIRLS working
    /// response before solving the internal LMM approximation.
    pub fn new_with_offset(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
        offset: Vec<f64>,
    ) -> Result<Self> {
        let mut model = Self::new_with_policy_internal(
            formula,
            data,
            family,
            link,
            None,
            false,
            CompilerPolicy::default(),
        )?;
        model.set_offset(offset)?;
        Ok(model)
    }

    /// Construct a GLMM with both case weights and a fixed linear predictor offset.
    pub fn new_with_weights_and_offset(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
        weights: Vec<f64>,
        offset: Vec<f64>,
    ) -> Result<Self> {
        let mut model = Self::new_with_offset(formula, data, family, link, offset)?;
        validate_case_weights(&weights, model.y.len())?;
        model.wt = weights;
        model.initialize_beta_from_response();
        Ok(model)
    }

    fn new_with_policy_internal(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
        negative_binomial_theta: Option<f64>,
        negative_binomial_estimate_theta: bool,
        compiler_policy: CompilerPolicy,
    ) -> Result<Self> {
        let link = link.unwrap_or_else(|| family.canonical_link());

        // For Normal + Identity, redirect to LMM
        if family == Family::Normal && link == LinkFunction::Identity {
            return Err(MixedModelError::InvalidArgument(
                "Use LinearMixedModel for Normal distribution with IdentityLink".to_string(),
            ));
        }
        validate_supported_glmm_family_link(family, link)?;
        validate_negative_binomial_theta_request(
            family,
            negative_binomial_theta,
            negative_binomial_estimate_theta,
        )?;

        // Data-boundary seam: lower the stateless in-formula transforms into
        // synthetic numeric columns here too, so the GLMM response-domain
        // check below sees a transformed response (e.g. `log(reaction)`).
        // The inner LMM build materializes again but the columns already
        // exist and are verified to match, so it is a no-op there. See
        // `docs/formula_transform_seam.md`.
        //
        // NOTE FOR FUTURE MAINTAINERS: GLMM currently has no public
        // `predict_new` path of its own — prediction delegates to the inner
        // LMM, which already calls `formula.materialize(newdata)` at its
        // own data boundary. If a direct GLMM `predict_new` is ever added
        // it MUST call `self.lmm.formula.materialize(newdata)` before
        // building the fixed-effects matrix; bypassing the seam would
        // silently omit transform re-evaluation on newdata.
        let materialized = formula.materialize(data)?;
        let data = &materialized;

        if let Some(y) = data.numeric(&formula.response) {
            validate_glmm_response_domain(family, link, y)?;
            if let Some(&first) = y.first() {
                if y.iter().all(|&value| value == first) {
                    return Err(MixedModelError::InvalidArgument(
                        "response is constant; GLMM construction requires variation in the response"
                            .to_string(),
                    ));
                }
            }
        }
        let negative_binomial_theta = initialize_negative_binomial_theta(
            family,
            negative_binomial_theta,
            negative_binomial_estimate_theta,
            data.numeric(&formula.response),
        )?;

        // Build the internal LMM
        let mut lmm =
            LinearMixedModel::new_with_compiler_policy(formula, data, None, compiler_policy)?;
        lmm.compiler_artifact
            .set_model_boundary(ModelBoundary::glmm(
                family_label(family),
                link_label(link),
                ObjectiveApproximation::Laplace {
                    inner: "pirls".to_string(),
                },
            ));
        let n = lmm.dims.n;
        let p = lmm.dims.p;

        let beta = DVector::zeros(p.min(lmm.feterm.rank));
        let theta = lmm.theta();
        let y = lmm.y();

        let u: Vec<DMatrix<f64>> = lmm
            .reterms
            .iter()
            .map(|rt| DMatrix::zeros(rt.vsize, rt.n_levels()))
            .collect();

        let b = u.clone();
        let u0 = u
            .iter()
            .map(|m| DMatrix::zeros(m.nrows(), m.ncols()))
            .collect();

        let eta = DVector::zeros(n);
        let mu = DVector::zeros(n);
        let offset = DVector::zeros(n);

        // AGQ vectors — only used for single RE term
        let agq_len = if u.len() == 1 {
            u[0].nrows() * u[0].ncols()
        } else {
            0
        };

        let mut model = GeneralizedLinearMixedModel {
            lmm,
            beta: beta.clone(),
            beta0: beta,
            theta,
            b,
            u,
            u0,
            eta,
            mu,
            y,
            offset,
            wt: Vec::new(),
            dispersion: negative_binomial_theta.unwrap_or(1.0),
            negative_binomial_theta,
            negative_binomial_estimate_theta,
            family,
            link,
            devc: vec![0.0; agq_len],
            devc0: vec![0.0; agq_len],
            sd: vec![0.0; agq_len],
            mult: vec![0.0; agq_len],
            pirls_profiled_optimum_certificate: None,
            pending_progress_error: None,
        };
        model.initialize_beta_from_response();
        Ok(model)
    }

    /// Construct a GLMM and apply a compiler policy to the internal compiled
    /// artifact before fitting.
    pub fn new_with_compiler_policy(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
        compiler_policy: CompilerPolicy,
    ) -> Result<Self> {
        Self::new_with_policy_internal(formula, data, family, link, None, false, compiler_policy)
    }

    /// Fixed NB2 theta used by a negative-binomial GLMM, when applicable.
    pub fn negative_binomial_theta(&self) -> Option<f64> {
        self.negative_binomial_theta
    }

    /// Whether this negative-binomial GLMM estimates the NB2 theta parameter.
    pub fn negative_binomial_theta_estimated(&self) -> bool {
        self.negative_binomial_estimate_theta
    }

    fn require_negative_binomial_theta(&self) -> Result<f64> {
        self.negative_binomial_theta.ok_or_else(|| {
            MixedModelError::InvalidArgument(
                "negative-binomial GLMM requires a positive fixed theta".to_string(),
            )
        })
    }

    pub(crate) fn random_effect_scale(&self) -> f64 {
        if self.family.has_dispersion() {
            self.dispersion(false)
        } else {
            1.0
        }
    }

    fn variance(&self, mu: f64) -> f64 {
        glmm_variance(self.family, mu, self.negative_binomial_theta)
    }

    /// Round-trippable compiler artifact attached to the internal model.
    pub fn compiler_artifact(&self) -> &CompiledModelArtifact {
        self.lmm.compiler_artifact()
    }

    /// Compiler policy attached to the internal compiled artifact.
    pub fn compiler_policy(&self) -> &CompilerPolicy {
        self.lmm.compiler_policy()
    }

    /// Apply a compiler policy before fitting.
    pub fn set_compiler_policy(&mut self, compiler_policy: CompilerPolicy) -> Result<&mut Self> {
        self.lmm.set_compiler_policy(compiler_policy)?;
        Ok(self)
    }

    /// Return a copy of this model with a compiler policy applied.
    pub fn with_compiler_policy(mut self, compiler_policy: CompilerPolicy) -> Result<Self> {
        self.set_compiler_policy(compiler_policy)?;
        Ok(self)
    }

    /// Stable user-facing audit report derived from the compiler artifact.
    pub fn audit_report(&self) -> ModelAuditReport {
        self.lmm.audit_report()
    }

    /// Explicit refusal for GLMM residual-scale profile likelihood.
    ///
    /// Profile likelihood is currently an LMM-only engine feature. This method
    /// gives downstream bindings a stable runtime reason instead of requiring
    /// them to infer support from the absence of a GLMM profile API.
    pub fn profile_sigma(&mut self, _threshold: f64) -> Result<MixedModelProfile> {
        Err(MixedModelError::Unsupported(
            glmm_profile_likelihood_unsupported_reason("profile_sigma"),
        ))
    }

    /// Explicit refusal for GLMM covariance-parameter profile likelihood.
    ///
    /// See [`Self::profile_sigma`] for the support boundary.
    pub fn profile_theta(&mut self, _index: usize, _threshold: f64) -> Result<MixedModelProfile> {
        Err(MixedModelError::Unsupported(
            glmm_profile_likelihood_unsupported_reason("profile_theta"),
        ))
    }

    /// Compact default print summary (PRD § 15).
    pub fn print_summary(&self) -> crate::compiler::ModelPrint {
        self.lmm.print_summary()
    }

    /// Source-to-fitted parameterization drilldown (PRD § 15).
    pub fn parameterization(&self) -> crate::compiler::ParameterizationDrilldown {
        self.lmm.parameterization()
    }

    /// Replace the fixed linear predictor offset before fitting.
    pub fn set_offset(&mut self, offset: Vec<f64>) -> Result<&mut Self> {
        if self.is_fitted() {
            return Err(MixedModelError::AlreadyFitted);
        }
        validate_offset(&offset, self.y.len())?;
        self.offset = DVector::from_vec(offset);
        self.update_eta();
        Ok(self)
    }

    fn initialize_beta_from_response(&mut self) {
        self.beta.fill(0.0);
        let Some(intercept_index) = self
            .lmm
            .feterm
            .cnames
            .iter()
            .take(self.lmm.feterm.rank)
            .position(|name| is_intercept_column(name))
        else {
            self.beta0 = self.beta.clone();
            self.update_eta();
            return;
        };
        let Some(mean) = initial_response_mean(self.family, &self.y, &self.wt) else {
            self.beta0 = self.beta.clone();
            self.update_eta();
            return;
        };
        let eta = self.link.link(initial_mean_for_link(self.family, mean));
        let offset_mean = if self.offset.is_empty() {
            0.0
        } else {
            self.offset.iter().sum::<f64>() / self.offset.len() as f64
        };
        self.beta[intercept_index] = eta - offset_mean;
        self.beta0 = self.beta.clone();
        self.update_eta();
    }

    /// Update the linear predictor η and conditional mean μ.
    pub fn update_eta(&mut self) {
        let n = self.eta.len();
        let x = self.lmm.feterm.full_rank_x();

        // η = offset + X * β
        self.eta = &self.offset + x * &self.beta;

        // Add random effects: η += Z_i * b_i
        for (i, rt) in self.lmm.reterms.iter().enumerate() {
            // b_i = λ_i * u_i
            self.b[i] = &rt.lambda * &self.u[i];
            // Multiply Z * vec(b) using refs for sparse multiplication
            let bvec = DVector::from_column_slice(self.b[i].as_slice());
            for (obs, &ref_idx) in rt.refs.iter().enumerate() {
                let r = ref_idx as usize;
                for s in 0..rt.vsize {
                    self.eta[obs] += rt.z[(s, obs)] * bvec[r * rt.vsize + s];
                }
            }
        }

        // μ = g⁻¹(η)
        for i in 0..n {
            self.mu[i] = self.link.linkinv(self.eta[i]);
        }
    }

    /// Variance-covariance summary for the fitted random effects.
    pub fn varcorr(&self) -> VarCorr {
        let scale = self.random_effect_scale();
        let residual_sd = if self.family.has_dispersion() {
            Some(scale)
        } else {
            None
        };
        VarCorr::from_reterms(&self.lmm.reterms, scale, residual_sd)
            .with_residual_source(self.lmm.residual_source)
    }

    /// Structural summary of the blocked `A`/`L` system.
    pub fn block_description(&self) -> BlockDescription {
        BlockDescription::from_generalized_model(self)
    }

    /// Fixed/random-effects summary table.
    pub fn summary(&self) -> ModelSummary {
        ModelSummary::from_generalized_model(self)
    }

    /// Render the model summary as markdown.
    pub fn summary_markdown(&self) -> String {
        self.summary().to_markdown()
    }

    /// Render the model summary as HTML.
    pub fn summary_html(&self) -> String {
        self.summary().to_html()
    }

    /// Render the model summary as LaTeX.
    pub fn summary_latex(&self) -> String {
        self.summary().to_latex()
    }

    /// Compute the deviance residuals for the current μ.
    #[allow(dead_code)] // exposed once GLMM diagnostics surface lands
    fn deviance_residuals(&self) -> DVector<f64> {
        let n = self.y.len();
        let mut devresid = DVector::zeros(n);
        for i in 0..n {
            devresid[i] = self.dev_resid_component(self.y[i], self.mu[i]);
        }
        devresid
    }

    /// Single observation deviance residual component.
    fn dev_resid_component(&self, y: f64, mu: f64) -> f64 {
        match self.family {
            Family::Bernoulli | Family::Binomial => {
                let eps = 1e-15;
                let mu = mu.max(eps).min(1.0 - eps);
                if y == 1.0 {
                    -2.0 * mu.ln()
                } else if y == 0.0 {
                    -2.0 * (1.0 - mu).ln()
                } else {
                    2.0 * (y * (y / mu).ln() + (1.0 - y) * ((1.0 - y) / (1.0 - mu)).ln())
                }
            }
            Family::Poisson => {
                if y == 0.0 {
                    2.0 * mu
                } else {
                    2.0 * (y * (y / mu).ln() - (y - mu))
                }
            }
            Family::NegativeBinomial => negative_binomial_deviance_component(
                y,
                mu,
                self.negative_binomial_theta
                    .expect("negative-binomial GLMM stores fixed theta"),
            ),
            Family::Normal => (y - mu).powi(2),
            Family::Gamma => {
                // Gamma requires μ > 0, but an inverse-link Gamma GLMM can
                // transiently propose η giving μ ≤ 0 during PIRLS. Unfloored,
                // `(y/μ).ln()` / `(y-μ)/μ` would yield NaN/±Inf from valid
                // data; that NaN then slips the `obj > halving_bound`
                // step-halving guard (`NaN > x` is false) and is silently
                // accepted. Floor μ to the same positive-mean ε (1e-6) this
                // module already uses for Gamma/Poisson/InverseGaussian
                // means: valid μ (O(y), far above 1e-6) is unaffected, while
                // a degenerate μ≤0 yields a finite, very large deviance that
                // step-halving correctly rejects as "worse". A tighter floor
                // (e.g. f64::MIN_POSITIVE) still overflows: `(y-μ)/μ` reaches
                // ~1e308 and doubles to +Inf.
                let mu = mu.max(1e-6);
                if y == 0.0 {
                    2.0 * (mu.ln())
                } else {
                    -2.0 * ((y / mu).ln() - (y - mu) / mu)
                }
            }
            Family::InverseGaussian => {
                // Same μ>0 requirement: μ=0 would divide by zero. Floor at
                // the same 1e-6 positive-mean ε so a transient degenerate
                // iterate is finite-but-rejected, not NaN/Inf silently
                // accepted.
                let mu = mu.max(1e-6);
                (y - mu).powi(2) / (y * mu * mu)
            }
        }
    }

    /// Reject `n_agq > 1` on models that don't satisfy the AGQ contract
    /// (single scalar RE term).
    pub fn validate_agq(&self, n_agq: usize) -> Result<()> {
        if n_agq > 1 && !self.is_single_scalar_re() {
            let sizes: Vec<usize> = self.lmm.reterms.iter().map(|rt| rt.vsize).collect();
            return Err(MixedModelError::InvalidArgument(format!(
                "n_agq = {n_agq} > 1 requires exactly one scalar random-effects term; \
                 this model has {} term(s) with vsizes {:?}",
                self.lmm.reterms.len(),
                sizes,
            )));
        }
        Ok(())
    }

    fn pirls_profiled_fd_gradient_component(
        &mut self,
        theta: &[f64],
        index: usize,
        h: f64,
        n_agq: usize,
        lower_bounds: &[f64],
    ) -> f64 {
        let value = theta[index];
        let lower = lower_bounds
            .get(index)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let mut plus = theta.to_vec();
        plus[index] = value + h;
        let fp = self.penalized_pirls_deviance_at_theta(&plus, n_agq);
        if value - h > lower {
            let mut minus = theta.to_vec();
            minus[index] = value - h;
            let fm = self.penalized_pirls_deviance_at_theta(&minus, n_agq);
            (fp - fm) / (2.0 * h)
        } else {
            let base = self.penalized_pirls_deviance_at_theta(theta, n_agq);
            (fp - base) / h
        }
    }

    /// Stationarity gradient of the profiled fast-PIRLS objective over theta,
    /// with the same PIRLS-noise-aware step escalation as the joint-Laplace
    /// certificate (see [`Self::joint_laplace_certification_gradient`]).
    fn pirls_profiled_certification_gradient(
        &mut self,
        theta: &[f64],
        n_agq: usize,
        lower_bounds: &[f64],
        gradient_tolerance: f64,
    ) -> JointLaplaceCertificationGradient {
        let probe_gradient = (0..theta.len())
            .map(|index| {
                let h = JOINT_LAPLACE_FD_RELATIVE_STEP * theta[index].abs().max(1.0);
                self.pirls_profiled_fd_gradient_component(theta, index, h, n_agq, lower_bounds)
            })
            .collect::<Vec<_>>();
        let mut gradient = probe_gradient.clone();
        let mut escalated_indices = Vec::new();
        let mut unassessable_indices = Vec::new();
        for (index, &value) in theta.iter().enumerate() {
            let raw = probe_gradient[index];
            if raw.is_finite() && raw.abs() <= gradient_tolerance {
                continue;
            }
            let scale = value.abs().max(1.0);
            let estimates = JOINT_LAPLACE_CERT_FD_ESCALATED_RELATIVE_STEPS.map(|step| {
                self.pirls_profiled_fd_gradient_component(
                    theta,
                    index,
                    step * scale,
                    n_agq,
                    lower_bounds,
                )
            });
            let consistent = estimates.iter().all(|estimate| estimate.is_finite())
                && (estimates[0] - estimates[1]).abs() <= gradient_tolerance;
            if consistent {
                gradient[index] = estimates[1];
                escalated_indices.push(index);
            } else {
                unassessable_indices.push(index);
            }
        }
        JointLaplaceCertificationGradient {
            gradient,
            probe_gradient,
            escalated_indices,
            unassessable_indices,
        }
    }

    /// Central-difference Hessian of the profiled fast-PIRLS objective over
    /// the interior (non-boundary) theta coordinates.
    fn finite_difference_pirls_profiled_hessian(
        &mut self,
        theta: &[f64],
        active_indices: &[usize],
        n_agq: usize,
    ) -> std::result::Result<DMatrix<f64>, String> {
        let n = active_indices.len();

        macro_rules! eval_profiled_probe {
            ($probe:expr, $context:expr) => {{
                let value = self.penalized_pirls_deviance_at_theta($probe, n_agq);
                if value.is_finite() {
                    value
                } else {
                    return Err(format!("{} is non-finite", $context));
                }
            }};
        }

        let base = eval_profiled_probe!(theta, "profiled fast-PIRLS Hessian base objective");
        let steps = active_indices
            .iter()
            .map(|&index| glmm_hessian_step(theta[index]))
            .collect::<Vec<_>>();

        let mut hessian = DMatrix::zeros(n, n);
        for active_i in 0..n {
            let i = active_indices[active_i];
            let hi = steps[active_i];
            let mut plus = theta.to_vec();
            plus[i] += hi;
            let f_plus = eval_profiled_probe!(
                &plus,
                format!(
                    "profiled fast-PIRLS Hessian diagonal plus probe for covariance parameter {}",
                    i + 1
                )
            );
            let mut minus = theta.to_vec();
            minus[i] -= hi;
            let f_minus = eval_profiled_probe!(
                &minus,
                format!(
                    "profiled fast-PIRLS Hessian diagonal minus probe for covariance parameter {}",
                    i + 1
                )
            );
            hessian[(active_i, active_i)] = (f_plus - 2.0 * base + f_minus) / (hi * hi);

            for active_j in 0..active_i {
                let j = active_indices[active_j];
                let hj = steps[active_j];
                let mut pp = theta.to_vec();
                pp[i] += hi;
                pp[j] += hj;
                let f_pp = eval_profiled_probe!(
                    &pp,
                    format!(
                        "profiled fast-PIRLS Hessian off-diagonal ++ probe for covariance parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );
                let mut pm = theta.to_vec();
                pm[i] += hi;
                pm[j] -= hj;
                let f_pm = eval_profiled_probe!(
                    &pm,
                    format!(
                        "profiled fast-PIRLS Hessian off-diagonal +- probe for covariance parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );
                let mut mp = theta.to_vec();
                mp[i] -= hi;
                mp[j] += hj;
                let f_mp = eval_profiled_probe!(
                    &mp,
                    format!(
                        "profiled fast-PIRLS Hessian off-diagonal -+ probe for covariance parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );
                let mut mm = theta.to_vec();
                mm[i] -= hi;
                mm[j] -= hj;
                let f_mm = eval_profiled_probe!(
                    &mm,
                    format!(
                        "profiled fast-PIRLS Hessian off-diagonal -- probe for covariance parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );
                let value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hi * hj);
                hessian[(active_i, active_j)] = value;
                hessian[(active_j, active_i)] = value;
            }
        }
        Ok(hessian)
    }

    /// Certify the fast-PIRLS profiled optimum after a profiled fit.
    ///
    /// The profiled objective minimizes the Laplace (or AGQ) deviance over
    /// theta with beta solved exactly by the penalized least-squares step at
    /// every theta, so a stationarity-plus-curvature certificate over theta
    /// is a complete optimum certificate for this estimator's own objective:
    /// the beta directions are exactly minimized by construction. The
    /// certificate gates are the joint-Laplace ones — noise-aware gradient
    /// tolerance for interior coordinates, a one-sided gradient condition for
    /// boundary coordinates, and positive-definiteness/conditioning of the
    /// interior-theta Hessian.
    fn certify_pirls_profiled_optimum(
        &mut self,
    ) -> std::result::Result<PirlsProfiledOptimumCertificate, String> {
        let theta = self.lmm.optsum.final_params.clone();
        if theta.len() != self.theta.len() {
            return Err(format!(
                "profiled fast-PIRLS final parameter vector has length {}, expected {} covariance parameters",
                theta.len(),
                self.theta.len()
            ));
        }
        if theta.len() > PIRLS_PROFILED_CERTIFICATE_MAX_THETA {
            return Err(format!(
                "profiled-optimum certificate skipped: {} covariance parameters exceed the certification budget of {PIRLS_PROFILED_CERTIFICATE_MAX_THETA}",
                theta.len()
            ));
        }
        let n_agq = self.lmm.optsum.n_agq.max(1);
        let lower_bounds = self.lmm.lower_bounds();
        let outcome = self.certify_pirls_profiled_optimum_probes(&theta, n_agq, &lower_bounds);
        // Probes left PIRLS state at an off-optimum theta; restore the fitted
        // state exactly the way finalize_theta_after_optimizer leaves it.
        let _ = self.penalized_pirls_deviance_at_theta(&theta, n_agq);
        self.beta = self.lmm.beta();
        self.refresh_dispersion();
        outcome
    }

    fn certify_pirls_profiled_optimum_probes(
        &mut self,
        theta: &[f64],
        n_agq: usize,
        lower_bounds: &[f64],
    ) -> std::result::Result<PirlsProfiledOptimumCertificate, String> {
        let mut boundary_theta_indices = Vec::new();
        let mut interior_indices = Vec::new();
        for (index, &value) in theta.iter().enumerate() {
            let lower = lower_bounds
                .get(index)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            if lower.is_finite() && value <= lower + glmm_hessian_step(value) {
                boundary_theta_indices.push(index + 1);
            } else {
                interior_indices.push(index);
            }
        }

        let certification = self.pirls_profiled_certification_gradient(
            theta,
            n_agq,
            lower_bounds,
            PIRLS_PROFILED_CERTIFICATE_GRADIENT_TOLERANCE,
        );
        if !certification.unassessable_indices.is_empty() {
            let labels = certification
                .unassessable_indices
                .iter()
                .map(|index| (index + 1).to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "profiled fast-PIRLS stationarity is not assessable for covariance parameter(s) {labels}: escalated finite-difference readings disagreed"
            ));
        }
        let mut gradient_max_abs = 0.0_f64;
        for (index, &value) in certification.gradient.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "profiled fast-PIRLS stationarity gradient is non-finite for covariance parameter {}",
                    index + 1
                ));
            }
            let boundary = boundary_theta_indices.contains(&(index + 1));
            let fails = if boundary {
                // At a lower bound the objective must not decrease moving
                // into the interior; the forward-difference gradient there
                // must be non-negative up to tolerance.
                value < -PIRLS_PROFILED_CERTIFICATE_GRADIENT_TOLERANCE
            } else {
                value.abs() > PIRLS_PROFILED_CERTIFICATE_GRADIENT_TOLERANCE
            };
            if fails {
                return Err(format!(
                    "profiled fast-PIRLS stationarity gradient {value:.6e} for covariance parameter {} exceeds tolerance {PIRLS_PROFILED_CERTIFICATE_GRADIENT_TOLERANCE:.1e}",
                    index + 1
                ));
            }
            gradient_max_abs = gradient_max_abs.max(value.abs());
        }

        if interior_indices.is_empty() {
            return Err(
                "all covariance parameters sit at their lower bounds; the profiled curvature certificate is not assessable for a fully boundary optimum"
                    .to_string(),
            );
        }
        let hessian =
            self.finite_difference_pirls_profiled_hessian(theta, &interior_indices, n_agq)?;
        let curvature = certify_glmm_joint_hessian(&hessian, "profiled fast-PIRLS theta Hessian")?;

        Ok(PirlsProfiledOptimumCertificate {
            gradient_max_abs,
            min_eigenvalue: curvature.min_eigenvalue,
            condition_number: curvature.condition_number,
            escalated_theta_indices: certification
                .escalated_indices
                .iter()
                .map(|index| index + 1)
                .collect(),
            boundary_theta_indices,
        })
    }

    fn glmm_fixed_effect_inference_artifacts(
        &mut self,
        metadata: &GlmmFitMetadata,
    ) -> GlmmFixedEffectInferenceArtifacts {
        let separation_diagnostics = self.conservative_binomial_separation_diagnostics();
        if !separation_diagnostics.is_empty() {
            let affected_terms = separation_diagnostics
                .iter()
                .flat_map(|diagnostic| diagnostic.affected_terms.iter().cloned())
                .collect::<Vec<_>>();
            let reason = if affected_terms.is_empty() {
                "GLMM Wald inference is unavailable because binomial separation diagnostics were detected"
                    .to_string()
            } else {
                format!(
                    "GLMM Wald inference is unavailable because binomial separation diagnostics were detected for {}",
                    affected_terms.join(", ")
                )
            };
            return GlmmFixedEffectInferenceArtifacts {
                table: self.glmm_fixed_effect_inference_unavailable_table(
                    reason,
                    FixedEffectInferenceStatus::NotAssessed,
                    vec![
                        "near-separation/rare-outcome GLMMs keep Wald SE/z/p unavailable until a separation-robust inference backend is implemented"
                            .to_string(),
                    ],
                ),
                covariance: None,
            };
        }

        if metadata.estimation_method == "joint_laplace" && self.lmm.optsum.n_agq <= 1 {
            match self.glmm_joint_laplace_fixed_effect_inference_artifacts() {
                Ok(artifacts) => return artifacts,
                Err(reason) => {
                    return GlmmFixedEffectInferenceArtifacts {
                        table: self.glmm_fixed_effect_inference_unavailable_table(
                            reason,
                            FixedEffectInferenceStatus::NotAssessed,
                            vec![
                                "joint-laplace GLMM fixed-effect Hessian certificate did not pass quality gates"
                                    .to_string(),
                            ],
                        ),
                        covariance: None,
                    };
                }
            }
        }

        GlmmFixedEffectInferenceArtifacts {
            table: self.glmm_fixed_effect_inference_unavailable_table(
                glmm_fixed_effect_inference_unsupported_reason(metadata.estimation_method.as_str()),
                FixedEffectInferenceStatus::Unsupported,
                vec![
                    "GLMM fixed-effect covariance geometry is recorded separately and is not a certified Wald inference backend"
                        .to_string(),
                ],
            ),
            covariance: None,
        }
    }

    fn glmm_fixed_effect_inference_unavailable_table(
        &self,
        reason: String,
        status: FixedEffectInferenceStatus,
        notes: Vec<String>,
    ) -> FixedEffectInferenceTable {
        let estimates = self.coef();
        let rows = self
            .coef_names()
            .into_iter()
            .enumerate()
            .map(|(index, label)| FixedEffectInferenceRow {
                label: label.clone(),
                kind: FixedEffectInferenceRowKind::Coefficient,
                estimate: estimates
                    .get(index)
                    .copied()
                    .filter(|value| value.is_finite()),
                std_error: None,
                numerator_df: None,
                denominator_df: None,
                statistic: None,
                statistic_name: None,
                p_value: None,
                method: FixedEffectInferenceMethod::NotComputed,
                status,
                reliability: ReliabilityGrade::NotAvailable,
                reliability_reason: None,
                estimability: EstimabilityAssessment::FixedContrast(
                    FixedContrastEstimability::not_assessed(label),
                ),
                reason: Some(reason.clone()),
                details: None,
                notes: notes.clone(),
            })
            .collect();
        FixedEffectInferenceTable::new(rows)
    }
}

/// RAII guard for the AGQ deviance sweep.
///
/// [`GeneralizedLinearMixedModel::deviance`] perturbs `u[0]` (and, via
/// `update_eta`, `eta`/`mu`) at each Gauss-Hermite node. If the sweep panics
/// mid-way the model would be left perturbed and inconsistent. This guard
/// restores the conditional modes and recomputes `eta`/`mu` on drop — on the
/// normal path *and* during unwinding.
struct AgqRestoreGuard<'a> {
    glmm: &'a mut GeneralizedLinearMixedModel,
    /// `u[0]` snapshot at the conditional modes (flat, length `n_levels`).
    u0_flat: Vec<f64>,
}

impl std::ops::Deref for AgqRestoreGuard<'_> {
    type Target = GeneralizedLinearMixedModel;
    fn deref(&self) -> &Self::Target {
        self.glmm
    }
}

impl std::ops::DerefMut for AgqRestoreGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.glmm
    }
}

impl Drop for AgqRestoreGuard<'_> {
    fn drop(&mut self) {
        for (g, &uv) in self.u0_flat.iter().enumerate() {
            self.glmm.u[0][(0, g)] = uv;
        }
        self.glmm.update_eta();
    }
}

/// Diagonal entries of a [`MatrixBlock`].
///
/// Used by AGQ to read per-level scalings off the (1,1) block of the
/// Cholesky factor. Pulled out of [`GeneralizedLinearMixedModel::l11_diag`]
/// so the variant logic is unit-testable without constructing a full LMM.
///
/// For a `Dense` block this returns `min(nrows, ncols)` diagonal entries.
fn matrix_block_diag(block: &MatrixBlock) -> Vec<f64> {
    match block {
        MatrixBlock::Dense(m) => {
            let n = m.nrows().min(m.ncols());
            (0..n).map(|i| m[(i, i)]).collect()
        }
        MatrixBlock::Diagonal(v) => v.iter().copied().collect(),
        MatrixBlock::BlockDiagonal(blocks) => {
            let mut out = Vec::new();
            for blk in blocks {
                for i in 0..blk.nrows() {
                    out.push(blk[(i, i)]);
                }
            }
            out
        }
        MatrixBlock::Sparse(m) => {
            let dense = MatrixBlock::Sparse(m.clone()).as_dense();
            let n = dense.nrows().min(dense.ncols());
            (0..n).map(|i| dense[(i, i)]).collect()
        }
    }
}

fn rc_refcell_into_inner_or_clone<T: Clone>(value: Rc<RefCell<Vec<T>>>) -> Vec<T> {
    match Rc::try_unwrap(value) {
        Ok(cell) => cell.into_inner(),
        Err(value) => value.borrow().clone(),
    }
}

impl std::fmt::Display for GeneralizedLinearMixedModel {
    /// Default print: the compact `ModelPrint` summary (PRD § 15).
    /// Heavier reports (`audit_report`, `parameterization`,
    /// `changes`, `explain_model`) remain one explicit method call
    /// away.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.print_summary(), f)
    }
}

impl MixedModelFit for GeneralizedLinearMixedModel {
    fn nobs(&self) -> usize {
        self.y.len()
    }

    fn dof(&self) -> usize {
        self.lmm.feterm.rank
            + self.lmm.parmap.len()
            + if self.family.has_dispersion() || self.negative_binomial_estimate_theta {
                1
            } else {
                0
            }
    }

    fn coef(&self) -> DVector<f64> {
        let mut full = DVector::from_element(self.lmm.feterm.piv.len(), 0.0);
        for (i, &val) in self.beta.iter().enumerate() {
            if i < self.lmm.feterm.piv.len() {
                full[self.lmm.feterm.piv[i]] = val;
            }
        }
        full
    }

    fn fixef(&self) -> DVector<f64> {
        self.beta.clone()
    }
    fn coef_names(&self) -> Vec<String> {
        self.lmm.coef_names()
    }
    fn vcov(&self) -> DMatrix<f64> {
        self.recorded_fixed_effect_covariance()
            .or_else(|| self.profiled_glmm_fixed_effect_covariance())
            .unwrap_or_else(|| self.lmm.vcov())
    }
    fn stderror(&self) -> DVector<f64> {
        self.fixed_effect_inference_standard_errors()
            .unwrap_or_else(|| self.lmm.stderror())
    }
    fn fitted(&self) -> DVector<f64> {
        self.mu.clone()
    }
    fn residuals(&self) -> DVector<f64> {
        &self.y - &self.mu
    }
    fn response(&self) -> &DVector<f64> {
        &self.y
    }
    fn model_matrix(&self) -> &DMatrix<f64> {
        &self.lmm.feterm.x
    }
    fn objective(&self) -> f64 {
        if self.is_fitted() {
            // Cached deviance at the optimum (Laplace or AGQ as fit specified).
            self.lmm.optsum.fmin
        } else {
            // Pre-fit: only the Laplace approximation is callable from &self.
            self.laplace_objective()
        }
    }

    fn loglikelihood(&self) -> f64 {
        // Fast-PIRLS stores the dropped-constant deviance, while labelled
        // joint fits store the included-constant objective optimized by the
        // joint path. The log-likelihood — and therefore AIC/BIC/AICc — must
        // always be on the full normalized `-2 logLik` scale to match
        // lme4/MixedModels.jl.
        if self.is_fitted() {
            if glmm_objective_includes_response_constants(&self.lmm.optsum.return_value) {
                -self.lmm.optsum.fmin / 2.0
            } else {
                -(self.lmm.optsum.fmin + self.response_constants_offset()) / 2.0
            }
        } else {
            -self.laplace_objective_with_response_constants() / 2.0
        }
    }

    fn formula_label(&self) -> Option<String> {
        Some(self.lmm.formula.to_string())
    }

    fn is_fitted(&self) -> bool {
        self.lmm.optsum.feval > 0
    }
    fn is_singular(&self) -> bool {
        self.lmm.is_singular()
    }
    fn opt_summary(&self) -> &OptSummary {
        &self.lmm.optsum
    }
    fn theta(&self) -> Vec<f64> {
        self.theta.clone()
    }

    fn dispersion(&self, sqr: bool) -> f64 {
        if let Some(theta) = self.negative_binomial_theta {
            return theta;
        }
        if sqr {
            self.dispersion * self.dispersion
        } else {
            self.dispersion
        }
    }

    fn ranef(&self) -> Vec<DMatrix<f64>> {
        self.b.clone()
    }

    fn random_effect_terms(&self) -> Vec<RandomEffectTermInfo> {
        self.lmm.random_effect_terms()
    }

    fn family_kind(&self) -> Option<crate::model::traits::Family> {
        Some(self.family)
    }

    fn link_kind(&self) -> Option<crate::model::traits::LinkFunction> {
        Some(self.link)
    }
}

#[cfg(test)]
mod tests;
