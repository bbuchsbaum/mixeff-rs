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
    prediction_interval_cutoff, CovarianceKktClassification, LinearMixedModel, NewReLevels,
    OptimizerControl, PredictionVarianceMethod, PredictionVariancePayload,
    PredictionVarianceStatus,
};
use crate::model::traits::{Family, LinkFunction, MixedModelFit, RandomEffectTermInfo};
use crate::optimizer::trust_bq::{
    minimize_with_progress as minimize_trust_bq_with_progress, TrustBqOptions, TrustBqProgress,
    TrustBqStopReason,
};
use crate::stats::{BlockDescription, MixedModelProfile, ModelSummary, VarCorr};
use crate::types::{gh_norm, FitLogEntry, MatrixBlock, OptSummary, Optimizer, ReMat};
use crate::unstable_internal_method;

mod certify;
mod pirls;
mod predictive;
pub(crate) use certify::*;
pub(crate) use pirls::*;
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

    /// Predictions for new data with configurable scale and unseen-level
    /// handling.
    ///
    /// The fixed-effect design is rebuilt with the training-time categorical
    /// encoding and the random-effects contribution uses the fitted GLMM
    /// conditional modes. On [`GlmmPredictionScale::Response`], the fitted
    /// inverse link is applied after the link-scale predictor is assembled.
    pub fn predict_new(
        &self,
        newdata: &DataFrame,
        scale: GlmmPredictionScale,
        new_re_levels: NewReLevels,
    ) -> Result<Vec<Option<f64>>> {
        self.predict_new_with_offset(newdata, None, scale, new_re_levels)
    }

    /// Predictions for new data with an explicit new-data offset vector.
    ///
    /// Use this variant for offset GLMMs. The offset is added on link scale
    /// before response-scale transformation. When `offset` is `None`, new rows
    /// are predicted with zero offset.
    pub fn predict_new_with_offset(
        &self,
        newdata: &DataFrame,
        offset: Option<&[f64]>,
        scale: GlmmPredictionScale,
        new_re_levels: NewReLevels,
    ) -> Result<Vec<Option<f64>>> {
        if self.lmm.optsum.feval <= 0 {
            return Err(MixedModelError::NotFitted);
        }
        if let Some(offset) = offset {
            validate_offset(offset, newdata.nrow())?;
        }

        let mut eta =
            self.lmm
                .linear_predict_new_with_state(newdata, &self.beta, &self.b, new_re_levels)?;
        if let Some(offset) = offset {
            for (prediction, offset_i) in eta.iter_mut().zip(offset.iter()) {
                if let Some(value) = prediction.as_mut() {
                    *value += *offset_i;
                }
            }
        }

        match scale {
            GlmmPredictionScale::Link => Ok(eta),
            GlmmPredictionScale::Response => Ok(eta
                .into_iter()
                .map(|prediction| prediction.map(|value| self.link.linkinv(value)))
                .collect()),
        }
    }

    /// Prediction-variance payload for GLMM new-data predictions.
    ///
    /// Rows are marked [`PredictionVarianceStatus::Available`] when the fit
    /// carries certified optimum evidence: joint-Laplace fits with an
    /// available fixed-effect inference artifact, or profiled fast-PIRLS fits
    /// whose post-fit profiled-optimum certificate passed its stationarity
    /// and curvature gates. Uncertified fits return the same working-Hessian
    /// delta-method numbers and mark rows
    /// [`PredictionVarianceStatus::Degraded`] with the certificate failure in
    /// the row reason. Response-scale rows additionally carry plug-in
    /// future-observation `prediction_variance` and predictive-quantile
    /// `prediction_lower`/`prediction_upper` columns for families that
    /// support them. New-level cases remain unavailable with row-level
    /// reasons.
    pub fn predict_new_variance(
        &self,
        newdata: &DataFrame,
        scale: GlmmPredictionScale,
        new_re_levels: NewReLevels,
    ) -> Result<PredictionVariancePayload> {
        self.predict_new_variance_with_level(newdata, scale, new_re_levels, 0.95)
    }

    /// Prediction-variance payload for GLMM new-data predictions at an
    /// explicit confidence level.
    pub fn predict_new_variance_with_level(
        &self,
        newdata: &DataFrame,
        scale: GlmmPredictionScale,
        new_re_levels: NewReLevels,
        level: f64,
    ) -> Result<PredictionVariancePayload> {
        let z = prediction_interval_cutoff(level)?;
        let link_predictions =
            self.predict_new(newdata, GlmmPredictionScale::Link, new_re_levels)?;
        let predictions = match scale {
            GlmmPredictionScale::Link => link_predictions.clone(),
            GlmmPredictionScale::Response => {
                self.predict_new(newdata, GlmmPredictionScale::Response, new_re_levels)?
            }
        };
        let mut payload =
            self.lmm
                .predict_new_variance_with_level(newdata, new_re_levels, level)?;
        let inner_lmm_scale = self.lmm.sigma();
        let glmm_covariance_scale = self
            .glmm_conditional_prediction_covariance_scale()
            .ok_or_else(|| {
                MixedModelError::InvalidArgument(
                    "GLMM prediction covariance scale is non-finite".to_string(),
                )
            })?;
        if !inner_lmm_scale.is_finite() || inner_lmm_scale <= 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "inner LMM prediction covariance scale must be positive and finite; got {inner_lmm_scale}"
            )));
        }
        // The delegated LMM payload is scaled by the inner LMM's sigma()
        // convention. GLMM predict(se.fit) parity follows lme4's GLMM scale:
        // 1 for unscaled families and ML sqrt(pwrss / N) for scaled families.
        let glmm_scale_multiplier = (glmm_covariance_scale / inner_lmm_scale).powi(2);
        let joint_laplace_conditional_variance =
            self.certified_joint_laplace_fixed_covariance().is_some();
        let pirls_certified_conditional_variance = !joint_laplace_conditional_variance
            && matches!(self.pirls_profiled_optimum_certificate, Some(Ok(_)));
        let certified_conditional_variance =
            joint_laplace_conditional_variance || pirls_certified_conditional_variance;
        let pirls_certificate_failure = match &self.pirls_profiled_optimum_certificate {
            Some(Err(reason)) if !joint_laplace_conditional_variance => Some(reason.clone()),
            _ => None,
        };

        let certified_geometry_label = if joint_laplace_conditional_variance {
            "final joint-laplace PIRLS/Laplace conditional-mode covariance"
        } else {
            "final fast-PIRLS profiled conditional-mode covariance at the certified profiled optimum"
        };
        let available_note = match scale {
            GlmmPredictionScale::Link => {
                format!(
                    "GLMM link-scale fitted-mean prediction variance uses the {certified_geometry_label} over fixed and random effects, rescaled to the GLMM covariance scale; theta uncertainty is not included"
                )
            }
            GlmmPredictionScale::Response => {
                format!(
                    "GLMM response-scale fitted-mean prediction variance uses delta-method link propagation from the {certified_geometry_label} over fixed and random effects, rescaled to the GLMM covariance scale; theta uncertainty is not included"
                )
            }
        };
        let fit_is_joint = self
            .lmm
            .compiler_artifact
            .glmm_fit_metadata
            .as_ref()
            .is_some_and(|metadata| metadata.estimation_method.starts_with("joint"));
        let uncertified_clause = match &pirls_certificate_failure {
            Some(failure) => format!(
                "the fast-PIRLS profiled optimum certificate was not issued ({failure}); refit with GlmmFitOptions::joint_laplace() for certified conditional prediction variance"
            ),
            None if fit_is_joint => {
                "the joint GLMM Hessian certificate did not pass quality gates, so conditional prediction variance is not certified for this fit"
                    .to_string()
            }
            None => "no certified optimum evidence is available for this fit; refit with GlmmFitOptions::joint_laplace() for certified conditional prediction variance"
                .to_string(),
        };
        let degraded_reason = match scale {
            GlmmPredictionScale::Link => {
                format!(
                    "GLMM link-scale prediction variance uses PIRLS/Laplace working-Hessian geometry; {uncertified_clause}"
                )
            }
            GlmmPredictionScale::Response => {
                format!(
                    "GLMM response-scale prediction variance uses delta-method link propagation from PIRLS/Laplace working-Hessian geometry; {uncertified_clause}"
                )
            }
        };
        let future_observation_support: std::result::Result<(), String> = match scale {
            GlmmPredictionScale::Link => Err(
                "future-observation prediction intervals are response-scale objects; request GlmmPredictionScale::Response for prediction_variance and prediction bounds"
                    .to_string(),
            ),
            GlmmPredictionScale::Response => self.glmm_future_observation_family_support(),
        };
        let mut future_observation_row_failures = std::collections::BTreeSet::new();

        for row in &mut payload.rows {
            row.prediction = predictions[row.row];
            row.fixed_variance = row.fixed_variance.and_then(|value| {
                clean_glmm_prediction_variance_component(value * glmm_scale_multiplier)
            });
            row.random_variance = row.random_variance.and_then(|value| {
                clean_glmm_prediction_variance_component(value * glmm_scale_multiplier)
            });
            row.fixed_random_covariance = row
                .fixed_random_covariance
                .map(|value| value * glmm_scale_multiplier)
                .filter(|value| value.is_finite());
            row.combined_variance = row.combined_variance.and_then(|value| {
                clean_glmm_prediction_variance_component(value * glmm_scale_multiplier)
            });
            row.se_fit = row.combined_variance.map(f64::sqrt);

            let link_scale_se = row.combined_variance.map(f64::sqrt);
            let derivative = match (scale, link_predictions[row.row]) {
                (GlmmPredictionScale::Link, Some(_)) => Some(1.0),
                (GlmmPredictionScale::Response, Some(eta)) => {
                    let value = self.link.mu_eta(eta);
                    (value.is_finite()).then_some(value)
                }
                (_, None) => None,
            };
            let variance_multiplier = derivative.map(|value| value * value);

            if let Some(multiplier) = variance_multiplier {
                row.fixed_variance = row
                    .fixed_variance
                    .map(|value| (value * multiplier).max(0.0));
                row.random_variance = row
                    .random_variance
                    .map(|value| (value * multiplier).max(0.0));
                row.fixed_random_covariance =
                    row.fixed_random_covariance.map(|value| value * multiplier);
                row.combined_variance = row
                    .combined_variance
                    .map(|value| (value * multiplier).max(0.0));
                row.se_fit = row.combined_variance.map(f64::sqrt);
            } else {
                row.random_variance = None;
                row.fixed_random_covariance = None;
                row.combined_variance = None;
                row.se_fit = None;
            }

            if row.status == PredictionVarianceStatus::Available {
                // Response-scale symmetric bounds can escape the family's
                // valid range near the boundary; compute the interval on the
                // link scale and map both ends through the inverse link
                // (ordered, since some links are decreasing).
                let bounds = match (scale, row.prediction, row.se_fit) {
                    (GlmmPredictionScale::Link, Some(prediction), Some(se_fit)) => {
                        Some((prediction - z * se_fit, prediction + z * se_fit))
                    }
                    (GlmmPredictionScale::Response, Some(_), Some(_)) => {
                        match (link_predictions[row.row], link_scale_se) {
                            (Some(eta), Some(link_se)) => {
                                let one = self.link.linkinv(eta - z * link_se);
                                let other = self.link.linkinv(eta + z * link_se);
                                (one.is_finite() && other.is_finite())
                                    .then(|| (one.min(other), one.max(other)))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some((lower, upper)) = bounds {
                    row.confidence_lower = Some(lower);
                    row.confidence_upper = Some(upper);
                } else {
                    row.confidence_lower = None;
                    row.confidence_upper = None;
                }
                row.prediction_variance = None;
                row.prediction_lower = None;
                row.prediction_upper = None;
                if future_observation_support.is_ok() {
                    if let (Some(eta), Some(link_se)) = (link_predictions[row.row], link_scale_se) {
                        match self.glmm_future_observation(eta, link_se, level) {
                            Ok(future) => {
                                row.prediction_variance =
                                    clean_glmm_prediction_variance_component(future.variance);
                                row.prediction_lower = Some(future.lower);
                                row.prediction_upper = Some(future.upper);
                            }
                            Err(reason) => {
                                future_observation_row_failures.insert(reason);
                            }
                        }
                    }
                }
                if certified_conditional_variance {
                    row.reason = None;
                } else {
                    row.status = PredictionVarianceStatus::Degraded;
                    row.reason = Some(degraded_reason.clone());
                }
            } else {
                row.prediction_variance = None;
                row.confidence_lower = None;
                row.confidence_upper = None;
                row.prediction_lower = None;
                row.prediction_upper = None;
                let existing_reason = row.reason.clone().unwrap_or_else(|| {
                    "GLMM prediction variance is unavailable for this row".to_string()
                });
                row.reason = Some(format!("{existing_reason}; {degraded_reason}"));
            }
        }

        payload.method = if joint_laplace_conditional_variance {
            PredictionVarianceMethod::GlmmJointLaplaceConditionalDelta
        } else if pirls_certified_conditional_variance {
            PredictionVarianceMethod::GlmmPirlsProfiledCertifiedConditionalDelta
        } else {
            PredictionVarianceMethod::GlmmPirlsLaplaceWorkingDelta
        };
        payload.notes = if certified_conditional_variance {
            vec![
                available_note,
                format!(
                    "fixed, random, and fixed/random covariance components are transformed together from the {certified_geometry_label} geometry"
                ),
            ]
        } else {
            vec![
                degraded_reason,
                "fixed/random components are transformed from the inner PIRLS working LMM variance geometry"
                    .to_string(),
            ]
        };
        match &future_observation_support {
            Ok(()) => {
                payload.notes.push(format!(
                    "future-observation prediction_variance and prediction bounds are plug-in predictive summaries: the family conditional distribution (dispersion/size parameters treated as known at their estimates, future case weight 1) is mixed over link-scale fitted-mean uncertainty with {GLMM_PREDICTIVE_QUADRATURE_POINTS}-point Gauss-Hermite quadrature; bounds are predictive-distribution quantiles and prediction_variance is the law-of-total-variance moment; theta uncertainty is not included"
                ));
            }
            Err(reason) => {
                payload.notes.push(format!(
                    "future-observation prediction intervals are not reported: {reason}"
                ));
            }
        }
        for failure in future_observation_row_failures {
            payload.notes.push(format!(
                "future-observation prediction columns are unavailable for some rows: {failure}"
            ));
        }
        if matches!(scale, GlmmPredictionScale::Response) {
            payload.notes.push(
                "response-scale confidence bounds are link-scale Wald bounds mapped through the inverse link so they respect the family's valid range; se_fit remains delta-method on the response scale"
                    .to_string(),
            );
        }
        Ok(payload)
    }

    fn glmm_conditional_prediction_covariance_scale(&self) -> Option<f64> {
        if !self.family.has_dispersion() {
            return Some(1.0);
        }
        let pwrss = self.lmm.pwrss();
        if !pwrss.is_finite() || pwrss < 0.0 {
            return None;
        }
        let denom = self.y.len().max(1) as f64;
        Some((pwrss / denom).max(f64::MIN_POSITIVE).sqrt())
    }

    /// Whether this model's family supports closed-form plug-in
    /// future-observation summaries for new rows.
    fn glmm_future_observation_family_support(&self) -> std::result::Result<(), String> {
        match self.family {
            Family::Binomial => Err(
                "future-observation prediction intervals for a grouped binomial response require the future observation's trial count, which newdata does not carry; model unit-trial rows with Family::Bernoulli for future-observation intervals"
                    .to_string(),
            ),
            _ => Ok(()),
        }
    }

    /// Plug-in predictive summary for one future observation.
    ///
    /// The family conditional distribution (dispersion / NB size treated as
    /// known at their estimates, future case weight 1) is mixed over the
    /// link-scale Gaussian fitted-mean uncertainty with normalized
    /// Gauss-Hermite quadrature. `variance` is the law-of-total-variance
    /// moment; `lower`/`upper` are predictive-distribution quantiles, so for
    /// discrete families they are integers and the interval is conservative
    /// (coverage at least `level`).
    fn glmm_future_observation(
        &self,
        eta: f64,
        link_se: f64,
        level: f64,
    ) -> std::result::Result<GlmmFutureObservation, String> {
        if !eta.is_finite() || !link_se.is_finite() || link_se < 0.0 {
            return Err(
                "link-scale prediction or its standard error is not finite and non-negative"
                    .to_string(),
            );
        }
        let lower_p = (1.0 - level) / 2.0;
        let upper_p = 1.0 - lower_p;
        let quadrature = gh_norm(GLMM_PREDICTIVE_QUADRATURE_POINTS);
        let mut nodes = Vec::with_capacity(quadrature.len());
        for (&z_node, &weight) in quadrature.z.iter().zip(quadrature.w.iter()) {
            let mu = self.link.linkinv(eta + link_se * z_node);
            if !mu.is_finite() {
                return Err(
                    "predictive quadrature produced a non-finite conditional mean".to_string(),
                );
            }
            nodes.push((mu, weight));
        }

        let mean: f64 = nodes.iter().map(|(mu, w)| w * mu).sum();
        let second_moment: f64 = nodes.iter().map(|(mu, w)| w * mu * mu).sum();
        let mean_variance = (second_moment - mean * mean).max(0.0);
        let dispersion = if self.family.has_dispersion() {
            self.dispersion(true)
        } else {
            1.0
        };
        if !dispersion.is_finite() || dispersion <= 0.0 {
            return Err(format!(
                "family dispersion estimate {dispersion} is not usable for predictive variance"
            ));
        }
        let family_variance: f64 = nodes
            .iter()
            .map(|(mu, w)| w * dispersion * self.variance(*mu))
            .sum();
        if !family_variance.is_finite() || family_variance < 0.0 {
            return Err(
                "family conditional variance is not finite over the predictive quadrature"
                    .to_string(),
            );
        }
        let variance = mean_variance + family_variance;
        let spread = variance.sqrt();

        let (lower, upper) = match self.family {
            Family::Bernoulli => {
                let prob_zero: f64 = nodes.iter().map(|(mu, w)| w * (1.0 - mu)).sum();
                let quantile = |p: f64| if prob_zero >= p { 0.0 } else { 1.0 };
                (quantile(lower_p), quantile(upper_p))
            }
            Family::Poisson => {
                let components = nodes
                    .iter()
                    .map(|(mu, w)| {
                        PoissonDist::new(mu.max(GLMM_PREDICTIVE_MEAN_FLOOR))
                            .map(|distribution| (distribution, *w))
                            .map_err(|err| format!("poisson predictive component: {err}"))
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let cdf = |t: u64| {
                    components
                        .iter()
                        .map(|(distribution, w)| w * distribution.cdf(t))
                        .sum::<f64>()
                };
                let quantile = |p: f64| {
                    discrete_mixture_quantile(&cdf, p, mean).ok_or_else(|| {
                        "poisson predictive quantile search did not converge".to_string()
                    })
                };
                (quantile(lower_p)?, quantile(upper_p)?)
            }
            Family::NegativeBinomial => {
                let size = self
                    .negative_binomial_theta
                    .filter(|theta| theta.is_finite() && *theta > 0.0)
                    .ok_or_else(|| {
                        "negative-binomial predictive quantiles require a positive finite size parameter"
                            .to_string()
                    })?;
                let components = nodes
                    .iter()
                    .map(|(mu, w)| {
                        let mu = mu.max(GLMM_PREDICTIVE_MEAN_FLOOR);
                        NegativeBinomialDist::new(size, size / (size + mu))
                            .map(|distribution| (distribution, *w))
                            .map_err(|err| format!("negative-binomial predictive component: {err}"))
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let cdf = |t: u64| {
                    components
                        .iter()
                        .map(|(distribution, w)| w * distribution.cdf(t))
                        .sum::<f64>()
                };
                let quantile = |p: f64| {
                    discrete_mixture_quantile(&cdf, p, mean).ok_or_else(|| {
                        "negative-binomial predictive quantile search did not converge".to_string()
                    })
                };
                (quantile(lower_p)?, quantile(upper_p)?)
            }
            Family::Gamma => {
                let shape = 1.0 / dispersion;
                let components = nodes
                    .iter()
                    .map(|(mu, w)| {
                        if !(*mu > 0.0) {
                            return Err(
                                "predictive quadrature produced conditional means outside the gamma family domain"
                                    .to_string(),
                            );
                        }
                        GammaDist::new(shape, shape / mu)
                            .map(|distribution| (distribution, *w))
                            .map_err(|err| format!("gamma predictive component: {err}"))
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let cdf = |t: f64| {
                    components
                        .iter()
                        .map(|(distribution, w)| w * distribution.cdf(t))
                        .sum::<f64>()
                };
                let quantile = |p: f64| {
                    continuous_mixture_quantile(&cdf, p, Some(0.0), mean, spread).ok_or_else(|| {
                        "gamma predictive quantile search did not converge".to_string()
                    })
                };
                (quantile(lower_p)?, quantile(upper_p)?)
            }
            Family::InverseGaussian => {
                let lambda = 1.0 / dispersion;
                for (mu, _) in &nodes {
                    if !(*mu > 0.0) {
                        return Err(
                            "predictive quadrature produced conditional means outside the inverse-Gaussian family domain"
                                .to_string(),
                        );
                    }
                }
                let cdf = |t: f64| {
                    nodes
                        .iter()
                        .map(|(mu, w)| w * inverse_gaussian_cdf(t, *mu, lambda))
                        .sum::<f64>()
                };
                let quantile = |p: f64| {
                    continuous_mixture_quantile(&cdf, p, Some(0.0), mean, spread).ok_or_else(|| {
                        "inverse-Gaussian predictive quantile search did not converge".to_string()
                    })
                };
                (quantile(lower_p)?, quantile(upper_p)?)
            }
            Family::Normal => {
                let sigma = dispersion.sqrt();
                let components = nodes
                    .iter()
                    .map(|(mu, w)| {
                        Normal::new(*mu, sigma)
                            .map(|distribution| (distribution, *w))
                            .map_err(|err| format!("gaussian predictive component: {err}"))
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()?;
                let cdf = |t: f64| {
                    components
                        .iter()
                        .map(|(distribution, w)| w * distribution.cdf(t))
                        .sum::<f64>()
                };
                let quantile = |p: f64| {
                    continuous_mixture_quantile(&cdf, p, None, mean, spread).ok_or_else(|| {
                        "gaussian predictive quantile search did not converge".to_string()
                    })
                };
                (quantile(lower_p)?, quantile(upper_p)?)
            }
            other => {
                return Err(format!(
                    "future-observation predictive quantiles are not implemented for {other:?}"
                ));
            }
        };
        if !(lower.is_finite() && upper.is_finite() && lower <= upper) {
            return Err("predictive quantiles are not finite and ordered".to_string());
        }

        Ok(GlmmFutureObservation {
            variance,
            lower,
            upper,
        })
    }

    fn certified_joint_laplace_fixed_covariance(&self) -> Option<DMatrix<f64>> {
        let covariance = self
            .lmm
            .compiler_artifact
            .fixed_effect_covariance_matrix
            .as_ref()?;
        if covariance.status != FixedEffectCovarianceStatus::Available
            || covariance.method != FixedEffectCovarianceMethod::JointLaplaceActiveHessian
        {
            return None;
        }
        let matrix = covariance.matrix.as_ref()?;
        let p = self.lmm.feterm.rank;
        if matrix.len() != p || matrix.iter().any(|row| row.len() != p) {
            return None;
        }
        let values = matrix
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        let dense = DMatrix::from_row_slice(p, p, &values);
        matrix_is_finite_local(&dense).then_some(dense)
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

    /// Simulate a new response vector under a fresh draw of the random
    /// effects (the parametric-bootstrap data step).
    ///
    /// Draws `b_i = Λ_i u_i` with `u_i ~ N(0, I)`, forms the linear
    /// predictor `η = offset + Xβ̂ + Σ Z_i b_i`, maps it to `μ = g⁻¹(η)`,
    /// and samples the response from the conditional family with mean `μ`:
    /// Bernoulli → `{0, 1}`, Poisson → counts, Binomial → success
    /// proportion over the per-observation trial size (prior weights,
    /// default `1`), and Gamma → positive draws with `shape = 1 / phi`
    /// and `scale = mu * phi`, where `phi = dispersion(true)`.
    /// InverseGaussian and Normal-as-GLM are refused because they do not
    /// yet have certified family-specific response simulators.
    pub fn simulate_response<R: rand::Rng>(&self, rng: &mut R) -> Result<Vec<f64>> {
        use rand_distr::{Binomial, Distribution, Gamma as GammaDistribution, Normal, Poisson};

        match self.family {
            Family::Bernoulli
            | Family::Binomial
            | Family::Poisson
            | Family::NegativeBinomial
            | Family::Gamma => {}
            Family::InverseGaussian | Family::Normal => {
                return Err(MixedModelError::Unsupported(format!(
                    "{:?} GLMM parametric bootstrap is not implemented; no certified \
                     family-specific response simulator is available",
                    self.family
                )));
            }
        }
        let gamma_phi = if matches!(self.family, Family::Gamma) {
            let phi = self.dispersion(true);
            if !phi.is_finite() || phi <= 0.0 {
                return Err(MixedModelError::InvalidArgument(format!(
                    "Gamma GLMM bootstrap requires positive finite phi = dispersion(true); got {phi}"
                )));
            }
            Some(phi)
        } else {
            None
        };
        let negative_binomial_theta = if matches!(self.family, Family::NegativeBinomial) {
            Some(self.require_negative_binomial_theta()?)
        } else {
            None
        };

        let n = self.eta.len();
        let x = self.lmm.feterm.full_rank_x();
        let mut eta = &self.offset + x * &self.beta;

        let normal01 = Normal::new(0.0, 1.0).unwrap();
        for rt in &self.lmm.reterms {
            let n_levels = rt.n_levels();
            let u = DMatrix::from_fn(rt.vsize, n_levels, |_, _| normal01.sample(rng));
            let b = &rt.lambda * &u;
            let bvec = DVector::from_column_slice(b.as_slice());
            for (obs, &ref_idx) in rt.refs.iter().enumerate() {
                let r = ref_idx as usize;
                for s in 0..rt.vsize {
                    eta[obs] += rt.z[(s, obs)] * bvec[r * rt.vsize + s];
                }
            }
        }

        let mut y = vec![0.0f64; n];
        for (i, yi) in y.iter_mut().enumerate() {
            let mu = self.link.linkinv(eta[i]);
            if !mu.is_finite() {
                return Err(MixedModelError::InvalidArgument(format!(
                    "simulated conditional mean is non-finite at observation {i}"
                )));
            }
            match self.family {
                Family::Bernoulli => {
                    let p = mu.clamp(0.0, 1.0);
                    *yi = f64::from(rng.gen::<f64>() < p);
                }
                Family::Binomial => {
                    let p = mu.clamp(0.0, 1.0);
                    let trials = if self.wt.is_empty() { 1.0 } else { self.wt[i] };
                    let n_trials = trials.round().max(0.0) as u64;
                    if n_trials == 0 {
                        *yi = 0.0;
                    } else {
                        let count = Binomial::new(n_trials, p)
                            .map_err(|e| {
                                MixedModelError::InvalidArgument(format!(
                                    "binomial draw failed at observation {i}: {e}"
                                ))
                            })?
                            .sample(rng) as f64;
                        *yi = count / trials;
                    }
                }
                Family::Poisson => {
                    let lambda = mu.max(f64::MIN_POSITIVE);
                    *yi = Poisson::new(lambda)
                        .map_err(|e| {
                            MixedModelError::InvalidArgument(format!(
                                "poisson draw failed at observation {i}: {e}"
                            ))
                        })?
                        .sample(rng);
                }
                Family::NegativeBinomial => {
                    let theta = negative_binomial_theta.expect("NB theta computed above");
                    let mean = mu.max(f64::MIN_POSITIVE);
                    let lambda = GammaDistribution::new(theta, mean / theta)
                        .map_err(|e| {
                            MixedModelError::InvalidArgument(format!(
                                "negative-binomial gamma-mixture draw failed at observation {i}: {e}"
                            ))
                        })?
                        .sample(rng);
                    *yi = Poisson::new(lambda.max(f64::MIN_POSITIVE))
                        .map_err(|e| {
                            MixedModelError::InvalidArgument(format!(
                                "negative-binomial poisson draw failed at observation {i}: {e}"
                            ))
                        })?
                        .sample(rng);
                }
                Family::Gamma => {
                    let phi = gamma_phi.expect("Gamma phi computed above");
                    let mean = if mu > 0.0 {
                        mu
                    } else if mu == 0.0 {
                        f64::MIN_POSITIVE
                    } else {
                        return Err(MixedModelError::InvalidArgument(format!(
                            "Gamma draw requires positive conditional mean at observation {i}; got {mu}"
                        )));
                    };
                    let shape = 1.0 / phi;
                    let scale = mean * phi;
                    *yi = GammaDistribution::new(shape, scale)
                        .map_err(|e| {
                            MixedModelError::InvalidArgument(format!(
                                "Gamma draw failed at observation {i}: {e}"
                            ))
                        })?
                        .sample(rng);
                }
                Family::InverseGaussian | Family::Normal => {
                    unreachable!("dispersion families refused above")
                }
            }
        }
        Ok(y)
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

    /// PIRLS: Penalized Iteratively Reweighted Least Squares.
    ///
    /// Updates β and u until convergence. The working response and weights
    /// are derived from the current μ = g⁻¹(Xβ + Zb).
    ///
    /// * `vary_beta` – if false, β is held fixed and only u is updated
    ///
    /// Returns `Ok(true)` if PIRLS reached its convergence tolerance within
    /// the iteration budget, `Ok(false)` if it exhausted the budget without
    /// converging (the conditional modes are the best seen but unverified).
    /// The non-converged case is deliberately *not* an `Err`: callers decide
    /// how to surface it (the final fit records a diagnostic; interior
    /// optimizer probes tolerate it). `Err` is reserved for hard linear-
    /// algebra/state failures.
    pub fn pirls(&mut self, vary_beta: bool, verbose: bool) -> Result<bool> {
        self.pirls_with_options(vary_beta, verbose, GLMM_PIRLS_MAX_ITER, true)
    }

    fn pirls_with_options(
        &mut self,
        vary_beta: bool,
        verbose: bool,
        max_iter: usize,
        reset_modes: bool,
    ) -> Result<bool> {
        // Mirrors MixedModels.jl/src/generalizedlinearmixedmodel.jl pirls!
        // (lines 614-669): step-halving toward the previous accepted iterate
        // whenever a fresh IRLS step would worsen the Laplace objective. Keeps
        // the outer optimizer's view of obj(θ) consistent across probes —
        // without this, BOBYQA on multi-RE GLMM surfaces (e.g. grouseticks
        // Poisson) sees noisy values and reports `RoundoffLimited`.
        let tol = 1.0e-5;
        let max_halvings = 10;

        let n = self.y.len();

        // Reset the conditional modes when callers need deterministic probe
        // values instead of path-dependent warm starts.
        if reset_modes {
            for u in self.u.iter_mut() {
                u.fill(0.0);
            }
        }
        for (i, rt) in self.lmm.reterms.iter().enumerate() {
            self.b[i] = &rt.lambda * &self.u[i];
        }
        self.update_eta();

        // Save the initial accepted state for halving. The 1.0001 slack is
        // only an acceptance bound for the first step-halving loop; convergence
        // is compared with the uninflated accepted objective.
        let mut u_prev: Vec<DMatrix<f64>> = self.u.clone();
        let mut beta_prev = self.beta.clone();
        let mut obj0 = self.laplace_objective();
        let mut halving_bound = obj0 * 1.0001;

        let mut sqrtwts = vec![0.0f64; n];
        let mut working_y = vec![0.0f64; n];

        // Whether PIRLS reached its convergence tolerance within `max_iter`.
        // Returned to the caller so a non-converged conditional-mode solve is
        // *observable* rather than silently accepted (audit 03·H1). We do not
        // hard-error inside the loop: the outer optimizer legitimately probes
        // near the variance-component boundary where an interior step may
        // exhaust halving, and turning that into an error perturbs the
        // soft-barrier search away from valid boundary fits.
        let mut converged = false;

        for iter in 0..max_iter.max(1) {
            // --- Compute IRLS weights and working response ---
            for obs in 0..n {
                let mu_obs = self.mu[obs];
                let eta_obs = self.eta[obs];
                let y_obs = self.y[obs];

                let case_w = if self.wt.is_empty() {
                    1.0
                } else {
                    self.wt[obs]
                };
                (sqrtwts[obs], working_y[obs]) =
                    pirls_working_observation_with_offset_and_family_parameters(
                        self.family,
                        self.link,
                        self.negative_binomial_theta,
                        y_obs,
                        eta_obs,
                        mu_obs,
                        case_w,
                        self.offset[obs],
                    );
            }

            // --- Update the LMM with new IRLS weights ---
            self.lmm.update_irls_weights(&sqrtwts, &working_y)?;
            self.lmm.update_l()?;

            // --- Propose new β / u from the LMM solution ---
            let new_u = if vary_beta {
                self.beta = self.lmm.beta();
                self.lmm.ranef_u()
            } else {
                self.ranef_u_given_beta(&self.beta)
            };
            for (i, rt) in self.lmm.reterms.iter().enumerate() {
                self.u[i].copy_from(&new_u[i]);
                self.b[i] = &rt.lambda * &self.u[i];
            }
            self.update_eta();
            let mut obj = self.laplace_objective();

            // --- Step-halving: average toward the previous accepted state
            //     until obj is no worse, up to `max_halvings` averagings. ---
            // A non-finite obj must count as "worse": `NaN > bound` is false,
            // so without the explicit check a NaN/Inf iterate would skip
            // halving and be silently accepted (audit 03·H2 defense-in-depth;
            // the family μ-floors above are the primary fix).
            let mut nhalf = 0;
            while (!obj.is_finite() || obj > halving_bound) && nhalf < max_halvings {
                nhalf += 1;
                for i in 0..self.u.len() {
                    self.u[i] = 0.5 * (&self.u[i] + &u_prev[i]);
                }
                if vary_beta {
                    self.beta = 0.5 * (&self.beta + &beta_prev);
                }
                for (i, rt) in self.lmm.reterms.iter().enumerate() {
                    self.b[i] = &rt.lambda * &self.u[i];
                }
                self.update_eta();
                obj = self.laplace_objective();
            }

            if verbose {
                eprintln!("  PIRLS iter {iter}: obj = {obj:.6} (nhalf = {nhalf})");
            }

            if pirls_converged(obj, obj0, tol) {
                converged = true;
                break;
            }

            // Accept iterate as the new previous state.
            for i in 0..self.u.len() {
                u_prev[i].copy_from(&self.u[i]);
            }
            beta_prev = self.beta.clone();
            obj0 = obj;
            halving_bound = obj;
        }

        self.refresh_dispersion();

        Ok(converged)
    }

    /// Conditional modes of the random effects with β held fixed.
    ///
    /// `LinearMixedModel::ranef_u()` intentionally profiles β before forming
    /// residuals. The joint GLMM objective needs the lme4-style
    /// `nAGQ > 0` surface where the candidate β is part of the outer parameter
    /// vector, so the inner PIRLS step must solve only for `u` conditional on
    /// that β.
    fn ranef_u_given_beta(&self, beta: &DVector<f64>) -> Vec<DMatrix<f64>> {
        let k = self.lmm.reterms.len();
        let p = self.lmm.feterm.rank;
        let n = self.lmm.dims.n;
        let wtxy = &self.lmm.xy_mat.wtxy;

        let mut wr = vec![0.0f64; n];
        for obs in 0..n {
            let mut val = wtxy[(obs, p)];
            for q in 0..p {
                val -= wtxy[(obs, q)] * beta[q];
            }
            wr[obs] = val;
        }

        let mut c_vecs = Vec::with_capacity(k);
        for re in &self.lmm.reterms {
            let vs = re.vsize;
            let nranef = re.n_ranef();
            let n_levels = re.n_levels();

            let mut c = vec![0.0; nranef];
            for (obs, &wr_obs) in wr.iter().enumerate() {
                let r = re.refs[obs] as usize;
                for s in 0..vs {
                    c[r * vs + s] += re.wtz[(s, obs)] * wr_obs;
                }
            }

            let lambda = &re.lambda;
            let mut c_scaled = vec![0.0; nranef];
            for lev in 0..n_levels {
                for i in 0..vs {
                    let mut val = 0.0;
                    for row in i..vs {
                        val += lambda[(row, i)] * c[lev * vs + row];
                    }
                    c_scaled[lev * vs + i] = val;
                }
            }
            c_vecs.push(DVector::from_vec(c_scaled));
        }

        let mut v_vecs: Vec<DVector<f64>> = Vec::with_capacity(k);
        for j in 0..k {
            let nranef_j = self.lmm.reterms[j].n_ranef();
            let mut rhs = c_vecs[j].clone();

            for (m, v_m) in v_vecs.iter().enumerate().take(j) {
                let l_jm = self.lmm.l_blocks[glmm_block_index(j, m)].as_dense();
                for row in 0..nranef_j {
                    let mut dot = 0.0;
                    for col in 0..v_m.len() {
                        dot += l_jm[(row, col)] * v_m[col];
                    }
                    rhs[row] -= dot;
                }
            }

            let mut v_j = rhs.as_slice().to_vec();
            solve_dense_lower_against_rhs(
                &self.lmm.l_blocks[glmm_block_index(j, j)].as_dense(),
                &mut v_j,
            );
            v_vecs.push(DVector::from_vec(v_j));
        }

        let mut u_vecs: Vec<DVector<f64>> = vec![DVector::zeros(0); k];
        for j in (0..k).rev() {
            let nranef_j = self.lmm.reterms[j].n_ranef();
            let mut rhs = v_vecs[j].clone();

            for m in (j + 1)..k {
                let l_mj = self.lmm.l_blocks[glmm_block_index(m, j)].as_dense();
                let u_m = &u_vecs[m];
                for row in 0..nranef_j {
                    let mut dot = 0.0;
                    for col in 0..u_m.len() {
                        dot += l_mj[(col, row)] * u_m[col];
                    }
                    rhs[row] -= dot;
                }
            }

            let mut u_j = rhs.as_slice().to_vec();
            solve_dense_upper_from_lower_transpose_against_rhs(
                &self.lmm.l_blocks[glmm_block_index(j, j)].as_dense(),
                &mut u_j,
            );
            u_vecs[j] = DVector::from_vec(u_j);
        }

        self.lmm
            .reterms
            .iter()
            .zip(u_vecs)
            .map(|(rt, u)| DMatrix::from_column_slice(rt.vsize, rt.n_levels(), u.as_slice()))
            .collect()
    }

    /// Laplace approximation objective: deviance residuals + u penalty + log|L|.
    pub fn laplace_objective(&self) -> f64 {
        // For binomial-with-trials data the response is a per-trial proportion
        // and `wt[i]` is the trial count; weighting the per-observation
        // deviance contribution by `wt[i]` recovers the binomial deviance.
        let dev: f64 = (0..self.y.len())
            .map(|i| self.case_weight(i) * self.dev_resid_component(self.y[i], self.mu[i]))
            .sum();
        let u_penalty: f64 = self
            .u
            .iter()
            .map(|u| u.iter().map(|x| x * x).sum::<f64>())
            .sum();
        dev + u_penalty + self.lmm_logdet()
    }

    fn u_penalty(&self) -> f64 {
        self.u
            .iter()
            .map(|u| u.iter().map(|x| x * x).sum::<f64>())
            .sum()
    }

    fn minus_two_loglik_observation(&self, index: usize) -> f64 {
        let y = self.y[index];
        let mu = self.mu[index].max(f64::MIN_POSITIVE);
        match self.family {
            Family::Bernoulli | Family::Binomial => {
                let trials = self.case_weight(index).max(0.0);
                let successes = (trials * y).clamp(0.0, trials);
                let failures = trials - successes;
                let p = mu.clamp(1.0e-15, 1.0 - 1.0e-15);
                let log_choose = if trials == 0.0 {
                    0.0
                } else {
                    ln_gamma(trials + 1.0) - ln_gamma(successes + 1.0) - ln_gamma(failures + 1.0)
                };
                let success_term = if successes == 0.0 {
                    0.0
                } else {
                    successes * p.ln()
                };
                let failure_term = if failures == 0.0 {
                    0.0
                } else {
                    failures * (1.0 - p).ln()
                };
                -2.0 * (log_choose + success_term + failure_term)
            }
            Family::Poisson => {
                let count_term = if y == 0.0 { 0.0 } else { y * mu.ln() };
                -2.0 * (count_term - mu - ln_gamma(y + 1.0))
            }
            Family::NegativeBinomial => {
                let theta = self
                    .negative_binomial_theta
                    .expect("negative-binomial GLMM stores fixed theta");
                let loglik = ln_gamma(y + theta) - ln_gamma(theta) - ln_gamma(y + 1.0)
                    + theta * (theta / (theta + mu)).ln()
                    + if y == 0.0 {
                        0.0
                    } else {
                        y * (mu / (theta + mu)).ln()
                    };
                -2.0 * loglik
            }
            Family::Gamma => {
                let phi = self.dispersion(true).max(f64::MIN_POSITIVE);
                let shape = 1.0 / phi;
                let scale = mu * phi;
                -2.0 * ((shape - 1.0) * y.ln() - y / scale - shape * scale.ln() - ln_gamma(shape))
            }
            Family::Normal => {
                let variance = self.dispersion(true).max(f64::MIN_POSITIVE);
                let residual = y - mu;
                (2.0 * std::f64::consts::PI * variance).ln() + residual * residual / variance
            }
            Family::InverseGaussian => {
                let phi = self.dispersion(true).max(f64::MIN_POSITIVE);
                (2.0 * std::f64::consts::PI * phi * y.powi(3)).ln()
                    + (y - mu).powi(2) / (phi * y * mu * mu)
            }
        }
    }

    /// Additive difference between the current dropped-constant Laplace
    /// objective and the same conditional objective with response constants
    /// retained.
    ///
    /// For Poisson, negative-binomial, and binomial-family GLMMs this is an
    /// observation-only constant once family parameters are fixed.
    /// Dispersion families also depend on the current scale
    /// convention, so callers should treat those values as explicit metadata
    /// rather than as a cross-engine parity claim.
    pub fn response_constants_offset(&self) -> f64 {
        let dropped: f64 = (0..self.y.len())
            .map(|i| self.case_weight(i) * self.dev_resid_component(self.y[i], self.mu[i]))
            .sum();
        let included: f64 = (0..self.y.len())
            .map(|i| self.minus_two_loglik_observation(i))
            .sum();
        included - dropped
    }

    /// Laplace objective with response normalising constants retained.
    ///
    /// This is the objective convention needed for meaningful comparison to
    /// `lme4`'s `-2 logLik` scale. It deliberately lives alongside
    /// [`laplace_objective`](Self::laplace_objective) so current fast-PIRLS
    /// fitting and comparison artifacts keep their existing dropped-constant
    /// semantics while certified joint GLMM parity is promoted row by row.
    pub fn laplace_objective_with_response_constants(&self) -> f64 {
        (0..self.y.len())
            .map(|i| self.minus_two_loglik_observation(i))
            .sum::<f64>()
            + self.u_penalty()
            + self.lmm_logdet()
    }

    /// Deviance of the GLMM.
    ///
    /// For `n_agq <= 1`, returns the Laplace approximation
    /// (`laplace_objective`).
    ///
    /// For `n_agq > 1`, returns the deviance evaluated by `n_agq`-point
    /// adaptive Gauss-Hermite quadrature. AGQ is only defined for models
    /// with a single scalar random-effects term; on multi-term or
    /// vector-valued RE models, calling with `n_agq > 1` is a programmer
    /// error (use [`validate_agq`](Self::validate_agq) up front, or call
    /// via [`fit_with_options`](Self::fit_with_options) which preflights).
    ///
    /// Mutates internal `u`, `eta`, `mu` during the AGQ sweep but restores
    /// observable state before returning.
    pub fn deviance(&mut self, n_agq: usize) -> f64 {
        if n_agq <= 1 {
            return self.laplace_objective();
        }
        // Hard runtime check (not debug_assert!): in release a violated
        // invariant here would otherwise feed a multi-/vector-valued RE model
        // into the single-scalar AGQ math below, silently producing wrong
        // numbers (or an opaque index panic) rather than a clear refusal.
        assert!(
            self.is_single_scalar_re(),
            "AGQ with n_agq > 1 requires exactly one scalar random-effects term; \
             callers must invoke validate_agq() before reaching this path"
        );

        let n_levels = self.u[0].ncols();
        let n_obs = self.y.len();

        // Snapshot u₀ (a flat vector of length n_levels since vsize == 1).
        let u0_flat: Vec<f64> = self.u[0].as_slice().to_vec();
        debug_assert_eq!(u0_flat.len(), n_levels);

        // Per-group sd from the diagonal of the (1,1) Cholesky block:
        // sd[g] = 1 / |L₁₁_diag[g]|.
        let l11_diag = self.l11_diag();
        debug_assert_eq!(l11_diag.len(), n_levels);
        let sd: Vec<f64> = l11_diag.iter().map(|d| 1.0 / d.abs()).collect();

        // Group index per observation. Clone to release the borrow on
        // `self.lmm` so we can call `update_eta(&mut self)` inside the loop.
        let refs: Vec<u32> = self.lmm.reterms[0].refs.clone();

        // devc0[g] = u₀[g]² + Σ_{i in group g} devresid_i  (at the conditional modes)
        let mut devc0 = vec![0.0_f64; n_levels];
        for (g, &uv) in u0_flat.iter().enumerate() {
            devc0[g] = uv * uv;
        }
        for i in 0..n_obs {
            devc0[refs[i] as usize] +=
                self.case_weight(i) * self.dev_resid_component(self.y[i], self.mu[i]);
        }

        // Sweep over GH nodes.
        let rule = crate::types::gh_norm(n_agq);
        let mut mult = vec![0.0_f64; n_levels];
        let mut devc = vec![0.0_f64; n_levels];

        // From here on `u[0]`/`eta`/`mu` are perturbed at each node. The guard
        // restores them when this scope ends — including if the sweep panics.
        let mut work = AgqRestoreGuard {
            glmm: self,
            u0_flat: u0_flat.clone(),
        };

        for (&z, &w) in rule.z.iter().zip(rule.w.iter()) {
            if w == 0.0 {
                continue;
            }
            if z == 0.0 {
                // devc == devc0, exp(0) * w simplifies to w
                for g in 0..n_levels {
                    mult[g] += w;
                }
                continue;
            }
            // u[g] = u₀[g] + z * sd[g]
            for g in 0..n_levels {
                work.u[0][(0, g)] = u0_flat[g] + z * sd[g];
            }
            work.update_eta();
            // devc[g] = u[g]² + Σ devresid_i (per group)
            for g in 0..n_levels {
                let uv = work.u[0][(0, g)];
                devc[g] = uv * uv;
            }
            for i in 0..n_obs {
                devc[refs[i] as usize] +=
                    work.case_weight(i) * work.dev_resid_component(work.y[i], work.mu[i]);
            }
            // mult[g] += exp((z² + devc0[g] - devc[g]) / 2) * w
            let z2 = z * z;
            for g in 0..n_levels {
                mult[g] += ((z2 + devc0[g] - devc[g]) * 0.5).exp() * w;
            }
        }

        // `work` drops here, restoring u and η/μ (also on a panic above).
        drop(work);

        let sum_devc0: f64 = devc0.iter().sum();
        let log_mult: f64 = mult.iter().map(|m| m.ln()).sum();
        let log_sd: f64 = sd.iter().map(|s| s.ln()).sum();
        sum_devc0 - 2.0 * (log_mult + log_sd)
    }

    /// Deviance with response normalising constants retained.
    ///
    /// For `n_agq <= 1`, this is the Laplace objective on the `-2 logLik`
    /// scale. For AGQ, the quadrature objective is shifted by the same
    /// response-constant offset used by the Laplace path.
    pub fn deviance_with_response_constants(&mut self, n_agq: usize) -> f64 {
        if n_agq <= 1 {
            return self.laplace_objective_with_response_constants();
        }
        let offset = self.response_constants_offset();
        self.deviance(n_agq) + offset
    }

    fn case_weight(&self, obs: usize) -> f64 {
        if self.wt.is_empty() {
            1.0
        } else {
            self.wt[obs]
        }
    }

    /// True iff the model has exactly one random-effects term and that
    /// term has `vsize == 1` (a scalar random effect).
    pub fn is_single_scalar_re(&self) -> bool {
        self.lmm.reterms.len() == 1 && self.lmm.reterms[0].vsize == 1
    }

    /// Diagonal of the (1,1) block of the lower-Cholesky factor `L`.
    ///
    /// For a single scalar RE term this is a per-level vector of length
    /// `n_levels`. Used by [`deviance`](Self::deviance) to derive AGQ
    /// node spacings.
    fn l11_diag(&self) -> Vec<f64> {
        matrix_block_diag(&self.lmm.l_blocks[0])
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

    fn record_invalid_agq_diagnostic(&mut self, n_agq: usize, reason: &str) {
        self.lmm
            .compiler_artifact
            .diagnostics
            .retain(|diagnostic| diagnostic.code != DiagnosticCode::InvalidAgqRequest);

        let affected_terms = self
            .lmm
            .reterms
            .iter()
            .map(random_effect_term_label)
            .collect::<Vec<_>>();
        let term_summaries = self
            .lmm
            .reterms
            .iter()
            .map(|term| {
                serde_json::json!({
                    "group": &term.grouping_name,
                    "columns": &term.cnames,
                    "basis_dimension": term.vsize,
                })
            })
            .collect::<Vec<_>>();
        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::InvalidAgqRequest,
            DiagnosticSeverity::Error,
            DiagnosticStage::Optimization,
            format!(
                "Invalid adaptive Gauss-Hermite quadrature request: n_agq = {n_agq} requires exactly one scalar random-effects term. Use n_agq = 1 for the Laplace approximation or simplify the random-effects structure."
            ),
        )
        .with_affected_terms(affected_terms)
        .with_suggested_actions(vec![
            "use n_agq = 1 for Laplace approximation on this random-effects structure"
                .to_string(),
            "fit AGQ only for a model with exactly one scalar random-effects term".to_string(),
        ]);
        diagnostic
            .payload
            .insert("n_agq".to_string(), serde_json::json!(n_agq));
        diagnostic
            .payload
            .insert("reason".to_string(), serde_json::json!(reason));
        diagnostic.payload.insert(
            "random_effect_term_count".to_string(),
            serde_json::json!(self.lmm.reterms.len()),
        );
        diagnostic.payload.insert(
            "random_effect_terms".to_string(),
            serde_json::json!(term_summaries),
        );
        self.lmm.compiler_artifact.diagnostics.push(diagnostic);
    }

    fn record_pirls_failure_diagnostic(&mut self, theta: &[f64], reason: &str) {
        self.lmm
            .compiler_artifact
            .diagnostics
            .retain(|diagnostic| diagnostic.code != DiagnosticCode::PirlsFailure);

        let affected_terms = self
            .lmm
            .reterms
            .iter()
            .map(random_effect_term_label)
            .collect::<Vec<_>>();
        let nonfinite_theta_indices = theta
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (!value.is_finite()).then_some(index))
            .collect::<Vec<_>>();
        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::PirlsFailure,
            DiagnosticSeverity::Error,
            DiagnosticStage::Optimization,
            "PIRLS failed while evaluating the final optimizer parameters for the GLMM; the fit was not completed.",
        )
        .with_affected_terms(affected_terms)
        .with_suggested_actions(vec![
            "inspect the optimizer return code and theta values before using this fit".to_string(),
            "try a different starting value, a simpler random-effects structure, or a lower optimizer step budget to localize the failure".to_string(),
            "check response domain, offsets, weights, and predictor scaling for invalid values".to_string(),
        ]);
        diagnostic
            .payload
            .insert("reason".to_string(), serde_json::json!(reason));
        diagnostic
            .payload
            .insert("theta_len".to_string(), serde_json::json!(theta.len()));
        diagnostic.payload.insert(
            "nonfinite_theta_indices".to_string(),
            serde_json::json!(nonfinite_theta_indices),
        );
        self.lmm.compiler_artifact.diagnostics.push(diagnostic);
    }

    /// Record a Warning when the inner PIRLS at the *final* optimizer θ did
    /// not reach its convergence tolerance within the iteration budget.
    ///
    /// The fit is still returned (mirroring MixedModels.jl, which also
    /// returns a model after a bounded PIRLS), but the non-convergence must
    /// not be *silent* (audit 03·H1): a downstream consumer can see this
    /// diagnostic instead of unknowingly trusting unverified conditional
    /// modes. Distinct from [`Self::record_pirls_failure_diagnostic`], which
    /// flags a hard PIRLS/linear-algebra failure that aborts the fit.
    fn record_pirls_nonconvergence_diagnostic(&mut self, theta: &[f64]) {
        self.lmm
            .compiler_artifact
            .diagnostics
            .retain(|d| d.code != DiagnosticCode::OptimizerNonconvergence);

        let affected_terms = self
            .lmm
            .reterms
            .iter()
            .map(random_effect_term_label)
            .collect::<Vec<_>>();
        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::OptimizerNonconvergence,
            DiagnosticSeverity::Warning,
            DiagnosticStage::Optimization,
            "the inner PIRLS conditional-mode solve did not reach its \
             convergence tolerance within the iteration budget at the final \
             optimizer parameters; the random-effect modes (and therefore the \
             Laplace/AGQ objective) are the best seen but unverified.",
        )
        .with_affected_terms(affected_terms)
        .with_suggested_actions(vec![
            "treat the conditional modes and objective as provisional and \
             cross-check against an alternate starting value"
                .to_string(),
            "simplify the random-effects structure or rescale predictors if \
             the GLMM surface is ill-conditioned near the optimum"
                .to_string(),
        ]);
        diagnostic
            .payload
            .insert("theta_len".to_string(), serde_json::json!(theta.len()));
        diagnostic
            .payload
            .insert("stage".to_string(), serde_json::json!("final_pirls"));
        self.lmm.compiler_artifact.diagnostics.push(diagnostic);
    }

    /// Log-determinant from the LMM's Cholesky factor.
    fn lmm_logdet(&self) -> f64 {
        // Delegate to the internal LMM's block structure
        let k = self.lmm.dims.nretrms;
        let mut logdet = 0.0;
        for j in 0..k {
            let idx = j * (j + 1) / 2 + j; // block_index(j, j)
            logdet += match &self.lmm.l_blocks[idx] {
                MatrixBlock::Dense(m) => {
                    let n = m.nrows().min(m.ncols());
                    (0..n).map(|i| m[(i, i)].abs().ln()).sum::<f64>()
                }
                MatrixBlock::Diagonal(v) => v.iter().map(|x| x.abs().ln()).sum::<f64>(),
                MatrixBlock::BlockDiagonal(blocks) => blocks
                    .iter()
                    .map(|blk| {
                        let n = blk.nrows();
                        (0..n).map(|i| blk[(i, i)].abs().ln()).sum::<f64>()
                    })
                    .sum::<f64>(),
                MatrixBlock::Sparse(m) => {
                    let dense = MatrixBlock::Sparse(m.clone()).as_dense();
                    let n = dense.nrows().min(dense.ncols());
                    (0..n).map(|i| dense[(i, i)].abs().ln()).sum::<f64>()
                }
            };
        }
        2.0 * logdet
    }

    /// Fit the GLMM.
    pub fn fit(&mut self) -> Result<&mut Self> {
        self.fit_with_glmm_options(GlmmFitOptions::default())
    }

    /// Refit the GLMM to a new response vector from the recorded initial θ.
    ///
    /// This mirrors Julia's `refit!` semantics for bootstrap and simulation
    /// workflows: the optimizer starts from `optsum.initial`, not from the
    /// previous optimum.
    pub fn refit(&mut self, new_y: &[f64]) -> Result<&mut Self> {
        let n_agq = self.lmm.optsum.n_agq.max(1);
        self.refit_with_options(new_y, n_agq, false)
    }

    /// Refit the GLMM to a new response vector with an explicit AGQ setting.
    pub fn refit_with_options(
        &mut self,
        new_y: &[f64],
        n_agq: usize,
        verbose: bool,
    ) -> Result<&mut Self> {
        if let Err(error) = self.validate_agq(n_agq) {
            self.record_invalid_agq_diagnostic(n_agq, &error.to_string());
            return Err(error);
        }
        self.reset_for_refit(Some(new_y))?;
        self.fit_with_options(true, n_agq, verbose)
    }

    /// Fit after first applying a compiler policy.
    pub fn fit_with_compiler_policy(
        &mut self,
        compiler_policy: CompilerPolicy,
    ) -> Result<&mut Self> {
        self.set_compiler_policy(compiler_policy)?;
        self.fit()
    }

    /// Fit with options.
    ///
    /// `fast` selects the MixedModels.jl-style fast path, which profiles over
    /// θ and updates β through PIRLS. `fast = false` selects the certified
    /// joint path: joint Laplace for `n_agq <= 1`, and joint AGQ for valid
    /// single-scalar random-effect models with `n_agq > 1`. NLopt builds use
    /// BOBYQA; dependency-light builds use the native TrustBQ joint path.
    ///
    /// `n_agq` selects the deviance approximation: `1` (or `0`) means the
    /// Laplace approximation; values `>= 2` request `n_agq`-point adaptive
    /// Gauss-Hermite quadrature, which is only valid for models with a
    /// single scalar random-effects term and is rejected up front
    /// otherwise.
    pub fn fit_with_options(
        &mut self,
        fast: bool,
        n_agq: usize,
        verbose: bool,
    ) -> Result<&mut Self> {
        self.fit_with_glmm_options(GlmmFitOptions {
            fast,
            n_agq,
            verbose,
            optimizer_control: OptimizerControl::default(),
        })
    }

    /// Fit with explicit GLMM options.
    pub fn fit_with_glmm_options(&mut self, options: GlmmFitOptions) -> Result<&mut Self> {
        let GlmmFitOptions {
            fast,
            n_agq,
            verbose,
            optimizer_control,
        } = options;
        if let Err(error) = self.validate_agq(n_agq) {
            self.record_invalid_agq_diagnostic(n_agq, &error.to_string());
            return Err(error);
        }
        if self.lmm.optsum.feval > 0 {
            return Err(MixedModelError::AlreadyFitted);
        }
        self.lmm.apply_optimizer_control(&optimizer_control)?;
        if let Some(start_theta) = &optimizer_control.start_theta {
            self.theta = start_theta.clone();
        }
        if self.family == Family::NegativeBinomial && self.negative_binomial_estimate_theta {
            return self.fit_negative_binomial_estimated_theta(fast, n_agq, verbose);
        }
        if !fast {
            return self.fit_joint_glmm_with_response_constants(n_agq, verbose);
        }
        self.fit_with_options_impl(n_agq, verbose)
    }

    fn fit_negative_binomial_estimated_theta(
        &mut self,
        fast: bool,
        n_agq: usize,
        verbose: bool,
    ) -> Result<&mut Self> {
        let initial_theta = self.require_negative_binomial_theta()?;
        let mut current_theta = clamp_negative_binomial_theta(initial_theta);
        let mut last_fit_theta = f64::NAN;
        let mut update_iterations = 0usize;
        let mut converged = false;

        for iteration in 0..NEGATIVE_BINOMIAL_THETA_MAX_ITERS {
            if self.lmm.optsum.feval > 0 {
                self.reset_for_refit(None)?;
            }
            self.negative_binomial_theta = Some(current_theta);
            self.fit_negative_binomial_conditional(fast, n_agq, verbose)?;
            last_fit_theta = current_theta;

            let next_theta = self.estimate_negative_binomial_theta_given_fit()?;
            update_iterations = iteration + 1;
            let relative_change = relative_theta_change(current_theta, next_theta);
            if verbose {
                eprintln!(
                    "  NB theta outer iter {}: theta = {:.6}, updated = {:.6}, rel_change = {:.3e}",
                    iteration + 1,
                    current_theta,
                    next_theta,
                    relative_change
                );
            }
            current_theta = next_theta;
            if relative_change <= NEGATIVE_BINOMIAL_THETA_TOL {
                converged = true;
                break;
            }
        }

        if relative_theta_change(last_fit_theta, current_theta)
            > NEGATIVE_BINOMIAL_THETA_FINAL_REFIT_TOL
        {
            self.reset_for_refit(None)?;
            self.negative_binomial_theta = Some(current_theta);
            self.fit_negative_binomial_conditional(fast, n_agq, verbose)?;
            last_fit_theta = current_theta;
        }

        self.negative_binomial_theta = Some(last_fit_theta);
        self.refresh_dispersion();
        self.record_negative_binomial_theta_estimation_metadata(
            initial_theta,
            last_fit_theta,
            update_iterations,
            converged,
        );
        Ok(self)
    }

    fn fit_negative_binomial_conditional(
        &mut self,
        fast: bool,
        n_agq: usize,
        verbose: bool,
    ) -> Result<()> {
        if fast {
            self.fit_with_options_impl(n_agq, verbose)?;
        } else {
            self.fit_joint_glmm_with_response_constants(n_agq, verbose)?;
        }
        Ok(())
    }

    fn configure_profile_start_optimizer(&mut self) {
        let optimizer = default_fast_glmm_optimizer();
        self.lmm.optsum.optimizer = optimizer;
        self.lmm.optsum.backend = optimizer.canonical_backend();
        self.lmm.optsum.optimizer_source = crate::types::OptimizerSource::Auto;
        self.lmm
            .optsum
            .caller_set_fields
            .retain(|field| field != "optimizer");
    }

    /// Labelled joint GLMM Laplace fit.
    ///
    /// This path optimizes `[β; θ]` against the included-response-constants
    /// Laplace objective. The public `fast = false` path delegates here for
    /// `n_agq <= 1` when NLopt is enabled, while summaries keep it distinct
    /// from the fast-PIRLS profiled path and from labelled fallback results.
    #[cfg(feature = "nlopt")]
    pub fn fit_experimental_joint_laplace_with_response_constants(
        &mut self,
        verbose: bool,
    ) -> Result<&mut Self> {
        self.fit_joint_glmm_with_response_constants(1, verbose)
    }

    /// Labelled joint GLMM fit with response constants retained.
    ///
    /// For `n_agq <= 1` this is joint Laplace; for `n_agq > 1` this is joint
    /// AGQ and is accepted only for the scalar random-effect shapes permitted
    /// by [`validate_agq`](Self::validate_agq).
    pub fn fit_joint_glmm_with_response_constants(
        &mut self,
        n_agq: usize,
        verbose: bool,
    ) -> Result<&mut Self> {
        if self.lmm.optsum.feval > 0 {
            return Err(MixedModelError::AlreadyFitted);
        }
        self.validate_agq(n_agq)?;
        let joint_optimizer = self
            .lmm
            .optsum
            .caller_selected_optimizer()
            .unwrap_or_else(default_joint_glmm_optimizer);
        validate_joint_glmm_optimizer(joint_optimizer)?;
        let saved_optimizer_source = self.lmm.optsum.optimizer_source;
        let saved_caller_set_fields = self.lmm.optsum.caller_set_fields.clone();

        // Use the supported fast path as the deterministic start. This keeps
        // the joint optimizer focused on whether [β; θ] can improve the same
        // included-constants objective for the requested approximation.
        if self.lmm.optsum.caller_selected_optimizer().is_some() {
            self.configure_profile_start_optimizer();
        }
        self.fit_with_options_impl(n_agq, verbose)?;
        let fallback_fast_pirls = self.clone();
        let start_beta = self.beta.as_slice().to_vec();
        let start_theta = self.theta.clone();
        let profiled_start_objective = self.deviance_with_response_constants(n_agq);
        let n_joint_params = start_beta.len() + start_theta.len();
        self.lmm.optsum.optimizer = joint_optimizer;
        self.lmm.optsum.backend = joint_optimizer.canonical_backend();
        self.lmm.optsum.optimizer_source = saved_optimizer_source;
        self.lmm.optsum.caller_set_fields = saved_caller_set_fields;
        let maxeval =
            joint_glmm_configured_maxeval_for(&self.lmm.optsum, n_joint_params, joint_optimizer);
        self.fit_joint_glmm_from_start(
            start_beta,
            start_theta,
            profiled_start_objective,
            n_agq,
            maxeval,
            Some(fallback_fast_pirls),
        )
    }

    fn fit_joint_glmm_from_start(
        &mut self,
        start_beta: Vec<f64>,
        start_theta: Vec<f64>,
        profiled_start_objective: f64,
        n_agq: usize,
        maxeval: u32,
        fallback_fast_pirls: Option<Self>,
    ) -> Result<&mut Self> {
        let optimizer = self
            .lmm
            .optsum
            .caller_selected_optimizer()
            .unwrap_or_else(|| match self.lmm.optsum.optimizer {
                Optimizer::TrustBq | Optimizer::NloptBobyqa => self.lmm.optsum.optimizer,
                _ => default_joint_glmm_optimizer(),
            });
        self.lmm.optsum.optimizer = optimizer;
        self.lmm.optsum.backend = optimizer.canonical_backend();
        match optimizer {
            Optimizer::TrustBq => self.fit_joint_glmm_from_start_trust_bq(
                start_beta,
                start_theta,
                profiled_start_objective,
                n_agq,
                maxeval,
                fallback_fast_pirls,
            ),
            Optimizer::NloptBobyqa => {
                #[cfg(feature = "nlopt")]
                {
                    self.fit_joint_glmm_from_start_nlopt_bobyqa(
                        start_beta,
                        start_theta,
                        profiled_start_objective,
                        n_agq,
                        maxeval,
                        fallback_fast_pirls,
                    )
                }
                #[cfg(not(feature = "nlopt"))]
                {
                    let _ = (
                        start_beta,
                        start_theta,
                        profiled_start_objective,
                        n_agq,
                        maxeval,
                        fallback_fast_pirls,
                    );
                    Err(MixedModelError::Unsupported(
                        "joint GLMM NloptBobyqa requires the `nlopt` feature; rebuild with `--features nlopt` or pick TrustBq"
                            .to_string(),
                    ))
                }
            }
            optimizer => Err(MixedModelError::Unsupported(format!(
                "Optimizer::{optimizer:?} is not wired for joint GLMM fits; pick TrustBq or NloptBobyqa where available"
            ))),
        }
    }

    #[cfg(feature = "nlopt")]
    fn fit_joint_glmm_from_start_nlopt_bobyqa(
        &mut self,
        start_beta: Vec<f64>,
        start_theta: Vec<f64>,
        profiled_start_objective: f64,
        n_agq: usize,
        maxeval: u32,
        fallback_fast_pirls: Option<Self>,
    ) -> Result<&mut Self> {
        use nlopt::{Algorithm as NloptAlgorithm, Nlopt, Target as NloptTarget};

        let n_beta = self.beta.len();
        let n_theta = self.theta.len();
        let n_params = n_beta + n_theta;
        let mut initial = start_beta;
        initial.extend(start_theta);
        debug_assert_eq!(initial.len(), n_params);

        let mut lower_bounds = vec![f64::NEG_INFINITY; n_beta];
        lower_bounds.extend(self.lmm.lower_bounds());
        self.lmm.optsum.optimizer = Optimizer::NloptBobyqa;
        self.lmm.optsum.backend = Optimizer::NloptBobyqa.canonical_backend();
        self.lmm.optsum.finitial = profiled_start_objective;
        let ftol_rel = if self.lmm.optsum.caller_set_field("ftol_rel") {
            self.lmm.optsum.ftol_rel
        } else {
            1e-10
        };
        let ftol_abs = if self.lmm.optsum.caller_set_field("ftol_abs") {
            self.lmm.optsum.ftol_abs
        } else {
            1e-7
        };
        let xtol_rel = self
            .lmm
            .optsum
            .caller_set_field("xtol_rel")
            .then_some(self.lmm.optsum.xtol_rel);
        let mut initial_step = vec![0.1; n_beta];
        if self.lmm.optsum.caller_set_field("initial_step") {
            initial_step.extend(self.lmm.optsum.initial_step.clone());
        } else {
            initial_step.extend(vec![0.5; n_theta]);
        }

        let feval_count = std::cell::Cell::new(0i64);
        let fit_log: Rc<RefCell<Vec<FitLogEntry>>> = Rc::new(RefCell::new(Vec::new()));
        let model = std::cell::RefCell::new(self);
        let obj_fn = |params: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let objective = model
                .borrow_mut()
                .joint_glmm_deviance_at_params(params, n_beta, n_agq);
            fit_log.borrow_mut().push(FitLogEntry {
                theta: params.to_vec(),
                objective,
            });
            objective
        };

        let mut optimizer = Nlopt::new(
            NloptAlgorithm::Bobyqa,
            n_params,
            obj_fn,
            NloptTarget::Minimize,
            (),
        );
        optimizer.set_lower_bounds(&lower_bounds).ok();
        optimizer.set_ftol_rel(ftol_rel).ok();
        optimizer.set_ftol_abs(ftol_abs).ok();
        if let Some(xtol_rel) = xtol_rel {
            optimizer.set_xtol_rel(xtol_rel).ok();
        }
        optimizer.set_maxeval(maxeval).ok();
        optimizer.set_initial_step(&initial_step).ok();

        let mut params = initial;
        let nlopt_result = optimizer.optimize(&mut params);
        drop(optimizer);

        let me = model.into_inner();
        let final_objective = me.joint_glmm_deviance_at_params(&params, n_beta, n_agq);
        me.refresh_dispersion();
        let status_prefix = joint_glmm_status_prefix(n_agq);
        let status_label = match &nlopt_result {
            Ok((status, _fmin)) => {
                format!(
                    "{status_prefix}:{}",
                    experimental_nlopt_status_label(&format!("{status:?}"))
                )
            }
            Err((status, _fmin)) => {
                format!(
                    "{status_prefix}_FAILED:{}",
                    experimental_nlopt_status_label(&format!("{status:?}"))
                )
            }
        };
        me.lmm.optsum.return_value = status_label;
        me.lmm.optsum.n_agq = n_agq;
        me.lmm.optsum.feval = feval_count.get();
        me.lmm.optsum.max_feval = maxeval as i64;
        me.lmm.optsum.fit_log = rc_refcell_into_inner_or_clone(fit_log);
        me.lmm.optsum.fmin = final_objective;
        me.lmm.optsum.final_params = params;
        let mut lower_bounds = vec![f64::NEG_INFINITY; n_beta];
        lower_bounds.extend(me.lmm.lower_bounds());
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &me.lmm.optsum,
            &me.lmm.optsum.final_params,
            &lower_bounds,
            Some(me.lmm.dims.n),
        );
        let certification_gradient = me.joint_laplace_certification_gradient(
            &me.lmm.optsum.final_params.clone(),
            n_beta,
            n_agq,
            &lower_bounds,
            2.0e-2,
        );
        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                gradient: certification_gradient.gradient.clone(),
                hessian: None,
            },
            2.0e-2,
            1.0e-6,
        );
        annotate_glmm_covariance_status(
            &mut certificate,
            &me.lmm.optsum.final_params,
            n_beta,
            &lower_bounds,
            &certification_gradient,
            2.0e-2,
        );
        if joint_certificate_requires_fallback(&certificate)
            && joint_candidate_materially_improves_profiled_start(&me.lmm.optsum)
        {
            record_uncertified_joint_candidate_diagnostic(&mut certificate, &me.lmm.optsum);
            me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
            me.record_glmm_fit_metadata();
            me.refresh_binomial_separation_diagnostics();
            me.refresh_near_unit_random_effect_correlation_diagnostics();
            return Ok(me);
        }
        if let Some(fallback) =
            uncertified_joint_fallback(&certificate, &me.lmm.optsum, fallback_fast_pirls)
        {
            *me = fallback;
            me.refresh_binomial_separation_diagnostics();
            me.refresh_near_unit_random_effect_correlation_diagnostics();
            return Ok(me);
        }
        me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
        me.record_glmm_fit_metadata();
        me.refresh_binomial_separation_diagnostics();
        me.refresh_near_unit_random_effect_correlation_diagnostics();
        Ok(me)
    }

    fn fit_joint_glmm_from_start_trust_bq(
        &mut self,
        start_beta: Vec<f64>,
        start_theta: Vec<f64>,
        profiled_start_objective: f64,
        n_agq: usize,
        maxeval: u32,
        fallback_fast_pirls: Option<Self>,
    ) -> Result<&mut Self> {
        let mut fallback_fast_pirls = fallback_fast_pirls;
        let n_beta = self.beta.len();
        let n_theta = self.theta.len();
        let n_params = n_beta + n_theta;
        let mut initial = start_beta;
        initial.extend(start_theta);
        debug_assert_eq!(initial.len(), n_params);

        let mut lower_bounds = vec![f64::NEG_INFINITY; n_beta];
        lower_bounds.extend(self.lmm.lower_bounds());
        let upper_bounds = vec![f64::INFINITY; n_params];
        self.lmm.optsum.optimizer = Optimizer::TrustBq;
        self.lmm.optsum.backend = Optimizer::TrustBq.canonical_backend();
        self.lmm.optsum.finitial = profiled_start_objective;
        self.lmm.optsum.max_feval = maxeval as i64;
        let ftol_abs = self.lmm.optsum.ftol_abs.max(1.0e-7);
        let ftol_rel = self.lmm.optsum.ftol_rel.max(1.0e-10);
        let initial_radius = joint_glmm_trust_bq_initial_radius(&initial, n_beta);
        let compact_joint_space = (5..=8).contains(&n_params);

        let invalid_objective = profiled_start_objective.abs().max(1.0)
            + 1.0e6 * (1.0 + profiled_start_objective.abs());
        let best_params = RefCell::new(initial.clone());
        let best_fmin = Cell::new(profiled_start_objective);
        let fit_log: Rc<RefCell<Vec<FitLogEntry>>> = Rc::new(RefCell::new(Vec::new()));

        let model = std::cell::RefCell::new(self);
        let mut objective_fn = |params: &[f64]| -> Result<f64> {
            let raw_objective = model
                .borrow_mut()
                .joint_glmm_deviance_at_params(params, n_beta, n_agq);
            let objective = if raw_objective.is_finite() {
                raw_objective
            } else {
                invalid_objective
            };
            fit_log.borrow_mut().push(FitLogEntry {
                theta: params.to_vec(),
                objective,
            });
            if raw_objective.is_finite() && objective < best_fmin.get() {
                best_fmin.set(objective);
                *best_params.borrow_mut() = params.to_vec();
            }
            Ok(objective)
        };
        let mut progress_fn = |_progress: &TrustBqProgress<'_>| -> Result<bool> { Ok(false) };

        let result = minimize_trust_bq_with_progress(
            &initial,
            &lower_bounds,
            &upper_bounds,
            TrustBqOptions {
                initial_radius,
                final_radius: 1.0e-5,
                max_evaluations: maxeval.max(1) as usize,
                ftol_abs,
                ftol_rel,
                max_cross_terms: if compact_joint_space { usize::MAX } else { 0 },
                stall_iterations: if compact_joint_space { 4 } else { 3 },
                stall_ftol_abs: if compact_joint_space { -1.0 } else { 1.0e-6 },
                stall_ftol_rel: if compact_joint_space { -1.0 } else { 1.0e-8 },
                stall_requires_stable_x: compact_joint_space,
                reuse_samples: true,
                ..TrustBqOptions::default()
            },
            &mut objective_fn,
            &mut progress_fn,
        )?;

        let logged_best_params = best_params.into_inner();
        let logged_best_fmin = best_fmin.get();
        let (mut params, candidate_objective) =
            if logged_best_fmin.is_finite() && logged_best_fmin <= result.fmin {
                (logged_best_params, logged_best_fmin)
            } else {
                (result.x, result.fmin)
            };
        let me = model.into_inner();
        let status_prefix = joint_glmm_status_prefix(n_agq);
        me.lmm.optsum.return_value = format!(
            "{status_prefix}:{}",
            trust_bq_status_label(result.stop_reason)
        );
        me.lmm.optsum.n_agq = n_agq;
        me.lmm.optsum.feval = result.fevals as i64;
        me.lmm.optsum.max_feval = maxeval as i64;
        me.lmm.optsum.fit_log = rc_refcell_into_inner_or_clone(fit_log);
        me.lmm.optsum.fmin = candidate_objective;
        me.lmm.optsum.final_trust_radius = Some(result.final_radius);
        me.lmm.optsum.final_params = params.clone();

        let mut lower_bounds = vec![f64::NEG_INFINITY; n_beta];
        lower_bounds.extend(me.lmm.lower_bounds());
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &me.lmm.optsum,
            &me.lmm.optsum.final_params,
            &lower_bounds,
            Some(me.lmm.dims.n),
        );
        let optimizer_stop_requires_fallback = !certificate.evidence.optimizer_stop.acceptable_stop;
        if optimizer_stop_requires_fallback
            && joint_candidate_materially_improves_profiled_start(&me.lmm.optsum)
        {
            let final_objective = me.joint_glmm_deviance_at_params(&params, n_beta, n_agq);
            me.refresh_dispersion();
            me.lmm.optsum.fmin = final_objective;
            me.lmm.optsum.final_params = std::mem::take(&mut params);
            certificate = OptimizerCertificate::from_opt_summary_with_context(
                &me.lmm.optsum,
                &me.lmm.optsum.final_params,
                &lower_bounds,
                Some(me.lmm.dims.n),
            );
            record_uncertified_joint_candidate_diagnostic(&mut certificate, &me.lmm.optsum);
            me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
            me.record_glmm_fit_metadata();
            me.refresh_binomial_separation_diagnostics();
            me.refresh_near_unit_random_effect_correlation_diagnostics();
            return Ok(me);
        }
        if fallback_fast_pirls.is_some() && optimizer_stop_requires_fallback {
            if let Some(fallback) =
                uncertified_joint_fallback(&certificate, &me.lmm.optsum, fallback_fast_pirls.take())
            {
                *me = fallback;
                me.refresh_binomial_separation_diagnostics();
                me.refresh_near_unit_random_effect_correlation_diagnostics();
                return Ok(me);
            }
        }

        if compact_joint_space {
            if let Some(polished) =
                me.polish_joint_laplace_stationarity(&params, &lower_bounds, 4, 2.0e-2)
            {
                params = polished;
            }
        }

        let final_objective = me.joint_glmm_deviance_at_params(&params, n_beta, n_agq);
        me.refresh_dispersion();
        me.lmm.optsum.fmin = final_objective;
        me.lmm.optsum.final_params = std::mem::take(&mut params);
        certificate = OptimizerCertificate::from_opt_summary_with_context(
            &me.lmm.optsum,
            &me.lmm.optsum.final_params,
            &lower_bounds,
            Some(me.lmm.dims.n),
        );
        let mut certification_gradient = me.joint_laplace_certification_gradient(
            &me.lmm.optsum.final_params.clone(),
            n_beta,
            n_agq,
            &lower_bounds,
            2.0e-2,
        );
        // trust_bq's derivative-free ftol stop can rest a steep, narrow
        // valley's width (~1e-3 deviance) short of the stationary point, where
        // the *assessed* gradient is genuinely above tolerance even though the
        // fit is reference-equivalent to several decimals. That failure is
        // polishable: take damped Newton steps to the stationary point and
        // re-certify, instead of surfacing fit_status=not_optimized on a fit
        // the polish can finish.
        if certificate.evidence.optimizer_stop.acceptable_stop
            && certification_gradient_assessed_free_failure(
                &certification_gradient,
                &me.lmm.optsum.final_params,
                &lower_bounds,
                2.0e-2,
            )
        {
            if let Some(polished) = me.polish_joint_laplace_stationarity(
                &me.lmm.optsum.final_params.clone(),
                &lower_bounds,
                4,
                2.0e-2,
            ) {
                let polished_objective = me.joint_glmm_deviance_at_params(&polished, n_beta, n_agq);
                if polished_objective.is_finite() && polished_objective <= me.lmm.optsum.fmin {
                    me.refresh_dispersion();
                    me.lmm.optsum.fmin = polished_objective;
                    me.lmm.optsum.final_params = polished;
                    certificate = OptimizerCertificate::from_opt_summary_with_context(
                        &me.lmm.optsum,
                        &me.lmm.optsum.final_params,
                        &lower_bounds,
                        Some(me.lmm.dims.n),
                    );
                    certification_gradient = me.joint_laplace_certification_gradient(
                        &me.lmm.optsum.final_params.clone(),
                        n_beta,
                        n_agq,
                        &lower_bounds,
                        2.0e-2,
                    );
                }
            }
        }
        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                gradient: certification_gradient.gradient.clone(),
                hessian: None,
            },
            2.0e-2,
            1.0e-6,
        );
        annotate_glmm_covariance_status(
            &mut certificate,
            &me.lmm.optsum.final_params,
            n_beta,
            &lower_bounds,
            &certification_gradient,
            2.0e-2,
        );
        if joint_certificate_requires_fallback(&certificate)
            && joint_candidate_materially_improves_profiled_start(&me.lmm.optsum)
        {
            record_uncertified_joint_candidate_diagnostic(&mut certificate, &me.lmm.optsum);
            me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
            me.record_glmm_fit_metadata();
            me.refresh_binomial_separation_diagnostics();
            me.refresh_near_unit_random_effect_correlation_diagnostics();
            return Ok(me);
        }
        if let Some(fallback) =
            uncertified_joint_fallback(&certificate, &me.lmm.optsum, fallback_fast_pirls)
        {
            *me = fallback;
            me.refresh_binomial_separation_diagnostics();
            me.refresh_near_unit_random_effect_correlation_diagnostics();
            return Ok(me);
        }
        me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
        me.record_glmm_fit_metadata();
        me.refresh_binomial_separation_diagnostics();
        me.refresh_near_unit_random_effect_correlation_diagnostics();
        Ok(me)
    }

    fn joint_glmm_deviance_at_params(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
    ) -> f64 {
        if params.len() != n_beta + self.theta.len()
            || !params.iter().all(|value| value.is_finite())
        {
            return f64::INFINITY;
        }
        self.beta = DVector::from_column_slice(&params[..n_beta]);
        let theta = &params[n_beta..];
        match self.update_pirls_at_theta(theta, false) {
            Ok(_) => {
                let deviance = self.deviance_with_response_constants(n_agq);
                if deviance.is_finite() {
                    deviance
                } else {
                    f64::INFINITY
                }
            }
            Err(_) => f64::INFINITY,
        }
    }

    fn joint_glmm_deviance_at_params_for_hessian(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
    ) -> std::result::Result<f64, String> {
        if params.len() != n_beta + self.theta.len() {
            return Err(format!(
                "parameter vector has length {}, expected {} fixed effects plus {} covariance parameters",
                params.len(),
                n_beta,
                self.theta.len()
            ));
        }
        if !params.iter().all(|value| value.is_finite()) {
            return Err("parameter vector contains non-finite values".to_string());
        }

        self.beta = DVector::from_column_slice(&params[..n_beta]);
        let theta = &params[n_beta..];
        self.update_pirls_at_theta_with_options(theta, false, GLMM_HESSIAN_PIRLS_MAX_ITER, true)
            .map_err(|error| format!("conditional-mode PIRLS probe failed: {error}"))?;

        let deviance = self.deviance_with_response_constants(n_agq);
        if deviance.is_finite() {
            Ok(deviance)
        } else {
            Err("probe objective is non-finite after the certification PIRLS probe".to_string())
        }
    }

    fn joint_laplace_finite_difference_gradient(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
        lower_bounds: &[f64],
    ) -> Vec<f64> {
        let gradient = (0..params.len())
            .map(|index| {
                let h = JOINT_LAPLACE_FD_RELATIVE_STEP * params[index].abs().max(1.0);
                self.joint_laplace_fd_gradient_component(
                    params,
                    index,
                    h,
                    n_beta,
                    n_agq,
                    lower_bounds,
                )
            })
            .collect();
        let _ = self.joint_glmm_deviance_at_params(params, n_beta, n_agq);
        gradient
    }

    fn joint_laplace_fd_gradient_component(
        &mut self,
        params: &[f64],
        index: usize,
        h: f64,
        n_beta: usize,
        n_agq: usize,
        lower_bounds: &[f64],
    ) -> f64 {
        let value = params[index];
        let lower = lower_bounds
            .get(index)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let mut plus = params.to_vec();
        plus[index] = value + h;
        let fp = self.joint_glmm_deviance_at_params(&plus, n_beta, n_agq);
        if value - h > lower {
            let mut minus = params.to_vec();
            minus[index] = value - h;
            let fm = self.joint_glmm_deviance_at_params(&minus, n_beta, n_agq);
            (fp - fm) / (2.0 * h)
        } else {
            let base = self.joint_glmm_deviance_at_params(params, n_beta, n_agq);
            (fp - base) / h
        }
    }

    /// Stationarity gradient for the joint-Laplace certificate, robust to the
    /// inner-PIRLS deviance noise floor.
    ///
    /// The deviance returned by a PIRLS solve carries an O(1e-5) absolute
    /// error from its own stopping rule, so a finite difference at the default
    /// step `1e-5 * scale` amplifies that error to an O(0.1-1) gradient
    /// reading in directions where the surface is nearly flat — exactly the
    /// directions a converged fit produces. Components whose default-step
    /// reading exceeds the tolerance are therefore re-probed at two larger
    /// steps where the deviance signal dominates the PIRLS noise. If the two
    /// large-step estimates agree, that estimate is the assessed gradient
    /// (which may still fail the tolerance — a genuine non-stationarity). If
    /// they disagree, the component cannot be assessed at any trusted step and
    /// is reported as such rather than as a failure.
    fn joint_laplace_certification_gradient(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
        lower_bounds: &[f64],
        gradient_tolerance: f64,
    ) -> JointLaplaceCertificationGradient {
        let probe_gradient =
            self.joint_laplace_finite_difference_gradient(params, n_beta, n_agq, lower_bounds);
        let mut gradient = probe_gradient.clone();
        let mut escalated_indices = Vec::new();
        let mut unassessable_indices = Vec::new();
        for (index, &value) in params.iter().enumerate() {
            let raw = probe_gradient[index];
            if raw.is_finite() && raw.abs() <= gradient_tolerance {
                continue;
            }
            let scale = value.abs().max(1.0);
            let estimates = JOINT_LAPLACE_CERT_FD_ESCALATED_RELATIVE_STEPS.map(|step| {
                self.joint_laplace_fd_gradient_component(
                    params,
                    index,
                    step * scale,
                    n_beta,
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
        if !(escalated_indices.is_empty() && unassessable_indices.is_empty()) {
            let _ = self.joint_glmm_deviance_at_params(params, n_beta, n_agq);
        }
        JointLaplaceCertificationGradient {
            gradient,
            probe_gradient,
            escalated_indices,
            unassessable_indices,
        }
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

    fn reset_for_refit(&mut self, new_y: Option<&[f64]>) -> Result<()> {
        if let Some(new_y) = new_y {
            if new_y.len() != self.y.len() {
                return Err(MixedModelError::InvalidArgument(format!(
                    "Response length {} does not match model ({} observations)",
                    new_y.len(),
                    self.y.len()
                )));
            }
            validate_glmm_response_domain(self.family, self.link, new_y)?;
            let y_max = new_y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let y_min = new_y.iter().copied().fold(f64::INFINITY, f64::min);
            if (y_max - y_min) < f64::EPSILON {
                return Err(MixedModelError::InvalidArgument(
                    "response is constant; GLMM refit requires variation in the response"
                        .to_string(),
                ));
            }

            let p = self.lmm.feterm.rank;
            for obs in 0..self.y.len() {
                let sw = if self.lmm.sqrtwts.is_empty() {
                    1.0
                } else {
                    self.lmm.sqrtwts[obs]
                };
                self.y[obs] = new_y[obs];
                self.lmm.y[obs] = new_y[obs];
                self.lmm.xy_mat.xy[(obs, p)] = new_y[obs];
                self.lmm.xy_mat.wtxy[(obs, p)] = sw * new_y[obs];
            }
            self.lmm.recompute_a_blocks()?;
        }

        let initial_theta = self.lmm.optsum.initial.clone();
        self.lmm.set_theta(&initial_theta)?;
        self.lmm.update_l()?;
        self.theta = initial_theta.clone();

        self.beta = DVector::zeros(self.lmm.feterm.rank);
        self.beta0 = self.beta.clone();
        for u in &mut self.u {
            u.fill(0.0);
        }
        for u0 in &mut self.u0 {
            u0.fill(0.0);
        }
        for b in &mut self.b {
            b.fill(0.0);
        }
        self.eta.fill(0.0);
        self.mu.fill(0.0);
        self.dispersion = 1.0;
        self.update_eta();

        self.lmm.optsum.finitial = f64::INFINITY;
        self.lmm.optsum.final_params = initial_theta;
        self.lmm.optsum.fmin = f64::INFINITY;
        self.lmm.optsum.feval = 0;
        self.lmm.optsum.return_value.clear();
        self.lmm.optsum.fit_log.clear();
        self.lmm.compiler_artifact.optimizer_certificate = None;
        self.lmm.compiler_artifact.glmm_fit_metadata = None;
        self.lmm.compiler_artifact.fixed_effect_covariance_matrix = None;
        self.lmm.compiler_artifact.effective_covariance.clear();
        self.pirls_profiled_optimum_certificate = None;
        Ok(())
    }

    fn record_glmm_fit_metadata(&mut self) {
        let mut metadata = GlmmFitMetadata::from_opt_summary(&self.lmm.optsum);
        if let Some(theta) = self.negative_binomial_theta {
            metadata
                .family_parameters
                .insert("negative_binomial_theta".to_string(), theta);
            metadata
                .family_parameters
                .insert("negative_binomial_variance_power".to_string(), 2.0);
            metadata.family_parameter_sources.insert(
                "negative_binomial_theta".to_string(),
                if self.negative_binomial_estimate_theta {
                    "estimated".to_string()
                } else {
                    "fixed".to_string()
                },
            );
            metadata.family_parameter_sources.insert(
                "negative_binomial_variance_power".to_string(),
                "fixed".to_string(),
            );
        }
        self.record_fast_pirls_parity_scope_diagnostic(&metadata);
        self.record_pirls_profiled_optimum_certificate(&metadata);
        let inference_artifacts = self.glmm_fixed_effect_inference_artifacts(&metadata);
        let inference_availability =
            glmm_inference_availability_for_table(&metadata, &inference_artifacts.table);
        let covariance = inference_artifacts
            .covariance
            .unwrap_or_else(|| self.lmm.glmm_fixed_effect_covariance_matrix());
        self.lmm
            .compiler_artifact
            .model_boundary
            .inference_availability = inference_availability;
        self.lmm.compiler_artifact.glmm_fit_metadata = Some(metadata);
        self.lmm.compiler_artifact.fixed_effect_covariance_matrix = Some(covariance);
        self.lmm.compiler_artifact.fixed_effect_inference_table = Some(inference_artifacts.table);
    }

    /// Run the post-fit profiled-optimum certificate for profiled fast-PIRLS
    /// fits, store the outcome for prediction-variance gating, and record a
    /// provenance diagnostic either way.
    fn record_pirls_profiled_optimum_certificate(&mut self, metadata: &GlmmFitMetadata) {
        if !matches!(
            metadata.estimation_method.as_str(),
            "fast_pirls_profiled" | "fallback_fast_pirls"
        ) {
            self.pirls_profiled_optimum_certificate = None;
            return;
        }
        // Fit drivers can record metadata more than once for the same final
        // fit (e.g. a joint fallback re-labelling a profiled fit); the
        // certificate and its diagnostic are per-fit, not per-recording.
        if self.pirls_profiled_optimum_certificate.is_some() {
            return;
        }
        let outcome = self.certify_pirls_profiled_optimum();

        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::SupportNote,
            DiagnosticSeverity::Info,
            DiagnosticStage::Certification,
            match &outcome {
                Ok(_) => "Fast-PIRLS profiled optimum certificate issued",
                Err(_) => "Fast-PIRLS profiled optimum certificate not issued",
            },
        );
        diagnostic.payload.insert(
            "glmm_pirls_profiled_optimum_certificate".to_string(),
            serde_json::json!(if outcome.is_ok() {
                "issued"
            } else {
                "not_issued"
            }),
        );
        diagnostic.payload.insert(
            "estimation_method".to_string(),
            serde_json::json!(metadata.estimation_method.as_str()),
        );
        match &outcome {
            Ok(certificate) => {
                diagnostic.payload.insert(
                    "gradient_max_abs".to_string(),
                    serde_json::json!(certificate.gradient_max_abs),
                );
                diagnostic.payload.insert(
                    "min_eigenvalue".to_string(),
                    serde_json::json!(certificate.min_eigenvalue),
                );
                diagnostic.payload.insert(
                    "condition_number".to_string(),
                    serde_json::json!(certificate.condition_number),
                );
                if !certificate.escalated_theta_indices.is_empty() {
                    diagnostic.payload.insert(
                        "escalated_theta_indices".to_string(),
                        serde_json::json!(certificate.escalated_theta_indices),
                    );
                }
                if !certificate.boundary_theta_indices.is_empty() {
                    diagnostic.payload.insert(
                        "boundary_theta_indices".to_string(),
                        serde_json::json!(certificate.boundary_theta_indices),
                    );
                }
            }
            Err(reason) => {
                diagnostic
                    .payload
                    .insert("reason".to_string(), serde_json::json!(reason));
            }
        }
        self.lmm.compiler_artifact.diagnostics.push(diagnostic);
        self.pirls_profiled_optimum_certificate = Some(outcome);
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

    fn glmm_joint_laplace_fixed_effect_inference_artifacts(
        &mut self,
    ) -> std::result::Result<GlmmFixedEffectInferenceArtifacts, String> {
        let p = self.beta.len();
        let n_theta = self.theta.len();
        let full_coef_names = self.coef_names();
        if self.lmm.feterm.rank != full_coef_names.len() {
            return Err(
                "joint-laplace GLMM Wald inference is unavailable for rank-deficient fixed effects"
                    .to_string(),
            );
        }

        let params = self.lmm.optsum.final_params.clone();
        if params.len() != p + n_theta {
            return Err(format!(
                "joint-laplace GLMM final parameter vector has length {}, expected {} fixed effects plus {} covariance parameters",
                params.len(),
                p,
                n_theta
            ));
        }

        let mut lower_bounds = vec![f64::NEG_INFINITY; p];
        lower_bounds.extend(self.lmm.lower_bounds());
        let mut active_indices = (0..p).collect::<Vec<_>>();
        let mut omitted_boundary_theta_indices = Vec::new();
        for index in p..params.len() {
            let lower = lower_bounds[index];
            if lower.is_finite() && params[index] <= lower + glmm_hessian_step(params[index]) {
                omitted_boundary_theta_indices.push(index - p + 1);
            } else {
                active_indices.push(index);
            }
        }

        let hessian = self.finite_difference_joint_laplace_hessian_for_indices(
            &params,
            &lower_bounds,
            &active_indices,
            true,
        )?;
        let certification = certify_glmm_joint_hessian(&hessian, "joint-laplace GLMM Hessian")?;
        let beta_covariance = 2.0 * certification.inverse.view((0, 0), (p, p)).into_owned();
        if !matrix_is_finite_local(&beta_covariance) {
            return Err(
                "joint-laplace GLMM fixed-effect covariance contains non-finite entries"
                    .to_string(),
            );
        }
        let full_covariance = unpivot_glmm_fixed_effect_covariance(
            &beta_covariance,
            &self.lmm.feterm.piv,
            full_coef_names.len(),
        );
        let covariance_payload = glmm_joint_laplace_fixed_effect_covariance_matrix(
            full_coef_names.clone(),
            &full_covariance,
            self.lmm.feterm.rank,
            &certification,
            &omitted_boundary_theta_indices,
        )?;
        let inference_notes =
            glmm_joint_laplace_hessian_notes(&certification, &omitted_boundary_theta_indices);

        let normal = Normal::new(0.0, 1.0)
            .map_err(|err| format!("normal reference distribution unavailable: {err}"))?;
        let estimates = self.coef();
        let mut std_errors = vec![f64::NAN; full_coef_names.len()];
        for full_index in 0..full_coef_names.len() {
            let variance = full_covariance[(full_index, full_index)];
            if !variance.is_finite() || variance <= 0.0 {
                return Err(format!(
                    "joint-laplace GLMM fixed-effect covariance has invalid variance for coefficient {}",
                    full_coef_names
                        .get(full_index)
                        .cloned()
                        .unwrap_or_else(|| full_index.to_string())
                ));
            }
            std_errors[full_index] = variance.sqrt();
        }

        let rows = full_coef_names
            .into_iter()
            .enumerate()
            .map(|(index, label)| {
                let estimate = estimates
                    .get(index)
                    .copied()
                    .filter(|value| value.is_finite());
                let std_error = std_errors
                    .get(index)
                    .copied()
                    .filter(|value| value.is_finite() && *value > 0.0);
                let statistic = estimate.zip(std_error).map(|(estimate, se)| estimate / se);
                let p_value = statistic.map(|z| 2.0 * (1.0 - normal.cdf(z.abs())));
                FixedEffectInferenceRow {
                    label: label.clone(),
                    kind: FixedEffectInferenceRowKind::Coefficient,
                    estimate,
                    std_error,
                    numerator_df: None,
                    denominator_df: None,
                    statistic,
                    statistic_name: Some(crate::compiler::FixedEffectStatisticName::Z),
                    p_value,
                    method: FixedEffectInferenceMethod::AsymptoticWaldZ,
                    status: FixedEffectInferenceStatus::Available,
                    reliability: ReliabilityGrade::Moderate,
                    reliability_reason: Some(
                        FixedEffectReliabilityReason::GlmmJointLaplaceActiveHessianWald,
                    ),
                    estimability: EstimabilityAssessment::FixedContrast(
                        FixedContrastEstimability::estimable(label, 1, 1),
                    ),
                    reason: None,
                    details: None,
                    notes: inference_notes.clone(),
                }
            })
            .collect();

        Ok(GlmmFixedEffectInferenceArtifacts {
            table: FixedEffectInferenceTable::new(rows),
            covariance: Some(covariance_payload),
        })
    }

    fn finite_difference_joint_laplace_hessian(
        &mut self,
        params: &[f64],
        lower_bounds: &[f64],
    ) -> std::result::Result<DMatrix<f64>, String> {
        let active_indices = (0..params.len()).collect::<Vec<_>>();
        self.finite_difference_joint_laplace_hessian_for_indices(
            params,
            lower_bounds,
            &active_indices,
            false,
        )
    }

    fn finite_difference_joint_laplace_hessian_for_indices(
        &mut self,
        params: &[f64],
        lower_bounds: &[f64],
        active_indices: &[usize],
        use_hessian_certification_probe: bool,
    ) -> std::result::Result<DMatrix<f64>, String> {
        let n = active_indices.len();
        let p = self.beta.len();
        let n_agq = self.lmm.optsum.n_agq;

        macro_rules! eval_hessian_probe {
            ($probe:expr, $context:expr) => {
                if use_hessian_certification_probe {
                    match self.joint_glmm_deviance_at_params_for_hessian($probe, p, n_agq) {
                        Ok(value) => value,
                        Err(reason) => {
                            let _ = self.joint_glmm_deviance_at_params(params, p, n_agq);
                            return Err(format!("{}: {}", $context, reason));
                        }
                    }
                } else {
                    let value = self.joint_glmm_deviance_at_params($probe, p, n_agq);
                    if value.is_finite() {
                        value
                    } else {
                        let _ = self.joint_glmm_deviance_at_params(params, p, n_agq);
                        return Err(format!("{} is non-finite", $context));
                    }
                }
            };
        }

        let base = eval_hessian_probe!(
            params,
            "joint-laplace GLMM Hessian certificate base objective"
        );

        let mut steps = Vec::with_capacity(n);
        for &index in active_indices {
            let value = *params.get(index).ok_or_else(|| {
                format!(
                    "joint-laplace GLMM Hessian active parameter index {} is out of range",
                    index + 1
                )
            })?;
            let h = glmm_hessian_step(value);
            let lower = lower_bounds
                .get(index)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            if lower.is_finite() && value - h <= lower {
                let _ = self.joint_glmm_deviance_at_params(params, p, n_agq);
                return Err(format!(
                    "joint-laplace GLMM Hessian central difference step for parameter {} would cross its lower bound",
                    index + 1
                ));
            }
            steps.push(h);
        }

        let mut hessian = DMatrix::zeros(n, n);
        for active_i in 0..n {
            let i = active_indices[active_i];
            let hi = steps[active_i];
            let mut plus = params.to_vec();
            plus[i] += hi;
            let f_plus = eval_hessian_probe!(
                &plus,
                format!(
                    "joint-laplace GLMM Hessian diagonal plus probe for parameter {}",
                    i + 1
                )
            );
            let mut minus = params.to_vec();
            minus[i] -= hi;
            let f_minus = eval_hessian_probe!(
                &minus,
                format!(
                    "joint-laplace GLMM Hessian diagonal minus probe for parameter {}",
                    i + 1
                )
            );
            hessian[(active_i, active_i)] = (f_plus - 2.0 * base + f_minus) / (hi * hi);

            for active_j in 0..active_i {
                let j = active_indices[active_j];
                let hj = steps[active_j];
                let mut pp = params.to_vec();
                pp[i] += hi;
                pp[j] += hj;
                let f_pp = eval_hessian_probe!(
                    &pp,
                    format!(
                        "joint-laplace GLMM Hessian off-diagonal ++ probe for parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );

                let mut pm = params.to_vec();
                pm[i] += hi;
                pm[j] -= hj;
                let f_pm = eval_hessian_probe!(
                    &pm,
                    format!(
                        "joint-laplace GLMM Hessian off-diagonal +- probe for parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );

                let mut mp = params.to_vec();
                mp[i] -= hi;
                mp[j] += hj;
                let f_mp = eval_hessian_probe!(
                    &mp,
                    format!(
                        "joint-laplace GLMM Hessian off-diagonal -+ probe for parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );

                let mut mm = params.to_vec();
                mm[i] -= hi;
                mm[j] -= hj;
                let f_mm = eval_hessian_probe!(
                    &mm,
                    format!(
                        "joint-laplace GLMM Hessian off-diagonal -- probe for parameters {} and {}",
                        i + 1,
                        j + 1
                    )
                );
                let value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hi * hj);
                hessian[(active_i, active_j)] = value;
                hessian[(active_j, active_i)] = value;
            }
        }
        let _ = self.joint_glmm_deviance_at_params(params, p, n_agq);

        Ok(hessian)
    }

    fn polish_joint_laplace_stationarity(
        &mut self,
        params: &[f64],
        lower_bounds: &[f64],
        max_iterations: usize,
        gradient_tolerance: f64,
    ) -> Option<Vec<f64>> {
        let p = self.beta.len();
        let n_agq = self.lmm.optsum.n_agq;
        let mut current = params.to_vec();
        let mut current_objective = self.joint_glmm_deviance_at_params(&current, p, n_agq);
        if !current_objective.is_finite() {
            return None;
        }

        for _ in 0..max_iterations {
            let certification = self.joint_laplace_certification_gradient(
                &current,
                p,
                n_agq,
                lower_bounds,
                gradient_tolerance,
            );
            // Polish only on assessed gradient signal: components the
            // noise-aware probe could not assess carry no usable descent
            // direction, and Newton steps on probe noise just burn a full
            // finite-difference Hessian before the line search rejects them.
            let mut gradient = certification.gradient;
            for &index in &certification.unassessable_indices {
                gradient[index] = 0.0;
            }
            let free_gradient_norm = gradient
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            if !free_gradient_norm.is_finite() || free_gradient_norm <= gradient_tolerance {
                break;
            }

            let hessian = self
                .finite_difference_joint_laplace_hessian(&current, lower_bounds)
                .ok()?;
            let step = hessian
                .cholesky()
                .map(|cholesky| cholesky.solve(&DVector::from_column_slice(&gradient)))?;
            if !step.iter().all(|value| value.is_finite()) {
                break;
            }
            let step_norm = step.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
            if step_norm <= 1.0e-8 {
                break;
            }

            let mut accepted = None;
            for damping in [1.0, 0.5, 0.25, 0.125, 0.0625] {
                let mut trial = current.clone();
                for (index, value) in trial.iter_mut().enumerate() {
                    *value -= damping * step[index];
                    let lower = lower_bounds
                        .get(index)
                        .copied()
                        .unwrap_or(f64::NEG_INFINITY);
                    if lower.is_finite() && *value <= lower {
                        *value = lower + 1.0e-8;
                    }
                }
                if !trial.iter().all(|value| value.is_finite()) {
                    continue;
                }
                let trial_objective = self.joint_glmm_deviance_at_params(&trial, p, n_agq);
                if trial_objective.is_finite()
                    && trial_objective
                        < current_objective
                            - (1.0e-9 * current_objective.abs().max(1.0)).max(1.0e-9)
                {
                    accepted = Some((trial, trial_objective));
                    break;
                }
            }

            let Some((trial, trial_objective)) = accepted else {
                break;
            };
            current = trial;
            current_objective = trial_objective;
        }

        let _ = self.joint_glmm_deviance_at_params(&current, p, n_agq);
        Some(current)
    }

    fn record_negative_binomial_theta_estimation_metadata(
        &mut self,
        initial_theta: f64,
        final_theta: f64,
        update_iterations: usize,
        converged: bool,
    ) {
        if let Some(metadata) = &mut self.lmm.compiler_artifact.glmm_fit_metadata {
            metadata
                .family_parameters
                .insert("negative_binomial_theta_initial".to_string(), initial_theta);
            metadata.family_parameters.insert(
                "negative_binomial_theta_outer_iterations".to_string(),
                update_iterations as f64,
            );
            metadata.family_parameters.insert(
                "negative_binomial_theta_outer_converged".to_string(),
                if converged { 1.0 } else { 0.0 },
            );
            metadata
                .family_parameters
                .insert("negative_binomial_theta".to_string(), final_theta);
            metadata.family_parameter_sources.insert(
                "negative_binomial_theta".to_string(),
                "estimated".to_string(),
            );
            metadata.family_parameter_sources.insert(
                "negative_binomial_theta_initial".to_string(),
                "method_of_moments_or_caller_start".to_string(),
            );
            metadata.family_parameter_sources.insert(
                "negative_binomial_theta_outer_iterations".to_string(),
                "estimated".to_string(),
            );
            metadata.family_parameter_sources.insert(
                "negative_binomial_theta_outer_converged".to_string(),
                "estimated".to_string(),
            );
        }
    }

    fn record_fast_pirls_parity_scope_diagnostic(&mut self, metadata: &GlmmFitMetadata) {
        if metadata.estimation_method != "fast_pirls_profiled" {
            return;
        }
        let scope = "fast_pirls_not_lme4_joint_parity";
        if self
            .lmm
            .compiler_artifact
            .diagnostics
            .iter()
            .any(|diagnostic| {
                diagnostic.code == DiagnosticCode::SupportNote
                    && diagnostic
                        .payload
                        .get("glmm_parity_scope")
                        .and_then(serde_json::Value::as_str)
                        == Some(scope)
            })
        {
            return;
        }

        let mut diagnostic = Diagnostic::new(
            DiagnosticCode::SupportNote,
            DiagnosticSeverity::Info,
            DiagnosticStage::Certification,
            "Fast-PIRLS GLMM fit is not certified as lme4 joint-Laplace parity",
        )
        .with_suggested_actions(vec![
            "treat this fit as profiled fast-PIRLS evidence, not an lme4 joint-Laplace parity row"
                .to_string(),
            "consult the parity scorecard or downstream mismatch ledger before applying strict lme4 tolerances"
                .to_string(),
        ]);
        diagnostic
            .payload
            .insert("glmm_parity_scope".to_string(), serde_json::json!(scope));
        diagnostic.payload.insert(
            "scorecard_class".to_string(),
            serde_json::json!("documented_divergence"),
        );
        diagnostic.payload.insert(
            "external_engine_parity".to_string(),
            serde_json::json!("not_certified"),
        );
        diagnostic.payload.insert(
            "reference_gate".to_string(),
            serde_json::json!("lme4_joint_laplace"),
        );
        diagnostic.payload.insert(
            "estimation_method".to_string(),
            serde_json::json!(metadata.estimation_method.as_str()),
        );
        diagnostic.payload.insert(
            "objective_definition".to_string(),
            serde_json::json!(metadata.objective_definition.as_str()),
        );
        diagnostic.payload.insert(
            "response_constants".to_string(),
            serde_json::json!(metadata.response_constants.as_str()),
        );
        self.lmm.compiler_artifact.diagnostics.push(diagnostic);
    }

    #[cfg(not(feature = "nlopt"))]
    fn fit_with_options_impl(&mut self, n_agq: usize, _verbose: bool) -> Result<&mut Self> {
        match self.lmm.optsum.optimizer {
            Optimizer::PatternSearch => self.fit_native_pattern_search(n_agq),
            Optimizer::Cobyla => self.fit_native_cobyla(n_agq),
            Optimizer::TrustBq => Err(MixedModelError::Unsupported(
                "TrustBQ is reserved for the dependency-light fast=false joint GLMM path; pick COBYLA or pattern_search for fast-PIRLS GLMMs"
                    .to_string(),
            )),
            Optimizer::NloptBobyqa | Optimizer::NloptNewuoa => Err(MixedModelError::Unsupported(
                "NLopt GLMM optimizers require the `nlopt` feature; rebuild with `--features nlopt` or pick a native optimizer"
                    .to_string(),
            )),
            Optimizer::PrimaBobyqa
            | Optimizer::PrimaCobyla
            | Optimizer::PrimaLincoa
            | Optimizer::PrimaNewuoa => Err(MixedModelError::Unsupported(
                "PRIMA GLMM optimizers are not wired; pick a native optimizer".to_string(),
            )),
        }
    }

    #[cfg(feature = "nlopt")]
    fn fit_with_options_impl(&mut self, n_agq: usize, _verbose: bool) -> Result<&mut Self> {
        use nlopt::{Algorithm as NloptAlgorithm, Nlopt, Target as NloptTarget};

        match self.lmm.optsum.caller_selected_optimizer() {
            Some(Optimizer::PatternSearch) => return self.fit_native_pattern_search(n_agq),
            Some(Optimizer::Cobyla) => return self.fit_native_cobyla(n_agq),
            Some(Optimizer::NloptBobyqa) | None => {}
            Some(Optimizer::TrustBq) => {
                return Err(MixedModelError::Unsupported(
                    "TrustBQ is reserved for fast=false joint GLMM fits; pick Cobyla, pattern_search, or NloptBobyqa for fast-PIRLS GLMMs"
                        .to_string(),
                ));
            }
            Some(Optimizer::NloptNewuoa) => {
                return Err(MixedModelError::Unsupported(
                    "NloptNewuoa is unconstrained and is not wired for bounded fast-PIRLS GLMM theta optimization; pick NloptBobyqa"
                        .to_string(),
                ));
            }
            Some(
                Optimizer::PrimaBobyqa
                | Optimizer::PrimaCobyla
                | Optimizer::PrimaLincoa
                | Optimizer::PrimaNewuoa,
            ) => {
                return Err(MixedModelError::Unsupported(
                    "PRIMA GLMM optimizers are not wired; pick Cobyla, pattern_search, or NloptBobyqa"
                        .to_string(),
                ));
            }
        }

        let n_theta = self.theta.len();
        let lb = self.lmm.lower_bounds();
        let initial_theta = self.lmm.optsum.initial.clone();
        self.lmm.optsum.optimizer = Optimizer::NloptBobyqa;
        self.lmm.optsum.backend = Optimizer::NloptBobyqa.canonical_backend();
        let ftol_rel = if self.lmm.optsum.caller_set_field("ftol_rel") {
            self.lmm.optsum.ftol_rel
        } else {
            1e-12
        };
        let ftol_abs = if self.lmm.optsum.caller_set_field("ftol_abs") {
            self.lmm.optsum.ftol_abs
        } else {
            1e-8
        };
        let xtol_rel = self
            .lmm
            .optsum
            .caller_set_field("xtol_rel")
            .then_some(self.lmm.optsum.xtol_rel);
        let xtol_abs = self
            .lmm
            .optsum
            .caller_set_field("xtol_abs")
            .then(|| self.lmm.optsum.xtol_abs.clone());
        let maxeval = if self.lmm.optsum.max_feval > 0 {
            self.lmm.optsum.max_feval.min(u32::MAX as i64).max(1) as u32
        } else {
            500
        };
        let initial_step = if self.lmm.optsum.caller_set_field("initial_step") {
            self.lmm.optsum.initial_step.clone()
        } else {
            vec![0.75; n_theta]
        };

        let feval_count = std::cell::Cell::new(0i64);
        let fit_log: Rc<RefCell<Vec<FitLogEntry>>> = Rc::new(RefCell::new(Vec::with_capacity(
            self.lmm.optsum.fit_log.capacity(),
        )));

        // Hand the model to the BOBYQA callback through a RefCell instead of a
        // raw `*mut Self`. The callback is the only borrower while the optimizer
        // is alive; `model.into_inner()` recovers `&mut self` once the optimizer
        // (and thus the closure) has been dropped.
        let model = std::cell::RefCell::new(self);
        let obj_fn = |theta: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let objective = model
                .borrow_mut()
                .penalized_pirls_deviance_at_theta(theta, n_agq);
            fit_log.borrow_mut().push(FitLogEntry {
                theta: theta.to_vec(),
                objective,
            });
            objective
        };

        let mut optimizer = Nlopt::new(
            NloptAlgorithm::Bobyqa,
            n_theta,
            obj_fn,
            NloptTarget::Minimize,
            (),
        );
        optimizer.set_lower_bounds(&lb).ok();
        // Match MixedModels.jl OptSummary defaults: ftol_rel=1e-12,
        // ftol_abs=1e-8, xtol_rel=0. Setting xtol_rel=1e-8 here previously
        // forced BOBYQA to shrink its trust region (ρ_beg → ρ_end) all the
        // way to 1e-8 before exploring multi-dim moves, which on multi-RE
        // GLMM surfaces (e.g. grouseticks Poisson) caused premature
        // termination at the initial θ with status `XtolReached`.
        optimizer.set_ftol_rel(ftol_rel).ok();
        optimizer.set_ftol_abs(ftol_abs).ok();
        if let Some(xtol_rel) = xtol_rel {
            optimizer.set_xtol_rel(xtol_rel).ok();
        }
        if let Some(xtol_abs) = &xtol_abs {
            optimizer.set_xtol_abs(xtol_abs).ok();
        }
        optimizer.set_maxeval(maxeval).ok();
        // Mirror the LMM cobyla initial step default; without an explicit
        // initial step BOBYQA falls back to per-axis defaults that may be
        // too small for parameters near the lower bound.
        optimizer.set_initial_step(&initial_step).ok();

        let mut theta = initial_theta;
        let nlopt_result = optimizer.optimize(&mut theta);
        drop(optimizer);

        // Optimizer (and its closure) dropped: reclaim exclusive `&mut self`.
        let me = model.into_inner();
        me.finalize_theta_after_optimizer(&mut theta, n_agq)?;
        me.lmm.optsum.return_value = match nlopt_result {
            Ok((status, _fmin)) => experimental_nlopt_status_label(&format!("{status:?}")),
            Err((status, _fmin)) => {
                format!(
                    "FAILED:{}",
                    experimental_nlopt_status_label(&format!("{status:?}"))
                )
            }
        };
        me.lmm.optsum.feval = feval_count.get();
        me.lmm.optsum.fit_log = rc_refcell_into_inner_or_clone(fit_log);
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &me.lmm.optsum,
            &me.theta,
            &me.lmm.lower_bounds(),
            Some(me.lmm.dims.n),
        );
        annotate_glmm_singular_covariance_status(&mut certificate, &me.theta, me.lmm.is_singular());
        me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
        me.record_glmm_fit_metadata();
        me.refresh_binomial_separation_diagnostics();
        me.refresh_near_unit_random_effect_correlation_diagnostics();

        Ok(me)
    }

    fn fit_native_cobyla(&mut self, n_agq: usize) -> Result<&mut Self> {
        let lb = self.lmm.lower_bounds();
        let initial_theta = self.lmm.optsum.initial.clone();
        self.lmm.optsum.optimizer = Optimizer::Cobyla;
        self.lmm.optsum.backend = Optimizer::Cobyla.canonical_backend();

        let best_theta: Rc<RefCell<Vec<f64>>> = Rc::new(RefCell::new(initial_theta.clone()));
        let best_fmin: Rc<Cell<f64>> = Rc::new(Cell::new(f64::INFINITY));
        let feval_count: Rc<Cell<i64>> = Rc::new(Cell::new(0i64));
        let fit_log: Rc<RefCell<Vec<FitLogEntry>>> = Rc::new(RefCell::new(Vec::with_capacity(
            self.lmm.optsum.fit_log.capacity(),
        )));

        // Compute every `self`-dependent input before handing the model to the
        // optimizer callback, so `self` is free to move into the RefCell.
        let bounds: Vec<(f64, f64)> = lb.iter().map(|&lo| (lo, f64::INFINITY)).collect();
        let constraint_fns: Vec<Box<dyn cobyla::Func<()>>> = lb
            .iter()
            .enumerate()
            .filter(|(_, &lo)| lo > f64::NEG_INFINITY)
            .map(|(i, &lo)| {
                Box::new(move |x: &[f64], _: &mut ()| -> f64 { x[i] - lo })
                    as Box<dyn cobyla::Func<()>>
            })
            .collect();
        let cons_refs: Vec<&dyn cobyla::Func<()>> =
            constraint_fns.iter().map(|f| f.as_ref()).collect();
        let maxeval = if self.lmm.optsum.max_feval > 0 {
            self.lmm.optsum.max_feval as usize
        } else {
            500
        };
        let stop_tol = cobyla::StopTols {
            ftol_rel: self.lmm.optsum.ftol_rel,
            ftol_abs: self.lmm.optsum.ftol_abs,
            xtol_rel: self.lmm.optsum.xtol_rel,
            xtol_abs: self.lmm.optsum.xtol_abs.clone(),
        };

        // Hand the model to the COBYLA callback through a RefCell instead of a
        // raw `*mut Self`. The callback is the only borrower while the optimizer
        // is alive; `model.into_inner()` recovers `&mut self` afterwards.
        let model = std::cell::RefCell::new(self);
        let objective_fn = |theta: &[f64], _data: &mut ()| -> f64 {
            feval_count.set(feval_count.get() + 1);
            let objective = model
                .borrow_mut()
                .penalized_pirls_deviance_at_theta(theta, n_agq);
            fit_log.borrow_mut().push(FitLogEntry {
                theta: theta.to_vec(),
                objective,
            });
            if objective < best_fmin.get() {
                best_fmin.set(objective);
                *best_theta.borrow_mut() = theta.to_vec();
            }
            objective
        };

        let result = cobyla::minimize(
            objective_fn,
            &initial_theta,
            &bounds,
            &cons_refs,
            (),
            maxeval,
            cobyla::RhoBeg::All(0.75),
            Some(stop_tol),
        );

        let (mut theta, return_value) = match result {
            Ok((status, x_opt, fmin)) if fmin.is_finite() => {
                (x_opt, Self::cobyla_success_status_label(status))
            }
            Ok((status, _x_opt, _fmin)) => (
                best_theta.borrow().clone(),
                Self::cobyla_success_status_label(status),
            ),
            Err((status @ cobyla::FailStatus::RoundoffLimited, _x_opt, _fmin)) => (
                best_theta.borrow().clone(),
                Self::cobyla_fail_status_label(status),
            ),
            Err((status, x_opt, fmin)) if fmin.is_finite() => {
                (x_opt, Self::cobyla_fail_status_label(status))
            }
            Err((status, _x_opt, _fmin)) if best_fmin.get().is_finite() => (
                best_theta.borrow().clone(),
                Self::cobyla_fail_status_label(status),
            ),
            Err((_status, _x_opt, _fmin)) => {
                return Err(MixedModelError::Optimization(
                    "COBYLA optimization failed while fitting GLMM".to_string(),
                ));
            }
        };

        // Optimizer finished and consumed its closure: reclaim `&mut self`.
        let me = model.into_inner();
        me.finalize_theta_after_optimizer(&mut theta, n_agq)?;
        me.lmm.optsum.return_value = return_value;
        me.lmm.optsum.feval = feval_count.get();
        me.lmm.optsum.fit_log = rc_refcell_into_inner_or_clone(fit_log);
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &me.lmm.optsum,
            &me.theta,
            &me.lmm.lower_bounds(),
            Some(me.lmm.dims.n),
        );
        annotate_glmm_singular_covariance_status(&mut certificate, &me.theta, me.lmm.is_singular());
        me.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
        me.record_glmm_fit_metadata();
        me.refresh_binomial_separation_diagnostics();
        me.refresh_near_unit_random_effect_correlation_diagnostics();

        Ok(me)
    }

    fn fit_native_pattern_search(&mut self, n_agq: usize) -> Result<&mut Self> {
        let lower_bounds = self.lmm.lower_bounds();
        let n_theta = self.theta.len();
        let maxeval = if self.lmm.optsum.max_feval > 0 {
            self.lmm.optsum.max_feval
        } else {
            500
        };
        let mut step_tol = self.lmm.optsum.xtol_abs.clone();
        if step_tol.len() != n_theta {
            step_tol = vec![1e-5; n_theta];
        }
        for tol in &mut step_tol {
            *tol = tol.max(1e-5);
        }
        let mut step = self.lmm.optsum.initial_step.clone();
        if step.len() != n_theta {
            step = vec![0.75; n_theta];
        }
        for (value, tol) in step.iter_mut().zip(step_tol.iter()) {
            *value = value.abs().max(*tol);
        }

        let mut theta = self.lmm.optsum.initial.clone();
        project_theta_to_bounds(&mut theta, &lower_bounds);
        let mut best_theta = theta.clone();
        let mut best_fmin = f64::INFINITY;
        let mut feval_count = 0i64;
        let mut fit_log = Vec::with_capacity(self.lmm.optsum.fit_log.capacity());
        let mut preferred_sign = vec![-1.0; n_theta];
        for (idx, lower) in lower_bounds.iter().enumerate() {
            if !lower.is_finite() {
                preferred_sign[idx] = 1.0;
            }
        }

        let mut current_f = record_pattern_search_eval(
            self,
            &theta,
            n_agq,
            &mut feval_count,
            &mut fit_log,
            &mut best_theta,
            &mut best_fmin,
        );
        self.lmm.optsum.finitial = current_f;

        while feval_count < maxeval && !steps_are_small(&step, &step_tol) {
            let base_theta = theta.clone();
            let base_f = current_f;
            let mut moved = false;

            for idx in 0..n_theta {
                let mut accepted = false;
                for dir in [preferred_sign[idx], -preferred_sign[idx]] {
                    let mut trial = theta.clone();
                    trial[idx] += dir * step[idx];
                    project_theta_to_bounds(&mut trial, &lower_bounds);
                    if (trial[idx] - theta[idx]).abs() <= step_tol[idx] * 0.5 {
                        continue;
                    }
                    let ftrial = record_pattern_search_eval(
                        self,
                        &trial,
                        n_agq,
                        &mut feval_count,
                        &mut fit_log,
                        &mut best_theta,
                        &mut best_fmin,
                    );
                    if ftrial + self.lmm.optsum.ftol_abs < current_f {
                        theta = trial;
                        current_f = ftrial;
                        preferred_sign[idx] = dir;
                        step[idx] = (step[idx] * 1.1).max(step_tol[idx]);
                        moved = true;
                        accepted = true;
                        break;
                    }
                    if feval_count >= maxeval {
                        break;
                    }
                }
                if !accepted {
                    preferred_sign[idx] = -preferred_sign[idx];
                    step[idx] *= 0.5;
                }
                if feval_count >= maxeval {
                    break;
                }
            }

            if moved && feval_count < maxeval {
                let mut pattern = theta.clone();
                for idx in 0..n_theta {
                    pattern[idx] += theta[idx] - base_theta[idx];
                }
                project_theta_to_bounds(&mut pattern, &lower_bounds);
                if pattern != theta {
                    let fpattern = record_pattern_search_eval(
                        self,
                        &pattern,
                        n_agq,
                        &mut feval_count,
                        &mut fit_log,
                        &mut best_theta,
                        &mut best_fmin,
                    );
                    if fpattern + self.lmm.optsum.ftol_abs < current_f {
                        theta = pattern;
                        current_f = fpattern;
                    }
                }
            }

            if !moved {
                for value in &mut step {
                    *value *= 0.5;
                }
            }
            if (base_f - current_f).abs() <= self.lmm.optsum.ftol_abs
                && steps_are_small(&step, &step_tol)
            {
                break;
            }
        }

        self.lmm.optsum.optimizer = Optimizer::PatternSearch;
        self.lmm.optsum.backend = Optimizer::PatternSearch.canonical_backend();
        self.finalize_theta_after_optimizer(&mut best_theta, n_agq)?;
        self.lmm.optsum.return_value = if feval_count >= maxeval {
            "MAXEVAL_REACHED".to_string()
        } else {
            "SUCCESS".to_string()
        };
        self.lmm.optsum.feval = feval_count;
        self.lmm.optsum.fit_log = fit_log;
        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &self.lmm.optsum,
            &self.theta,
            &self.lmm.lower_bounds(),
            Some(self.lmm.dims.n),
        );
        annotate_glmm_singular_covariance_status(
            &mut certificate,
            &self.theta,
            self.lmm.is_singular(),
        );
        self.lmm.compiler_artifact.optimizer_certificate = Some(certificate);
        self.record_glmm_fit_metadata();
        self.refresh_binomial_separation_diagnostics();
        self.refresh_near_unit_random_effect_correlation_diagnostics();

        Ok(self)
    }

    fn finalize_theta_after_optimizer(&mut self, theta: &mut [f64], n_agq: usize) -> Result<()> {
        LinearMixedModel::rectify_theta_columns(theta, &self.lmm.parmap, self.lmm.reterms.len());

        // Final PIRLS at optimal θ, after matching MixedModels.jl's
        // post-optimizer sign convention for Cholesky columns.
        let pirls_converged = match self.update_pirls_at_theta(theta, true) {
            Ok(converged) => converged,
            Err(error) => {
                self.record_pirls_failure_diagnostic(theta, &error.to_string());
                return Err(error);
            }
        };
        if !pirls_converged {
            // Not a hard failure (Julia also returns a model here), but the
            // unverified modes must be observable rather than silently
            // accepted as a good fit (audit 03·H1).
            self.record_pirls_nonconvergence_diagnostic(theta);
        }
        self.beta = self.lmm.beta();
        self.refresh_dispersion();

        self.lmm.optsum.n_agq = n_agq;
        self.lmm.optsum.fmin = self.deviance(n_agq);
        self.lmm.optsum.final_params = theta.to_vec();
        Ok(())
    }

    /// Returns whether the inner PIRLS converged (see [`Self::pirls`]).
    fn update_pirls_at_theta(&mut self, theta: &[f64], vary_beta: bool) -> Result<bool> {
        self.update_pirls_at_theta_with_options(theta, vary_beta, GLMM_PIRLS_MAX_ITER, true)
    }

    fn update_pirls_at_theta_with_options(
        &mut self,
        theta: &[f64],
        vary_beta: bool,
        max_iter: usize,
        reset_modes: bool,
    ) -> Result<bool> {
        if theta.len() != self.theta.len() {
            return Err(MixedModelError::DimensionMismatch(format!(
                "theta vector length {} does not match fitted GLMM theta length {}",
                theta.len(),
                self.theta.len()
            )));
        }
        if !theta.iter().all(|value| value.is_finite()) {
            return Err(MixedModelError::InvalidArgument(
                "theta values must be finite".to_string(),
            ));
        }
        self.lmm.set_theta(theta)?;
        self.lmm.update_l()?;
        self.theta = theta.to_vec();
        let converged = self.pirls_with_options(vary_beta, false, max_iter, reset_modes)?;
        Ok(converged)
    }

    fn refresh_dispersion(&mut self) {
        self.dispersion = self.estimated_dispersion_scale();
    }

    fn estimated_dispersion_scale(&self) -> f64 {
        if let Some(theta) = self.negative_binomial_theta {
            return theta;
        }
        if !self.family.has_dispersion() {
            return 1.0;
        }

        let pearson = self.pearson_dispersion_numerator();
        let denom = self.y.len().saturating_sub(self.lmm.feterm.rank).max(1) as f64;
        let variance = (pearson / denom).max(f64::MIN_POSITIVE);
        variance.sqrt()
    }

    fn estimate_negative_binomial_theta_given_fit(&self) -> Result<f64> {
        if self.family != Family::NegativeBinomial {
            return Err(MixedModelError::InvalidArgument(
                "negative-binomial theta estimation is only valid for Family::NegativeBinomial"
                    .to_string(),
            ));
        }
        let weights = (!self.wt.is_empty()).then_some(self.wt.as_slice());
        Ok(estimate_negative_binomial_theta_conditional(
            self.y.as_slice(),
            self.mu.as_slice(),
            weights,
        ))
    }

    fn pearson_dispersion_numerator(&self) -> f64 {
        let mut total = 0.0;
        for obs in 0..self.y.len() {
            let mu = self.mu[obs];
            let variance = self.variance(mu);
            if !variance.is_finite() || variance <= 0.0 {
                continue;
            }
            let residual = self.y[obs] - mu;
            total += self.case_weight(obs) * residual * residual / variance;
        }
        total
    }

    fn penalized_pirls_deviance_at_theta(&mut self, theta: &[f64], n_agq: usize) -> f64 {
        match self.update_pirls_at_theta(theta, true) {
            Ok(_) => {
                let deviance = self.deviance(n_agq);
                if deviance.is_finite() {
                    deviance
                } else {
                    f64::INFINITY
                }
            }
            Err(_) => f64::INFINITY,
        }
    }

    fn refresh_near_unit_random_effect_correlation_diagnostics(&mut self) {
        const NEAR_UNIT_CORR_THRESHOLD: f64 = 0.99;

        let varcorr = self.varcorr();
        let mut diagnostics = Vec::new();
        for component in &varcorr.components {
            for (offset, &corr) in component.correlations.iter().enumerate() {
                if corr.abs() < NEAR_UNIT_CORR_THRESHOLD {
                    continue;
                }
                let (row, col) = lower_triangle_pair(offset);
                let row_name = component
                    .names
                    .get(row)
                    .cloned()
                    .unwrap_or_else(|| format!("basis[{row}]"));
                let col_name = component
                    .names
                    .get(col)
                    .cloned()
                    .unwrap_or_else(|| format!("basis[{col}]"));
                let mut diagnostic = Diagnostic::new(
                    DiagnosticCode::NearUnitRandomEffectCorrelation,
                    DiagnosticSeverity::Warning,
                    DiagnosticStage::Certification,
                    format!(
                        "random-effect correlation for group {} between {} and {} is {:.3}; the fitted covariance is nearly one-dimensional",
                        component.group, col_name, row_name, corr
                    ),
                )
                .with_affected_terms(vec![component.group.clone()])
                .with_suggested_actions(vec![
                    "consider a zero-correlation (`||`) or reduced-rank random-effect structure".to_string(),
                    "treat correlation estimates and Hessian-based standard errors cautiously".to_string(),
                ]);
                diagnostic
                    .payload
                    .insert("group".to_string(), serde_json::json!(component.group));
                diagnostic
                    .payload
                    .insert("correlation".to_string(), serde_json::json!(corr));
                diagnostic.payload.insert(
                    "threshold".to_string(),
                    serde_json::json!(NEAR_UNIT_CORR_THRESHOLD),
                );
                diagnostics.push(diagnostic);
            }
        }

        if diagnostics.is_empty() {
            return;
        }

        self.lmm
            .compiler_artifact
            .diagnostics
            .extend(diagnostics.clone());
        if let Some(certificate) = &mut self.lmm.compiler_artifact.optimizer_certificate {
            certificate.diagnostics.extend(diagnostics);
        }
    }

    fn refresh_binomial_separation_diagnostics(&mut self) {
        self.lmm
            .compiler_artifact
            .diagnostics
            .retain(|diagnostic| diagnostic.code != DiagnosticCode::BinomialSeparation);
        if let Some(certificate) = &mut self.lmm.compiler_artifact.optimizer_certificate {
            certificate
                .diagnostics
                .retain(|diagnostic| diagnostic.code != DiagnosticCode::BinomialSeparation);
        }

        let diagnostics = self.conservative_binomial_separation_diagnostics();
        if diagnostics.is_empty() {
            return;
        }

        self.lmm
            .compiler_artifact
            .diagnostics
            .extend(diagnostics.clone());
        if let Some(certificate) = &mut self.lmm.compiler_artifact.optimizer_certificate {
            certificate.diagnostics.extend(diagnostics);
        }
    }

    fn conservative_binomial_separation_diagnostics(&self) -> Vec<Diagnostic> {
        if !matches!(self.family, Family::Bernoulli | Family::Binomial)
            || !self.y.iter().all(|value| is_binary_response(*value))
        {
            return Vec::new();
        }

        let mut diagnostics = Vec::new();
        for column_index in 0..self.lmm.feterm.rank {
            let column_name = self
                .lmm
                .feterm
                .cnames
                .get(column_index)
                .cloned()
                .unwrap_or_else(|| format!("fixed_effect[{column_index}]"));
            if is_intercept_column(&column_name) {
                continue;
            }

            let column_values = self.lmm.feterm.x.column(column_index);
            let Some(split) = binary_column_split(column_values.iter().copied()) else {
                continue;
            };

            let low_counts = outcome_counts_for_value(
                column_values.iter().copied(),
                self.y.iter().copied(),
                split.low,
            );
            let high_counts = outcome_counts_for_value(
                column_values.iter().copied(),
                self.y.iter().copied(),
                split.high,
            );

            if let Some(diagnostic) =
                separation_diagnostic_for_side(&column_name, split.low, low_counts, high_counts)
            {
                diagnostics.push(diagnostic);
            }
            if let Some(diagnostic) =
                separation_diagnostic_for_side(&column_name, split.high, high_counts, low_counts)
            {
                diagnostics.push(diagnostic);
            }
        }

        diagnostics
    }

    fn cobyla_success_status_label(status: cobyla::SuccessStatus) -> String {
        match status {
            cobyla::SuccessStatus::Success => "SUCCESS".to_string(),
            cobyla::SuccessStatus::StopValReached => "STOPVAL_REACHED".to_string(),
            cobyla::SuccessStatus::FtolReached => "FTOL_REACHED".to_string(),
            cobyla::SuccessStatus::XtolReached => "XTOL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxEvalReached => "MAXEVAL_REACHED".to_string(),
            cobyla::SuccessStatus::MaxTimeReached => "MAXTIME_REACHED".to_string(),
        }
    }

    fn cobyla_fail_status_label(status: cobyla::FailStatus) -> String {
        match status {
            cobyla::FailStatus::Failure => "FAILURE".to_string(),
            cobyla::FailStatus::InvalidArgs => "INVALID_ARGS".to_string(),
            cobyla::FailStatus::OutOfMemory => "OUT_OF_MEMORY".to_string(),
            cobyla::FailStatus::RoundoffLimited => "ROUNDOFF_LIMITED".to_string(),
            cobyla::FailStatus::ForcedStop => "FORCED_STOP".to_string(),
            cobyla::FailStatus::UnexpectedError => "UNEXPECTED_ERROR".to_string(),
        }
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

fn project_theta_to_bounds(theta: &mut [f64], lower_bounds: &[f64]) {
    for (value, lower) in theta.iter_mut().zip(lower_bounds.iter()) {
        if lower.is_finite() && *value < *lower {
            *value = *lower;
        }
    }
}

fn steps_are_small(step: &[f64], step_tol: &[f64]) -> bool {
    step.iter()
        .zip(step_tol.iter())
        .all(|(step, tol)| *step <= *tol)
}

fn record_pattern_search_eval(
    model: &mut GeneralizedLinearMixedModel,
    theta: &[f64],
    n_agq: usize,
    feval_count: &mut i64,
    fit_log: &mut Vec<FitLogEntry>,
    best_theta: &mut Vec<f64>,
    best_fmin: &mut f64,
) -> f64 {
    *feval_count += 1;
    let objective = model.penalized_pirls_deviance_at_theta(theta, n_agq);
    fit_log.push(FitLogEntry {
        theta: theta.to_vec(),
        objective,
    });
    if objective < *best_fmin {
        *best_fmin = objective;
        *best_theta = theta.to_vec();
    }
    objective
}

fn lower_triangle_pair(offset: usize) -> (usize, usize) {
    let mut row = 1usize;
    let mut remaining = offset;
    while remaining >= row {
        remaining -= row;
        row += 1;
    }
    (row, remaining)
}

fn default_fast_glmm_optimizer() -> Optimizer {
    #[cfg(feature = "nlopt")]
    {
        Optimizer::NloptBobyqa
    }
    #[cfg(not(feature = "nlopt"))]
    {
        Optimizer::Cobyla
    }
}

fn default_joint_glmm_optimizer() -> Optimizer {
    #[cfg(feature = "nlopt")]
    {
        Optimizer::NloptBobyqa
    }
    #[cfg(not(feature = "nlopt"))]
    {
        Optimizer::TrustBq
    }
}

fn trust_bq_joint_glmm_default_maxeval(n_params: usize) -> u32 {
    // Native TrustBQ uses local quadratic models, so it should need far fewer
    // objective calls than the previous COBYLA fallback while still leaving
    // enough budget for mixed beta/theta scales on large-intercept Bernoulli
    // models.
    (500usize + 80usize * n_params.max(1)).min(8_000) as u32
}

fn joint_glmm_default_maxeval_for(optimizer: Optimizer, n_params: usize) -> u32 {
    match optimizer {
        Optimizer::TrustBq => trust_bq_joint_glmm_default_maxeval(n_params),
        Optimizer::NloptBobyqa => 200,
        _ => trust_bq_joint_glmm_default_maxeval(n_params),
    }
}

fn joint_glmm_configured_maxeval_for(
    optsum: &OptSummary,
    n_params: usize,
    optimizer: Optimizer,
) -> u32 {
    if optsum.max_feval > 0 {
        optsum.max_feval.min(u32::MAX as i64).max(1) as u32
    } else {
        joint_glmm_default_maxeval_for(optimizer, n_params)
    }
}

fn validate_joint_glmm_optimizer(optimizer: Optimizer) -> Result<()> {
    match optimizer {
        Optimizer::TrustBq => Ok(()),
        Optimizer::NloptBobyqa => {
            #[cfg(feature = "nlopt")]
            {
                Ok(())
            }
            #[cfg(not(feature = "nlopt"))]
            {
                Err(MixedModelError::Unsupported(
                    "joint GLMM NloptBobyqa requires the `nlopt` feature; rebuild with `--features nlopt` or pick TrustBq"
                        .to_string(),
                ))
            }
        }
        other => Err(MixedModelError::Unsupported(format!(
            "Optimizer::{other:?} is not wired for joint GLMM fits; pick TrustBq or NloptBobyqa where available"
        ))),
    }
}

fn joint_glmm_trust_bq_initial_radius(initial: &[f64], n_beta: usize) -> f64 {
    let beta_scale = initial
        .iter()
        .take(n_beta)
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    // Keep beta moves large enough to repair high-baseline intercept starts,
    // but do not let one large coefficient make theta probes excessive.
    (0.25 * beta_scale).clamp(0.25, 1.0)
}

fn trust_bq_status_label(status: TrustBqStopReason) -> &'static str {
    match status {
        TrustBqStopReason::RadiusBelowTolerance => "RADIUS_REACHED",
        TrustBqStopReason::ObjectiveTolerance => "FTOL_REACHED",
        TrustBqStopReason::MaxEvaluations => "MAXEVAL_REACHED",
        TrustBqStopReason::StepBelowTolerance => "XTOL_REACHED",
        TrustBqStopReason::ObjectiveStagnation => "FTOL_REACHED",
        TrustBqStopReason::CertifiedConvergence => "FTOL_REACHED",
    }
}

fn glmm_block_index(row: usize, col: usize) -> usize {
    debug_assert!(row >= col);
    row * (row + 1) / 2 + col
}

fn solve_dense_lower_against_rhs(l: &DMatrix<f64>, rhs: &mut [f64]) {
    for i in 0..rhs.len() {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= l[(i, j)] * rhs[j];
        }
        rhs[i] = sum / l[(i, i)];
    }
}

fn solve_dense_upper_from_lower_transpose_against_rhs(l: &DMatrix<f64>, rhs: &mut [f64]) {
    for i in (0..rhs.len()).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..rhs.len() {
            sum -= l[(j, i)] * rhs[j];
        }
        rhs[i] = sum / l[(i, i)];
    }
}

#[cfg(feature = "nlopt")]
fn experimental_nlopt_status_label(name: &str) -> String {
    match name {
        "Success" => "SUCCESS".to_string(),
        "StopValReached" => "STOPVAL_REACHED".to_string(),
        "FtolReached" => "FTOL_REACHED".to_string(),
        "XtolReached" => "XTOL_REACHED".to_string(),
        "MaxEvalReached" => "MAXEVAL_REACHED".to_string(),
        "MaxTimeReached" => "MAXTIME_REACHED".to_string(),
        "RoundoffLimited" => "ROUNDOFF_LIMITED".to_string(),
        "InvalidArgs" => "INVALID_ARGS".to_string(),
        "OutOfMemory" => "OUT_OF_MEMORY".to_string(),
        "ForcedStop" => "FORCED_STOP".to_string(),
        "Failure" => "FAILURE".to_string(),
        other => other.to_ascii_uppercase(),
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
        self.lmm.vcov()
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
