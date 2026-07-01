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

fn joint_glmm_status_prefix(n_agq: usize) -> &'static str {
    if n_agq <= 1 {
        "JOINT_LAPLACE"
    } else {
        "JOINT_AGQ"
    }
}

fn glmm_objective_includes_response_constants(return_value: &str) -> bool {
    return_value.starts_with("JOINT_LAPLACE:")
        || return_value.starts_with("JOINT_LAPLACE_FAILED:")
        || return_value.starts_with("JOINT_AGQ:")
        || return_value.starts_with("JOINT_AGQ_FAILED:")
        || return_value.starts_with("EXPERIMENTAL_JOINT:")
        || return_value.starts_with("EXPERIMENTAL_JOINT_FAILED:")
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

#[derive(Debug, Clone, Copy)]
struct BinaryColumnSplit {
    low: f64,
    high: f64,
}

#[derive(Debug, Clone, Copy)]
struct OutcomeCounts {
    n: usize,
    successes: usize,
    failures: usize,
}

fn is_binary_response(value: f64) -> bool {
    (value - 0.0).abs() < 1e-12 || (value - 1.0).abs() < 1e-12
}

fn is_nonnegative_integer_response(value: f64) -> bool {
    value >= 0.0 && (value - value.round()).abs() < 1e-12
}

fn is_intercept_column(name: &str) -> bool {
    matches!(name, "1" | "(Intercept)" | "Intercept" | "intercept")
}

fn random_effect_term_label(reterm: &ReMat) -> String {
    let columns = reterm
        .cnames
        .iter()
        .map(|name| {
            if is_intercept_column(name) {
                "1"
            } else {
                name.as_str()
            }
        })
        .collect::<Vec<_>>()
        .join(" + ");
    format!("({columns} | {})", reterm.grouping_name)
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

fn annotate_glmm_covariance_status(
    certificate: &mut OptimizerCertificate,
    params: &[f64],
    n_beta: usize,
    lower_bounds: &[f64],
    certification: &JointLaplaceCertificationGradient,
    gradient_tolerance: f64,
) {
    if !certificate.evidence.optimizer_stop.acceptable_stop || params.len() <= n_beta {
        return;
    }
    let boundary_tolerance = 1.0e-8;
    let gradient = certification.gradient.as_slice();

    if let Some(free_gradient_norm) = certificate.free_gradient_norm {
        if !free_gradient_norm.is_finite() || free_gradient_norm > gradient_tolerance {
            // The assembled gradient failed the free-component KKT check. A
            // failure is only an *assessed* non-stationarity when some
            // failing free component carries a trusted reading; if every
            // failing component is one the noise-aware probe could not
            // assess, the honest verdict is "not assessable", not "not
            // optimized".
            let assessed_failure = certification_gradient_assessed_free_failure(
                certification,
                params,
                lower_bounds,
                gradient_tolerance,
            );
            if assessed_failure {
                certificate.status = crate::compiler::FitStatus::NotOptimized;
                if !certificate.diagnostics.iter().any(|diagnostic| {
                    diagnostic.code == DiagnosticCode::OptimizerNonconvergence
                        && diagnostic
                            .payload
                            .get("stationarity_check")
                            .and_then(serde_json::Value::as_str)
                            == Some("free_gradient_kkt")
                }) {
                    let mut diagnostic = Diagnostic::new(
                        DiagnosticCode::OptimizerNonconvergence,
                        DiagnosticSeverity::Warning,
                        DiagnosticStage::Certification,
                        "GLMM joint optimizer stop failed finite-difference stationarity; convergence is not certified",
                    )
                    .with_suggested_actions(vec![
                        "treat this joint GLMM result as not optimized until a tighter run or alternate optimizer certifies stationarity".to_string(),
                        "fall back to the labelled fast-PIRLS GLMM result when available rather than reporting a silent interior convergence".to_string(),
                    ]);
                    diagnostic
                        .payload
                        .insert("fit_mode".to_string(), serde_json::json!("joint_glmm"));
                    diagnostic.payload.insert(
                        "stationarity_check".to_string(),
                        serde_json::json!("free_gradient_kkt"),
                    );
                    diagnostic.payload.insert(
                        "free_gradient_norm".to_string(),
                        serde_json::json!(free_gradient_norm),
                    );
                    diagnostic.payload.insert(
                        "gradient_tolerance".to_string(),
                        serde_json::json!(gradient_tolerance),
                    );
                    insert_certification_gradient_payload(&mut diagnostic, certification);
                    if let Some(return_code) = &certificate.evidence.optimizer_stop.return_code {
                        diagnostic
                            .payload
                            .insert("return_code".to_string(), serde_json::json!(return_code));
                    }
                    certificate.diagnostics.push(diagnostic);
                }
            } else {
                certificate.status = crate::compiler::FitStatus::NotAssessed;
                let mut diagnostic = Diagnostic::new(
                    DiagnosticCode::OptimizerNotAssessed,
                    DiagnosticSeverity::Warning,
                    DiagnosticStage::Certification,
                    "GLMM joint stationarity could not be assessed: the finite-difference probe is noise-dominated on a flat deviance direction even at escalated steps",
                )
                .with_suggested_actions(vec![
                    "treat this fit as an acceptable optimizer stop whose stationarity is unverifiable, not as an assessed optimization failure".to_string(),
                    "certify externally (reference fit or refit with a tighter inner PIRLS tolerance) before promoting this row to strict parity".to_string(),
                ]);
                diagnostic
                    .payload
                    .insert("fit_mode".to_string(), serde_json::json!("joint_glmm"));
                diagnostic.payload.insert(
                    "stationarity_check".to_string(),
                    serde_json::json!("free_gradient_kkt_noise_dominated"),
                );
                diagnostic.payload.insert(
                    "free_gradient_norm".to_string(),
                    serde_json::json!(free_gradient_norm),
                );
                diagnostic.payload.insert(
                    "gradient_tolerance".to_string(),
                    serde_json::json!(gradient_tolerance),
                );
                insert_certification_gradient_payload(&mut diagnostic, certification);
                if let Some(return_code) = &certificate.evidence.optimizer_stop.return_code {
                    diagnostic
                        .payload
                        .insert("return_code".to_string(), serde_json::json!(return_code));
                }
                certificate.diagnostics.push(diagnostic);
            }
            return;
        }
    }
    let theta_params = &params[n_beta..];
    let theta_lower = lower_bounds.get(n_beta..).unwrap_or(&[]);
    let theta_gradient = gradient.get(n_beta..).unwrap_or(&[]);
    let boundary_indices = theta_params
        .iter()
        .zip(theta_lower.iter())
        .enumerate()
        .filter_map(|(index, (value, lower))| {
            lower
                .is_finite()
                .then_some(())
                .filter(|_| *value <= *lower + boundary_tolerance)
                .map(|_| index)
        })
        .collect::<Vec<_>>();

    if boundary_indices.is_empty() {
        certificate.status = crate::compiler::FitStatus::ConvergedInterior;
        record_certification_gradient_escalation(certificate, certification, gradient_tolerance);
        return;
    }

    // A boundary KKT violation must be *proven* on an assessed reading; an
    // unassessable component cannot demote a boundary stop.
    let invalid_boundary = boundary_indices.iter().any(|&index| {
        !certification
            .unassessable_indices
            .contains(&(index + n_beta))
            && theta_gradient
                .get(index)
                .is_some_and(|value| *value < -gradient_tolerance)
    });
    let classification = if invalid_boundary {
        certificate.status = crate::compiler::FitStatus::NotOptimized;
        CovarianceKktClassification::InvalidBoundaryStop
    } else {
        certificate.status = crate::compiler::FitStatus::ConvergedBoundary;
        CovarianceKktClassification::ValidZeroVariance
    };

    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::BoundaryParameter,
        DiagnosticSeverity::Info,
        DiagnosticStage::Certification,
        format!("GLMM joint covariance state classified as {classification:?}"),
    )
    .with_suggested_actions(vec![
        "interpret zero random-effect scales as boundary covariance estimates, not missing optimizer metadata".to_string(),
        "use the recorded stationarity residual and scorecard class before promoting a GLMM row to parity".to_string(),
    ]);
    diagnostic.payload.insert(
        "covariance_kkt_classification".to_string(),
        serde_json::json!(format!("{classification:?}")),
    );
    diagnostic.payload.insert(
        "boundary_theta_indices".to_string(),
        serde_json::json!(boundary_indices),
    );
    diagnostic.payload.insert(
        "gradient_tolerance".to_string(),
        serde_json::json!(gradient_tolerance),
    );
    insert_certification_gradient_payload(&mut diagnostic, certification);
    certificate.diagnostics.push(diagnostic);
    record_certification_gradient_escalation(certificate, certification, gradient_tolerance);
}

/// True when some free (non-boundary) component fails the stationarity
/// tolerance on an *assessed* reading — i.e. the failure is a proven
/// non-stationarity rather than a noise-dominated probe artifact.
fn certification_gradient_assessed_free_failure(
    certification: &JointLaplaceCertificationGradient,
    params: &[f64],
    lower_bounds: &[f64],
    gradient_tolerance: f64,
) -> bool {
    let boundary_tolerance = 1.0e-8;
    certification
        .gradient
        .iter()
        .enumerate()
        .any(|(index, value)| {
            let at_bound = lower_bounds.get(index).copied().is_some_and(|lower| {
                lower.is_finite()
                    && params.get(index).copied().unwrap_or(f64::NAN) <= lower + boundary_tolerance
            });
            !at_bound
                && (!value.is_finite() || value.abs() > gradient_tolerance)
                && !certification.unassessable_indices.contains(&index)
        })
}

/// Records the noise-aware probe context on a certification diagnostic so the
/// evidence trail shows which components were assessed at escalated
/// finite-difference steps (and which could not be assessed at all).
fn insert_certification_gradient_payload(
    diagnostic: &mut Diagnostic,
    certification: &JointLaplaceCertificationGradient,
) {
    if !certification.was_escalated() {
        return;
    }
    let max_abs = |values: &[f64]| {
        values
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
    };
    diagnostic.payload.insert(
        "probe_gradient_max_abs".to_string(),
        serde_json::json!(max_abs(&certification.probe_gradient)),
    );
    diagnostic.payload.insert(
        "assessed_gradient_max_abs".to_string(),
        serde_json::json!(max_abs(&certification.gradient)),
    );
    diagnostic.payload.insert(
        "escalated_indices".to_string(),
        serde_json::json!(certification.escalated_indices),
    );
    diagnostic.payload.insert(
        "unassessable_indices".to_string(),
        serde_json::json!(certification.unassessable_indices),
    );
    diagnostic.payload.insert(
        "escalated_relative_steps".to_string(),
        serde_json::json!(JOINT_LAPLACE_CERT_FD_ESCALATED_RELATIVE_STEPS),
    );
}

/// Leaves an Info-severity evidence trail on certificates whose stationarity
/// verdict relied on escalated finite-difference steps, so a passing status
/// never hides that the default-step probe was noise-dominated.
fn record_certification_gradient_escalation(
    certificate: &mut OptimizerCertificate,
    certification: &JointLaplaceCertificationGradient,
    gradient_tolerance: f64,
) {
    if !certification.was_escalated() {
        return;
    }
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::OptimizerRecovery,
        DiagnosticSeverity::Info,
        DiagnosticStage::Certification,
        "GLMM joint stationarity was assessed with escalated finite-difference steps; the default-step probe was dominated by inner-PIRLS deviance noise",
    )
    .with_suggested_actions(vec![
        "read the recorded probe and assessed gradient norms before applying strict external-engine tolerances".to_string(),
    ]);
    diagnostic
        .payload
        .insert("fit_mode".to_string(), serde_json::json!("joint_glmm"));
    diagnostic.payload.insert(
        "stationarity_check".to_string(),
        serde_json::json!("free_gradient_kkt_escalated_step"),
    );
    diagnostic.payload.insert(
        "gradient_tolerance".to_string(),
        serde_json::json!(gradient_tolerance),
    );
    insert_certification_gradient_payload(&mut diagnostic, certification);
    certificate.diagnostics.push(diagnostic);
}

fn annotate_glmm_singular_covariance_status(
    certificate: &mut OptimizerCertificate,
    theta: &[f64],
    is_singular: bool,
) {
    let near_zero_theta = theta
        .iter()
        .any(|value| value.is_finite() && value.abs() <= 1.0e-4);
    if !(is_singular || near_zero_theta) {
        return;
    }
    let boundary_roundoff = certificate
        .evidence
        .optimizer_stop
        .return_code
        .as_deref()
        .is_some_and(|code| code == "ROUNDOFF_LIMITED" || code == "FAILED:ROUNDOFF_LIMITED");
    if !certificate.evidence.optimizer_stop.acceptable_stop && !boundary_roundoff {
        return;
    }
    if boundary_roundoff {
        certificate.evidence.optimizer_stop.acceptable_stop = true;
        certificate.evidence.certification_quality = EvidenceQuality::Approximate {
            reason: "roundoff-limited optimizer stop accepted only for a near-zero GLMM covariance boundary"
                .to_string(),
        };
        certificate
            .checks
            .retain(|check| !matches!(check, crate::compiler::CertificateCheck::Failed { .. }));
        certificate
            .diagnostics
            .retain(|diagnostic| diagnostic.code != DiagnosticCode::OptimizerNonconvergence);
    }
    if matches!(
        certificate.status,
        crate::compiler::FitStatus::ConvergedInterior | crate::compiler::FitStatus::NotOptimized
    ) {
        certificate.status = crate::compiler::FitStatus::ConvergedBoundary;
    }
    let classification = CovarianceKktClassification::ValidZeroVariance;
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::BoundaryParameter,
        DiagnosticSeverity::Info,
        DiagnosticStage::Certification,
        format!("GLMM covariance state classified as {classification:?}"),
    )
    .with_suggested_actions(vec![
        "treat the singular random-effect covariance as an explicit boundary state in downstream summaries".to_string(),
        "do not promote the GLMM row to external-engine parity without the joint optimizer scorecard gate".to_string(),
    ]);
    diagnostic.payload.insert(
        "covariance_kkt_classification".to_string(),
        serde_json::json!(format!("{classification:?}")),
    );
    diagnostic
        .payload
        .insert("is_singular".to_string(), serde_json::json!(true));
    certificate.diagnostics.push(diagnostic);
}

fn uncertified_joint_fallback(
    joint_certificate: &OptimizerCertificate,
    joint_optsum: &OptSummary,
    fallback_fast_pirls: Option<GeneralizedLinearMixedModel>,
) -> Option<GeneralizedLinearMixedModel> {
    if !joint_certificate_requires_fallback(joint_certificate) {
        return None;
    }
    let mut fallback = fallback_fast_pirls?;
    let joint_return_code = joint_optsum.return_value.clone();
    let fast_return_code = fallback.lmm.optsum.return_value.clone();
    let fallback_prefix = if joint_return_code.starts_with("JOINT_AGQ") {
        "JOINT_AGQ_FALLBACK_FAST_PIRLS"
    } else {
        "JOINT_LAPLACE_FALLBACK_FAST_PIRLS"
    };
    fallback.lmm.optsum.return_value =
        format!("{fallback_prefix}(joint={joint_return_code}; fast={fast_return_code})");
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::OptimizerRecovery,
        DiagnosticSeverity::Warning,
        DiagnosticStage::Certification,
        "joint GLMM did not certify; returning labelled fast-PIRLS fallback",
    )
    .with_suggested_actions(vec![
        "treat this as a documented-divergence fast-PIRLS GLMM result, not a certified joint fit"
            .to_string(),
        "inspect the joint optimizer return code before promoting this row to parity".to_string(),
    ]);
    diagnostic.payload.insert(
        "fit_mode".to_string(),
        serde_json::json!("fallback_fast_pirls"),
    );
    diagnostic.payload.insert(
        "scorecard_class".to_string(),
        serde_json::json!("documented_divergence"),
    );
    diagnostic.payload.insert(
        "joint_return_code".to_string(),
        serde_json::json!(joint_return_code),
    );
    diagnostic.payload.insert(
        "joint_fit_status".to_string(),
        serde_json::json!(format!("{:?}", joint_certificate.status)),
    );
    if let Some(free_gradient_norm) = joint_certificate.free_gradient_norm {
        diagnostic.payload.insert(
            "joint_free_gradient_norm".to_string(),
            serde_json::json!(free_gradient_norm),
        );
    }
    diagnostic.payload.insert(
        "fast_pirls_return_code".to_string(),
        serde_json::json!(fast_return_code),
    );
    diagnostic.payload.insert(
        "joint_optimizer".to_string(),
        serde_json::json!(joint_optsum.optimizer_name()),
    );
    diagnostic.payload.insert(
        "joint_optimizer_backend".to_string(),
        serde_json::json!(joint_optsum.backend_name()),
    );
    diagnostic.payload.insert(
        "joint_feval".to_string(),
        serde_json::json!(joint_optsum.feval),
    );
    diagnostic.payload.insert(
        "joint_max_feval".to_string(),
        serde_json::json!(joint_optsum.max_feval),
    );
    diagnostic.payload.insert(
        "joint_fmin".to_string(),
        serde_json::json!(joint_optsum.fmin),
    );
    fallback
        .lmm
        .compiler_artifact
        .diagnostics
        .push(diagnostic.clone());
    if let Some(certificate) = &mut fallback.lmm.compiler_artifact.optimizer_certificate {
        certificate.diagnostics.push(diagnostic);
    }
    fallback.record_glmm_fit_metadata();
    Some(fallback)
}

fn joint_certificate_requires_fallback(joint_certificate: &OptimizerCertificate) -> bool {
    !joint_certificate.evidence.optimizer_stop.acceptable_stop
        || matches!(
            joint_certificate.status,
            crate::compiler::FitStatus::NotOptimized
        )
}

fn joint_candidate_materially_improves_profiled_start(optsum: &OptSummary) -> bool {
    let (Some(initial), Some(final_value)) = (
        optsum.finitial.is_finite().then_some(optsum.finitial),
        optsum.fmin.is_finite().then_some(optsum.fmin),
    ) else {
        return false;
    };
    if final_value >= initial {
        return false;
    }
    let scale = initial.abs().max(final_value.abs()).max(1.0);
    let tolerance =
        (optsum.ftol_abs.max(1.0e-8) + optsum.ftol_rel.max(1.0e-10) * scale).max(1.0e-8) * 10.0;
    initial - final_value > tolerance
}

fn record_uncertified_joint_candidate_diagnostic(
    certificate: &mut OptimizerCertificate,
    optsum: &OptSummary,
) {
    let objective_delta = optsum.finitial - optsum.fmin;
    let budget_limited = certificate.evidence.optimizer_stop.budget_exhausted
        || optsum.return_value.contains("MAXEVAL_REACHED")
        || optsum.return_value.contains("MAXTIME_REACHED");
    let stationarity_uncertified = certificate.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == DiagnosticCode::OptimizerNonconvergence
            && diagnostic
                .payload
                .get("stationarity_check")
                .and_then(serde_json::Value::as_str)
                == Some("free_gradient_kkt")
    });
    let (message, scorecard_class, certification_gap, first_action) = if budget_limited {
        (
            "returning improved joint GLMM candidate after budget exhaustion; convergence is not certified",
            "budget_limited_joint_candidate",
            "budget_exhausted",
            "treat fixed effects and log-likelihood as a budget-limited joint-Laplace candidate, not a certified optimizer convergence",
        )
    } else if stationarity_uncertified {
        (
            "returning improved joint GLMM candidate with uncertified stationarity; convergence is not certified",
            "stationarity_uncertified_joint_candidate",
            "stationarity_uncertified",
            "treat fixed effects and log-likelihood as an uncertified joint-Laplace candidate, not a certified optimizer convergence",
        )
    } else {
        (
            "returning improved joint GLMM candidate without full optimizer certification",
            "uncertified_joint_candidate",
            "uncertified",
            "treat fixed effects and log-likelihood as an uncertified joint-Laplace candidate until an external reference verifies it",
        )
    };
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::OptimizerNonconvergence,
        DiagnosticSeverity::Warning,
        DiagnosticStage::Certification,
        message,
    )
    .with_suggested_actions(vec![
        first_action.to_string(),
        "increase max_feval or compare against an external joint-Laplace reference before promoting this row to strict parity".to_string(),
    ]);
    diagnostic.payload.insert(
        "fit_mode".to_string(),
        serde_json::json!("uncertified_joint_candidate"),
    );
    diagnostic.payload.insert(
        "scorecard_class".to_string(),
        serde_json::json!(scorecard_class),
    );
    diagnostic.payload.insert(
        "certification_gap".to_string(),
        serde_json::json!(certification_gap),
    );
    diagnostic.payload.insert(
        "objective_delta".to_string(),
        serde_json::json!(objective_delta),
    );
    diagnostic
        .payload
        .insert("joint_fmin".to_string(), serde_json::json!(optsum.fmin));
    diagnostic
        .payload
        .insert("joint_feval".to_string(), serde_json::json!(optsum.feval));
    diagnostic.payload.insert(
        "joint_max_feval".to_string(),
        serde_json::json!(optsum.max_feval),
    );
    certificate.diagnostics.push(diagnostic);
}

fn binary_column_split(values: impl Iterator<Item = f64>) -> Option<BinaryColumnSplit> {
    let mut unique = Vec::new();
    for value in values {
        if !value.is_finite() {
            return None;
        }
        if unique
            .iter()
            .all(|seen: &f64| (value - *seen).abs() > 1e-12)
        {
            unique.push(value);
            if unique.len() > 2 {
                return None;
            }
        }
    }
    if unique.len() != 2 {
        return None;
    }
    unique.sort_by(|a, b| a.total_cmp(b));
    Some(BinaryColumnSplit {
        low: unique[0],
        high: unique[1],
    })
}

fn outcome_counts_for_value(
    values: impl Iterator<Item = f64>,
    y: impl Iterator<Item = f64>,
    target: f64,
) -> OutcomeCounts {
    let mut counts = OutcomeCounts {
        n: 0,
        successes: 0,
        failures: 0,
    };
    for (value, response) in values.zip(y) {
        if (value - target).abs() <= 1e-12 {
            counts.n += 1;
            if response > 0.5 {
                counts.successes += 1;
            } else {
                counts.failures += 1;
            }
        }
    }
    counts
}

fn separation_diagnostic_for_side(
    column_name: &str,
    value: f64,
    side: OutcomeCounts,
    complement: OutcomeCounts,
) -> Option<Diagnostic> {
    if side.n == 0
        || complement.n == 0
        || !side_is_pure(side)
        || !complement_has_opposite(side, complement)
    {
        return None;
    }

    let outcome = if side.successes == side.n { 1 } else { 0 };
    let kind = if side_is_pure(complement) {
        "complete_fixed_effect"
    } else {
        "quasi_complete_fixed_effect"
    };
    let rows = if side.n == 1 { "row" } else { "rows" };
    let value_label = format_column_value(value);
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::BinomialSeparation,
        DiagnosticSeverity::Warning,
        DiagnosticStage::Certification,
        format!(
            "Possible separation in binomial model: `{column_name} = {value_label}` occurs in {} {rows}, and all such rows have y = {outcome}. The coefficient for `{column_name}` may be unbounded; standard errors, Wald tests, and p-values for this term are unreliable.",
            side.n
        ),
    )
    .with_affected_terms(vec![column_name.to_string()])
    .with_suggested_actions(vec![
        "inspect the corresponding rows or levels for sparse outcome support".to_string(),
        "consider removing or combining rare predictors, or use penalized/Bayesian logistic mixed modeling".to_string(),
        "report inference for this term as unreliable if the model is retained".to_string(),
    ]);
    diagnostic
        .payload
        .insert("term".to_string(), serde_json::json!(column_name));
    diagnostic
        .payload
        .insert("value".to_string(), serde_json::json!(value));
    diagnostic
        .payload
        .insert("n_at_value".to_string(), serde_json::json!(side.n));
    diagnostic.payload.insert(
        "successes_at_value".to_string(),
        serde_json::json!(side.successes),
    );
    diagnostic.payload.insert(
        "failures_at_value".to_string(),
        serde_json::json!(side.failures),
    );
    diagnostic.payload.insert(
        "complement_successes".to_string(),
        serde_json::json!(complement.successes),
    );
    diagnostic.payload.insert(
        "complement_failures".to_string(),
        serde_json::json!(complement.failures),
    );
    diagnostic
        .payload
        .insert("separation_kind".to_string(), serde_json::json!(kind));
    Some(diagnostic)
}

fn side_is_pure(counts: OutcomeCounts) -> bool {
    counts.n > 0 && (counts.successes == counts.n || counts.failures == counts.n)
}

fn complement_has_opposite(side: OutcomeCounts, complement: OutcomeCounts) -> bool {
    if side.successes == side.n {
        complement.failures > 0
    } else {
        complement.successes > 0
    }
}

fn format_column_value(value: f64) -> String {
    if (value.round() - value).abs() < 1e-12 {
        format!("{value:.0}")
    } else {
        format!("{value:.6}")
    }
}

#[cfg(test)]
fn pirls_working_observation(
    family: Family,
    link: LinkFunction,
    y: f64,
    eta: f64,
    mu: f64,
    case_weight: f64,
) -> (f64, f64) {
    pirls_working_observation_with_family_parameters(family, link, None, y, eta, mu, case_weight)
}

fn pirls_working_observation_with_family_parameters(
    family: Family,
    link: LinkFunction,
    negative_binomial_theta: Option<f64>,
    y: f64,
    eta: f64,
    mu: f64,
    case_weight: f64,
) -> (f64, f64) {
    let (working_mu, eta_for_derivative) = bounded_pirls_mean_and_eta(family, link, mu, eta);
    let dmu_deta = link.mu_eta(eta_for_derivative);
    let var_mu = glmm_variance(family, working_mu, negative_binomial_theta);
    let weight = if dmu_deta.is_finite() && var_mu.is_finite() && var_mu > 0.0 {
        case_weight * dmu_deta * dmu_deta / var_mu
    } else {
        0.0
    };
    let resid = if !dmu_deta.is_finite() || dmu_deta.abs() < 1e-15 {
        0.0
    } else {
        (y - working_mu) / dmu_deta
    };
    (weight.max(0.0).sqrt(), eta + resid)
}

#[cfg(test)]
fn pirls_working_observation_with_offset(
    family: Family,
    link: LinkFunction,
    y: f64,
    eta: f64,
    mu: f64,
    case_weight: f64,
    offset: f64,
) -> (f64, f64) {
    let (sqrt_weight, working_response) =
        pirls_working_observation(family, link, y, eta, mu, case_weight);
    (sqrt_weight, working_response - offset)
}

fn pirls_working_observation_with_offset_and_family_parameters(
    family: Family,
    link: LinkFunction,
    negative_binomial_theta: Option<f64>,
    y: f64,
    eta: f64,
    mu: f64,
    case_weight: f64,
    offset: f64,
) -> (f64, f64) {
    let (sqrt_weight, working_response) = pirls_working_observation_with_family_parameters(
        family,
        link,
        negative_binomial_theta,
        y,
        eta,
        mu,
        case_weight,
    );
    (sqrt_weight, working_response - offset)
}

fn bounded_pirls_mean_and_eta(family: Family, link: LinkFunction, mu: f64, eta: f64) -> (f64, f64) {
    const BOUNDED_MEAN_EPS: f64 = 1e-15;
    const LOG_LINK_ETA_BOUND: f64 = 30.0;
    if matches!(family, Family::Bernoulli | Family::Binomial) {
        let bounded_mu = mu.clamp(BOUNDED_MEAN_EPS, 1.0 - BOUNDED_MEAN_EPS);
        (bounded_mu, link.link(bounded_mu))
    } else if matches!(family, Family::Poisson | Family::NegativeBinomial) {
        match link {
            LinkFunction::Log => {
                let bounded_eta = eta.clamp(-LOG_LINK_ETA_BOUND, LOG_LINK_ETA_BOUND);
                (bounded_eta.exp(), bounded_eta)
            }
            LinkFunction::Sqrt => {
                let bounded_mu = mu.max(BOUNDED_MEAN_EPS);
                let min_eta = bounded_mu.sqrt();
                let bounded_eta = if eta.abs() < min_eta {
                    if eta.is_sign_negative() {
                        -min_eta
                    } else {
                        min_eta
                    }
                } else {
                    eta
                };
                (bounded_eta * bounded_eta, bounded_eta)
            }
            _ => (mu, eta),
        }
    } else {
        (mu, eta)
    }
}

fn glmm_variance(family: Family, mu: f64, negative_binomial_theta: Option<f64>) -> f64 {
    match family {
        Family::NegativeBinomial => {
            let theta = negative_binomial_theta
                .unwrap_or(1.0)
                .max(f64::MIN_POSITIVE);
            mu + mu * mu / theta
        }
        _ => family.variance(mu),
    }
}

fn clean_glmm_prediction_variance_component(value: f64) -> Option<f64> {
    if !value.is_finite() || value < -1.0e-10 {
        return None;
    }
    Some(value.max(0.0))
}

/// Plug-in predictive summary for one future observation on the response
/// scale: law-of-total-variance moment plus predictive-distribution quantile
/// bounds.
struct GlmmFutureObservation {
    variance: f64,
    lower: f64,
    upper: f64,
}

/// Gauss-Hermite node count for predictive (future-observation) mixtures.
const GLMM_PREDICTIVE_QUADRATURE_POINTS: usize = 21;
/// Floor applied to conditional means before constructing count-family
/// predictive components, since the statrs distributions require a strictly
/// positive rate/mean.
const GLMM_PREDICTIVE_MEAN_FLOOR: f64 = 1.0e-12;

/// Smallest `t` (as f64) with `cdf(t) >= p` for a discrete mixture supported
/// on the non-negative integers. Doubles an upper bracket from `mean_hint`,
/// then binary-searches. `None` if the bracket never reaches `p`.
fn discrete_mixture_quantile(cdf: &dyn Fn(u64) -> f64, p: f64, mean_hint: f64) -> Option<f64> {
    let mut hi: u64 = if mean_hint.is_finite() && mean_hint > 1.0 {
        mean_hint.ceil() as u64
    } else {
        1
    };
    let mut expansions = 0;
    while cdf(hi) < p {
        if expansions >= 96 {
            return None;
        }
        hi = hi.saturating_mul(2).saturating_add(1);
        expansions += 1;
    }
    let mut lo: u64 = 0;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Some(lo as f64)
}

/// Quantile of a continuous mixture CDF by bracket expansion and bisection.
/// `domain_floor` clamps the lower bracket for positive-support families.
fn continuous_mixture_quantile(
    cdf: &dyn Fn(f64) -> f64,
    p: f64,
    domain_floor: Option<f64>,
    center: f64,
    spread: f64,
) -> Option<f64> {
    if !center.is_finite() || !spread.is_finite() {
        return None;
    }
    let step = spread.max(center.abs() * 1.0e-6).max(1.0e-12);
    let mut lo = center - 10.0 * step;
    let mut hi = center + 10.0 * step;
    if let Some(floor) = domain_floor {
        lo = lo.max(floor);
        hi = hi.max(floor + step);
    }
    let mut expansions = 0;
    while cdf(hi) < p {
        if expansions >= 256 || !hi.is_finite() {
            return None;
        }
        hi += (hi - lo).max(step);
        expansions += 1;
    }
    expansions = 0;
    while cdf(lo) > p {
        if expansions >= 256 || !lo.is_finite() {
            return None;
        }
        lo -= (hi - lo).max(step);
        if let Some(floor) = domain_floor {
            if lo <= floor {
                lo = floor;
                break;
            }
        }
        expansions += 1;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if !(mid > lo && mid < hi) {
            break;
        }
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(hi)
}

/// `ln Phi(x)` with an asymptotic tail for arguments where the direct CDF
/// underflows (needed by the inverse-Gaussian CDF's exponentially weighted
/// second term).
fn standard_normal_ln_cdf(x: f64) -> f64 {
    if x < -37.0 {
        // Mills-ratio asymptotic: Phi(x) ~ phi(x) / |x| for x -> -inf.
        -0.5 * x * x - (-x).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    } else {
        Normal::new(0.0, 1.0)
            .unwrap()
            .cdf(x)
            .max(f64::MIN_POSITIVE)
            .ln()
    }
}

/// CDF of the inverse-Gaussian distribution with mean `mu` and shape
/// `lambda`, evaluated in log space so the `exp(2 lambda / mu)` factor cannot
/// overflow against the matching normal tail.
fn inverse_gaussian_cdf(t: f64, mu: f64, lambda: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let sqrt_term = (lambda / t).sqrt();
    let first = standard_normal.cdf(sqrt_term * (t / mu - 1.0));
    let second = (2.0 * lambda / mu + standard_normal_ln_cdf(-sqrt_term * (t / mu + 1.0))).exp();
    (first + second).clamp(0.0, 1.0)
}

const NEGATIVE_BINOMIAL_THETA_MIN: f64 = 1.0e-8;
const NEGATIVE_BINOMIAL_THETA_MAX: f64 = 1.0e8;
const NEGATIVE_BINOMIAL_THETA_MAX_ITERS: usize = 8;
const NEGATIVE_BINOMIAL_THETA_TOL: f64 = 1.0e-5;
const NEGATIVE_BINOMIAL_THETA_FINAL_REFIT_TOL: f64 = 1.0e-8;

fn clamp_negative_binomial_theta(theta: f64) -> f64 {
    theta.clamp(NEGATIVE_BINOMIAL_THETA_MIN, NEGATIVE_BINOMIAL_THETA_MAX)
}

fn negative_binomial_deviance_component(y: f64, mu: f64, theta: f64) -> f64 {
    let mu = mu.max(f64::MIN_POSITIVE);
    let theta = theta.max(f64::MIN_POSITIVE);
    let first = if y == 0.0 { 0.0 } else { y * (y / mu).ln() };
    let second = (y + theta) * ((y + theta) / (mu + theta)).ln();
    2.0 * (first - second)
}

fn negative_binomial_loglik_observation(y: f64, mu: f64, theta: f64) -> f64 {
    let mu = mu.max(f64::MIN_POSITIVE);
    let theta = theta.max(f64::MIN_POSITIVE);
    ln_gamma(y + theta) - ln_gamma(theta) - ln_gamma(y + 1.0)
        + theta * (theta / (theta + mu)).ln()
        + if y == 0.0 {
            0.0
        } else {
            y * (mu / (theta + mu)).ln()
        }
}

fn negative_binomial_theta_moment_start(y: &[f64], weights: Option<&[f64]>) -> f64 {
    let (sum_w, mean_num) =
        y.iter()
            .enumerate()
            .fold((0.0, 0.0), |(sum_w, mean_num), (idx, &value)| {
                let weight = weights
                    .and_then(|weights| weights.get(idx).copied())
                    .unwrap_or(1.0);
                (sum_w + weight, mean_num + weight * value)
            });
    if sum_w <= 0.0 {
        return 1.0;
    }
    let mean = mean_num / sum_w;
    let variance = y.iter().enumerate().fold(0.0, |acc, (idx, &value)| {
        let weight = weights
            .and_then(|weights| weights.get(idx).copied())
            .unwrap_or(1.0);
        acc + weight * (value - mean).powi(2)
    }) / sum_w.max(1.0);

    if variance > mean && mean > 0.0 {
        clamp_negative_binomial_theta(mean * mean / (variance - mean))
    } else {
        NEGATIVE_BINOMIAL_THETA_MAX.sqrt()
    }
}

fn estimate_negative_binomial_theta_conditional(
    y: &[f64],
    mu: &[f64],
    weights: Option<&[f64]>,
) -> f64 {
    if y.len() != mu.len() || y.is_empty() {
        return 1.0;
    }
    let log_min = NEGATIVE_BINOMIAL_THETA_MIN.ln();
    let log_max = NEGATIVE_BINOMIAL_THETA_MAX.ln();
    let weighted_loglik = |log_theta: f64| -> f64 {
        let theta = log_theta.exp();
        let mut total = 0.0;
        for (idx, (&y_i, &mu_i)) in y.iter().zip(mu.iter()).enumerate() {
            let weight = weights
                .and_then(|weights| weights.get(idx).copied())
                .unwrap_or(1.0);
            if !weight.is_finite() || weight <= 0.0 {
                continue;
            }
            let contribution = negative_binomial_loglik_observation(y_i, mu_i, theta);
            if !contribution.is_finite() {
                return f64::NEG_INFINITY;
            }
            total += weight * contribution;
        }
        total
    };

    let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut a = log_min;
    let mut b = log_max;
    let mut c = b - inv_phi * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = weighted_loglik(c);
    let mut fd = weighted_loglik(d);

    for _ in 0..96 {
        if (b - a).abs() <= 1.0e-8 {
            break;
        }
        if fc < fd {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = weighted_loglik(d);
        } else {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = weighted_loglik(c);
        }
    }

    let mut candidates = vec![a, b, c, d];
    let moment = negative_binomial_theta_moment_start(y, weights);
    candidates.push(moment.ln());
    candidates
        .into_iter()
        .filter(|value| value.is_finite())
        .max_by(|left, right| weighted_loglik(*left).total_cmp(&weighted_loglik(*right)))
        .map(|log_theta| clamp_negative_binomial_theta(log_theta.exp()))
        .unwrap_or(moment)
}

fn relative_theta_change(old: f64, new: f64) -> f64 {
    if !old.is_finite() || !new.is_finite() {
        return f64::INFINITY;
    }
    (new - old).abs() / old.abs().max(1.0)
}

fn pirls_converged(obj: f64, accepted_obj: f64, tol: f64) -> bool {
    (obj - accepted_obj).abs() < tol
}

fn validate_case_weights(weights: &[f64], n_obs: usize) -> Result<()> {
    if weights.len() != n_obs {
        return Err(MixedModelError::InvalidArgument(format!(
            "case weights length ({}) does not match number of observations ({n_obs})",
            weights.len()
        )));
    }
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w <= 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "case weight at index {i} must be finite and positive (got {w})"
            )));
        }
    }
    Ok(())
}

fn validate_offset(offset: &[f64], n_obs: usize) -> Result<()> {
    if offset.len() != n_obs {
        return Err(MixedModelError::InvalidArgument(format!(
            "offset length ({}) does not match number of observations ({n_obs})",
            offset.len()
        )));
    }
    for (idx, &value) in offset.iter().enumerate() {
        if !value.is_finite() {
            return Err(MixedModelError::InvalidArgument(format!(
                "offset at index {idx} must be finite (got {value})"
            )));
        }
    }
    Ok(())
}

fn validate_supported_glmm_family_link(family: Family, link: LinkFunction) -> Result<()> {
    let supported = match family {
        Family::Bernoulli | Family::Binomial => {
            matches!(
                link,
                LinkFunction::Logit | LinkFunction::Probit | LinkFunction::Cloglog
            )
        }
        Family::Poisson => matches!(link, LinkFunction::Log | LinkFunction::Sqrt),
        Family::NegativeBinomial => matches!(link, LinkFunction::Log),
        // Dispersion-family GLMMs predate this explicit binary/Poisson support
        // matrix; keep their existing sensible links while preserving the
        // Normal+Identity LMM redirect above.
        Family::Gamma | Family::InverseGaussian => {
            matches!(link, LinkFunction::Log | LinkFunction::Inverse)
        }
        Family::Normal => matches!(
            link,
            LinkFunction::Log | LinkFunction::Inverse | LinkFunction::Sqrt
        ),
    };
    if supported {
        Ok(())
    } else {
        Err(MixedModelError::UnsupportedFamilyLink {
            family: family_label(family).to_string(),
            link: link_label(link).to_string(),
        })
    }
}

fn validate_negative_binomial_theta_request(
    family: Family,
    theta: Option<f64>,
    estimate_theta: bool,
) -> Result<()> {
    match (family, theta, estimate_theta) {
        (Family::NegativeBinomial, Some(theta), _) if theta.is_finite() && theta > 0.0 => Ok(()),
        (Family::NegativeBinomial, Some(theta), true) => Err(MixedModelError::InvalidArgument(
            format!("negative-binomial theta start must be positive and finite (got {theta})"),
        )),
        (Family::NegativeBinomial, Some(theta), false) => Err(MixedModelError::InvalidArgument(
            format!("negative-binomial fixed theta must be positive and finite (got {theta})"),
        )),
        (Family::NegativeBinomial, None, true) => Ok(()),
        (Family::NegativeBinomial, None, false) => Err(MixedModelError::InvalidArgument(
            "negative-binomial GLMM requires a positive fixed theta, or explicit theta \
             estimation via GeneralizedLinearMixedModel::new_negative_binomial_estimated(...) \
             / GeneralizedLinearMixedModelBuilder::estimate_negative_binomial_theta(...); use \
             GeneralizedLinearMixedModel::new_negative_binomial(...) or \
             GeneralizedLinearMixedModelBuilder::negative_binomial_theta(...) for fixed theta"
                .to_string(),
        )),
        (_, Some(_), _) | (_, None, true) => Err(MixedModelError::InvalidArgument(
            "negative-binomial theta options can only be supplied with Family::NegativeBinomial"
                .to_string(),
        )),
        (_, None, false) => Ok(()),
    }
}

fn initialize_negative_binomial_theta(
    family: Family,
    theta: Option<f64>,
    estimate_theta: bool,
    response: Option<&[f64]>,
) -> Result<Option<f64>> {
    if family != Family::NegativeBinomial {
        return Ok(None);
    }
    if let Some(theta) = theta {
        return Ok(Some(clamp_negative_binomial_theta(theta)));
    }
    if estimate_theta {
        let y = response.ok_or_else(|| {
            MixedModelError::InvalidArgument(
                "negative-binomial theta estimation requires a numeric response".to_string(),
            )
        })?;
        return Ok(Some(negative_binomial_theta_moment_start(y, None)));
    }
    Err(MixedModelError::InvalidArgument(
        "negative-binomial GLMM requires a positive fixed theta".to_string(),
    ))
}

fn validate_glmm_response_domain(family: Family, link: LinkFunction, y: &[f64]) -> Result<()> {
    for (idx, &value) in y.iter().enumerate() {
        if !value.is_finite() {
            return Err(MixedModelError::InvalidArgument(format!(
                "response at index {idx} must be finite for GLMM construction (got {value})"
            )));
        }
        if family == Family::Bernoulli && !is_binary_response(value) {
            return Err(MixedModelError::InvalidArgument(format!(
                "bernoulli GLMM response must be exactly 0 or 1; index {idx} has {value}"
            )));
        }
        if family == Family::Binomial
            && !(0.0..=1.0).contains(&value)
            && !is_nonnegative_integer_response(value)
        {
            return Err(MixedModelError::InvalidArgument(format!(
                "binomial GLMM response must be a proportion in [0, 1] or a non-negative integer count; index {idx} has {value}"
            )));
        }
        if family == Family::Poisson && value < 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "poisson GLMM response must be non-negative; index {idx} has {value}"
            )));
        }
        if family == Family::NegativeBinomial && !is_nonnegative_integer_response(value) {
            return Err(MixedModelError::InvalidArgument(format!(
                "negative-binomial GLMM response must be a non-negative integer count; index {idx} has {value}"
            )));
        }
        if matches!(family, Family::Gamma | Family::InverseGaussian) && value <= 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "{} GLMM response must be strictly positive; index {idx} has {value}",
                family_label(family)
            )));
        }
        if family == Family::Normal && link == LinkFunction::Sqrt && value < 0.0 {
            return Err(MixedModelError::InvalidArgument(format!(
                "gaussian GLMM with sqrt link requires non-negative responses; index {idx} has {value}"
            )));
        }
    }
    Ok(())
}

fn initial_response_mean(family: Family, y: &DVector<f64>, weights: &[f64]) -> Option<f64> {
    if y.is_empty() {
        return None;
    }
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for (idx, value) in y.iter().enumerate() {
        let weight = weights.get(idx).copied().unwrap_or(1.0);
        weighted_sum += weight * value;
        weight_sum += weight;
    }
    if weight_sum <= 0.0 {
        return None;
    }
    let mean = weighted_sum / weight_sum;
    Some(match family {
        Family::Bernoulli | Family::Binomial => mean.clamp(1e-6, 1.0 - 1e-6),
        Family::Poisson | Family::NegativeBinomial | Family::Gamma | Family::InverseGaussian => {
            mean.max(1e-6)
        }
        Family::Normal => mean.max(0.0),
    })
}

fn initial_mean_for_link(family: Family, mean: f64) -> f64 {
    match family {
        Family::Bernoulli | Family::Binomial => mean.clamp(1e-6, 1.0 - 1e-6),
        Family::Poisson | Family::NegativeBinomial | Family::Gamma | Family::InverseGaussian => {
            mean.max(1e-6)
        }
        Family::Normal => mean.max(0.0),
    }
}

fn family_label(family: Family) -> &'static str {
    match family {
        Family::Normal => "gaussian",
        Family::Bernoulli => "bernoulli",
        Family::Binomial => "binomial",
        Family::Poisson => "poisson",
        Family::NegativeBinomial => "negative_binomial",
        Family::Gamma => "gamma",
        Family::InverseGaussian => "inverse_gaussian",
    }
}

fn link_label(link: LinkFunction) -> &'static str {
    match link {
        LinkFunction::Identity => "identity",
        LinkFunction::Log => "log",
        LinkFunction::Logit => "logit",
        LinkFunction::Probit => "probit",
        LinkFunction::Cloglog => "cloglog",
        LinkFunction::Inverse => "inverse",
        LinkFunction::Sqrt => "sqrt",
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

fn glmm_profile_likelihood_unsupported_reason(operation: &str) -> String {
    format!(
        "{operation}: GLMM profile likelihood is not implemented in this release; \
         profile_sigma/profile_theta remain LMM-only, so GLMM callers must use \
         certified Wald intervals when available or an explicit bootstrap/profile \
         implementation rather than fabricated profile-likelihood intervals"
    )
}

struct GlmmJointHessianCertification {
    inverse: DMatrix<f64>,
    min_eigenvalue: f64,
    condition_number: f64,
    rank: usize,
}

struct GlmmFixedEffectInferenceArtifacts {
    table: FixedEffectInferenceTable,
    covariance: Option<FixedEffectCovarianceMatrix>,
}

const GLMM_PIRLS_MAX_ITER: usize = 10;
const GLMM_HESSIAN_PIRLS_MAX_ITER: usize = 50;

/// Default relative finite-difference step for joint-Laplace gradients.
const JOINT_LAPLACE_FD_RELATIVE_STEP: f64 = 1.0e-5;

/// Escalated relative steps for the stationarity certification gradient.
///
/// The inner PIRLS stopping rule leaves an O(1e-5) absolute error in the
/// deviance, so a central difference at relative step `h` carries a noise
/// term of roughly `1e-5 / h` in the gradient. Against the 2e-2 stationarity
/// tolerance the default 1e-5 step is useless on flat directions (noise
/// O(1)); these two steps put the noise term at roughly tolerance/2 and
/// tolerance/8 while keeping the central-difference truncation error far
/// below tolerance, and disagreement between them flags a component whose
/// surface is too rough to assess at any trusted step.
const JOINT_LAPLACE_CERT_FD_ESCALATED_RELATIVE_STEPS: [f64; 2] = [1.0e-3, 4.0e-3];

/// Stationarity tolerance for the post-fit profiled fast-PIRLS optimum
/// certificate; matches the joint-Laplace fit-time certification tolerance.
const PIRLS_PROFILED_CERTIFICATE_GRADIENT_TOLERANCE: f64 = 2.0e-2;
/// Theta-dimension budget for the post-fit profiled-optimum certificate. The
/// curvature probe costs ~2k^2 PIRLS solves; beyond this the certificate is
/// skipped with an explicit reason rather than silently slowing every fit.
const PIRLS_PROFILED_CERTIFICATE_MAX_THETA: usize = 12;

/// Result of the noise-aware stationarity gradient probe used by the
/// joint-Laplace optimizer certificate.
struct JointLaplaceCertificationGradient {
    /// Assessed gradient: default-step readings where those already pass the
    /// tolerance, escalated-step readings where the default step was
    /// noise-dominated but the larger steps agreed, and the raw default-step
    /// readings for unassessable components.
    gradient: Vec<f64>,
    /// Raw default-step readings, kept for the certificate evidence trail.
    probe_gradient: Vec<f64>,
    /// Components certified (or honestly failed) via escalated steps.
    escalated_indices: Vec<usize>,
    /// Components whose escalated-step readings disagreed: the probe cannot
    /// distinguish noise from signal, so stationarity is not assessable there.
    unassessable_indices: Vec<usize>,
}

impl JointLaplaceCertificationGradient {
    fn was_escalated(&self) -> bool {
        !self.escalated_indices.is_empty() || !self.unassessable_indices.is_empty()
    }
}

/// Evidence that a profiled fast-PIRLS fit sits at a certified optimum of its
/// own objective: assessed stationarity over theta plus positive-definite,
/// well-conditioned curvature over the interior theta coordinates. Beta is
/// exactly minimized by the penalized least-squares step at every probed
/// theta, so no separate beta-direction evidence is required.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PirlsProfiledOptimumCertificate {
    /// Largest assessed absolute gradient component over theta.
    gradient_max_abs: f64,
    /// Smallest eigenvalue of the interior-theta profiled Hessian.
    min_eigenvalue: f64,
    /// Condition number of the interior-theta profiled Hessian.
    condition_number: f64,
    /// One-based theta indices whose gradient needed escalated FD steps.
    escalated_theta_indices: Vec<usize>,
    /// One-based theta indices held at their lower bounds (one-sided
    /// stationarity check; omitted from the curvature probe).
    boundary_theta_indices: Vec<usize>,
}

fn glmm_hessian_step(value: f64) -> f64 {
    1.0e-4 * value.abs().max(1.0)
}

fn certify_glmm_joint_hessian(
    hessian: &DMatrix<f64>,
    context: &str,
) -> std::result::Result<GlmmJointHessianCertification, String> {
    const CONDITION_NUMBER_MAX: f64 = 1.0e10;

    if hessian.nrows() == 0 || hessian.nrows() != hessian.ncols() {
        return Err(format!(
            "{context} has shape {}x{}",
            hessian.nrows(),
            hessian.ncols()
        ));
    }
    if !matrix_is_finite_local(hessian) {
        return Err(format!("{context} contains non-finite entries"));
    }

    let symmetric = 0.5 * (hessian + hessian.transpose());
    let diagonal_scale = (0..symmetric.nrows())
        .map(|index| symmetric[(index, index)].abs())
        .fold(1.0_f64, f64::max);
    let eigen_tolerance = 1.0e-7 * diagonal_scale;
    let eigen = SymmetricEigen::new(symmetric.clone());
    let mut min_eigenvalue = f64::INFINITY;
    let mut max_eigenvalue = 0.0_f64;
    let mut rank = 0usize;
    for value in eigen.eigenvalues.iter().copied() {
        min_eigenvalue = min_eigenvalue.min(value);
        max_eigenvalue = max_eigenvalue.max(value);
        if value > eigen_tolerance {
            rank += 1;
        }
    }

    if min_eigenvalue <= eigen_tolerance {
        return Err(format!(
            "{context} is not positive definite on the active parameter space: min eigenvalue {min_eigenvalue:.6e} <= tolerance {eigen_tolerance:.6e}"
        ));
    }
    if rank != symmetric.nrows() {
        return Err(format!(
            "{context} rank {rank} is below expected rank {}",
            symmetric.nrows()
        ));
    }

    let condition_number = max_eigenvalue / min_eigenvalue;
    if !condition_number.is_finite() || condition_number > CONDITION_NUMBER_MAX {
        return Err(format!(
            "{context} condition number {condition_number:.6e} exceeds certification threshold {CONDITION_NUMBER_MAX:.6e}"
        ));
    }

    let cholesky = symmetric
        .cholesky()
        .ok_or_else(|| format!("{context} Cholesky factorization failed"))?;
    let inverse = cholesky.inverse();
    if !matrix_is_finite_local(&inverse) {
        return Err(format!("{context} inverse contains non-finite entries"));
    }

    Ok(GlmmJointHessianCertification {
        inverse,
        min_eigenvalue,
        condition_number,
        rank,
    })
}

fn matrix_is_finite_local(matrix: &DMatrix<f64>) -> bool {
    matrix.iter().all(|value| value.is_finite())
}

fn matrix_rows_local(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
        .collect()
}

fn matrix_max_asymmetry_local(matrix: &DMatrix<f64>) -> f64 {
    if matrix.nrows() != matrix.ncols() {
        return f64::INFINITY;
    }
    let mut max_asymmetry = 0.0_f64;
    for row in 0..matrix.nrows() {
        for col in 0..row {
            max_asymmetry = max_asymmetry.max((matrix[(row, col)] - matrix[(col, row)]).abs());
        }
    }
    max_asymmetry
}

fn unpivot_glmm_fixed_effect_covariance(
    active_covariance: &DMatrix<f64>,
    pivot: &[usize],
    full_p: usize,
) -> DMatrix<f64> {
    let mut result = DMatrix::zeros(full_p, full_p);
    for active_row in 0..active_covariance.nrows() {
        let full_row = pivot[active_row];
        for active_col in 0..active_covariance.ncols() {
            let full_col = pivot[active_col];
            result[(full_row, full_col)] = active_covariance[(active_row, active_col)];
        }
    }
    result
}

fn glmm_joint_laplace_fixed_effect_covariance_matrix(
    coef_names: Vec<String>,
    covariance: &DMatrix<f64>,
    rank: usize,
    certification: &GlmmJointHessianCertification,
    omitted_boundary_theta_indices: &[usize],
) -> std::result::Result<FixedEffectCovarianceMatrix, String> {
    let finite = matrix_is_finite_local(covariance);
    let symmetric = finite && matrix_max_asymmetry_local(covariance) <= 1.0e-8;
    let details = FixedEffectCovarianceDetails {
        rank: Some(rank),
        expected_rank: Some(coef_names.len()),
        aliased: Vec::new(),
        matrix_rows: covariance.nrows(),
        matrix_cols: covariance.ncols(),
        finite: Some(finite),
        symmetric: Some(symmetric),
    };

    if !finite {
        return Err(
            "joint-laplace GLMM active-Hessian covariance contains non-finite entries".to_string(),
        );
    }
    if !symmetric {
        return Err(
            "joint-laplace GLMM active-Hessian covariance failed symmetry validation".to_string(),
        );
    }

    Ok(FixedEffectCovarianceMatrix::joint_laplace_active_hessian(
        coef_names,
        matrix_rows_local(covariance),
        details,
        glmm_joint_laplace_hessian_notes(certification, omitted_boundary_theta_indices),
    ))
}

fn glmm_joint_laplace_hessian_notes(
    certification: &GlmmJointHessianCertification,
    omitted_boundary_theta_indices: &[usize],
) -> Vec<String> {
    let mut notes = vec![
        "fixed-effect covariance derived from the beta block of the inverse finite-difference Hessian over joint-laplace beta plus interior theta parameters"
            .to_string(),
        format!(
            "joint Hessian certificate: min eigenvalue {:.6e}, condition number {:.6e}, rank {}",
            certification.min_eigenvalue,
            certification.condition_number,
            certification.rank
        ),
    ];
    if !omitted_boundary_theta_indices.is_empty() {
        let labels = omitted_boundary_theta_indices
            .iter()
            .map(|index| index.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        notes.push(format!(
            "boundary covariance parameters held fixed at their lower bounds and omitted from the active Hessian: theta {labels}"
        ));
    }
    notes
}

fn glmm_inference_availability_for_table(
    metadata: &GlmmFitMetadata,
    table: &FixedEffectInferenceTable,
) -> InferenceAvailability {
    if !table.rows.is_empty()
        && table
            .rows
            .iter()
            .all(|row| row.status == FixedEffectInferenceStatus::Available)
    {
        return InferenceAvailability::Available {
            method: "asymptotic_wald_z_joint_laplace_active_hessian".to_string(),
        };
    }

    if metadata.estimation_method == "joint_laplace" {
        return InferenceAvailability::NotAssessed {
            reason: table
                .rows
                .first()
                .and_then(|row| row.reason.clone())
                .unwrap_or_else(|| {
                    "joint-laplace GLMM fixed-effect Hessian certificate did not pass quality gates"
                        .to_string()
                }),
        };
    }

    InferenceAvailability::Unsupported {
        reason: glmm_fixed_effect_inference_unsupported_reason(&metadata.estimation_method),
    }
}

fn glmm_fixed_effect_inference_unsupported_reason(estimation_method: &str) -> String {
    format!(
        "certified GLMM fixed-effect Wald inference is not implemented for {estimation_method}; \
         fast-PIRLS/profiled covariance geometry remains a working-Hessian payload, while only \
         joint-laplace fits with a passing certified active-subspace Hessian over active beta plus \
         interior theta parameters can report Wald SE/z/p/confint"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::model::data::DataFrame;
    use crate::model::linear::FitToleranceOverrides;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn agq_poisson_fixture() -> GeneralizedLinearMixedModel {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut g = Vec::new();
        for grp in 0..5 {
            for obs in 0..8 {
                let xv = obs as f64 - 3.5;
                let eta = 0.8 + 0.1 * xv + [-0.2, 0.1, 0.0, 0.15, -0.05][grp];
                y.push(eta.exp().round().max(1.0));
                x.push(xv);
                g.push(format!("g{}", grp + 1));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("g", g).unwrap();
        let formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
        model.fit().unwrap();
        model
    }

    #[cfg(feature = "nlopt")]
    fn small_joint_poisson_fixture() -> GeneralizedLinearMixedModel {
        let mut data = DataFrame::new();
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        let mut obs = Vec::new();
        let group_effects = [-0.7, 0.2, 0.8, -0.3];
        for (g, effect) in group_effects.iter().enumerate() {
            for j in 0..6 {
                let xv = j as f64 - 2.5;
                let eta = 0.4 + 0.18 * xv + effect;
                let base = eta.exp();
                let overdispersion_bump = if j % 3 == 0 { 2.0 } else { 0.0 };
                y.push((base + overdispersion_bump).round().max(0.0));
                x.push(xv);
                group.push(format!("g{}", g + 1));
                obs.push(format!("o{}_{}", g + 1, j + 1));
            }
        }
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();
        data.add_categorical("obs", obs).unwrap();
        let formula = parse_formula("y ~ 1 + x + (1 | group) + (1 | obs)").unwrap();
        GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, Some(LinkFunction::Log))
            .unwrap()
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn experimental_joint_failed_stop_records_uncertified_certificate() {
        let mut model = small_joint_poisson_fixture();
        model.fit_with_options_impl(1, false).unwrap();
        let start_beta = model.beta.as_slice().to_vec();
        let start_theta = model.theta.clone();
        let start_objective = model.deviance_with_response_constants(1);

        model
            .fit_joint_glmm_from_start(start_beta, start_theta, start_objective, 1, 1, None)
            .unwrap();

        let certificate = model
            .compiler_artifact()
            .optimizer_certificate
            .as_ref()
            .expect("failed joint attempt should still record an optimizer certificate");
        assert!(
            !certificate.evidence.optimizer_stop.acceptable_stop,
            "forced one-evaluation joint fit must not be certified as an acceptable stop"
        );
        assert!(
            certificate.free_gradient_norm.is_none(),
            "failed optimizer stop must not report a passing stationarity residual"
        );
        assert!(
            model
                .opt_summary()
                .return_value
                .starts_with("JOINT_LAPLACE"),
            "forced failure must keep a joint-Laplace return-code namespace"
        );
        assert!(
            model.opt_summary().return_value.contains("MAXEVAL_REACHED"),
            "forced one-evaluation joint fit must report MAXEVAL_REACHED, got {}",
            model.opt_summary().return_value
        );
    }

    #[test]
    fn joint_glmm_stationarity_failure_is_not_converged_interior() {
        let params = vec![1.41606, 0.08172, 0.45, 0.68];
        let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 0.0];
        let gradient = vec![3.5e-2, 1.0e-3, 0.0, 0.0];
        let gradient_tolerance = 2.0e-2;

        let mut optsum = OptSummary::new(params.clone());
        optsum.optimizer = Optimizer::TrustBq;
        optsum.backend = Optimizer::TrustBq.canonical_backend();
        optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
        optsum.finitial = 2845.394;
        optsum.fmin = 2845.394;
        optsum.feval = 23;
        optsum.max_feval = 5000;
        optsum.final_params = params.clone();

        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &optsum,
            &params,
            &lower_bounds,
            Some(2854),
        );
        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                gradient: gradient.clone(),
                hessian: None,
            },
            gradient_tolerance,
            1.0e-6,
        );

        let certification = JointLaplaceCertificationGradient {
            gradient: gradient.clone(),
            probe_gradient: gradient.clone(),
            escalated_indices: Vec::new(),
            unassessable_indices: Vec::new(),
        };
        annotate_glmm_covariance_status(
            &mut certificate,
            &params,
            2,
            &lower_bounds,
            &certification,
            gradient_tolerance,
        );

        assert_eq!(certificate.status, crate::compiler::FitStatus::NotOptimized);
        assert!(
            joint_certificate_requires_fallback(&certificate),
            "assessed stationarity failure should still trigger labelled fallback"
        );
        assert!(certificate.checks.iter().any(|check| {
            matches!(
                check,
                crate::compiler::CertificateCheck::DerivativeMismatch { kind, .. }
                    if kind == "free_gradient_kkt_mismatch"
            )
        }));
        let diagnostic = certificate
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.code == DiagnosticCode::OptimizerNonconvergence
                    && diagnostic
                        .payload
                        .get("stationarity_check")
                        .and_then(serde_json::Value::as_str)
                        == Some("free_gradient_kkt")
            })
            .expect("failed stationarity should be reported as optimizer nonconvergence");
        assert_eq!(
            diagnostic.payload.get("return_code"),
            Some(&serde_json::json!("JOINT_LAPLACE:FTOL_REACHED"))
        );
        assert_eq!(
            diagnostic.payload.get("free_gradient_norm"),
            Some(&serde_json::json!(3.5e-2))
        );
    }

    #[test]
    fn joint_glmm_noise_dominated_stationarity_is_not_assessed() {
        // Probe readings on the two theta components are pure inner-PIRLS
        // noise (bd-01KTQFTH6J0ZFGR5RMV28HAX44 measured 0.703/0.365 at a
        // glmer-equivalent optimum); the escalated steps disagreed, so the
        // components are unassessable. The certificate must say NotAssessed,
        // not NotOptimized, and must not trigger the fast-PIRLS fallback.
        let params = vec![1.43958, 0.08172, 0.3861, 0.5219];
        let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 0.0];
        let probe_gradient = vec![-2.7e-5, 1.0e-3, 0.703, 0.365];
        let gradient_tolerance = 2.0e-2;

        let mut optsum = OptSummary::new(params.clone());
        optsum.optimizer = Optimizer::TrustBq;
        optsum.backend = Optimizer::TrustBq.canonical_backend();
        optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
        optsum.finitial = 2851.2;
        optsum.fmin = 2845.375;
        optsum.feval = 55;
        optsum.max_feval = 820;
        optsum.final_params = params.clone();

        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &optsum,
            &params,
            &lower_bounds,
            Some(2880),
        );
        let certification = JointLaplaceCertificationGradient {
            gradient: probe_gradient.clone(),
            probe_gradient: probe_gradient.clone(),
            escalated_indices: Vec::new(),
            unassessable_indices: vec![2, 3],
        };
        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                gradient: certification.gradient.clone(),
                hessian: None,
            },
            gradient_tolerance,
            1.0e-6,
        );
        annotate_glmm_covariance_status(
            &mut certificate,
            &params,
            2,
            &lower_bounds,
            &certification,
            gradient_tolerance,
        );

        assert_eq!(certificate.status, crate::compiler::FitStatus::NotAssessed);
        assert!(
            !joint_certificate_requires_fallback(&certificate),
            "an unassessable stationarity probe must not discard the joint candidate"
        );
        let diagnostic = certificate
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.code == DiagnosticCode::OptimizerNotAssessed
                    && diagnostic
                        .payload
                        .get("stationarity_check")
                        .and_then(serde_json::Value::as_str)
                        == Some("free_gradient_kkt_noise_dominated")
            })
            .expect("noise-dominated stationarity should be reported as not assessed");
        assert_eq!(
            diagnostic.payload.get("unassessable_indices"),
            Some(&serde_json::json!([2, 3]))
        );
        assert_eq!(
            diagnostic.payload.get("return_code"),
            Some(&serde_json::json!("JOINT_LAPLACE:FTOL_REACHED"))
        );
        assert!(
            !certificate
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == DiagnosticCode::OptimizerNonconvergence),
            "an unassessable probe must not be labelled optimizer nonconvergence"
        );
    }

    #[test]
    fn joint_glmm_escalated_stationarity_pass_certifies_with_evidence_trail() {
        // The default-step probe was noise-dominated on theta but the
        // escalated steps agreed on a near-zero gradient: the fit certifies
        // as interior-converged with an Info trail recording the escalation.
        let params = vec![1.43958, 0.08172, 0.3861, 0.5219];
        let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 0.0];
        let probe_gradient = vec![-2.7e-5, 1.0e-3, 0.703, 0.365];
        let assessed_gradient = vec![-2.7e-5, 1.0e-3, 2.4e-3, -1.1e-3];
        let gradient_tolerance = 2.0e-2;

        let mut optsum = OptSummary::new(params.clone());
        optsum.optimizer = Optimizer::TrustBq;
        optsum.backend = Optimizer::TrustBq.canonical_backend();
        optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
        optsum.finitial = 2851.2;
        optsum.fmin = 2845.375;
        optsum.feval = 55;
        optsum.max_feval = 820;
        optsum.final_params = params.clone();

        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &optsum,
            &params,
            &lower_bounds,
            Some(2880),
        );
        let certification = JointLaplaceCertificationGradient {
            gradient: assessed_gradient.clone(),
            probe_gradient,
            escalated_indices: vec![2, 3],
            unassessable_indices: Vec::new(),
        };
        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                gradient: certification.gradient.clone(),
                hessian: None,
            },
            gradient_tolerance,
            1.0e-6,
        );
        annotate_glmm_covariance_status(
            &mut certificate,
            &params,
            2,
            &lower_bounds,
            &certification,
            gradient_tolerance,
        );

        assert_eq!(
            certificate.status,
            crate::compiler::FitStatus::ConvergedInterior
        );
        assert!(!joint_certificate_requires_fallback(&certificate));
        let trail = certificate
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.code == DiagnosticCode::OptimizerRecovery
                    && diagnostic
                        .payload
                        .get("stationarity_check")
                        .and_then(serde_json::Value::as_str)
                        == Some("free_gradient_kkt_escalated_step")
            })
            .expect("escalated certification must leave an evidence trail");
        assert_eq!(trail.severity, DiagnosticSeverity::Info);
        assert_eq!(
            trail.payload.get("escalated_indices"),
            Some(&serde_json::json!([2, 3]))
        );
        assert_eq!(
            trail.payload.get("probe_gradient_max_abs"),
            Some(&serde_json::json!(0.703))
        );
    }

    #[test]
    fn joint_glmm_nonfinite_objective_stop_is_not_converged_interior() {
        let params = vec![448.9995, 0.79586, 0.42];
        let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];

        let mut optsum = OptSummary::new(params.clone());
        optsum.optimizer = Optimizer::TrustBq;
        optsum.backend = Optimizer::TrustBq.canonical_backend();
        optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
        optsum.finitial = 2540.376;
        optsum.fmin = f64::INFINITY;
        optsum.feval = 61;
        optsum.max_feval = 5000;
        optsum.final_params = params.clone();

        let certificate = OptimizerCertificate::from_opt_summary_with_context(
            &optsum,
            &params,
            &lower_bounds,
            Some(5279),
        );

        assert_eq!(certificate.status, crate::compiler::FitStatus::NotOptimized);
        assert_eq!(certificate.objective_value, None);
        assert!(
            !certificate.evidence.optimizer_stop.acceptable_stop,
            "non-finite objective must invalidate an otherwise acceptable joint stop"
        );
        assert!(
            joint_certificate_requires_fallback(&certificate),
            "non-finite objective joint attempts should trigger the labelled fallback path"
        );
        assert!(certificate.checks.iter().any(|check| {
            matches!(
                check,
                crate::compiler::CertificateCheck::Failed { code, .. }
                    if code == "non_finite_objective"
            )
        }));
        let diagnostic = certificate
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.code == DiagnosticCode::OptimizerNonconvergence
                    && diagnostic
                        .payload
                        .get("objective_finite")
                        .and_then(serde_json::Value::as_bool)
                        == Some(false)
            })
            .expect("non-finite objective should be reported as optimizer nonconvergence");
        assert_eq!(
            diagnostic.payload.get("return_code"),
            Some(&serde_json::json!("JOINT_LAPLACE:FTOL_REACHED"))
        );

        let fallback = agq_poisson_fixture();
        let recovered = uncertified_joint_fallback(&certificate, &optsum, Some(fallback)).unwrap();
        assert!(
            recovered
                .opt_summary()
                .return_value
                .starts_with("JOINT_LAPLACE_FALLBACK_FAST_PIRLS"),
            "non-finite joint objective should return the labelled fallback result"
        );
    }

    #[test]
    fn joint_glmm_not_assessed_stationarity_keeps_joint_candidate() {
        let params = vec![1.2, -0.25, 0.42, 0.68];
        let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 0.0];

        let mut optsum = OptSummary::new(params.clone());
        optsum.optimizer = Optimizer::TrustBq;
        optsum.backend = Optimizer::TrustBq.canonical_backend();
        optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
        optsum.finitial = 1137.42;
        optsum.fmin = 1136.50;
        optsum.feval = 578;
        optsum.max_feval = 1140;
        optsum.final_params = params.clone();

        let mut certificate = OptimizerCertificate::from_opt_summary_with_context(
            &optsum,
            &params,
            &lower_bounds,
            Some(1427),
        );
        certificate.status = crate::compiler::FitStatus::NotAssessed;
        certificate.mark_derivative_checks_not_assessed(
            "objective gradient is not exposed by the current derivative-free optimizer path",
        );

        assert!(
            certificate.evidence.optimizer_stop.acceptable_stop,
            "an acceptable optimizer stop with unassessed derivatives is not a hard optimizer failure"
        );
        assert!(
            matches!(
                certificate.evidence.gradient.method,
                EvidenceMethod::NotAssessed { .. }
            ),
            "regression must exercise the no-gradient/not-assessed derivative path"
        );
        assert!(
            !joint_certificate_requires_fallback(&certificate),
            "not-assessed stationarity should not be conflated with an assessed optimizer failure"
        );

        let fallback = agq_poisson_fixture();
        assert!(
            uncertified_joint_fallback(&certificate, &optsum, Some(fallback)).is_none(),
            "acceptable joint candidates with unassessed stationarity should remain joint fits"
        );
    }

    #[test]
    fn joint_glmm_ftol_at_budget_boundary_keeps_not_available_joint_candidate() {
        let params = vec![1.2, -0.25, 0.42, 0.68];
        let lower_bounds = vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 0.0];

        let mut optsum = OptSummary::new(params.clone());
        optsum.optimizer = Optimizer::TrustBq;
        optsum.backend = Optimizer::TrustBq.canonical_backend();
        optsum.return_value = "JOINT_LAPLACE:FTOL_REACHED".to_string();
        optsum.finitial = 1137.42;
        optsum.fmin = 1136.50;
        optsum.feval = 578;
        optsum.max_feval = 578;
        optsum.final_params = params.clone();

        let certificate = OptimizerCertificate::from_opt_summary_with_context(
            &optsum,
            &params,
            &lower_bounds,
            Some(1427),
        );

        assert!(
            certificate.evidence.optimizer_stop.acceptable_stop,
            "joint FTOL at the evaluation cap is a clean stop, not budget exhaustion"
        );
        assert!(!certificate.evidence.optimizer_stop.budget_exhausted);
        assert_eq!(
            certificate.status,
            crate::compiler::FitStatus::ConvergedInterior
        );
        assert!(
            matches!(
                certificate.evidence.gradient.method,
                EvidenceMethod::NotAvailable { .. }
            ),
            "regression must exercise the production no-gradient/NotAvailable path"
        );
        assert!(
            !joint_certificate_requires_fallback(&certificate),
            "production NotAvailable derivative evidence on an acceptable joint FTOL stop should not discard the joint candidate"
        );

        let fallback = agq_poisson_fixture();
        assert!(
            uncertified_joint_fallback(&certificate, &optsum, Some(fallback)).is_none(),
            "acceptable joint candidates with NotAvailable derivatives should remain joint fits"
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn experimental_joint_failed_stop_returns_labelled_fast_pirls_fallback() {
        let mut model = small_joint_poisson_fixture();
        model.fit_with_options_impl(1, false).unwrap();
        let fallback = model.clone();
        let start_beta = model.beta.as_slice().to_vec();
        let start_theta = model.theta.clone();
        let start_objective = model.deviance_with_response_constants(1);

        model
            .fit_joint_glmm_from_start(
                start_beta,
                start_theta,
                start_objective,
                1,
                1,
                Some(fallback),
            )
            .unwrap();

        assert!(
            model
                .opt_summary()
                .return_value
                .starts_with("JOINT_LAPLACE_FALLBACK_FAST_PIRLS"),
            "fallback result must label the returned estimates, got {}",
            model.opt_summary().return_value
        );
        let certificate = model
            .compiler_artifact()
            .optimizer_certificate
            .as_ref()
            .expect("fallback fit should retain the fast-PIRLS certificate");
        assert!(
            !matches!(certificate.status, crate::compiler::FitStatus::NotOptimized),
            "fallback certificate should describe the returned fast-PIRLS fit, not the failed joint attempt"
        );
        assert!(
            certificate.diagnostics.iter().any(|diagnostic| {
                diagnostic.code == DiagnosticCode::OptimizerRecovery
                    && diagnostic.payload.get("fit_mode")
                        == Some(&serde_json::json!("fallback_fast_pirls"))
                    && diagnostic.payload.get("scorecard_class")
                        == Some(&serde_json::json!("documented_divergence"))
            }),
            "fallback artifact must record the documented-divergence fallback path"
        );
    }

    #[test]
    fn stateless_transform_glmm_end_to_end() {
        // A transformed predictor `I(x^2)` flows through the GLMM build
        // (which wraps an internal LMM) — proving the materialization seam
        // is wired on the GLMM path too.
        use crate::model::traits::MixedModelFit;

        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut g = Vec::new();
        for grp in 0..5 {
            for obs in 0..8 {
                let xv = obs as f64 - 3.5;
                let eta = 0.6 + 0.05 * xv + 0.01 * xv * xv + [-0.2, 0.1, 0.0, 0.15, -0.05][grp];
                y.push(eta.exp().round().max(1.0));
                x.push(xv);
                g.push(format!("g{}", grp + 1));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("g", g).unwrap();

        let formula = parse_formula("y ~ 1 + x + I(x^2) + (1 | g)").unwrap();
        assert!(formula.derived.iter().any(|d| d.label == "I(x^2)"));
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
        model.fit().unwrap();

        let names = model.coef_names();
        assert!(
            names.iter().any(|n| n == "I(x^2)"),
            "GLMM coef_names should contain `I(x^2)`, got {names:?}"
        );
        assert!(model.objective().is_finite());
    }

    #[test]
    fn agq_deviance_restores_state_on_normal_path() {
        let mut model = agq_poisson_fixture();
        let u0 = model.u[0].clone();
        let eta0 = model.eta.clone();
        let mu0 = model.mu.clone();

        let dev = model.deviance(5);
        assert!(dev.is_finite());

        // The AGQ sweep perturbs u/eta/mu; the guard must restore them exactly.
        assert_eq!(model.u[0], u0, "u not restored after deviance(5)");
        assert_eq!(model.eta, eta0, "eta not restored after deviance(5)");
        assert_eq!(model.mu, mu0, "mu not restored after deviance(5)");
    }

    #[test]
    fn agq_restore_guard_restores_state_on_panic() {
        let mut model = agq_poisson_fixture();
        let u0 = model.u[0].clone();
        let eta0 = model.eta.clone();
        let mu0 = model.mu.clone();
        let u0_flat: Vec<f64> = model.u[0].as_slice().to_vec();
        let n_levels = model.u[0].ncols();

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut work = AgqRestoreGuard {
                glmm: &mut model,
                u0_flat: u0_flat.clone(),
            };
            // Desync state the way the AGQ sweep would, then blow up mid-sweep.
            for g in 0..n_levels {
                work.u[0][(0, g)] += 7.0;
            }
            work.update_eta();
            panic!("simulated panic inside AGQ sweep");
        }));

        assert!(result.is_err(), "the closure was expected to panic");
        // Guard's Drop ran during unwinding and restored the model.
        assert_eq!(model.u[0], u0, "u not restored after panic");
        assert_eq!(model.eta, eta0, "eta not restored after panic");
        assert_eq!(model.mu, mu0, "mu not restored after panic");
    }

    #[test]
    fn glmm_builder_matches_direct_construction_byte_for_byte() {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut g = Vec::new();
        for grp in 0..5 {
            for obs in 0..8 {
                let xv = obs as f64 - 3.5;
                let eta = 0.8 + 0.1 * xv + [-0.2, 0.1, 0.0, 0.15, -0.05][grp];
                y.push(eta.exp().round().max(1.0));
                x.push(xv);
                g.push(format!("g{}", grp + 1));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("g", g).unwrap();

        let mut direct = GeneralizedLinearMixedModel::new(
            parse_formula("y ~ 1 + x + (1 | g)").unwrap(),
            &data,
            Family::Poisson,
            None,
        )
        .unwrap();
        direct.fit().unwrap();

        let built = GeneralizedLinearMixedModelBuilder::new(
            parse_formula("y ~ 1 + x + (1 | g)").unwrap(),
            &data,
            Family::Poisson,
        )
        .fit()
        .unwrap();

        assert_eq!(
            built.coef(),
            direct.coef(),
            "builder coef must match direct"
        );
        assert_eq!(built.theta, direct.theta, "builder theta must match direct");
    }

    fn assert_glmm_theta_diagonals_nonnegative(model: &GeneralizedLinearMixedModel) {
        for (idx, &(_, row, col)) in model.lmm.parmap.iter().enumerate() {
            if row == col {
                assert!(
                    model.theta[idx] >= 0.0,
                    "GLMM theta diagonal {idx} should be rectified, got {}",
                    model.theta[idx]
                );
                assert_eq!(
                    model.lmm.optsum.final_params[idx], model.theta[idx],
                    "GLMM OptSummary must store the rectified theta value"
                );
            }
        }
    }

    fn resampled_contra_response(data: &DataFrame) -> Vec<f64> {
        data.numeric("use_num")
            .unwrap()
            .iter()
            .enumerate()
            .map(
                |(idx, &value)| {
                    if idx % 11 == 0 {
                        1.0 - value
                    } else {
                        value
                    }
                },
            )
            .collect()
    }

    fn refit_cold_contra_model(new_y: &[f64]) -> GeneralizedLinearMixedModel {
        let mut data = contra_fixture();
        data.add_numeric("use_num", new_y.to_vec()).unwrap();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit_with_options(true, 1, false).unwrap();
        model
    }

    fn glmm_retained_state_slots(model: &GeneralizedLinearMixedModel) -> usize {
        let matrix_slots = |matrix: &DMatrix<f64>| matrix.nrows() * matrix.ncols();
        let block_slots = |block: &MatrixBlock| block.nrows() * block.ncols();

        model.beta.len()
            + model.beta0.len()
            + model.theta.capacity()
            + model.b.iter().map(matrix_slots).sum::<usize>()
            + model.u.iter().map(matrix_slots).sum::<usize>()
            + model.u0.iter().map(matrix_slots).sum::<usize>()
            + model.eta.len()
            + model.mu.len()
            + model.y.len()
            + model.offset.len()
            + model.wt.capacity()
            + model.devc.capacity()
            + model.devc0.capacity()
            + model.sd.capacity()
            + model.mult.capacity()
            + model.lmm.y.len()
            + matrix_slots(&model.lmm.xy_mat.xy)
            + matrix_slots(&model.lmm.xy_mat.wtxy)
            + model
                .lmm
                .reterms
                .iter()
                .map(|rt| matrix_slots(&rt.z) + matrix_slots(&rt.wtz) + matrix_slots(&rt.lambda))
                .sum::<usize>()
            + model.lmm.a_blocks.iter().map(block_slots).sum::<usize>()
            + model.lmm.l_blocks.iter().map(block_slots).sum::<usize>()
            + model.lmm.optsum.initial.capacity()
            + model.lmm.optsum.final_params.capacity()
            + model.lmm.optsum.fit_log.capacity()
    }

    #[test]
    fn test_gamma_pirls_components_are_link_specific() {
        let eta = 2.0_f64.ln();
        let (sqrtw_log, z_log) =
            pirls_working_observation(Family::Gamma, LinkFunction::Log, 3.0, eta, 2.0, 1.0);
        assert!(
            (sqrtw_log - 1.0).abs() < 1e-12,
            "Gamma-log should use dmu/deta=mu, giving unit IRLS weight"
        );
        assert!(
            (z_log - (eta + 0.5)).abs() < 1e-12,
            "Gamma-log working response should divide by dmu/deta=2"
        );

        let (sqrtw_inverse, z_inverse) =
            pirls_working_observation(Family::Gamma, LinkFunction::Inverse, 3.0, 0.5, 2.0, 1.0);
        assert!(
            (sqrtw_inverse - 2.0).abs() < 1e-12,
            "Gamma-inverse should retain |dmu/deta|=mu^2 in the weight"
        );
        assert!(
            (z_inverse - 0.25).abs() < 1e-12,
            "Gamma-inverse working response must preserve the negative derivative"
        );
    }

    #[test]
    fn test_pirls_no_iter0_break_on_first_step_halving_slack() {
        let accepted_obj = 100.0_f64;
        let old_inflated_reference = accepted_obj * 1.0001;
        let first_step_obj = old_inflated_reference + 0.5e-5;
        let tol = 1e-5_f64;

        assert!(
            (first_step_obj - old_inflated_reference).abs() < tol,
            "this is the old false-convergence case when the halving slack is reused"
        );
        assert!(
            !pirls_converged(first_step_obj, accepted_obj, tol),
            "PIRLS convergence must compare against the uninflated accepted objective"
        );
        assert!(pirls_converged(accepted_obj + tol * 0.5, accepted_obj, tol));
    }

    #[test]
    fn test_pirls_handles_bernoulli_near_separation() {
        let (sqrtw_low, z_low) = pirls_working_observation(
            Family::Bernoulli,
            LinkFunction::Logit,
            0.0,
            -1000.0,
            0.0,
            1.0,
        );
        let (sqrtw_high, z_high) =
            pirls_working_observation(Family::Bernoulli, LinkFunction::Log, 1.0, 1000.0, 1.0, 1.0);

        assert!(sqrtw_low.is_finite());
        assert!(z_low.is_finite());
        assert!(sqrtw_high.is_finite());
        assert!(z_high.is_finite());
        assert!(
            sqrtw_high < 4.0e7,
            "clamped Bernoulli variance should keep sqrt weight bounded, got {sqrtw_high}"
        );
    }

    #[test]
    fn test_pirls_no_inf_weight_under_logit() {
        for (y, eta, mu) in [(0.0, -1000.0, 0.0), (1.0, 1000.0, 1.0)] {
            let (sqrtw, z) =
                pirls_working_observation(Family::Binomial, LinkFunction::Logit, y, eta, mu, 25.0);
            assert!(sqrtw.is_finite());
            assert!(z.is_finite());
        }
    }

    #[test]
    fn test_pirls_no_inf_weight_under_binary_noncanonical_links() {
        for link in [LinkFunction::Probit, LinkFunction::Cloglog] {
            for (y, eta, mu) in [(0.0, -1000.0, 0.0), (1.0, 1000.0, 1.0)] {
                let (sqrtw, z) =
                    pirls_working_observation(Family::Binomial, link, y, eta, mu, 25.0);
                assert!(sqrtw.is_finite(), "{link:?} sqrt weight was {sqrtw}");
                assert!(z.is_finite(), "{link:?} working response was {z}");
            }
        }
    }

    #[test]
    fn test_glmm_offset_enters_linear_predictor() {
        let data = constant_response_fixture(vec![0.0, 1.0, 0.0, 1.0]);
        let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();
        let offset = vec![0.1, -0.2, 0.3, -0.4];

        let model = GeneralizedLinearMixedModel::new_with_offset(
            formula,
            &data,
            Family::Bernoulli,
            None,
            offset.clone(),
        )
        .unwrap();

        for (idx, want) in offset.iter().enumerate() {
            assert!((model.offset[idx] - want).abs() < 1e-12);
            assert!((model.eta[idx] - want).abs() < 1e-12);
            assert!((model.mu[idx] - LinkFunction::Logit.linkinv(*want)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_glmm_offset_validation() {
        let data = constant_response_fixture(vec![0.0, 1.0, 0.0, 1.0]);
        let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();

        let err = GeneralizedLinearMixedModel::new_with_offset(
            formula,
            &data,
            Family::Bernoulli,
            None,
            vec![0.0],
        )
        .unwrap_err();

        match err {
            MixedModelError::InvalidArgument(message) => {
                assert!(message.contains("offset length"));
                assert!(message.contains("number of observations"));
            }
            other => panic!("expected InvalidArgument error, got {other:?}"),
        }
    }

    #[test]
    fn test_pirls_working_response_subtracts_offset() {
        let eta = 1.25_f64;
        let mu = eta.exp();
        let offset = -0.75_f64;
        let (sqrtw_plain, z_plain) =
            pirls_working_observation(Family::Poisson, LinkFunction::Log, 3.0, eta, mu, 2.0);
        let (sqrtw_offset, z_offset) = pirls_working_observation_with_offset(
            Family::Poisson,
            LinkFunction::Log,
            3.0,
            eta,
            mu,
            2.0,
            offset,
        );

        assert!((sqrtw_offset - sqrtw_plain).abs() < 1e-12);
        assert!((z_offset - (z_plain - offset)).abs() < 1e-12);
    }

    #[test]
    fn test_pirls_handles_poisson_log_extreme_offset_scale() {
        for (y, eta, mu) in [
            (0.0, -1000.0, 0.0),
            (1.0, -1000.0, 0.0),
            (0.0, 1000.0, f64::INFINITY),
            (1.0, 1000.0, f64::INFINITY),
        ] {
            let (sqrtw, z) =
                pirls_working_observation(Family::Poisson, LinkFunction::Log, y, eta, mu, 1.0);

            assert!(sqrtw.is_finite(), "sqrt weight was {sqrtw}");
            assert!(sqrtw > 0.0, "sqrt weight should stay positive");
            assert!(sqrtw < 4.0e6, "sqrt weight was {sqrtw}");
            assert!(z.is_finite(), "working response was {z}");
        }
    }

    #[test]
    fn test_pirls_handles_poisson_sqrt_zero_mean_start() {
        for (y, eta, mu) in [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, -0.1, 0.01)] {
            let (sqrtw, z) =
                pirls_working_observation(Family::Poisson, LinkFunction::Sqrt, y, eta, mu, 1.0);
            assert!(sqrtw.is_finite(), "sqrt weight was {sqrtw}");
            assert!(sqrtw > 0.0, "sqrt weight should stay positive");
            assert!(z.is_finite(), "working response was {z}");
        }
    }

    #[test]
    fn test_negative_binomial_pirls_uses_fixed_theta_variance() {
        let theta = 4.0;
        let eta = 2.0_f64.ln();
        let mu = 2.0;
        let (sqrtw, z) = pirls_working_observation_with_family_parameters(
            Family::NegativeBinomial,
            LinkFunction::Log,
            Some(theta),
            3.0,
            eta,
            mu,
            1.0,
        );

        let expected_variance = mu + mu * mu / theta;
        let expected_sqrtw = (mu * mu / expected_variance).sqrt();
        assert_relative_eq!(sqrtw, expected_sqrtw, epsilon = 1e-12);
        assert_relative_eq!(z, eta + 0.5, epsilon = 1e-12);
    }

    fn gamma_dispersion_fixture() -> DataFrame {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        let group_effects = [-0.25, 0.1, 0.3, -0.15];
        for g in 0..4 {
            for obs in 0..5 {
                let xv = obs as f64 - 2.0;
                let eta = 1.2 + 0.25 * xv + group_effects[g];
                let wiggle = 1.0 + 0.06 * ((g + obs) % 3) as f64;
                y.push(eta.exp() * wiggle);
                x.push(xv);
                group.push(format!("g{}", g + 1));
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();
        data
    }

    fn negative_binomial_fixture() -> DataFrame {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        let group_effects = [-0.35, 0.1, 0.25, -0.05];
        for g in 0..4 {
            for obs in 0..6 {
                let xv = obs as f64 - 2.5;
                let eta = 1.0 + 0.18 * xv + group_effects[g];
                let base = eta.exp();
                let overdispersion_bump = if (g + obs) % 3 == 0 { 2.0 } else { 0.0 };
                y.push((base + overdispersion_bump).round().max(0.0));
                x.push(xv);
                group.push(format!("g{}", g + 1));
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();
        data
    }

    #[cfg(not(feature = "nlopt"))]
    fn two_term_poisson_fixture() -> DataFrame {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut g1 = Vec::new();
        let mut g2 = Vec::new();
        for a in 0..4 {
            for b in 0..3 {
                for obs in 0..3 {
                    let xv = obs as f64 - 1.0;
                    let eta = 1.0 + 0.15 * xv + [-0.25, 0.05, 0.2, -0.1][a] + [0.1, -0.15, 0.05][b];
                    y.push(eta.exp().round().max(1.0));
                    x.push(xv);
                    g1.push(format!("g1_{}", a + 1));
                    g2.push(format!("g2_{}", b + 1));
                }
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("g1", g1).unwrap();
        data.add_categorical("g2", g2).unwrap();
        data
    }

    #[test]
    fn test_glmm_constructor_accepts_gamma_with_positive_response() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();

        let model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();

        assert_eq!(model.family, Family::Gamma);
        assert_eq!(model.dispersion(false), 1.0);
        assert_eq!(model.dispersion(true), 1.0);
    }

    #[test]
    fn test_negative_binomial_constructor_requires_fixed_theta() {
        let data = negative_binomial_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();

        let missing_theta = GeneralizedLinearMixedModel::new(
            formula.clone(),
            &data,
            Family::NegativeBinomial,
            None,
        )
        .expect_err("plain NB constructor should require fixed theta");
        match missing_theta {
            MixedModelError::InvalidArgument(message) => {
                assert!(message.contains("negative-binomial"));
                assert!(message.contains("fixed theta"));
            }
            other => panic!("expected InvalidArgument error, got {other:?}"),
        }

        let bad_theta =
            GeneralizedLinearMixedModel::new_negative_binomial(formula.clone(), &data, 0.0, None)
                .expect_err("NB theta must be positive");
        match bad_theta {
            MixedModelError::InvalidArgument(message) => {
                assert!(message.contains("positive"));
                assert!(message.contains("theta"));
            }
            other => panic!("expected InvalidArgument error, got {other:?}"),
        }

        let estimated = GeneralizedLinearMixedModel::new_negative_binomial_estimated(
            formula.clone(),
            &data,
            None,
            None,
        )
        .unwrap();
        assert!(estimated.negative_binomial_theta_estimated());
        assert!(estimated
            .negative_binomial_theta()
            .is_some_and(|theta| theta.is_finite() && theta > 0.0));

        let bad_link = GeneralizedLinearMixedModel::new_negative_binomial(
            formula,
            &data,
            2.5,
            Some(LinkFunction::Sqrt),
        )
        .expect_err("fixed-theta NB only supports log link in this slice");
        match bad_link {
            MixedModelError::UnsupportedFamilyLink { family, link } => {
                assert_eq!(family, "negative_binomial");
                assert_eq!(link, "sqrt");
            }
            other => panic!("expected UnsupportedFamilyLink error, got {other:?}"),
        }
    }

    #[test]
    fn test_negative_binomial_fixed_theta_fit_records_metadata() {
        let data = negative_binomial_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new_negative_binomial(formula, &data, 2.5, None).unwrap();
        model.lmm.optsum.max_feval = 80;

        model.fit_with_options(true, 1, false).unwrap();

        assert_eq!(model.family, Family::NegativeBinomial);
        assert_eq!(model.link, LinkFunction::Log);
        assert_eq!(model.negative_binomial_theta(), Some(2.5));
        assert_eq!(model.dispersion(false), 2.5);
        assert_eq!(model.dispersion(true), 2.5);
        assert_eq!(model.dof(), model.lmm.feterm.rank + model.lmm.parmap.len());
        assert!(model.objective().is_finite());
        assert!(model.loglikelihood().is_finite());

        let metadata = model
            .compiler_artifact()
            .glmm_fit_metadata
            .as_ref()
            .expect("fitted NB GLMM should record fit metadata");
        assert_eq!(
            metadata.family_parameters.get("negative_binomial_theta"),
            Some(&2.5)
        );
        assert_eq!(
            metadata
                .family_parameters
                .get("negative_binomial_variance_power"),
            Some(&2.0)
        );
        assert_eq!(
            metadata
                .family_parameter_sources
                .get("negative_binomial_theta")
                .map(String::as_str),
            Some("fixed")
        );
        assert_eq!(
            model
                .compiler_artifact()
                .model_boundary
                .response_distribution,
            "negative_binomial"
        );
        let payload = crate::stats::FitSummaryPayload::from_generalized_model(&model);
        assert_eq!(
            payload.family_parameters.get("negative_binomial_theta"),
            Some(&2.5)
        );
        assert_eq!(
            payload
                .family_parameter_sources
                .get("negative_binomial_theta")
                .map(String::as_str),
            Some("fixed")
        );

        let vc = model.varcorr();
        assert!(vc.residual_sd.is_none());
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let y_sim = model.simulate_response(&mut rng).unwrap();
        assert_eq!(y_sim.len(), model.nobs());
        assert!(y_sim
            .iter()
            .all(|value| is_nonnegative_integer_response(*value)));
    }

    #[test]
    fn test_negative_binomial_estimated_theta_fit_records_metadata() {
        let data = negative_binomial_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new_negative_binomial_estimated(
            formula, &data, None, None,
        )
        .unwrap();
        let start_theta = model.negative_binomial_theta().unwrap();

        let control = OptimizerControl::auto()
            .with_optimizer(Optimizer::PatternSearch)
            .with_max_feval(80);
        model
            .fit_with_glmm_options(GlmmFitOptions::fast_laplace().with_optimizer_control(control))
            .unwrap();

        let theta = model.negative_binomial_theta().unwrap();
        assert!(model.negative_binomial_theta_estimated());
        assert!(theta.is_finite() && theta > 0.0);
        assert_eq!(model.dispersion(false), theta);
        assert_eq!(model.dispersion(true), theta);
        assert_eq!(
            model.dof(),
            model.lmm.feterm.rank + model.lmm.parmap.len() + 1
        );
        assert!(model.objective().is_finite());
        assert!(model.loglikelihood().is_finite());

        let metadata = model
            .compiler_artifact()
            .glmm_fit_metadata
            .as_ref()
            .expect("estimated NB GLMM should record fit metadata");
        assert_eq!(
            metadata.family_parameters.get("negative_binomial_theta"),
            Some(&theta)
        );
        assert_eq!(
            metadata
                .family_parameters
                .get("negative_binomial_theta_initial"),
            Some(&start_theta)
        );
        assert!(metadata
            .family_parameters
            .get("negative_binomial_theta_outer_iterations")
            .is_some_and(|value| *value >= 1.0));
        assert_eq!(
            metadata
                .family_parameter_sources
                .get("negative_binomial_theta")
                .map(String::as_str),
            Some("estimated")
        );

        let json = serde_json::to_string(model.compiler_artifact()).unwrap();
        let artifact: crate::compiler::CompiledModelArtifact = serde_json::from_str(&json).unwrap();
        let roundtrip_metadata = artifact.glmm_fit_metadata.unwrap();
        assert_eq!(
            roundtrip_metadata
                .family_parameter_sources
                .get("negative_binomial_theta")
                .map(String::as_str),
            Some("estimated")
        );

        let payload = crate::stats::FitSummaryPayload::from_generalized_model(&model);
        assert_eq!(
            payload.family_parameters.get("negative_binomial_theta"),
            Some(&theta)
        );
        assert_eq!(
            payload
                .family_parameter_sources
                .get("negative_binomial_theta")
                .map(String::as_str),
            Some("estimated")
        );
    }

    #[test]
    fn test_gamma_inverse_gaussian_deviance_finite_at_nonpositive_mu() {
        // Regression for audit 03·H2 / mote bd-01KRXCQ8T7J50F739C7ADHFD41:
        // an inverse-link Gamma/InverseGaussian GLMM can transiently propose
        // μ ≤ 0 during PIRLS. The per-observation deviance component must stay
        // finite (a large penalty step-halving can reject), never NaN/Inf
        // that would slip the `obj > halving_bound` guard.
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let gamma = GeneralizedLinearMixedModel::new(
            formula.clone(),
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();
        for &mu in &[0.0_f64, -1e-12, -1.0, -1e6] {
            let d = gamma.dev_resid_component(2.5, mu);
            assert!(d.is_finite(), "Gamma dev at μ={mu} must be finite, got {d}");
        }

        let inv_g = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::InverseGaussian,
            Some(LinkFunction::Log),
        )
        .unwrap();
        for &mu in &[0.0_f64, -1e-9, -3.0] {
            let d = inv_g.dev_resid_component(2.5, mu);
            assert!(
                d.is_finite(),
                "InverseGaussian dev at μ={mu} must be finite, got {d}"
            );
        }
    }

    #[cfg(not(feature = "nlopt"))]
    #[test]
    fn test_glmm_fit_uses_native_cobyla_without_nlopt() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();
        model.lmm.optsum.max_feval = 50;

        model.fit_with_options(true, 1, false).unwrap();

        assert_eq!(model.lmm.optsum.optimizer, Optimizer::Cobyla);
        assert_eq!(model.lmm.optsum.backend.label(), "native");
        assert!(model.lmm.optsum.feval > 0);
        assert!(model.lmm.optsum.fmin.is_finite());
        assert!(!model.lmm.optsum.fit_log.is_empty());
        assert!(model.lmm.compiler_artifact.optimizer_certificate.is_some());
    }

    #[cfg(not(feature = "nlopt"))]
    #[test]
    fn test_glmm_fit_uses_native_pattern_search_when_requested() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();
        model.lmm.optsum.optimizer = Optimizer::PatternSearch;
        model.lmm.optsum.max_feval = 120;

        model.fit_with_options(true, 1, false).unwrap();

        assert_eq!(model.lmm.optsum.optimizer, Optimizer::PatternSearch);
        assert_eq!(model.lmm.optsum.backend.label(), "native");
        assert!(model.lmm.optsum.feval > 0);
        assert!(model.lmm.optsum.fmin.is_finite());
        assert!(!model.lmm.optsum.fit_log.is_empty());
        assert!(model.lmm.compiler_artifact.optimizer_certificate.is_some());
    }

    #[cfg(not(feature = "nlopt"))]
    #[test]
    fn test_glmm_pattern_search_handles_multitheta_poisson_fit() {
        let data = two_term_poisson_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | g1) + (1 | g2)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
        model.lmm.optsum.optimizer = Optimizer::PatternSearch;
        model.lmm.optsum.max_feval = 180;

        model.fit_with_options(true, 1, false).unwrap();

        assert_eq!(model.theta.len(), 2);
        assert_eq!(model.lmm.optsum.optimizer, Optimizer::PatternSearch);
        assert!(model.theta.iter().all(|value| value.is_finite()));
        assert!(model.theta.iter().all(|value| *value >= 0.0));
        assert!(model.objective().is_finite());
    }

    #[test]
    fn test_glmm_constructor_rejects_nonpositive_gamma_response() {
        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let err = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .expect_err("Gamma GLMM should reject zero responses");

        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("gamma"));
                assert!(msg.contains("strictly positive"));
            }
            other => panic!("expected InvalidArgument error, got {other:?}"),
        }
    }

    #[test]
    fn test_gamma_glmm_refit_rejects_nonpositive_response() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();
        let mut new_y = data.numeric("y").unwrap().to_vec();
        new_y[0] = 0.0;

        let err = model
            .refit(&new_y)
            .expect_err("Gamma GLMM refit/bootstrap response must stay strictly positive");

        match err {
            MixedModelError::InvalidArgument(msg) => {
                assert!(msg.contains("gamma"));
                assert!(msg.contains("strictly positive"));
            }
            other => panic!("expected InvalidArgument error, got {other:?}"),
        }
    }

    #[test]
    fn test_glmm_constructor_accepts_normal_nonidentity_dispersion_family() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();

        let model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Normal,
            Some(LinkFunction::Sqrt),
        )
        .unwrap();

        assert_eq!(model.family, Family::Normal);
        assert_eq!(model.link, LinkFunction::Sqrt);
        assert_eq!(model.dispersion(false), 1.0);
    }

    #[test]
    fn test_gamma_glmm_fit_estimates_pearson_dispersion() {
        let data = gamma_dispersion_fixture();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();

        model.fit_with_options(true, 1, false).unwrap();

        let sigma = model.dispersion(false);
        let phi = model.dispersion(true);
        let expected_phi =
            model.pearson_dispersion_numerator() / (model.nobs() - model.lmm.feterm.rank) as f64;

        assert!(sigma.is_finite());
        assert!(sigma > 0.0);
        assert_relative_eq!(phi, sigma * sigma, epsilon = 1e-12);
        assert_relative_eq!(phi, expected_phi, epsilon = 1e-12, max_relative = 1e-12);
        assert_eq!(
            model.dof(),
            model.lmm.feterm.rank + model.lmm.parmap.len() + 1
        );
        assert_relative_eq!(model.varcorr().residual_sd.unwrap(), sigma, epsilon = 1e-12);
    }

    /// Difficult-model corpus row `gamma_near_zero_random_effect_unit`
    /// (see `comparison/difficult_model_scoreboard.toml`). Gamma is
    /// implemented but not 1.0-certified and there is no Gamma comparison
    /// fixture, so the near-zero-random-effect axis is represented here as a
    /// deterministic unit-test diagnostic, never an lme4-parity claim.
    #[test]
    fn test_gamma_glmm_near_zero_random_effect_is_diagnostic() {
        // Every group shares the same linear predictor: the between-group
        // variance is structurally negligible, so the MLE of theta sits at
        // (or against) the zero boundary.
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        for g in 0..5 {
            for obs in 0..6 {
                let xv = obs as f64 - 2.5;
                let eta = 1.1 + 0.2 * xv;
                let wiggle = 1.0 + 0.04 * ((g + obs) % 3) as f64;
                y.push(eta.exp() * wiggle);
                x.push(xv);
                group.push(format!("g{}", g + 1));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();

        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();
        model.lmm_mut().optsum.optimizer = Optimizer::PatternSearch;
        model.lmm_mut().optsum.initial = vec![0.0];
        model.lmm_mut().optsum.max_feval = 1000;

        model.fit_with_options(true, 1, false).unwrap();

        // optimizer_status: a certificate must be recorded for this fit.
        assert!(
            model.lmm.compiler_artifact.optimizer_certificate.is_some(),
            "Gamma near-zero RE fit must record an optimizer certificate"
        );
        assert!(model.lmm.optsum.feval > 0);

        // time_to_certified_fit input: the objective is finite and computable.
        assert!(model.objective().is_finite());

        // certification_status: this is a near-zero boundary diagnostic, not
        // an lme4-parity claim. theta is non-negative and pinned near zero.
        let theta = model.theta();
        assert_eq!(theta.len(), 1);
        assert!(theta[0].is_finite() && theta[0] >= 0.0);
        assert!(
            theta[0] < 1e-1,
            "near-zero random-effect axis: expected theta pinned near the \
             zero boundary, got {}",
            theta[0]
        );

        // The corpus criterion is a *diagnostic*, not just a small number:
        // the near-zero random effect must be reported through the artifact
        // as a singular/boundary covariance, so an ordinary interior fit that
        // merely happened to land on a small theta would NOT satisfy this.
        assert!(
            model.is_singular(),
            "near-zero random-effect axis must surface as a singular/boundary \
             covariance in the artifact, not be inferred from theta alone"
        );
        let certificate = model
            .lmm
            .compiler_artifact
            .optimizer_certificate
            .as_ref()
            .expect("near-zero Gamma GLMM should retain optimizer certificate");
        assert_eq!(
            certificate.status,
            crate::compiler::FitStatus::ConvergedBoundary,
            "near-zero Gamma GLMM should classify as a boundary covariance state; return={}",
            model.lmm.optsum.return_value
        );
        assert!(
            certificate.diagnostics.iter().any(|diagnostic| {
                diagnostic.payload.get("covariance_kkt_classification")
                    == Some(&serde_json::json!("ValidZeroVariance"))
            }),
            "near-zero Gamma GLMM should expose the existing covariance classification leaf"
        );
    }

    #[test]
    fn test_poisson_glmm_near_zero_random_effect_classifies_boundary() {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        for g in 0..5 {
            for obs in 0..6 {
                let xv = obs as f64 - 2.5;
                let eta = 0.8 + 0.15 * xv;
                y.push(eta.exp().round().max(0.0));
                x.push(xv);
                group.push(format!("g{}", g + 1));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();

        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Poisson,
            Some(LinkFunction::Log),
        )
        .unwrap();
        model.fit_with_options(true, 1, false).unwrap();

        let theta = model.theta();
        assert!(
            theta.iter().any(|value| value.abs() <= 1.0e-4),
            "near-zero Poisson random effect should pin a covariance scale near zero, got {theta:?}"
        );
        let certificate = model
            .lmm
            .compiler_artifact
            .optimizer_certificate
            .as_ref()
            .expect("near-zero Poisson GLMM should retain optimizer certificate");
        assert_eq!(
            certificate.status,
            crate::compiler::FitStatus::ConvergedBoundary,
            "near-zero Poisson GLMM should classify as a boundary covariance state"
        );
        assert!(
            certificate.diagnostics.iter().any(|diagnostic| {
                diagnostic.payload.get("covariance_kkt_classification")
                    == Some(&serde_json::json!("ValidZeroVariance"))
            }),
            "near-zero Poisson GLMM should expose the existing covariance classification leaf"
        );
    }

    #[cfg(not(feature = "nlopt"))]
    #[test]
    fn test_glmm_fast_false_uses_native_joint_or_fallback_path_without_nlopt() {
        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.lmm.optsum.max_feval = 80;

        model.fit_with_options(false, 1, false).unwrap();

        assert!(
            model.lmm.optsum.return_value.contains("JOINT_LAPLACE"),
            "fast=false without nlopt must use the labelled joint Laplace path or fallback, got {}",
            model.lmm.optsum.return_value
        );
        assert_eq!(model.lmm.optsum.backend.label(), "native");
        let trust_bq_attempted = model.lmm.optsum.optimizer == Optimizer::TrustBq
            || model
                .lmm
                .compiler_artifact
                .diagnostics
                .iter()
                .any(|diagnostic| {
                    diagnostic.code == DiagnosticCode::OptimizerRecovery
                        && diagnostic.payload.get("joint_optimizer")
                            == Some(&serde_json::json!("trust_bq"))
                });
        assert!(
            trust_bq_attempted,
            "native fast=false should attempt the TrustBQ joint optimizer or record it in fallback diagnostics"
        );
        let metadata = model
            .lmm
            .compiler_artifact
            .glmm_fit_metadata
            .as_ref()
            .expect("native fast=false fit should record GLMM metadata");
        assert_eq!(metadata.optimizer_max_feval, Some(80));
        assert!(metadata.optimizer_feval.unwrap_or_default() >= 0);
        assert!(
            matches!(
                metadata.estimation_method.as_str(),
                "joint_laplace" | "fallback_fast_pirls"
            ),
            "native fast=false must record either joint Laplace or a labelled fallback, got {:?}",
            metadata
        );

        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.lmm.optsum.max_feval = 80;
        model.fit_with_options(false, 7, false).unwrap();
        assert!(
            model.lmm.optsum.return_value.contains("JOINT_AGQ"),
            "valid scalar-RE AGQ should also use the labelled native joint path, got {}",
            model.lmm.optsum.return_value
        );
    }

    #[cfg(not(feature = "nlopt"))]
    #[test]
    fn test_glmm_joint_laplace_honors_configured_max_feval_without_nlopt() {
        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit_with_options_impl(1, false).unwrap();
        let start_beta = model.beta.as_slice().to_vec();
        let start_theta = model.theta.clone();
        let start_objective = model.deviance_with_response_constants(1);

        model
            .fit_joint_glmm_from_start(start_beta, start_theta, start_objective, 1, 3, None)
            .unwrap();

        assert_eq!(model.lmm.optsum.optimizer, Optimizer::TrustBq);
        assert_eq!(model.lmm.optsum.max_feval, 3);
        assert!(model.lmm.optsum.feval <= 3);
        assert!(
            model.lmm.optsum.return_value.contains("MAXEVAL_REACHED"),
            "forced tiny budget should report maxeval, got {}",
            model.lmm.optsum.return_value
        );
        let metadata = model
            .lmm
            .compiler_artifact
            .glmm_fit_metadata
            .as_ref()
            .expect("joint fit should record GLMM metadata");
        assert_eq!(metadata.optimizer, "trust_bq");
        assert_eq!(metadata.optimizer_feval, Some(model.lmm.optsum.feval));
        assert_eq!(metadata.optimizer_max_feval, Some(3));
        assert_eq!(
            metadata.optimizer_fit_log_len,
            Some(model.lmm.optsum.fit_log.len())
        );
        assert_eq!(metadata.optimizer_convergence_status, "budget_exhausted");
    }

    #[cfg(not(feature = "nlopt"))]
    #[test]
    fn test_budgeted_native_joint_laplace_records_high_baseline_multi_re_metadata() {
        let mut correct = Vec::new();
        let mut x = Vec::new();
        let mut participant = Vec::new();
        let mut item = Vec::new();
        for subj in 0..8 {
            let subj_shift = (subj as f64 - 3.5) * 0.08;
            for trial in 0..8 {
                let xv = if trial % 2 == 0 { -0.5 } else { 0.5 };
                let item_id = trial % 4;
                let eta = 2.8 + 0.35 * xv + subj_shift - 0.04 * item_id as f64;
                let p = 1.0 / (1.0 + (-eta).exp());
                let deterministic_u = ((subj * 17 + trial * 11) % 101) as f64 / 101.0;
                correct.push((deterministic_u < p) as i32 as f64);
                x.push(xv);
                participant.push(format!("s{subj}"));
                item.push(format!("i{item_id}"));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("correct", correct).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("participant", participant).unwrap();
        data.add_categorical("item", item).unwrap();
        let formula =
            parse_formula("correct ~ 1 + x + (1 + x | participant) + (1 | item)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.lmm.optsum.max_feval = 40;

        model.fit_with_options(false, 1, false).unwrap();

        assert!(
            model.lmm.optsum.return_value.contains("JOINT_LAPLACE"),
            "budgeted high-baseline multi-RE fit should attempt the labelled joint route, got {}",
            model.lmm.optsum.return_value
        );
        let metadata = model
            .lmm
            .compiler_artifact
            .glmm_fit_metadata
            .as_ref()
            .expect("budgeted joint route should record GLMM metadata");
        assert_eq!(metadata.n_agq, 1);
        assert_eq!(metadata.optimizer_max_feval, Some(40));
        assert!(metadata.optimizer_feval.unwrap_or_default() <= 40);
        assert_eq!(
            metadata.estimation_method, "joint_laplace",
            "budgeted high-baseline multi-RE fit should keep the native joint candidate instead of returning the fast-PIRLS fallback"
        );
        if model.lmm.optsum.return_value.contains("MAXEVAL_REACHED") {
            let certificate = model
                .lmm
                .compiler_artifact
                .optimizer_certificate
                .as_ref()
                .expect("budget-limited joint candidate should retain certificate");
            assert!(certificate.diagnostics.iter().any(|diagnostic| {
                diagnostic.code == DiagnosticCode::OptimizerNonconvergence
                    && diagnostic.payload.get("fit_mode")
                        == Some(&serde_json::json!("uncertified_joint_candidate"))
                    && diagnostic.payload.get("scorecard_class")
                        == Some(&serde_json::json!("budget_limited_joint_candidate"))
            }));
        }
        let trust_bq_attempted = model.lmm.optsum.optimizer == Optimizer::TrustBq
            || model
                .lmm
                .compiler_artifact
                .diagnostics
                .iter()
                .any(|diagnostic| {
                    diagnostic.code == DiagnosticCode::OptimizerRecovery
                        && diagnostic.payload.get("joint_optimizer")
                            == Some(&serde_json::json!("trust_bq"))
                });
        assert!(trust_bq_attempted);
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_glmm_fast_false_uses_labelled_joint_or_fallback_path() {
        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        model.fit_with_options(false, 1, false).unwrap();

        assert!(model.lmm.optsum.return_value.contains("JOINT_LAPLACE"));
        let metadata = model
            .lmm
            .compiler_artifact
            .glmm_fit_metadata
            .as_ref()
            .expect("fast=false fit should record GLMM metadata");
        assert!(
            matches!(
                metadata.estimation_method.as_str(),
                "joint_laplace" | "fallback_fast_pirls"
            ),
            "fast=false must record either certified joint Laplace or a labelled fallback, got {:?}",
            metadata
        );
        if metadata.estimation_method == "joint_laplace" {
            assert_eq!(metadata.objective_definition, "joint_glmm_laplace_deviance");
            assert_eq!(metadata.response_constants, "included");
        } else {
            assert_eq!(metadata.objective_definition, "profiled_glmm_deviance");
            assert_eq!(metadata.response_constants, "dropped");
            assert_eq!(
                metadata.fallback_status.as_deref(),
                Some("fallback_fast_pirls")
            );
        }
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_glmm_fast_false_nagq_uses_labelled_joint_agq_or_fallback_path() {
        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        model.fit_with_options(false, 7, false).unwrap();

        assert!(
            model.lmm.optsum.return_value.contains("JOINT_AGQ"),
            "fast=false n_agq>1 must label the joint AGQ path, got {}",
            model.lmm.optsum.return_value
        );
        let metadata = model
            .lmm
            .compiler_artifact
            .glmm_fit_metadata
            .as_ref()
            .expect("fast=false AGQ fit should record GLMM metadata");
        assert!(
            matches!(
                metadata.estimation_method.as_str(),
                "joint_agq" | "fallback_fast_pirls"
            ),
            "fast=false AGQ must record either certified joint AGQ or a labelled fallback, got {:?}",
            metadata
        );
        if metadata.estimation_method == "joint_agq" {
            assert_eq!(metadata.objective_definition, "joint_glmm_agq_deviance");
            assert_eq!(metadata.response_constants, "included");
            assert_eq!(metadata.n_agq, 7);
        } else {
            assert_eq!(metadata.objective_definition, "profiled_glmm_deviance");
            assert_eq!(metadata.response_constants, "dropped");
            assert_eq!(
                metadata.fallback_status.as_deref(),
                Some("fallback_fast_pirls")
            );
        }
    }

    fn constant_response_fixture(y: Vec<f64>) -> DataFrame {
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_categorical(
            "g",
            vec![
                "a".to_string(),
                "a".to_string(),
                "b".to_string(),
                "b".to_string(),
            ],
        )
        .unwrap();
        data
    }

    fn assert_constant_response_rejected(family: Family, y: Vec<f64>) {
        let data = constant_response_fixture(y);
        let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();

        let err = GeneralizedLinearMixedModel::new(formula, &data, family, None).unwrap_err();

        match err {
            MixedModelError::InvalidArgument(message) => {
                assert!(message.contains("response is constant"));
            }
            other => panic!("expected InvalidArgument error, got {other:?}"),
        }
    }

    #[test]
    fn test_glmm_rejects_constant_response_bernoulli() {
        assert_constant_response_rejected(Family::Bernoulli, vec![0.0, 0.0, 0.0, 0.0]);
        assert_constant_response_rejected(Family::Bernoulli, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_glmm_rejects_constant_response_poisson() {
        assert_constant_response_rejected(Family::Poisson, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_glmm_accepts_near_constant() {
        let data = constant_response_fixture(vec![0.0, 0.0, 0.0, 1.0]);
        let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();

        GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
    }

    #[test]
    fn test_glmm_constructor_supports_requested_family_link_pairs() {
        let mut binomial_data = constant_response_fixture(vec![0.0, 0.25, 0.75, 1.0]);
        binomial_data
            .add_numeric("x", vec![-1.0, -0.5, 0.5, 1.0])
            .unwrap();
        let binomial_formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
        for link in [LinkFunction::Probit, LinkFunction::Cloglog] {
            let model = GeneralizedLinearMixedModel::new(
                binomial_formula.clone(),
                &binomial_data,
                Family::Binomial,
                Some(link),
            )
            .unwrap();
            assert_eq!(model.link, link);
            assert!(model.mu.iter().all(|mu| *mu > 0.0 && *mu < 1.0));
        }

        let mut poisson_data = constant_response_fixture(vec![0.0, 1.0, 2.0, 4.0]);
        poisson_data
            .add_numeric("x", vec![-1.0, -0.5, 0.5, 1.0])
            .unwrap();
        let poisson_formula = parse_formula("y ~ 1 + x + (1 | g)").unwrap();
        let model = GeneralizedLinearMixedModel::new(
            poisson_formula,
            &poisson_data,
            Family::Poisson,
            Some(LinkFunction::Sqrt),
        )
        .unwrap();
        assert_eq!(model.link, LinkFunction::Sqrt);
        assert!(model.mu.iter().all(|mu| *mu >= 0.0));
    }

    #[test]
    fn test_glmm_constructor_rejects_unsupported_family_link_pairs() {
        let data = constant_response_fixture(vec![0.0, 0.0, 0.0, 1.0]);
        let formula = parse_formula("y ~ 1 + (1 | g)").unwrap();

        for (family, link) in [
            (Family::Binomial, LinkFunction::Sqrt),
            (Family::Poisson, LinkFunction::Probit),
        ] {
            let err = GeneralizedLinearMixedModel::new(formula.clone(), &data, family, Some(link))
                .unwrap_err();
            match err {
                MixedModelError::UnsupportedFamilyLink {
                    family: got_family,
                    link: got_link,
                } => {
                    assert_eq!(got_family, family_label(family));
                    assert_eq!(got_link, link_label(link));
                }
                other => panic!("expected UnsupportedFamilyLink error, got {other:?}"),
            }
        }
    }

    /// Build a DataFrame from the embedded contra.csv.
    ///
    /// Columns: use_num (numeric 0/1), age, age2 (= age²), urban (Y/N),
    ///          livch (0+/1/2/3+), urban_dist (interaction string).
    fn contra_fixture() -> DataFrame {
        let csv = include_str!("contra.csv");
        let mut use_num = Vec::new();
        let mut age = Vec::new();
        let mut age2 = Vec::new();
        let mut urban = Vec::new();
        let mut livch = Vec::new();
        let mut urban_dist = Vec::new();

        for line in csv.lines() {
            let parts: Vec<&str> = line.split(',').collect();
            use_num.push(parts[0].parse::<f64>().unwrap());
            age.push(parts[1].parse::<f64>().unwrap());
            age2.push(parts[2].parse::<f64>().unwrap());
            urban.push(parts[3].to_string());
            livch.push(parts[4].to_string());
            urban_dist.push(parts[5].to_string());
        }

        let mut df = DataFrame::new();
        df.add_numeric("use_num", use_num).unwrap();
        df.add_numeric("age", age).unwrap();
        df.add_numeric("age2", age2).unwrap();
        df.add_categorical("urban", urban).unwrap();
        df.add_categorical("livch", livch).unwrap();
        df.add_categorical("urban_dist", urban_dist).unwrap();
        df
    }

    #[test]
    fn glmm_fast_options_record_caller_native_optimizer_override() {
        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        let control = OptimizerControl::auto()
            .with_optimizer(Optimizer::Cobyla)
            .with_max_feval(120)
            .with_tolerances(FitToleranceOverrides::default().with_ftol_abs(1.0e-8));

        model
            .fit_with_glmm_options(GlmmFitOptions::fast_laplace().with_optimizer_control(control))
            .unwrap();

        assert_eq!(model.lmm.optsum.optimizer, Optimizer::Cobyla);
        assert_eq!(model.lmm.optsum.optimizer_source_name(), "caller");
        assert!(model.lmm.optsum.caller_set_field("optimizer"));
        assert!(model.lmm.optsum.caller_set_field("max_feval"));

        let certificate = model
            .lmm
            .optimizer_certificate()
            .expect("GLMM fit should attach optimizer certificate");
        assert_eq!(certificate.optimizer_control.optimizer_source, "caller");
        assert!(certificate
            .optimizer_control
            .caller_set_fields
            .iter()
            .any(|field| field == "optimizer"));
        let metadata = model
            .lmm
            .compiler_artifact
            .glmm_fit_metadata
            .as_ref()
            .expect("GLMM fit should record metadata");
        assert_eq!(metadata.optimizer, "cobyla");
        assert_eq!(metadata.optimizer_source.as_deref(), Some("caller"));
        assert!(metadata
            .caller_set_fields
            .iter()
            .any(|field| field == "max_feval"));
    }

    #[test]
    fn glmm_joint_options_reject_unwired_optimizer_before_fitting() {
        let data = contra_fixture();
        let formula = parse_formula("use_num ~ 1 + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        let err = model
            .fit_with_glmm_options(
                GlmmFitOptions::joint_laplace().with_optimizer(Optimizer::Cobyla),
            )
            .expect_err("joint GLMM Cobyla override should be unsupported");

        assert_eq!(err.code(), "unsupported");
        assert!(!model.is_fitted());
    }

    // ── GLMM parity tests (pirls.jl) ─────────────────────────────────────────

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_contra_glmm_theta_and_deviance() {
        // pirls.jl:
        //   gm0 = fit(MixedModel, first(gfms[:contra]), contra, Bernoulli(); fast=true)
        //   @test isapprox(gm0.θ, [0.5720746212924732], atol=0.001)
        //   @test isapprox(deviance(gm0), 2361.657202855648, atol=0.001)
        //
        // Equivalent formula (pre-computed age² and urban×dist interaction):
        //   use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        model.fit_with_options(true, 1, false).unwrap();

        let theta = &model.theta;
        assert_eq!(theta.len(), 1);
        assert_relative_eq!(theta[0], 0.5720746212924732, epsilon = 0.01);

        let dev = model.deviance(1);
        assert_relative_eq!(dev, 2361.657202855648, epsilon = 1.0);
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_cbpp_binomial_glmm_with_case_weights() {
        // pirls.jl:125-147, MixedModels.jl/test/pirls.jl
        //   gm2 = fit(MixedModel, first(gfms[:cbpp]), cbpp, Binomial();
        //              wts=float(cbpp.hsz), init_from_lmm=[:β, :θ])
        //   @test deviance(gm2, true) ≈ 100.09585620707632 rtol=0.0001
        //   @test loglikelihood(gm2)  ≈ -92.02628187247377 atol=0.001
        //
        // Formula in modelcache.jl:
        //   (incid / hsz) ~ 1 + period + (1 | herd)
        //
        // Bundled cbpp dataset uses lme4 column names: incidence, size, herd,
        // period. The response is the per-trial proportion (incidence/size)
        // and `size` provides the case weights.
        let (data, _) = crate::datasets::load("cbpp").unwrap();
        let incidence = data.numeric("incidence").unwrap();
        let size = data.numeric("size").unwrap();

        let proportion: Vec<f64> = incidence
            .iter()
            .zip(size.iter())
            .map(|(&y, &n)| y / n)
            .collect();
        let weights: Vec<f64> = size.to_vec();

        let mut data_with_proportion = data.clone();
        data_with_proportion
            .add_numeric("proportion", proportion)
            .unwrap();

        let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();

        let mut model = GeneralizedLinearMixedModel::new_with_weights(
            formula,
            &data_with_proportion,
            Family::Binomial,
            None,
            weights,
        )
        .unwrap();

        model.fit_with_options(true, 1, false).unwrap();

        let dev = model.deviance(1);
        // Julia ref: `deviance(gm2, true) ≈ 100.09585620707632`, rtol=0.0001.
        assert_relative_eq!(dev, 100.09585620707632, max_relative = 1e-3);

        // `MixedModelFit::loglikelihood` is on the full normalized `-2 logLik`
        // scale (response normalising constants retained), so it is now
        // directly comparable to Julia's `loglikelihood(gm2) ≈
        // -92.02628187247377` (pirls.jl:125-147). This pins the B1 fix:
        // before it, `loglikelihood` was `-objective/2` on the
        // dropped-constant scale and AIC/BIC were offset by `2·Σ ln C(nᵢ,kᵢ)`.
        // Same fast-PIRLS-vs-joint band as the deviance check above (the
        // log-likelihood inherits that divergence): rtol 1e-3.
        let ll = MixedModelFit::loglikelihood(&model);
        assert_relative_eq!(ll, -92.02628187247377, max_relative = 1e-3);
        // AIC/BIC follow from the corrected log-likelihood + dof.
        let dof = MixedModelFit::dof(&model) as f64;
        assert_relative_eq!(model.aic(), -2.0 * ll + 2.0 * dof, epsilon = 1e-9);
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn experimental_joint_cbpp_objective_matches_lme4_at_lme4_parameters() {
        let (data, _) = crate::datasets::load("cbpp").unwrap();
        let incidence = data.numeric("incidence").unwrap();
        let size = data.numeric("size").unwrap();

        let proportion: Vec<f64> = incidence
            .iter()
            .zip(size.iter())
            .map(|(&y, &n)| y / n)
            .collect();
        let weights: Vec<f64> = size.to_vec();

        let mut data_with_proportion = data.clone();
        data_with_proportion
            .add_numeric("proportion", proportion)
            .unwrap();

        let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new_with_weights(
            formula,
            &data_with_proportion,
            Family::Binomial,
            None,
            weights,
        )
        .unwrap();
        let params = vec![
            -1.398_342_864_233_42,
            -0.991_924_975_185_999,
            -1.128_216_216_147_423,
            -1.579_745_413_889_423,
            0.642_069_926_557_109,
        ];
        let objective = model.joint_glmm_deviance_at_params(&params, 4, 1);
        let lme4_objective = 184.053_132_779_073_5;
        let delta = (objective - lme4_objective).abs();
        assert!(
            delta <= 1.0e-3,
            "cbpp joint objective should match lme4 at the exact lme4 optimum; rust={objective:.9}, lme4={lme4_objective:.9}, delta={delta:.9}"
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_cbpp_agq_deviance_uses_case_weights() {
        let (data, _) = crate::datasets::load("cbpp").unwrap();
        let incidence = data.numeric("incidence").unwrap();
        let size = data.numeric("size").unwrap();

        let proportion: Vec<f64> = incidence
            .iter()
            .zip(size.iter())
            .map(|(&y, &n)| y / n)
            .collect();
        let weights: Vec<f64> = size.to_vec();

        let mut data_with_proportion = data.clone();
        data_with_proportion
            .add_numeric("proportion", proportion)
            .unwrap();

        let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new_with_weights(
            formula,
            &data_with_proportion,
            Family::Binomial,
            None,
            weights,
        )
        .unwrap();
        model.fit_with_options(true, 1, false).unwrap();

        let weighted_agq = model.deviance(5);
        model.wt = vec![1.0; model.y.len()];
        let unit_weight_agq = model.deviance(5);
        assert!(
            (weighted_agq - unit_weight_agq).abs() > 1.0,
            "AGQ deviance must include binomial case weights; weighted={weighted_agq}, unit={unit_weight_agq}"
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_grouseticks_poisson_glmm_deviance() {
        // pirls.jl:194-227, MixedModels.jl/test/pirls.jl
        //   gm4 = fit(MixedModel, only(gfms[:grouseticks]), grouseticks,
        //              Poisson(); fast=true)
        //   @test isapprox(deviance(gm4), 851.4046, atol=0.001)
        //
        // Formula in modelcache.jl:
        //   ticks ~ 1 + year + ch + (1 | index) + (1 | brood) + (1 | location)
        let (data, _) = crate::datasets::load("grouseticks").unwrap();
        let formula = parse_formula(
            "TICKS ~ 1 + YEAR + cHEIGHT + (1 | INDEX) + (1 | BROOD) + (1 | LOCATION)",
        )
        .unwrap();

        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();

        model.fit_with_options(true, 1, false).unwrap();

        let theta = model.theta.clone();
        assert_eq!(theta.len(), 3, "expected three scalar-RE θ components");
        for (i, &t) in theta.iter().enumerate() {
            assert!(t >= 0.0, "θ[{i}] = {t} should be nonnegative");
            assert!(t.is_finite(), "θ[{i}] = {t} should be finite");
        }

        let dev = model.deviance(1);
        // Julia uses atol=0.001; we allow a slightly larger absolute slack
        // to absorb any remaining BOBYQA-vs-NEWUOA optimizer-driver
        // differences. Julia ref deviance: 851.4046.
        assert_relative_eq!(dev, 851.4046, max_relative = 1e-3);
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_contra_glmm_nagq_7_deviance() {
        // pirls.jl (contra testset, lines 94-97):
        //   refit!(gm0; nAGQ=7)
        //   @test isapprox(deviance(gm0), 2360.876, atol=0.001)
        //
        // After re-fitting with 7-point adaptive Gauss-Hermite quadrature
        // the deviance should drop slightly from the Laplace value.
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        model.fit_with_options(true, 7, false).unwrap();

        // optsum should record the AGQ choice and a cached AGQ deviance.
        assert_eq!(model.lmm.optsum.n_agq, 7);
        eprintln!(
            "contra nAGQ=7 deviance: rust = {:.6}, julia ref = 2360.876",
            model.lmm.optsum.fmin
        );
        assert_relative_eq!(model.lmm.optsum.fmin, 2360.876, epsilon = 1.0);

        // Re-evaluating at the converged state should match the cached value
        // exactly (no further optimization between the two calls).
        let dev_agq = model.deviance(7);
        assert_relative_eq!(dev_agq, model.lmm.optsum.fmin, epsilon = 1e-9);

        // The Laplace value (n_agq = 1) should be close to but distinct from
        // the AGQ value at the same θ.
        let dev_lap = model.deviance(1);
        assert!(
            (dev_lap - dev_agq).abs() < 5.0,
            "Laplace and AGQ deviances should be within ~5 units (got {dev_lap} vs {dev_agq})",
        );
    }

    #[test]
    fn test_matrix_block_diag_covers_all_variants() {
        // Direct unit test of the diagonal-extraction helper that AGQ uses
        // on the (1,1) Cholesky block. Contra exercises one variant in
        // practice; this guards the other two so refactors of the L block
        // layout can't silently break AGQ.
        use crate::types::MatrixBlock;
        use nalgebra::{DMatrix, DVector};

        let diag = MatrixBlock::Diagonal(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(matrix_block_diag(&diag), vec![1.0, 2.0, 3.0]);

        let blk0 = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let blk1 = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let bd = MatrixBlock::BlockDiagonal(vec![blk0, blk1]);
        // Diagonal of each 2x2 block in order:
        // blk0 -> (0,0)=1, (1,1)=4; blk1 -> (0,0)=5, (1,1)=8.
        assert_eq!(matrix_block_diag(&bd), vec![1.0, 4.0, 5.0, 8.0]);

        // Dense, rectangular: returns min(rows,cols) diagonals.
        let m = DMatrix::from_row_slice(
            3,
            4,
            &[
                10.0, 0.0, 0.0, 0.0, //
                0.0, 20.0, 0.0, 0.0, //
                0.0, 0.0, 30.0, 0.0,
            ],
        );
        let dense = MatrixBlock::Dense(m);
        assert_eq!(matrix_block_diag(&dense), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_glmm_validate_agq_accepts_single_scalar_re() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        // Single scalar RE: validation should accept any n_agq.
        assert!(model.is_single_scalar_re());
        assert!(model.validate_agq(0).is_ok());
        assert!(model.validate_agq(1).is_ok());
        assert!(model.validate_agq(7).is_ok());
        assert!(model.validate_agq(25).is_ok());
    }

    #[test]
    fn test_glmm_validate_agq_rejects_vector_random_effect() {
        // (1 + age | urban_dist) has vsize == 2 — vector-valued RE.
        // AGQ is only defined for scalar REs; n_agq > 1 must be refused.
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)").unwrap();
        let model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        assert!(
            !model.is_single_scalar_re(),
            "expected vector-RE model not to be classified single-scalar"
        );
        assert_eq!(model.lmm.reterms.len(), 1);
        assert_eq!(model.lmm.reterms[0].vsize, 2);

        // n_agq <= 1 is always allowed (Laplace).
        assert!(model.validate_agq(0).is_ok());
        assert!(model.validate_agq(1).is_ok());

        // n_agq > 1 must error with InvalidArgument citing the vsize mismatch.
        for n_agq in [2_usize, 3, 7, 11] {
            let err = model.validate_agq(n_agq).expect_err(&format!(
                "validate_agq({n_agq}) should error on a vector RE model"
            ));
            match err {
                MixedModelError::InvalidArgument(msg) => {
                    assert!(
                        msg.contains("scalar"),
                        "error message should mention 'scalar' requirement; got {msg}"
                    );
                }
                other => panic!("expected InvalidArgument, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_glmm_validate_agq_rejects_multi_term_random_effects() {
        // Two grouping factors (urban_dist + livch) — multi-term RE.
        // Even with each term scalar, AGQ is undefined.
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + (1 | urban_dist) + (1 | livch)")
                .unwrap();
        let model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        assert_eq!(model.lmm.reterms.len(), 2);
        assert!(!model.is_single_scalar_re());

        assert!(model.validate_agq(1).is_ok());
        for n_agq in [2_usize, 7] {
            let err = model
                .validate_agq(n_agq)
                .expect_err("validate_agq should error on multi-term model");
            assert!(matches!(err, MixedModelError::InvalidArgument(_)));
        }
    }

    #[test]
    fn test_glmm_fit_with_options_rejects_invalid_nagq_up_front() {
        // The fit entry point must preflight the AGQ guard, so users never
        // get a partial fit followed by a panic deep inside deviance().
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        // Laplace fit is fine on a vector RE.
        let lap_result = model.fit_with_options(true, 1, false);
        assert!(lap_result.is_ok());

        // But asking for AGQ on the same shape must error before any work.
        let mut model2 = {
            let data = contra_fixture();
            let formula =
                parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)")
                    .unwrap();
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap()
        };
        let feval_before = model2.lmm.optsum.feval;
        let err = model2.fit_with_options(true, 7, false).expect_err(
            "fit_with_options(_, 7, _) on a vector-RE model should error before fitting",
        );
        assert!(matches!(err, MixedModelError::InvalidArgument(_)));
        assert_eq!(
            model2.lmm.optsum.feval, feval_before,
            "no objective evaluations should have happened on the rejected fit",
        );

        let mut model3 = {
            let data = contra_fixture();
            let formula =
                parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)")
                    .unwrap();
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap()
        };
        let feval_before = model3.lmm.optsum.feval;
        let err = model3
            .fit_with_options(false, 7, false)
            .expect_err("fast=false AGQ must reject invalid RE shape before fitting");
        assert!(matches!(err, MixedModelError::InvalidArgument(_)));
        assert_eq!(
            model3.lmm.optsum.feval, feval_before,
            "fast=false invalid AGQ request must not run the joint optimizer",
        );
    }

    #[test]
    fn test_glmm_refit_resets_theta_to_initial() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        let initial_theta = model.lmm.optsum.initial.clone();

        model.fit_with_options(true, 1, false).unwrap();
        assert!(
            (model.theta[0] - initial_theta[0]).abs() > 1e-6,
            "fixture should move away from its starting theta"
        );

        let new_y = resampled_contra_response(&data);
        model.reset_for_refit(Some(&new_y)).unwrap();

        assert_eq!(model.theta, initial_theta);
        assert_eq!(model.lmm.optsum.final_params, initial_theta);
        assert_eq!(model.lmm.optsum.feval, 0);
        assert!(model.lmm.optsum.return_value.is_empty());
    }

    #[test]
    fn test_glmm_bootstrap_does_not_warm_start() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        let initial_theta = model.lmm.optsum.initial.clone();
        model.fit_with_options(true, 1, false).unwrap();
        let fitted_theta = model.theta.clone();

        let err = model
            .fit_with_options(true, 1, false)
            .expect_err("plain fit_with_options must not silently warm-start a fitted GLMM");
        assert!(matches!(err, MixedModelError::AlreadyFitted));

        let new_y = resampled_contra_response(&data);
        model.reset_for_refit(Some(&new_y)).unwrap();
        assert_eq!(
            model.theta, initial_theta,
            "bootstrap/refit reset must ignore the previous optimum"
        );
        assert_ne!(model.theta, fitted_theta);
    }

    #[test]
    fn test_glmm_refit_after_resample_matches_cold_fit() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut warm_model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        warm_model.fit_with_options(true, 1, false).unwrap();

        let new_y = resampled_contra_response(&data);
        warm_model.refit(&new_y).unwrap();
        let cold_model = refit_cold_contra_model(&new_y);

        assert_relative_eq!(warm_model.theta[0], cold_model.theta[0], epsilon = 1e-8);
        assert_relative_eq!(
            warm_model.lmm.optsum.fmin,
            cold_model.lmm.optsum.fmin,
            epsilon = 1e-8
        );
        for (warm, cold) in warm_model.beta.iter().zip(cold_model.beta.iter()) {
            assert_relative_eq!(warm, cold, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_glmm_repeated_refit_does_not_accumulate_retained_state() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let original_y = data.numeric("use_num").unwrap().to_vec();
        let perturbed_y = resampled_contra_response(&data);
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit_with_options(true, 1, false).unwrap();

        let baseline_slots = glmm_retained_state_slots(&model);
        let baseline_fit_log_capacity = model.lmm.optsum.fit_log.capacity();

        for iteration in 0..8 {
            let y = if iteration % 2 == 0 {
                &perturbed_y
            } else {
                &original_y
            };
            model.refit(y).unwrap();

            assert_eq!(
                glmm_retained_state_slots(&model),
                baseline_slots,
                "GLMM refit should reuse bounded work buffers rather than accumulating retained state"
            );
            assert_eq!(
                model.lmm.optsum.fit_log.capacity(),
                baseline_fit_log_capacity,
                "GLMM optimizer logging must not retain one entry per refit iteration"
            );
        }
    }

    #[test]
    fn test_glmm_theta_probe_penalizes_invalid_theta() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        let value = model.penalized_pirls_deviance_at_theta(&[f64::NAN], 1);
        assert!(
            value.is_infinite() && value.is_sign_positive(),
            "invalid optimizer probes should be penalized, not evaluated from stale state"
        );
    }

    #[test]
    fn test_glmm_final_theta_update_propagates_invalid_theta_error() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();

        let err = model
            .update_pirls_at_theta(&[f64::NAN], true)
            .expect_err("final theta update must propagate invalid-theta errors");
        assert!(matches!(err, MixedModelError::InvalidArgument(_)));
    }

    fn glmm_prediction_data() -> DataFrame {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        let group_effects = [-0.45, 0.1, 0.35, -0.05, 0.25];
        for (g, effect) in group_effects.iter().enumerate() {
            for obs in 0..8 {
                let xv = obs as f64 - 3.5;
                let eta = 0.6 + 0.2 * xv + effect;
                y.push(eta.exp().round().max(0.0));
                x.push(xv);
                group.push(format!("g{}", g + 1));
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();
        data
    }

    fn glmm_certified_prediction_data() -> DataFrame {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        for g in 0..4 {
            for obs in 0..5 {
                let xv = obs as f64 - 2.0;
                let eta = 0.5 + 0.2 * xv + (g as f64 - 1.5) * 0.08;
                y.push(eta.exp() * (0.95 + 0.02 * ((g + obs) % 3) as f64));
                x.push(xv);
                group.push(format!("g{}", g + 1));
            }
        }

        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();
        data
    }

    fn glmm_prediction_fixture() -> (GeneralizedLinearMixedModel, DataFrame) {
        let data = glmm_prediction_data();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
        model.fit().unwrap();
        (model, data)
    }

    #[test]
    fn test_glmm_predict_new_same_data_matches_fitted_on_response_and_link_scale() {
        let (model, data) = glmm_prediction_fixture();

        let response = model
            .predict_new(&data, GlmmPredictionScale::Response, NewReLevels::Error)
            .unwrap();
        let fitted = model.fitted();
        assert_eq!(response.len(), fitted.len());
        for (idx, prediction) in response.iter().enumerate() {
            assert_relative_eq!(
                prediction.expect("training rows have known random-effect levels"),
                fitted[idx],
                epsilon = 1e-9,
                max_relative = 1e-9
            );
        }

        let link = model
            .predict_new(&data, GlmmPredictionScale::Link, NewReLevels::Error)
            .unwrap();
        assert_eq!(link.len(), model.eta.len());
        for (idx, prediction) in link.iter().enumerate() {
            assert_relative_eq!(
                prediction.expect("training rows have known random-effect levels"),
                model.eta[idx],
                epsilon = 1e-9,
                max_relative = 1e-9
            );
        }
    }

    #[test]
    fn test_glmm_predict_new_unseen_levels_follow_policy() {
        let (model, _) = glmm_prediction_fixture();

        let mut newdata = DataFrame::new();
        newdata.add_numeric("y", vec![0.0, 0.0]).unwrap();
        newdata.add_numeric("x", vec![0.0, 0.0]).unwrap();
        newdata
            .add_categorical("group", vec!["NEW".to_string(), "g1".to_string()])
            .unwrap();

        let err = model
            .predict_new(&newdata, GlmmPredictionScale::Response, NewReLevels::Error)
            .unwrap_err();
        assert_eq!(err.code(), "invalid_argument");
        assert!(err.to_string().contains("NEW"));
        assert!(err.to_string().contains("group"));

        let population = model
            .predict_new(
                &newdata,
                GlmmPredictionScale::Response,
                NewReLevels::Population,
            )
            .unwrap();
        assert_eq!(population.len(), 2);
        assert!(population[0].is_some());
        assert!(population[1].is_some());

        let missing = model
            .predict_new(
                &newdata,
                GlmmPredictionScale::Response,
                NewReLevels::Missing,
            )
            .unwrap();
        assert_eq!(missing[0], None);
        assert!(missing[1].is_some());
    }

    #[test]
    fn test_glmm_predict_new_with_offset_applies_offset_on_link_scale() {
        let (model, data) = glmm_prediction_fixture();

        let base = model
            .predict_new(&data, GlmmPredictionScale::Link, NewReLevels::Error)
            .unwrap();
        let offset = vec![0.25; data.nrow()];
        let shifted = model
            .predict_new_with_offset(
                &data,
                Some(&offset),
                GlmmPredictionScale::Link,
                NewReLevels::Error,
            )
            .unwrap();

        for (base, shifted) in base.iter().zip(shifted.iter()) {
            assert_relative_eq!(
                shifted.expect("known level"),
                base.expect("known level") + 0.25,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_glmm_predict_new_variance_returns_degraded_working_delta_payload() {
        let (model, data) = glmm_prediction_fixture();

        let payload = model
            .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
            .unwrap();
        assert_eq!(
            payload.method,
            PredictionVarianceMethod::GlmmPirlsLaplaceWorkingDelta
        );
        assert_eq!(payload.confidence_level, Some(0.95));
        assert_eq!(payload.rows.len(), data.nrow());
        let fitted = model.fitted();
        let first = &payload.rows[0];
        assert_eq!(first.status, PredictionVarianceStatus::Degraded);
        assert_relative_eq!(
            first.prediction.expect("GLMM point prediction"),
            fitted[0],
            epsilon = 1e-9,
            max_relative = 1e-9
        );
        assert!(first.fixed_variance.unwrap() > 0.0);
        assert!(first.random_variance.unwrap() >= 0.0);
        assert!(first.fixed_random_covariance.unwrap().is_finite());
        assert!(first.combined_variance.unwrap() > 0.0);
        assert!(first.se_fit.unwrap() > 0.0);
        assert!(first.prediction_variance.unwrap() > 0.0);
        assert!(first.confidence_lower.unwrap() < first.prediction.unwrap());
        assert!(first.confidence_upper.unwrap() > first.prediction.unwrap());
        let prediction_lower = first.prediction_lower.unwrap();
        let prediction_upper = first.prediction_upper.unwrap();
        assert!(prediction_lower >= 0.0);
        assert!(prediction_lower <= prediction_upper);
        assert_eq!(prediction_lower.fract(), 0.0, "poisson bounds are counts");
        assert_eq!(prediction_upper.fract(), 0.0, "poisson bounds are counts");
        assert!(first.prediction_variance.unwrap() > first.combined_variance.unwrap());
        let reason = first.reason.as_deref().unwrap_or("");
        assert!(reason.contains("the fast-PIRLS profiled optimum certificate was not issued"));
        assert!(reason.contains("GlmmFitOptions::joint_laplace()"));
    }

    fn glmm_certified_pirls_poisson_fixture() -> (GeneralizedLinearMixedModel, DataFrame) {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        let group_effects = [-0.9_f64, -0.3, 0.2, 0.7, 1.1, -0.5];
        for (g, effect) in group_effects.iter().enumerate() {
            for obs in 0..10 {
                let xv = (obs as f64 - 4.5) / 3.0;
                let eta = 1.0 + 0.3 * xv + effect;
                let noise = 0.85 + 0.3 * (((g * 13 + obs * 7) % 11) as f64 / 10.0);
                y.push((eta.exp() * noise).round().max(0.0));
                x.push(xv);
                group.push(format!("g{}", g + 1));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Poisson, None).unwrap();
        model.fit().unwrap();
        (model, data)
    }

    #[test]
    #[cfg(feature = "nlopt")]
    fn test_glmm_pirls_certified_prediction_variance_rows_available() {
        let (model, data) = glmm_certified_pirls_poisson_fixture();
        assert!(
            matches!(model.pirls_profiled_optimum_certificate, Some(Ok(_))),
            "fixture should certify: {:?}",
            model.pirls_profiled_optimum_certificate
        );

        let payload = model
            .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
            .unwrap();
        assert_eq!(
            payload.method,
            PredictionVarianceMethod::GlmmPirlsProfiledCertifiedConditionalDelta
        );
        assert!(payload
            .notes
            .iter()
            .any(|note| note.contains("certified profiled optimum")));
        for row in &payload.rows {
            assert_eq!(row.status, PredictionVarianceStatus::Available);
            assert_eq!(row.reason, None);
            let prediction = row.prediction.expect("point prediction");
            assert!(row.se_fit.unwrap() > 0.0);
            // The Poisson future-observation variance is dominated by the
            // family term E[mu], so it must exceed the fitted-mean variance.
            assert!(row.prediction_variance.unwrap() > row.combined_variance.unwrap());
            let lower = row.prediction_lower.unwrap();
            let upper = row.prediction_upper.unwrap();
            assert_eq!(lower.fract(), 0.0);
            assert_eq!(upper.fract(), 0.0);
            assert!(lower >= 0.0);
            assert!(lower <= prediction.ceil());
            assert!(upper >= prediction.floor());
            assert!(upper > lower);
        }

        assert!(model
            .compiler_artifact()
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic
                .payload
                .get("glmm_pirls_profiled_optimum_certificate")
                .and_then(serde_json::Value::as_str)
                == Some("issued")));
    }

    #[test]
    #[cfg(not(feature = "nlopt"))]
    fn test_glmm_pirls_native_prediction_variance_rows_degrade_without_certificate() {
        let (model, data) = glmm_certified_pirls_poisson_fixture();
        assert!(
            matches!(model.pirls_profiled_optimum_certificate, Some(Err(_))),
            "native fixture should keep uncertified geometry explicit: {:?}",
            model.pirls_profiled_optimum_certificate
        );

        let payload = model
            .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
            .unwrap();
        assert_eq!(
            payload.method,
            PredictionVarianceMethod::GlmmPirlsLaplaceWorkingDelta
        );
        for row in &payload.rows {
            assert_eq!(row.status, PredictionVarianceStatus::Degraded);
            let reason = row.reason.as_deref().unwrap_or("");
            assert!(reason.contains("the fast-PIRLS profiled optimum certificate was not issued"));
            assert!(reason.contains("GlmmFitOptions::joint_laplace()"));
            assert!(row.se_fit.unwrap() > 0.0);
            assert!(row.prediction_variance.unwrap() > row.combined_variance.unwrap());
        }
    }

    #[test]
    fn test_glmm_pirls_uncertified_fit_keeps_degraded_with_refit_guidance() {
        let (mut model, data) = glmm_certified_pirls_poisson_fixture();
        model.pirls_profiled_optimum_certificate =
            Some(Err("forced certificate failure for test".to_string()));

        let payload = model
            .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
            .unwrap();
        assert_eq!(
            payload.method,
            PredictionVarianceMethod::GlmmPirlsLaplaceWorkingDelta
        );
        let first = &payload.rows[0];
        assert_eq!(first.status, PredictionVarianceStatus::Degraded);
        let reason = first.reason.as_deref().unwrap();
        assert!(reason.contains("forced certificate failure for test"));
        assert!(reason.contains("GlmmFitOptions::joint_laplace()"));
        // Degraded rows still carry the (uncertified) predictive columns so
        // downstream layers can surface them together with the reason.
        assert!(first.prediction_variance.unwrap() > 0.0);
    }

    #[test]
    fn test_glmm_link_scale_rows_do_not_carry_future_observation_columns() {
        let (model, data) = glmm_certified_pirls_poisson_fixture();
        let payload = model
            .predict_new_variance(&data, GlmmPredictionScale::Link, NewReLevels::Error)
            .unwrap();
        let first = &payload.rows[0];
        if matches!(model.pirls_profiled_optimum_certificate, Some(Ok(_))) {
            assert_eq!(first.status, PredictionVarianceStatus::Available);
        } else {
            assert_eq!(first.status, PredictionVarianceStatus::Degraded);
        }
        assert_eq!(first.prediction_variance, None);
        assert_eq!(first.prediction_lower, None);
        assert_eq!(first.prediction_upper, None);
        assert!(payload
            .notes
            .iter()
            .any(|note| note.contains("response-scale objects")));
    }

    #[test]
    fn test_glmm_bernoulli_future_observation_bounds_are_support_points() {
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        for g in 0..8usize {
            for obs in 0..12usize {
                let idx = g * 12 + obs;
                let xv = (obs as f64 - 5.5) / 2.2;
                let eta = -0.3 + 1.8 * xv + (g as f64 - 3.5) * 0.25;
                let p = 1.0 / (1.0 + (-eta).exp());
                let u = ((idx * 37 + 11) % 97) as f64 / 97.0;
                y.push(if p > u { 1.0 } else { 0.0 });
                x.push(xv);
                group.push(format!("g{}", g + 1));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();

        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit_with_options(false, 1, false).unwrap();

        let payload = model
            .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
            .unwrap();
        let mut saw_zero_lower = false;
        let mut saw_unit_upper = false;
        for row in &payload.rows {
            let lower = row.prediction_lower.unwrap();
            let upper = row.prediction_upper.unwrap();
            assert!(lower == 0.0 || lower == 1.0);
            assert!(upper == 0.0 || upper == 1.0);
            assert!(lower <= upper);
            let variance = row.prediction_variance.unwrap();
            // Law of total variance for a Bernoulli future observation:
            // bounded by the maximal Bernoulli variance.
            assert!(variance > 0.0 && variance <= 0.25 + 1.0e-9);
            saw_zero_lower |= lower == 0.0;
            saw_unit_upper |= upper == 1.0;
        }
        assert!(saw_zero_lower && saw_unit_upper);
    }

    #[test]
    fn test_glmm_binomial_future_observation_refused_with_trial_count_reason() {
        let (data, _) = crate::datasets::load("cbpp").unwrap();
        let incidence = data.numeric("incidence").unwrap();
        let size = data.numeric("size").unwrap();
        let proportion: Vec<f64> = incidence
            .iter()
            .zip(size.iter())
            .map(|(&y, &n)| y / n)
            .collect();
        let weights: Vec<f64> = size.to_vec();
        let mut data_with_proportion = data.clone();
        data_with_proportion
            .add_numeric("proportion", proportion)
            .unwrap();
        let formula = parse_formula("proportion ~ 1 + period + (1 | herd)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new_with_weights(
            formula,
            &data_with_proportion,
            Family::Binomial,
            None,
            weights,
        )
        .unwrap();
        model.fit().unwrap();

        let payload = model
            .predict_new_variance(
                &data_with_proportion,
                GlmmPredictionScale::Response,
                NewReLevels::Error,
            )
            .unwrap();
        let first = &payload.rows[0];
        assert_eq!(first.prediction_variance, None);
        assert_eq!(first.prediction_lower, None);
        assert_eq!(first.prediction_upper, None);
        assert!(first.confidence_lower.is_some());
        assert!(payload
            .notes
            .iter()
            .any(|note| note.contains("trial count")));
    }

    #[test]
    fn test_discrete_mixture_quantile_matches_single_poisson_reference() {
        let poisson = PoissonDist::new(4.2).unwrap();
        let cdf = |t: u64| poisson.cdf(t);
        // scipy.stats.poisson.ppf reference values for lambda = 4.2.
        assert_eq!(discrete_mixture_quantile(&cdf, 0.025, 4.2), Some(1.0));
        assert_eq!(discrete_mixture_quantile(&cdf, 0.975, 4.2), Some(9.0));
        assert_eq!(discrete_mixture_quantile(&cdf, 0.005, 4.2), Some(0.0));
        assert_eq!(discrete_mixture_quantile(&cdf, 0.995, 4.2), Some(10.0));
    }

    #[test]
    fn test_inverse_gaussian_cdf_matches_scipy_reference() {
        // scipy.stats.invgauss(mu=1, scale=1).cdf(1) and
        // scipy.stats.invgauss(mu=4, scale=0.5).cdf(3) (mean 2, shape 0.5).
        // statrs's erfc-based normal CDF carries ~1e-11 absolute error, so
        // the comparison tolerance reflects that, not the IG formula.
        assert_relative_eq!(
            inverse_gaussian_cdf(1.0, 1.0, 1.0),
            0.6681020012231706,
            epsilon = 1.0e-9
        );
        assert_relative_eq!(
            inverse_gaussian_cdf(3.0, 2.0, 0.5),
            0.8343083811593116,
            epsilon = 1.0e-9
        );
        assert_eq!(inverse_gaussian_cdf(0.0, 1.0, 1.0), 0.0);
        assert_eq!(inverse_gaussian_cdf(-1.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn test_standard_normal_ln_cdf_tail_is_continuous_and_consistent() {
        let direct = Normal::new(0.0, 1.0).unwrap().cdf(-5.0).ln();
        assert_relative_eq!(standard_normal_ln_cdf(-5.0), direct, epsilon = 1.0e-12);
        let just_above = standard_normal_ln_cdf(-36.9);
        let just_below = standard_normal_ln_cdf(-37.1);
        assert!(just_below < just_above);
        assert!((just_below - just_above).abs() < 8.0);
    }

    #[test]
    fn test_continuous_mixture_quantile_matches_single_normal_reference() {
        let normal = Normal::new(2.0, 3.0).unwrap();
        let cdf = |t: f64| normal.cdf(t);
        let q = continuous_mixture_quantile(&cdf, 0.975, None, 2.0, 3.0).unwrap();
        assert_relative_eq!(q, 2.0 + 1.959963984540054 * 3.0, epsilon = 1.0e-6);
        let q_low = continuous_mixture_quantile(&cdf, 0.025, None, 2.0, 3.0).unwrap();
        assert_relative_eq!(q_low, 2.0 - 1.959963984540054 * 3.0, epsilon = 1.0e-6);
    }

    #[test]
    fn test_glmm_response_scale_confidence_bounds_stay_in_family_range() {
        // Strong slope pushes fitted probabilities near 0 and 1 so symmetric
        // response-scale bounds would escape (0, 1).
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut group = Vec::new();
        for g in 0..8usize {
            for obs in 0..12usize {
                let idx = g * 12 + obs;
                let xv = (obs as f64 - 5.5) / 2.2;
                let eta = -0.3 + 1.8 * xv + (g as f64 - 3.5) * 0.25;
                let p = 1.0 / (1.0 + (-eta).exp());
                let u = ((idx * 37 + 11) % 97) as f64 / 97.0;
                y.push(if p > u { 1.0 } else { 0.0 });
                x.push(xv);
                group.push(format!("g{}", g + 1));
            }
        }
        let mut data = DataFrame::new();
        data.add_numeric("y", y).unwrap();
        data.add_numeric("x", x).unwrap();
        data.add_categorical("group", group).unwrap();

        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit_with_options(false, 1, false).unwrap();

        let response = model
            .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
            .unwrap();
        let link = model
            .predict_new_variance(&data, GlmmPredictionScale::Link, NewReLevels::Error)
            .unwrap();
        assert!(response
            .notes
            .iter()
            .any(|note| note.contains("mapped through the inverse link")));

        let z = 1.959963984540054;
        let mut rows_with_bounds = 0;
        let mut symmetric_would_escape = false;
        for (row, link_row) in response.rows.iter().zip(link.rows.iter()) {
            let (Some(fit), Some(se_fit), Some(lower), Some(upper)) = (
                row.prediction,
                row.se_fit,
                row.confidence_lower,
                row.confidence_upper,
            ) else {
                continue;
            };
            rows_with_bounds += 1;
            assert!(
                lower > 0.0 && upper < 1.0,
                "row {}: response bounds ({lower}, {upper}) escape (0, 1)",
                row.row
            );
            assert!(lower < fit && fit < upper);
            if fit - z * se_fit < 0.0 || fit + z * se_fit > 1.0 {
                symmetric_would_escape = true;
            }
            // The response bounds must be the link-scale bounds mapped
            // through the inverse link.
            let link_lower = link_row.confidence_lower.expect("link lower bound");
            let link_upper = link_row.confidence_upper.expect("link upper bound");
            assert_relative_eq!(
                lower,
                1.0 / (1.0 + (-link_lower).exp()),
                epsilon = 1e-12,
                max_relative = 1e-12
            );
            assert_relative_eq!(
                upper,
                1.0 / (1.0 + (-link_upper).exp()),
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
        assert!(rows_with_bounds > 0, "fixture should yield bounded rows");
        assert!(
            symmetric_would_escape,
            "fixture should reproduce the symmetric-bounds escape this test guards against"
        );
    }

    #[test]
    fn test_glmm_predict_new_variance_reports_joint_laplace_conditional_rows_available() {
        let data = glmm_certified_prediction_data();
        let formula = parse_formula("y ~ 1 + x + (1 | group)").unwrap();
        let mut model = GeneralizedLinearMixedModel::new(
            formula,
            &data,
            Family::Gamma,
            Some(LinkFunction::Log),
        )
        .unwrap();
        model.fit_with_options(false, 1, false).unwrap();

        let artifact = model.compiler_artifact();
        let covariance = artifact
            .fixed_effect_covariance_matrix
            .as_ref()
            .expect("joint-laplace fit should expose fixed covariance");
        assert_eq!(
            covariance.method,
            FixedEffectCovarianceMethod::JointLaplaceActiveHessian
        );
        let matrix = covariance
            .matrix
            .as_ref()
            .expect("certified covariance should carry matrix values");

        let payload = model
            .predict_new_variance(&data, GlmmPredictionScale::Response, NewReLevels::Error)
            .unwrap();
        let first = &payload.rows[0];
        assert_eq!(
            payload.method,
            PredictionVarianceMethod::GlmmJointLaplaceConditionalDelta
        );
        assert_eq!(first.status, PredictionVarianceStatus::Available);
        assert_eq!(first.reason, None);
        assert!(payload
            .notes
            .iter()
            .any(|note| note.contains("conditional-mode covariance")));

        assert!(matrix.iter().flatten().all(|value| value.is_finite()));
        assert!(first.fixed_variance.expect("fixed component") > 0.0);
        assert!(first.random_variance.expect("random component") >= 0.0);
        assert!(first
            .fixed_random_covariance
            .expect("fixed/random covariance")
            .is_finite());
        assert_relative_eq!(
            first.combined_variance.expect("combined component"),
            first.fixed_variance.unwrap()
                + first.random_variance.unwrap()
                + 2.0 * first.fixed_random_covariance.unwrap(),
            epsilon = 1.0e-8,
            max_relative = 1.0e-8
        );
        assert!(first.se_fit.unwrap() > 0.0);
        assert!(first.prediction_variance.unwrap() > 0.0);
        assert!(first.confidence_lower.unwrap() < first.prediction.unwrap());
        assert!(first.confidence_upper.unwrap() > first.prediction.unwrap());
        let prediction_lower = first.prediction_lower.unwrap();
        let prediction_upper = first.prediction_upper.unwrap();
        assert!(prediction_lower > 0.0, "gamma future bounds stay positive");
        assert!(prediction_lower < first.prediction.unwrap());
        assert!(prediction_upper > first.prediction.unwrap());
        assert!(prediction_lower <= first.confidence_lower.unwrap() + 1.0e-9);
        assert!(prediction_upper >= first.confidence_upper.unwrap() - 1.0e-9);

        let link_payload = model
            .predict_new_variance(&data, GlmmPredictionScale::Link, NewReLevels::Error)
            .unwrap();

        // lme4 2.0.1 reference:
        // glmer(y ~ 1 + x + (1 | group), data, family = Gamma(link = "log"),
        //       nAGQ = 1, control = glmerControl(optimizer = "bobyqa"))
        // predict(..., newdata = data[1:5,], re.form = NULL, se.fit = TRUE)
        // emits lme4's documented approximation warning for se.fit.
        let lme4_response_fit = [0.9529792, 1.1645747, 1.4231520, 1.7391427, 2.1252947];
        let lme4_response_se = [0.01402705, 0.01476743, 0.01696936, 0.02205325, 0.03128255];
        for (idx, (row, (expected_fit, expected_se))) in payload
            .rows
            .iter()
            .take(lme4_response_fit.len())
            .zip(lme4_response_fit.into_iter().zip(lme4_response_se))
            .enumerate()
        {
            let fit = row.prediction.expect("response-scale GLMM prediction");
            assert!(
                (fit - expected_fit).abs() <= 5.0e-5_f64.max(5.0e-5 * expected_fit.abs()),
                "response-scale lme4 fit parity row {idx}: observed {fit}, expected {expected_fit}"
            );
            let se_fit = row.se_fit.expect("response-scale GLMM se.fit");
            assert!(
                (se_fit - expected_se).abs() <= 5.0e-5_f64.max(5.0e-5 * expected_se.abs()),
                "response-scale lme4 se.fit parity row {idx}: observed {se_fit}, expected {expected_se}"
            );
        }

        let lme4_link_fit = [-0.0481622, 0.1523560, 0.3528741, 0.5533923, 0.7539105];
        let lme4_link_se = [0.01471916, 0.01268053, 0.01192378, 0.01268053, 0.01471916];
        let lme4_link_fixed = [
            0.0006883062,
            0.0006324485,
            0.0006138292,
            0.0006324485,
            0.0006883062,
        ];
        let lme4_link_random = [0.0006815289; 5];
        let lme4_link_cross = [-0.0005765908; 5];
        let lme4_link_combined = [
            0.0002166536,
            0.0001607959,
            0.0001421766,
            0.0001607959,
            0.0002166536,
        ];
        for (idx, (row, (expected_fit, expected_se))) in link_payload
            .rows
            .iter()
            .take(lme4_link_fit.len())
            .zip(lme4_link_fit.into_iter().zip(lme4_link_se))
            .enumerate()
        {
            assert_eq!(row.status, PredictionVarianceStatus::Available);
            let fit = row.prediction.expect("link-scale GLMM prediction");
            assert!(
                (fit - expected_fit).abs() <= 5.0e-5_f64.max(5.0e-5 * expected_fit.abs()),
                "link-scale lme4 fit parity row {idx}: observed {fit}, expected {expected_fit}"
            );
            let se_fit = row.se_fit.expect("link-scale GLMM se.fit");
            assert!(
                (se_fit - expected_se).abs() <= 5.0e-5_f64.max(5.0e-5 * expected_se.abs()),
                "link-scale lme4 se.fit parity row {idx}: observed {se_fit}, expected {expected_se}"
            );
            let fixed = row.fixed_variance.expect("link-scale GLMM fixed component");
            assert!(
                (fixed - lme4_link_fixed[idx]).abs()
                    <= 1.0e-6_f64.max(1.0e-6 * lme4_link_fixed[idx].abs()),
                "link-scale lme4 fixed component parity row {idx}: observed {fixed}, expected {}",
                lme4_link_fixed[idx]
            );
            let random = row
                .random_variance
                .expect("link-scale GLMM random component");
            assert!(
                (random - lme4_link_random[idx]).abs()
                    <= 1.0e-6_f64.max(1.0e-6 * lme4_link_random[idx].abs()),
                "link-scale lme4 random component parity row {idx}: observed {random}, expected {}",
                lme4_link_random[idx]
            );
            let cross = row
                .fixed_random_covariance
                .expect("link-scale GLMM fixed/random component");
            assert!(
                (cross - lme4_link_cross[idx]).abs()
                    <= 1.0e-6_f64.max(1.0e-6 * lme4_link_cross[idx].abs()),
                "link-scale lme4 fixed/random component parity row {idx}: observed {cross}, expected {}",
                lme4_link_cross[idx]
            );
            let combined = row
                .combined_variance
                .expect("link-scale GLMM combined component");
            assert!(
                (combined - lme4_link_combined[idx]).abs()
                    <= 1.0e-6_f64.max(1.0e-6 * lme4_link_combined[idx].abs()),
                "link-scale lme4 combined component parity row {idx}: observed {combined}, expected {}",
                lme4_link_combined[idx]
            );
        }
    }

    #[test]
    fn test_glmm_predict_new_variance_unseen_level_keeps_unavailable_reason() {
        let (model, _) = glmm_prediction_fixture();

        let mut newdata = DataFrame::new();
        newdata.add_numeric("y", vec![0.0, 0.0]).unwrap();
        newdata.add_numeric("x", vec![0.0, 0.0]).unwrap();
        newdata
            .add_categorical("group", vec!["NEW".to_string(), "g1".to_string()])
            .unwrap();

        let payload = model
            .predict_new_variance(
                &newdata,
                GlmmPredictionScale::Response,
                NewReLevels::Population,
            )
            .unwrap();
        let unseen = &payload.rows[0];
        assert_eq!(unseen.status, PredictionVarianceStatus::Unavailable);
        assert!(unseen.prediction.is_some());
        assert!(unseen.fixed_variance.is_some());
        assert_eq!(unseen.random_variance, None);
        assert_eq!(unseen.fixed_random_covariance, None);
        assert_eq!(unseen.combined_variance, None);
        assert_eq!(unseen.se_fit, None);
        assert!(unseen
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("new level 'NEW'"));

        let known = &payload.rows[1];
        assert_eq!(known.status, PredictionVarianceStatus::Degraded);
        assert!(known.se_fit.unwrap() > 0.0);
    }

    #[test]
    fn test_glmm_profile_likelihood_methods_refuse_with_explicit_reason() {
        let (mut model, _) = glmm_prediction_fixture();

        let sigma_err = model.profile_sigma(4.0).unwrap_err();
        assert_eq!(sigma_err.code(), "unsupported");
        let sigma_msg = sigma_err.to_string();
        assert!(sigma_msg.contains("profile_sigma"));
        assert!(sigma_msg.contains("GLMM profile likelihood is not implemented"));
        assert!(sigma_msg.contains("LMM-only"));

        let theta_err = model.profile_theta(0, 4.0).unwrap_err();
        assert_eq!(theta_err.code(), "unsupported");
        let theta_msg = theta_err.to_string();
        assert!(theta_msg.contains("profile_theta"));
        assert!(theta_msg.contains("GLMM profile likelihood is not implemented"));
        assert!(theta_msg.contains("LMM-only"));
    }

    #[test]
    fn test_glmm_rectify_after_fit() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age2 + urban + livch + (1 + age | urban_dist)").unwrap();
        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        let mut theta = vec![-0.5, 0.05, -0.25];

        model.finalize_theta_after_optimizer(&mut theta, 1).unwrap();

        assert_eq!(theta, vec![0.5, -0.05, 0.25]);
        assert_eq!(model.theta, theta);
        assert_eq!(model.lmm.optsum.final_params, theta);
        assert!(model.lmm.optsum.fmin.is_finite());
        assert_glmm_theta_diagonals_nonnegative(&model);
    }

    #[test]
    fn test_glmm_deviance_agq_restores_state() {
        // After a Laplace fit, snapshotting (u, eta, mu) and then calling
        // deviance(7) must leave those vectors bit-equivalent on return:
        // AGQ is supposed to perturb-and-restore.
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit_with_options(true, 1, false).unwrap(); // Laplace fit.

        let u_snap: Vec<DMatrix<f64>> = model.u.clone();
        let eta_snap = model.eta.clone();
        let mu_snap = model.mu.clone();

        let _agq = model.deviance(7);

        // u must be byte-identical: the AGQ sweep restores from u₀.
        assert_eq!(model.u.len(), u_snap.len());
        for (after, before) in model.u.iter().zip(u_snap.iter()) {
            assert_eq!(
                after.shape(),
                before.shape(),
                "u shape must not change across deviance(n_agq)"
            );
            for (a, b) in after.iter().zip(before.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "u entry diverged: before={b}, after={a}"
                );
            }
        }

        // eta and mu may pick up tiny fp differences from the final
        // update_eta() call, but should be within ~1e-12 absolute.
        for (a, b) in model.eta.iter().zip(eta_snap.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "eta drifted across AGQ sweep: before={b}, after={a}"
            );
        }
        for (a, b) in model.mu.iter().zip(mu_snap.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "mu drifted across AGQ sweep: before={b}, after={a}"
            );
        }

        // And a Laplace re-eval must match its pre-AGQ value.
        let lap_after = model.deviance(1);
        let lap_before = {
            // Recompute a fresh Laplace from the pre-AGQ snapshot for parity.
            let dev_resid: f64 = (0..model.y.len())
                .map(|i| model.dev_resid_component(model.y[i], mu_snap[i]))
                .sum();
            let u_pen: f64 = u_snap
                .iter()
                .map(|u| u.iter().map(|x| x * x).sum::<f64>())
                .sum();
            dev_resid + u_pen + model.lmm_logdet()
        };
        assert!(
            (lap_after - lap_before).abs() < 1e-9,
            "Laplace deviance drifted across AGQ sweep: before={lap_before}, after={lap_after}"
        );
    }

    #[cfg(feature = "nlopt")]
    #[test]
    fn test_glmm_nagq_sweep_converges_on_contra() {
        // At a fixed θ, the n-point AGQ deviance should approach a limit
        // as n_agq grows. We assert:
        //   * all values lie within a small band around the Julia reference
        //     (~2360.876, our Rust fit ~2360.98)
        //   * successive doublings of n_agq move by less than 0.05 (well
        //     below the 1.0 tolerance pattern used elsewhere)
        //   * n_agq=1 path equals laplace_objective() exactly.
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

        let mut model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        model.fit_with_options(true, 7, false).unwrap();

        let lap_direct = model.laplace_objective();
        let lap_via_dev = model.deviance(1);
        assert_eq!(
            lap_direct.to_bits(),
            lap_via_dev.to_bits(),
            "deviance(1) and laplace_objective() must agree bit-for-bit"
        );

        let dev3 = model.deviance(3);
        let dev5 = model.deviance(5);
        let dev7 = model.deviance(7);
        let dev9 = model.deviance(9);
        let dev15 = model.deviance(15);

        // Rough band: all AGQ evaluations should sit within ~2 deviance units
        // of the Julia reference 2360.876.
        for (label, val) in [
            ("nAGQ=3", dev3),
            ("nAGQ=5", dev5),
            ("nAGQ=7", dev7),
            ("nAGQ=9", dev9),
            ("nAGQ=15", dev15),
        ] {
            assert!(
                (val - 2360.876_f64).abs() < 2.0,
                "{label} deviance {val} too far from Julia ref 2360.876"
            );
        }

        // Convergence: successive refinements should change by < 0.05.
        for (a_label, a, b_label, b) in [
            ("nAGQ=3", dev3, "nAGQ=5", dev5),
            ("nAGQ=5", dev5, "nAGQ=7", dev7),
            ("nAGQ=7", dev7, "nAGQ=9", dev9),
            ("nAGQ=9", dev9, "nAGQ=15", dev15),
        ] {
            assert!(
                (a - b).abs() < 0.05,
                "AGQ refinement |{a_label} - {b_label}| = {} should be < 0.05",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn test_glmm_compiler_artifact_records_boundary_metadata() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();

        let model =
            GeneralizedLinearMixedModel::new(formula, &data, Family::Bernoulli, None).unwrap();
        let artifact = model.compiler_artifact();

        assert_eq!(
            artifact.model_boundary.model_kind,
            crate::compiler::ModelKind::GeneralizedLinearMixedModel
        );
        assert_eq!(artifact.model_boundary.response_distribution, "bernoulli");
        assert_eq!(artifact.model_boundary.link, "logit");
        assert!(matches!(
            artifact.model_boundary.objective_approximation,
            crate::compiler::ObjectiveApproximation::Laplace { .. }
        ));
        assert!(matches!(
            artifact.model_boundary.inference_availability,
            crate::compiler::InferenceAvailability::Unsupported { .. }
        ));
    }

    #[test]
    fn test_glmm_new_with_compiler_policy_applies_internal_policy() {
        let data = contra_fixture();
        let formula =
            parse_formula("use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)").unwrap();
        let mut policy = CompilerPolicy::as_specified();
        policy.thresholds.effective_rank_relative_tolerance = 0.125;

        let model = GeneralizedLinearMixedModel::new_with_compiler_policy(
            formula,
            &data,
            Family::Bernoulli,
            None,
            policy,
        )
        .unwrap();

        assert_eq!(
            model.compiler_policy().random_strategy,
            crate::compiler::RandomStrategy::AsSpecified
        );
        assert!(model
            .compiler_artifact()
            .reproducibility
            .thresholds
            .iter()
            .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "0.125"));
    }
}
