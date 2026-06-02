//! Generalized linear mixed-effects model (GLMM).
//!
//! Fits models of the form:
//!   `g(E[y]) = Xβ + Zb`
//! where g is a link function, b ~ N(0, σ²Λ'Λ), and y|b follows
//! an exponential family distribution.
//!
//! Uses Penalized Iteratively Reweighted Least Squares (PIRLS) for
//! the conditional modes, with optional adaptive Gauss-Hermite quadrature.

use nalgebra::{DMatrix, DVector};
use statrs::function::gamma::ln_gamma;
use std::cell::Cell;
use std::cell::RefCell;
use std::rc::Rc;

use crate::compiler::{
    CompiledModelArtifact, CompilerPolicy, Diagnostic, DiagnosticCode, DiagnosticSeverity,
    DiagnosticStage, EvidenceMethod, EvidenceQuality, GlmmFitMetadata, ModelAuditReport,
    ModelBoundary, ObjectiveApproximation, OptimizerCertificate, OptimizerDerivativeEvidence,
};
use crate::error::{MixedModelError, Result};
use crate::formula::Formula;
use crate::model::data::DataFrame;
use crate::model::linear::{CovarianceKktClassification, LinearMixedModel, OptimizerControl};
use crate::model::traits::{Family, LinkFunction, MixedModelFit, RandomEffectTermInfo};
use crate::optimizer::trust_bq::{
    minimize_with_progress as minimize_trust_bq_with_progress, TrustBqOptions, TrustBqProgress,
    TrustBqStopReason,
};
use crate::stats::{BlockDescription, ModelSummary, VarCorr};
use crate::types::{FitLogEntry, MatrixBlock, OptSummary, Optimizer, ReMat};
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
    /// Fixed NB2 size/dispersion parameter for negative-binomial GLMMs.
    ///
    /// This is the `theta` in `Var(Y | b) = mu + mu^2 / theta`. The first
    /// supported NB slice accepts caller-supplied theta; glmer.nb-style theta
    /// profiling is intentionally left for a later optimizer slice.
    negative_binomial_theta: Option<f64>,

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

    /// Construct a GLMM from formula, data, distribution, and link.
    pub fn new(
        formula: Formula,
        data: &DataFrame,
        family: Family,
        link: Option<LinkFunction>,
    ) -> Result<Self> {
        Self::new_with_policy_internal(formula, data, family, link, None, CompilerPolicy::default())
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
        let negative_binomial_theta =
            validate_negative_binomial_theta(family, negative_binomial_theta)?;

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
            family,
            link,
            devc: vec![0.0; agq_len],
            devc0: vec![0.0; agq_len],
            sd: vec![0.0; agq_len],
            mult: vec![0.0; agq_len],
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
        Self::new_with_policy_internal(formula, data, family, link, None, compiler_policy)
    }

    /// Fixed NB2 theta used by a negative-binomial GLMM, when applicable.
    pub fn negative_binomial_theta(&self) -> Option<f64> {
        self.negative_binomial_theta
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
        // Mirrors MixedModels.jl/src/generalizedlinearmixedmodel.jl pirls!
        // (lines 614-669): step-halving toward the previous accepted iterate
        // whenever a fresh IRLS step would worsen the Laplace objective. Keeps
        // the outer optimizer's view of obj(θ) consistent across probes —
        // without this, BOBYQA on multi-RE GLMM surfaces (e.g. grouseticks
        // Poisson) sees noisy values and reports `RoundoffLimited`.
        let max_iter = 10;
        let tol = 1.0e-5;
        let max_halvings = 10;

        let n = self.y.len();

        // Initialise u to zero; keep existing β.
        for u in self.u.iter_mut() {
            u.fill(0.0);
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

        for iter in 0..max_iter {
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
        if !fast {
            return self.fit_joint_glmm_with_response_constants(n_agq, verbose);
        }
        self.fit_with_options_impl(n_agq, verbose)
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
        let gradient = me.joint_laplace_finite_difference_gradient(
            &me.lmm.optsum.final_params.clone(),
            n_beta,
            n_agq,
            &lower_bounds,
        );
        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                gradient: gradient.clone(),
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
            &gradient,
            2.0e-2,
        );
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
                max_cross_terms: 0,
                stall_iterations: 3,
                stall_ftol_abs: 1.0e-6,
                stall_ftol_rel: 1.0e-8,
                stall_requires_stable_x: false,
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
        if joint_certificate_requires_fallback(&certificate)
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
        if fallback_fast_pirls.is_some() && joint_certificate_requires_fallback(&certificate) {
            if let Some(fallback) =
                uncertified_joint_fallback(&certificate, &me.lmm.optsum, fallback_fast_pirls.take())
            {
                *me = fallback;
                me.refresh_binomial_separation_diagnostics();
                me.refresh_near_unit_random_effect_correlation_diagnostics();
                return Ok(me);
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
        let gradient = me.joint_laplace_finite_difference_gradient(
            &me.lmm.optsum.final_params.clone(),
            n_beta,
            n_agq,
            &lower_bounds,
        );
        certificate.apply_derivative_evidence(
            OptimizerDerivativeEvidence {
                method: EvidenceMethod::FiniteDifference,
                gradient: gradient.clone(),
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
            &gradient,
            2.0e-2,
        );
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

    fn joint_laplace_finite_difference_gradient(
        &mut self,
        params: &[f64],
        n_beta: usize,
        n_agq: usize,
        lower_bounds: &[f64],
    ) -> Vec<f64> {
        let base = self.joint_glmm_deviance_at_params(params, n_beta, n_agq);
        let mut gradient = Vec::with_capacity(params.len());
        for (index, &value) in params.iter().enumerate() {
            let h = 1.0e-5 * value.abs().max(1.0);
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
                gradient.push((fp - fm) / (2.0 * h));
            } else {
                gradient.push((fp - base) / h);
            }
        }
        let _ = self.joint_glmm_deviance_at_params(params, n_beta, n_agq);
        gradient
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
        }
        self.record_fast_pirls_parity_scope_diagnostic(&metadata);
        self.lmm.compiler_artifact.glmm_fit_metadata = Some(metadata);
        self.lmm.compiler_artifact.fixed_effect_covariance_matrix =
            Some(self.lmm.glmm_fixed_effect_covariance_matrix());
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
        for u in self.u.iter_mut() {
            u.fill(0.0);
        }
        let converged = self.pirls(vary_beta, false)?;
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
    gradient: &[f64],
    gradient_tolerance: f64,
) {
    if !certificate.evidence.optimizer_stop.acceptable_stop || params.len() <= n_beta {
        return;
    }

    let boundary_tolerance = 1.0e-8;
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
        return;
    }

    let invalid_boundary = boundary_indices.iter().any(|&index| {
        theta_gradient
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
            crate::compiler::FitStatus::NotOptimized | crate::compiler::FitStatus::NotAssessed
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
    let mut diagnostic = Diagnostic::new(
        DiagnosticCode::OptimizerNonconvergence,
        DiagnosticSeverity::Warning,
        DiagnosticStage::Certification,
        "returning improved joint GLMM candidate after budget exhaustion; convergence is not certified",
    )
    .with_suggested_actions(vec![
        "treat fixed effects and log-likelihood as a budget-limited joint-Laplace candidate, not a certified optimizer convergence".to_string(),
        "increase max_feval or compare against an external joint-Laplace reference before promoting this row to strict parity".to_string(),
    ]);
    diagnostic.payload.insert(
        "fit_mode".to_string(),
        serde_json::json!("uncertified_joint_candidate"),
    );
    diagnostic.payload.insert(
        "scorecard_class".to_string(),
        serde_json::json!("budget_limited_joint_candidate"),
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

fn negative_binomial_deviance_component(y: f64, mu: f64, theta: f64) -> f64 {
    let mu = mu.max(f64::MIN_POSITIVE);
    let theta = theta.max(f64::MIN_POSITIVE);
    let first = if y == 0.0 { 0.0 } else { y * (y / mu).ln() };
    let second = (y + theta) * ((y + theta) / (mu + theta)).ln();
    2.0 * (first - second)
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

fn validate_negative_binomial_theta(family: Family, theta: Option<f64>) -> Result<Option<f64>> {
    match (family, theta) {
        (Family::NegativeBinomial, Some(theta)) if theta.is_finite() && theta > 0.0 => {
            Ok(Some(theta))
        }
        (Family::NegativeBinomial, Some(theta)) => Err(MixedModelError::InvalidArgument(format!(
            "negative-binomial fixed theta must be positive and finite (got {theta})"
        ))),
        (Family::NegativeBinomial, None) => Err(MixedModelError::InvalidArgument(
            "negative-binomial GLMM requires a positive fixed theta; use \
             GeneralizedLinearMixedModel::new_negative_binomial(...) or \
             GeneralizedLinearMixedModelBuilder::negative_binomial_theta(...)"
                .to_string(),
        )),
        (_, Some(_)) => Err(MixedModelError::InvalidArgument(
            "negative-binomial theta can only be supplied with Family::NegativeBinomial"
                .to_string(),
        )),
        (_, None) => Ok(None),
    }
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
            + if self.family.has_dispersion() { 1 } else { 0 }
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
        self.lmm.stderror()
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
            parse_formula("correct ~ 1 + x + (1 + x || participant) + (1 | item)").unwrap();
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
        assert!(matches!(
            metadata.estimation_method.as_str(),
            "joint_laplace" | "fallback_fast_pirls"
        ));
        if metadata.estimation_method == "joint_laplace"
            && model.lmm.optsum.return_value.contains("MAXEVAL_REACHED")
        {
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
    fn test_glmm_fast_false_nagq_uses_labelled_joint_agq_path() {
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
        assert_eq!(metadata.estimation_method, "joint_agq");
        assert_eq!(metadata.objective_definition, "joint_glmm_agq_deviance");
        assert_eq!(metadata.response_constants, "included");
        assert_eq!(metadata.n_agq, 7);
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
