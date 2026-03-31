//! Optimization summary for mixed-effects model fitting.
//!
//! This is a port of `OptSummary` from Julia's MixedModels.jl.
//! It tracks the optimizer state, tolerances, fit log, and
//! model-fitting options (REML, adaptive Gauss-Hermite quadrature,
//! known σ).

/// Choice of optimizer algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    /// COBYLA — Constrained Optimization By Linear Approximations.
    /// A derivative-free optimizer that handles bound constraints
    /// via inequality constraints.
    Cobyla,
}

/// One entry in the fit log, recording the parameter vector and the
/// objective value at a particular evaluation.
#[derive(Debug, Clone)]
pub struct FitLogEntry {
    /// Parameter vector (θ) at this evaluation.
    pub theta: Vec<f64>,
    /// Objective function value (deviance or REML criterion).
    pub objective: f64,
}

/// Summary of the optimization used to fit a mixed-effects model.
///
/// Stores initial and final parameter values, convergence information,
/// tolerances, and a log of all function evaluations. The defaults
/// match those in Julia's MixedModels.jl.
#[derive(Debug, Clone)]
pub struct OptSummary {
    // ---- Parameter values ----
    /// Initial parameter vector (θ₀).
    pub initial: Vec<f64>,

    /// Objective value at the initial parameters.
    pub finitial: f64,

    /// Final (optimised) parameter vector.
    pub final_params: Vec<f64>,

    /// Minimum objective value found.
    pub fmin: f64,

    /// Number of function evaluations. A value of −1 means the model
    /// has not yet been fitted.
    pub feval: i64,

    /// Return status string from the optimizer (e.g. `"FTOL_REACHED"`).
    pub return_value: String,

    // ---- Tolerances ----
    /// Absolute tolerance on θ components for declaring convergence to
    /// zero. Default `1e-12`.
    pub xtol_zero_abs: f64,

    /// Absolute tolerance on the objective for declaring convergence
    /// to zero. Default `1e-12`.
    pub ftol_zero_abs: f64,

    /// Relative tolerance on the objective. Default `1e-8`.
    pub ftol_rel: f64,

    /// Absolute tolerance on the objective. Default `1e-12`.
    pub ftol_abs: f64,

    /// Relative tolerance on θ. Default `0.0` (inactive by default).
    pub xtol_rel: f64,

    /// Per-component absolute tolerance on θ.
    pub xtol_abs: Vec<f64>,

    /// Initial step sizes for derivative-free optimizers.
    pub initial_step: Vec<f64>,

    /// Maximum number of function evaluations. Default `-1` (unlimited).
    pub max_feval: i64,

    /// Maximum wall-clock time (seconds). Default `-1.0` (unlimited).
    pub max_time: f64,

    // ---- Optimizer ----
    /// Which optimizer to use.
    pub optimizer: Optimizer,

    // ---- Model-fitting options ----
    /// Whether to use REML (restricted maximum likelihood) rather
    /// than ML.
    pub reml: bool,

    /// Number of adaptive Gauss-Hermite quadrature points for GLMMs.
    /// 1 means Laplace approximation.
    pub n_agq: usize,

    /// Known residual standard deviation. `None` means σ is estimated
    /// from the data (the usual case). Setting `Some(σ)` fixes it,
    /// as in `MixedModel(..., σ = σ)` in Julia.
    pub sigma: Option<f64>,

    // ---- Fit log ----
    /// Log of `(θ, objective)` at every function evaluation.
    pub fit_log: Vec<FitLogEntry>,
}

impl OptSummary {
    /// Create a new `OptSummary` with default tolerances matching
    /// Julia's MixedModels.jl.
    ///
    /// The `feval` field is set to −1 to indicate that the model has
    /// not yet been fitted.
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial parameter vector (θ₀).
    pub fn new(initial: Vec<f64>) -> Self {
        let n = initial.len();
        OptSummary {
            finitial: f64::INFINITY,
            final_params: initial.clone(),
            fmin: f64::INFINITY,
            feval: -1,
            return_value: String::new(),

            // Tolerances (matching Julia defaults)
            xtol_zero_abs: 1e-12,
            ftol_zero_abs: 1e-12,
            ftol_rel: 1e-8,
            ftol_abs: 1e-12,
            xtol_rel: 0.0,
            xtol_abs: vec![1e-10; n],
            initial_step: vec![0.75; n],
            max_feval: -1,
            max_time: -1.0,

            optimizer: Optimizer::Cobyla,

            reml: true,
            n_agq: 1,
            sigma: None,

            fit_log: Vec::new(),

            initial,
        }
    }

    /// Whether the model has been fitted (at least one function
    /// evaluation has been performed).
    pub fn is_fitted(&self) -> bool {
        self.feval > 0
    }

    /// Record a function evaluation in the fit log.
    ///
    /// # Arguments
    ///
    /// * `theta` - Parameter vector at this evaluation.
    /// * `objective` - Objective value at this evaluation.
    pub fn log_eval(&mut self, theta: Vec<f64>, objective: f64) {
        self.fit_log.push(FitLogEntry { theta, objective });
    }

    /// Number of parameters (length of the θ vector).
    pub fn n_params(&self) -> usize {
        self.initial.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_defaults() {
        let opt = OptSummary::new(vec![1.0, 0.5]);
        assert_eq!(opt.initial, vec![1.0, 0.5]);
        assert_eq!(opt.feval, -1);
        assert!(!opt.is_fitted());
        assert_eq!(opt.optimizer, Optimizer::Cobyla);
        assert!(opt.reml);
        assert_eq!(opt.n_agq, 1);
        assert!(opt.sigma.is_none());
        assert_eq!(opt.xtol_abs.len(), 2);
        assert_eq!(opt.initial_step.len(), 2);
        assert!(opt.fmin.is_infinite());
    }

    #[test]
    fn test_is_fitted() {
        let mut opt = OptSummary::new(vec![1.0]);
        assert!(!opt.is_fitted());
        opt.feval = 1;
        assert!(opt.is_fitted());
    }

    #[test]
    fn test_log_eval() {
        let mut opt = OptSummary::new(vec![1.0, 0.5]);
        opt.log_eval(vec![1.0, 0.5], 100.0);
        opt.log_eval(vec![0.8, 0.4], 95.0);
        assert_eq!(opt.fit_log.len(), 2);
        assert!((opt.fit_log[1].objective - 95.0).abs() < 1e-12);
    }

    #[test]
    fn test_n_params() {
        let opt = OptSummary::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(opt.n_params(), 3);
    }

    #[test]
    fn test_empty_initial() {
        let opt = OptSummary::new(Vec::new());
        assert_eq!(opt.n_params(), 0);
        assert!(!opt.is_fitted());
    }
}
