//! Optimization summary for mixed-effects model fitting.
//!
//! This is a port of `OptSummary` from Julia's MixedModels.jl.
//! It tracks the optimizer state, tolerances, fit log, and
//! model-fitting options (REML, adaptive Gauss-Hermite quadrature,
//! known σ).

use std::fmt;

/// Choice of optimizer algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Optimizer {
    /// COBYLA — Constrained Optimization By Linear Approximations.
    /// A derivative-free optimizer that handles bound constraints
    /// via inequality constraints.
    Cobyla,
    /// Bound-aware pattern search used for small multivariate θ vectors.
    PatternSearch,
    /// In-tree bounded quadratic trust-region optimizer.
    TrustBq,
    /// NLopt NEWUOA for unconstrained larger θ vectors.
    NloptNewuoa,
    /// NLopt BOBYQA for bound-constrained θ vectors.
    NloptBobyqa,
    /// PRIMA BOBYQA — Powell's bound-constrained derivative-free optimizer.
    PrimaBobyqa,
    /// PRIMA COBYLA — Powell's general-constraints derivative-free optimizer.
    PrimaCobyla,
    /// PRIMA LINCOA — Powell's linearly-constrained derivative-free optimizer.
    PrimaLincoa,
    /// PRIMA NEWUOA — Powell's unconstrained derivative-free optimizer.
    PrimaNewuoa,
}

/// Optimization backend providing the optimizer.
///
/// Mirrors `OptSummary.backend` from MixedModels.jl. The default backend is
/// `Native` (in-tree TrustBQ/pattern-search plus the COBYLA crate).
/// `Nlopt` is the upstream default in the Julia reference and is the active
/// backend for any `Optimizer::Nlopt*` variant. `Prima` is reserved for the
/// PRIMA derivative-free family; `Optimizer::PrimaBobyqa` is wired for LMMs
/// when the non-default `prima` Cargo feature is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OptimizerBackend {
    /// In-tree Rust optimizers and native fallback crates.
    Native,
    /// NLopt-backed optimizers (BOBYQA, NEWUOA).
    Nlopt,
    /// PRIMA-backed optimizers (bobyqa/cobyla/lincoa/newuoa).
    Prima,
}

/// Source of the optimizer algorithm recorded in an [`OptSummary`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OptimizerSource {
    /// The fit driver selected the optimizer automatically.
    Auto,
    /// A caller explicitly requested the optimizer.
    Caller,
}

impl Optimizer {
    /// Canonical backend for this optimizer.
    pub fn canonical_backend(self) -> OptimizerBackend {
        match self {
            Optimizer::Cobyla | Optimizer::PatternSearch | Optimizer::TrustBq => {
                OptimizerBackend::Native
            }
            Optimizer::NloptNewuoa | Optimizer::NloptBobyqa => OptimizerBackend::Nlopt,
            Optimizer::PrimaBobyqa
            | Optimizer::PrimaCobyla
            | Optimizer::PrimaLincoa
            | Optimizer::PrimaNewuoa => OptimizerBackend::Prima,
        }
    }
}

impl OptimizerBackend {
    /// Human-readable label, lowercase to match Julia's `:nlopt` / `:prima`.
    pub fn label(self) -> &'static str {
        match self {
            OptimizerBackend::Native => "native",
            OptimizerBackend::Nlopt => "nlopt",
            OptimizerBackend::Prima => "prima",
        }
    }

    /// The list of `OptSummary` fields used by this backend, in the order
    /// they should appear in MIME renderings. Mirrors Julia's
    /// `opt_params(::Val{:backend})`.
    pub fn opt_params(self) -> &'static [&'static str] {
        match self {
            OptimizerBackend::Native | OptimizerBackend::Nlopt => &[
                "ftol_rel",
                "ftol_abs",
                "xtol_rel",
                "xtol_abs",
                "initial_step",
                "maxfeval",
                "maxtime",
            ],
            OptimizerBackend::Prima => &["rhobeg", "rhoend", "maxfeval"],
        }
    }
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

/// Typed classification of an optimizer's termination status.
///
/// The optimizer's raw outcome is stored as a free-form
/// [`OptSummary::return_value`] string (`"FTOL_REACHED"`,
/// `"MAXEVAL_REACHED"`, …). Forcing every caller to string-match that to
/// learn whether the fit actually converged is exactly the brittle
/// anti-pattern the crate warns against, and it makes it easy to ship a
/// budget-truncated (non-optimal) fit as if it were good. This enum is the
/// single typed contract; prefer [`OptSummary::converged`] /
/// [`OptSummary::convergence_status`] over inspecting the string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConvergenceStatus {
    /// Stopped at a genuine convergence criterion (objective/parameter
    /// tolerance, trust radius, or target value reached). The returned
    /// parameters are a verified local optimum to the requested tolerance.
    Converged,
    /// An evaluation/time/iteration budget was hit before a convergence
    /// criterion. The returned parameters are the best seen so far but are
    /// **not** a verified optimum — treating this as success is the
    /// "non-convergence masquerading as a fit" hazard.
    BudgetExhausted,
    /// The optimizer halted because progress was limited by floating-point
    /// roundoff/stagnation. Often near-optimal, but not a clean convergence;
    /// callers should treat the fit as provisional.
    RoundoffLimited,
    /// The optimizer failed outright (invalid input, non-finite objective,
    /// forced stop, out of memory, …). The fit is not usable.
    Failed,
    /// No optimizer status is recorded yet (model not fitted).
    NotRun,
}

/// Summary of the optimization used to fit a mixed-effects model.
///
/// Stores initial and final parameter values, convergence information,
/// tolerances, and a log of all function evaluations. The defaults
/// match those in Julia's MixedModels.jl.
#[derive(Debug, Clone)]
#[non_exhaustive]
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

    /// Final trust-region radius for optimizers that expose one.
    ///
    /// `None` for optimizers that do not use or report a trust radius.
    pub final_trust_radius: Option<f64>,

    // ---- Optimizer ----
    /// Which optimizer to use.
    pub optimizer: Optimizer,

    /// Optimization backend providing the optimizer. Defaults to
    /// `OptimizerBackend::Native`; PRIMA dispatch requires the non-default
    /// `prima` Cargo feature and a system `libprimac`.
    pub backend: OptimizerBackend,

    /// Whether the recorded optimizer was chosen automatically or requested
    /// by the caller.
    pub optimizer_source: OptimizerSource,

    /// Stable labels for fit-control fields deliberately set by the caller.
    ///
    /// Empty means the fit used driver defaults. This is intentionally a
    /// compact audit vector rather than a second copy of the numeric fields,
    /// which are already present in this summary.
    pub caller_set_fields: Vec<String>,

    // ---- PRIMA-specific tolerances ----
    /// PRIMA initial trust-region radius. Default `1.0`, matching
    /// MixedModels.jl. Ignored by NLopt and the in-tree backend.
    pub rhobeg: f64,

    /// PRIMA final trust-region radius. Default `1e-6`, matching
    /// MixedModels.jl. Ignored by NLopt and the in-tree backend.
    pub rhoend: f64,

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
            final_trust_radius: None,

            optimizer: Optimizer::Cobyla,
            backend: OptimizerBackend::Native,
            optimizer_source: OptimizerSource::Auto,
            caller_set_fields: Vec::new(),
            // PRIMA defaults from MixedModels.jl: rhobeg = 1.0, rhoend = rhobeg / 1e6.
            rhobeg: 1.0,
            rhoend: 1e-6,

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

    /// Typed classification of the optimizer's termination status.
    ///
    /// Interprets [`return_value`](Self::return_value) (the union of the
    /// status vocabularies of every backend: NLopt, COBYLA, the bounded
    /// trust-region solver, and PRIMA), so callers never have to string-match
    /// it. The KKT boundary-restart wrapper (`"KKT_BOUNDARY_RESTART(n):
    /// <inner>"`) is unwrapped and classified by its inner status.
    pub fn convergence_status(&self) -> ConvergenceStatus {
        if !self.is_fitted() {
            return ConvergenceStatus::NotRun;
        }
        let raw = self.return_value.trim();
        if raw.is_empty() {
            return ConvergenceStatus::NotRun;
        }
        let status = optimizer_final_status_code(raw);
        match status {
            // Clean convergence criteria across all backends.
            "SUCCESS" | "STOPVAL_REACHED" | "FTOL_REACHED" | "XTOL_REACHED" | "RADIUS_REACHED"
            | "SMALL_TR_RADIUS" | "FTARGET_ACHIEVED" => ConvergenceStatus::Converged,
            // Budget/iteration limits — best-effort, NOT a verified optimum.
            "MAXEVAL_REACHED" | "MAXTIME_REACHED" | "MAXFUN_REACHED" | "MAXTR_REACHED"
            | "CALLBACK_TERMINATE" => ConvergenceStatus::BudgetExhausted,
            // Roundoff/stagnation: provisional, not a clean convergence.
            "ROUNDOFF_LIMITED" | "DAMAGING_ROUNDING" => ConvergenceStatus::RoundoffLimited,
            // Everything else (FAILURE, INVALID_ARGS, OUT_OF_MEMORY,
            // FORCED_STOP, UNEXPECTED_ERROR, FORCED_BAD_BOUNDARY,
            // TRSUBP_FAILED, NAN_INF_*, NO_SPACE_BETWEEN_BOUNDS,
            // ZERO_LINEAR_CONSTRAINT, INVALID_INPUT, …) is a hard failure.
            _ => ConvergenceStatus::Failed,
        }
    }

    /// Whether the optimizer reached a genuine convergence criterion.
    ///
    /// `true` only for [`ConvergenceStatus::Converged`]: a budget-exhausted,
    /// roundoff-limited, failed, or not-yet-run fit all return `false`. This
    /// is the honest gate — a non-converged fit must never report `true`.
    pub fn converged(&self) -> bool {
        matches!(self.convergence_status(), ConvergenceStatus::Converged)
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

    /// The optimization backend label used for display.
    pub fn backend_name(&self) -> &'static str {
        match self.optimizer.canonical_backend() {
            OptimizerBackend::Native => self.backend.label(),
            backend => backend.label(),
        }
    }

    /// Label for the optimizer-selection source.
    pub fn optimizer_source_name(&self) -> &'static str {
        match self.optimizer_source {
            OptimizerSource::Auto => "auto",
            OptimizerSource::Caller => "caller",
        }
    }

    /// Whether a named fit-control field was supplied by the caller.
    pub fn caller_set_field(&self, field: &str) -> bool {
        self.caller_set_fields
            .iter()
            .any(|candidate| candidate == field)
    }

    /// Caller-requested optimizer, if the optimizer was explicitly pinned.
    pub fn caller_selected_optimizer(&self) -> Option<Optimizer> {
        match self.optimizer_source {
            OptimizerSource::Caller => Some(self.optimizer),
            OptimizerSource::Auto => None,
        }
    }

    /// The optimizer label used for display.
    pub fn optimizer_name(&self) -> &'static str {
        match self.optimizer {
            Optimizer::Cobyla => "cobyla",
            Optimizer::PatternSearch => "pattern_search",
            Optimizer::TrustBq => "trust_bq",
            Optimizer::NloptNewuoa => "newuoa",
            Optimizer::NloptBobyqa => "bobyqa",
            Optimizer::PrimaBobyqa => "bobyqa",
            Optimizer::PrimaCobyla => "cobyla",
            Optimizer::PrimaLincoa => "lincoa",
            Optimizer::PrimaNewuoa => "newuoa",
        }
    }

    /// The Julia-style optimizer code used in MIME renderers. NLopt-backed
    /// optimizers use the `LN_*` codes from NLopt; PRIMA-backed optimizers
    /// use the lowercase Julia symbol (`bobyqa`, `cobyla`, ...).
    pub fn optimizer_code(&self) -> &'static str {
        match self.optimizer {
            Optimizer::Cobyla => "LN_COBYLA",
            Optimizer::PatternSearch => "PATTERN_SEARCH",
            Optimizer::TrustBq => "TRUST_BQ",
            Optimizer::NloptNewuoa => "LN_NEWUOA",
            Optimizer::NloptBobyqa => "LN_BOBYQA",
            Optimizer::PrimaBobyqa => "bobyqa",
            Optimizer::PrimaCobyla => "cobyla",
            Optimizer::PrimaLincoa => "lincoa",
            Optimizer::PrimaNewuoa => "newuoa",
        }
    }

    fn optimizer_setting_pairs(&self) -> Vec<(&'static str, String)> {
        self.optimizer
            .canonical_backend()
            .opt_params()
            .iter()
            .map(|name| {
                let value = match *name {
                    "ftol_rel" => self.ftol_rel.to_string(),
                    "ftol_abs" => self.ftol_abs.to_string(),
                    "xtol_rel" => self.xtol_rel.to_string(),
                    "xtol_abs" => format!("{:?}", self.xtol_abs),
                    "initial_step" => format!("{:?}", self.initial_step),
                    "maxfeval" => self.max_feval.to_string(),
                    "maxtime" => self.max_time.to_string(),
                    "rhobeg" => self.rhobeg.to_string(),
                    "rhoend" => self.rhoend.to_string(),
                    other => format!("<unknown opt param {other}>"),
                };
                (*name, value)
            })
            .collect()
    }

    fn markdown_optimizer_settings(&self) -> String {
        self.optimizer_setting_pairs()
            .into_iter()
            .map(|(name, value)| format!("| {name:<24} | {value} |\n"))
            .collect()
    }

    fn html_optimizer_settings(&self) -> String {
        self.optimizer_setting_pairs()
            .into_iter()
            .map(|(name, value)| {
                format!("<tr><td align=\"left\">{name}</td><td align=\"left\">{value}</td></tr>")
            })
            .collect()
    }

    fn latex_optimizer_settings(&self) -> String {
        self.optimizer_setting_pairs()
            .into_iter()
            .map(|(name, value)| format!("{} & {} \\\\\n", latex_escape_code(name), value))
            .collect()
    }

    /// Render a markdown summary table.
    pub fn to_markdown(&self) -> String {
        format!(
            concat!(
                "|                          |                   |\n",
                "|:------------------------ |:----------------- |\n",
                "| **Initialization**       |                   |\n",
                "| Initial parameter vector | {} |\n",
                "| Initial objective value  | {} |\n",
                "| **Optimizer settings**   |                   |\n",
                "| Optimizer                | `{}` |\n",
                "| Backend                  | `{}` |\n",
                "{}",
                "| xtol_zero_abs            | {} |\n",
                "| ftol_zero_abs            | {} |\n",
                "| **Result**               |                   |\n",
                "| Function evaluations     | {} |\n",
                "| Final parameter vector   | {} |\n",
                "| Final objective value    | {} |\n",
                "| Return code              | `{}` |\n"
            ),
            format!("{:?}", self.initial),
            self.finitial,
            self.optimizer_code(),
            self.backend_name(),
            self.markdown_optimizer_settings(),
            self.xtol_zero_abs,
            self.ftol_zero_abs,
            self.feval,
            format!("{:?}", self.final_params),
            self.fmin,
            self.return_value
        )
    }

    /// Render an HTML summary table.
    pub fn to_html(&self) -> String {
        format!(
            concat!(
                "<table>",
                "<tr><td align=\"left\"><b>Initialization</b></td><td align=\"left\"></td></tr>",
                "<tr><td align=\"left\">Initial parameter vector</td><td align=\"left\">{:?}</td></tr>",
                "<tr><td align=\"left\">Initial objective value</td><td align=\"left\">{}</td></tr>",
                "<tr><td align=\"left\"><b>Optimizer settings</b></td><td align=\"left\"></td></tr>",
                "<tr><td align=\"left\">Optimizer</td><td align=\"left\"><code>{}</code></td></tr>",
                "<tr><td align=\"left\">Backend</td><td align=\"left\"><code>{}</code></td></tr>",
                "{}",
                "<tr><td align=\"left\">xtol_zero_abs</td><td align=\"left\">{}</td></tr>",
                "<tr><td align=\"left\">ftol_zero_abs</td><td align=\"left\">{}</td></tr>",
                "<tr><td align=\"left\"><b>Result</b></td><td align=\"left\"></td></tr>",
                "<tr><td align=\"left\">Function evaluations</td><td align=\"left\">{}</td></tr>",
                "<tr><td align=\"left\">Final parameter vector</td><td align=\"left\">{:?}</td></tr>",
                "<tr><td align=\"left\">Final objective value</td><td align=\"left\">{}</td></tr>",
                "<tr><td align=\"left\">Return code</td><td align=\"left\"><code>{}</code></td></tr>",
                "</table>\n"
            ),
            self.initial,
            self.finitial,
            self.optimizer_code(),
            self.backend_name(),
            self.html_optimizer_settings(),
            self.xtol_zero_abs,
            self.ftol_zero_abs,
            self.feval,
            self.final_params,
            self.fmin,
            self.return_value
        )
    }

    /// Render a LaTeX summary table.
    pub fn to_latex(&self) -> String {
        format!(
            concat!(
                "\\begin{{tabular}}\n",
                "{{l | l}}\n",
                "\\textbf{{Initialization}} &  \\\\\n",
                "Initial parameter vector & {:?} \\\\\n",
                "Initial objective value & {} \\\\\n",
                "\\textbf{{Optimizer settings}} &  \\\\\n",
                "Optimizer & \\texttt{{{}}} \\\\\n",
                "Backend & \\texttt{{{}}} \\\\\n",
                "{}",
                "xtol_zero_abs & {} \\\\\n",
                "ftol_zero_abs & {} \\\\\n",
                "\\textbf{{Result}} &  \\\\\n",
                "Function evaluations & {} \\\\\n",
                "Final parameter vector & {:?} \\\\\n",
                "Final objective value & {} \\\\\n",
                "Return code & \\texttt{{{}}} \\\\\n",
                "\\end{{tabular}}\n"
            ),
            self.initial,
            self.finitial,
            latex_escape_code(self.optimizer_code()),
            latex_escape_code(self.backend_name()),
            self.latex_optimizer_settings(),
            self.xtol_zero_abs,
            self.ftol_zero_abs,
            self.feval,
            self.final_params,
            self.fmin,
            latex_escape_code(&self.return_value)
        )
    }
}

/// Extract the stop code that describes the estimates actually installed in
/// the model. Post-fit wrappers describe recovery/provenance; direct joint
/// prefixes describe the joint optimizer; labelled fallback wrappers carry
/// both the failed joint code and the returned fast-PIRLS code. Classification
/// must use the latter so `OptSummary::converged()` and the optimizer
/// certificate describe the same returned fit.
pub(crate) fn optimizer_final_status_code(mut status: &str) -> &str {
    loop {
        let stripped = ["KKT_BOUNDARY_RESTART", "START_LADDER", "ACTIVE_FACE"]
            .iter()
            .find_map(|prefix| {
                status
                    .strip_prefix(prefix)
                    .and_then(|rest| rest.split_once(": "))
                    .map(|(_, inner)| inner.trim())
            });
        match stripped {
            Some(inner) => status = inner,
            None => break,
        }
    }

    let is_fallback = status.starts_with("JOINT_LAPLACE_FALLBACK_FAST_PIRLS(")
        || status.starts_with("JOINT_AGQ_FALLBACK_FAST_PIRLS(")
        || status.starts_with("EXPERIMENTAL_JOINT_FALLBACK_FAST_PIRLS(");
    if is_fallback {
        if let Some((_, fast_code)) = status.rsplit_once("; fast=") {
            return fast_code.strip_suffix(')').unwrap_or(fast_code).trim();
        }
    }

    status
        .strip_prefix("JOINT_LAPLACE:")
        .or_else(|| status.strip_prefix("JOINT_AGQ:"))
        .or_else(|| status.strip_prefix("EXPERIMENTAL_JOINT:"))
        .or_else(|| status.strip_prefix("JOINT_LAPLACE_FAILED:"))
        .or_else(|| status.strip_prefix("JOINT_AGQ_FAILED:"))
        .or_else(|| status.strip_prefix("EXPERIMENTAL_JOINT_FAILED:"))
        .unwrap_or(status)
}

impl fmt::Display for OptSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Initial parameter vector: {:?}", self.initial)?;
        writeln!(f, "Initial objective value:  {}", self.finitial)?;
        writeln!(f)?;
        writeln!(f, "Backend:                  {}", self.backend_name())?;
        writeln!(f, "Optimizer:                {}", self.optimizer_name())?;
        for (name, value) in self.optimizer_setting_pairs() {
            writeln!(f, "{:<26} {}", format!("{name}:"), value)?;
        }
        writeln!(f)?;
        writeln!(f, "Function evaluations:     {}", self.feval)?;
        writeln!(f, "xtol_zero_abs:            {}", self.xtol_zero_abs)?;
        writeln!(f, "ftol_zero_abs:            {}", self.ftol_zero_abs)?;
        writeln!(f, "Final parameter vector:   {:?}", self.final_params)?;
        writeln!(f, "Final objective value:    {}", self.fmin)?;
        writeln!(f, "Return code:              {}", self.return_value)
    }
}

fn latex_escape_code(code: &str) -> String {
    code.replace('_', "\\_")
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

    #[test]
    fn test_backend_name() {
        let mut opt = OptSummary::new(vec![1.0]);
        assert_eq!(opt.backend_name(), "native");
        opt.optimizer = Optimizer::NloptNewuoa;
        assert_eq!(opt.backend_name(), "nlopt");
        opt.optimizer = Optimizer::PrimaBobyqa;
        assert_eq!(opt.backend_name(), "prima");
    }

    #[test]
    fn test_display_contains_core_fields() {
        let mut opt = OptSummary::new(vec![1.0]);
        opt.finitial = 2595.85;
        opt.optimizer = Optimizer::NloptBobyqa;
        opt.final_params = vec![0.2612];
        opt.fmin = 2486.42;
        opt.feval = 17;
        opt.return_value = "MAXEVAL_REACHED".to_string();

        let out = format!("{opt}");
        assert!(out.contains("Initial parameter vector: [1.0]"));
        assert!(out.contains("Backend:                  nlopt"));
        assert!(out.contains("Optimizer:                bobyqa"));
        assert!(out.contains("Function evaluations:     17"));
        assert!(out.contains("Final objective value:    2486.42"));
        assert!(out.contains("Return code:              MAXEVAL_REACHED"));
    }

    #[test]
    fn test_markdown_contains_core_rows() {
        let mut opt = OptSummary::new(vec![1.0]);
        opt.finitial = 2595.85;
        opt.optimizer = Optimizer::NloptBobyqa;
        opt.initial_step = vec![0.75];
        opt.final_params = vec![0.2612];
        opt.fmin = 2486.42;
        opt.feval = 17;
        opt.return_value = "MAXEVAL_REACHED".to_string();

        let out = opt.to_markdown();
        assert!(out.contains("| Initial parameter vector | [1.0] |"));
        assert!(out.contains("| Optimizer                | `LN_BOBYQA` |"));
        assert!(out.contains("| Backend                  | `nlopt` |"));
        assert!(out.contains("| initial_step             | [0.75] |"));
        assert!(out.contains("| xtol_zero_abs            | 0.000000000001 |"));
        assert!(out.contains("| ftol_zero_abs            | 0.000000000001 |"));
        assert!(out.contains("| Function evaluations     | 17 |"));
        assert!(out.contains("| Return code              | `MAXEVAL_REACHED` |"));
    }

    #[test]
    fn test_prima_renderers_use_backend_specific_opt_params() {
        let mut opt = OptSummary::new(vec![1.0]);
        opt.optimizer = Optimizer::PrimaBobyqa;
        opt.rhobeg = 1.0;
        opt.rhoend = 1e-6;
        opt.max_feval = 500;

        let markdown = opt.to_markdown();
        assert!(markdown.contains("| Optimizer                | `bobyqa` |"));
        assert!(markdown.contains("| Backend                  | `prima` |"));
        assert!(markdown.contains("| rhobeg                   | 1 |"));
        assert!(markdown.contains("| rhoend                   | 0.000001 |"));
        assert!(markdown.contains("| maxfeval                 | 500 |"));
        assert!(!markdown.contains("| ftol_rel"));
        assert!(!markdown.contains("| initial_step"));

        let html = opt.to_html();
        assert!(html.contains("<code>prima</code>"));
        assert!(html.contains("<td align=\"left\">rhobeg</td>"));
        assert!(!html.contains("<td align=\"left\">ftol_rel</td>"));

        let latex = opt.to_latex();
        assert!(latex.contains("Backend & \\texttt{prima}"));
        assert!(latex.contains("rhobeg & 1"));
        assert!(!latex.contains("ftol\\_rel"));

        let display = format!("{opt}");
        assert!(display.contains("Backend:                  prima"));
        assert!(display.contains("rhobeg:"));
        assert!(!display.contains("ftol_rel:"));
    }

    #[test]
    fn test_html_contains_core_markup() {
        let mut opt = OptSummary::new(vec![1.0, 0.5]);
        opt.optimizer = Optimizer::NloptBobyqa;
        opt.return_value = "FTOL_REACHED".to_string();

        let out = opt.to_html();
        assert!(out.contains("<b>Initialization</b>"));
        assert!(out.contains("<code>LN_BOBYQA</code>"));
        assert!(out.contains("<code>FTOL_REACHED</code>"));
    }

    #[test]
    fn test_latex_contains_core_markup() {
        let mut opt = OptSummary::new(vec![1.0, 0.5]);
        opt.optimizer = Optimizer::NloptBobyqa;
        opt.return_value = "FTOL_REACHED".to_string();

        let out = opt.to_latex();
        assert!(out.contains("\\textbf{Initialization}"));
        assert!(out.contains("\\texttt{LN\\_BOBYQA}"));
        assert!(out.contains("\\texttt{FTOL\\_REACHED}"));
    }

    fn status_of(feval: i64, code: &str) -> ConvergenceStatus {
        let mut opt = OptSummary::new(vec![1.0]);
        opt.feval = feval;
        opt.return_value = code.to_string();
        opt.convergence_status()
    }

    #[test]
    fn convergence_status_classifies_every_backend_vocabulary() {
        // Not fitted / no status -> NotRun.
        assert_eq!(status_of(-1, "FTOL_REACHED"), ConvergenceStatus::NotRun);
        assert_eq!(status_of(10, ""), ConvergenceStatus::NotRun);

        // Clean convergence across NLopt / COBYLA / trust-region / PRIMA.
        for code in [
            "SUCCESS",
            "STOPVAL_REACHED",
            "FTOL_REACHED",
            "XTOL_REACHED",
            "RADIUS_REACHED",
            "SMALL_TR_RADIUS",
            "FTARGET_ACHIEVED",
        ] {
            assert_eq!(
                status_of(10, code),
                ConvergenceStatus::Converged,
                "{code} should be Converged"
            );
        }

        // Budget exhaustion must never be reported as converged — this is the
        // "non-convergence masquerading as a fit" hazard.
        for code in [
            "MAXEVAL_REACHED",
            "MAXTIME_REACHED",
            "MAXFUN_REACHED",
            "MAXTR_REACHED",
            "CALLBACK_TERMINATE",
        ] {
            assert_eq!(
                status_of(10, code),
                ConvergenceStatus::BudgetExhausted,
                "{code} should be BudgetExhausted"
            );
            assert!(!status_of(10, code).eq(&ConvergenceStatus::Converged));
        }

        assert_eq!(
            status_of(10, "ROUNDOFF_LIMITED"),
            ConvergenceStatus::RoundoffLimited
        );

        for code in [
            "FAILURE",
            "INVALID_ARGS",
            "OUT_OF_MEMORY",
            "FORCED_STOP",
            "FORCED_BAD_BOUNDARY",
            "TRSUBP_FAILED",
            "NAN_INF_F",
            "INVALID_INPUT",
        ] {
            assert_eq!(
                status_of(10, code),
                ConvergenceStatus::Failed,
                "{code} should be Failed"
            );
        }
    }

    #[test]
    fn convergence_status_unwraps_kkt_boundary_restart() {
        // The KKT restart wrapper must classify by its inner outcome.
        assert_eq!(
            status_of(10, "KKT_BOUNDARY_RESTART(2): FTOL_REACHED"),
            ConvergenceStatus::Converged
        );
        assert_eq!(
            status_of(10, "KKT_BOUNDARY_RESTART(1): MAXEVAL_REACHED"),
            ConvergenceStatus::BudgetExhausted
        );
    }

    #[test]
    fn convergence_status_unwraps_start_ladder_and_active_face() {
        // The TrustBQ start-ladder and active-face wrappers must classify by
        // the inner outcome, including when wrappers stack.
        assert_eq!(
            status_of(10, "START_LADDER(diagonal_first:45 evals): FTOL_REACHED"),
            ConvergenceStatus::Converged
        );
        assert_eq!(
            status_of(10, "START_LADDER(diagonal_first:45 evals): MAXEVAL_REACHED"),
            ConvergenceStatus::BudgetExhausted
        );
        assert_eq!(
            status_of(
                10,
                "ACTIVE_FACE(rank4of8:312 evals:certified): FTOL_REACHED"
            ),
            ConvergenceStatus::Converged
        );
        assert_eq!(
            status_of(
                10,
                "ACTIVE_FACE(rank4of8:312 evals:uncertified): KKT_BOUNDARY_RESTART(1): FTOL_REACHED"
            ),
            ConvergenceStatus::Converged
        );
    }

    #[test]
    fn convergence_status_unwraps_joint_glmm_prefixes() {
        assert_eq!(
            status_of(10, "JOINT_LAPLACE:SUCCESS"),
            ConvergenceStatus::Converged
        );
        assert_eq!(
            status_of(10, "JOINT_AGQ:MAXEVAL_REACHED"),
            ConvergenceStatus::BudgetExhausted
        );
        assert_eq!(
            status_of(10, "JOINT_LAPLACE_FAILED:ROUNDOFF_LIMITED"),
            ConvergenceStatus::RoundoffLimited
        );
        assert_eq!(
            status_of(
                78,
                "JOINT_LAPLACE_FALLBACK_FAST_PIRLS(joint=JOINT_LAPLACE:MAXEVAL_REACHED; fast=FTOL_REACHED)"
            ),
            ConvergenceStatus::Converged,
            "a labelled fallback must classify the returned fast-PIRLS fit"
        );
        assert_eq!(
            status_of(
                25,
                "JOINT_LAPLACE_FALLBACK_FAST_PIRLS(joint=JOINT_LAPLACE:MAXEVAL_REACHED; fast=MAXEVAL_REACHED)"
            ),
            ConvergenceStatus::BudgetExhausted
        );
    }

    #[test]
    fn converged_is_true_only_for_clean_convergence() {
        assert!(status_of(10, "FTOL_REACHED") == ConvergenceStatus::Converged);
        let mut opt = OptSummary::new(vec![1.0]);
        opt.feval = 10;
        opt.return_value = "MAXEVAL_REACHED".to_string();
        assert!(!opt.converged());
        opt.return_value = "FTOL_REACHED".to_string();
        assert!(opt.converged());
    }
}
