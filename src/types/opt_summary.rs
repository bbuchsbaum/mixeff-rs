//! Optimization summary for mixed-effects model fitting.
//!
//! This is a port of `OptSummary` from Julia's MixedModels.jl.
//! It tracks the optimizer state, tolerances, fit log, and
//! model-fitting options (REML, adaptive Gauss-Hermite quadrature,
//! known σ).

use std::fmt;

/// Choice of optimizer algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    /// COBYLA — Constrained Optimization By Linear Approximations.
    /// A derivative-free optimizer that handles bound constraints
    /// via inequality constraints.
    Cobyla,
    /// Bound-aware pattern search used for small multivariate θ vectors.
    PatternSearch,
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
/// `Native` (the COBYLA crate and the in-tree pattern-search optimizer).
/// `Nlopt` is the upstream default in the Julia reference and is the active
/// backend for any `Optimizer::Nlopt*` variant. `Prima` is reserved for the
/// PRIMA derivative-free family; `Optimizer::PrimaBobyqa` is wired for LMMs
/// when the non-default `prima` Cargo feature is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerBackend {
    /// In-tree Rust optimizers (COBYLA crate, pattern search).
    Native,
    /// NLopt-backed optimizers (BOBYQA, NEWUOA).
    Nlopt,
    /// PRIMA-backed optimizers (bobyqa/cobyla/lincoa/newuoa).
    Prima,
}

impl Optimizer {
    /// Canonical backend for this optimizer.
    pub fn canonical_backend(self) -> OptimizerBackend {
        match self {
            Optimizer::Cobyla | Optimizer::PatternSearch => OptimizerBackend::Native,
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

    /// Optimization backend providing the optimizer. Defaults to
    /// `OptimizerBackend::Native`; PRIMA dispatch requires the non-default
    /// `prima` Cargo feature and a system `libprimac`.
    pub backend: OptimizerBackend,

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

            optimizer: Optimizer::Cobyla,
            backend: OptimizerBackend::Native,
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

    /// The optimizer label used for display.
    pub fn optimizer_name(&self) -> &'static str {
        match self.optimizer {
            Optimizer::Cobyla => "cobyla",
            Optimizer::PatternSearch => "pattern_search",
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
            Optimizer::NloptNewuoa => "LN_NEWUOA",
            Optimizer::NloptBobyqa => "LN_BOBYQA",
            Optimizer::PrimaBobyqa => "bobyqa",
            Optimizer::PrimaCobyla => "cobyla",
            Optimizer::PrimaLincoa => "lincoa",
            Optimizer::PrimaNewuoa => "newuoa",
        }
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
                "| ftol_rel                 | {} |\n",
                "| ftol_abs                 | {} |\n",
                "| xtol_rel                 | {} |\n",
                "| xtol_abs                 | {} |\n",
                "| initial_step             | {} |\n",
                "| maxfeval                 | {} |\n",
                "| maxtime                  | {} |\n",
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
            self.ftol_rel,
            self.ftol_abs,
            self.xtol_rel,
            format!("{:?}", self.xtol_abs),
            format!("{:?}", self.initial_step),
            self.max_feval,
            self.max_time,
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
                "<tr><td align=\"left\">ftol_rel</td><td align=\"left\">{}</td></tr>",
                "<tr><td align=\"left\">ftol_abs</td><td align=\"left\">{}</td></tr>",
                "<tr><td align=\"left\">xtol_rel</td><td align=\"left\">{}</td></tr>",
                "<tr><td align=\"left\">xtol_abs</td><td align=\"left\">{:?}</td></tr>",
                "<tr><td align=\"left\">initial_step</td><td align=\"left\">{:?}</td></tr>",
                "<tr><td align=\"left\">maxfeval</td><td align=\"left\">{}</td></tr>",
                "<tr><td align=\"left\">maxtime</td><td align=\"left\">{}</td></tr>",
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
            self.ftol_rel,
            self.ftol_abs,
            self.xtol_rel,
            self.xtol_abs,
            self.initial_step,
            self.max_feval,
            self.max_time,
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
                "ftol_rel & {} \\\\\n",
                "ftol_abs & {} \\\\\n",
                "xtol_rel & {} \\\\\n",
                "xtol_abs & {:?} \\\\\n",
                "initial_step & {:?} \\\\\n",
                "maxfeval & {} \\\\\n",
                "maxtime & {} \\\\\n",
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
            self.ftol_rel,
            self.ftol_abs,
            self.xtol_rel,
            self.xtol_abs,
            self.initial_step,
            self.max_feval,
            self.max_time,
            self.xtol_zero_abs,
            self.ftol_zero_abs,
            self.feval,
            self.final_params,
            self.fmin,
            latex_escape_code(&self.return_value)
        )
    }
}

impl fmt::Display for OptSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Initial parameter vector: {:?}", self.initial)?;
        writeln!(f, "Initial objective value:  {}", self.finitial)?;
        writeln!(f)?;
        writeln!(f, "Backend:                  {}", self.backend_name())?;
        writeln!(f, "Optimizer:                {}", self.optimizer_name())?;
        writeln!(f, "ftol_rel:                 {}", self.ftol_rel)?;
        writeln!(f, "ftol_abs:                 {}", self.ftol_abs)?;
        writeln!(f, "xtol_rel:                 {}", self.xtol_rel)?;
        writeln!(f, "xtol_abs:                 {:?}", self.xtol_abs)?;
        writeln!(f, "maxfeval:                 {}", self.max_feval)?;
        writeln!(f, "maxtime:                  {}", self.max_time)?;
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
}
