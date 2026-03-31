//! AST types for mixed-model formula terms.
//!
//! These types represent the parsed structure of an R/Julia-style formula such as
//! `y ~ 1 + x1 + x2 + (1 + x1 | group)`.  The design mirrors the term representation
//! used by Julia's MixedModels.jl.

/// A fully parsed mixed-model formula.
///
/// A formula has three parts:
/// - **response**: the left-hand side variable name (e.g. `y`).
/// - **fixed_terms**: the fixed-effect terms on the right-hand side.
/// - **random_terms**: the random-effect specifications `(terms | grouping)`.
///
/// # Implicit intercept
///
/// If the fixed terms do not contain an explicit [`FixedTerm::Intercept`] or
/// [`FixedTerm::NoIntercept`], an intercept is assumed to be present (the
/// parser inserts one automatically).
#[derive(Debug, Clone)]
pub struct Formula {
    /// Name of the response (outcome) variable.
    pub response: String,
    /// Fixed-effect terms on the RHS.
    pub fixed_terms: Vec<FixedTerm>,
    /// Random-effect specifications `(... | group)`.
    pub random_terms: Vec<RandomTerm>,
}

/// A single fixed-effect term.
#[derive(Debug, Clone, PartialEq)]
pub enum FixedTerm {
    /// Explicit intercept (`1`).
    Intercept,
    /// Suppress the intercept (`0` or `-1`).
    NoIntercept,
    /// A single column / predictor (`x1`).
    Column(String),
    /// An interaction between two or more predictors (`a:b`).
    Interaction(Vec<String>),
    /// A nesting specification (`a/b`), which expands to `a + a:b`.
    Nested(Vec<String>),
}

/// A random-effect specification, corresponding to `(terms | grouping)` or
/// `(terms || grouping)` in the formula string.
#[derive(Debug, Clone)]
pub struct RandomTerm {
    /// The model terms inside the random-effect parentheses.
    pub terms: Vec<FixedTerm>,
    /// The grouping factor (right of `|` or `||`).
    pub grouping: GroupingFactor,
    /// If `true`, the `||` (zero-correlation) syntax was used, meaning the
    /// covariance between random-effect terms is forced to zero.
    pub zerocorr: bool,
}

/// The grouping factor for a random-effect term.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupingFactor {
    /// A single grouping variable, e.g. `subject`.
    Single(String),
    /// An interaction of grouping variables, e.g. `g1 & g2`.
    Interaction(Vec<String>),
}

impl Formula {
    /// Whether the formula includes a fixed-effects intercept.
    ///
    /// Returns `true` if an explicit `Intercept` term is present,
    /// or if no explicit intercept/no-intercept directive was given
    /// (the parser inserts an implicit intercept).
    pub fn has_intercept(&self) -> bool {
        self.fixed_terms.iter().any(|t| matches!(t, FixedTerm::Intercept))
    }
}
