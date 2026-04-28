//! AST types for mixed-model formula terms.
//!
//! These types represent the parsed structure of an R/Julia-style formula such as
//! `y ~ 1 + x1 + x2 + (1 + x1 | group)`.  The design mirrors the term representation
//! used by Julia's MixedModels.jl.

use std::fmt;

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
    /// Source text and parser-level canonicalization metadata, when the term
    /// came from the formula parser rather than a manually constructed AST.
    pub source: Option<RandomTermSource>,
}

/// The grouping factor for a random-effect term.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupingFactor {
    /// A single grouping variable, e.g. `subject`.
    Single(String),
    /// An interaction of grouping variables using legacy `&` syntax, e.g. `g1 & g2`.
    Interaction(Vec<String>),
    /// A cell-level interaction grouping factor, e.g. `subject:item`.
    Cell(Vec<String>),
}

/// Source metadata for a parsed random-effect term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomTermSource {
    /// Parenthesized source text exactly as written by the user, modulo
    /// leading/trailing formula whitespace.
    pub written: String,
    /// Parser-level expansion that produced this canonical term, if any.
    pub expansion: Option<RandomTermExpansion>,
}

/// Parser-level canonicalization applied to a random-effect grouping form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomTermExpansion {
    /// `(b | a/b)` expanded to `(b | a) + (b | a:b)`.
    NestedGrouping,
    /// `(b | a*b)` expanded to `(b | a) + (b | b) + (b | a:b)`.
    CrossedGrouping,
}

impl Formula {
    /// Whether the formula includes a fixed-effects intercept.
    ///
    /// Returns `true` if an explicit `Intercept` term is present,
    /// or if no explicit intercept/no-intercept directive was given
    /// (the parser inserts an implicit intercept).
    pub fn has_intercept(&self) -> bool {
        self.fixed_terms
            .iter()
            .any(|t| matches!(t, FixedTerm::Intercept))
    }
}

impl fmt::Display for FixedTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FixedTerm::Intercept => write!(f, "1"),
            FixedTerm::NoIntercept => write!(f, "0"),
            FixedTerm::Column(name) => write!(f, "{name}"),
            FixedTerm::Interaction(names) => write!(f, "{}", names.join(":")),
            FixedTerm::Nested(names) => write!(f, "{}", names.join("/")),
        }
    }
}

impl fmt::Display for GroupingFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GroupingFactor::Single(name) => write!(f, "{name}"),
            GroupingFactor::Interaction(names) => write!(f, "{}", names.join(" & ")),
            GroupingFactor::Cell(names) => write!(f, "{}", names.join(":")),
        }
    }
}

impl fmt::Display for RandomTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bar = if self.zerocorr { "||" } else { "|" };
        let terms = self
            .terms
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" + ");
        write!(f, "({terms} {bar} {})", self.grouping)
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rhs = self
            .fixed_terms
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        rhs.extend(self.random_terms.iter().map(ToString::to_string));
        write!(f, "{} ~ {}", self.response, rhs.join(" + "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formula_display_canonicalizes_terms() {
        let formula = Formula {
            response: "reaction".to_string(),
            fixed_terms: vec![FixedTerm::Intercept, FixedTerm::Column("days".to_string())],
            random_terms: vec![RandomTerm {
                terms: vec![FixedTerm::Intercept, FixedTerm::Column("days".to_string())],
                grouping: GroupingFactor::Single("subj".to_string()),
                zerocorr: false,
                source: None,
            }],
        };

        assert_eq!(
            formula.to_string(),
            "reaction ~ 1 + days + (1 + days | subj)"
        );
    }

    #[test]
    fn test_formula_display_handles_zero_correlation_and_group_interactions() {
        let formula = Formula {
            response: "y".to_string(),
            fixed_terms: vec![FixedTerm::NoIntercept, FixedTerm::Column("x".to_string())],
            random_terms: vec![RandomTerm {
                terms: vec![FixedTerm::Intercept, FixedTerm::Column("x".to_string())],
                grouping: GroupingFactor::Interaction(vec![
                    "subject".to_string(),
                    "item".to_string(),
                ]),
                zerocorr: true,
                source: None,
            }],
        };

        assert_eq!(formula.to_string(), "y ~ 0 + x + (1 + x || subject & item)");
    }
}
