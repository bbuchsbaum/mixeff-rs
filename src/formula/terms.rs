//! AST types for mixed-model formula terms.
//!
//! These types represent the parsed structure of an R/Julia-style formula such as
//! `y ~ 1 + x1 + x2 + (1 + x1 | group)`.  The design mirrors the term representation
//! used by Julia's MixedModels.jl.

use std::fmt;

use super::transform::DerivedColumn;
use crate::error::Result;
use crate::model::data::DataFrame;

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
    ///
    /// If the response was an in-formula stateless transform (e.g.
    /// `log(reaction)`), this holds the canonical R-style label and a
    /// matching entry exists in [`Formula::derived`]. Above the data
    /// boundary the response is just "a column by this name".
    pub response: String,
    /// Fixed-effect terms on the RHS.
    pub fixed_terms: Vec<FixedTerm>,
    /// Random-effect specifications `(... | group)`.
    pub random_terms: Vec<RandomTerm>,
    /// Synthetic numeric columns lowered from stateless in-formula
    /// transforms (`I(days^2)`, `log(reaction)`, …). Each is keyed by its
    /// canonical R-style label, which doubles as the column name, the
    /// coefficient name, and (if it is the response) the response name.
    /// Materialized into the working [`DataFrame`] at the data boundary
    /// (LMM/GLMM build and predict) by [`Formula::materialize`]; the layered
    /// tower above that boundary never sees a transform AST.
    pub derived: Vec<DerivedColumn>,
}

/// A single fixed-effect term.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FixedTerm {
    /// Explicit intercept (`1`).
    Intercept,
    /// Suppress the intercept (`0` or `-1`).
    NoIntercept,
    /// A single column / predictor (`x1`).
    Column(String),
    /// An interaction between two or more predictors (`a:b`).
    Interaction(Vec<String>),
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
    /// Requested covariance family for this random-effect block. Ordinary
    /// `|` syntax uses [`RandomCovariance::Full`], `||` uses
    /// [`RandomCovariance::Diagonal`], and lme4-style wrappers such as
    /// `cs(...)` / `ar1(...)` use the corresponding structured family.
    pub covariance: RandomCovariance,
    /// Source text and parser-level canonicalization metadata, when the term
    /// came from the formula parser rather than a manually constructed AST.
    pub source: Option<RandomTermSource>,
}

/// Requested random-effect covariance family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RandomCovariance {
    /// Unstructured full lower-Cholesky covariance.
    Full,
    /// Diagonal covariance in the compiled random-effect basis.
    Diagonal,
    /// Compound-symmetry covariance. Parsed for v1.0 contract readiness but
    /// refused by the current fitting engine.
    CompoundSymmetry,
    /// Random-effect autoregressive order-one covariance. Parsed for v1.0
    /// contract readiness but refused by the current fitting engine.
    Ar1,
}

impl RandomCovariance {
    pub fn label(self) -> &'static str {
        match self {
            RandomCovariance::Full => "full",
            RandomCovariance::Diagonal => "diagonal",
            RandomCovariance::CompoundSymmetry => "compound_symmetry",
            RandomCovariance::Ar1 => "ar1",
        }
    }

    pub fn wrapper(self) -> Option<&'static str> {
        match self {
            RandomCovariance::Full => None,
            RandomCovariance::Diagonal => Some("diag"),
            RandomCovariance::CompoundSymmetry => Some("cs"),
            RandomCovariance::Ar1 => Some("ar1"),
        }
    }

    pub fn is_supported_for_fit(self) -> bool {
        matches!(self, RandomCovariance::Full | RandomCovariance::Diagonal)
    }
}

/// The grouping factor for a random-effect term.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
#[non_exhaustive]
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

    /// Lower the stateless in-formula transforms into synthetic numeric
    /// columns at the data boundary.
    ///
    /// Returns a clone of `data` with one numeric column appended per entry
    /// in [`Formula::derived`], named by its canonical label. This is the
    /// single seam where transforms become real columns; every layer above
    /// keeps seeing "a column by name", so `FixedTerm`/`RandomTerm`/
    /// `FixedDesign`/`predict`/`coef_names` are untouched.
    ///
    /// The same call runs at fit time (against training data) and at
    /// prediction time (against `newdata`): because each transform is a pure
    /// pointwise recipe there is no stored basis to diverge from.
    ///
    /// **Collision policy**: if the caller supplies a column whose name
    /// byte-equals a derived label (e.g. a raw column literally named
    /// `"I(x^2)"`), the engine **recomputes** the recipe and asserts
    /// agreement within a tight tolerance (`1e-10` relative + `1e-12`
    /// absolute). A mismatch is an error — there must be exactly one source
    /// of truth for the recipe, and the engine owns it (see
    /// `docs/formula_transform_seam.md` §"hidden model surgery"). Silently
    /// accepting a diverging pre-supplied column would recreate the exact
    /// two-implementations-of-the-recipe failure the seam contract forbids.
    pub fn materialize(&self, data: &DataFrame) -> Result<DataFrame> {
        if self.derived.is_empty() {
            return Ok(data.clone());
        }
        let mut out = data.clone();
        for d in &self.derived {
            let engine_values = super::transform::materialize_column(d, &out)?;

            if let Some(existing) = out.numeric(&d.label) {
                // A column with this canonical label already exists.  The
                // engine owns the recipe, so recompute and verify — silently
                // trusting a pre-supplied column would create two
                // implementations of the recipe that could diverge at
                // prediction time.
                let existing = existing.to_vec();
                if existing.len() != engine_values.len() {
                    return Err(crate::error::MixedModelError::InvalidArgument(format!(
                        "in-formula transform `{}`: a column with this name already \
                         exists in the data but has {} rows, expected {} — the engine \
                         owns this derived column; rename the raw column to avoid \
                         the collision",
                        d.label,
                        existing.len(),
                        engine_values.len()
                    )));
                }
                for (row, (&supplied, &computed)) in
                    existing.iter().zip(engine_values.iter()).enumerate()
                {
                    // Absolute tolerance for values near zero; relative for
                    // larger magnitudes. 1e-10 rel / 1e-12 abs matches the
                    // precision of double-precision arithmetic.
                    let abs_diff = (supplied - computed).abs();
                    let rel_diff = if computed.abs() > 1e-12 {
                        abs_diff / computed.abs()
                    } else {
                        abs_diff
                    };
                    if abs_diff > 1e-12 && rel_diff > 1e-10 {
                        return Err(crate::error::MixedModelError::InvalidArgument(format!(
                            "in-formula transform `{}` at row {}: the engine \
                             computed {computed} but the pre-supplied column \
                             contains {supplied} (relative diff {rel_diff:.3e}). \
                             The engine owns this derived-column recipe; there \
                             must be exactly one source of truth. Rename the raw \
                             column to avoid the collision, or remove it and let \
                             the engine compute it.",
                            d.label, row
                        )));
                    }
                }
                // Values agree — the column is already correct; leave it.
            } else {
                out.add_numeric(&d.label, engine_values)?;
            }
        }
        Ok(out)
    }
}

impl fmt::Display for FixedTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FixedTerm::Intercept => write!(f, "1"),
            FixedTerm::NoIntercept => write!(f, "0"),
            FixedTerm::Column(name) => write!(f, "{name}"),
            FixedTerm::Interaction(names) => write!(f, "{}", names.join(":")),
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
        let terms = self
            .terms
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" + ");
        if self.zerocorr {
            let bar = if self.zerocorr { "||" } else { "|" };
            write!(f, "({terms} {bar} {})", self.grouping)
        } else if let Some(wrapper) = self.covariance.wrapper() {
            write!(f, "{wrapper}({terms} | {})", self.grouping)
        } else {
            write!(f, "({terms} | {})", self.grouping)
        }
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
                covariance: RandomCovariance::Full,
                source: None,
            }],
            derived: Vec::new(),
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
                covariance: RandomCovariance::Diagonal,
                source: None,
            }],
            derived: Vec::new(),
        };

        assert_eq!(formula.to_string(), "y ~ 0 + x + (1 + x || subject & item)");
    }
}
