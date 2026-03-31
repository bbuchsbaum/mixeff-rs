//! Formula parsing for mixed-effects models.
//!
//! This module provides a parser for R/Julia-style mixed-model formulas such as
//! `y ~ 1 + x1 + x2 + (1 + x1 | group)`.
//!
//! # Quick start
//!
//! ```
//! use mixedmodels::formula::{parse_formula, Formula, FixedTerm, RandomTerm, GroupingFactor};
//!
//! let f = parse_formula("y ~ x1 + (1 | group)").unwrap();
//! assert_eq!(f.response, "y");
//! ```

pub mod parser;
pub mod terms;

// Re-export the main public types for convenience.
pub use parser::{parse_formula, FormulaError};
pub use terms::{FixedTerm, Formula, GroupingFactor, RandomTerm};
