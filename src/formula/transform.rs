//! Stateless in-formula transform subset.
//!
//! This module owns the *stateless / pointwise* slice of R's formula language
//! that the engine is allowed to evaluate itself, per the frozen contract in
//! `docs/formula_transform_seam.md`. The defining property is that every
//! transform here is a pure pointwise function `R -> R` with **no
//! fitting-time state** (no centering means, no QR basis, no knots, no level
//! set). Because the closed-form expression *is* the recipe, prediction on new
//! data is correct simply by re-evaluating the same expression — there is no
//! `predvars` and no second source of truth.
//!
//! Supported:
//! - `I(<arith>)` where `<arith>` is `+ - * / ^`, unary `-`, parentheses,
//!   f64 literals, and column references. `^` is real-valued power
//!   (`f64::powf`).
//! - Bare pointwise calls outside `I()`: `log` (natural), `log2`, `log10`,
//!   `exp`, `sqrt`, `abs`. Single argument, which may itself be `I(...)`, a
//!   column, or a nested whitelisted call (e.g. `sqrt(I(x + 1))`,
//!   `log(reaction)`).
//!
//! Anything else (`poly`, `scale`, `ns`, `bs`, `cut`, `factor`, `center`,
//! unknown functions, multi-argument `log(x, base)`, non-whitelisted
//! operators) is **forbidden**, not merely unimplemented: it is stateful and
//! belongs to the host wrapper. The forbidden set keeps the existing
//! actionable refusal in [`super::parser`].
//!
//! # Canonical label rule
//!
//! Each derived column is identified by a **canonical R-style label** that is
//! byte-identical to what R would print and is used simultaneously as the
//! synthetic column name, the coefficient name, and (on the LHS) the response
//! name. The single normalization rule is: **no whitespace anywhere inside the
//! label.** Operators are emitted bare (`I(days^2)`, `I(a*b)`, `I(1/x)`,
//! `I(-x)`), function calls are `name(arg)` (`log(reaction)`,
//! `sqrt(I(x+1))`), and parentheses are emitted only where operator
//! precedence requires them. Numeric literals print without a trailing `.0`
//! for integral values (`I(x+1)`, not `I(x+1.0)`).

use std::fmt::Write as _;

use super::parser::FormulaError;
use crate::error::{MixedModelError, Result};
use crate::model::data::{Column, DataFrame};

/// A whitelisted single-argument pointwise function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformFn {
    /// Natural logarithm.
    Ln,
    /// Base-2 logarithm.
    Log2,
    /// Base-10 logarithm.
    Log10,
    /// Exponential.
    Exp,
    /// Square root.
    Sqrt,
    /// Absolute value.
    Abs,
}

impl TransformFn {
    /// Resolve a function name to a whitelisted [`TransformFn`], if any.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "log" => Some(TransformFn::Ln),
            "log2" => Some(TransformFn::Log2),
            "log10" => Some(TransformFn::Log10),
            "exp" => Some(TransformFn::Exp),
            "sqrt" => Some(TransformFn::Sqrt),
            "abs" => Some(TransformFn::Abs),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            TransformFn::Ln => "log",
            TransformFn::Log2 => "log2",
            TransformFn::Log10 => "log10",
            TransformFn::Exp => "exp",
            TransformFn::Sqrt => "sqrt",
            TransformFn::Abs => "abs",
        }
    }

    fn apply(self, x: f64) -> f64 {
        match self {
            TransformFn::Ln => x.ln(),
            TransformFn::Log2 => x.log2(),
            TransformFn::Log10 => x.log10(),
            TransformFn::Exp => x.exp(),
            TransformFn::Sqrt => x.sqrt(),
            TransformFn::Abs => x.abs(),
        }
    }
}

/// A binary arithmetic operator allowed inside `I(...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Division (`/`).
    Div,
    /// Real-valued exponentiation (`^`).
    Pow,
}

impl BinOp {
    fn symbol(self) -> char {
        match self {
            BinOp::Add => '+',
            BinOp::Sub => '-',
            BinOp::Mul => '*',
            BinOp::Div => '/',
            BinOp::Pow => '^',
        }
    }

    /// Binding power: higher binds tighter. `^` is highest, `* /` next,
    /// `+ -` lowest.
    fn precedence(self) -> u8 {
        match self {
            BinOp::Add | BinOp::Sub => 1,
            BinOp::Mul | BinOp::Div => 2,
            BinOp::Pow => 3,
        }
    }

    fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
            BinOp::Pow => a.powf(b),
        }
    }
}

/// Stateless transform expression AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Numeric literal.
    Lit(f64),
    /// A reference to a numeric data column by name.
    Col(String),
    /// Unary negation.
    Neg(Box<Expr>),
    /// Binary arithmetic.
    Bin(BinOp, Box<Expr>, Box<Expr>),
    /// A whitelisted single-argument pointwise call.
    Call(TransformFn, Box<Expr>),
}

/// A derived (synthetic) numeric column produced by lowering a stateless
/// transform at the data boundary.
///
/// `label` is the canonical R-style label (see module docs); it is the
/// synthetic column name, the coefficient name, and — when the response was
/// transformed — the response name. `expr` is the closed-form recipe,
/// re-evaluated verbatim on prediction `newdata`.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivedColumn {
    /// Canonical R-style label (column name == coef name == response name).
    pub label: String,
    /// The stateless expression evaluated to produce the column.
    pub expr: Expr,
}

impl DerivedColumn {
    /// Build a derived-column descriptor from a parsed expression.
    ///
    /// The canonical label is computed immediately and used later as the
    /// synthetic column name at the fit/predict data boundary.
    pub fn new(expr: Expr) -> Self {
        Self {
            label: canonical_label(&expr),
            expr,
        }
    }
}

// ---------------------------------------------------------------------------
// Canonical label formatting
// ---------------------------------------------------------------------------

/// Canonical R-style label for a derived column. The top-level wrapper is
/// `I(...)` for an arithmetic expression and `name(arg)` for a function call;
/// a bare column reference (degenerate) prints as the column name.
pub fn canonical_label(expr: &Expr) -> String {
    match expr {
        Expr::Col(name) => name.clone(),
        Expr::Call(f, arg) => {
            let mut s = String::new();
            let _ = write!(s, "{}(", f.label());
            s.push_str(&inner_arg_label(arg));
            s.push(')');
            s
        }
        // Arithmetic / literal / negation at top level → wrap in `I(...)`.
        _ => {
            let mut s = String::from("I(");
            write_expr(&mut s, expr, 0);
            s.push(')');
            s
        }
    }
}

/// Label for a function argument: a nested call keeps its own `name(arg)`
/// form, a bare column prints as the name, anything arithmetic is wrapped in
/// `I(...)` (matching how R prints e.g. `sqrt(I(x + 1))`).
fn inner_arg_label(expr: &Expr) -> String {
    match expr {
        Expr::Col(name) => name.clone(),
        Expr::Call(_, _) => canonical_label(expr),
        _ => {
            let mut s = String::from("I(");
            write_expr(&mut s, expr, 0);
            s.push(')');
            s
        }
    }
}

/// Minimal canonical f64 rendering: integral values print without a trailing
/// `.0` so `I(x + 1)` (not `I(x + 1.0)`); other values use the default
/// shortest round-trip representation.
fn fmt_lit(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// Write `expr` into `out` with parentheses added only where operator
/// precedence requires them. `parent_prec` is the binding power of the
/// enclosing operator (0 for the outermost context).
fn write_expr(out: &mut String, expr: &Expr, parent_prec: u8) {
    match expr {
        Expr::Lit(v) => out.push_str(&fmt_lit(*v)),
        Expr::Col(name) => out.push_str(name),
        Expr::Neg(inner) => {
            // Unary minus has lower precedence than `^` (R rule: `-x^2` is
            // `-(x^2)`). Parentheses around the whole `-...` are needed only
            // when the Neg is in a position where an unparenthesized `-`
            // would be misread — specifically as the right operand of `^`
            // (e.g. `2^(-x)` must keep the parens so it is not read as
            // `(2^2)^x` after re-parsing). At any lower parent precedence
            // the leading `-` is unambiguous.
            let need = parent_prec > BinOp::Mul.precedence(); // only inside ^
            if need {
                out.push('(');
            }
            out.push('-');
            // Write the inner expression at the `^` precedence level so that
            // a Pow inside the operand (e.g. `-x^2` → Neg(Bin(Pow,x,2)))
            // renders without extra parens: `-x^2` not `-(x^2)`.
            write_expr(out, inner, BinOp::Pow.precedence());
            if need {
                out.push(')');
            }
        }
        Expr::Bin(op, a, b) => {
            let p = op.precedence();
            let need = p < parent_prec;
            if need {
                out.push('(');
            }
            write_expr(out, a, p);
            out.push(op.symbol());
            // Right operand: use p+1 so same-precedence right children are
            // parenthesized (keeps `a-(b-c)` distinct from `a-b-c`).
            write_expr(out, b, p + 1);
            if need {
                out.push(')');
            }
        }
        Expr::Call(f, arg) => {
            let _ = write!(out, "{}(", f.label());
            out.push_str(&inner_arg_label(arg));
            out.push(')');
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Evaluate a stateless transform expression against `data`, producing one
/// `f64` per row.
///
/// Errors are actionable [`MixedModelError`]s (never panics):
/// - a referenced column is missing,
/// - a referenced column is categorical (cannot be used in arithmetic),
/// - the result contains a non-finite value (e.g. `log(-1)`, `sqrt(-1)`,
///   `1/0`). NaN/Inf from a transform on training data is an error, not a
///   silent pass — it would otherwise surface only as an opaque Cholesky
///   failure.
pub fn eval(expr: &Expr, data: &DataFrame) -> Result<Vec<f64>> {
    let n = data.nrow();
    let mut out = Vec::with_capacity(n);
    for row in 0..n {
        out.push(eval_row(expr, data, row)?);
    }
    Ok(out)
}

fn eval_row(expr: &Expr, data: &DataFrame, row: usize) -> Result<f64> {
    match expr {
        Expr::Lit(v) => Ok(*v),
        Expr::Col(name) => match data.column(name) {
            Some(Column::Numeric(v)) => Ok(v[row]),
            Some(Column::Categorical(_)) => Err(MixedModelError::InvalidArgument(format!(
                "in-formula transform references categorical column `{name}`; \
                 stateless transforms operate on numeric columns only — \
                 precompute a numeric encoding or use the host wrapper for \
                 factor handling"
            ))),
            None => Err(MixedModelError::InvalidArgument(format!(
                "in-formula transform references column `{name}`, which is not \
                 present in the data"
            ))),
        },
        Expr::Neg(inner) => Ok(-eval_row(inner, data, row)?),
        Expr::Bin(op, a, b) => {
            let lhs = eval_row(a, data, row)?;
            let rhs = eval_row(b, data, row)?;
            Ok(op.apply(lhs, rhs))
        }
        Expr::Call(f, arg) => Ok(f.apply(eval_row(arg, data, row)?)),
    }
}

/// Evaluate `derived` against `data` and return the resulting column values,
/// rejecting any non-finite result with an actionable error keyed to the
/// canonical label.
pub fn materialize_column(derived: &DerivedColumn, data: &DataFrame) -> Result<Vec<f64>> {
    let values = eval(&derived.expr, data)?;
    if let Some(pos) = values.iter().position(|v| !v.is_finite()) {
        return Err(MixedModelError::InvalidArgument(format!(
            "in-formula transform `{}` produced a non-finite value ({}) at \
             row {}; the transform is undefined there (e.g. log/sqrt of a \
             non-positive value, or division by zero) — clean or restrict \
             the data before fitting",
            derived.label, values[pos], pos
        )));
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// Parsing the stateless subset from a token slice
// ---------------------------------------------------------------------------

/// A minimal token used only by the transform sub-parser. The formula lexer
/// hands the raw character span between the opening and closing parenthesis
/// of an `I(...)`/`fn(...)` call to [`parse_transform_arith`] or
/// [`parse_bare_call`].
///
/// Whitelist enforcement is by *parsed construct*, never by surface syntax:
/// only the operators/functions enumerated here are accepted; everything else
/// (including `poly`, `scale`, two-argument `log(x, base)`, `%in%`, …) returns
/// an actionable [`FormulaError`] so [`super::parser`] keeps refusing.
pub fn parse_transform_arith(src: &str) -> std::result::Result<Expr, FormulaError> {
    let toks = lex(src)?;
    let mut p = TParser {
        toks,
        pos: 0,
        depth: 0,
    };
    let e = p.parse_expr(0)?;
    if p.pos != p.toks.len() {
        return Err(refuse(src));
    }
    Ok(e)
}

/// Parse a bare pointwise call `fn(arg)` (outside `I()`), where `arg` may be
/// a column, a nested whitelisted call, or `I(...)`.
pub fn parse_bare_call(name: &str, arg_src: &str) -> std::result::Result<Expr, FormulaError> {
    let Some(f) = TransformFn::from_name(name) else {
        return Err(refuse(&format!("{name}(…)")));
    };
    let arg = parse_call_argument(arg_src)?;
    Ok(Expr::Call(f, Box::new(arg)))
}

/// Parse the single argument of a whitelisted call. The argument is either an
/// `I(...)` arithmetic expression, a nested whitelisted call, or a bare
/// column reference. A comma anywhere (i.e. a second argument such as
/// `log(x, base)`) is forbidden.
fn parse_call_argument(src: &str) -> std::result::Result<Expr, FormulaError> {
    let trimmed = src.trim();
    if trimmed.contains(',') {
        return Err(FormulaError::Other(format!(
            "multi-argument call `…({trimmed})` is not a stateless pointwise \
             transform and is out of scope for the engine — base changes like \
             `log(x, base)` are stateful; precompute the column or handle it \
             in the host wrapper"
        )));
    }
    let toks = lex(trimmed)?;
    let mut p = TParser {
        toks,
        pos: 0,
        depth: 0,
    };
    let e = p.parse_primary()?;
    if p.pos != p.toks.len() {
        // e.g. `log(x + 1)` written without the I() wrapper — arithmetic
        // outside I() is not part of the subset.
        return Err(FormulaError::Other(format!(
            "argument `{trimmed}` to a pointwise transform must be a column, \
             a nested whitelisted call, or an `I(...)` arithmetic expression \
             — wrap arithmetic in `I(...)` (e.g. `sqrt(I(x + 1))`)"
        )));
    }
    Ok(e)
}

fn refuse(construct: &str) -> FormulaError {
    FormulaError::Other(format!(
        "in-formula construct `{construct}` is not in the engine's stateless \
         transform subset (allowed: `I(<+ - * / ^, unary -, parens, \
         literals, columns>)` and pointwise `log`/`log2`/`log10`/`exp`/`sqrt`\
         /`abs`). Stateful transforms (`poly`, `scale`, `ns`, `bs`, `cut`, \
         `factor`, `center`, …) carry fitting-time state and must be \
         precomputed as data columns or handled by the host wrapper."
    ))
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Op(BinOp),
    LParen,
    RParen,
    Call(TransformFn),
    /// An `I(` opener (the inner arithmetic follows up to the matching `)`).
    IOpen,
}

fn lex(src: &str) -> std::result::Result<Vec<Tok>, FormulaError> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '+' => {
                out.push(Tok::Op(BinOp::Add));
                i += 1;
            }
            '-' => {
                out.push(Tok::Op(BinOp::Sub));
                i += 1;
            }
            '*' => {
                out.push(Tok::Op(BinOp::Mul));
                i += 1;
            }
            '/' => {
                out.push(Tok::Op(BinOp::Div));
                i += 1;
            }
            '^' => {
                out.push(Tok::Op(BinOp::Pow));
                i += 1;
            }
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            ',' => {
                return Err(FormulaError::Other(
                    "multi-argument calls are not stateless pointwise transforms \
                     and are out of scope for the engine — precompute the column \
                     or handle it in the host wrapper"
                        .to_string(),
                ));
            }
            '`' => {
                // Backtick-quoted column name: verbatim until the next backtick.
                let start = i + 1;
                let mut j = start;
                while j < chars.len() && chars[j] != '`' {
                    j += 1;
                }
                if j >= chars.len() {
                    return Err(FormulaError::Other(
                        "unterminated backtick-quoted identifier inside transform".to_string(),
                    ));
                }
                out.push(Tok::Ident(chars[start..j].iter().collect()));
                i = j + 1;
            }
            '0'..='9' | '.' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E'
                        || ((chars[i] == '+' || chars[i] == '-')
                            && i > start
                            && (chars[i - 1] == 'e' || chars[i - 1] == 'E')))
                {
                    i += 1;
                }
                let lit: String = chars[start..i].iter().collect();
                let v = lit.parse::<f64>().map_err(|_| {
                    FormulaError::Other(format!("invalid numeric literal `{lit}` in transform"))
                })?;
                out.push(Tok::Num(v));
            }
            _ if c.is_alphabetic() || c == '_' || c == '.' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '.')
                {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                // Is this an `I(` or `fn(` opener? Look past whitespace for `(`.
                let mut k = i;
                while k < chars.len() && chars[k].is_ascii_whitespace() {
                    k += 1;
                }
                if k < chars.len() && chars[k] == '(' {
                    if word == "I" {
                        out.push(Tok::IOpen);
                        out.push(Tok::LParen);
                        i = k + 1;
                    } else if let Some(f) = TransformFn::from_name(&word) {
                        out.push(Tok::Call(f));
                        out.push(Tok::LParen);
                        i = k + 1;
                    } else {
                        return Err(refuse(&format!("{word}(…)")));
                    }
                } else {
                    out.push(Tok::Ident(word));
                }
            }
            other => {
                return Err(FormulaError::Other(format!(
                    "unexpected character `{other}` in in-formula transform; \
                     allowed inside `I(...)`: `+ - * / ^`, unary `-`, \
                     parentheses, numeric literals, and column references"
                )));
            }
        }
    }
    Ok(out)
}

/// Maximum nesting depth of an in-formula transform expression.
///
/// Recursive descent here (and the subsequent `write_expr`/`eval_row` walks
/// over the produced tree) recurses once per nesting level, so an adversarial
/// or typo'd formula such as `I(((((…)))))` or `sqrt(sqrt(…))` would otherwise
/// overflow the stack and **abort the process** — uncatchable by
/// `catch_unwind`, so a host wrapper passing untrusted formulas could not
/// defend. Capping parse depth bounds the produced `Expr` tree, which in turn
/// bounds every later recursive walk over it. Real transforms nest only a
/// handful deep (e.g. `sqrt(I(log(x) + a*b))` is depth ~5); 64 is far beyond
/// any legitimate use yet a trivial amount of stack.
const MAX_TRANSFORM_DEPTH: usize = 64;

struct TParser {
    toks: Vec<Tok>,
    pos: usize,
    /// Current recursive-descent nesting depth (see [`MAX_TRANSFORM_DEPTH`]).
    depth: usize,
}

impl TParser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    /// Bump the recursion-depth counter, refusing once the budget is
    /// exhausted. Paired with [`Self::leave`]; every recursive grammar entry
    /// point (`parse_expr`, `parse_primary`) brackets its body between the two
    /// so no descent path can overflow the stack.
    fn enter(&mut self) -> std::result::Result<(), FormulaError> {
        self.depth += 1;
        if self.depth > MAX_TRANSFORM_DEPTH {
            return Err(FormulaError::Other(format!(
                "in-formula transform nesting exceeds the maximum supported \
                 depth ({MAX_TRANSFORM_DEPTH}); deeply nested \
                 parentheses/calls are rejected to bound parser recursion"
            )));
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.depth -= 1;
    }

    /// Pratt parser over the arithmetic grammar. `min_prec` is the minimum
    /// binding power this call will consume.
    fn parse_expr(&mut self, min_prec: u8) -> std::result::Result<Expr, FormulaError> {
        self.enter()?;
        let r = self.parse_expr_inner(min_prec);
        self.leave();
        r
    }

    fn parse_expr_inner(&mut self, min_prec: u8) -> std::result::Result<Expr, FormulaError> {
        let mut lhs = self.parse_unary()?;
        while let Some(Tok::Op(op)) = self.peek().cloned() {
            let p = op.precedence();
            if p < min_prec {
                break;
            }
            self.bump();
            // `^` is right-associative; the others left-associative.
            let next_min = if op == BinOp::Pow { p } else { p + 1 };
            let rhs = self.parse_expr(next_min)?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> std::result::Result<Expr, FormulaError> {
        if let Some(Tok::Op(BinOp::Sub)) = self.peek() {
            self.bump();
            // R precedence: `^` binds tighter than unary `-`.
            // Parse the operand at the `^` precedence level so that `-x^2`
            // becomes `Neg(x^2)` = `-(x^2)`, matching R's evaluation order.
            // Recursively calling parse_unary here would give `(-x)^2` which
            // is wrong (R: `-2^2` == -4, not +4).
            let inner = self.parse_expr(BinOp::Pow.precedence())?;
            return Ok(Expr::Neg(Box::new(inner)));
        }
        // Unary plus is a no-op; consume any run of leading `+` iteratively
        // so `+ + + … x` cannot recurse (and overflow) through parse_unary.
        while let Some(Tok::Op(BinOp::Add)) = self.peek() {
            self.bump();
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> std::result::Result<Expr, FormulaError> {
        self.enter()?;
        let r = self.parse_primary_inner();
        self.leave();
        r
    }

    fn parse_primary_inner(&mut self) -> std::result::Result<Expr, FormulaError> {
        match self.bump() {
            Some(Tok::Num(v)) => Ok(Expr::Lit(v)),
            Some(Tok::Ident(name)) => Ok(Expr::Col(name)),
            Some(Tok::LParen) => {
                let e = self.parse_expr(0)?;
                match self.bump() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err(FormulaError::Other(
                        "unbalanced parentheses in in-formula transform".to_string(),
                    )),
                }
            }
            Some(Tok::IOpen) => {
                // `I(` consumes the following `LParen`, an arithmetic
                // expression, and the matching `RParen`.
                match self.bump() {
                    Some(Tok::LParen) => {}
                    _ => {
                        return Err(FormulaError::Other(
                            "malformed `I(...)` in in-formula transform".to_string(),
                        ))
                    }
                }
                // Detect empty `I()` / `I(   )` before trying to parse an
                // expression, so the error is actionable rather than generic.
                if let Some(Tok::RParen) = self.peek() {
                    return Err(FormulaError::Other(
                        "empty `I(...)` — expected an arithmetic expression inside \
                         the parentheses (e.g. `I(x^2)`, `I(a*b)`, `I(1/x)`)"
                            .to_string(),
                    ));
                }
                let e = self.parse_expr(0)?;
                match self.bump() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err(FormulaError::Other(
                        "unbalanced parentheses in `I(...)`".to_string(),
                    )),
                }
            }
            Some(Tok::Call(f)) => {
                match self.bump() {
                    Some(Tok::LParen) => {}
                    _ => {
                        return Err(FormulaError::Other(
                            "malformed pointwise call in in-formula transform".to_string(),
                        ))
                    }
                }
                // The argument is itself a primary (column / nested call /
                // I(...)); arithmetic must be wrapped in I(...).
                let arg = self.parse_primary()?;
                match self.bump() {
                    Some(Tok::RParen) => Ok(Expr::Call(f, Box::new(arg))),
                    _ => Err(FormulaError::Other(format!(
                        "pointwise `{}` takes exactly one argument (a column, a \
                         nested whitelisted call, or `I(...)`); a second \
                         argument is stateful and out of scope",
                        f.label()
                    ))),
                }
            }
            other => Err(FormulaError::Other(format!(
                "unexpected token {other:?} in in-formula transform"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn df() -> DataFrame {
        let mut d = DataFrame::new();
        d.add_numeric("x", vec![1.0, 2.0, 3.0]).unwrap();
        d.add_numeric("days", vec![0.0, 1.0, 2.0]).unwrap();
        d.add_numeric("a", vec![2.0, 3.0, 4.0]).unwrap();
        d.add_numeric("b", vec![5.0, 6.0, 7.0]).unwrap();
        d.add_numeric("reaction", vec![10.0, 20.0, 30.0]).unwrap();
        d.add_categorical("g", vec!["a".into(), "b".into(), "a".into()])
            .unwrap();
        d
    }

    #[test]
    fn parse_and_label_power() {
        let e = parse_transform_arith("days^2").unwrap();
        assert_eq!(canonical_label(&e), "I(days^2)");
    }

    #[test]
    fn deeply_nested_parens_are_refused_not_aborted() {
        // Regression for B1/B2 (mote bd-01KRXCQ8BQZMP51GMYB0BPP0C9):
        // pathological nesting must return an actionable FormulaError, not
        // overflow the stack and abort the process (uncatchable). Use a depth
        // well past MAX_TRANSFORM_DEPTH; the parser must bail early and cheap.
        let n = MAX_TRANSFORM_DEPTH + 50;
        let src = format!("{}x{}", "(".repeat(n), ")".repeat(n));
        let err = parse_transform_arith(&src).expect_err("must refuse, not abort");
        match err {
            FormulaError::Other(m) => assert!(
                m.contains("maximum supported") && m.contains("depth"),
                "unexpected message: {m}"
            ),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn deeply_nested_calls_are_refused_not_aborted() {
        let n = MAX_TRANSFORM_DEPTH + 50;
        let src = format!("{}x{}", "sqrt(".repeat(n), ")".repeat(n));
        let err = parse_transform_arith(&src).expect_err("must refuse, not abort");
        assert!(matches!(err, FormulaError::Other(_)));
    }

    #[test]
    fn nesting_within_budget_still_parses() {
        // A realistic transform nests only a few deep; ensure the guard does
        // not regress legitimate use.
        let e = parse_transform_arith("((x + 1) * (a - b)) / 2").unwrap();
        assert_eq!(canonical_label(&e), "I((x+1)*(a-b)/2)");
        // Exactly at a comfortable depth (well under the cap) must succeed.
        let n = 16;
        let src = format!("{}x{}", "(".repeat(n), ")".repeat(n));
        assert!(parse_transform_arith(&src).is_ok());
    }

    #[test]
    fn parse_and_label_product_and_div_and_neg() {
        assert_eq!(
            canonical_label(&parse_transform_arith("a*b").unwrap()),
            "I(a*b)"
        );
        assert_eq!(
            canonical_label(&parse_transform_arith("1/x").unwrap()),
            "I(1/x)"
        );
        assert_eq!(
            canonical_label(&parse_transform_arith("-x").unwrap()),
            "I(-x)"
        );
    }

    #[test]
    fn canonical_label_drops_whitespace_and_trailing_zero() {
        let e = parse_transform_arith("  x  +  1  ").unwrap();
        assert_eq!(canonical_label(&e), "I(x+1)");
    }

    #[test]
    fn nested_parens_precedence() {
        let e = parse_transform_arith("(a+b)*x").unwrap();
        assert_eq!(canonical_label(&e), "I((a+b)*x)");
        let e2 = parse_transform_arith("a+b*x").unwrap();
        assert_eq!(canonical_label(&e2), "I(a+b*x)");
    }

    #[test]
    fn bare_call_log_reaction() {
        let e = parse_bare_call("log", "reaction").unwrap();
        assert_eq!(canonical_label(&e), "log(reaction)");
    }

    #[test]
    fn nested_call_sqrt_of_i() {
        let e = parse_bare_call("sqrt", "I(x+1)").unwrap();
        assert_eq!(canonical_label(&e), "sqrt(I(x+1))");
        let vals = eval(&e, &df()).unwrap();
        assert_eq!(vals, vec![2f64.sqrt(), 3f64.sqrt(), 4f64.sqrt()]);
    }

    #[test]
    fn backtick_identifier_in_transform() {
        let mut d = DataFrame::new();
        d.add_numeric("odd name", vec![4.0, 9.0]).unwrap();
        let e = parse_bare_call("sqrt", "`odd name`").unwrap();
        assert_eq!(eval(&e, &d).unwrap(), vec![2.0, 3.0]);
    }

    #[test]
    fn evaluator_numeric_correctness() {
        let d = df();
        let e = parse_transform_arith("days^2").unwrap();
        assert_eq!(eval(&e, &d).unwrap(), vec![0.0, 1.0, 4.0]);
        let e = parse_transform_arith("a*b").unwrap();
        assert_eq!(eval(&e, &d).unwrap(), vec![10.0, 18.0, 28.0]);
        let e = parse_transform_arith("1/x").unwrap();
        assert_eq!(eval(&e, &d).unwrap(), vec![1.0, 0.5, 1.0 / 3.0]);
        let e = parse_transform_arith("-x").unwrap();
        assert_eq!(eval(&e, &d).unwrap(), vec![-1.0, -2.0, -3.0]);
        let e = parse_bare_call("log", "reaction").unwrap();
        assert_eq!(
            eval(&e, &d).unwrap(),
            vec![10f64.ln(), 20f64.ln(), 30f64.ln()]
        );
    }

    #[test]
    fn pow_is_right_associative() {
        // 2^3^2 == 2^(3^2) == 512, not (2^3)^2 == 64.
        let mut d = DataFrame::new();
        d.add_numeric("two", vec![2.0]).unwrap();
        let e = parse_transform_arith("two^3^2").unwrap();
        assert_eq!(eval(&e, &d).unwrap(), vec![512.0]);
    }

    #[test]
    fn error_on_missing_column() {
        let err = eval(&parse_transform_arith("nope+1").unwrap(), &df()).unwrap_err();
        match err {
            MixedModelError::InvalidArgument(m) => {
                assert!(m.contains("`nope`"), "got {m}");
                assert!(m.contains("not"), "got {m}");
            }
            o => panic!("unexpected {o:?}"),
        }
    }

    #[test]
    fn error_on_categorical_in_arithmetic() {
        let err = eval(&parse_transform_arith("g+1").unwrap(), &df()).unwrap_err();
        match err {
            MixedModelError::InvalidArgument(m) => {
                assert!(m.contains("categorical"), "got {m}");
                assert!(m.contains("`g`"), "got {m}");
            }
            o => panic!("unexpected {o:?}"),
        }
    }

    #[test]
    fn error_on_non_finite_result() {
        let mut d = DataFrame::new();
        d.add_numeric("z", vec![1.0, -1.0]).unwrap();
        let dc = DerivedColumn::new(parse_bare_call("sqrt", "z").unwrap());
        let err = materialize_column(&dc, &d).unwrap_err();
        match err {
            MixedModelError::InvalidArgument(m) => {
                assert!(m.contains("non-finite"), "got {m}");
                assert!(m.contains("sqrt(z)"), "got {m}");
            }
            o => panic!("unexpected {o:?}"),
        }
    }

    #[test]
    fn forbidden_constructs_are_refused() {
        for src in ["poly(x,2)", "scale(x)", "ns(x,3)", "bs(x)", "cut(x,3)"] {
            assert!(
                parse_transform_arith(src).is_err(),
                "{src} should be refused"
            );
        }
        // Two-argument log inside an argument position.
        assert!(parse_bare_call("log", "x, 2").is_err());
        // Unknown function.
        assert!(parse_bare_call("frobnicate", "x").is_err());
        // Non-whitelisted operator surfaces from the lexer.
        assert!(parse_transform_arith("x %in% y").is_err());
    }

    // ── Unary-minus precedence (R semantics) ────────────────────────────────

    #[test]
    fn unary_minus_lower_precedence_than_power() {
        // R: `-x^2` is `-(x^2)`, not `(-x)^2`.
        // With x=3: -(3^2) = -9, not (-3)^2 = +9.
        let d = df(); // x = [1, 2, 3]
        let e = parse_transform_arith("-x^2").unwrap();
        let vals = eval(&e, &d).unwrap();
        assert_eq!(
            vals,
            vec![-1.0, -4.0, -9.0],
            "`-x^2` must be `-(x^2)`, got {vals:?}"
        );
    }

    #[test]
    fn unary_minus_neg_literal_power() {
        // R: `-2^2` == -(2^2) == -4, not (-2)^2 == +4.
        let mut d = DataFrame::new();
        d.add_numeric("dummy", vec![0.0]).unwrap();
        // Use a literal: `-2^2` should evaluate to -4. But since our
        // expression only uses column refs and literals, build it manually.
        let e = parse_transform_arith("-2^2").unwrap();
        // Evaluate against any single-row frame — no column refs needed.
        let result = eval_row(&e, &d, 0).unwrap();
        assert_eq!(result, -4.0, "`-2^2` must be -4, got {result}");
    }

    #[test]
    fn unary_minus_power_canonical_label_round_trips() {
        // `-x^2` must round-trip to `I(-x^2)`, NOT `I((-x)^2)`.
        // The byte-identical-to-R label promise requires this.
        let e = parse_transform_arith("-x^2").unwrap();
        assert_eq!(
            canonical_label(&e),
            "I(-x^2)",
            "label must be `I(-x^2)` (R-style), got `{}`",
            canonical_label(&e)
        );

        // Also verify the AST shape: Neg(Bin(Pow, Col(x), Lit(2))).
        match &e {
            Expr::Neg(inner) => match inner.as_ref() {
                Expr::Bin(BinOp::Pow, lhs, rhs) => {
                    assert!(matches!(lhs.as_ref(), Expr::Col(n) if n == "x"));
                    assert!(matches!(rhs.as_ref(), Expr::Lit(v) if *v == 2.0));
                }
                other => panic!("expected Bin(Pow, x, 2), got {other:?}"),
            },
            other => panic!("expected Neg(...), got {other:?}"),
        }
    }

    #[test]
    fn explicit_paren_neg_base_parses_and_labels_correctly() {
        // `(-x)^2` with explicit parens must still work and label as
        // `I((-x)^2)` (the extra parens are user-supplied, not by the
        // canonical formatter — but the formatter should add them because
        // the Neg is in the base of a Pow).
        let e = parse_transform_arith("(-x)^2").unwrap();
        // AST: Bin(Pow, Neg(Col(x)), Lit(2))
        match &e {
            Expr::Bin(BinOp::Pow, base, exp_) => {
                assert!(matches!(base.as_ref(), Expr::Neg(_)));
                assert!(matches!(exp_.as_ref(), Expr::Lit(v) if *v == 2.0));
            }
            other => panic!("expected Bin(Pow, Neg(x), 2), got {other:?}"),
        }
        // With x=3: (-3)^2 = 9.
        let d = df();
        let vals = eval(&e, &d).unwrap();
        assert_eq!(
            vals,
            vec![1.0, 4.0, 9.0],
            "`(-x)^2` must be +9 for x=3, got {vals:?}"
        );
        // Label must add parens around Neg when it is the base of Pow.
        assert_eq!(
            canonical_label(&e),
            "I((-x)^2)",
            "got `{}`",
            canonical_label(&e)
        );
    }

    // ── Collision policy ────────────────────────────────────────────────────

    #[test]
    fn collision_with_correct_precomputed_column_is_accepted() {
        use crate::formula::parse_formula;

        // Build a DataFrame that already contains `I(x^2)` with the right values.
        let mut data = DataFrame::new();
        data.add_numeric("x", vec![1.0, 2.0, 3.0]).unwrap();
        data.add_numeric("y", vec![10.0, 20.0, 30.0]).unwrap();
        data.add_categorical("g", vec!["a".into(), "b".into(), "c".into()])
            .unwrap();
        // Pre-supply the exact correct values (1, 4, 9).
        data.add_numeric("I(x^2)", vec![1.0, 4.0, 9.0]).unwrap();

        let formula = parse_formula("y ~ x + I(x^2) + (1 | g)").unwrap();
        // materialize should succeed — values agree.
        let out = formula.materialize(&data).unwrap();
        assert_eq!(out.numeric("I(x^2)").unwrap(), &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn collision_with_wrong_precomputed_column_is_rejected() {
        use crate::formula::parse_formula;

        let mut data = DataFrame::new();
        data.add_numeric("x", vec![1.0, 2.0, 3.0]).unwrap();
        data.add_numeric("y", vec![10.0, 20.0, 30.0]).unwrap();
        data.add_categorical("g", vec!["a".into(), "b".into(), "c".into()])
            .unwrap();
        // Pre-supply wrong values (1, 5, 9 instead of 1, 4, 9).
        data.add_numeric("I(x^2)", vec![1.0, 5.0, 9.0]).unwrap();

        let formula = parse_formula("y ~ x + I(x^2) + (1 | g)").unwrap();
        let err = formula.materialize(&data).unwrap_err();
        match err {
            crate::error::MixedModelError::InvalidArgument(m) => {
                assert!(m.contains("I(x^2)"), "message should name the label: {m}");
                assert!(
                    m.contains("engine"),
                    "message should mention engine ownership: {m}"
                );
            }
            o => panic!("expected InvalidArgument, got {o:?}"),
        }
    }

    // ── Empty I() actionable error ───────────────────────────────────────────

    #[test]
    fn empty_i_gives_actionable_error() {
        let err = parse_transform_arith("I()").unwrap_err();
        match err {
            FormulaError::Other(m) => {
                assert!(m.contains("empty"), "message should say 'empty': {m}");
                assert!(m.contains("I("), "message should name I(...): {m}");
            }
            o => panic!("expected Other, got {o:?}"),
        }
        // Whitespace-only is the same as empty.
        let err2 = parse_transform_arith("I(   )").unwrap_err();
        match err2 {
            FormulaError::Other(m) => {
                assert!(
                    m.contains("empty"),
                    "whitespace-only I() should say 'empty': {m}"
                );
            }
            o => panic!("expected Other, got {o:?}"),
        }
    }

    // ── Numeric literal canonical labels ─────────────────────────────────────

    #[test]
    fn integral_literal_labels_no_trailing_dot_zero() {
        // Small integer: printed as integer (no `.0`).
        assert_eq!(
            canonical_label(&parse_transform_arith("x+1").unwrap()),
            "I(x+1)"
        );
        assert_eq!(
            canonical_label(&parse_transform_arith("x+0").unwrap()),
            "I(x+0)"
        );
        // Large integral value inside 1e15 threshold.
        assert_eq!(
            canonical_label(&parse_transform_arith("x+1000").unwrap()),
            "I(x+1000)"
        );
        // Negative literal: -1 inside I.
        assert_eq!(
            canonical_label(&parse_transform_arith("x + -1").unwrap()),
            "I(x+-1)"
        );
    }

    #[test]
    fn large_integral_literal_round_trips() {
        // 1e14 is within the 1e15 threshold → prints as integer.
        let e = parse_transform_arith("1e14").unwrap();
        // 1e14 as i64 is 100_000_000_000_000.
        assert_eq!(canonical_label(&e), "I(100000000000000)");
    }

    #[test]
    fn scientific_notation_literal_round_trips_as_float() {
        // 1e300 is outside the integral-print range → uses float format.
        let e = parse_transform_arith("1e300").unwrap();
        let label = canonical_label(&e);
        // Should contain the exponential notation, not integer.
        assert!(
            label.contains('e') || label.contains('E') || label.starts_with("I(10"),
            "expected scientific notation for 1e300, got {label}"
        );
    }

    #[test]
    fn negative_literal_power_numeric_correctness() {
        // `-3^2` should compute as -(3^2) = -9 (R semantics), not (-3)^2 = +9.
        let mut d = DataFrame::new();
        d.add_numeric("dummy", vec![0.0]).unwrap();
        let e = parse_transform_arith("-3^2").unwrap();
        let result = eval_row(&e, &d, 0).unwrap();
        assert_eq!(
            result, -9.0,
            "`-3^2` must be -9 (R semantics), got {result}"
        );
    }
}
