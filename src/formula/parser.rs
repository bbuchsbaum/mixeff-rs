//! Recursive-descent parser for R/Julia-style mixed-model formulas.
//!
//! The entry point is [`parse_formula`], which accepts a string such as
//! `"y ~ 1 + x1 + (1 + x1 | group)"` and returns a [`Formula`] AST.
//!
//! # Supported syntax
//!
//! | Pattern | Meaning |
//! |---|---|
//! | `y ~ x1 + x2` | Fixed effects (implicit intercept) |
//! | `y ~ 1 + x1` | Explicit intercept |
//! | `y ~ 0 + x1` or `y ~ -1 + x1` | No intercept |
//! | `y ~ x1 + (1 \| group)` | Random intercept |
//! | `y ~ x1 + (1 + x1 \| group)` | Random intercept + slope |
//! | `y ~ x1 + (1 + x1 \|\| group)` | Zero-correlation parameterisation |
//! | `y ~ x1 + (1 \| g1) + (1 \| g2)` | Crossed random effects |
//! | `y ~ x1 + (1 \| g1 & g2)` | Interaction grouping factor |
//! | `y ~ x1 + x2:x3` | Interaction term |
//! | `y ~ x1 * x2` | Main effects + interaction (`x1 + x2 + x1:x2`) |
//! | `y ~ x1 / x2` | Nesting (`x1 + x1:x2`) |

use thiserror::Error;

use super::terms::{
    FixedTerm, Formula, GroupingFactor, RandomCovariance, RandomTerm, RandomTermExpansion,
    RandomTermSource,
};
use super::transform::{parse_bare_call, parse_transform_arith, DerivedColumn, TransformFn};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced while parsing a formula string.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FormulaError {
    /// The input string is empty.
    #[error("empty formula string")]
    Empty,

    /// The formula is missing the `~` separator between LHS and RHS.
    #[error("formula must contain '~' separating response from predictors")]
    MissingTilde,

    /// The left-hand side of `~` is empty or invalid.
    #[error("missing response variable on the left-hand side of '~'")]
    MissingResponse,

    /// An unexpected token was encountered during parsing.
    #[error("unexpected token '{0}' at position {1}")]
    UnexpectedToken(String, usize),

    /// A closing parenthesis was expected but not found.
    #[error("unmatched opening parenthesis — expected ')'")]
    UnmatchedParen,

    /// A random-effect term `(... | group)` is missing the `|` or `||` separator.
    #[error("random-effect term is missing '|' or '||' separator")]
    MissingBar,

    /// A random-effect term has an empty grouping factor.
    #[error("random-effect term has an empty grouping factor")]
    EmptyGrouping,

    /// A random-effect term has no model terms (left of `|`).
    #[error("random-effect term has no model terms before '|'")]
    EmptyRandomTerms,

    /// The right-hand side of `~` has no terms (e.g. `"y ~"`).
    #[error("formula has no terms on the right-hand side of '~'")]
    EmptyRhs,

    /// The formula ends with a dangling `+`/`-` operator (e.g. `"y ~ x +"`).
    #[error("formula ends with a dangling '+'/'-' operator at position {0}")]
    TrailingOperator(usize),

    /// Two terms appear without a `+`/`-` separator (e.g. `"y ~ (1|g) (1|h)"`).
    #[error("expected '+' or '-' separating model terms at position {0}")]
    MissingTermSeparator(usize),

    /// A bare numeric literal was used as a model term (e.g. `"y ~ 2 * x"`).
    /// Only the `0`/`1` intercept literals are meaningful.
    #[error(
        "numeric literal '{0}' at position {1} is not a valid model term \
         (only the 0/1 intercept literals are allowed)"
    )]
    NumericLiteralTerm(String, usize),

    /// `-` was applied to a random-effect block (e.g. `"y ~ x - (1|g)"`),
    /// which is not a supported term-removal target.
    #[error("'-' cannot remove a random-effect term at position {0}")]
    NegatedRandomEffect(usize),

    /// Generic parse error with a custom message.
    #[error("{0}")]
    Other(String),
}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Tilde,      // ~
    Plus,       // +
    Minus,      // -
    Star,       // *
    Colon,      // :
    Slash,      // /
    Pipe,       // |
    DoublePipe, // ||
    Ampersand,  // &
    LParen,     // (
    RParen,     // )
    Zero,       // 0  (literal digit)
    One,        // 1  (literal digit)
    Ident(String),
}

/// Position-annotated token.
#[derive(Debug, Clone)]
struct Spanned {
    token: Token,
    pos: usize,
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

/// Find the index just past the matching `)` for an opening `(` at
/// `open_idx`, skipping over backtick-quoted spans (which may legally contain
/// unbalanced parentheses as part of a column name).
fn matching_paren(chars: &[char], open_idx: usize) -> Result<usize, FormulaError> {
    debug_assert_eq!(chars[open_idx], '(');
    let mut depth = 0i32;
    let mut j = open_idx;
    while j < chars.len() {
        match chars[j] {
            '`' => {
                j += 1;
                while j < chars.len() && chars[j] != '`' {
                    j += 1;
                }
                if j >= chars.len() {
                    return Err(FormulaError::Other(
                        "unterminated backtick-quoted identifier inside an \
                         in-formula transform"
                            .to_string(),
                    ));
                }
            }
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(j + 1);
                }
            }
            _ => {}
        }
        j += 1;
    }
    Err(FormulaError::UnmatchedParen)
}

/// Tokenize the raw formula string, collecting any stateless in-formula
/// transforms into `derived`. Each transform call (`I(...)` or a whitelisted
/// pointwise `fn(...)`) is replaced by an `Ident` carrying its canonical
/// label, so the recursive-descent parser keeps treating it as a plain
/// column reference (the layered tower above the data boundary is unchanged).
fn tokenize(input: &str, derived: &mut Vec<DerivedColumn>) -> Result<Vec<Spanned>, FormulaError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        // Skip whitespace.
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        let pos = i;

        match c {
            '~' => {
                tokens.push(Spanned {
                    token: Token::Tilde,
                    pos,
                });
                i += 1;
            }
            '+' => {
                tokens.push(Spanned {
                    token: Token::Plus,
                    pos,
                });
                i += 1;
            }
            '-' => {
                tokens.push(Spanned {
                    token: Token::Minus,
                    pos,
                });
                i += 1;
            }
            '*' => {
                tokens.push(Spanned {
                    token: Token::Star,
                    pos,
                });
                i += 1;
            }
            ':' => {
                tokens.push(Spanned {
                    token: Token::Colon,
                    pos,
                });
                i += 1;
            }
            '/' => {
                tokens.push(Spanned {
                    token: Token::Slash,
                    pos,
                });
                i += 1;
            }
            '&' => {
                tokens.push(Spanned {
                    token: Token::Ampersand,
                    pos,
                });
                i += 1;
            }
            '`' => {
                // Backtick-quoted identifier: column names with spaces,
                // reserved words, or characters the bare lexer would split
                // (e.g. `` `my col` ``, `` `weird-name` ``, `` `2024` ``).
                // Content is taken verbatim; only the closing backtick ends it.
                let start = i + 1;
                let mut j = start;
                while j < len && chars[j] != '`' {
                    j += 1;
                }
                if j >= len {
                    return Err(FormulaError::Other(format!(
                        "unterminated backtick-quoted identifier starting at position {pos}"
                    )));
                }
                let name: String = chars[start..j].iter().collect();
                if name.is_empty() {
                    return Err(FormulaError::Other(format!(
                        "empty backtick-quoted identifier at position {pos}"
                    )));
                }
                tokens.push(Spanned {
                    token: Token::Ident(name),
                    pos,
                });
                i = j + 1;
            }
            '(' => {
                tokens.push(Spanned {
                    token: Token::LParen,
                    pos,
                });
                i += 1;
            }
            ')' => {
                tokens.push(Spanned {
                    token: Token::RParen,
                    pos,
                });
                i += 1;
            }
            '|' => {
                // Distinguish `|` from `||`.
                if i + 1 < len && chars[i + 1] == '|' {
                    tokens.push(Spanned {
                        token: Token::DoublePipe,
                        pos,
                    });
                    i += 2;
                } else {
                    tokens.push(Spanned {
                        token: Token::Pipe,
                        pos,
                    });
                    i += 1;
                }
            }
            '0' => {
                // If `0` is followed by an identifier character, treat as ident start.
                if i + 1 < len
                    && (chars[i + 1].is_alphanumeric()
                        || chars[i + 1] == '_'
                        || chars[i + 1] == '.')
                {
                    let start = i;
                    while i < len
                        && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '.')
                    {
                        i += 1;
                    }
                    let word: String = chars[start..i].iter().collect();
                    tokens.push(Spanned {
                        token: Token::Ident(word),
                        pos,
                    });
                } else {
                    tokens.push(Spanned {
                        token: Token::Zero,
                        pos,
                    });
                    i += 1;
                }
            }
            '1' => {
                // Same logic: bare `1` is a special token; `1abc` is an ident.
                if i + 1 < len
                    && (chars[i + 1].is_alphanumeric()
                        || chars[i + 1] == '_'
                        || chars[i + 1] == '.')
                {
                    let start = i;
                    while i < len
                        && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '.')
                    {
                        i += 1;
                    }
                    let word: String = chars[start..i].iter().collect();
                    tokens.push(Spanned {
                        token: Token::Ident(word),
                        pos,
                    });
                } else {
                    tokens.push(Spanned {
                        token: Token::One,
                        pos,
                    });
                    i += 1;
                }
            }
            _ if c.is_alphabetic() || c == '_' || c == '.' => {
                let start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '.')
                {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();

                // Is this identifier *immediately* function-applied, i.e.
                // `word(…)` with no intervening whitespace? In lme4 formulas
                // a bare identifier is never called except for an in-formula
                // transform, so an adjacent `(` is the unambiguous transform
                // seam. `I(...)` and the whitelisted pointwise functions are
                // lowered into a derived column; anything else (`poly(`,
                // `scale(`, `factor(`, …) is refused via the transform
                // whitelist. A *space* before `(` (e.g. `x1 (1|g)`) is two
                // terms, matching R — handled by the normal term path.
                if i < len && chars[i] == '(' && is_covariance_wrapper_name(&word) {
                    tokens.push(Spanned {
                        token: Token::Ident(word),
                        pos,
                    });
                } else if i < len && chars[i] == '(' {
                    let k = i;
                    let end = matching_paren(&chars, k)?;
                    let inner: String = chars[k + 1..end - 1].iter().collect();
                    let expr = if word == "I" {
                        parse_transform_arith(&inner)?
                    } else if TransformFn::from_name(&word).is_some() {
                        parse_bare_call(&word, &inner)?
                    } else {
                        // Unknown / stateful function: actionable refusal,
                        // naming the construct and pointing at the host
                        // wrapper / precompute.
                        return Err(FormulaError::Other(format!(
                            "in-formula construct `{word}(...)` at position \
                             {pos} is not in the engine's stateless transform \
                             subset (allowed: `I(<+ - * / ^, unary -, parens, \
                             literals, columns>)` and pointwise \
                             `log`/`log2`/`log10`/`exp`/`sqrt`/`abs`). \
                             Stateful transforms (`poly`, `scale`, `ns`, \
                             `bs`, `cut`, `factor`, `center`, …) carry \
                             fitting-time state and must be precomputed as \
                             data columns or handled by the host wrapper."
                        )));
                    };
                    let dc = DerivedColumn::new(expr);
                    let label = dc.label.clone();
                    if !derived.iter().any(|d| d.label == label) {
                        derived.push(dc);
                    }
                    tokens.push(Spanned {
                        token: Token::Ident(label),
                        pos,
                    });
                    i = end;
                } else {
                    tokens.push(Spanned {
                        token: Token::Ident(word),
                        pos,
                    });
                }
            }
            '2'..='9' => {
                // Digits 2-9 at the start of a token — treat as identifier start
                // (R allows variable names starting with digits in formulas via backticks,
                // but we keep it simple and reject bare digits other than 0/1).
                let start = i;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '.')
                {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                tokens.push(Spanned {
                    token: Token::Ident(word),
                    pos,
                });
            }
            '^' | '%' | '=' | '!' | '<' | '>' => {
                // Reached only for an operator that is *not* inside a
                // transform call (the inner span of `I(...)`/`fn(...)` is
                // consumed by the identifier branch above and never re-lexed
                // here). Bare formula-level arithmetic like `y ~ x^2` is not
                // part of the stateless subset: the subset requires the
                // explicit `I(...)` wrapper. Refuse with an actionable
                // message instead of a bare "unexpected token".
                return Err(FormulaError::Other(format!(
                    "unexpected '{c}' at position {pos}: bare formula-level \
                     arithmetic is not supported — wrap a stateless \
                     expression in `I(...)` (e.g. `I(x^2)`, `I(a*b)`, \
                     `I(1/x)`) or use a pointwise transform \
                     (`log`/`log2`/`log10`/`exp`/`sqrt`/`abs`, e.g. \
                     `sqrt(I(x + 1))`). Stateful transforms (`poly`, \
                     `scale`, `ns`, `bs`, `cut`, `factor`, `center`, …) \
                     carry fitting-time state and must be precomputed as \
                     data columns or handled by the host wrapper. If the \
                     column name itself contains unusual characters, quote \
                     it with backticks (e.g. `` `log x` ``)."
                )));
            }
            _ => {
                return Err(FormulaError::UnexpectedToken(c.to_string(), pos));
            }
        }
    }

    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Internal parser state.
struct Parser {
    tokens: Vec<Spanned>,
    cursor: usize,
    input: String,
}

impl Parser {
    fn new(tokens: Vec<Spanned>, input: &str) -> Self {
        Self {
            tokens,
            cursor: 0,
            input: input.to_string(),
        }
    }

    /// Peek at the current token without consuming it.
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.cursor).map(|s| &s.token)
    }

    /// Peek one token beyond the current token without consuming it.
    fn peek_next(&self) -> Option<&Token> {
        self.tokens.get(self.cursor + 1).map(|s| &s.token)
    }

    /// Current position (for error messages).
    fn pos(&self) -> usize {
        self.tokens
            .get(self.cursor)
            .map(|s| s.pos)
            .unwrap_or(self.tokens.last().map(|s| s.pos + 1).unwrap_or(0))
    }

    /// Advance and return the current token.
    fn advance(&mut self) -> Option<&Spanned> {
        if self.cursor < self.tokens.len() {
            let t = &self.tokens[self.cursor];
            self.cursor += 1;
            Some(t)
        } else {
            None
        }
    }

    /// Consume a token if it matches `expected`.
    fn expect(&mut self, expected: &Token) -> Result<(), FormulaError> {
        match self.peek() {
            Some(t) if t == expected => {
                self.advance();
                Ok(())
            }
            Some(t) => Err(FormulaError::UnexpectedToken(
                format!("{:?}", t),
                self.pos(),
            )),
            None => Err(FormulaError::UnexpectedToken(
                "end of input".to_string(),
                self.pos(),
            )),
        }
    }

    /// True when there are no more tokens.
    fn at_end(&self) -> bool {
        self.cursor >= self.tokens.len()
    }

    fn source_span(&self, start: usize, end_inclusive: usize) -> String {
        self.input
            .chars()
            .skip(start)
            .take(end_inclusive.saturating_sub(start) + 1)
            .collect::<String>()
            .trim()
            .to_string()
    }

    // ----- LHS -----

    /// Parse the response variable (LHS of `~`).
    fn parse_response(&mut self) -> Result<String, FormulaError> {
        match self.peek() {
            Some(Token::Ident(_)) => {
                if let Some(spanned) = self.advance() {
                    if let Token::Ident(ref name) = spanned.token {
                        return Ok(name.clone());
                    }
                }
                unreachable!()
            }
            _ => Err(FormulaError::MissingResponse),
        }
    }

    // ----- RHS: top-level term list -----

    /// Parse the entire RHS (after `~`), returning fixed and random terms.
    fn parse_rhs(&mut self) -> Result<(Vec<FixedTerm>, Vec<RandomTerm>), FormulaError> {
        let mut fixed: Vec<FixedTerm> = Vec::new();
        let mut random: Vec<RandomTerm> = Vec::new();

        // `-` is term removal at the top level (lme4 semantics): `- 1` /
        // `- 0` suppress the intercept, `- x` drops `x` from the term set.
        let mut negate = false;
        // True when the next token must be a term (start of input, or just
        // after a `+`/`-`). False once a term has been consumed: the next
        // token must then be a `+`/`-` separator or end of input.
        let mut expect_term = true;
        // True iff a `+`/`-` was the most recently consumed token, so a
        // dangling trailing operator can be detected at end of input.
        let mut pending_operator = false;

        loop {
            if self.at_end() {
                if pending_operator {
                    return Err(FormulaError::TrailingOperator(self.pos()));
                }
                break;
            }

            match self.peek() {
                Some(Token::Plus) => {
                    self.advance();
                    negate = false;
                    expect_term = true;
                    pending_operator = true;
                    continue;
                }
                Some(Token::Minus) => {
                    self.advance();
                    negate = true;
                    expect_term = true;
                    pending_operator = true;
                    continue;
                }
                Some(Token::LParen) => {
                    if !expect_term {
                        return Err(FormulaError::MissingTermSeparator(self.pos()));
                    }
                    if negate {
                        return Err(FormulaError::NegatedRandomEffect(self.pos()));
                    }
                    let rts = self.parse_random_term()?;
                    random.extend(rts);
                    expect_term = false;
                    pending_operator = false;
                }
                Some(Token::One) => {
                    if !expect_term {
                        return Err(FormulaError::MissingTermSeparator(self.pos()));
                    }
                    self.advance();
                    if negate {
                        // `-1` means suppress intercept.
                        fixed.push(FixedTerm::NoIntercept);
                    } else {
                        fixed.push(FixedTerm::Intercept);
                    }
                    negate = false;
                    expect_term = false;
                    pending_operator = false;
                }
                Some(Token::Zero) => {
                    if !expect_term {
                        return Err(FormulaError::MissingTermSeparator(self.pos()));
                    }
                    self.advance();
                    fixed.push(FixedTerm::NoIntercept);
                    negate = false;
                    expect_term = false;
                    pending_operator = false;
                }
                Some(Token::Ident(name))
                    if self.peek_next() == Some(&Token::LParen)
                        && is_covariance_wrapper_name(name) =>
                {
                    if !expect_term {
                        return Err(FormulaError::MissingTermSeparator(self.pos()));
                    }
                    if negate {
                        return Err(FormulaError::NegatedRandomEffect(self.pos()));
                    }
                    let source_start = self.pos();
                    let wrapper = if let Some(spanned) = self.advance() {
                        if let Token::Ident(name) = &spanned.token {
                            name.clone()
                        } else {
                            unreachable!()
                        }
                    } else {
                        unreachable!()
                    };
                    let covariance = covariance_wrapper(&wrapper).ok_or_else(|| {
                        FormulaError::Other(format!(
                            "unknown random-effect covariance wrapper `{wrapper}`"
                        ))
                    })?;
                    let rts = self.parse_random_term_with_covariance(source_start, covariance)?;
                    random.extend(rts);
                    negate = false;
                    expect_term = false;
                    pending_operator = false;
                }
                Some(Token::Ident(_)) => {
                    if !expect_term {
                        return Err(FormulaError::MissingTermSeparator(self.pos()));
                    }
                    let terms = self.parse_term_expr()?;
                    if negate {
                        // Top-level term removal (lme4 `-` semantics).
                        for t in &terms {
                            fixed.retain(|existing| existing != t);
                        }
                    } else {
                        fixed.extend(terms);
                    }
                    negate = false;
                    expect_term = false;
                    pending_operator = false;
                }
                Some(other) => {
                    return Err(FormulaError::UnexpectedToken(
                        format!("{:?}", other),
                        self.pos(),
                    ));
                }
                None => break,
            }
        }

        Ok((fixed, random))
    }

    // ----- Term expression: handles `*`, `:`, `/` between identifiers -----

    /// Parse one "term expression" which may involve `*`, `:`, or `/` operators.
    ///
    /// For example:
    /// - `x1` -> `[Column("x1")]`
    /// - `x1:x2` -> `[Interaction(["x1","x2"])]`
    /// - `x1*x2` -> `[Column("x1"), Column("x2"), Interaction(["x1","x2"])]`
    /// - `x1*x2*x3` -> all main effects, two-way interactions, and the
    ///   three-way interaction.
    /// - `x1/x2` -> `[Column("x1"), Interaction(["x1","x2"])]`  (nesting)
    fn parse_term_expr(&mut self) -> Result<Vec<FixedTerm>, FormulaError> {
        let first = self.parse_atom()?;

        match self.peek() {
            Some(Token::Colon) => {
                // a:b[:c...]
                let mut names = vec![first];
                while self.peek() == Some(&Token::Colon) {
                    self.advance();
                    names.push(self.parse_atom()?);
                }
                Ok(vec![FixedTerm::Interaction(names)])
            }
            Some(Token::Star) => {
                // a * b * c => all non-empty products in formula order.
                let mut names = vec![first];
                while self.peek() == Some(&Token::Star) {
                    self.advance();
                    names.push(self.parse_atom()?);
                }
                Ok(Self::expand_star_terms(&names))
            }
            Some(Token::Slash) => {
                // a / b / c => a + a:b + a:b:c (nesting path).
                let mut names = vec![first];
                let mut terms = vec![FixedTerm::Column(names[0].clone())];
                while self.peek() == Some(&Token::Slash) {
                    self.advance();
                    names.push(self.parse_atom()?);
                    terms.push(FixedTerm::Interaction(names.clone()));
                }
                Ok(terms)
            }
            _ => Ok(vec![FixedTerm::Column(first)]),
        }
    }

    /// Parse a single identifier (atom in a term expression).
    fn parse_atom(&mut self) -> Result<String, FormulaError> {
        match self.peek() {
            Some(Token::Ident(_)) => {
                if let Some(spanned) = self.advance() {
                    if let Token::Ident(ref name) = spanned.token {
                        // A bare numeric literal (`2`, `2.5`, `2e3`) tokenizes
                        // as an Ident but is not a valid model term; reject it
                        // at parse time instead of deferring to design build.
                        if name.parse::<f64>().is_ok() {
                            return Err(FormulaError::NumericLiteralTerm(
                                name.clone(),
                                spanned.pos,
                            ));
                        }
                        return Ok(name.clone());
                    }
                }
                unreachable!()
            }
            Some(other) => Err(FormulaError::UnexpectedToken(
                format!("{:?}", other),
                self.pos(),
            )),
            None => Err(FormulaError::UnexpectedToken(
                "end of input".to_string(),
                self.pos(),
            )),
        }
    }

    fn expand_star_terms(names: &[String]) -> Vec<FixedTerm> {
        let mut terms = Vec::new();
        for size in 1..=names.len() {
            let mut current = Vec::with_capacity(size);
            Self::append_star_combinations(names, size, 0, &mut current, &mut terms);
        }
        terms
    }

    fn append_star_combinations(
        names: &[String],
        size: usize,
        start: usize,
        current: &mut Vec<String>,
        terms: &mut Vec<FixedTerm>,
    ) {
        if current.len() == size {
            if size == 1 {
                terms.push(FixedTerm::Column(current[0].clone()));
            } else {
                terms.push(FixedTerm::Interaction(current.clone()));
            }
            return;
        }

        let remaining = size - current.len();
        for index in start..=names.len() - remaining {
            current.push(names[index].clone());
            Self::append_star_combinations(names, size, index + 1, current, terms);
            current.pop();
        }
    }

    // ----- Random term: (terms | grouping) -----

    /// Parse a random-effect specification `(terms | group)` or `(terms || group)`.
    ///
    /// Grouping expressions that imply multiple variance components are
    /// expanded immediately:
    /// - `(1 | a/b)` becomes `(1 | a) + (1 | a:b)`
    /// - `(1 | a*b)` becomes `(1 | a) + (1 | b) + (1 | a:b)`
    fn parse_random_term(&mut self) -> Result<Vec<RandomTerm>, FormulaError> {
        let source_start = self.pos();
        self.parse_random_term_with_covariance(source_start, RandomCovariance::Full)
    }

    fn parse_random_term_with_covariance(
        &mut self,
        source_start: usize,
        requested_covariance: RandomCovariance,
    ) -> Result<Vec<RandomTerm>, FormulaError> {
        self.expect(&Token::LParen)?;

        // Collect tokens inside the parentheses to find the bar position.
        let mut terms: Vec<FixedTerm> = Vec::new();
        let zerocorr;

        // Parse terms before `|` or `||`.
        // We need to handle `+`, `-`, `1`, `0`, identifiers, `:` within
        // the random effect specification.
        let mut negate = false;

        loop {
            match self.peek() {
                Some(Token::Pipe) => {
                    self.advance();
                    zerocorr = false;
                    break;
                }
                Some(Token::DoublePipe) => {
                    self.advance();
                    zerocorr = true;
                    break;
                }
                Some(Token::RParen) => {
                    return Err(FormulaError::MissingBar);
                }
                None => {
                    return Err(FormulaError::UnmatchedParen);
                }
                Some(Token::Plus) => {
                    self.advance();
                    negate = false;
                }
                Some(Token::Minus) => {
                    self.advance();
                    negate = true;
                }
                Some(Token::One) => {
                    self.advance();
                    if negate {
                        terms.push(FixedTerm::NoIntercept);
                    } else {
                        terms.push(FixedTerm::Intercept);
                    }
                    negate = false;
                }
                Some(Token::Zero) => {
                    self.advance();
                    terms.push(FixedTerm::NoIntercept);
                    negate = false;
                }
                Some(Token::Ident(_)) => {
                    let expr_terms = self.parse_term_expr()?;
                    terms.extend(expr_terms);
                    negate = false;
                }
                Some(other) => {
                    return Err(FormulaError::UnexpectedToken(
                        format!("{:?}", other),
                        self.pos(),
                    ));
                }
            }
        }

        if terms.is_empty() {
            return Err(FormulaError::EmptyRandomTerms);
        }
        if !terms
            .iter()
            .any(|term| matches!(term, FixedTerm::Intercept | FixedTerm::NoIntercept))
        {
            terms.insert(0, FixedTerm::Intercept);
        }
        let covariance = if requested_covariance == RandomCovariance::Full && zerocorr {
            RandomCovariance::Diagonal
        } else {
            requested_covariance
        };

        // Parse grouping factor(s), expanding nested/crossed shorthand.
        let parsed_grouping = self.parse_grouping()?;

        let source_end = self.pos();
        self.expect(&Token::RParen)?;
        let written = self.source_span(source_start, source_end);

        Ok(parsed_grouping
            .groupings
            .into_iter()
            .map(|grouping| RandomTerm {
                terms: terms.clone(),
                grouping,
                zerocorr,
                covariance,
                source: Some(RandomTermSource {
                    written: written.clone(),
                    expansion: parsed_grouping.expansion,
                }),
            })
            .collect())
    }

    /// Parse the grouping factor side of a random-effect term.
    ///
    /// Supported forms:
    /// - `g`
    /// - `g1 & g2` (legacy interaction syntax)
    /// - `g1:g2` (cell-level grouping)
    /// - `g1/g2[/g3...]` (nested expansion)
    /// - `g1*g2[*g3...]` (main effects plus interaction expansion)
    fn parse_grouping(&mut self) -> Result<ParsedGrouping, FormulaError> {
        let first = match self.peek() {
            Some(Token::Ident(_)) => self.parse_atom()?,
            _ => return Err(FormulaError::EmptyGrouping),
        };

        if self.peek() == Some(&Token::Ampersand) {
            let mut names = vec![first];
            while self.peek() == Some(&Token::Ampersand) {
                self.advance();
                names.push(self.parse_atom()?);
            }
            Ok(ParsedGrouping::new(vec![GroupingFactor::Interaction(
                names,
            )]))
        } else if self.peek() == Some(&Token::Colon) {
            let mut names = vec![first];
            while self.peek() == Some(&Token::Colon) {
                self.advance();
                names.push(self.parse_atom()?);
            }
            Ok(ParsedGrouping::new(vec![GroupingFactor::Cell(names)]))
        } else if self.peek() == Some(&Token::Slash) {
            let mut names = vec![first];
            while self.peek() == Some(&Token::Slash) {
                self.advance();
                names.push(self.parse_atom()?);
            }
            Ok(ParsedGrouping {
                groupings: expand_nested_grouping(&names),
                expansion: Some(RandomTermExpansion::NestedGrouping),
            })
        } else if self.peek() == Some(&Token::Star) {
            let mut names = vec![first];
            while self.peek() == Some(&Token::Star) {
                self.advance();
                names.push(self.parse_atom()?);
            }
            Ok(ParsedGrouping {
                groupings: expand_crossed_grouping(&names),
                expansion: Some(RandomTermExpansion::CrossedGrouping),
            })
        } else {
            Ok(ParsedGrouping::new(vec![GroupingFactor::Single(first)]))
        }
    }
}

fn is_covariance_wrapper_name(name: &str) -> bool {
    covariance_wrapper(name).is_some()
}

fn covariance_wrapper(name: &str) -> Option<RandomCovariance> {
    match name {
        "us" => Some(RandomCovariance::Full),
        "diag" => Some(RandomCovariance::Diagonal),
        "cs" => Some(RandomCovariance::CompoundSymmetry),
        "ar1" => Some(RandomCovariance::Ar1),
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct ParsedGrouping {
    groupings: Vec<GroupingFactor>,
    expansion: Option<RandomTermExpansion>,
}

impl ParsedGrouping {
    fn new(groupings: Vec<GroupingFactor>) -> Self {
        Self {
            groupings,
            expansion: None,
        }
    }
}

fn expand_nested_grouping(names: &[String]) -> Vec<GroupingFactor> {
    let mut result = Vec::new();
    for end in 1..=names.len() {
        if end == 1 {
            result.push(GroupingFactor::Single(names[0].clone()));
        } else {
            result.push(GroupingFactor::Cell(names[..end].to_vec()));
        }
    }
    result
}

fn expand_crossed_grouping(names: &[String]) -> Vec<GroupingFactor> {
    let mut result = Vec::new();
    for size in 1..=names.len() {
        append_combinations(names, size, 0, &mut Vec::new(), &mut result);
    }
    result
}

fn append_combinations(
    names: &[String],
    size: usize,
    start: usize,
    current: &mut Vec<String>,
    result: &mut Vec<GroupingFactor>,
) {
    if current.len() == size {
        if size == 1 {
            result.push(GroupingFactor::Single(current[0].clone()));
        } else {
            result.push(GroupingFactor::Cell(current.clone()));
        }
        return;
    }

    for idx in start..names.len() {
        current.push(names[idx].clone());
        append_combinations(names, size, idx + 1, current, result);
        current.pop();
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a mixed-model formula string into a [`Formula`] AST.
///
/// # Examples
///
/// ```
/// use mixeff_rs::formula::parser::parse_formula;
///
/// let f = parse_formula("y ~ x1 + (1 | group)").unwrap();
/// assert_eq!(f.response, "y");
/// assert_eq!(f.fixed_terms.len(), 2); // implicit intercept + x1
/// assert_eq!(f.random_terms.len(), 1);
/// ```
///
/// # Errors
///
/// Returns a [`FormulaError`] if the input cannot be parsed.
pub fn parse_formula(input: &str) -> Result<Formula, FormulaError> {
    let input = input.trim();
    if input.is_empty() {
        return Err(FormulaError::Empty);
    }

    let mut derived: Vec<DerivedColumn> = Vec::new();
    let tokens = tokenize(input, &mut derived)?;
    if tokens.is_empty() {
        return Err(FormulaError::Empty);
    }

    let mut parser = Parser::new(tokens, input);

    // --- LHS ---
    let response = parser.parse_response()?;

    // --- Tilde ---
    parser.expect(&Token::Tilde)?;

    // --- RHS ---
    let (mut fixed, random) = parser.parse_rhs()?;

    // Reject formulae with nothing on the RHS (e.g. "y ~"). Without this
    // check the implicit-intercept rule would silently canonicalize them
    // to "y ~ 1", which masks user typos and diverges from R/lme4.
    if fixed.is_empty() && random.is_empty() {
        return Err(FormulaError::EmptyRhs);
    }

    // --- Implicit intercept ---
    // If no explicit `Intercept` or `NoIntercept` was given in the fixed terms,
    // insert an implicit intercept at the front.
    let has_explicit_intercept = fixed
        .iter()
        .any(|t| matches!(t, FixedTerm::Intercept | FixedTerm::NoIntercept));

    if !has_explicit_intercept {
        fixed.insert(0, FixedTerm::Intercept);
    }

    // If NoIntercept was given, remove any Intercept tokens.
    let has_no_intercept = fixed.iter().any(|t| matches!(t, FixedTerm::NoIntercept));
    if has_no_intercept {
        fixed.retain(|t| !matches!(t, FixedTerm::Intercept | FixedTerm::NoIntercept));
    }

    // Keep only the derived columns actually referenced by the final term
    // set (a `-`-removed transform term should not materialize a column).
    let mut referenced: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    referenced.insert(response.as_str());
    for t in &fixed {
        match t {
            FixedTerm::Column(name) => {
                referenced.insert(name.as_str());
            }
            FixedTerm::Interaction(names) => {
                for n in names {
                    referenced.insert(n.as_str());
                }
            }
            FixedTerm::Intercept | FixedTerm::NoIntercept => {}
        }
    }
    for rt in &random {
        for t in &rt.terms {
            match t {
                FixedTerm::Column(name) => {
                    referenced.insert(name.as_str());
                }
                FixedTerm::Interaction(names) => {
                    for n in names {
                        referenced.insert(n.as_str());
                    }
                }
                FixedTerm::Intercept | FixedTerm::NoIntercept => {}
            }
        }
    }
    derived.retain(|d| referenced.contains(d.label.as_str()));

    Ok(Formula {
        response,
        fixed_terms: fixed,
        random_terms: random,
        derived,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::terms::{FixedTerm, GroupingFactor};

    // ---- Tokenizer tests ----

    /// Tokenizer test helper: discards collected derived columns.
    fn tokenize(input: &str) -> Result<Vec<Spanned>, FormulaError> {
        let mut derived = Vec::new();
        super::tokenize(input, &mut derived)
    }

    #[test]
    fn tokenize_simple() {
        let tokens = tokenize("y ~ 1 + x1").unwrap();
        let kinds: Vec<_> = tokens.iter().map(|s| &s.token).collect();
        assert_eq!(
            kinds,
            vec![
                &Token::Ident("y".into()),
                &Token::Tilde,
                &Token::One,
                &Token::Plus,
                &Token::Ident("x1".into()),
            ]
        );
    }

    #[test]
    fn tokenize_double_pipe() {
        let tokens = tokenize("(1 || g)").unwrap();
        let kinds: Vec<_> = tokens.iter().map(|s| &s.token).collect();
        assert!(kinds.contains(&&Token::DoublePipe));
        assert!(!kinds.contains(&&Token::Pipe));
    }

    #[test]
    fn tokenize_ampersand_grouping() {
        let tokens = tokenize("(1 | g1 & g2)").unwrap();
        let kinds: Vec<_> = tokens.iter().map(|s| &s.token).collect();
        assert!(kinds.contains(&&Token::Ampersand));
    }

    // ---- Fixed effects only ----

    #[test]
    fn fixed_effects_explicit_intercept() {
        let f = parse_formula("y ~ 1 + x1 + x2").unwrap();
        assert_eq!(f.response, "y");
        assert_eq!(
            f.fixed_terms,
            vec![
                FixedTerm::Intercept,
                FixedTerm::Column("x1".into()),
                FixedTerm::Column("x2".into()),
            ]
        );
        assert!(f.random_terms.is_empty());
    }

    #[test]
    fn fixed_effects_implicit_intercept() {
        let f = parse_formula("y ~ x1 + x2").unwrap();
        // Implicit intercept should be inserted.
        assert_eq!(f.fixed_terms[0], FixedTerm::Intercept);
        assert_eq!(f.fixed_terms.len(), 3);
    }

    #[test]
    fn empty_rhs_is_rejected() {
        // "y ~" used to canonicalize to "y ~ 1" via the implicit-intercept
        // rule. We now reject it to match R/lme4.
        assert!(matches!(parse_formula("y ~"), Err(FormulaError::EmptyRhs)));
        assert!(matches!(
            parse_formula("y ~   "),
            Err(FormulaError::EmptyRhs)
        ));
    }

    #[test]
    fn no_intercept_with_zero() {
        let f = parse_formula("y ~ 0 + x1 + x2").unwrap();
        // No intercept — the NoIntercept and Intercept are stripped.
        assert!(!f
            .fixed_terms
            .iter()
            .any(|t| matches!(t, FixedTerm::Intercept)));
        assert!(!f
            .fixed_terms
            .iter()
            .any(|t| matches!(t, FixedTerm::NoIntercept)));
        assert_eq!(
            f.fixed_terms,
            vec![
                FixedTerm::Column("x1".into()),
                FixedTerm::Column("x2".into()),
            ]
        );
    }

    #[test]
    fn no_intercept_with_minus_one() {
        let f = parse_formula("y ~ -1 + x1").unwrap();
        assert!(!f
            .fixed_terms
            .iter()
            .any(|t| matches!(t, FixedTerm::Intercept)));
        assert_eq!(f.fixed_terms, vec![FixedTerm::Column("x1".into())]);
    }

    // ---- Interaction ----

    #[test]
    fn interaction_colon() {
        let f = parse_formula("y ~ x1 + x2:x3").unwrap();
        assert!(f
            .fixed_terms
            .contains(&FixedTerm::Interaction(vec!["x2".into(), "x3".into()])));
    }

    #[test]
    fn star_expansion() {
        // x1 * x2 => x1 + x2 + x1:x2
        let f = parse_formula("y ~ x1 * x2").unwrap();
        assert!(f.fixed_terms.contains(&FixedTerm::Column("x1".into())));
        assert!(f.fixed_terms.contains(&FixedTerm::Column("x2".into())));
        assert!(f
            .fixed_terms
            .contains(&FixedTerm::Interaction(vec!["x1".into(), "x2".into()])));
    }

    #[test]
    fn three_way_star_expansion() {
        let f = parse_formula("y ~ A * B * C").unwrap();
        assert_eq!(
            f.fixed_terms,
            vec![
                FixedTerm::Intercept,
                FixedTerm::Column("A".into()),
                FixedTerm::Column("B".into()),
                FixedTerm::Column("C".into()),
                FixedTerm::Interaction(vec!["A".into(), "B".into()]),
                FixedTerm::Interaction(vec!["A".into(), "C".into()]),
                FixedTerm::Interaction(vec!["B".into(), "C".into()]),
                FixedTerm::Interaction(vec!["A".into(), "B".into(), "C".into()]),
            ]
        );
    }

    #[test]
    fn nesting_slash() {
        // x1 / x2 => x1 + x1:x2
        let f = parse_formula("y ~ x1 / x2").unwrap();
        assert!(f.fixed_terms.contains(&FixedTerm::Column("x1".into())));
        assert!(f
            .fixed_terms
            .contains(&FixedTerm::Interaction(vec!["x1".into(), "x2".into()])));
        // Should NOT contain a standalone Column("x2").
        assert!(!f.fixed_terms.contains(&FixedTerm::Column("x2".into())));
    }

    // ---- Random effects ----

    #[test]
    fn random_intercept() {
        let f = parse_formula("y ~ x1 + (1 | group)").unwrap();
        assert_eq!(f.random_terms.len(), 1);
        let rt = &f.random_terms[0];
        assert_eq!(rt.terms, vec![FixedTerm::Intercept]);
        assert_eq!(rt.grouping, GroupingFactor::Single("group".into()));
        assert!(!rt.zerocorr);
    }

    #[test]
    fn random_intercept_and_slope() {
        let f = parse_formula("y ~ x1 + x2 + (1 + x1 | group)").unwrap();
        assert_eq!(f.random_terms.len(), 1);
        let rt = &f.random_terms[0];
        assert_eq!(
            rt.terms,
            vec![FixedTerm::Intercept, FixedTerm::Column("x1".into())]
        );
        assert_eq!(rt.grouping, GroupingFactor::Single("group".into()));
        assert!(!rt.zerocorr);
    }

    #[test]
    fn random_slope_syntax_has_implicit_intercept() {
        let f = parse_formula("y ~ x1 + (x1 | group)").unwrap();
        let rt = &f.random_terms[0];
        assert_eq!(
            rt.terms,
            vec![FixedTerm::Intercept, FixedTerm::Column("x1".into())]
        );
        assert_eq!(
            rt.source.as_ref().map(|source| source.written.as_str()),
            Some("(x1 | group)")
        );
    }

    #[test]
    fn zerocorr() {
        let f = parse_formula("y ~ x1 + (1 + x1 || group)").unwrap();
        assert_eq!(f.random_terms.len(), 1);
        let rt = &f.random_terms[0];
        assert!(rt.zerocorr);
        assert_eq!(rt.covariance, RandomCovariance::Diagonal);
        assert_eq!(
            rt.terms,
            vec![FixedTerm::Intercept, FixedTerm::Column("x1".into())]
        );
    }

    #[test]
    fn covariance_wrappers_parse_as_random_terms() {
        let cases = [
            ("us(1 + x1 | group)", RandomCovariance::Full, false),
            ("diag(1 + x1 | group)", RandomCovariance::Diagonal, false),
            (
                "cs(1 + x1 | group)",
                RandomCovariance::CompoundSymmetry,
                false,
            ),
            ("ar1(0 + x1 | group)", RandomCovariance::Ar1, false),
        ];

        for (term, covariance, zerocorr) in cases {
            let f = parse_formula(&format!("y ~ x1 + {term}")).unwrap();
            assert_eq!(f.random_terms.len(), 1);
            let rt = &f.random_terms[0];
            assert_eq!(rt.covariance, covariance);
            assert_eq!(rt.zerocorr, zerocorr);
            assert_eq!(
                rt.source.as_ref().map(|source| source.written.as_str()),
                Some(term)
            );
        }
    }

    #[test]
    fn crossed_random_effects() {
        let f = parse_formula("y ~ x1 + (1 | g1) + (1 | g2)").unwrap();
        assert_eq!(f.random_terms.len(), 2);
        assert_eq!(
            f.random_terms[0].grouping,
            GroupingFactor::Single("g1".into())
        );
        assert_eq!(
            f.random_terms[1].grouping,
            GroupingFactor::Single("g2".into())
        );
    }

    #[test]
    fn interaction_grouping() {
        let f = parse_formula("y ~ x1 + (1 | g1 & g2)").unwrap();
        assert_eq!(f.random_terms.len(), 1);
        assert_eq!(
            f.random_terms[0].grouping,
            GroupingFactor::Interaction(vec!["g1".into(), "g2".into()])
        );
    }

    #[test]
    fn cell_grouping_with_colon() {
        let f = parse_formula("y ~ x1 + (1 | subject:item)").unwrap();
        assert_eq!(f.random_terms.len(), 1);
        assert_eq!(
            f.random_terms[0].grouping,
            GroupingFactor::Cell(vec!["subject".into(), "item".into()])
        );
        assert_eq!(f.to_string(), "y ~ 1 + x1 + (1 | subject:item)");
    }

    #[test]
    fn nested_grouping_expands_to_main_and_cell() {
        let f = parse_formula("y ~ x1 + (1 | school/class)").unwrap();
        assert_eq!(f.random_terms.len(), 2);
        assert_eq!(
            f.random_terms[0].grouping,
            GroupingFactor::Single("school".into())
        );
        assert_eq!(
            f.random_terms[1].grouping,
            GroupingFactor::Cell(vec!["school".into(), "class".into()])
        );
        assert_eq!(
            f.to_string(),
            "y ~ 1 + x1 + (1 | school) + (1 | school:class)"
        );
        assert!(f.random_terms.iter().all(|term| {
            term.source.as_ref().is_some_and(|source| {
                source.written == "(1 | school/class)"
                    && source.expansion == Some(RandomTermExpansion::NestedGrouping)
            })
        }));
    }

    #[test]
    fn crossed_star_grouping_expands_to_main_effects_and_cell() {
        let f = parse_formula("y ~ x1 + (1 | subject*item)").unwrap();
        assert_eq!(f.random_terms.len(), 3);
        assert_eq!(
            f.random_terms[0].grouping,
            GroupingFactor::Single("subject".into())
        );
        assert_eq!(
            f.random_terms[1].grouping,
            GroupingFactor::Single("item".into())
        );
        assert_eq!(
            f.random_terms[2].grouping,
            GroupingFactor::Cell(vec!["subject".into(), "item".into()])
        );
        assert_eq!(
            f.to_string(),
            "y ~ 1 + x1 + (1 | subject) + (1 | item) + (1 | subject:item)"
        );
        assert!(f.random_terms.iter().all(|term| {
            term.source.as_ref().is_some_and(|source| {
                source.written == "(1 | subject*item)"
                    && source.expansion == Some(RandomTermExpansion::CrossedGrouping)
            })
        }));
    }

    // ---- Edge cases ----

    #[test]
    fn no_fixed_intercept_with_random() {
        // y ~ 0 + x1 + (1|g)  — fixed intercept suppressed, random intercept present.
        let f = parse_formula("y ~ 0 + x1 + (1 | g)").unwrap();
        assert!(!f
            .fixed_terms
            .iter()
            .any(|t| matches!(t, FixedTerm::Intercept)));
        assert_eq!(f.fixed_terms, vec![FixedTerm::Column("x1".into())]);
        assert_eq!(f.random_terms.len(), 1);
    }

    #[test]
    fn only_random_term_implicit_intercept() {
        // y ~ (1|g) — fixed intercept is implicit.
        let f = parse_formula("y ~ (1 | g)").unwrap();
        assert_eq!(f.fixed_terms, vec![FixedTerm::Intercept]);
        assert_eq!(f.random_terms.len(), 1);
    }

    #[test]
    fn whitespace_handling() {
        let f1 = parse_formula("y~1+x1+(1|g)").unwrap();
        let f2 = parse_formula("  y  ~  1 +  x1  + ( 1 | g )  ").unwrap();
        assert_eq!(f1.response, f2.response);
        assert_eq!(f1.fixed_terms, f2.fixed_terms);
        assert_eq!(f1.random_terms.len(), f2.random_terms.len());
    }

    #[test]
    fn dotted_identifier() {
        let f = parse_formula("y.resp ~ x.pred + (1 | g.group)").unwrap();
        assert_eq!(f.response, "y.resp");
        assert!(f.fixed_terms.contains(&FixedTerm::Column("x.pred".into())));
        assert_eq!(
            f.random_terms[0].grouping,
            GroupingFactor::Single("g.group".into())
        );
    }

    #[test]
    fn underscored_identifier() {
        let f = parse_formula("y_resp ~ x_pred + (1 | g_group)").unwrap();
        assert_eq!(f.response, "y_resp");
        assert!(f.fixed_terms.contains(&FixedTerm::Column("x_pred".into())));
    }

    #[test]
    fn backtick_identifier_with_spaces_and_odd_chars() {
        let f = parse_formula("`reaction time` ~ `day of study` + (1 | `subject id`)").unwrap();
        assert_eq!(f.response, "reaction time");
        assert!(f
            .fixed_terms
            .contains(&FixedTerm::Column("day of study".into())));
        assert_eq!(
            f.random_terms[0].grouping,
            GroupingFactor::Single("subject id".into())
        );
    }

    #[test]
    fn backtick_identifier_allows_reserved_and_digit_starts() {
        // `2024` and a hyphenated name would not lex as bare identifiers.
        let f = parse_formula("y ~ `2024-cohort` + `x-1`").unwrap();
        assert!(f
            .fixed_terms
            .contains(&FixedTerm::Column("2024-cohort".into())));
        assert!(f.fixed_terms.contains(&FixedTerm::Column("x-1".into())));
    }

    #[test]
    fn unterminated_backtick_is_actionable_error() {
        let err = parse_formula("y ~ `oops + (1 | g)").unwrap_err();
        match err {
            FormulaError::Other(msg) => {
                assert!(msg.contains("unterminated backtick"), "got: {msg}");
            }
            other => panic!("expected Other, got {other:?}"),
        }
    }

    #[test]
    fn empty_backtick_is_rejected() {
        let err = parse_formula("y ~ `` + x").unwrap_err();
        match err {
            FormulaError::Other(msg) => {
                assert!(msg.contains("empty backtick"), "got: {msg}");
            }
            other => panic!("expected Other, got {other:?}"),
        }
    }

    #[test]
    fn in_formula_transformation_gives_actionable_refusal() {
        // Bare formula-level arithmetic / comparison operators are still
        // refused: the stateless subset requires the explicit `I(...)`
        // wrapper.
        for src in ["y ~ x^2", "y ~ x %in% g", "y ~ x > 0"] {
            let err = parse_formula(src).unwrap_err();
            match err {
                FormulaError::Other(msg) => {
                    assert!(
                        msg.contains("not supported") && msg.contains("precompute"),
                        "expected actionable transform refusal, got: {msg}"
                    );
                }
                other => panic!("expected Other for `{src}`, got {other:?}"),
            }
        }
    }

    #[test]
    fn stateful_transforms_are_refused_with_actionable_message() {
        // The line is drawn at *stateless*, by whitelist, not by surface
        // syntax. Stateful basis transforms and unknown functions carry
        // fitting-time state and must be refused with an actionable message
        // (naming the construct, pointing at precompute / host wrapper).
        for src in [
            "y ~ poly(x, 2) + (1 | g)",
            "y ~ scale(x) + (1 | g)",
            "y ~ ns(x, 3) + (1 | g)",
            "y ~ bs(x) + (1 | g)",
            "y ~ factor(g2) + (1 | g)",
            "y ~ cut(x, 3) + (1 | g)",
            "y ~ center(x) + (1 | g)",
            "y ~ frobnicate(x) + (1 | g)",
            "log(reaction, 2) ~ x + (1 | g)",
            "y ~ log(x, 2) + (1 | g)",
        ] {
            let err = parse_formula(src).unwrap_err();
            match err {
                FormulaError::Other(msg) => {
                    assert!(
                        (msg.contains("stateless") || msg.contains("stateful"))
                            && (msg.contains("precompute")
                                || msg.contains("host wrapper")
                                || msg.contains("out of scope")),
                        "expected actionable refusal for `{src}`, got: {msg}"
                    );
                }
                other => panic!("expected Other for `{src}`, got {other:?}"),
            }
        }
    }

    #[test]
    fn stateless_subset_parses_with_canonical_labels() {
        // `I(...)` arithmetic and pointwise calls on both sides.
        let f = parse_formula("log(reaction) ~ days + I(days^2) + (1 | subj)").unwrap();
        assert_eq!(f.response, "log(reaction)");
        assert!(f
            .fixed_terms
            .contains(&FixedTerm::Column("I(days^2)".into())));
        assert!(f.fixed_terms.contains(&FixedTerm::Column("days".into())));
        // Derived set carries the transformed response and the I() term.
        let labels: Vec<&str> = f.derived.iter().map(|d| d.label.as_str()).collect();
        assert!(labels.contains(&"log(reaction)"));
        assert!(labels.contains(&"I(days^2)"));
        assert_eq!(f.derived.len(), 2);

        // Canonical-label exactness for the supported shapes.
        for (src, label) in [
            ("y ~ I(a*b) + (1|g)", "I(a*b)"),
            ("y ~ I(1/x) + (1|g)", "I(1/x)"),
            ("y ~ I(-x) + (1|g)", "I(-x)"),
            ("y ~ I( x  +  1 ) + (1|g)", "I(x+1)"),
            ("y ~ sqrt(I(x+1)) + (1|g)", "sqrt(I(x+1))"),
            ("y ~ I((a+b)*x) + (1|g)", "I((a+b)*x)"),
        ] {
            let f = parse_formula(src).unwrap();
            assert!(
                f.fixed_terms.contains(&FixedTerm::Column(label.into())),
                "`{src}` should yield column `{label}`, got {:?}",
                f.fixed_terms
            );
        }

        // Backtick identifiers still work alongside transforms.
        let f = parse_formula("`reaction time` ~ I(`day of study`^2) + (1 | g)").unwrap();
        assert_eq!(f.response, "reaction time");
        assert!(f
            .fixed_terms
            .contains(&FixedTerm::Column("I(day of study^2)".into())));
    }

    #[test]
    fn triple_interaction() {
        let f = parse_formula("y ~ a:b:c").unwrap();
        assert!(f.fixed_terms.contains(&FixedTerm::Interaction(vec![
            "a".into(),
            "b".into(),
            "c".into(),
        ])));
    }

    #[test]
    fn random_slope_no_intercept() {
        let f = parse_formula("y ~ x + (0 + x | g)").unwrap();
        let rt = &f.random_terms[0];
        // NoIntercept should be present in the random terms.
        assert!(rt.terms.contains(&FixedTerm::NoIntercept));
    }

    // ---- Error cases ----

    #[test]
    fn error_empty() {
        assert!(parse_formula("").is_err());
    }

    #[test]
    fn error_no_tilde() {
        assert!(parse_formula("y + x1").is_err());
    }

    #[test]
    fn error_missing_response() {
        assert!(parse_formula("~ x1 + (1|g)").is_err());
    }

    #[test]
    fn error_unmatched_paren() {
        assert!(parse_formula("y ~ x1 + (1 | g").is_err());
    }

    #[test]
    fn error_missing_bar_in_random() {
        assert!(parse_formula("y ~ x1 + (1 + x1)").is_err());
    }

    #[test]
    fn complex_formula() {
        // A realistic complex formula.
        let f = parse_formula("rt ~ 1 + condition + age + (1 + condition | subject) + (1 | item)")
            .unwrap();
        assert_eq!(f.response, "rt");
        assert_eq!(
            f.fixed_terms,
            vec![
                FixedTerm::Intercept,
                FixedTerm::Column("condition".into()),
                FixedTerm::Column("age".into()),
            ]
        );
        assert_eq!(f.random_terms.len(), 2);

        let rt0 = &f.random_terms[0];
        assert_eq!(
            rt0.terms,
            vec![FixedTerm::Intercept, FixedTerm::Column("condition".into()),]
        );
        assert_eq!(rt0.grouping, GroupingFactor::Single("subject".into()));

        let rt1 = &f.random_terms[1];
        assert_eq!(rt1.terms, vec![FixedTerm::Intercept]);
        assert_eq!(rt1.grouping, GroupingFactor::Single("item".into()));
    }

    // ---- Grammar property tests (strictness) ----

    /// Deterministic xorshift PRNG so the generated-grammar property test is
    /// reproducible without pulling in a proptest/quickcheck dependency.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Property: any RHS assembled from the grammar
    ///   term      := ident | ident ':' ident | ident '*' ident | '1' | '0'
    ///   re_block  := '(' ('1' | '1 + ' ident) ('|' | '||') ident ')'
    ///   rhs       := term-or-block ( (' + ' | ' - ') term-or-block )*
    /// must parse successfully and yield at least one fixed or random term.
    /// `-` is exercised as lme4 term removal, which never makes a well-formed
    /// formula unparseable.
    #[test]
    fn grammar_property_wellformed_formulas_parse() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let idents = ["x", "y1", "cond", "age", "grp", "subj", "item"];
        let groups = ["g", "subj", "item", "site"];
        for _ in 0..2000 {
            let n_parts = 1 + rng.below(4);
            let mut rhs = String::new();
            for part in 0..n_parts {
                let kind = rng.below(6);
                let is_block = kind == 5;
                if part > 0 {
                    // `-` is term removal and is invalid before a RE block, so
                    // a RE-block part is always introduced with `+`.
                    let use_minus = !is_block && rng.below(2) == 0;
                    rhs.push_str(if use_minus { " - " } else { " + " });
                }
                match kind {
                    0 => rhs.push('1'),
                    1 => rhs.push('0'),
                    2 => rhs.push_str(idents[rng.below(idents.len())]),
                    3 => {
                        rhs.push_str(idents[rng.below(idents.len())]);
                        rhs.push(':');
                        rhs.push_str(idents[rng.below(idents.len())]);
                    }
                    4 => {
                        rhs.push_str(idents[rng.below(idents.len())]);
                        rhs.push('*');
                        rhs.push_str(idents[rng.below(idents.len())]);
                    }
                    _ => {
                        let bar = if rng.below(2) == 0 { "|" } else { "||" };
                        rhs.push_str(&format!(
                            "(1{} {} {})",
                            if rng.below(2) == 0 {
                                String::new()
                            } else {
                                format!(" + {}", idents[rng.below(idents.len())])
                            },
                            bar,
                            groups[rng.below(groups.len())]
                        ));
                    }
                }
            }
            // Guarantee at least one additive term that the generator can
            // never reference (and therefore never `-`-remove), so a
            // removal-heavy RHS still leaves a non-degenerate model.
            let formula = format!("resp ~ keep_sentinel + {rhs}");
            let parsed = parse_formula(&formula);
            assert!(
                parsed.is_ok(),
                "well-formed formula failed to parse: {formula:?} -> {:?}",
                parsed.err()
            );
            let f = parsed.unwrap();
            assert!(
                !f.fixed_terms.is_empty() || !f.random_terms.is_empty(),
                "formula {formula:?} produced no terms"
            );
        }
    }

    #[test]
    fn grammar_property_trailing_operator_rejected() {
        for bad in ["y ~ x +", "y ~ x -", "y ~ x + (1|g) +", "y ~ 1 + x1 - "] {
            assert!(
                matches!(parse_formula(bad), Err(FormulaError::TrailingOperator(_))),
                "expected TrailingOperator for {bad:?}, got {:?}",
                parse_formula(bad)
            );
        }
    }

    #[test]
    fn grammar_property_missing_separator_rejected() {
        for bad in ["y ~ (1|g) (1|h)", "y ~ x1 x2", "y ~ x1 (1|g)"] {
            assert!(
                matches!(
                    parse_formula(bad),
                    Err(FormulaError::MissingTermSeparator(_))
                ),
                "expected MissingTermSeparator for {bad:?}, got {:?}",
                parse_formula(bad)
            );
        }
    }

    #[test]
    fn grammar_property_numeric_literal_term_rejected() {
        for bad in ["y ~ 2 * x1", "y ~ x1 + 3", "y ~ 2.5 + x1", "y ~ x1:4"] {
            assert!(
                matches!(
                    parse_formula(bad),
                    Err(FormulaError::NumericLiteralTerm(_, _))
                ),
                "expected NumericLiteralTerm for {bad:?}, got {:?}",
                parse_formula(bad)
            );
        }
    }

    #[test]
    fn minus_is_top_level_term_removal() {
        // lme4 semantics: `-` removes a term from the set, it does not add it.
        let f = parse_formula("y ~ 1 + x1 + x2 - x2").unwrap();
        assert_eq!(
            f.fixed_terms,
            vec![FixedTerm::Intercept, FixedTerm::Column("x1".into()),],
            "x2 should have been removed, not added"
        );

        // `- 1` still suppresses the intercept. Per the documented
        // normalization (parse_formula), intercept suppression is represented
        // as the *absence* of a FixedTerm::Intercept, not a NoIntercept token.
        let f = parse_formula("y ~ x1 - 1").unwrap();
        assert!(!f.fixed_terms.contains(&FixedTerm::Intercept));
        assert!(!f.fixed_terms.contains(&FixedTerm::NoIntercept));
        assert!(f.fixed_terms.contains(&FixedTerm::Column("x1".into())));

        // Removing a never-added term is a harmless no-op (the implicit
        // intercept is still inserted since none was given explicitly).
        let f = parse_formula("y ~ x1 - x9").unwrap();
        assert_eq!(
            f.fixed_terms,
            vec![FixedTerm::Intercept, FixedTerm::Column("x1".into())]
        );
    }

    #[test]
    fn minus_on_random_effect_block_rejected() {
        assert!(matches!(
            parse_formula("y ~ x1 - (1 | g)"),
            Err(FormulaError::NegatedRandomEffect(_))
        ));
    }
}
