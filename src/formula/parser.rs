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
    FixedTerm, Formula, GroupingFactor, RandomTerm, RandomTermExpansion, RandomTermSource,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced while parsing a formula string.
#[derive(Debug, Error)]
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

/// Tokenize the raw formula string.
fn tokenize(input: &str) -> Result<Vec<Spanned>, FormulaError> {
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
                tokens.push(Spanned {
                    token: Token::Ident(word),
                    pos,
                });
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

        // Track whether next term is subtracted (e.g. `- 1`).
        let mut negate = false;

        loop {
            if self.at_end() {
                break;
            }

            match self.peek() {
                Some(Token::Plus) => {
                    self.advance();
                    negate = false;
                    continue;
                }
                Some(Token::Minus) => {
                    self.advance();
                    negate = true;
                    continue;
                }
                Some(Token::LParen) => {
                    // Random effect group.
                    let rts = self.parse_random_term()?;
                    random.extend(rts);
                    negate = false;
                }
                Some(Token::One) => {
                    self.advance();
                    if negate {
                        // `-1` means suppress intercept.
                        fixed.push(FixedTerm::NoIntercept);
                    } else {
                        fixed.push(FixedTerm::Intercept);
                    }
                    negate = false;
                }
                Some(Token::Zero) => {
                    self.advance();
                    fixed.push(FixedTerm::NoIntercept);
                    negate = false;
                }
                Some(Token::Ident(_)) => {
                    let terms = self.parse_term_expr()?;
                    fixed.extend(terms);
                    negate = false;
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
/// use mixedmodels::formula::parser::parse_formula;
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

    let tokens = tokenize(input)?;
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

    Ok(Formula {
        response,
        fixed_terms: fixed,
        random_terms: random,
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
        assert_eq!(
            rt.terms,
            vec![FixedTerm::Intercept, FixedTerm::Column("x1".into())]
        );
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
}
