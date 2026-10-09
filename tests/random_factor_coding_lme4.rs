//! lme4 parity for factor random-effect bases: `||` with a factor (lme4's
//! `expandDoubleVerts`) and R's model.matrix coding of `(0 + f + h | g)`.
//!
//! Reference values from R 4.3 / lme4 (`lmer(..., REML = FALSE)`) on the
//! deterministic data built by `factor_slope_data` (same construction in R:
//! see the comments there).

use approx::assert_relative_eq;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{DataFrame, LinearMixedModel, MixedModelFit};

/// R:
/// ```r
/// G <- 12; R <- 6; i <- 0:(G*R-1); gi <- i %/% R; ri <- i %% R
/// g <- factor(sprintf("g%02d", gi)); f <- factor(c("a","b","c")[(ri %% 3)+1])
/// h <- factor(c("p","q")[((ri %/% 3) + gi) %% 2 + 1]); x <- ri - 2.5 + 0.1*gi
/// u0 <- sin(gi*1.7)*1.2; ub <- cos(gi*2.3)*0.8; uc <- sin(gi*0.9+1)*0.6
/// us <- cos(gi*1.1)*0.3; eff <- ifelse(f=="b", ub, ifelse(f=="c", uc, 0))
/// y <- 5 + 0.5*x + c(0,1,-0.5)[as.integer(f)] + u0 + eff + us*x +
///      0.4*sin(i*12.9898)*sqrt(3)
/// ```
fn factor_slope_data() -> DataFrame {
    let (groups, reps) = (12usize, 6usize);
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut f = Vec::new();
    let mut h = Vec::new();
    let mut g = Vec::new();
    for i in 0..groups * reps {
        let gi = i / reps;
        let ri = i % reps;
        let gf = gi as f64;
        let level = ri % 3;
        let xv = ri as f64 - 2.5 + 0.1 * gf;
        let u0 = (gf * 1.7).sin() * 1.2;
        let ub = (gf * 2.3).cos() * 0.8;
        let uc = (gf * 0.9 + 1.0).sin() * 0.6;
        let us = (gf * 1.1).cos() * 0.3;
        let eff = [0.0, ub, uc][level];
        let fixed = [0.0, 1.0, -0.5][level];
        let noise = 0.4 * (i as f64 * 12.9898).sin() * 3f64.sqrt();
        y.push(5.0 + 0.5 * xv + fixed + u0 + eff + us * xv + noise);
        x.push(xv);
        f.push(["a", "b", "c"][level].to_string());
        h.push(["p", "q"][(ri / 3 + gi) % 2].to_string());
        g.push(format!("g{gi:02}"));
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("f", f).unwrap();
    data.add_categorical("h", h).unwrap();
    data.add_categorical("g", g).unwrap();
    data
}

fn fit_ml(formula: &str) -> LinearMixedModel {
    let data = factor_slope_data();
    let mut model = LinearMixedModel::new(parse_formula(formula).unwrap(), &data, None).unwrap();
    model.fit(false).unwrap();
    model
}

#[test]
fn double_bar_with_factor_expands_like_lme4() {
    // lme4: (1 + f || g) -> (1 | g) + (0 + f | g); 1 + 6 = 7 theta.
    let model = fit_ml("y ~ f + x + (1 + f || g)");
    assert_eq!(model.theta().len(), 7);
    assert_relative_eq!(model.objective(), 165.48433362, epsilon = 2e-4);
    let bases: Vec<Vec<String>> = model.reterms().iter().map(|t| t.cnames.clone()).collect();
    assert!(bases.contains(&vec!["(Intercept)".to_string()]));
    assert!(bases.iter().any(|b| b.len() == 3), "{bases:?}");
    // The expansion is announced on the effective formula.
    let shown = model.formula().to_string();
    assert!(shown.contains("(0 + f | g)"), "{shown}");
}

#[test]
fn double_bar_with_numeric_and_factor_expands_like_lme4() {
    // lme4: (1 + x + f || g) -> (1 | g) + (0 + x | g) + (0 + f | g); 8 theta.
    let model = fit_ml("y ~ f + x + (1 + x + f || g)");
    assert_eq!(model.theta().len(), 8);
    assert_relative_eq!(model.objective(), 106.860296627, epsilon = 2e-4);
}

#[test]
fn double_bar_numeric_only_is_unchanged() {
    let model = fit_ml("y ~ f + x + (1 + x || g)");
    assert_eq!(model.theta().len(), 2);
    assert_eq!(model.reterms().len(), 1);
    assert_relative_eq!(model.objective(), 147.843802796, epsilon = 2e-4);
}

#[test]
fn no_intercept_random_basis_uses_r_model_matrix_coding() {
    // R: model.matrix(~ 0 + f + h) = fa, fb, fc, hq (4 columns, full rank);
    // cell-means coding of both factors would give 5 rank-deficient columns.
    let model = fit_ml("y ~ f + h + x + (0 + f + h | g)");
    assert_eq!(model.reterms()[0].cnames.len(), 4);
    assert_eq!(model.theta().len(), 10);
    assert_relative_eq!(model.objective(), 116.018309689, epsilon = 2e-4);
}
