//! Small LMM comparison examples against lme4 reference fits.
//!
//! This is the quick, inspectable companion to the full batch workflow:
//!
//! ```text
//! cargo run --release --example compare_rust
//! Rscript scripts/compare_lme4.R
//! cargo run --release --example compare_report
//! ```
//!
//! Run this focused example with:
//!
//! ```text
//! cargo run --example lmm_lme4_comparison
//! ```

use mixeff_rs::datasets;
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::{LinearMixedModel, MixedModelFit};

const TOL: f64 = 5.0e-3;

struct Case {
    dataset: &'static str,
    formula: &'static str,
    estimator: Estimator,
    lme4_call: &'static str,
    lme4: Lme4Reference,
}

#[derive(Clone, Copy)]
enum Estimator {
    Reml,
    Ml,
}

impl Estimator {
    fn reml(self) -> bool {
        matches!(self, Estimator::Reml)
    }

    fn label(self) -> &'static str {
        match self {
            Estimator::Reml => "REML",
            Estimator::Ml => "ML",
        }
    }
}

struct Lme4Reference {
    beta: &'static [f64],
    sigma: f64,
    theta: &'static [f64],
    objective: f64,
    loglik: f64,
}

struct FitSummary {
    beta: Vec<f64>,
    sigma: f64,
    theta: Vec<f64>,
    objective: f64,
    loglik: f64,
}

const CASES: &[Case] = &[
    Case {
        dataset: "dyestuff",
        formula: "Yield ~ 1 + (1 | Batch)",
        estimator: Estimator::Reml,
        lme4_call: r#"lme4::lmer(Yield ~ 1 + (1 | Batch), data = Dyestuff, REML = TRUE)"#,
        lme4: Lme4Reference {
            beta: &[1527.5],
            sigma: 49.5101,
            theta: &[0.8483],
            objective: 319.6543,
            loglik: -159.8271,
        },
    },
    Case {
        dataset: "dyestuff",
        formula: "Yield ~ 1 + (1 | Batch)",
        estimator: Estimator::Ml,
        lme4_call: r#"lme4::lmer(Yield ~ 1 + (1 | Batch), data = Dyestuff, REML = FALSE)"#,
        lme4: Lme4Reference {
            beta: &[1527.5],
            sigma: 49.5101,
            theta: &[0.7526],
            objective: 327.3271,
            loglik: -163.6635,
        },
    },
    Case {
        dataset: "rail",
        formula: "travel ~ 1 + (1 | Rail)",
        estimator: Estimator::Reml,
        lme4_call: r#"lme4::lmer(travel ~ 1 + (1 | Rail), data = Rail, REML = TRUE)"#,
        lme4: Lme4Reference {
            beta: &[66.5],
            sigma: 4.0208,
            theta: &[6.1693],
            objective: 122.1770,
            loglik: -61.0885,
        },
    },
    Case {
        dataset: "sleepstudy",
        formula: "Reaction ~ 1 + Days + (1 + Days | Subject)",
        estimator: Estimator::Reml,
        lme4_call: r#"lme4::lmer(Reaction ~ 1 + Days + (1 + Days | Subject), data = sleepstudy, REML = TRUE)"#,
        lme4: Lme4Reference {
            beta: &[251.4051, 10.4673],
            sigma: 25.5918,
            theta: &[0.9667, 0.0152, 0.2309],
            objective: 1743.6283,
            loglik: -871.8141,
        },
    },
    Case {
        dataset: "penicillin",
        formula: "diameter ~ 1 + (1 | plate) + (1 | sample)",
        estimator: Estimator::Reml,
        lme4_call: r#"lme4::lmer(diameter ~ 1 + (1 | plate) + (1 | sample), data = Penicillin, REML = TRUE)"#,
        lme4: Lme4Reference {
            beta: &[22.9722],
            sigma: 0.5499,
            theta: &[1.5397, 3.5125],
            objective: 330.8606,
            loglik: -165.4303,
        },
    },
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("LMM/lme4 comparison examples");
    println!("tolerance: max abs delta <= {TOL}\n");
    println!("| dataset | est | max Δβ | Δσ | max Δθ | Δobjective | ΔlogLik | status |");
    println!("|---|---|---:|---:|---:|---:|---:|---|");

    let mut failed = Vec::new();
    for case in CASES {
        let rust = fit_case(case)?;
        let row = compare(case, &rust);
        println!(
            "| {} | {} | {:.4e} | {:.4e} | {:.4e} | {:.4e} | {:.4e} | {} |",
            case.dataset,
            case.estimator.label(),
            row.beta_delta,
            row.sigma_delta,
            row.theta_delta,
            row.objective_delta,
            row.loglik_delta,
            if row.passed { "ok" } else { "check" }
        );
        if !row.passed {
            failed.push(format!(
                "{} [{}] :: {}",
                case.dataset,
                case.estimator.label(),
                case.formula
            ));
        }
    }

    println!("\nEquivalent lme4 calls:");
    for case in CASES {
        println!("- {}", case.lme4_call);
    }

    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} comparison(s) exceeded tolerance: {:?}",
            failed.len(),
            failed
        )
        .into())
    }
}

fn fit_case(case: &Case) -> Result<FitSummary, Box<dyn std::error::Error>> {
    let (data, _) = datasets::load(case.dataset)?;
    let formula = parse_formula(case.formula)?;
    let mut model = LinearMixedModel::new(formula, &data, None)?;
    model.fit(case.estimator.reml())?;

    Ok(FitSummary {
        beta: MixedModelFit::coef(&model).iter().copied().collect(),
        sigma: model.sigma(),
        theta: model.theta(),
        objective: model.objective_value(),
        loglik: MixedModelFit::loglikelihood(&model),
    })
}

struct ComparisonRow {
    beta_delta: f64,
    sigma_delta: f64,
    theta_delta: f64,
    objective_delta: f64,
    loglik_delta: f64,
    passed: bool,
}

fn compare(case: &Case, rust: &FitSummary) -> ComparisonRow {
    let beta_delta = max_abs_delta(&rust.beta, case.lme4.beta);
    let theta_delta = max_abs_delta(&rust.theta, case.lme4.theta);
    let sigma_delta = (rust.sigma - case.lme4.sigma).abs();
    let objective_delta = (rust.objective - case.lme4.objective).abs();
    let loglik_delta = (rust.loglik - case.lme4.loglik).abs();
    let passed = [
        beta_delta,
        theta_delta,
        sigma_delta,
        objective_delta,
        loglik_delta,
    ]
    .iter()
    .all(|delta| *delta <= TOL);

    ComparisonRow {
        beta_delta,
        sigma_delta,
        theta_delta,
        objective_delta,
        loglik_delta,
        passed,
    }
}

fn max_abs_delta(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(
        actual.len(),
        expected.len(),
        "comparison vectors must have the same length"
    );
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f64::max)
}
