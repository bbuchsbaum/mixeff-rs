use std::env;

use mixedmodels::formula::parse_formula;
use mixedmodels::model::data::DataFrame;
use mixedmodels::model::generalized::GeneralizedLinearMixedModel;
use mixedmodels::model::linear::{LinearMixedModel, MatrixBlock};
use mixedmodels::model::traits::{Family, MixedModelFit};
use serde::Serialize;

fn simulate_data(n_subjects: usize, n_obs_per_subject: usize, seed: u64) -> DataFrame {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let beta = [250.0, 10.0];
    let sigma = 25.0;
    let lambda = [[24.0, 0.0], [1.68, 5.23]];

    let total_n = n_subjects * n_obs_per_subject;
    let mut reaction = Vec::with_capacity(total_n);
    let mut days = Vec::with_capacity(total_n);
    let mut subj_labels = Vec::with_capacity(total_n);

    for i in 0..n_subjects {
        let u0 = normal.sample(&mut rng);
        let u1 = normal.sample(&mut rng);
        let b0 = lambda[0][0] * u0;
        let b1 = lambda[1][0] * u0 + lambda[1][1] * u1;

        let label = format!("S{:04}", i + 1);
        for d in 0..n_obs_per_subject {
            let x = d as f64;
            let mu = beta[0] + beta[1] * x + b0 + b1 * x;
            let y = mu + sigma * normal.sample(&mut rng);
            reaction.push(y);
            days.push(x);
            subj_labels.push(label.clone());
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("reaction", reaction);
    df.add_numeric("days", days);
    df.add_categorical("subj", subj_labels);
    df
}

fn centered_mod(value: usize, modulus: usize, center: f64, scale: f64) -> f64 {
    ((value % modulus) as f64 - center) * scale
}

fn simulate_large_theta_data(
    n_subjects: usize,
    n_items: usize,
    n_sites: usize,
    n_rep: usize,
) -> DataFrame {
    let beta = [250.0, 9.5];

    let total_n = n_subjects * n_items * n_rep;
    let mut reaction = Vec::with_capacity(total_n);
    let mut days = Vec::with_capacity(total_n);
    let mut subj_labels = Vec::with_capacity(total_n);
    let mut item_labels = Vec::with_capacity(total_n);
    let mut site_labels = Vec::with_capacity(total_n);

    for s in 0..n_subjects {
        let subj_b0 = centered_mod(7 * s + 3, 19, 9.0, 2.4);
        let subj_b1 = centered_mod(11 * s + 5, 17, 8.0, 0.38) + 0.05 * subj_b0;
        let subj_label = format!("S{:03}", s + 1);

        for i in 0..n_items {
            let item_b0 = centered_mod(13 * i + 2, 23, 11.0, 1.6);
            let item_b1 = centered_mod(5 * i + 7, 19, 9.0, 0.27) - 0.04 * item_b0;
            let item_label = format!("I{:03}", i + 1);

            for r in 0..n_rep {
                let site = (5 * s + 3 * i + r) % n_sites;
                let site_b0 = centered_mod(3 * site + 1, 13, 6.0, 1.2);
                let site_b1 = centered_mod(7 * site + 4, 11, 5.0, 0.18) + 0.03 * site_b0;
                let eps = centered_mod(13 * s + 7 * i + 3 * r + 2 * site, 29, 14.0, 0.9);
                let x = r as f64 + (i % 4) as f64 * 0.35 + (s % 3) as f64 * 0.1;

                let mu = beta[0]
                    + beta[1] * x
                    + subj_b0
                    + subj_b1 * x
                    + item_b0
                    + item_b1 * x
                    + site_b0
                    + site_b1 * x;

                reaction.push(mu + eps);
                days.push(x);
                subj_labels.push(subj_label.clone());
                item_labels.push(item_label.clone());
                site_labels.push(format!("K{:03}", site + 1));
            }
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("reaction", reaction);
    df.add_numeric("days", days);
    df.add_categorical("subj", subj_labels);
    df.add_categorical("item", item_labels);
    df.add_categorical("site", site_labels);
    df
}

fn block_logdet_factor(block: &MatrixBlock) -> f64 {
    match block {
        MatrixBlock::Diagonal(diag) => diag.iter().filter(|&&d| d > 0.0).map(|d| d.ln()).sum(),
        MatrixBlock::BlockDiagonal(blocks) => blocks
            .iter()
            .map(|blk| {
                (0..blk.nrows())
                    .map(|i| blk[(i, i)])
                    .filter(|&d| d > 0.0)
                    .map(f64::ln)
                    .sum::<f64>()
            })
            .sum(),
        MatrixBlock::Dense(mat) => (0..mat.nrows().min(mat.ncols()))
            .map(|i| mat[(i, i)])
            .filter(|&d| d > 0.0)
            .map(f64::ln)
            .sum(),
        MatrixBlock::Sparse(mat) => {
            let dense = MatrixBlock::Sparse(mat.clone()).as_dense();
            (0..dense.nrows().min(dense.ncols()))
                .map(|i| dense[(i, i)])
                .filter(|&d| d > 0.0)
                .map(f64::ln)
                .sum()
        }
    }
}

fn objective_components(model: &LinearMixedModel) -> (f64, f64, f64) {
    let k = model.reterms.len();
    let logdet_re = (0..k)
        .map(|j| {
            let idx = j * (j + 1) / 2 + j;
            2.0 * block_logdet_factor(&model.l_blocks[idx])
        })
        .sum();

    let last_idx = k * (k + 1) / 2 + k;
    let last = model.l_blocks[last_idx].as_dense();
    let mut logdet_xx = 0.0;
    for i in 0..(last.nrows().saturating_sub(1)) {
        let d = last[(i, i)];
        if d > 0.0 {
            logdet_xx += d.ln();
        }
    }

    let logdet_total = if model.optsum.reml {
        logdet_re + 2.0 * logdet_xx
    } else {
        logdet_re
    };

    (logdet_re, logdet_xx * 2.0, logdet_total)
}

#[derive(Serialize)]
struct ParityDump {
    model: String,
    formula: String,
    n_rows: usize,
    n_subjects: usize,
    n_obs_per_subject: usize,
    n_items: Option<usize>,
    n_sites: Option<usize>,
    n_rep: Option<usize>,
    data_reaction_sum: f64,
    data_days_sum: f64,
    seed: u64,
    reml: bool,
    fit_theta: Vec<f64>,
    fit_beta: Vec<f64>,
    fit_sigma: f64,
    fit_objective: f64,
    fit_pwrss: f64,
    fit_logdet_re: f64,
    fit_logdet_xx: f64,
    fit_logdet_total: f64,
    fit_feval: i64,
    input_theta: Option<Vec<f64>>,
    objective_at_input_theta: Option<f64>,
    input_theta_pwrss: Option<f64>,
    input_theta_logdet_re: Option<f64>,
    input_theta_logdet_xx: Option<f64>,
    input_theta_logdet_total: Option<f64>,
}

fn parse_flag<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    let prefix = format!("--{name}=");
    env::args()
        .skip(1)
        .find_map(|arg| arg.strip_prefix(&prefix).map(str::to_owned))
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}

fn parse_theta() -> Option<Vec<f64>> {
    let prefix = "--theta=";
    env::args().skip(1).find_map(|arg| {
        arg.strip_prefix(prefix).map(|value| {
            value
                .split(',')
                .filter(|part| !part.is_empty())
                .map(|part| part.parse::<f64>().expect("invalid theta value"))
                .collect::<Vec<_>>()
        })
    })
}

fn parse_model() -> String {
    let prefix = "--model=";
    env::args()
        .skip(1)
        .find_map(|arg| arg.strip_prefix(prefix).map(str::to_owned))
        .unwrap_or_else(|| "scalar".to_string())
}

/// Contra dataset embedded as CSV. Columns (no header):
/// `use_num` (0/1), `age`, `age2`, `urban` (Y/N), `livch` (0/1/2/3+),
/// `urban_dist` (urban × district interaction string).
const CONTRA_CSV: &str = include_str!("../src/model/contra.csv");

fn load_contra() -> DataFrame {
    let mut use_num = Vec::new();
    let mut age = Vec::new();
    let mut age2 = Vec::new();
    let mut urban = Vec::new();
    let mut livch = Vec::new();
    let mut urban_dist = Vec::new();

    for line in CONTRA_CSV.lines() {
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        use_num.push(parts[0].parse::<f64>().expect("use_num"));
        age.push(parts[1].parse::<f64>().expect("age"));
        age2.push(parts[2].parse::<f64>().expect("age2"));
        urban.push(parts[3].to_string());
        livch.push(parts[4].to_string());
        urban_dist.push(parts[5].to_string());
    }

    let mut df = DataFrame::new();
    df.add_numeric("use_num", use_num);
    df.add_numeric("age", age);
    df.add_numeric("age2", age2);
    df.add_categorical("urban", urban);
    df.add_categorical("livch", livch);
    df.add_categorical("urban_dist", urban_dist);
    df
}

#[derive(Serialize)]
struct GlmmParityDump {
    model: String,
    formula: String,
    family: String,
    link: String,
    n_rows: usize,
    n_groups: usize,
    fit_n_agq: usize,
    fit_theta: Vec<f64>,
    fit_beta: Vec<f64>,
    fit_objective: f64,
    fit_deviance_laplace: f64,
    fit_deviance_agq: f64,
    fit_feval: i64,
}

fn dump_contra_glmm(n_agq: usize) {
    let formula_str = "use_num ~ 1 + age + age2 + urban + livch + (1 | urban_dist)";
    let parsed = parse_formula(formula_str).expect("failed to parse contra formula");
    let data = load_contra();

    let mut model = GeneralizedLinearMixedModel::new(parsed, &data, Family::Bernoulli, None)
        .expect("failed to build contra GLMM");
    model
        .fit_with_options(false, n_agq, false)
        .expect("contra GLMM fit failed");

    let dev_lap = model.deviance(1);
    let dev_agq = if n_agq > 1 {
        model.deviance(n_agq)
    } else {
        dev_lap
    };

    let dump = GlmmParityDump {
        model: "contra-glmm".to_string(),
        formula: formula_str.to_string(),
        family: "bernoulli".to_string(),
        link: "logit".to_string(),
        n_rows: data.nrow(),
        n_groups: model.lmm.reterms[0].n_levels(),
        fit_n_agq: n_agq,
        fit_theta: model.theta.clone(),
        fit_beta: model.beta.iter().copied().collect(),
        fit_objective: model.objective(),
        fit_deviance_laplace: dev_lap,
        fit_deviance_agq: dev_agq,
        fit_feval: model.lmm.optsum.feval,
    };

    println!("{}", serde_json::to_string_pretty(&dump).unwrap());
}

fn main() {
    let model_name = parse_model();
    if model_name == "contra-glmm" {
        // GLMM parity dump uses a fixed real dataset; only --n-agq is honoured.
        let n_agq = parse_flag("n-agq", 7usize);
        dump_contra_glmm(n_agq);
        return;
    }
    let n_subjects = parse_flag("n-subj", 18usize);
    let n_obs_per_subject = parse_flag("n-obs", 10usize);
    let n_items = parse_flag("n-items", 12usize);
    let n_sites = parse_flag("n-sites", 6usize);
    let n_rep = parse_flag("n-rep", 4usize);
    let seed = parse_flag("seed", 42u64);
    let reml = parse_flag("reml", true);
    let input_theta = parse_theta();

    let formula = match model_name.as_str() {
        "scalar" => "reaction ~ 1 + days + (1 | subj)",
        "vector" => "reaction ~ 1 + days + (1 + days | subj)",
        "crossed" => {
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)"
        }
        other => panic!("unknown model {other}"),
    };

    let data = match model_name.as_str() {
        "scalar" | "vector" => simulate_data(n_subjects, n_obs_per_subject, seed),
        "crossed" => simulate_large_theta_data(n_subjects, n_items, n_sites, n_rep),
        _ => unreachable!(),
    };
    let parsed = parse_formula(formula).expect("failed to parse formula");
    let mut model = LinearMixedModel::new(parsed, &data, None).expect("failed to build model");
    model.fit(reml).expect("fit failed");
    let data_reaction_sum = data.numeric("reaction").unwrap().iter().sum::<f64>();
    let data_days_sum = data.numeric("days").unwrap().iter().sum::<f64>();

    let fit_theta = model.theta();
    let fit_beta = model.coef().iter().copied().collect::<Vec<_>>();
    let fit_sigma = model.sigma();
    let fit_objective = model.objective();
    let fit_pwrss = model.pwrss();
    let (fit_logdet_re, fit_logdet_xx, fit_logdet_total) = objective_components(&model);
    let fit_feval = model.optsum.feval;

    let input_theta_summary = input_theta.as_ref().map(|theta| {
        let mut probe = model.clone();
        let obj = probe
            .objective_at(theta)
            .expect("failed to evaluate objective at input theta");
        let pwrss = probe.pwrss();
        let (logdet_re, logdet_xx, logdet_total) = objective_components(&probe);
        (obj, pwrss, logdet_re, logdet_xx, logdet_total)
    });

    let dump = ParityDump {
        model: model_name,
        formula: formula.to_string(),
        n_rows: data.nrow(),
        n_subjects,
        n_obs_per_subject,
        n_items: (formula.contains("| item")).then_some(n_items),
        n_sites: (formula.contains("| site")).then_some(n_sites),
        n_rep: (formula.contains("| site")).then_some(n_rep),
        data_reaction_sum,
        data_days_sum,
        seed,
        reml,
        fit_theta,
        fit_beta,
        fit_sigma,
        fit_objective,
        fit_pwrss,
        fit_logdet_re,
        fit_logdet_xx,
        fit_logdet_total,
        fit_feval,
        input_theta,
        objective_at_input_theta: input_theta_summary.as_ref().map(|x| x.0),
        input_theta_pwrss: input_theta_summary.as_ref().map(|x| x.1),
        input_theta_logdet_re: input_theta_summary.as_ref().map(|x| x.2),
        input_theta_logdet_xx: input_theta_summary.as_ref().map(|x| x.3),
        input_theta_logdet_total: input_theta_summary.as_ref().map(|x| x.4),
    };

    println!("{}", serde_json::to_string_pretty(&dump).unwrap());
}
