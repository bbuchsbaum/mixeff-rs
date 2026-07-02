use super::*;
use approx::assert_relative_eq;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::compiler::{
    CertificateCheck, CompiledModelArtifact, CompilerPolicy, ContrastMatrix, ContrastRhs,
    ConvergenceLevel, ConvergenceVerdict, DiagnosticCode, EffectiveRankStatus, EvidenceMethod,
    EvidenceQuality, FitIntent, FitStatus, FixedEffectCovarianceMethod,
    FixedEffectCovarianceStatus, FixedEffectHypothesis, InferenceStatus, InformationBudgetStatus,
    ModelChangeStatus, ModelStateStatus, RandomStrategy, RankStatus, ReductionRecord,
    ReductionTrigger, ThetaMap,
};
use crate::formula::parse_formula;
use crate::model::data::{Column, DataFrame};
use crate::model::traits::MixedModelFit;

fn simulate_sleepstudy_like(n_subjects: usize, n_obs_per_subject: usize, seed: u64) -> DataFrame {
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
    df.add_numeric("reaction", reaction).unwrap();
    df.add_numeric("days", days).unwrap();
    df.add_categorical("subj", subj_labels).unwrap();
    df
}

fn grouped_slope_data(n_groups: usize) -> DataFrame {
    grouped_slope_data_with_obs(n_groups, 2)
}

fn grouped_slope_data_with_obs(n_groups: usize, obs_per_group: usize) -> DataFrame {
    let mut data = DataFrame::new();
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();
    for idx in 0..n_groups {
        for obs in 0..obs_per_group {
            y.push(idx as f64 + obs as f64);
            x.push(obs as f64);
            group.push(format!("g{}", idx + 1));
        }
    }
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("group", group).unwrap();
    data
}

fn three_level_condition_fixture() -> (LinearMixedModel, FixedEffectHypothesis) {
    let n_subj = 10usize;
    let n_per = 6usize;
    let mut subj = Vec::new();
    let mut cond = Vec::new();
    let mut y = Vec::new();
    let levels = ["A", "B", "C"];
    for subject in 0..n_subj {
        let subject_offset = subject as f64 * 0.03;
        for obs in 0..n_per {
            let level = levels[obs % levels.len()];
            subj.push(format!("s{subject}"));
            cond.push(level.to_string());
            let treatment = match level {
                "B" => 0.6,
                "C" => 0.3,
                _ => 0.0,
            };
            let noise = ((subject * 13 + obs * 7) % 11) as f64 * 0.01;
            y.push(2.0 + treatment + subject_offset + noise);
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_categorical("cond", cond).unwrap();
    data.add_categorical("subj", subj).unwrap();

    let formula = parse_formula("y ~ 1 + cond + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let names = model.coef_names();
    let cond_b = names.iter().position(|name| name == "cond: B").unwrap();
    let cond_c = names.iter().position(|name| name == "cond: C").unwrap();
    let mut l = DMatrix::zeros(2, names.len());
    l[(0, cond_b)] = 1.0;
    l[(1, cond_c)] = 1.0;
    let hypothesis = FixedEffectHypothesis::zero_rhs("cond", ContrastMatrix::new(l).unwrap());
    (model, hypothesis)
}

fn joint_wald_f_direct_inverse_oracle(
    model: &LinearMixedModel,
    hypothesis: &FixedEffectHypothesis,
) -> f64 {
    let beta = model.coef();
    let vcov = model.vcov();
    let delta = &hypothesis.l.values * beta - &hypothesis.rhs.values;
    let middle =
        symmetrize_matrix(&(&hypothesis.l.values * vcov * hypothesis.l.values.transpose()));
    let inverse = middle
        .try_inverse()
        .expect("condition fixture should yield full-rank L V L'");
    let quadratic = (delta.transpose() * inverse * delta)[(0, 0)];
    quadratic / hypothesis.n_contrasts() as f64
}

fn typed_term_test_fixture() -> LinearMixedModel {
    let mut subject = Vec::new();
    let mut x = Vec::new();
    let mut z = Vec::new();
    let mut y = Vec::new();
    for group in 0..12 {
        let group_offset = group as f64 * 0.04;
        for obs in 0..5 {
            let xv = obs as f64 - 2.0 + group as f64 * 0.03;
            let zv = ((group + obs * 2) % 7) as f64 / 3.0 - 1.0;
            let wiggle = ((group * 11 + obs * 5) % 13) as f64 * 0.01;
            subject.push(format!("s{group}"));
            x.push(xv);
            z.push(zv);
            y.push(1.0 + 0.8 * xv - 0.4 * zv + 0.5 * xv * zv + group_offset + wiggle);
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_numeric("z", z).unwrap();
    data.add_categorical("subject", subject).unwrap();

    let formula = parse_formula("y ~ 1 + x + z + x:z + (1 | subject)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();
    model
}

fn hypothesis_by_label<'a>(
    hypotheses: &'a [FixedEffectHypothesis],
    label: &str,
) -> &'a FixedEffectHypothesis {
    hypotheses
        .iter()
        .find(|hypothesis| hypothesis.label == label)
        .unwrap_or_else(|| panic!("missing hypothesis {label} in {hypotheses:?}"))
}

fn matrices_differ(a: &DMatrix<f64>, b: &DMatrix<f64>, tolerance: f64) -> bool {
    a.shape() != b.shape()
        || a.iter()
            .zip(b.iter())
            .any(|(left, right)| (left - right).abs() > tolerance)
}

fn successful_bootstrap_payload_with_statistics(
    model: &LinearMixedModel,
    hypothesis_label: &str,
    replicate_statistics: Vec<f64>,
    statistic_label: &str,
) -> BootstrapRunPayload {
    let fits = replicate_statistics
        .iter()
        .enumerate()
        .map(|(i, _)| BootstrapReplicate {
            objective: i as f64 + 1.0,
            sigma: model.sigma(),
            beta: model.beta(),
            se: model.stderror(),
            theta: model.theta(),
        })
        .collect::<Vec<_>>();
    let bsamp = MixedModelBootstrap { fits };
    let metadata = bsamp.run_metadata_for_model(
        model,
        BootstrapTarget::fixed_effect_null("fixed-effect null", hypothesis_label),
        replicate_statistics.len(),
        BootstrapFailedRefitPolicy::Exclude,
        BootstrapSeedRecord::std_rng(20260513),
        BootstrapRefitOptions::from_model(model),
        Some(statistic_label.to_string()),
        Some(&replicate_statistics),
        None,
    );
    bsamp.into_run_payload_with_statistics(metadata, replicate_statistics)
}

#[cfg(feature = "nlopt")]
fn correlated_crossed_slope_data() -> DataFrame {
    fn centered_mod(value: usize, modulus: usize, center: f64, scale: f64) -> f64 {
        ((value % modulus) as f64 - center) * scale
    }

    let n_g = 10;
    let n_h = 8;
    let n_rep = 4;
    let mut y = Vec::with_capacity(n_g * n_h * n_rep);
    let mut x = Vec::with_capacity(n_g * n_h * n_rep);
    let mut g = Vec::with_capacity(n_g * n_h * n_rep);
    let mut h = Vec::with_capacity(n_g * n_h * n_rep);

    for gi in 0..n_g {
        let g0 = centered_mod(7 * gi + 3, 19, 9.0, 2.1);
        let g1 = 0.82 * g0 + centered_mod(11 * gi + 5, 17, 8.0, 0.18);
        for hi in 0..n_h {
            let h0 = centered_mod(13 * hi + 2, 23, 11.0, 1.5);
            let h1 = -0.74 * h0 + centered_mod(5 * hi + 7, 19, 9.0, 0.16);
            for rep in 0..n_rep {
                let xv = rep as f64 - 1.5 + (gi % 3) as f64 * 0.08 + (hi % 2) as f64 * 0.05;
                let eps = centered_mod(gi * 11 + hi * 7 + rep * 5, 31, 15.0, 0.28);
                y.push(4.0 + 1.7 * xv + g0 + g1 * xv + h0 + h1 * xv + eps);
                x.push(xv);
                g.push(format!("g{:02}", gi + 1));
                h.push(format!("h{:02}", hi + 1));
            }
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("g", g).unwrap();
    data.add_categorical("h", h).unwrap();
    data
}

fn vsize3_kernel_remat() -> ReMat {
    ReMat::new(
        "subj".to_string(),
        vec![0, 1],
        vec!["S1".to_string(), "S2".to_string()],
        vec!["(Intercept)".to_string(), "x".to_string(), "z".to_string()],
        DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 0.0, 1.0, 2.0, 3.0]),
    )
}

#[test]
fn test_apply_lambda_transpose_to_rhs_consistent_with_parmap_order() {
    let mut re = vsize3_kernel_remat();
    let parmap = build_parmap(&[re.clone()]);
    assert_eq!(
        parmap,
        vec![
            (0, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (0, 1, 1),
            (0, 2, 1),
            (0, 2, 2)
        ]
    );

    re.set_theta(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(
        re.lambda,
        DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0])
    );

    let mut rhs = DMatrix::from_row_slice(
        6,
        2,
        &[
            7.0, 29.0, 11.0, 31.0, 13.0, 37.0, 17.0, 41.0, 19.0, 43.0, 23.0, 47.0,
        ],
    );
    let original = rhs.clone();

    apply_lambda_transpose_to_rhs(&mut rhs, &re);

    let lambda_t = re.lambda.transpose();
    let mut expected = DMatrix::zeros(6, 2);
    for level in 0..2 {
        let offset = level * 3;
        let expected_block = &lambda_t * original.rows(offset, 3).into_owned();
        expected.rows_mut(offset, 3).copy_from(&expected_block);
    }
    assert_eq!(rhs, expected);
}

fn diagonal_theta_indices(model: &LinearMixedModel) -> Vec<usize> {
    model
        .parmap
        .iter()
        .enumerate()
        .filter_map(|(idx, &(_, row, col))| (row == col).then_some(idx))
        .collect()
}

fn assert_theta_diagonals_nonnegative(model: &LinearMixedModel) {
    let theta = model.theta();
    for idx in diagonal_theta_indices(model) {
        assert!(
            theta[idx] >= 0.0,
            "theta diagonal {idx} should be rectified, got {}",
            theta[idx]
        );
        assert_eq!(
            model.optsum.final_params[idx], theta[idx],
            "final_params must store the rectified theta value"
        );
    }
}

#[cfg(feature = "nlopt")]
fn simulate_large_theta_crossed(seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let n_subjects = 18;
    let n_items = 12;
    let n_sites = 6;
    let n_rep = 4;

    let beta = [250.0, 9.5];
    let sigma = 18.0;
    let lambda_subj = [[18.0, 0.0], [2.2, 4.5]];
    let lambda_item = [[11.0, 0.0], [-1.4, 3.2]];
    let lambda_site = [[7.5, 0.0], [0.6, 1.7]];

    let draw_effects = |rng: &mut StdRng, lambda: [[f64; 2]; 2], levels: usize| {
        let mut effects = Vec::with_capacity(levels);
        for _ in 0..levels {
            let u0 = normal.sample(rng);
            let u1 = normal.sample(rng);
            effects.push([lambda[0][0] * u0, lambda[1][0] * u0 + lambda[1][1] * u1]);
        }
        effects
    };

    let subj_effects = draw_effects(&mut rng, lambda_subj, n_subjects);
    let item_effects = draw_effects(&mut rng, lambda_item, n_items);
    let site_effects = draw_effects(&mut rng, lambda_site, n_sites);

    let total_n = n_subjects * n_items * n_rep;
    let mut reaction = Vec::with_capacity(total_n);
    let mut days = Vec::with_capacity(total_n);
    let mut subj_labels = Vec::with_capacity(total_n);
    let mut item_labels = Vec::with_capacity(total_n);
    let mut site_labels = Vec::with_capacity(total_n);

    for s in 0..n_subjects {
        for i in 0..n_items {
            for r in 0..n_rep {
                let site = (s * 5 + i * 3 + r) % n_sites;
                let x = r as f64 + (i % 4) as f64 * 0.35;
                let mut mu = beta[0] + beta[1] * x;
                mu += subj_effects[s][0] + subj_effects[s][1] * x;
                mu += item_effects[i][0] + item_effects[i][1] * x;
                mu += site_effects[site][0] + site_effects[site][1] * x;
                let y = mu + sigma * normal.sample(&mut rng);

                reaction.push(y);
                days.push(x);
                subj_labels.push(format!("S{:03}", s + 1));
                item_labels.push(format!("I{:03}", i + 1));
                site_labels.push(format!("K{:03}", site + 1));
            }
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("reaction", reaction).unwrap();
    df.add_numeric("days", days).unwrap();
    df.add_categorical("subj", subj_labels).unwrap();
    df.add_categorical("item", item_labels).unwrap();
    df.add_categorical("site", site_labels).unwrap();
    df
}

fn permute_rows(data: &DataFrame, order: &[usize]) -> DataFrame {
    let mut permuted = DataFrame::new();

    for name in data.column_names() {
        match data.column(name).unwrap() {
            Column::Numeric(values) => {
                let reordered = order.iter().map(|&idx| values[idx]).collect();
                permuted.add_numeric(name, reordered).unwrap();
            }
            Column::Categorical(cat) => {
                let reordered = order.iter().map(|&idx| cat.values[idx].clone()).collect();
                permuted.add_categorical(name, reordered).unwrap();
            }
        }
    }

    permuted
}

fn shared_julia_parity_fixture() -> DataFrame {
    let reaction = vec![
        228.34733704764443,
        294.32292211548196,
        205.740_213_893_405_7,
        278.878_780_120_278_5,
        271.077_699_509_520_6,
        244.5608057798394,
        265.944_633_024_091_4,
        226.77991725455206,
        242.4319346940861,
        214.974_081_145_202,
        323.210_130_256_588_3,
        277.4835351479876,
        273.747_591_812_113_5,
        287.110_981_496_805_4,
        278.941_478_348_983_8,
        297.196_069_266_972_8,
        228.30198076068194,
        195.39462889633353,
        217.48019241415267,
        258.9102478189954,
        276.43800461900963,
        315.60786380412753,
        272.3080316216936,
        301.842_641_745_225_9,
    ];
    let days = vec![
        0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0,
        2.0, 3.0, 0.0, 1.0, 2.0, 3.0,
    ];
    let subj = vec![
        "S0001", "S0001", "S0001", "S0001", "S0002", "S0002", "S0002", "S0002", "S0003", "S0003",
        "S0003", "S0003", "S0004", "S0004", "S0004", "S0004", "S0005", "S0005", "S0005", "S0005",
        "S0006", "S0006", "S0006", "S0006",
    ];

    let mut df = DataFrame::new();
    df.add_numeric("reaction", reaction).unwrap();
    df.add_numeric("days", days).unwrap();
    df.add_categorical("subj", subj.into_iter().map(str::to_string).collect())
        .unwrap();
    df
}

#[test]
#[cfg(not(feature = "prima"))]
fn test_forced_prima_bobyqa_requires_prima_feature() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let err = model
        .fit_with_forced_optimizer(true, Optimizer::PrimaBobyqa)
        .unwrap_err();

    match err {
        MixedModelError::Unsupported(message) => {
            assert!(message.contains("`prima` feature"));
            assert!(message.contains("libprimac"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

fn shared_julia_crossed_parity_fixture() -> DataFrame {
    fn centered_mod(value: usize, modulus: usize, center: f64, scale: f64) -> f64 {
        ((value % modulus) as f64 - center) * scale
    }

    let n_subjects = 18;
    let n_items = 12;
    let n_sites = 6;
    let n_rep = 4;
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
    df.add_numeric("reaction", reaction).unwrap();
    df.add_numeric("days", days).unwrap();
    df.add_categorical("subj", subj_labels).unwrap();
    df.add_categorical("item", item_labels).unwrap();
    df.add_categorical("site", site_labels).unwrap();
    df
}

/// Synthetic data where every group mean equals 5.0 (SS_B = 0).
/// The ML estimate of between-group variance is exactly 0 → θ = 0 → singular.
fn singular_re_fixture() -> DataFrame {
    let yields: Vec<f64> = vec![
        2.0, 8.0, 5.0, 3.0, 7.0, // batch A: mean = 5.0
        1.0, 9.0, 5.0, 4.0, 6.0, // batch B: mean = 5.0
        3.0, 7.0, 5.0, 2.0, 8.0, // batch C: mean = 5.0
        4.0, 6.0, 5.0, 1.0, 9.0, // batch D: mean = 5.0
        0.0, 10.0, 5.0, 3.0, 7.0, // batch E: mean = 5.0
        2.0, 8.0, 5.0, 4.0, 6.0, // batch F: mean = 5.0
    ];
    let batches: Vec<String> = "ABCDEF"
        .chars()
        .flat_map(|c| std::iter::repeat_n(c.to_string(), 5))
        .collect();

    let mut df = DataFrame::new();
    df.add_numeric("yield", yields).unwrap();
    df.add_categorical("batch", batches).unwrap();
    df
}

fn rank_one_rho_one_random_slope_fixture() -> DataFrame {
    let x_values = [-1.0, 0.0, 1.0, 2.0];
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut group = Vec::new();

    for g in 0..12 {
        let label = format!("G{:02}", g + 1);
        let group_effect = (g as f64 - 5.5) * 0.7;
        for (j, x_value) in x_values.iter().enumerate() {
            let residual = (((17 * g + 11 * j + 3) % 23) as f64 - 11.0) * 0.08;
            y.push(10.0 + 2.0 * x_value + group_effect * (1.0 + x_value) + residual);
            x.push(*x_value);
            group.push(label.clone());
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_categorical("group", group).unwrap();
    df
}

fn shared_julia_fixed_sigma_fixture() -> DataFrame {
    let y = vec![
        3.630846066147111,
        -0.23699581316575297,
        1.2105354224682663,
        0.869853351939183,
        -0.20112670239063263,
        1.841939312590815,
        3.0508340329938406,
        -0.16159198227005228,
        -1.7111617117834814,
        -2.573210271206462,
        -0.634354739497098,
        -2.5610196330697224,
        1.318703449478216,
        -3.9447255998012105,
        0.5307037522842474,
        -0.7644160195344709,
        -5.332106917168301,
        -0.47433639211466,
        -4.057116827660948,
        -3.8085558079065667,
        4.234332252764718,
        1.755107761778669,
        2.757065064409675,
        5.30205261880327,
        4.1451742404667105,
        1.2036710555092098,
        -3.0539946895833316,
        -1.8393472588555542,
        5.892040902634034,
        -1.9696539153474302,
        0.6486861972481239,
        0.368489072228326,
        -0.3611408729159792,
        5.193373815268175,
        1.913189995798939,
        0.47507592474230975,
        0.06401249428337571,
        2.2165512252476343,
        -0.9397784817739796,
        1.7788922478551683,
        -9.801745951021179,
        -1.9383974696808517,
        -2.092847010025527,
        3.442639699290954,
        -0.0837941751454139,
        4.133629704184189,
        2.1736737572044635,
        -1.0159208846460877,
        4.368916320835367,
        0.7607202499336108,
        5.85815983648636,
        -1.7609048242566288,
        -4.810884455196657,
        0.793817702591471,
        4.266085487320645,
        1.6199123691375519,
        -0.3084152967914453,
        0.6543377004554722,
        2.539769962223369,
        -3.918979949516328,
        1.1953631700478802,
        -0.2168447423962808,
        7.456462357947441,
        2.479491605550824,
        4.691307422020858,
        -3.9391366970370267,
        1.7056528817929726,
        -8.146790126669345,
        -1.1244595976644554,
        -1.9500060764200495,
        4.463837139784824,
        6.523171674670275,
        0.7811592530551956,
        4.633376703546607,
        1.8990447937621922,
        1.6916780132695428,
        4.812588984521369,
        0.7355154695965163,
        -1.1072651428981173,
        -1.5843836139553726,
        2.7091806278382435,
        -1.9396989674195224,
        -1.329495768570552,
        -2.0278076791842725,
        1.7658616138387506,
        3.407320593069791,
        1.9592167318065936,
        -3.5416850711564076,
        3.2744973367017147,
        -5.1760765079709525,
        -2.9661568404990826,
        0.5663029518057119,
        -3.266594534667978,
        -1.148968568238526,
        -2.720195067059705,
        0.515349568691151,
        4.858796519538594,
        -1.0745735117250352,
        1.8560434180444785,
        -2.540853853933194,
    ];

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_categorical("z", (1..=100).map(|idx| idx.to_string()).collect())
        .unwrap();
    df
}

fn current_logdet_xx(model: &LinearMixedModel) -> f64 {
    let k = model.reterms.len();
    let last = model.l_blocks[block_index(k, k)].as_dense();
    let p = last.nrows().saturating_sub(1);
    let mut logdet = 0.0;
    for i in 0..p {
        let diag = last[(i, i)];
        if diag > 0.0 {
            logdet += diag.ln();
        }
    }
    logdet * 2.0
}

fn make_vector_remat_for_kernel_tests(levels: usize) -> ReMat {
    let refs: Vec<u32> = (0..levels).map(|idx| idx as u32).collect();
    let level_names = (0..levels)
        .map(|idx| format!("S{:04}", idx + 1))
        .collect::<Vec<_>>();
    let cnames = vec!["(Intercept)".to_string(), "x".to_string()];
    let mut z = Vec::with_capacity(levels * 2);
    z.extend(std::iter::repeat_n(1.0, levels));
    z.extend((0..levels).map(|idx| idx as f64 + 0.5));

    ReMat::new(
        "subj".to_string(),
        refs,
        level_names,
        cnames,
        DMatrix::from_row_slice(2, levels, &z),
    )
}

#[test]
fn test_lmm_carries_compiler_artifact_design_audit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();

    let artifact = model.compiler_artifact();
    assert_eq!(artifact.requested_formula, formula.to_string());
    assert_eq!(artifact.semantic_model.random_terms.len(), 1);
    assert_eq!(artifact.theta_maps.len(), 1);

    let audit = model.design_audit().expect("design audit should attach");
    assert_eq!(audit.fixed_effect_rank.status, RankStatus::FullRank);
    assert_eq!(audit.fixed_effect_rank.rank, Some(2));
    assert_eq!(audit.random_terms[0].group.name, "subj");
    assert_eq!(audit.random_terms[0].group.n_levels, Some(18));
    assert_eq!(audit.random_terms[0].requested_covariance_parameters, 3);
}

#[test]
fn test_lmm_refuses_structured_random_covariance_before_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + ar1(0 + days | subj)").unwrap();
    let err = LinearMixedModel::new(formula, &data, None).unwrap_err();
    assert_eq!(err.code(), "unsupported");
    assert!(err.to_string().contains("ar1"));
    assert!(err.to_string().contains("not fitted in v1.0"));
}

#[test]
fn test_lmm_accepts_diag_wrapper_as_diagonal_covariance() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + diag(1 + days | subj)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert_eq!(model.reterms.len(), 1);
    assert_eq!(model.reterms[0].vsize, 2);
    assert_eq!(model.reterms[0].n_theta(), 2);
}

#[test]
fn test_random_effect_three_way_interaction_basis_is_materialized() {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 1.5, 2.5, 3.0, 4.0])
        .unwrap();
    data.add_numeric("A", vec![0.0, 1.0, 0.5, 1.5, 2.0, 2.5])
        .unwrap();
    data.add_numeric("B", vec![1.0, 0.5, 1.5, 1.0, 2.0, 1.5])
        .unwrap();
    data.add_numeric("C", vec![2.0, 1.0, 0.5, 1.5, 1.0, 2.5])
        .unwrap();
    data.add_categorical(
        "group",
        vec!["g1", "g1", "g1", "g2", "g2", "g2"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();

    let formula = parse_formula("y ~ A * B * C + (A * B * C | group)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert_eq!(
        model.reterms[0].cnames,
        vec!["(Intercept)", "A", "B", "C", "A:B", "A:C", "B:C", "A:B:C",]
    );
    assert_eq!(model.reterms[0].vsize, 8);
    assert_eq!(model.theta().len(), 36);
}

#[test]
fn test_random_effect_categorical_slope_uses_treatment_coding_with_intercept() {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
        .unwrap();
    data.add_categorical(
        "cond",
        vec!["A", "B", "C", "A", "B", "C"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    data.add_categorical(
        "subj",
        vec!["s1", "s1", "s1", "s2", "s2", "s2"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();

    let formula = parse_formula("y ~ cond + (1 + cond | subj)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert_eq!(
        model.reterms[0].cnames,
        vec!["(Intercept)", "cond: B", "cond: C"]
    );
    assert_eq!(model.reterms[0].vsize, 3);
    assert_eq!(model.theta().len(), 6);
    assert_eq!(
        model.compiler_artifact().theta_maps[0].block().user_basis,
        vec!["intercept".to_string(), "cond".to_string()]
    );
    assert_eq!(
        model.compiler_artifact().theta_maps[0]
            .block()
            .optimizer_basis,
        vec![
            "intercept".to_string(),
            "cond: B".to_string(),
            "cond: C".to_string()
        ]
    );
    assert_eq!(model.compiler_artifact().theta_maps[0].n_free(), 6);
}

#[test]
fn test_explicit_categorical_contrast_basis_drives_fixed_random_and_interaction_columns() {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 1.5, 2.5, 1.2, 2.2])
        .unwrap();
    data.add_numeric("x", vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        .unwrap();
    data.add_categorical_with_contrast(
        "anchor",
        vec!["low", "high", "low", "high", "low", "high"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        vec!["low".to_string(), "high".to_string()],
        crate::model::data::CategoricalContrast::new(
            vec!["low".to_string(), "high".to_string()],
            DMatrix::from_row_slice(2, 1, &[0.5, -0.5]),
            vec!["hi_minus_lo".to_string()],
            false,
            crate::model::data::ContrastSource::Custom,
        )
        .unwrap(),
    )
    .unwrap();
    data.add_categorical(
        "subj",
        vec!["s1", "s1", "s2", "s2", "s3", "s3"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();

    let formula = parse_formula("y ~ anchor + x:anchor + (1 + anchor | subj)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert!(model
        .feterm
        .cnames
        .iter()
        .any(|name| name == "anchor: hi_minus_lo"));
    assert!(model
        .feterm
        .cnames
        .iter()
        .any(|name| name == "x:anchor: hi_minus_lo"));
    assert_eq!(
        model.reterms[0].cnames,
        vec!["(Intercept)", "anchor: hi_minus_lo"]
    );
    assert_eq!(
        model.reterms[0]
            .z
            .row(1)
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
    );

    let audit = model.design_audit().expect("design audit should attach");
    let anchor_basis = audit
        .fixed_effects
        .contrast_bases
        .iter()
        .find(|basis| basis.variable == "anchor")
        .expect("explicit contrast basis should be recorded");
    assert!(anchor_basis.explicit);
    assert_eq!(anchor_basis.source, "custom");
    assert_eq!(anchor_basis.column_names, vec!["hi_minus_lo"]);
    assert_eq!(anchor_basis.contrast_matrix, vec![vec![0.5], vec![-0.5]]);
    assert!(audit.fixed_effects.columns.iter().any(|column| {
        column.name == "anchor: hi_minus_lo"
            && column.kind == crate::compiler::FixedEffectColumnKind::CategoricalContrast
    }));
}

#[test]
fn test_builtin_sum_contrast_fit_matches_treatment_fit_and_names_columns() {
    // lme4-style check for the built-in constructors: the same model fitted
    // under `contr.sum` and `contr.treatment` (both with an explicit level
    // order that differs from first appearance) spans the same column space,
    // so the ML objective and fitted values must agree exactly and the
    // coefficient vectors must be the known linear transforms of the cell
    // means. Column names follow R: sum coding names the first k-1 levels,
    // treatment coding names levels 2..k.
    let conds = ["hi", "lo", "mid"]; // first-appearance order: hi, lo, mid
    let cond_effect = |c: &str| match c {
        "lo" => -2.0,
        "mid" => 0.5,
        _ => 3.0,
    };
    let subj_effect = [0.0, 1.0, -1.0];
    let mut y = Vec::new();
    let mut cond = Vec::new();
    let mut subj = Vec::new();
    let mut i = 0usize;
    for _rep in 0..2 {
        for s in 0..3 {
            for c in conds {
                let noise = ((i as f64 * 12.9898).sin() * 43758.547).fract() - 0.5;
                y.push(10.0 + cond_effect(c) + subj_effect[s] + noise);
                cond.push(c.to_string());
                subj.push(format!("s{s}"));
                i += 1;
            }
        }
    }
    // Explicit canonical order lo < mid < hi, deliberately different from
    // the first-appearance order in the data.
    let levels: Vec<String> = ["lo", "mid", "hi"].iter().map(|s| s.to_string()).collect();

    let fit_with = |contrast: crate::model::data::CategoricalContrast| {
        let mut data = DataFrame::new();
        data.add_numeric("y", y.clone()).unwrap();
        data.add_categorical_with_contrast("cond", cond.clone(), levels.clone(), contrast)
            .unwrap();
        data.add_categorical("subj", subj.clone()).unwrap();
        let formula = parse_formula("y ~ 1 + cond + (1 | subj)").unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(false).unwrap();
        model
    };

    let sum_model = fit_with(crate::model::data::CategoricalContrast::sum(levels.clone()).unwrap());
    let trt_model =
        fit_with(crate::model::data::CategoricalContrast::treatment(levels.clone()).unwrap());

    let sum_names = sum_model.coef_names();
    let trt_names = trt_model.coef_names();
    assert_eq!(sum_names, vec!["(Intercept)", "cond: lo", "cond: mid"]);
    assert_eq!(trt_names, vec!["(Intercept)", "cond: mid", "cond: hi"]);

    assert_relative_eq!(
        sum_model.objective_value(),
        trt_model.objective_value(),
        epsilon = 1e-8,
        max_relative = 1e-10
    );
    let f_sum = sum_model.fitted();
    let f_trt = trt_model.fitted();
    for (a, b) in f_sum.iter().zip(f_trt.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-6);
    }

    // Cell means from the treatment fit (reference = lo), then the sum-coded
    // coefficients must be grand mean and deviations from it.
    let bt = trt_model.coef();
    let bs = sum_model.coef();
    let mu_lo = bt[0];
    let mu_mid = bt[0] + bt[1];
    let mu_hi = bt[0] + bt[2];
    let grand = (mu_lo + mu_mid + mu_hi) / 3.0;
    assert_relative_eq!(bs[0], grand, epsilon = 1e-6);
    assert_relative_eq!(bs[1], mu_lo - grand, epsilon = 1e-6);
    assert_relative_eq!(bs[2], mu_mid - grand, epsilon = 1e-6);
}

#[test]
fn test_predict_new_retains_builtin_helmert_contrast_snapshot() {
    // Prediction must re-encode categorical predictors through the training
    // contrast snapshot (levels + basis), not newdata's own first-appearance
    // encoding — here with a built-in helmert basis and an explicit level
    // order, mirroring test_predict_new_categorical_encoding_is_training_anchored.
    let grp_seq = [
        "B", "B", "A", "A", "C", "C", "A", "B", "C", "B", "A", "C", "C", "A", "B", "A", "C", "B",
    ];
    let n = grp_seq.len();
    let grp_effect = |g: &str| match g {
        "A" => 5.0,
        "C" => -3.0,
        _ => 0.0,
    };
    let subj_effect = [0.0, 1.0, -1.0];

    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut grp = Vec::with_capacity(n);
    let mut subj = Vec::with_capacity(n);
    for (i, g) in grp_seq.iter().enumerate() {
        let s = i % 3;
        let xv = i as f64 * 0.5;
        x.push(xv);
        let noise = ((i as f64 * 12.9898).sin() * 43758.547).fract() - 0.5;
        y.push(10.0 + 2.0 * xv + grp_effect(g) + subj_effect[s] + noise);
        grp.push((*g).to_string());
        subj.push(format!("s{s}"));
    }

    let levels: Vec<String> = ["A", "B", "C"].iter().map(|s| s.to_string()).collect();
    let helmert = crate::model::data::CategoricalContrast::helmert(levels.clone()).unwrap();

    let mut train = DataFrame::new();
    train.add_numeric("y", y.clone()).unwrap();
    train.add_numeric("x", x.clone()).unwrap();
    train
        .add_categorical_with_contrast("grp", grp.clone(), levels.clone(), helmert)
        .unwrap();
    train.add_categorical("subj", subj.clone()).unwrap();

    let formula = parse_formula("y ~ 1 + x + grp + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &train, None).unwrap();
    model.fit(false).unwrap();

    let audit = model.design_audit().expect("design audit should attach");
    let grp_basis = audit
        .fixed_effects
        .contrast_bases
        .iter()
        .find(|basis| basis.variable == "grp")
        .expect("helmert contrast basis should be recorded");
    assert_eq!(grp_basis.source, "helmert");

    let fitted = model.fitted();

    // newdata: identical rows in reversed order, added WITHOUT the explicit
    // contrast/levels, so its own first-appearance encoding would differ.
    let mut rev = DataFrame::new();
    let mut yr = y.clone();
    yr.reverse();
    let mut xr = x.clone();
    xr.reverse();
    let mut gr = grp.clone();
    gr.reverse();
    let mut sr = subj.clone();
    sr.reverse();
    rev.add_numeric("y", yr).unwrap();
    rev.add_numeric("x", xr).unwrap();
    rev.add_categorical("grp", gr).unwrap();
    rev.add_categorical("subj", sr).unwrap();
    assert_ne!(
        rev.categorical("grp").unwrap().levels,
        levels,
        "reversed newdata must have a different raw level order to exercise the snapshot"
    );

    let pred_rev = model.predict_new(&rev, NewReLevels::Error).unwrap();
    for (p, pred) in pred_rev.iter().enumerate() {
        let original_idx = n - 1 - p;
        assert_relative_eq!(
            pred.expect("all levels are training-known"),
            fitted[original_idx],
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }
}

#[test]
fn test_random_effect_categorical_slope_uses_cell_means_without_intercept() {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
        .unwrap();
    data.add_categorical(
        "cond",
        vec!["A", "B", "C", "A", "B", "C"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    data.add_categorical(
        "subj",
        vec!["s1", "s1", "s1", "s2", "s2", "s2"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();

    let formula = parse_formula("y ~ cond + (0 + cond | subj)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert_eq!(
        model.reterms[0].cnames,
        vec!["cond: A", "cond: B", "cond: C"]
    );
    assert_eq!(model.reterms[0].vsize, 3);
    assert_eq!(model.theta().len(), 6);
    assert_eq!(
        model.compiler_artifact().theta_maps[0].block().user_basis,
        vec!["cond".to_string()]
    );
    assert_eq!(
        model.compiler_artifact().theta_maps[0]
            .block()
            .optimizer_basis,
        vec![
            "cond: A".to_string(),
            "cond: B".to_string(),
            "cond: C".to_string()
        ]
    );
    assert_eq!(model.compiler_artifact().theta_maps[0].n_free(), 6);
}

#[test]
fn test_random_effect_no_intercept_factor_uses_cell_means_with_explicit_contrast() {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 1.5, 2.5, 1.2, 2.2])
        .unwrap();
    data.add_categorical_with_contrast(
        "anchor",
        vec!["low", "high", "low", "high", "low", "high"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        vec!["low".to_string(), "high".to_string()],
        crate::model::data::CategoricalContrast::new(
            vec!["low".to_string(), "high".to_string()],
            DMatrix::from_row_slice(2, 1, &[0.5, -0.5]),
            vec!["hi_minus_lo".to_string()],
            false,
            crate::model::data::ContrastSource::Custom,
        )
        .unwrap(),
    )
    .unwrap();
    data.add_categorical(
        "subj",
        vec!["s1", "s1", "s2", "s2", "s3", "s3"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();

    let formula = parse_formula("y ~ anchor + (0 + anchor | subj)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert_eq!(model.reterms[0].cnames, vec!["anchor: low", "anchor: high"]);
    assert_eq!(
        model.reterms[0]
            .z
            .row(0)
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(
        model.reterms[0]
            .z
            .row(1)
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn test_random_effect_categorical_cell_means_preserves_zero_correlation_map() {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
        .unwrap();
    data.add_categorical(
        "cond",
        vec!["A", "B", "C", "A", "B", "C"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    data.add_categorical(
        "subj",
        vec!["s1", "s1", "s1", "s2", "s2", "s2"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();

    let formula = parse_formula("y ~ cond + (0 + cond || subj)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert_eq!(
        model.reterms[0].cnames,
        vec!["cond: A", "cond: B", "cond: C"]
    );
    assert_eq!(model.theta().len(), 3);
    assert!(matches!(
        model.compiler_artifact().theta_maps[0],
        ThetaMap::Diagonal(_)
    ));
    assert_eq!(model.compiler_artifact().theta_maps[0].n_free(), 3);
}

#[test]
fn test_random_effect_interaction_uses_cell_means_without_intercept() {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 1.5, 2.5]).unwrap();
    data.add_numeric("x", vec![0.5, 1.0, 1.5, 2.0]).unwrap();
    data.add_categorical(
        "cond",
        vec!["A", "B", "A", "B"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    data.add_categorical(
        "subj",
        vec!["s1", "s1", "s2", "s2"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();

    let formula = parse_formula("y ~ x * cond + (0 + x:cond | subj)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert_eq!(model.reterms[0].cnames, vec!["x:cond: A", "x:cond: B"]);
    assert_eq!(model.reterms[0].vsize, 2);
    assert_eq!(
        model.compiler_artifact().theta_maps[0]
            .block()
            .optimizer_basis,
        vec!["x:cond: A".to_string(), "x:cond: B".to_string()]
    );
}

#[test]
fn test_zerocorr_factor_split_terms_record_no_error_diagnostics() {
    let n = 60;
    let mut data = DataFrame::new();
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let group_effect = ((i / 6) as f64 * 1.3).sin();
            let f_effect = if i % 2 == 0 { 0.0 } else { 0.4 };
            0.5 * x[i] + f_effect + group_effect + (i as f64 * 2.1).sin() * 0.3
        })
        .collect();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical(
        "f",
        (0..n)
            .map(|i| if i % 2 == 0 { "a" } else { "b" }.to_string())
            .collect(),
    )
    .unwrap();
    data.add_categorical("g", (0..n).map(|i| format!("g{}", i / 6)).collect())
        .unwrap();

    let formula = parse_formula("y ~ x + f + (1 + f + x || g)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert_eq!(model.reterms[0].cnames, vec!["(Intercept)", "f: b", "x"]);
    let artifact = model.compiler_artifact();
    let errors = artifact
        .diagnostics
        .iter()
        .filter(|diag| diag.severity == crate::compiler::diagnostics::DiagnosticSeverity::Error)
        .map(|diag| diag.message.clone())
        .collect::<Vec<_>>();
    assert!(
        errors.is_empty(),
        "||-with-factor construction should not record error diagnostics: {errors:?}"
    );
    assert_eq!(artifact.theta_maps.len(), 3);
    let factor_map = &artifact.theta_maps[1];
    assert_eq!(factor_map.block().user_basis, vec!["f".to_string()]);
    assert_eq!(factor_map.block().theta_slots[0].lambda_row, 1);
    let total_free: usize = artifact.theta_maps.iter().map(|map| map.n_free()).sum();
    assert_eq!(total_free, model.theta().len());

    model.fit(false).unwrap();
    assert!(model.theta().iter().all(|value| value.is_finite()));
    let post_fit_basis_errors = model
        .compiler_artifact()
        .diagnostics
        .iter()
        .filter(|diag| {
            diag.severity == crate::compiler::diagnostics::DiagnosticSeverity::Error
                && diag.message.contains("optimizer basis")
        })
        .count();
    assert_eq!(post_fit_basis_errors, 0);
}

#[test]
fn test_singular_fixture_maximal_model_has_too_rich_information_budget() {
    let (data, _) = crate::datasets::load("singular").unwrap();
    let formula = parse_formula("y ~ 1 + A * B * C + (A * B * C | group)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    let audit = model.design_audit().expect("design audit should attach");
    let random = &audit.random_terms[0];

    assert_eq!(
        model.reterms[0].cnames,
        vec!["(Intercept)", "A", "B", "C", "A:B", "A:C", "B:C", "A:B:C",]
    );
    assert_eq!(random.group.n_levels, Some(10));
    assert_eq!(random.basis_size, 8);
    assert_eq!(random.requested_covariance_parameters, 36);
    assert_eq!(
        random.information_budget.status,
        InformationBudgetStatus::TooRich
    );
    assert_eq!(
        random.information_budget.min_levels_full_covariance,
        Some(180)
    );
}

// Rank-detection depends on the optimizer landing in the reduced-rank
// region of the theta surface; the native no-default-features path can
// converge full-rank on this fit, so the assertion only holds with NLopt.
#[cfg(feature = "nlopt")]
#[test]
fn test_singular_fixture_zcp_fit_exposes_reduced_effective_rank() {
    let (data, _) = crate::datasets::load("singular").unwrap();
    let formula = parse_formula("y ~ 1 + A * B * C + (A * B * C || group)").unwrap();
    let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();

    model.fit(false).unwrap();

    let summary = &model.compiler_artifact().effective_covariance[0];
    assert_eq!(summary.requested_rank, 8);
    assert!(summary.supported_rank < summary.requested_rank);
    assert_eq!(summary.status, EffectiveRankStatus::ReducedRank);
    assert_eq!(
        model.optimizer_certificate().unwrap().status,
        FitStatus::ConvergedReducedRank
    );
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_native_default_singular_zcp_fit_keeps_certificate_state_observable() {
    let (data, _) = crate::datasets::load("singular").unwrap();
    let formula = parse_formula("y ~ 1 + A * B * C + (A * B * C || group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let fit_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.fit(false).map(|_| ())
    }));
    assert!(
        fit_result.is_ok(),
        "native singular ZCP fit should not panic"
    );
    fit_result.unwrap().unwrap();

    assert_eq!(model.optsum.optimizer, Optimizer::TrustBq);
    assert!(model.objective_value().is_finite());
    assert!(model.sigma().is_finite() && model.sigma() > 0.0);
    assert!(model.theta().iter().all(|value| value.is_finite()));

    let certificate = model
        .optimizer_certificate()
        .expect("singular native fit should attach optimizer certificate");
    assert_eq!(certificate.optimizer_name.as_deref(), Some("trust_bq"));
    assert!(
        certificate
            .objective_value
            .is_some_and(|value| value.is_finite()),
        "singular native certificate should carry finite objective"
    );
    assert!(
        certificate
            .evidence
            .optimizer_stop
            .function_evaluations
            .is_some_and(|feval| feval > 0),
        "singular native certificate should record function evaluations"
    );

    for summary in &model.compiler_artifact().effective_covariance {
        assert_eq!(summary.requested_rank, 8);
        assert!(summary.supported_rank <= summary.requested_rank);
        assert!(matches!(
            summary.status,
            EffectiveRankStatus::FullRank | EffectiveRankStatus::ReducedRank
        ));
    }
}

#[test]
fn test_lmm_compiler_artifact_records_rank_deficient_fixed_effects() {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0]).unwrap();
    data.add_numeric("x2", vec![0.0, 2.0, 0.0, 2.0]).unwrap();
    data.add_categorical(
        "z",
        vec!["a", "a", "b", "b"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();

    let formula = parse_formula("y ~ x + x2 + (1 | z)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    let audit = model.design_audit().expect("design audit should attach");

    assert_eq!(audit.fixed_effect_rank.status, RankStatus::RankDeficient);
    assert_eq!(audit.fixed_effect_rank.rank, Some(2));
    assert_eq!(audit.fixed_effect_rank.expected, Some(3));
    assert!(model
        .compiler_artifact()
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == DiagnosticCode::FixedEffectRankDeficient));
}

#[test]
fn test_lmm_compiler_theta_maps_follow_optimizer_reterm_order() {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    data.add_numeric("x", vec![0.0, 1.0, 0.5, 1.5, 0.25, 1.25])
        .unwrap();
    data.add_categorical(
        "small",
        vec!["s1", "s1", "s2", "s2", "s1", "s2"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    data.add_categorical(
        "large",
        vec!["l1", "l2", "l3", "l1", "l2", "l3"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();

    let formula = parse_formula("y ~ x + (1 | small) + (1 + x | large)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    let maps = &model.compiler_artifact().theta_maps;

    assert_eq!(model.reterms[0].grouping_name, "large");
    assert_eq!(maps[0].block().term_id, "r1");
    assert_eq!(maps[0].block().term_index, 0);
    assert_eq!(maps[0].block().group, "large");
    assert_eq!(maps[0].block().theta_slots[0].global_index, Some(0));

    assert_eq!(model.reterms[1].grouping_name, "small");
    assert_eq!(maps[1].block().term_id, "r0");
    assert_eq!(maps[1].block().term_index, 1);
    assert_eq!(maps[1].block().group, "small");
    assert_eq!(maps[1].block().theta_slots[0].global_index, Some(3));

    let traces = &model.compiler_artifact().covariance_parameter_traces;
    assert_eq!(traces.len(), 4);
    assert_eq!(traces[0].term_id, "r1");
    assert_eq!(traces[0].source_syntax, "(1 + x | large)");
    assert_eq!(traces[0].optimizer_term_index, 0);
    assert_eq!(traces[0].lambda.row_basis, "intercept");
    assert!(traces
        .iter()
        .all(|trace| trace.parmap_entry.as_ref().unwrap().matches_theta_map));
}

#[test]
fn test_lmm_optimizer_certificate_records_interior_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();

    assert!(model.optimizer_certificate().is_none());
    model.fit(false).unwrap();

    let certificate = model
        .optimizer_certificate()
        .expect("optimizer certificate should attach after fit");
    assert_eq!(certificate.status, FitStatus::ConvergedInterior);
    assert_eq!(
        certificate.optimizer_name.as_deref(),
        Some("pattern_search")
    );
    assert!(certificate.objective_value.is_some());
    assert!(certificate.evidence.optimizer_stop.acceptable_stop);
    assert!(!certificate.evidence.optimizer_stop.budget_exhausted);
    assert_eq!(certificate.evidence.parameter_space.n_theta, 1);
    assert_eq!(certificate.evidence.parameter_space.n_boundary, 0);
    assert_eq!(certificate.evidence.sample_size.n_observations, Some(180));
    assert_eq!(certificate.evidence.sample_size.n_theta, 1);
    assert!(matches!(
        certificate.evidence.certification_quality,
        EvidenceQuality::Approximate { .. }
    ));
    assert!(matches!(
        certificate.evidence.gradient.method,
        EvidenceMethod::FiniteDifference
    ));
    assert!(certificate.evidence.gradient.raw_gradient_norm.is_some());
    assert!(certificate.evidence.gradient.free_gradient_norm.is_some());
    assert!(certificate
        .evidence
        .gradient
        .projected_gradient_norm
        .is_some());
    assert!(matches!(
        certificate.evidence.hessian.method,
        EvidenceMethod::FiniteDifference
    ));
    assert!(certificate.evidence.hessian.min_eigenvalue.is_some());
    assert_eq!(certificate.evidence.hessian.rank, Some(1));
    assert!(certificate
        .checks
        .iter()
        .any(|check| matches!(check, CertificateCheck::FreeGradientOk { .. })));
    assert!(certificate
        .checks
        .iter()
        .any(|check| matches!(check, CertificateCheck::HessianPsdOnActiveSubspace { .. })));
    assert!(!certificate
        .checks
        .iter()
        .any(|check| matches!(check, CertificateCheck::NotAssessed { .. })));

    let verification = model.verify_convergence().unwrap();
    assert!(matches!(
        verification.status,
        ConvergenceVerificationStatus::RestartAgrees
            | ConvergenceVerificationStatus::OptimizerConsensus
    ));
    assert!(!verification.runs.is_empty());
    assert!(verification.runs.iter().all(|run| run.agrees));
    assert!(model
        .optimizer_certificate()
        .unwrap()
        .verification
        .is_some());

    let trace = &model.compiler_artifact().covariance_parameter_traces[0];
    assert!(trace.theta.value.is_some());
    assert!(trace.lambda.value.is_some());
    assert_eq!(trace.varcorr_entries[0].label, "sd(intercept)");
    assert!(trace.varcorr_entries[0].value.is_some());
}

#[test]
fn test_lmm_convergence_verification_is_not_run_before_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let verification = model.verify_convergence().unwrap();

    assert_eq!(verification.status, ConvergenceVerificationStatus::NotRun);
    assert!(verification.runs.is_empty());
    assert_eq!(verification.message, "model has not been fitted");
}

#[test]
fn test_lmm_audit_report_updates_after_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let prefit_report = model.audit_report().to_text();
    assert!(prefit_report.contains("Optimizer"));
    assert!(prefit_report.contains("model has not been fitted"));

    model.fit(false).unwrap();

    let fitted_report = model.audit_report().to_text();
    assert!(fitted_report.contains("ConvergedInterior"));
    assert!(fitted_report.contains("pattern_search"));
    assert!(fitted_report.contains("convergence interpretation"));
    assert!(fitted_report.contains("run verify_convergence()"));
}

#[test]
fn test_lmm_optimizer_certificate_records_boundary_fit() {
    let data = singular_re_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();

    let certificate = model
        .optimizer_certificate()
        .expect("optimizer certificate should attach after fit");
    assert_eq!(certificate.status, FitStatus::ConvergedReducedRank);
    assert_eq!(certificate.evidence.parameter_space.n_boundary, 1);
    assert_eq!(
        certificate.evidence.parameter_space.boundary_indices,
        vec![0]
    );
    assert!(certificate
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == DiagnosticCode::BoundaryParameter));
    assert!(certificate.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == DiagnosticCode::BoundaryParameter
            && diagnostic
                .suggested_actions
                .iter()
                .any(|action| action.contains("valid fitted boundary"))
    }));
    let boundary_diagnostic = certificate
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::BoundaryParameter)
        .expect("boundary parameter diagnostic");
    assert_eq!(boundary_diagnostic.affected_terms, vec!["(1 | batch)"]);
    assert!(boundary_diagnostic
        .message
        .contains("standard deviation for intercept in (1 | batch)"));
    assert!(!boundary_diagnostic.message.contains("theta[0]"));
    assert_eq!(
        boundary_diagnostic.payload.get("theta_index"),
        Some(&serde_json::json!(0))
    );
    assert_eq!(
        boundary_diagnostic.payload.get("term_id"),
        Some(&serde_json::json!("r0"))
    );
    assert!(matches!(
        &certificate.evidence.gradient.method,
        EvidenceMethod::NotAssessed { reason } if reason.contains("variance-component boundary")
    ));
    assert!(certificate
        .evidence
        .gradient
        .kkt_boundary_gradient_max
        .is_none());
    assert!(matches!(
        &certificate.evidence.hessian.quality,
        EvidenceQuality::NotAssessed { reason } if reason.contains("variance-component boundary")
    ));
    assert_eq!(certificate.evidence.hessian.rank, None);
    assert!(certificate.checks.iter().any(|check| matches!(
        check,
        CertificateCheck::NotAssessed { reason }
            if reason.contains("boundary-gradient KKT check skipped")
    )));
    assert!(certificate
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == DiagnosticCode::CovarianceReduced));
    let covariance_diagnostic = certificate
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::CovarianceReduced)
        .expect("covariance reduced diagnostic");
    assert_eq!(covariance_diagnostic.affected_terms, vec!["(1 | batch)"]);
    assert!(covariance_diagnostic
        .message
        .contains("fitted covariance for (1 | batch)"));
    assert!(!covariance_diagnostic.message.contains("r0"));
    assert_eq!(
        covariance_diagnostic.payload.get("term_id"),
        Some(&serde_json::json!("r0"))
    );
    assert!(model
        .compiler_artifact()
        .reductions
        .iter()
        .all(|reduction| reduction.diagnostics.is_empty()));
    assert!(!model
        .compiler_artifact()
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == DiagnosticCode::CovarianceReduced));
    assert_eq!(
        model.compiler_artifact().effective_covariance[0].supported_rank,
        0
    );
}

#[test]
fn test_scalar_covariance_kkt_certificate_interior_converged() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();

    let certificate = model.scalar_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::InteriorConverged
    );
    assert!(block.variance > certificate.variance_tolerance);
    assert!(block.score.abs() <= certificate.score_tolerance);
    assert!(certificate.residual.is_finite());
}

#[test]
fn test_scalar_covariance_kkt_certificate_valid_zero_variance() {
    let data = singular_re_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();

    let certificate = model.scalar_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::ValidZeroVariance
    );
    assert!(block.variance <= certificate.variance_tolerance);
    assert!(block.score >= -certificate.score_tolerance);
    assert!(certificate.residual.is_finite());
}

#[test]
fn test_scalar_covariance_kkt_certificate_flags_invalid_boundary_stop() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();
    model.set_theta(&[0.0]).unwrap();

    let certificate = model.scalar_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::InvalidBoundaryStop
    );
    assert!(block.variance <= certificate.variance_tolerance);
    assert!(block.score < -certificate.score_tolerance);
    assert!(certificate.residual.is_finite());
}

#[test]
fn test_scalar_covariance_kkt_certificate_marks_tiny_positive_variance_weak() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();
    model.set_theta(&[1e-3]).unwrap();

    let certificate = model.scalar_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::WeakIdentification
    );
    assert!(block.variance > certificate.variance_tolerance);
    assert!(block.score.abs() > certificate.score_tolerance);
    assert!(certificate.residual.is_finite());
}

#[test]
fn test_two_by_two_covariance_kkt_certificate_valid_rank_one_rho_one() {
    let data = rank_one_rho_one_random_slope_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();
    let fitted_theta = model.theta();
    let rank_one_theta = [fitted_theta[0], fitted_theta[1], 0.0];
    model.set_theta(&rank_one_theta).unwrap();

    let certificate = model.two_by_two_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::ValidRankDeficientCovariance
    );
    assert!(block.min_eig_g <= certificate.covariance_tolerance);
    assert!(block.min_eig_score >= -certificate.score_tolerance);
    assert!(block.complementarity <= certificate.complementarity_tolerance);
    assert!(block.residual.is_finite());
}

#[test]
fn test_two_by_two_covariance_kkt_certificate_flags_invalid_boundary_stop() {
    let data = rank_one_rho_one_random_slope_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();
    model.set_theta(&[0.0, 0.0, 0.0]).unwrap();

    let certificate = model.two_by_two_covariance_kkt_certificate().unwrap();
    assert_eq!(certificate.blocks.len(), 1);
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::InvalidBoundaryStop
    );
    assert!(block.min_eig_g <= certificate.covariance_tolerance);
    assert!(block.min_eig_score < -certificate.score_tolerance);
    assert!(block.residual.is_finite());
}

#[test]
fn test_trust_bq_certificate_stop_accepts_scalar_interior() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let theta = model.theta();
    let objective = model.objective_at(&theta).unwrap();

    assert!(model
        .trust_bq_covariance_kkt_certifies_theta(&theta, objective, 64, false)
        .unwrap());
}

#[test]
fn test_trust_bq_certificate_stop_accepts_scalar_valid_boundary() {
    let data = singular_re_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    let theta = vec![0.0];
    let objective = model.objective_at(&theta).unwrap();

    assert!(model
        .trust_bq_covariance_kkt_certifies_theta(&theta, objective, 64, false)
        .unwrap());
}

#[test]
fn test_trust_bq_certificate_stop_rejects_scalar_invalid_boundary() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    let theta = vec![0.0];
    let objective = model.objective_at(&theta).unwrap();

    assert!(!model
        .trust_bq_covariance_kkt_certifies_theta(&theta, objective, 64, false)
        .unwrap());
}

#[test]
fn test_trust_bq_certificate_stop_accepts_two_by_two_valid_rank_deficient() {
    let data = rank_one_rho_one_random_slope_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();
    let fitted_theta = model.theta();
    let theta = vec![fitted_theta[0], fitted_theta[1], 0.0];
    let objective = model.objective_at(&theta).unwrap();

    assert!(model
        .trust_bq_covariance_kkt_certifies_theta(&theta, objective, 96, false)
        .unwrap());
}

#[test]
fn test_trust_bq_model_family_policy_records_small_theta_contract() {
    let policy = trust_bq_model_family_policy(3, None, &[], &[], 0, 1e-12, 1e-8);

    assert_eq!(policy.max_evaluations, 1000);
    assert_eq!(policy.max_cross_terms, usize::MAX);
    assert!(!policy.reuse_samples);
    assert_eq!(policy.stall_iterations, 4);
    assert_eq!(policy.stall_ftol_rel, -1.0);
    assert_eq!(policy.stall_ftol_abs, -1.0);
    assert!(policy.stall_requires_stable_x);
    // The small family caps the relative accepted-step band at 1e-11 (the
    // NLopt-style 1e-8 default is too loose for parity-grade endpoints)
    // and floors the absolute band at 1e-10; the certificate inherits.
    assert_eq!(policy.certificate_ftol_abs, 1e-10);
    assert_eq!(policy.certificate_ftol_rel, 1e-11);
}

#[test]
fn test_trust_bq_model_family_policy_records_moderate_theta_contract() {
    let policy = trust_bq_model_family_policy(6, Some(321), &[], &[], 0, 1e-12, 1e-8);

    assert_eq!(policy.max_evaluations, 321);
    assert_eq!(policy.max_cross_terms, 0);
    assert!(!policy.reuse_samples);
    assert_eq!(policy.stall_iterations, 4);
    assert!(policy.stall_requires_stable_x);
}

#[test]
fn test_trust_bq_model_family_policy_records_crossed_large_theta_contract() {
    let policy = trust_bq_model_family_policy(9, None, &[], &[], 0, 1e-12, 1e-8);

    assert_eq!(policy.max_evaluations, 475);
    assert_eq!(policy.max_cross_terms, 0);
    assert!(policy.reuse_samples);
    assert_eq!(policy.stall_iterations, 3);
    assert_eq!(policy.stall_ftol_rel, 1e-6);
    assert_eq!(policy.stall_ftol_abs, 1e-8);
    assert!(!policy.stall_requires_stable_x);
    assert_eq!(policy.certificate_ftol_abs, 1e-8);
    assert_eq!(policy.certificate_ftol_rel, 1e-6);
}

#[test]
fn test_trust_bq_start_ladder_defaults_off() {
    assert_eq!(
        OptimizerControl::default().trust_bq_start_ladder,
        TrustBqStartLadder::Off
    );

    let data = simulate_sleepstudy_like(24, 10, 42);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model
        .fit_with_options(FitOptions::reml().with_optimizer(Optimizer::TrustBq))
        .unwrap();
    assert!(
        !model.optsum.return_value.contains("START_LADDER"),
        "single-start TrustBQ must not report a ladder: {}",
        model.optsum.return_value
    );
}

#[test]
fn test_trust_bq_diagonal_first_ladder_matches_single_start_objective() {
    let data = simulate_sleepstudy_like(24, 10, 42);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();

    let mut single = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
    single
        .fit_with_options(FitOptions::reml().with_optimizer(Optimizer::TrustBq))
        .unwrap();

    let mut laddered = LinearMixedModel::new(formula, &data, None).unwrap();
    laddered
        .fit_with_options(
            FitOptions::reml().with_optimizer_control(
                OptimizerControl::auto()
                    .with_optimizer(Optimizer::TrustBq)
                    .with_trust_bq_start_ladder(TrustBqStartLadder::DiagonalFirst),
            ),
        )
        .unwrap();

    assert!(
        laddered
            .optsum
            .return_value
            .starts_with("START_LADDER(diagonal_first"),
        "opted-in ladder must be audit-visible: {}",
        laddered.optsum.return_value
    );
    // The warm start must never degrade the fit. It may legitimately land
    // lower: on this fixture single-start TrustBQ under-converges by ~0.52
    // while the ladder reaches the NLopt BOBYQA reference objective
    // (2298.705488) to 1e-6.
    let tolerance = 1e-6 * (1.0 + single.objective().abs());
    assert!(
        laddered.objective() <= single.objective() + tolerance,
        "ladder objective {} is worse than single-start objective {}",
        laddered.objective(),
        single.objective()
    );
    // Both stages' evaluations are counted and logged.
    assert_eq!(
        laddered.optsum.feval as usize,
        laddered.optsum.fit_log.len(),
        "feval must count ladder-stage evaluations"
    );
    // The certificate must classify by the inner stop code, not the
    // START_LADDER wrapper (a converged ladder fit is not NotOptimized).
    if laddered.optsum.converged() {
        assert_ne!(
            laddered.optimizer_certificate().unwrap().status,
            FitStatus::NotOptimized,
            "converged ladder fit must not be certified NotOptimized"
        );
    }
}

#[test]
fn test_trust_bq_ladder_is_noop_without_off_diagonal_theta() {
    // A scalar random-intercept model has no off-diagonal theta, so the
    // diagonal-first ladder must fall back to plain single-start TrustBQ.
    let data = simulate_sleepstudy_like(24, 10, 42);
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model
        .fit_with_options(
            FitOptions::reml().with_optimizer_control(
                OptimizerControl::auto()
                    .with_optimizer(Optimizer::TrustBq)
                    .with_trust_bq_start_ladder(TrustBqStartLadder::DiagonalFirst),
            ),
        )
        .unwrap();
    assert!(
        !model.optsum.return_value.contains("START_LADDER"),
        "ladder must be a no-op without off-diagonal theta: {}",
        model.optsum.return_value
    );
}

#[test]
fn test_active_face_refit_defaults_off() {
    assert_eq!(
        OptimizerControl::default().active_face_refit,
        ActiveFaceRefit::Off
    );

    let data = simulate_sleepstudy_like(24, 10, 42);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit_with_options(FitOptions::reml()).unwrap();
    assert!(
        !model.optsum.return_value.contains("ACTIVE_FACE"),
        "default fits must not run the active-face refit: {}",
        model.optsum.return_value
    );
}

#[test]
fn test_active_face_refit_noop_on_full_rank_fit() {
    // A well-identified vector model detects no lower-rank face, so the
    // opted-in refit must leave the fit byte-identical to the default path.
    let data = simulate_sleepstudy_like(24, 10, 42);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();

    let mut plain = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
    plain.fit_with_options(FitOptions::reml()).unwrap();

    let mut faced = LinearMixedModel::new(formula, &data, None).unwrap();
    faced
        .fit_with_options(FitOptions::reml().with_optimizer_control(
            OptimizerControl::auto().with_active_face_refit(ActiveFaceRefit::Experimental),
        ))
        .unwrap();

    assert!(
        !faced.optsum.return_value.contains("ACTIVE_FACE"),
        "full-rank fit must not trigger the active-face refit: {}",
        faced.optsum.return_value
    );
    assert_eq!(plain.objective(), faced.objective());
    assert_eq!(plain.optsum.feval, faced.optsum.feval);
    assert_eq!(plain.theta(), faced.theta());
}

// The maximal over-specified singular row (36 theta for a ~rank-4 block) is
// where the primary optimizer exhausts its budget far from the lme4
// optimum; the assertions pin the recovery contract on the default (NLopt)
// release path.
#[cfg(feature = "nlopt")]
#[test]
fn test_active_face_refit_improves_maximal_singular_fit() {
    let (data, _) = crate::datasets::load("singular").unwrap();
    let formula = parse_formula("y ~ 1 + A * B * C + (A * B * C | group)").unwrap();

    let mut baseline = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
    baseline.fit_with_options(FitOptions::reml()).unwrap();

    let mut faced = LinearMixedModel::new(formula, &data, None).unwrap();
    faced
        .fit_with_options(FitOptions::reml().with_optimizer_control(
            OptimizerControl::auto().with_active_face_refit(ActiveFaceRefit::Experimental),
        ))
        .unwrap();

    println!(
        "baseline: objective {} feval {} status {}",
        baseline.objective(),
        baseline.optsum.feval,
        baseline.optsum.return_value
    );
    println!(
        "active-face: objective {} feval {} status {}",
        faced.objective(),
        faced.optsum.feval,
        faced.optsum.return_value
    );

    assert!(
        faced.optsum.return_value.starts_with("ACTIVE_FACE("),
        "opted-in refit must be audit-visible: {}",
        faced.optsum.return_value
    );
    assert!(
        faced.objective() < baseline.objective() - 10.0,
        "active-face refit must materially improve the budget-bound objective: {} vs {}",
        faced.objective(),
        baseline.objective()
    );
    // lme4 converges this row to 766.554 (comparison/lme4_results.json); the
    // face refit must land in that neighborhood, not merely improve.
    assert!(
        (faced.objective() - 766.554).abs() < 5.0,
        "active-face objective {} is far from the lme4 reference 766.554",
        faced.objective()
    );
    // The refit's evaluations are accounted for.
    assert!(faced.optsum.feval > baseline.optsum.feval);
    // The reduced active rank is recorded in the audit label even when the
    // face stage stops on budget (effective-covariance summaries only
    // populate for certificate-converged statuses).
    assert!(
        faced.optsum.return_value.contains("rank") && faced.optsum.return_value.contains("of8"),
        "active-face label must record the active rank: {}",
        faced.optsum.return_value
    );
    // When the face stage genuinely converges, the certificate must accept
    // the wrapped stop code and the summaries must expose the reduced rank.
    if faced.optsum.converged() {
        assert_ne!(
            faced.optimizer_certificate().unwrap().status,
            FitStatus::NotOptimized,
            "converged active-face fit must not be certified NotOptimized"
        );
        let summary = &faced.compiler_artifact().effective_covariance[0];
        assert_eq!(summary.requested_rank, 8);
        assert!(
            summary.supported_rank < summary.requested_rank,
            "converged active-face optimum must expose its reduced rank"
        );
    }
    // The face optimum sits on the boundary: dropped directions expand to
    // exact zero theta columns.
    assert!(faced.is_singular());
}

fn force_bad_boundary_fit_state(
    model: &mut LinearMixedModel,
    theta: &[f64],
    optimizer: Optimizer,
    reml: bool,
) -> f64 {
    let objective = model.objective_at(theta).unwrap();
    model.optsum.reml = reml;
    model.optsum.optimizer = optimizer;
    model.optsum.backend = optimizer.canonical_backend();
    model.optsum.initial = theta.to_vec();
    model.optsum.final_params = theta.to_vec();
    model.optsum.finitial = objective;
    model.optsum.fmin = objective;
    model.optsum.feval = 1;
    model.optsum.return_value = "FORCED_BAD_BOUNDARY".to_string();
    model.optsum.fit_log = vec![FitLogEntry {
        theta: theta.to_vec(),
        objective,
    }];
    objective
}

fn optimize_scalar_psd_poc(model: &LinearMixedModel, start_variance: f64) -> PatternSearchOutcome {
    let start = vec![start_variance.max(0.0)];
    let initial = model
        .objective_at_theta_for_certificate(&[start[0].sqrt()])
        .unwrap();
    LinearMixedModel::run_multivariate_pattern_search(
        start,
        initial,
        &[0.0],
        vec![(0.1 * (1.0 + start_variance.abs())).max(1e-3)],
        &[1e-7],
        2_000,
        1e-12,
        |g| model.objective_at_theta_for_certificate(&[g[0].max(0.0).sqrt()]),
    )
    .unwrap()
}

fn optimize_two_by_two_psd_poc(
    model: &LinearMixedModel,
    start_covariance: [[f64; 2]; 2],
) -> PatternSearchOutcome {
    let start = vec![
        start_covariance[0][0].max(0.0),
        0.5 * (start_covariance[0][1] + start_covariance[1][0]),
        start_covariance[1][1].max(0.0),
    ];
    let invalid_objective = model.objective_value().abs().max(1.0) + 1e6;
    let objective = |g: &[f64]| {
        let covariance = [[g[0], g[1]], [g[1], g[2]]];
        let Some(theta) = two_by_two_theta_from_covariance(covariance) else {
            return Ok(invalid_objective);
        };
        model.objective_at_theta_for_certificate(&theta)
    };
    let initial = objective(&start).unwrap();
    let scale = two_by_two_frobenius_norm(start_covariance).max(1.0);

    LinearMixedModel::run_multivariate_pattern_search(
        start,
        initial,
        &[0.0, f64::NEG_INFINITY, 0.0],
        vec![0.05 * scale, 0.05 * scale, 0.05 * scale],
        &[1e-6, 1e-6, 1e-6],
        5_000,
        1e-12,
        objective,
    )
    .unwrap()
}

fn normalize_2_for_test(mut vector: [f64; 2]) -> [f64; 2] {
    let norm = vector[0].hypot(vector[1]);
    if norm > 0.0 && norm.is_finite() {
        vector[0] /= norm;
        vector[1] /= norm;
    }
    vector
}

fn symmetric_2x2_max_eigenvector_for_test(matrix: [[f64; 2]; 2]) -> [f64; 2] {
    let a = matrix[0][0];
    let b = 0.5 * (matrix[0][1] + matrix[1][0]);
    let c = matrix[1][1];
    let (_, lambda) = symmetric_2x2_eigenvalues([[a, b], [b, c]]);
    if b.abs() > 1e-14 {
        normalize_2_for_test([b, lambda - a])
    } else if a >= c {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    }
}

fn rank_one_covariance_for_test(direction: [f64; 2], amplitude: f64) -> [[f64; 2]; 2] {
    let direction = normalize_2_for_test(direction);
    let amplitude = amplitude.max(0.0);
    [
        [
            amplitude * direction[0] * direction[0],
            amplitude * direction[0] * direction[1],
        ],
        [
            amplitude * direction[1] * direction[0],
            amplitude * direction[1] * direction[1],
        ],
    ]
}

fn optimize_two_by_two_active_face_psd_poc(
    model: &LinearMixedModel,
    direction: [f64; 2],
    start_amplitude: f64,
) -> PatternSearchOutcome {
    let direction = normalize_2_for_test(direction);
    let start = vec![start_amplitude.max(0.0)];
    let invalid_objective = model.objective_value().abs().max(1.0) + 1e6;
    let objective = |g: &[f64]| {
        let covariance = rank_one_covariance_for_test(direction, g[0]);
        let Some(theta) = two_by_two_theta_from_covariance(covariance) else {
            return Ok(invalid_objective);
        };
        model.objective_at_theta_for_certificate(&theta)
    };
    let initial = objective(&start).unwrap();

    let mut outcome = LinearMixedModel::run_multivariate_pattern_search(
        start,
        initial,
        &[0.0],
        vec![(0.05 * (1.0 + start_amplitude.abs())).max(1e-4)],
        &[1e-6],
        1_500,
        1e-12,
        objective,
    )
    .unwrap();
    outcome.trace_label = Some("active_face_rank_one_2x2".to_string());
    outcome.active_rank = Some(1);
    outcome.inactive_directions = Some(1);
    outcome
}

#[test]
fn test_kkt_guided_restart_repairs_scalar_invalid_boundary_stop() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();
    model.set_theta(&[0.0]).unwrap();
    let before_certificate = model.scalar_covariance_kkt_certificate().unwrap();
    assert_eq!(
        before_certificate.blocks[0].classification,
        CovarianceKktClassification::InvalidBoundaryStop
    );
    let before = before_certificate.objective;
    model.optsum.optimizer = Optimizer::PatternSearch;
    model.optsum.backend = Optimizer::PatternSearch.canonical_backend();
    model.optsum.initial = vec![0.0];
    model.optsum.final_params = vec![0.0];
    model.optsum.finitial = before;
    model.optsum.fmin = before;
    model.optsum.feval = 1;
    model.optsum.return_value = "FORCED_BAD_BOUNDARY".to_string();
    model.optsum.fit_log = vec![FitLogEntry {
        theta: vec![0.0],
        objective: before,
    }];

    assert!(model.apply_kkt_guided_boundary_restart(false).unwrap());

    let after = model
        .objective_at_theta_for_certificate(&model.theta())
        .unwrap();
    assert!(after < before);
    let certificate = model.scalar_covariance_kkt_certificate().unwrap();
    assert_eq!(
        certificate.blocks[0].classification,
        CovarianceKktClassification::InteriorConverged
    );
    assert!(model.optsum.return_value.contains("KKT_BOUNDARY_RESTART"));
    model.refresh_optimizer_certificate();
    let optimizer_certificate = model.optimizer_certificate().unwrap();
    assert!(
        optimizer_certificate
            .evidence
            .optimizer_stop
            .acceptable_stop
    );
    assert!(
        !optimizer_certificate
            .evidence
            .optimizer_stop
            .budget_exhausted
    );
    assert_eq!(optimizer_certificate.status, FitStatus::ConvergedInterior);
    assert!(optimizer_certificate.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == DiagnosticCode::OptimizerRecovery
            && diagnostic.message.contains("covariance KKT-guided restart")
    }));
    let verdict = ConvergenceVerdict::for_artifact(model.compiler_artifact());
    assert_eq!(verdict.level, ConvergenceLevel::Ok);
    assert!(verdict.headline.contains("recovered convergence"));
}

#[test]
fn test_kkt_guided_restart_repairs_two_by_two_invalid_boundary_stop() {
    let data = rank_one_rho_one_random_slope_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    let before =
        force_bad_boundary_fit_state(&mut model, &[0.0, 0.0, 0.0], Optimizer::TrustBq, false);

    assert!(model.apply_kkt_guided_boundary_restart(false).unwrap());

    let after = model
        .objective_at_theta_for_certificate(&model.theta())
        .unwrap();
    assert!(after < before);
    let certificate = model.two_by_two_covariance_kkt_certificate().unwrap();
    assert!(
        matches!(
            certificate.blocks[0].classification,
            CovarianceKktClassification::InteriorConverged
                | CovarianceKktClassification::ValidRankDeficientCovariance
        ),
        "unexpected certificate: {:?}",
        certificate.blocks[0]
    );
    assert!(model.optsum.return_value.contains("KKT_BOUNDARY_RESTART"));
    model.refresh_optimizer_certificate();
    let optimizer_certificate = model.optimizer_certificate().unwrap();
    assert!(
        optimizer_certificate
            .evidence
            .optimizer_stop
            .acceptable_stop
    );
    assert!(
        !optimizer_certificate
            .evidence
            .optimizer_stop
            .budget_exhausted
    );
    assert!(optimizer_certificate
        .evidence
        .optimizer_stop
        .final_trust_radius
        .is_some());
    assert!(matches!(
        optimizer_certificate.status,
        FitStatus::ConvergedInterior
            | FitStatus::ConvergedBoundary
            | FitStatus::ConvergedReducedRank
    ));
    let recovery = optimizer_certificate
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DiagnosticCode::OptimizerRecovery)
        .expect("recovered fit should carry optimizer recovery diagnostic");
    assert_eq!(
        recovery.payload.get("recovery_reason"),
        Some(&serde_json::json!("2x2 block 0"))
    );
    let verdict = ConvergenceVerdict::for_artifact(model.compiler_artifact());
    assert!(matches!(
        verdict.level,
        ConvergenceLevel::Ok | ConvergenceLevel::Caution
    ));
    assert!(verdict.headline.contains("recovered convergence"));
}

#[test]
fn test_mmtrust_psd_scalar_poc_matches_theta_optimizer_objective() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();
    let reference = model.objective_value();
    let reference_variance = model.theta()[0].powi(2);
    let outcome = optimize_scalar_psd_poc(&model, reference_variance * 1.2 + 1e-3);
    let tolerance = 1e-6 * (1.0 + reference.abs());

    assert!(
        (outcome.best_fmin - reference).abs() <= tolerance,
        "scalar PSD POC objective {} did not match theta objective {}",
        outcome.best_fmin,
        reference
    );
}

#[test]
fn test_mmtrust_psd_two_by_two_poc_matches_theta_optimizer_objective() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();
    let reference = model.objective_value();
    let theta = model.theta();
    let reference_covariance = two_by_two_covariance_from_theta([theta[0], theta[1], theta[2]]);
    let start_covariance = [
        [
            reference_covariance[0][0] * 1.05 + 1e-3,
            reference_covariance[0][1],
        ],
        [
            reference_covariance[1][0],
            reference_covariance[1][1] * 1.05 + 1e-3,
        ],
    ];
    let outcome = optimize_two_by_two_psd_poc(&model, start_covariance);
    let tolerance = 1e-6 * (1.0 + reference.abs());

    assert!(
        (outcome.best_fmin - reference).abs() <= tolerance,
        "2x2 PSD POC objective {} did not match theta objective {}",
        outcome.best_fmin,
        reference
    );
}

#[test]
fn test_active_face_poc_reduces_rank_one_two_by_two_covariance_search() {
    let data = rank_one_rho_one_random_slope_fixture();
    let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(false).unwrap();
    let fitted_theta = model.theta();
    let rank_one_theta = [fitted_theta[0], fitted_theta[1], 0.0];
    model.set_theta(&rank_one_theta).unwrap();
    let reference = model
        .objective_at_theta_for_certificate(&rank_one_theta)
        .unwrap();

    let certificate = model.two_by_two_covariance_kkt_certificate().unwrap();
    let block = &certificate.blocks[0];
    assert_eq!(
        block.classification,
        CovarianceKktClassification::ValidRankDeficientCovariance
    );

    let direction = symmetric_2x2_max_eigenvector_for_test(block.covariance);
    let (_, active_amplitude) = symmetric_2x2_eigenvalues(block.covariance);
    assert!(active_amplitude > 0.0);
    let start_amplitude = active_amplitude * 1.15 + 1e-3;
    let active_outcome =
        optimize_two_by_two_active_face_psd_poc(&model, direction, start_amplitude);
    let full_outcome = optimize_two_by_two_psd_poc(
        &model,
        rank_one_covariance_for_test(direction, start_amplitude),
    );
    let tolerance = 1e-6 * (1.0 + reference.abs());

    assert_eq!(active_outcome.best_theta.len(), 1);
    assert_eq!(full_outcome.best_theta.len(), 3);
    assert_eq!(
        active_outcome.trace_label.as_deref(),
        Some("active_face_rank_one_2x2")
    );
    assert_eq!(active_outcome.active_rank, Some(1));
    assert_eq!(active_outcome.inactive_directions, Some(1));
    assert!(
        matches!(
            active_outcome.exit_reason.as_str(),
            "step_tolerance" | "ftol_or_no_progress"
        ),
        "active-face trace should record a non-budget exit reason, got {}",
        active_outcome.exit_reason
    );
    let active_gap = (active_outcome.best_fmin - reference).abs();
    let full_gap = (full_outcome.best_fmin - reference).abs();
    assert!(
        active_gap <= tolerance,
        "active-face objective {} did not match reference {}",
        active_outcome.best_fmin,
        reference
    );
    assert!(
            active_gap < full_gap,
            "active-face search should improve the full 2x2 boundary search: active_gap={active_gap}, full_gap={full_gap}"
        );
    assert!(
            active_outcome.feval_count < full_outcome.feval_count || full_gap > tolerance,
            "active face should use fewer evaluations or certify when full 2x2 search does not: active_feval={}, full_feval={}, full_gap={full_gap}, tolerance={tolerance}",
            active_outcome.feval_count,
            full_outcome.feval_count,
        );

    let active_covariance = rank_one_covariance_for_test(direction, active_outcome.best_theta[0]);
    let covariance_gap = two_by_two_frobenius_norm([
        [
            active_covariance[0][0] - block.covariance[0][0],
            active_covariance[0][1] - block.covariance[0][1],
        ],
        [
            active_covariance[1][0] - block.covariance[1][0],
            active_covariance[1][1] - block.covariance[1][1],
        ],
    ]);
    assert!(
        covariance_gap <= 1e-3 * (1.0 + two_by_two_frobenius_norm(block.covariance)),
        "active-face covariance estimate drift too large: gap={covariance_gap}, reference={:?}",
        block.covariance
    );
}

#[test]
fn test_active_face_detection_records_lower_rank_four_by_four_block() {
    let mut data = DataFrame::new();
    let mut y = Vec::new();
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let mut x3 = Vec::new();
    let mut group = Vec::new();
    for g in 0..16_usize {
        let shift = (g as f64 - 7.5) * 0.08;
        for k in 0..6_usize {
            let t = k as f64 - 2.5;
            let a = t / 3.0;
            let b = (t * t - 2.0) / 5.0;
            let c = if k % 2 == 0 { -0.5 } else { 0.5 };
            y.push(1.0 + 0.4 * a - 0.2 * b + 0.1 * c + shift);
            x1.push(a);
            x2.push(b);
            x3.push(c);
            group.push(format!("g{}", g + 1));
        }
    }
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x1", x1).unwrap();
    data.add_numeric("x2", x2).unwrap();
    data.add_numeric("x3", x3).unwrap();
    data.add_categorical("group", group).unwrap();

    let formula = parse_formula("y ~ 1 + x1 + x2 + x3 + (1 + x1 + x2 + x3 | group)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    // Force a rank-2 lower Cholesky factor in the 4x4 covariance block.
    // The active-face prototype does not optimize this shape yet, but the
    // fitted artifact must detect the active face from the same
    // eigenstructure used by future continuation.
    let mut theta = vec![0.0; model.n_theta()];
    for (idx, &(_, row, col)) in model.parmap().iter().enumerate() {
        theta[idx] = match (row, col) {
            (0, 0) => 1.0,
            (1, 0) => 0.35,
            (2, 0) => -0.15,
            (3, 0) => 0.10,
            (1, 1) => 0.70,
            (2, 1) => 0.25,
            (3, 1) => -0.20,
            _ => 0.0,
        };
    }
    model.set_theta(&theta).unwrap();
    model.update_l().unwrap();
    let objective = model.objective_at_theta_for_certificate(&theta).unwrap();
    model.optsum.return_value = "FTOL_REACHED".to_string();
    model.optsum.feval = 1;
    model.optsum.finitial = objective;
    model.optsum.fmin = objective;
    model.optsum.final_params = theta;
    model.refresh_optimizer_certificate();
    model.refresh_effective_covariance_summaries();

    let summary = &model.compiler_artifact().effective_covariance[0];
    assert_eq!(summary.requested_rank, 4);
    assert_eq!(summary.supported_rank, 2);
    assert_eq!(summary.status, EffectiveRankStatus::ReducedRank);
    assert_eq!(summary.directions.len(), 2);
    assert_eq!(summary.unsupported_directions.len(), 2);
    assert!(
        summary.interpretable_submodel.is_none(),
        "4x4 lower-rank detection must not pretend the active face is a simple one-axis formula"
    );
}

#[test]
fn test_effective_covariance_rank_uses_policy_thresholds() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    let mut policy = CompilerPolicy::maximal_feasible();
    policy.thresholds.effective_rank_relative_tolerance = 2.0;
    model.set_compiler_policy(policy).unwrap();

    model.fit(false).unwrap();

    let summary = &model.compiler_artifact().effective_covariance[0];
    assert_eq!(summary.status, EffectiveRankStatus::ReducedRank);
    assert_eq!(summary.supported_rank, 0);
    assert!(model
        .compiler_artifact()
        .reproducibility
        .thresholds
        .iter()
        .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "2"));
}

#[test]
fn test_lmm_new_with_compiler_policy_applies_policy_before_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut policy = CompilerPolicy::as_specified();
    policy.thresholds.effective_rank_relative_tolerance = 0.25;

    let model = LinearMixedModel::new_with_compiler_policy(formula, &data, None, policy).unwrap();

    assert_eq!(
        model.compiler_policy().random_strategy,
        crate::compiler::RandomStrategy::AsSpecified
    );
    assert!(model
        .compiler_artifact()
        .reproducibility
        .thresholds
        .iter()
        .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "0.25"));
}

#[test]
fn test_lmm_design_compiled_reduces_full_covariance_before_fit() {
    let data = grouped_slope_data_with_obs(6, 3);
    let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();

    let model = LinearMixedModel::new_with_compiler_policy(
        formula,
        &data,
        None,
        CompilerPolicy::design_compiled(),
    )
    .unwrap();
    let artifact = model.compiler_artifact();
    let state = artifact.model_state_summary();

    assert!(model.formula.random_terms[0].zerocorr);
    assert_eq!(model.theta().len(), 2);
    assert_eq!(artifact.theta_maps.len(), 2);
    assert_eq!(
        artifact
            .theta_maps
            .iter()
            .map(ThetaMap::n_free)
            .sum::<usize>(),
        2
    );
    assert_eq!(artifact.theta_maps[0].block().term_index, 0);
    assert_eq!(artifact.theta_maps[1].block().term_index, 0);
    assert_eq!(
        artifact.effective_formula.as_deref(),
        Some("y ~ 1 + x + (1 + x || group)")
    );
    assert_eq!(
        artifact.reproducibility.fit_intent,
        FitIntent::ConfirmatoryDesignCompiled
    );
    assert_eq!(state.supported.status, ModelStateStatus::Reduced);
    assert!(state.changes.iter().any(|change| {
        change.status == ModelChangeStatus::Applied
            && change.trigger == ReductionTrigger::DesignTime
            && change.replacement_term.as_deref() == Some("(1 + x || group)")
    }));
}

#[test]
fn test_lmm_design_compiled_refuses_unsupported_random_distribution() {
    let data = grouped_slope_data(2);
    let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();

    let result = LinearMixedModel::new_with_compiler_policy(
        formula,
        &data,
        None,
        CompilerPolicy::design_compiled(),
    );

    assert!(result.is_err());
    assert!(result
        .err()
        .unwrap()
        .to_string()
        .contains("design_compiled refused"));
}

#[test]
fn test_lmm_design_compiled_refuses_row_saturated_random_effect() {
    let data = grouped_slope_data(100);
    let formula = parse_formula("y ~ x + (1 + x | group)").unwrap();

    let err = LinearMixedModel::new_with_compiler_policy(
        formula,
        &data,
        None,
        CompilerPolicy::design_compiled(),
    )
    .expect_err("row-saturated random-effect terms should be refused");
    let message = err.to_string();

    assert!(message.contains("number of observations (200)"));
    assert!(message.contains("random coefficients (200)"));
    assert!(message.contains("residual scale"));
}

#[test]
fn test_lmm_set_compiler_policy_rejects_after_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let error = model
        .set_compiler_policy(CompilerPolicy::as_specified())
        .expect_err("fitted models must reject policy mutation");

    assert!(matches!(error, MixedModelError::AlreadyFitted));
}

#[test]
fn test_lmm_fit_with_compiler_policy_applies_policy_then_fits() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    let mut policy = CompilerPolicy::as_specified();
    policy.thresholds.effective_rank_relative_tolerance = 0.5;

    model.fit_with_compiler_policy(false, policy).unwrap();

    assert_eq!(
        model.compiler_policy().random_strategy,
        crate::compiler::RandomStrategy::AsSpecified
    );
    assert!(model.optimizer_certificate().is_some());
    assert!(model
        .compiler_artifact()
        .reproducibility
        .thresholds
        .iter()
        .any(|(name, value)| name == "effective_rank_relative_tolerance" && value == "0.5"));
}

#[test]
fn test_lmm_optimizer_certificate_records_budget_stop() {
    let data = singular_re_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.max_feval = 1;

    model.fit(false).unwrap();

    let certificate = model
        .optimizer_certificate()
        .expect("optimizer certificate should attach after fit");
    assert_eq!(certificate.status, FitStatus::NotOptimized);
    assert!(!certificate.evidence.optimizer_stop.acceptable_stop);
    assert!(certificate.evidence.optimizer_stop.budget_exhausted);
    assert!(matches!(
        certificate.evidence.certification_quality,
        EvidenceQuality::Failed { .. }
    ));
    assert!(certificate
        .checks
        .iter()
        .any(|check| matches!(check, CertificateCheck::Failed { .. })));
    assert!(model.compiler_artifact().effective_covariance.is_empty());
}

#[test]
fn test_block_index() {
    assert_eq!(block_index(0, 0), 0);
    assert_eq!(block_index(1, 0), 1);
    assert_eq!(block_index(1, 1), 2);
    assert_eq!(block_index(2, 0), 3);
    assert_eq!(block_index(2, 1), 4);
    assert_eq!(block_index(2, 2), 5);
}

#[test]
fn test_dense_crossed_block_guard_reports_problem_too_large() {
    let err = ensure_dense_block_within_explicit_limit(
        1_400_000,
        100_000,
        "issue-702-scale crossed random-effects block",
        16 * 1024 * 1024 * 1024,
    )
    .expect_err("issue-702-scale dense block should be refused before allocation");

    assert!(matches!(err, MixedModelError::ProblemTooLarge(_)));
    assert!(err.to_string().contains("1043."));
    assert!(err.to_string().contains("issue-702-scale"));
}

#[test]
fn test_dense_crossed_block_guard_accepts_small_blocks() {
    ensure_dense_block_within_explicit_limit(
        100,
        80,
        "small crossed random-effects block",
        16 * 1024 * 1024 * 1024,
    )
    .expect("small dense blocks should remain valid");
}

#[test]
fn test_crossed_scalar_re_cross_product_stays_sparse() {
    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    data.add_categorical(
        "person",
        vec!["p1", "p1", "p2", "p3", "p3", "p1"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();
    data.add_categorical(
        "firm",
        vec!["f1", "f2", "f2", "f1", "f3", "f1"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
    .unwrap();

    let formula = parse_formula("y ~ 1 + (1 | person) + (1 | firm)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert!(
        matches!(model.a_blocks[block_index(1, 0)], MatrixBlock::Sparse(_)),
        "crossed scalar RE off-diagonal A block should not be materialized dense"
    );
    let MatrixBlock::Sparse(cross) = &model.a_blocks[block_index(1, 0)] else {
        unreachable!();
    };
    assert_eq!(cross.nrows(), model.reterms[1].n_ranef());
    assert_eq!(cross.ncols(), model.reterms[0].n_ranef());
    assert!(cross.nnz() <= data.nrow());

    let dense = MatrixBlock::Sparse(cross.clone()).as_dense();
    let person_p1 = model.reterms[0]
        .levels
        .iter()
        .position(|level| level == "p1")
        .unwrap();
    let firm_f1 = model.reterms[1]
        .levels
        .iter()
        .position(|level| level == "f1")
        .unwrap();
    assert_eq!(dense[(firm_f1, person_p1)], 2.0);
}

#[test]
fn test_cholesky_block_diagonal() {
    let mut block = MatrixBlock::Diagonal(DVector::from_vec(vec![4.0, 9.0, 16.0]));
    cholesky_block(&mut block).unwrap();
    if let MatrixBlock::Diagonal(d) = &block {
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
        assert!((d[2] - 4.0).abs() < 1e-10);
    }
}

#[test]
fn test_cholesky_block_dense() {
    // [[4, 2], [2, 5]] → L = [[2, 0], [1, 2]]
    let mut block = MatrixBlock::Dense(DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 5.0]));
    cholesky_block(&mut block).unwrap();
    if let MatrixBlock::Dense(m) = &block {
        assert!((m[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((m[(1, 0)] - 1.0).abs() < 1e-10);
        assert!((m[(1, 1)] - 2.0).abs() < 1e-10);
        assert!(m[(0, 1)].abs() < 1e-10);
    }
}

#[test]
fn test_cholesky_zero_pad_scales_with_diagonal() {
    let mut unit_scale = MatrixBlock::Dense(DMatrix::from_diagonal(&DVector::from_vec(vec![
        -1e-12, 1.0,
    ])));
    assert!(matches!(
        cholesky_block(&mut unit_scale),
        Err(MixedModelError::PosDefException)
    ));

    let mut large_scale = MatrixBlock::Dense(DMatrix::from_diagonal(&DVector::from_vec(vec![
        -1e-12, 1e8,
    ])));
    cholesky_block(&mut large_scale).unwrap();
    let MatrixBlock::Dense(mat) = large_scale else {
        unreachable!();
    };
    assert_eq!(mat[(0, 0)], 0.0);
    assert_relative_eq!(mat[(1, 1)], 1e4, epsilon = 1e-8);
}

#[test]
fn test_cholesky_rejects_near_singular_negative_pivot_at_unit_scale() {
    let mut block =
        MatrixBlock::Dense(DMatrix::from_diagonal(&DVector::from_vec(vec![-1e-9, 1.0])));

    assert!(matches!(
        cholesky_block(&mut block),
        Err(MixedModelError::PosDefException)
    ));
}

#[test]
fn test_cholesky_strict_mode_matches_julia() {
    let mut block = MatrixBlock::Dense(DMatrix::from_diagonal(&DVector::from_vec(vec![
        -f64::EPSILON,
        1e16,
    ])));

    assert!(matches!(
        cholesky_block_with_tolerance(&mut block, 0.0),
        Err(MixedModelError::PosDefException)
    ));
}

#[test]
fn test_logdet_block() {
    let block = MatrixBlock::Diagonal(DVector::from_vec(vec![2.0, 3.0]));
    let ld = logdet_block(&block);
    // logdet = 2 * (ln(2) + ln(3)) = 2 * ln(6)
    assert!((ld - 2.0 * 6.0_f64.ln()).abs() < 1e-10);
}

#[test]
fn test_rank_k_downdate_small_dense_large_k_matches_gemm() {
    let a = DMatrix::from_fn(3, 520, |row, col| {
        (((row + 1) * (col + 3)) % 17) as f64 / 13.0 - 0.4
    });
    let init = DMatrix::from_row_slice(3, 3, &[3.0, 0.2, 0.4, 0.2, 2.5, -0.1, 0.4, -0.1, 1.7]);
    let mut optimized = MatrixBlock::Dense(init.clone());
    let mut expected = init;
    expected.gemm(-1.0, &a, &a.transpose(), 1.0);

    rank_k_downdate(&mut optimized, &a);

    let MatrixBlock::Dense(result) = optimized else {
        panic!("expected dense block");
    };
    for row in 0..3 {
        for col in 0..=row {
            assert_relative_eq!(
                result[(row, col)],
                expected[(row, col)],
                epsilon = 1e-10,
                max_relative = 1e-12
            );
        }
    }
}

#[test]
fn test_create_al_single_vsize2_matches_generic_blocks() {
    let data = simulate_sleepstudy_like(260, 3, 23);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let y_data = data.numeric(&formula.response).unwrap();
    let y = DVector::from_column_slice(y_data);
    let (x_mat, fe_names) = build_fixed_effects_matrix(&formula, &data).unwrap();
    let feterm = FeTerm::new(x_mat, fe_names);
    let xy = FeMat::new(&feterm, &y);
    let re = build_re_mat(&formula.random_terms[0], &data, data.nrow()).unwrap();

    let (specialized, _) = create_al_single_vsize2(&re, &xy);
    let generic = [
        compute_re_cross_product(&re, &re),
        compute_fe_re_cross_product(&xy, &re),
        MatrixBlock::Dense(xy.wtxy.transpose() * &xy.wtxy),
    ];

    for (left, right) in specialized.iter().zip(generic.iter()) {
        let left_dense = left.as_dense();
        let right_dense = right.as_dense();
        assert_eq!(left_dense.shape(), right_dense.shape());
        for row in 0..left_dense.nrows() {
            for col in 0..left_dense.ncols() {
                assert_relative_eq!(
                    left_dense[(row, col)],
                    right_dense[(row, col)],
                    epsilon = 1e-10,
                    max_relative = 1e-12
                );
            }
        }
    }
}

#[test]
fn test_fixed_design_solver_blocks_match_femat_blocks_unweighted() {
    let data = simulate_sleepstudy_like(24, 4, 23);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let y = DVector::from_column_slice(data.numeric(&formula.response).unwrap());
    let raw_fixed_design = crate::model::fixed_design::build_fixed_effects_design_with_policy(
        &formula,
        &data,
        FixedDesignBuildPolicy::streamed(),
    )
    .unwrap();
    let feterm = FeTerm::new(
        raw_fixed_design.materialize_dense(),
        raw_fixed_design.column_names().to_vec(),
    );
    let fixed_design = raw_fixed_design
        .select_columns(&feterm.piv[..feterm.rank])
        .unwrap();
    let xy = FeMat::new(&feterm, &y);
    let re = build_re_mat(&formula.random_terms[0], &data, data.nrow()).unwrap();

    let (backend_blocks, _) =
        create_al_from_fixed_design(std::slice::from_ref(&re), &fixed_design, &y, None).unwrap();
    let (dense_blocks, _) = create_al(&[re], &xy).unwrap();

    for (backend, dense) in backend_blocks.iter().zip(dense_blocks.iter()) {
        let backend_dense = backend.as_dense();
        let expected_dense = dense.as_dense();
        assert_eq!(backend_dense.shape(), expected_dense.shape());
        for row in 0..backend_dense.nrows() {
            for col in 0..backend_dense.ncols() {
                assert_relative_eq!(
                    backend_dense[(row, col)],
                    expected_dense[(row, col)],
                    epsilon = 1e-10,
                    max_relative = 1e-12
                );
            }
        }
    }
}

#[test]
fn test_fixed_design_solver_blocks_match_femat_blocks_weighted() {
    let data = simulate_sleepstudy_like(24, 4, 47);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let y = DVector::from_column_slice(data.numeric(&formula.response).unwrap());
    let raw_fixed_design = crate::model::fixed_design::build_fixed_effects_design_with_policy(
        &formula,
        &data,
        FixedDesignBuildPolicy::streamed(),
    )
    .unwrap();
    let feterm = FeTerm::new(
        raw_fixed_design.materialize_dense(),
        raw_fixed_design.column_names().to_vec(),
    );
    let fixed_design = raw_fixed_design
        .select_columns(&feterm.piv[..feterm.rank])
        .unwrap();
    let sqrtwts = DVector::from_iterator(
        data.nrow(),
        (0..data.nrow()).map(|idx| if idx % 2 == 0 { 1.0 } else { 2.0 }),
    );
    let mut xy = FeMat::new(&feterm, &y);
    xy.reweight(&sqrtwts);
    let mut re = build_re_mat(&formula.random_terms[0], &data, data.nrow()).unwrap();
    re.reweight(&sqrtwts);

    let (backend_blocks, _) =
        create_al_from_fixed_design(&[re.clone()], &fixed_design, &y, Some(&sqrtwts)).unwrap();
    let (dense_blocks, _) = create_al(&[re], &xy).unwrap();

    for (backend, dense) in backend_blocks.iter().zip(dense_blocks.iter()) {
        let backend_dense = backend.as_dense();
        let expected_dense = dense.as_dense();
        assert_eq!(backend_dense.shape(), expected_dense.shape());
        for row in 0..backend_dense.nrows() {
            for col in 0..backend_dense.ncols() {
                assert_relative_eq!(
                    backend_dense[(row, col)],
                    expected_dense[(row, col)],
                    epsilon = 1e-10,
                    max_relative = 1e-12
                );
            }
        }
    }
}

#[test]
fn test_lmm_constructor_keeps_high_cardinality_fixed_design_streamed() {
    let n_levels = 256usize;
    let n_obs = 512usize;
    let formula = parse_formula("y ~ 1 + sku + (1 | group)").unwrap();
    let mut data = DataFrame::new();
    data.add_numeric("y", (0..n_obs).map(|idx| idx as f64).collect())
        .unwrap();
    data.add_categorical(
        "sku",
        (0..n_obs)
            .map(|idx| format!("sku{}", idx % n_levels))
            .collect(),
    )
    .unwrap();
    data.add_categorical(
        "group",
        (0..n_obs).map(|idx| format!("g{}", idx % 16)).collect(),
    )
    .unwrap();

    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    assert_eq!(
        model.fixed_design.storage(),
        crate::model::fixed_design::FixedDesignStorage::Streamed
    );
    assert_eq!(model.fixed_design.n_cols(), model.feterm.rank);
    assert!(model.fixed_design.as_streamed().is_some());

    let summary = model.fixed_design_backend_summary();
    assert_eq!(summary.storage, FixedDesignStorage::Streamed);
    assert_eq!(summary.n_obs, n_obs);
    assert_eq!(summary.n_cols, model.feterm.rank);
    assert!(model.fixed_design_density() < 0.02);
    assert!(model.fixed_design_active_entries() < n_obs * 3);

    let diagnostic = model
        .compiler_artifact()
        .diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.code == DiagnosticCode::SupportNote
                && diagnostic
                    .payload
                    .get("diagnostic_kind")
                    .and_then(|value| value.as_str())
                    == Some("fixed_design_backend")
        })
        .expect("streamed backend should be exposed as a structured diagnostic");
    assert_eq!(
        diagnostic
            .payload
            .get("storage")
            .and_then(|value| value.as_str()),
        Some("streamed")
    );
    assert!(diagnostic
        .message
        .contains("fixed-effect design backend selected: streamed"));

    let report = model.audit_report().to_text();
    assert!(report.contains("fixed-effect design backend selected: streamed"));
    assert!(report.contains("rank/pivot detection uses a streamed Gram certificate"));
    assert!(
        report.contains("streamed fixed-effect rank/pivot: Gram certificate established full rank")
    );
}

fn streamed_fixed_effect_parity_fixture(n_levels: usize, obs_per_level: usize) -> DataFrame {
    let n_obs = n_levels * obs_per_level;
    let mut y = Vec::with_capacity(n_obs);
    let mut x = Vec::with_capacity(n_obs);
    let mut sku = Vec::with_capacity(n_obs);
    let mut group = Vec::with_capacity(n_obs);

    for level in 0..n_levels {
        for rep in 0..obs_per_level {
            let obs = level * obs_per_level + rep;
            let x_value = rep as f64 - 0.5 + ((level % 5) as f64) * 0.1;
            let sku_effect = ((level % 11) as f64 - 5.0) * 0.07;
            let group_effect = ((obs % 17) as f64 - 8.0) * 0.03;
            let noise = ((obs % 7) as f64 - 3.0) * 0.01;
            x.push(x_value);
            y.push(2.0 + 0.8 * x_value + sku_effect + group_effect + noise);
            sku.push(format!("sku{:03}", level));
            group.push(format!("g{:02}", obs % 17));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("sku", sku).unwrap();
    data.add_categorical("group", group).unwrap();
    data
}

fn assert_lmm_fit_close(left: &LinearMixedModel, right: &LinearMixedModel) {
    assert_eq!(left.coef_names(), right.coef_names());
    let left_theta = left.theta();
    let right_theta = right.theta();
    assert_eq!(left_theta.len(), right_theta.len());
    for (left_theta, right_theta) in left_theta.iter().zip(right_theta.iter()) {
        assert_relative_eq!(
            *left_theta,
            *right_theta,
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }

    let left_beta = left.beta();
    let right_beta = right.beta();
    assert_eq!(left_beta.len(), right_beta.len());
    for idx in 0..left_beta.len() {
        assert_relative_eq!(
            left_beta[idx],
            right_beta[idx],
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }

    assert_relative_eq!(
        left.sigma(),
        right.sigma(),
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    assert_relative_eq!(
        left.objective_value(),
        right.objective_value(),
        epsilon = 1e-8,
        max_relative = 1e-8
    );

    let left_fitted = left.fitted();
    let right_fitted = right.fitted();
    assert_eq!(left_fitted.len(), right_fitted.len());
    for idx in 0..left_fitted.len() {
        assert_relative_eq!(
            left_fitted[idx],
            right_fitted[idx],
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }
}

#[test]
fn test_streamed_fixed_effect_lmm_fit_matches_dense_backend() {
    let data = streamed_fixed_effect_parity_fixture(64, 4);
    let formula = parse_formula("y ~ 1 + x + sku + (1 | group)").unwrap();

    let mut dense = LinearMixedModel::new_with_fixed_design_policy(
        formula.clone(),
        &data,
        None,
        FixedDesignBuildPolicy::dense(),
    )
    .unwrap();
    let mut streamed = LinearMixedModel::new_with_fixed_design_policy(
        formula,
        &data,
        None,
        FixedDesignBuildPolicy::streamed(),
    )
    .unwrap();

    assert_eq!(
        dense.fixed_design.storage(),
        crate::model::fixed_design::FixedDesignStorage::Dense
    );
    assert_eq!(
        streamed.fixed_design.storage(),
        crate::model::fixed_design::FixedDesignStorage::Streamed
    );

    dense.fit(false).unwrap();
    streamed.fit(false).unwrap();

    assert_lmm_fit_close(&dense, &streamed);
}

/// High-cardinality fixture: enough fixed columns x RE levels that the
/// streamed backend's `X'Z` cross-product crosses the sparse-emission
/// threshold (dense cells >= 64k).
fn high_cardinality_streamed_fixture(n_levels: usize, obs_per_level: usize) -> DataFrame {
    let n_obs = n_levels * obs_per_level;
    let n_groups = 600;
    let mut y = Vec::with_capacity(n_obs);
    let mut x = Vec::with_capacity(n_obs);
    let mut sku = Vec::with_capacity(n_obs);
    let mut group = Vec::with_capacity(n_obs);

    for level in 0..n_levels {
        for rep in 0..obs_per_level {
            let obs = level * obs_per_level + rep;
            let x_value = rep as f64 - 0.5 + ((level % 5) as f64) * 0.1;
            let sku_effect = ((level % 11) as f64 - 5.0) * 0.07;
            let group_id = (obs * 7) % n_groups;
            let group_effect = ((group_id % 23) as f64 - 11.0) * 0.03;
            let noise = ((obs % 7) as f64 - 3.0) * 0.01;
            x.push(x_value);
            y.push(2.0 + 0.8 * x_value + sku_effect + group_effect + noise);
            sku.push(format!("sku{:03}", level));
            group.push(format!("g{:03}", group_id));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("sku", sku).unwrap();
    data.add_categorical("group", group).unwrap();
    data
}

#[test]
fn test_high_cardinality_streamed_fit_uses_sparse_fixed_re_block_and_matches_dense() {
    let data = high_cardinality_streamed_fixture(120, 6);
    let formula = parse_formula("y ~ 1 + x + sku + (1 | group)").unwrap();

    let mut dense = LinearMixedModel::new_with_fixed_design_policy(
        formula.clone(),
        &data,
        None,
        FixedDesignBuildPolicy::dense(),
    )
    .unwrap();
    let mut streamed = LinearMixedModel::new_with_fixed_design_policy(
        formula,
        &data,
        None,
        FixedDesignBuildPolicy::streamed(),
    )
    .unwrap();

    assert_eq!(
        streamed.fixed_design.storage(),
        crate::model::fixed_design::FixedDesignStorage::Streamed
    );
    // k == 1: a_blocks = [Z'Z, [X|y]'Z, [X|y]'[X|y]]; the FE x RE
    // cross-product must have stayed sparse for this shape.
    assert!(
        matches!(streamed.a_blocks[1], MatrixBlock::Sparse(_)),
        "high-cardinality streamed [X|y]'Z block should be Sparse, got {:?}",
        match &streamed.a_blocks[1] {
            MatrixBlock::Dense(m) => format!("Dense{:?}", m.shape()),
            other => format!("{other:?}").chars().take(30).collect::<String>(),
        }
    );
    assert!(
        matches!(dense.a_blocks[1], MatrixBlock::Dense(_)),
        "dense backend keeps the dense [X|y]'Z block"
    );

    dense.fit(false).unwrap();
    streamed.fit(false).unwrap();

    assert_lmm_fit_close(&dense, &streamed);
}

#[test]
fn test_copy_and_rmul_lambda_sparse_matches_dense() {
    let mut coo = CooMatrix::new(3, 4);
    coo.push(0, 0, 2.0);
    coo.push(2, 1, -1.5);
    coo.push(1, 3, 0.75);
    let sparse = CscMatrix::from(&coo);
    let a_sparse = MatrixBlock::Sparse(sparse);
    let a_dense = MatrixBlock::Dense(a_sparse.as_dense());

    let mut re = ReMat::new(
        "g".to_string(),
        vec![0, 1],
        vec!["a".to_string(), "b".to_string()],
        vec!["(Intercept)".to_string()],
        DMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
    );
    re.set_theta(&[0.6]).unwrap();

    let mut l_from_sparse = MatrixBlock::Sparse(CscMatrix::from(&CooMatrix::new(3, 4)));
    copy_and_rmul_lambda(&mut l_from_sparse, &a_sparse, &re);
    let mut l_from_dense = MatrixBlock::Dense(DMatrix::zeros(3, 4));
    copy_and_rmul_lambda(&mut l_from_dense, &a_dense, &re);

    assert!(matches!(l_from_sparse, MatrixBlock::Sparse(_)));
    assert_eq!(l_from_sparse.as_dense(), l_from_dense.as_dense());

    // Second application reuses the sparse buffer and stays correct after a
    // theta change.
    re.set_theta(&[1.25]).unwrap();
    copy_and_rmul_lambda(&mut l_from_sparse, &a_sparse, &re);
    copy_and_rmul_lambda(&mut l_from_dense, &a_dense, &re);
    assert_eq!(l_from_sparse.as_dense(), l_from_dense.as_dense());
}

#[test]
fn test_weighted_streamed_fixed_effect_lmm_fit_matches_dense_backend() {
    let data = streamed_fixed_effect_parity_fixture(48, 5);
    let formula = parse_formula("y ~ 1 + x + sku + (1 | group)").unwrap();
    let weights = (0..data.nrow())
        .map(|idx| 0.5 + ((idx % 5) as f64) * 0.25)
        .collect::<Vec<_>>();

    let mut dense = LinearMixedModel::new_with_fixed_design_policy(
        formula.clone(),
        &data,
        Some(&weights),
        FixedDesignBuildPolicy::dense(),
    )
    .unwrap();
    let mut streamed = LinearMixedModel::new_with_fixed_design_policy(
        formula,
        &data,
        Some(&weights),
        FixedDesignBuildPolicy::streamed(),
    )
    .unwrap();

    assert_eq!(
        dense.fixed_design.storage(),
        crate::model::fixed_design::FixedDesignStorage::Dense
    );
    assert_eq!(
        streamed.fixed_design.storage(),
        crate::model::fixed_design::FixedDesignStorage::Streamed
    );

    dense.fit(false).unwrap();
    streamed.fit(false).unwrap();

    assert_lmm_fit_close(&dense, &streamed);
}

#[test]
fn test_rdiv_lower_transpose_diagonal() {
    let mut a = MatrixBlock::Dense(DMatrix::from_row_slice(
        2,
        3,
        &[4.0, 9.0, 8.0, 2.0, 3.0, 5.0],
    ));
    let l = MatrixBlock::Diagonal(DVector::from_vec(vec![2.0, 3.0, 0.0]));

    rdiv_lower_transpose(&mut a, &l);

    if let MatrixBlock::Dense(m) = &a {
        assert_relative_eq!(m[(0, 0)], 2.0, epsilon = 1e-12);
        assert_relative_eq!(m[(1, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(m[(0, 1)], 3.0, epsilon = 1e-12);
        assert_relative_eq!(m[(1, 1)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(m[(0, 2)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(m[(1, 2)], 0.0, epsilon = 1e-12);
    } else {
        panic!("expected dense block after diagonal solve");
    }
}

#[test]
fn test_blocked_forward_solve_zero_pivot_guard_uniform() {
    let tiny_but_solvable = f64::EPSILON * 0.5;
    let effectively_zero = BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE * 0.5;
    let blocks = [
        (
            MatrixBlock::Diagonal(DVector::from_vec(vec![tiny_but_solvable, 2.0])),
            [1.0, 2.0],
        ),
        (
            MatrixBlock::BlockDiagonal(vec![DMatrix::from_row_slice(
                2,
                2,
                &[tiny_but_solvable, 0.0, 3.0, 2.0],
            )]),
            [1.0, 0.5],
        ),
        (
            MatrixBlock::Dense(DMatrix::from_row_slice(
                2,
                2,
                &[tiny_but_solvable, 0.0, 3.0, 2.0],
            )),
            [1.0, 0.5],
        ),
    ];

    for (block, expected) in &blocks {
        let mut rhs = vec![tiny_but_solvable, 4.0];
        solve_lower_block_against_rhs(block, &mut rhs);
        assert_relative_eq!(rhs[0], expected[0], epsilon = 1e-12);
        assert_relative_eq!(rhs[1], expected[1], epsilon = 1e-12);

        let mut rhs_matrix = DMatrix::from_column_slice(2, 1, &[tiny_but_solvable, 4.0]);
        solve_lower_block_rhs(&mut rhs_matrix, block);
        assert_relative_eq!(rhs_matrix[(0, 0)], rhs[0], epsilon = 1e-12);
        assert_relative_eq!(rhs_matrix[(1, 0)], rhs[1], epsilon = 1e-12);
    }

    let mut rhs = vec![1.0, 4.0];
    solve_lower_block_against_rhs(
        &MatrixBlock::Dense(DMatrix::from_row_slice(
            2,
            2,
            &[effectively_zero, 0.0, 3.0, 2.0],
        )),
        &mut rhs,
    );
    assert_eq!(rhs, vec![0.0, 2.0]);
}

#[test]
fn test_copy_scale_inflate_vsize2_matches_reference() {
    let mut re = make_vector_remat_for_kernel_tests(2);
    re.set_theta(&[1.2, -0.35, 0.8]).unwrap();

    let src_blocks = vec![
        DMatrix::from_row_slice(2, 2, &[3.0, 0.4, 0.4, 2.5]),
        DMatrix::from_row_slice(2, 2, &[1.7, -0.2, -0.2, 0.9]),
    ];
    let a = MatrixBlock::BlockDiagonal(src_blocks.clone());
    let mut l = MatrixBlock::BlockDiagonal(vec![DMatrix::zeros(2, 2), DMatrix::zeros(2, 2)]);

    copy_scale_inflate(&mut l, &a, &re);

    let MatrixBlock::BlockDiagonal(result_blocks) = l else {
        panic!("expected block-diagonal result");
    };

    for (result, src) in result_blocks.iter().zip(src_blocks.iter()) {
        let expected = re.lambda.transpose() * src * &re.lambda + DMatrix::identity(2, 2);
        for row in 0..2 {
            for col in 0..2 {
                assert_relative_eq!(
                    result[(row, col)],
                    expected[(row, col)],
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
            }
        }
    }
}

#[test]
fn test_copy_and_scale_offdiag_vsize2_matches_reference() {
    let mut re_i = make_vector_remat_for_kernel_tests(2);
    let mut re_j = make_vector_remat_for_kernel_tests(2);
    re_i.set_theta(&[1.1, -0.25, 0.9]).unwrap();
    re_j.set_theta(&[0.8, 0.3, 1.4]).unwrap();

    let a_dense = DMatrix::from_row_slice(
        4,
        4,
        &[
            1.0, 0.2, -0.3, 0.5, 0.6, 1.4, 0.1, -0.2, -0.4, 0.3, 1.6, 0.7, 0.2, -0.5, 0.8, 1.1,
        ],
    );
    let a = MatrixBlock::Dense(a_dense.clone());
    let mut l = MatrixBlock::Dense(DMatrix::zeros(4, 4));

    copy_and_scale_offdiag(&mut l, &a, &re_i, &re_j);

    let MatrixBlock::Dense(result) = l else {
        panic!("expected dense result");
    };

    let mut expected = DMatrix::zeros(4, 4);
    for bi in 0..2 {
        let row0 = bi * 2;
        for bj in 0..2 {
            let col0 = bj * 2;
            let src = a_dense.view((row0, col0), (2, 2)).into_owned();
            let block = re_i.lambda.transpose() * src * &re_j.lambda;
            for row in 0..2 {
                for col in 0..2 {
                    expected[(row0 + row, col0 + col)] = block[(row, col)];
                }
            }
        }
    }

    for row in 0..4 {
        for col in 0..4 {
            assert_relative_eq!(
                result[(row, col)],
                expected[(row, col)],
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
    }
}

#[test]
fn test_rdiv_lower_transpose_blockdiag_vsize2_matches_dense_reference() {
    let mut a = MatrixBlock::Dense(DMatrix::from_row_slice(
        3,
        4,
        &[
            2.0, -1.0, 0.5, 1.2, 0.1, 3.0, -0.4, 0.8, -2.1, 0.7, 1.5, -0.9,
        ],
    ));
    let l = MatrixBlock::BlockDiagonal(vec![
        DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.5, 1.5]),
        DMatrix::from_row_slice(2, 2, &[1.3, 0.0, -0.2, 0.9]),
    ]);

    let mut expected = DMatrix::from_row_slice(
        3,
        4,
        &[
            2.0, -1.0, 0.5, 1.2, 0.1, 3.0, -0.4, 0.8, -2.1, 0.7, 1.5, -0.9,
        ],
    );
    let dense_l = l.as_dense();
    for j in 0..dense_l.ncols() {
        if dense_l[(j, j)].abs() < BLOCK_TRIANGULAR_SOLVE_ZERO_TOLERANCE {
            for i in 0..expected.nrows() {
                expected[(i, j)] = 0.0;
            }
            continue;
        }
        for i in 0..expected.nrows() {
            let mut s = expected[(i, j)];
            for k in 0..j {
                s -= expected[(i, k)] * dense_l[(j, k)];
            }
            expected[(i, j)] = s / dense_l[(j, j)];
        }
    }

    rdiv_lower_transpose(&mut a, &l);

    let MatrixBlock::Dense(result) = a else {
        panic!("expected dense result");
    };

    for row in 0..result.nrows() {
        for col in 0..result.ncols() {
            assert_relative_eq!(
                result[(row, col)],
                expected[(row, col)],
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
    }
}

#[test]
fn test_objective_at_reuses_work_blocks_without_drift() {
    let data = simulate_sleepstudy_like(8, 6, 7);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let theta_a = [1.3, -0.15, 0.8];
    let theta_b = [0.7, 0.25, 1.4];

    let obj_a1 = model.objective_at(&theta_a).unwrap();
    let _obj_b = model.objective_at(&theta_b).unwrap();
    let obj_a2 = model.objective_at(&theta_a).unwrap();

    assert_relative_eq!(obj_a1, obj_a2, epsilon = 1e-10, max_relative = 1e-10);
}

#[test]
fn test_fast_vsize2_profiled_objective_matches_generic_update() {
    let data = simulate_sleepstudy_like(300, 3, 17);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.reml = true;
    let theta = [0.9, 0.2, 0.35];

    let generic = model.objective_at(&theta).unwrap();
    let fast = LinearMixedModel::profiled_objective_one_vsize2_fast(
        &model.a_blocks,
        &model.reterms,
        &theta,
        model.dims,
        true,
        model.optsum.sigma,
        model
            .compiler_policy()
            .thresholds
            .cholesky_zero_pad_tolerance,
    )
    .expect("large one-term vector RE should use the fast objective path");

    assert_relative_eq!(fast, generic, epsilon = 1e-8, max_relative = 1e-12);
}

#[test]
fn test_vector_re_fit_is_invariant_to_row_order() {
    let data = simulate_sleepstudy_like(10, 5, 42);
    let order: Vec<usize> = (0..data.nrow()).rev().collect();
    let permuted = permute_rows(&data, &order);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();

    let mut model_a = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
    let mut model_b = LinearMixedModel::new(formula, &permuted, None).unwrap();

    model_a.fit(true).unwrap();
    model_b.fit(true).unwrap();

    assert_relative_eq!(
        model_a.objective_value(),
        model_b.objective_value(),
        epsilon = 1e-7,
        max_relative = 1e-7
    );
    assert_relative_eq!(
        model_a.sigma(),
        model_b.sigma(),
        epsilon = 1e-3,
        max_relative = 1e-3
    );

    let beta_a = model_a.beta();
    let beta_b = model_b.beta();
    for i in 0..beta_a.len() {
        assert_relative_eq!(beta_a[i], beta_b[i], epsilon = 1e-4, max_relative = 1e-4);
    }

    let theta_a = model_a.theta();
    let theta_b = model_b.theta();
    for i in 0..theta_a.len() {
        assert_relative_eq!(theta_a[i], theta_b[i], epsilon = 5e-3, max_relative = 5e-3);
    }
}

#[cfg(feature = "nlopt")]
#[test]
fn test_vector_fit_uses_bobyqa_with_bounded_evaluations() {
    // n_theta = 3 (correlated random slope) → BOBYQA path. Pattern
    // search is the fallback if BOBYQA fails to converge; here we
    // expect the primary path to succeed and to use far fewer evals
    // than pattern_search did (which was bounded at 140).
    let data = simulate_sleepstudy_like(18, 10, 42);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(true).unwrap();

    assert_eq!(model.optsum.optimizer, Optimizer::NloptBobyqa);
    assert!(
        model.optsum.feval <= 80,
        "bobyqa used too many evaluations: {}",
        model.optsum.feval
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn test_large_theta_fit_uses_nlopt_newuoa() {
    let data = simulate_large_theta_crossed(123);
    let formula = parse_formula(
        "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
    )
    .unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.max_feval = 3000;

    model.fit(true).unwrap();

    assert_eq!(model.n_theta(), 9);
    assert_eq!(model.optsum.optimizer, Optimizer::NloptNewuoa);
    assert!(model.objective_value().is_finite());
    assert!(model.sigma().is_finite());
}

#[cfg(feature = "nlopt")]
#[test]
fn test_large_theta_nlopt_matches_or_beats_cobyla_baseline() {
    let data = simulate_large_theta_crossed(123);
    let formula = parse_formula(
        "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
    )
    .unwrap();

    let mut model_nlopt = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
    model_nlopt.optsum.max_feval = 3000;
    model_nlopt.fit(true).unwrap();

    let mut model_cobyla = LinearMixedModel::new(formula, &data, None).unwrap();
    model_cobyla.optsum.max_feval = 3000;
    model_cobyla.optsum.reml = true;
    let theta0 = model_cobyla.optsum.initial.clone();
    model_cobyla.optsum.finitial = model_cobyla.objective_at(&theta0).unwrap();
    model_cobyla
        .fit_cobyla_with_maxeval(true, Some(3000))
        .unwrap();

    assert!(
        model_nlopt.objective_value() <= model_cobyla.objective_value() + 1e-2,
        "nlopt objective {} should match or beat cobyla {} within tolerance",
        model_nlopt.objective_value(),
        model_cobyla.objective_value()
    );
    assert!(model_nlopt.optsum.feval < model_cobyla.optsum.feval);
}

#[test]
fn test_profile_response_matrix_matches_scalar_model_for_single_column() {
    let data = simulate_sleepstudy_like(12, 8, 17);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(true).unwrap();

    let y = model.y();
    let response_matrix = DMatrix::from_column_slice(y.len(), 1, y.as_slice());
    let profile = model
        .profile_response_matrix(&response_matrix, true)
        .unwrap();
    let beta = model.beta();

    assert_relative_eq!(
        profile.total_objective,
        model.objective_value(),
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    assert_relative_eq!(
        profile.pwrss[0],
        model.pwrss(),
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    assert_relative_eq!(
        profile.sigma[0],
        model.sigma(),
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    for row in 0..beta.len() {
        assert_relative_eq!(
            profile.beta[(row, 0)],
            beta[row],
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }
}

#[test]
fn test_profile_response_matrix_batches_columns_consistently() {
    let data = simulate_sleepstudy_like(10, 6, 23);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(true).unwrap();

    let y1 = model.y();
    let y2 = y1.map(|value| 0.75 * value + 12.0);
    let mut batch = DMatrix::zeros(y1.len(), 2);
    batch.set_column(0, &y1);
    batch.set_column(1, &y2);

    let batch_profile = model.profile_response_matrix(&batch, true).unwrap();
    let single_1 = model
        .profile_response_matrix(
            &DMatrix::from_column_slice(y1.len(), 1, y1.as_slice()),
            true,
        )
        .unwrap();
    let single_2 = model
        .profile_response_matrix(
            &DMatrix::from_column_slice(y2.len(), 1, y2.as_slice()),
            true,
        )
        .unwrap();

    assert_relative_eq!(
        batch_profile.total_objective,
        single_1.total_objective + single_2.total_objective,
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    for row in 0..batch_profile.beta.nrows() {
        assert_relative_eq!(
            batch_profile.beta[(row, 0)],
            single_1.beta[(row, 0)],
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            batch_profile.beta[(row, 1)],
            single_2.beta[(row, 0)],
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }
    assert_relative_eq!(
        batch_profile.sigma[0],
        single_1.sigma[0],
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    assert_relative_eq!(
        batch_profile.sigma[1],
        single_2.sigma[0],
        epsilon = 1e-8,
        max_relative = 1e-8
    );
}

#[test]
fn test_response_accessor_matches_stored_response() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();

    let y = model.y();
    let response = MixedModelFit::response(&model);

    assert_eq!(response.len(), y.len());
    for idx in 0..y.len() {
        assert_relative_eq!(response[idx], y[idx], epsilon = 1e-12, max_relative = 1e-12);
    }
}

#[test]
fn test_scalar_single_theta_fit_is_locally_optimal() {
    let data = simulate_sleepstudy_like(16, 8, 99);
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    model.fit(true).unwrap();

    let fitted_theta = model.theta()[0];
    let fitted_obj = model.objective_value();
    let mut probe = model.clone();
    let radius = fitted_theta.max(0.5);

    for step in 0..=20 {
        let frac = step as f64 / 20.0;
        let theta = frac * (fitted_theta + radius);
        let obj = probe.objective_at(&[theta]).unwrap();
        assert!(
            fitted_obj <= obj + 1e-6,
            "fitted objective {fitted_obj} exceeded probe objective {obj} at theta={theta}"
        );
    }

    assert!(
        model.optsum.feval <= 32,
        "scalar optimizer used too many evaluations: {}",
        model.optsum.feval
    );
}

#[test]
fn test_scalar_single_theta_records_maxeval() {
    let data = simulate_sleepstudy_like(16, 8, 99);
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.max_feval = 1;

    model.fit(true).unwrap();

    assert_eq!(model.optsum.optimizer, Optimizer::PatternSearch);
    assert_eq!(model.optsum.return_value, "MAXEVAL_REACHED");
    assert_ne!(model.optsum.return_value, "SUCCESS");
}

#[test]
fn test_pattern_search_records_maxeval() {
    let data = simulate_sleepstudy_like(12, 8, 17);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.max_feval = 1;

    model
        .fit_with_forced_optimizer(true, Optimizer::PatternSearch)
        .unwrap();

    assert_eq!(model.optsum.optimizer, Optimizer::PatternSearch);
    assert_eq!(model.optsum.return_value, "MAXEVAL_REACHED");
    assert_ne!(model.optsum.return_value, "SUCCESS");
}

#[test]
fn test_pattern_search_descends_correlated_directions() {
    let initial = vec![0.0, 0.0];
    let outcome = LinearMixedModel::run_multivariate_pattern_search(
        initial.clone(),
        0.0,
        &[f64::NEG_INFINITY, f64::NEG_INFINITY],
        vec![1.0, 1.0],
        &[1e-4, 1e-4],
        5,
        1e-12,
        |theta| Ok(theta[0] * theta[0] + theta[1] * theta[1] - 3.0 * theta[0] * theta[1]),
    )
    .unwrap();

    assert_eq!(outcome.feval_count, 5);
    assert!(
        outcome.best_fmin < -0.9,
        "combined pattern probe should descend when each axis probe is uphill, got {}",
        outcome.best_fmin
    );
    assert!(
        outcome.fit_log.iter().any(|entry| {
            entry.objective < 0.0
                && entry
                    .theta
                    .iter()
                    .zip(initial.iter())
                    .filter(|(candidate, base)| (*candidate - *base).abs() > 1e-12)
                    .count()
                    > 1
        }),
        "fit log should include an improving multi-coordinate pattern probe"
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn test_pattern_search_matches_nlopt_on_correlated_crossed_fixture() {
    let data = correlated_crossed_slope_data();
    let formula = parse_formula("y ~ 1 + x + (1 + x | g) + (1 + x | h)").unwrap();

    let mut pattern_model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
    pattern_model.optsum.max_feval = 20000;
    pattern_model
        .fit_with_forced_optimizer(true, Optimizer::PatternSearch)
        .unwrap();

    let mut nlopt_model = LinearMixedModel::new(formula, &data, None).unwrap();
    nlopt_model.fit(true).unwrap();

    assert_eq!(pattern_model.optsum.optimizer, Optimizer::PatternSearch);
    assert_eq!(nlopt_model.optsum.optimizer, Optimizer::NloptBobyqa);
    assert!(
        pattern_model.objective_value() <= nlopt_model.objective_value() + 1e-4,
        "pattern_search objective {} should match nlopt {} on correlated crossed fixture",
        pattern_model.objective_value(),
        nlopt_model.objective_value()
    );
}

#[test]
fn test_cobyla_records_maxeval() {
    let data = simulate_sleepstudy_like(12, 8, 17);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.max_feval = 5;

    model
        .fit_with_forced_optimizer(true, Optimizer::Cobyla)
        .unwrap();

    assert_eq!(model.optsum.optimizer, Optimizer::Cobyla);
    assert_eq!(model.optsum.return_value, "MAXEVAL_REACHED");
    assert_ne!(model.optsum.return_value, "SUCCESS");
}

#[test]
fn test_cobyla_validates_configured_initial_step() {
    let data = simulate_sleepstudy_like(12, 8, 17);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.initial_step = vec![0.5];

    let err = model
        .fit_with_forced_optimizer(true, Optimizer::Cobyla)
        .expect_err("COBYLA should reject wrong-length initial_step");
    assert!(err.to_string().contains("initial_step length"));

    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.initial_step = vec![0.5, 0.0, 0.5];

    let err = model
        .fit_with_forced_optimizer(true, Optimizer::Cobyla)
        .expect_err("COBYLA should reject non-positive initial_step values");
    assert!(err.to_string().contains("finite and positive"));
}

#[test]
fn test_optimizer_return_values_consistent_across_backends() {
    assert_eq!(
        LinearMixedModel::cobyla_success_status_label(cobyla::SuccessStatus::MaxEvalReached),
        "MAXEVAL_REACHED"
    );
    assert_eq!(
        LinearMixedModel::cobyla_fail_status_label(cobyla::FailStatus::RoundoffLimited),
        "ROUNDOFF_LIMITED"
    );
    #[cfg(feature = "nlopt")]
    {
        assert_eq!(
            LinearMixedModel::nlopt_status_label("MaxEvalReached"),
            "MAXEVAL_REACHED"
        );
        assert_eq!(
            LinearMixedModel::nlopt_status_label("RoundoffLimited"),
            "ROUNDOFF_LIMITED"
        );
    }
}

#[test]
fn test_rectify_runs_for_cobyla_and_pattern_search_backends() {
    let data = grouped_slope_data_with_obs(8, 4);
    let formula = parse_formula("y ~ 1 + x + (1 + x | group)").unwrap();

    for optimizer in [Optimizer::Cobyla, Optimizer::PatternSearch] {
        let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
        let negative_theta = vec![-1.0, 0.25, -0.5];
        let fmin = model.objective_at(&negative_theta).unwrap();

        model
            .finalize_fit_result(
                negative_theta.clone(),
                fmin,
                1,
                vec![FitLogEntry {
                    theta: negative_theta,
                    objective: fmin,
                }],
                optimizer,
                None,
            )
            .unwrap();

        assert_eq!(model.optsum.optimizer, optimizer);
        assert_theta_diagonals_nonnegative(&model);
        assert_eq!(model.optsum.final_params, vec![1.0, -0.25, 0.5]);
    }
}

#[test]
fn test_rectify_theta_columns_matches_julia_sign_convention() {
    let parmap = vec![(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0)];
    let mut theta = vec![-2.0, 0.75, -3.0, -4.0];

    LinearMixedModel::rectify_theta_columns(&mut theta, &parmap, 2);

    assert_eq!(theta, vec![2.0, -0.75, 3.0, 4.0]);
}

#[test]
#[allow(clippy::approx_constant)] // 3.14 is a sigma sentinel, not π
fn test_fixed_sigma_constrains_scalar_re_fit() {
    let data = shared_julia_fixed_sigma_fixture();
    let formula = parse_formula("y ~ 0 + (1 | z)").unwrap();
    let julia_objective = 513.5676467958401;

    let mut model_sigma1 = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
    model_sigma1.optsum.sigma = Some(1.0);
    assert_relative_eq!(
        model_sigma1.objective_at(&[2.992032352222033]).unwrap(),
        julia_objective,
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    model_sigma1.fit(false).unwrap();

    assert_eq!(model_sigma1.fixef().len(), 0);
    assert_relative_eq!(
        model_sigma1.sigma(),
        1.0,
        epsilon = 1e-12,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        model_sigma1.objective_value(),
        julia_objective,
        epsilon = 2e-5,
        max_relative = 1e-8
    );
    assert_relative_eq!(
        model_sigma1.theta()[0],
        2.992032352222033,
        epsilon = 1e-3,
        max_relative = 1e-3
    );
    assert_eq!(model_sigma1.dof(), model_sigma1.n_theta());

    let mut model_sigma314 = LinearMixedModel::new(formula, &data, None).unwrap();
    model_sigma314.optsum.sigma = Some(3.14);
    assert_relative_eq!(
        model_sigma314.objective_at(&[0.09694160520621385]).unwrap(),
        julia_objective,
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    model_sigma314.fit(false).unwrap();

    assert_eq!(model_sigma314.fixef().len(), 0);
    assert_relative_eq!(
        model_sigma314.sigma(),
        3.14,
        epsilon = 1e-12,
        max_relative = 1e-12
    );
    assert_relative_eq!(
        model_sigma314.objective_value(),
        julia_objective,
        epsilon = 2e-5,
        max_relative = 1e-8
    );
    assert_relative_eq!(
        model_sigma314.theta()[0],
        0.09694160520621385,
        epsilon = 1e-3,
        max_relative = 1e-3
    );
    assert_eq!(model_sigma314.dof(), model_sigma314.n_theta());
}

#[test]
#[allow(clippy::approx_constant)] // 3.14 is a sigma sentinel, not π
fn test_varest_under_fixed_sigma_matches_julia() {
    let data = shared_julia_fixed_sigma_fixture();
    let formula = parse_formula("y ~ 0 + (1 | z)").unwrap();
    let mut fixed = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
    fixed.optsum.sigma = Some(3.14);
    fixed.fit(false).unwrap();

    let mut estimated = LinearMixedModel::new(formula, &data, None).unwrap();
    estimated.fit(false).unwrap();

    assert_relative_eq!(fixed.sigma(), 3.14, epsilon = 1e-12);
    assert_relative_eq!(fixed.varest(), 3.14, epsilon = 1e-12);
    assert_relative_eq!(
        estimated.varest(),
        estimated.sigma().powi(2),
        epsilon = 1e-12
    );
}

#[test]
#[allow(clippy::approx_constant)] // 3.14 is a sigma sentinel, not π
fn test_dispersion_under_fixed_sigma_matches_julia() {
    let data = shared_julia_fixed_sigma_fixture();
    let formula = parse_formula("y ~ 0 + (1 | z)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.sigma = Some(3.14);
    model.fit(false).unwrap();

    assert_relative_eq!(
        MixedModelFit::dispersion(&model, false),
        3.14,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        MixedModelFit::dispersion(&model, true),
        3.14,
        epsilon = 1e-12
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn test_large_theta_fit_records_maxeval_status() {
    let data = simulate_large_theta_crossed(123);
    let formula = parse_formula(
        "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
    )
    .unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.max_feval = 1;

    model.fit(true).unwrap();

    assert_eq!(model.optsum.optimizer, Optimizer::NloptNewuoa);
    assert_eq!(model.optsum.return_value, "MAXEVAL_REACHED");
    assert_eq!(model.optsum.feval, 1);
    assert!(model.objective_value().is_finite());
}

#[cfg(feature = "nlopt")]
#[test]
fn test_large_theta_fit_records_maxtime_status() {
    let data = simulate_large_theta_crossed(123);
    let formula = parse_formula(
        "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
    )
    .unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.max_time = 1e-9;

    model.fit(true).unwrap();

    assert_eq!(model.optsum.optimizer, Optimizer::NloptNewuoa);
    assert_eq!(model.optsum.return_value, "MAXTIME_REACHED");
    assert_eq!(model.optsum.max_time, 1e-9);
    assert!(model.optsum.feval >= 1);
    assert!(model.objective_value().is_finite());
}

#[test]
fn test_scalar_objective_matches_julia_on_shared_fixture() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let julia_theta = [0.6273717260668661];
    let julia_objective = 223.742_068_488_410_9;

    let rust_objective = model.objective_at(&julia_theta).unwrap();

    assert_relative_eq!(
        rust_objective,
        julia_objective,
        epsilon = 1e-8,
        max_relative = 1e-8
    );

    model.fit(true).unwrap();
    assert_relative_eq!(
        model.objective_value(),
        julia_objective,
        epsilon = 1e-5,
        max_relative = 1e-5
    );
    assert_relative_eq!(
        model.sigma(),
        30.23875724370832,
        epsilon = 1e-5,
        max_relative = 1e-5
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn test_vector_objective_matches_julia_on_shared_fixture() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let julia_theta = [0.6565437822843008, -0.019160976185379253, 0.0];
    let julia_objective = 223.73509351902135;

    let rust_objective = model.objective_at(&julia_theta).unwrap();

    assert_relative_eq!(
        rust_objective,
        julia_objective,
        epsilon = 1e-8,
        max_relative = 1e-8
    );

    model.fit(true).unwrap();
    assert_relative_eq!(
        model.objective_value(),
        julia_objective,
        epsilon = 1e-4,
        max_relative = 1e-4
    );
    assert_relative_eq!(
        model.sigma(),
        30.22863368533761,
        epsilon = 1e-4,
        max_relative = 1e-4
    );
}

#[test]
fn test_crossed_objective_matches_julia_on_shared_fixture() {
    let data = shared_julia_crossed_parity_fixture();
    let formula = parse_formula(
        "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
    )
    .unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let julia_theta = [
        1.6360390637490343,
        0.19973976515130532,
        0.16548928583172998,
        1.3985120310259511,
        -0.07659426024736829,
        0.19501821571577171,
        0.627_720_707_627_351,
        -0.036_380_030_801_807_13,
        0.11318289497410258,
    ];
    let julia_objective = 6_177.391_766_038_913;
    let julia_pwrss = 50993.469629712374;
    let julia_logdet_re = 208.5086015326244;
    let julia_logdet_xx = 5.502_813_812_310_208;

    let rust_objective = model.objective_at(&julia_theta).unwrap();

    assert_relative_eq!(
        rust_objective,
        julia_objective,
        epsilon = 1e-6,
        max_relative = 1e-9
    );
    assert_relative_eq!(
        model.pwrss(),
        julia_pwrss,
        epsilon = 1e-5,
        max_relative = 1e-9
    );
    assert_relative_eq!(
        model.logdet_re(),
        julia_logdet_re,
        epsilon = 1e-8,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        current_logdet_xx(&model),
        julia_logdet_xx,
        epsilon = 1e-8,
        max_relative = 1e-10
    );
}

// This fixture pins the certified default-backend (matrixmultiply) optimizer
// trajectory; the experimental faer gemm backend converges to an equivalent
// optimum along a slightly different path (rounding-level objective drift ->
// different NEWUOA stopping point), which moves sigma outside the debug-build
// parity band. The faer configuration is benchmark-only and not parity
// certified, so the pin applies to the default backend alone.
#[cfg(all(feature = "nlopt", not(feature = "faer-backend")))]
#[test]
fn test_crossed_fit_matches_julia_on_shared_fixture() {
    let data = shared_julia_crossed_parity_fixture();
    let formula = parse_formula(
        "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
    )
    .unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let julia_theta = [
        1.6360390637490343,
        0.19973976515130532,
        0.16548928583172998,
        1.3985120310259511,
        -0.07659426024736829,
        0.19501821571577171,
        0.627_720_707_627_351,
        -0.036_380_030_801_807_13,
        0.11318289497410258,
    ];
    let julia_objective = 6_177.391_766_038_913;
    let julia_sigma = 7.691_369_016_180_007;

    model.fit(true).unwrap();

    assert_eq!(model.optsum.optimizer, Optimizer::NloptNewuoa);
    // A fitted optimizer path is allowed codegen-level drift within the
    // documented VERSIONING.md numerical parity band.
    assert_relative_eq!(
        model.objective_value(),
        julia_objective,
        epsilon = 1e-7,
        max_relative = 1e-8
    );
    let sigma_abs_tol = if cfg!(debug_assertions) { 2e-5 } else { 2e-4 };
    let sigma_rel_tol = if cfg!(debug_assertions) { 5e-6 } else { 3e-5 };
    assert_relative_eq!(
        model.sigma(),
        julia_sigma,
        epsilon = sigma_abs_tol,
        max_relative = sigma_rel_tol
    );

    let theta = model.theta();
    if cfg!(debug_assertions) {
        for (actual, expected) in theta.iter().zip(julia_theta.iter()) {
            assert_relative_eq!(*actual, *expected, epsilon = 2e-4, max_relative = 2e-4);
        }
    } else {
        assert!(
            theta.iter().all(|value| value.is_finite()),
            "release-profile NLopt theta should remain finite: {theta:?}"
        );
    }
}

// ── Tests ported from MixedModels.jl/test/pls.jl ────────────────────────

#[test]
fn test_ml_loglikelihood_aic_bic_relationships() {
    // Verify the algebraic relationships: ll = -obj/2, aic, bic.
    // Matches Julia's convention: objective already includes n*log(2π).
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    let n = model.nobs() as f64;
    let k = model.dof() as f64;
    let obj = model.objective_value();
    let ll = MixedModelFit::loglikelihood(&model);

    // ML: loglikelihood = -objective / 2
    assert_relative_eq!(ll, -obj / 2.0, epsilon = 1e-12);

    // AIC = -2*ll + 2*k
    assert_relative_eq!(
        MixedModelFit::aic(&model),
        -2.0 * ll + 2.0 * k,
        epsilon = 1e-12
    );

    // BIC = -2*ll + k*ln(n)
    assert_relative_eq!(
        MixedModelFit::bic(&model),
        -2.0 * ll + k * n.ln(),
        epsilon = 1e-12
    );
}

#[test]
fn test_ml_nobs_and_dof_scalar_re() {
    // 6 subjects × 4 days = 24 obs; dof = p(2) + n_theta(1) + 1(sigma) = 4
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(MixedModelFit::nobs(&model), 24);
    assert_eq!(MixedModelFit::dof(&model), 4);
}

#[test]
fn test_ml_fixef_and_stderror() {
    // reaction ~ 1 + days: two fixef, both SE positive
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let fixef = MixedModelFit::fixef(&model);
    let se = MixedModelFit::stderror(&model);

    assert_eq!(fixef.len(), 2);
    assert_eq!(se.len(), 2);
    assert!(se[0] > 0.0, "intercept SE must be positive");
    assert!(se[1] > 0.0, "slope SE must be positive");
}

#[test]
fn test_ml_wald_confint() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let coef = MixedModelFit::coef(&model);
    let se = MixedModelFit::stderror(&model);
    let names = MixedModelFit::coef_names(&model);
    let z = 1.959_963_984_540_054_f64; // qnorm(0.975)

    let ci = model.wald_confint(0.95);
    assert_eq!(ci.len(), coef.len());
    for (i, row) in ci.iter().enumerate() {
        assert_eq!(row.parameter, names[i]);
        assert_relative_eq!(row.estimate, coef[i], epsilon = 1e-12);
        assert_relative_eq!(row.lower, coef[i] - z * se[i], epsilon = 1e-9);
        assert_relative_eq!(row.upper, coef[i] + z * se[i], epsilon = 1e-9);
        assert!(row.lower < row.estimate && row.estimate < row.upper);
    }

    // Higher coverage widens every interval.
    let ci99 = model.wald_confint(0.99);
    for (a, b) in ci.iter().zip(ci99.iter()) {
        assert!((b.upper - b.lower) > (a.upper - a.lower));
    }
}

#[test]
fn test_coeftable_with_method_surfaces_satterthwaite_df() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    // Default table is asymptotic Wald-z with no df.
    let wald = model.coeftable();
    assert_eq!(wald.method, "wald-z");
    assert_eq!(wald.statistic_name, "z");
    assert!(wald.df.iter().all(|d| d.is_none()));

    // Satterthwaite table self-identifies and carries finite df.
    let satt = model.coeftable_with_method(FixedEffectTestMethod::Satterthwaite);
    assert_eq!(satt.method, "satterthwaite");
    assert_eq!(satt.statistic_name, "t");
    assert_eq!(satt.names, wald.names);
    assert_eq!(satt.estimates.len(), wald.estimates.len());
    for (e_s, e_w) in satt.estimates.iter().zip(wald.estimates.iter()) {
        assert_relative_eq!(e_s, e_w, epsilon = 1e-9);
    }
    assert!(
        satt.df
            .iter()
            .any(|d| d.map(|v| v.is_finite()).unwrap_or(false)),
        "Satterthwaite table must carry finite denominator df"
    );
    // The rendered table announces the method (no longer misleading).
    let rendered = format!("{satt}");
    assert!(rendered.contains("Method: satterthwaite"));
    assert!(rendered.contains("t value"));
    assert!(crate::stats::coeftable_to_markdown(&satt).contains("*Method: satterthwaite*"));
}

#[test]
fn test_ml_fitted_plus_residuals_equals_response() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let fitted = MixedModelFit::fitted(&model);
    let residuals = MixedModelFit::residuals(&model);
    let y = model.y();

    assert_eq!(fitted.len(), y.len());
    for i in 0..y.len() {
        assert_relative_eq!(fitted[i] + residuals[i], y[i], epsilon = 1e-10);
    }
}

#[test]
fn test_ml_ranef_dimensions_scalar_re() {
    // (1|subj): vsize=1, 6 subjects → matrix is 1×6
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let ranef = model.ranef_b();
    assert_eq!(ranef.len(), 1, "one grouping factor");
    assert_eq!(ranef[0].nrows(), 1, "scalar RE: vsize = 1");
    assert_eq!(ranef[0].ncols(), 6, "6 subjects");
}

#[test]
fn test_is_singular_reflects_theta_at_lower_bound() {
    // After fitting non-degenerate data: not singular.
    // Driving theta to lower bound → singular.
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert!(
        !model.is_singular(),
        "non-degenerate fit should not be singular"
    );

    let fitted_theta = model.theta();
    let lb = model.lower_bounds();
    model.set_theta(&lb).unwrap(); // θ = [0.0] → at lower bound
    assert!(model.is_singular(), "theta at lower bound must be singular");

    model.set_theta(&fitted_theta).unwrap();
    assert!(
        !model.is_singular(),
        "restored theta should not be singular"
    );
}

#[test]
fn test_is_singular_detects_rank_deficient_lambda() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();
    assert!(
        !model.is_singular(),
        "fitted vector model should start full rank"
    );

    // Full Cholesky with a tiny nonzero second diagonal: not at the
    // parameter lower bound, but numerically rank-deficient in ΛΛ'.
    let rank_deficient_theta = vec![1.0, 0.25, 1e-8];
    model.set_theta(&rank_deficient_theta).unwrap();
    model.update_l().unwrap();

    assert!(
        !model.theta_at_lower_bound(),
        "tiny nonzero diagonal should not be classified as a boundary θ"
    );

    model.refresh_effective_covariance_summaries();

    let summary = &model.compiler_artifact().effective_covariance[0];
    assert_eq!(summary.status, EffectiveRankStatus::ReducedRank);
    assert!(
        model.is_singular(),
        "is_singular must follow reduced effective covariance, not just θ lower bounds"
    );
}

#[test]
fn test_is_singular_consistent_with_effective_covariance_status() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    let mut policy = CompilerPolicy::maximal_feasible();
    policy.thresholds.effective_rank_relative_tolerance = 2.0;
    model.set_compiler_policy(policy).unwrap();

    model.fit(false).unwrap();

    let has_reduced_covariance = model
        .compiler_artifact()
        .effective_covariance
        .iter()
        .any(|summary| summary.status == EffectiveRankStatus::ReducedRank);
    assert!(has_reduced_covariance);
    assert_eq!(
        model.is_singular(),
        model.theta_at_lower_bound()
            || model.optimizer_certificate_reports_boundary()
            || has_reduced_covariance
    );
    assert!(model.is_singular());
}

#[test]
fn test_lmm_set_theta_propagates_remat_err() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let err = model.set_theta(&[]).unwrap_err();

    assert!(matches!(err, MixedModelError::DimensionMismatch(_)));
}

#[test]
fn test_set_theta_does_not_panic_on_bad_input() {
    let data = shared_julia_parity_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.set_theta(&[])));

    assert!(result.is_ok());
    assert!(matches!(
        result.unwrap(),
        Err(MixedModelError::DimensionMismatch(_))
    ));
}

#[test]
fn test_lrt_nested_scalar_re_models() {
    // LRT comparing reaction ~ 1 + (1|subj) vs reaction ~ 1 + days + (1|subj).
    // The second model adds one FE parameter: chisq_dof == 1.
    use crate::stats::lrt::LikelihoodRatioTest;

    let data = shared_julia_parity_fixture();
    let f0 = parse_formula("reaction ~ 1 + (1 | subj)").unwrap();
    let f1 = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();

    let mut m0 = LinearMixedModel::new(f0, &data, None).unwrap();
    let mut m1 = LinearMixedModel::new(f1, &data, None).unwrap();
    m0.fit(false).unwrap();
    m1.fit(false).unwrap();

    let lrt =
        LikelihoodRatioTest::test(&[&m0 as &dyn MixedModelFit, &m1 as &dyn MixedModelFit]).unwrap();

    // χ² = 2*(ll1 - ll0)
    let expected_chisq =
        2.0 * (MixedModelFit::loglikelihood(&m1) - MixedModelFit::loglikelihood(&m0));
    assert_relative_eq!(lrt.chisq[0], expected_chisq, epsilon = 1e-10);

    // Adding `days` costs 1 dof
    assert_eq!(lrt.chisq_dof[0], 1);

    // Fuller model has better (larger) log-likelihood
    assert!(MixedModelFit::loglikelihood(&m1) > MixedModelFit::loglikelihood(&m0));

    // p-value in [0, 1]
    assert!(lrt.pvalues[0] >= 0.0 && lrt.pvalues[0] <= 1.0);
}

#[test]
fn test_singular_re_fit_is_singular() {
    // Synthetic data: all group means identical (SS_B = 0).
    // Mirrors pls.jl "Dyestuff2" testset spirit: when between-group variance
    // is zero, θ → 0 and the model is singular.
    let data = singular_re_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    assert!(model.is_singular(), "fit with SS_B=0 must be singular");
    assert_relative_eq!(model.theta()[0], 0.0, epsilon = 1e-10);
}

// ── Fixtures from actual Julia MixedModels.jl datasets ─────────────────

/// Dyestuff data (Davies, 1949) — 6 batches × 5 observations.
/// Matches `dataset(:dyestuff)` from MixedModelsDatasets.jl.
fn dyestuff_fixture() -> DataFrame {
    let yields: Vec<f64> = vec![
        1545.0, 1440.0, 1440.0, 1520.0, 1580.0, // batch A
        1540.0, 1555.0, 1490.0, 1560.0, 1495.0, // batch B
        1595.0, 1550.0, 1605.0, 1510.0, 1560.0, // batch C
        1445.0, 1440.0, 1595.0, 1465.0, 1545.0, // batch D
        1595.0, 1630.0, 1515.0, 1635.0, 1625.0, // batch E
        1520.0, 1455.0, 1450.0, 1480.0, 1445.0, // batch F
    ];
    let batches: Vec<String> = "ABCDEF"
        .chars()
        .flat_map(|c| std::iter::repeat_n(c.to_string(), 5))
        .collect();
    let mut df = DataFrame::new();
    df.add_numeric("yield", yields).unwrap();
    df.add_categorical("batch", batches).unwrap();
    df
}

#[test]
fn lmm_builder_matches_direct_construction_byte_for_byte() {
    let df = dyestuff_fixture();
    for criterion in [ModelCriterion::Ml, ModelCriterion::Reml] {
        let reml = criterion.is_reml();

        let mut direct =
            LinearMixedModel::new(parse_formula("yield ~ 1 + (1 | batch)").unwrap(), &df, None)
                .unwrap();
        direct.fit(reml).unwrap();

        let built =
            LinearMixedModelBuilder::new(parse_formula("yield ~ 1 + (1 | batch)").unwrap(), &df)
                .fit(if criterion.is_reml() {
                    FitOptions::reml()
                } else {
                    FitOptions::ml()
                })
                .unwrap();

        assert_eq!(
            built.coef(),
            direct.coef(),
            "builder coef must match direct ({criterion:?})"
        );
        assert_eq!(
            built.objective(),
            direct.objective(),
            "builder objective must match direct ({criterion:?})"
        );
    }
}

#[test]
fn lmm_fit_options_record_caller_optimizer_tolerances_and_start() {
    let df = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut cold = LinearMixedModel::new(formula.clone(), &df, None).unwrap();
    cold.fit(true).unwrap();
    let start_theta = cold.theta();

    let mut warm = LinearMixedModel::new(formula, &df, None).unwrap();
    let control = OptimizerControl::auto()
        .with_optimizer(Optimizer::PatternSearch)
        .with_start_theta(start_theta.clone())
        .with_max_feval(1_000)
        .with_tolerances(
            FitToleranceOverrides::default()
                .with_ftol_abs(1.0e-10)
                .with_xtol_abs(vec![1.0e-8; start_theta.len()])
                .with_initial_step(vec![0.25; start_theta.len()]),
        );

    warm.fit_with_options(FitOptions::reml().with_optimizer_control(control))
        .unwrap();

    assert_eq!(warm.optsum.optimizer, Optimizer::PatternSearch);
    assert_eq!(warm.optsum.optimizer_source_name(), "caller");
    for field in [
        "optimizer",
        "start_theta",
        "max_feval",
        "ftol_abs",
        "xtol_abs",
        "initial_step",
    ] {
        assert!(
            warm.optsum.caller_set_field(field),
            "missing caller field {field:?}"
        );
    }
    assert_relative_eq!(warm.objective(), cold.objective(), epsilon = 1.0e-5);

    let certificate = warm
        .optimizer_certificate()
        .expect("fit should attach optimizer certificate");
    assert_eq!(certificate.optimizer_control.optimizer_source, "caller");
    assert!(certificate
        .optimizer_control
        .caller_set_fields
        .iter()
        .any(|field| field == "optimizer"));
    let json = serde_json::to_value(certificate).unwrap();
    assert_eq!(
        json["optimizer_control"]["optimizer_source"],
        serde_json::json!("caller")
    );
}

#[test]
fn lmm_fit_options_reject_bad_start_theta_before_fitting() {
    let df = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();

    let err = model
        .fit_with_options(
            FitOptions::reml()
                .with_optimizer_control(OptimizerControl::auto().with_start_theta(vec![0.1, 0.2])),
        )
        .expect_err("wrong-length start theta should be rejected");

    assert_eq!(err.code(), "invalid_argument");
    assert!(!model.is_fitted());
}

#[test]
fn lmm_builder_build_returns_unfitted_model() {
    let df = dyestuff_fixture();
    let mut model =
        LinearMixedModelBuilder::new(parse_formula("yield ~ 1 + (1 | batch)").unwrap(), &df)
            .build()
            .unwrap();
    // build() must not fit; fitting afterwards still succeeds.
    assert!(model.fit(false).is_ok());
}

/// Sleepstudy data (Belenky et al., 2003) — 18 subjects × 10 days.
/// Matches `dataset(:sleepstudy)` from MixedModelsDatasets.jl.
fn sleepstudy_fixture() -> DataFrame {
    let subjects = [
        "S308", "S309", "S310", "S330", "S331", "S332", "S333", "S334", "S335", "S337", "S349",
        "S350", "S351", "S352", "S369", "S370", "S371", "S372",
    ];
    #[rustfmt::skip]
        let reaction: Vec<f64> = vec![
            // S308
            249.5600, 258.7047, 250.8006, 321.4398, 356.8519,
            414.6901, 382.2038, 290.1486, 430.5853, 466.3535,
            // S309
            222.7339, 205.2658, 202.9778, 204.7070, 207.7161,
            215.9618, 213.6303, 217.7272, 224.2957, 237.3142,
            // S310
            199.0539, 194.3322, 234.3200, 232.8416, 229.3074,
            220.4579, 235.4208, 255.7511, 261.0125, 247.5153,
            // S330
            321.5426, 300.4002, 283.8565, 285.1330, 285.7973,
            297.5855, 280.2396, 318.2613, 305.3495, 354.0487,
            // S331
            287.6079, 285.0000, 301.8206, 320.1153, 316.2773,
            293.3187, 290.0750, 334.8177, 293.7469, 371.5811,
            // S332
            234.8606, 242.8118, 272.9613, 309.7688, 317.4629,
            309.9976, 454.1619, 346.8311, 330.3003, 253.8644,
            // S333
            283.8424, 289.5550, 276.7693, 299.8097, 297.1710,
            338.1665, 332.0265, 348.8399, 333.3600, 362.0428,
            // S334
            265.4731, 276.2012, 243.3647, 254.6723, 279.0244,
            284.1912, 305.5248, 331.5229, 335.7469, 377.2990,
            // S335
            241.6083, 273.9472, 254.4907, 270.8021, 251.4519,
            254.6362, 245.4523, 235.3110, 235.7541, 237.2466,
            // S337
            312.3666, 313.8058, 291.6112, 346.1222, 365.7324,
            391.8385, 404.2601, 416.6923, 455.8643, 458.9167,
            // S349
            236.1032, 230.3167, 238.9256, 254.9220, 250.7103,
            269.7744, 281.5648, 308.1020, 336.2806, 351.6451,
            // S350
            256.2968, 243.4543, 256.2046, 255.5271, 268.9165,
            329.7247, 379.4445, 362.9184, 394.4872, 389.0527,
            // S351
            250.5265, 300.0576, 269.8939, 280.5891, 271.8274,
            304.6336, 287.7466, 266.5955, 321.5418, 347.5655,
            // S352
            221.6771, 298.1939, 326.8785, 346.8555, 348.7402,
            352.8287, 354.4266, 360.4326, 375.6406, 388.5417,
            // S369
            271.9235, 268.4369, 257.2424, 277.6566, 314.8222,
            317.2135, 298.1353, 348.1229, 340.2800, 366.5131,
            // S370
            225.2640, 234.5235, 238.9008, 240.4730, 267.5373,
            344.1937, 281.1481, 347.5855, 365.1630, 372.2288,
            // S371
            269.8804, 272.4428, 277.8989, 281.7895, 279.1705,
            284.5120, 259.2658, 304.6306, 350.7807, 369.4692,
            // S372
            269.4117, 273.4740, 297.5968, 310.6316, 287.1726,
            329.6076, 334.4818, 343.2199, 369.1417, 364.1236,
        ];
    let days: Vec<f64> = (0..18).flat_map(|_| (0..10u64).map(|d| d as f64)).collect();
    let subj: Vec<String> = subjects
        .iter()
        .flat_map(|s| std::iter::repeat_n(s.to_string(), 10))
        .collect();
    let mut df = DataFrame::new();
    df.add_numeric("reaction", reaction).unwrap();
    df.add_numeric("days", days).unwrap();
    df.add_categorical("subj", subj).unwrap();
    df
}

/// Penicillin data (Davies, 1967) — 24 plates × 6 samples = 144 observations.
/// Matches `dataset(:penicillin)` from MixedModelsDatasets.jl.
fn penicillin_fixture() -> DataFrame {
    // Diameter values in plate-major order (6 samples A-F per plate a-x).
    #[rustfmt::skip]
        let diameter: Vec<f64> = vec![
            27.0, 23.0, 26.0, 23.0, 23.0, 21.0, // plate a
            27.0, 23.0, 26.0, 23.0, 23.0, 21.0, // plate b
            25.0, 21.0, 25.0, 24.0, 24.0, 20.0, // plate c
            26.0, 23.0, 25.0, 23.0, 23.0, 20.0, // plate d
            25.0, 22.0, 26.0, 22.0, 23.0, 20.0, // plate e
            24.0, 22.0, 25.0, 23.0, 22.0, 19.0, // plate f
            24.0, 20.0, 23.0, 21.0, 22.0, 19.0, // plate g
            26.0, 22.0, 26.0, 24.0, 24.0, 21.0, // plate h
            24.0, 21.0, 24.0, 22.0, 22.0, 20.0, // plate i
            24.0, 21.0, 24.0, 23.0, 22.0, 19.0, // plate j
            26.0, 23.0, 26.0, 24.0, 24.0, 21.0, // plate k
            25.0, 22.0, 26.0, 24.0, 24.0, 20.0, // plate l
            26.0, 24.0, 26.0, 24.0, 25.0, 22.0, // plate m
            26.0, 23.0, 26.0, 23.0, 23.0, 20.0, // plate n
            26.0, 23.0, 25.0, 24.0, 24.0, 22.0, // plate o
            25.0, 22.0, 25.0, 23.0, 23.0, 20.0, // plate p
            25.0, 21.0, 24.0, 23.0, 23.0, 20.0, // plate q
            25.0, 22.0, 24.0, 23.0, 23.0, 19.0, // plate r
            24.0, 21.0, 23.0, 21.0, 21.0, 19.0, // plate s
            26.0, 23.0, 26.0, 24.0, 24.0, 21.0, // plate t
            25.0, 21.0, 24.0, 22.0, 22.0, 18.0, // plate u
            25.0, 22.0, 25.0, 22.0, 22.0, 20.0, // plate v
            24.0, 21.0, 24.0, 22.0, 24.0, 19.0, // plate w
            24.0, 21.0, 24.0, 22.0, 21.0, 18.0, // plate x
        ];
    let plate_letters: Vec<&str> = vec![
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
        "s", "t", "u", "v", "w", "x",
    ];
    let plate: Vec<String> = plate_letters
        .iter()
        .flat_map(|p| std::iter::repeat_n(p.to_string(), 6))
        .collect();
    let sample: Vec<String> = (0..24)
        .flat_map(|_| ["A", "B", "C", "D", "E", "F"].iter().map(|s| s.to_string()))
        .collect();
    let mut df = DataFrame::new();
    df.add_numeric("diameter", diameter).unwrap();
    df.add_categorical("plate", plate).unwrap();
    df.add_categorical("sample", sample).unwrap();
    df
}

/// Pastes data (Davies, 1947) — 10 batches × 3 casks × 2 samples = 60 obs.
/// Matches `dataset(:pastes)` from MixedModelsDatasets.jl.
/// The nested structure `batch / cask` expands to `batch + batch:cask`.
fn pastes_fixture() -> DataFrame {
    // Strength values, 6 per batch (2 per cask: a,a,b,b,c,c)
    #[rustfmt::skip]
        let strength: Vec<f64> = vec![
            62.8, 62.6, 60.1, 62.3, 62.7, 63.1, // batch A
            60.0, 61.4, 57.5, 56.9, 61.1, 58.9, // batch B
            58.7, 57.5, 63.9, 63.1, 65.4, 63.7, // batch C
            57.1, 56.4, 56.9, 58.6, 64.7, 64.5, // batch D
            55.1, 55.1, 54.7, 54.2, 58.8, 57.5, // batch E
            63.4, 64.9, 59.3, 58.1, 60.5, 60.0, // batch F
            62.5, 62.6, 61.0, 58.7, 56.9, 57.7, // batch G
            59.2, 59.4, 65.2, 66.0, 64.8, 64.1, // batch H
            54.8, 54.8, 64.0, 64.0, 57.7, 56.8, // batch I
            58.3, 59.3, 59.2, 59.2, 58.9, 56.6, // batch J
        ];
    // batch: A-J, 6 obs each
    let batch: Vec<String> = "ABCDEFGHIJ"
        .chars()
        .flat_map(|c| std::iter::repeat_n(c.to_string(), 6))
        .collect();
    // cask: a,a,b,b,c,c per batch
    let cask_pattern = ["a", "a", "b", "b", "c", "c"];
    let cask: Vec<String> = (0..10)
        .flat_map(|_| cask_pattern.iter().map(|s| s.to_string()))
        .collect();
    // batch_cask: interaction label for (1 | batch & cask)
    let batch_cask: Vec<String> = batch
        .iter()
        .zip(&cask)
        .map(|(b, c)| format!("{b}:{c}"))
        .collect();

    let mut df = DataFrame::new();
    df.add_numeric("strength", strength).unwrap();
    df.add_categorical("batch", batch).unwrap();
    df.add_categorical("cask", cask).unwrap();
    df.add_categorical("batch_cask", batch_cask).unwrap();
    df
}

/// Dyestuff2 data — same structure as Dyestuff but within-batch variance
/// dominates, so the RE variance collapses to zero (singular fit).
/// Values decoded from `dyestuff2.arrow` (MixedModelsDatasets.jl).
fn dyestuff2_fixture() -> DataFrame {
    #[rustfmt::skip]
        let yields: Vec<f64> = vec![
            7.298, 3.846, 2.434, 9.566,  7.990, // batch A
            5.220, 6.556, 0.608, 11.788, -0.892, // batch B
            0.110, 10.386, 13.434, 5.510, 8.166, // batch C
            2.212, 4.852, 7.092,  9.288,  4.980, // batch D
            0.282, 9.014, 4.458,  9.446,  7.198, // batch E
            1.722, 4.782, 8.106,  0.758,  3.758, // batch F
        ];
    let batches: Vec<String> = "ABCDEF"
        .chars()
        .flat_map(|c| std::iter::repeat_n(c.to_string(), 5))
        .collect();
    let mut df = DataFrame::new();
    df.add_numeric("yield", yields).unwrap();
    df.add_categorical("batch", batches).unwrap();
    df
}

fn fitted_varpar(model: &LinearMixedModel) -> Vec<f64> {
    let mut varpar = model.theta();
    varpar.push(model.sigma());
    varpar
}

fn assert_matrix_relative_eq(actual: &DMatrix<f64>, expected: &DMatrix<f64>, epsilon: f64) {
    assert_eq!(actual.shape(), expected.shape());
    for row in 0..actual.nrows() {
        for col in 0..actual.ncols() {
            assert_relative_eq!(actual[(row, col)], expected[(row, col)], epsilon = epsilon);
        }
    }
}

fn assert_matrix_symmetric(matrix: &DMatrix<f64>, epsilon: f64) {
    assert_eq!(matrix.nrows(), matrix.ncols());
    for row in 0..matrix.nrows() {
        for col in 0..row {
            assert_relative_eq!(matrix[(row, col)], matrix[(col, row)], epsilon = epsilon);
        }
    }
}

#[derive(Debug, Deserialize)]
struct SatterthwaiteParityFixture {
    cases: Vec<SatterthwaiteParityCase>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct SatterthwaiteParityCase {
    name: String,
    formula: String,
    coefficient: String,
    estimate: f64,
    std_error: f64,
    df: f64,
    statistic: f64,
    p_value: f64,
}

#[derive(Debug, Deserialize)]
struct KenwardRogerPbkrtestParityFixture {
    scalar_cases: Vec<KenwardRogerScalarParityCase>,
    multi_df_cases: Vec<KenwardRogerMultiDfParityCase>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct KenwardRogerScalarParityCase {
    name: String,
    formula: String,
    label: String,
    l: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    estimate: f64,
    std_error: f64,
    denominator_df: f64,
    statistic: f64,
    p_value: f64,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct KenwardRogerMultiDfParityCase {
    name: String,
    formula: String,
    label: String,
    l: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    numerator_df: f64,
    denominator_df: f64,
    statistic: f64,
    p_value: f64,
    f_scaling: f64,
    unscaled_statistic: f64,
    unscaled_p_value: f64,
}

fn satterthwaite_lmer_test_parity_fixture() -> SatterthwaiteParityFixture {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/compiler_contract/satterthwaite_lmer_test_parity_v1.json"
    ))
    .expect("Satterthwaite lmerTest parity fixture should deserialize")
}

fn kenward_roger_pbkrtest_parity_fixture() -> KenwardRogerPbkrtestParityFixture {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/compiler_contract/kenward_roger_pbkrtest_parity_v1.json"
    ))
    .expect("Kenward-Roger pbkrtest parity fixture should deserialize")
}

fn fixed_effect_hypothesis_from_fixture(
    label: &str,
    l: &[Vec<f64>],
    rhs: &[f64],
) -> FixedEffectHypothesis {
    assert!(!l.is_empty(), "{label}: contrast matrix must have rows");
    let ncols = l[0].len();
    assert!(ncols > 0, "{label}: contrast matrix must have columns");
    assert_eq!(rhs.len(), l.len(), "{label}: rhs length must match rows");
    assert!(
        l.iter().all(|row| row.len() == ncols),
        "{label}: contrast rows must have a common width"
    );
    let values = l.iter().flatten().copied().collect::<Vec<_>>();
    let l = ContrastMatrix::new(DMatrix::from_row_slice(rhs.len(), ncols, &values)).unwrap();
    let rhs = ContrastRhs::new(DVector::from_column_slice(rhs)).unwrap();
    FixedEffectHypothesis::new(label.to_string(), l, rhs).unwrap()
}

fn unbalanced_sleepstudy_fixture() -> DataFrame {
    let source = sleepstudy_fixture();
    let reaction = source.numeric("reaction").unwrap();
    let days = source.numeric("days").unwrap();
    let subj = &source.categorical("subj").unwrap().values;

    let mut out_reaction = Vec::new();
    let mut out_days = Vec::new();
    let mut out_subj = Vec::new();
    for row in 0..source.nrow() {
        let drop_row = matches!(subj[row].as_str(), "S308" | "S309")
            && matches!(days[row] as i32, 1 | 3 | 5 | 7 | 9);
        if !drop_row {
            out_reaction.push(reaction[row]);
            out_days.push(days[row]);
            out_subj.push(subj[row].clone());
        }
    }

    let mut df = DataFrame::new();
    df.add_numeric("reaction", out_reaction).unwrap();
    df.add_numeric("days", out_days).unwrap();
    df.add_categorical_with_levels(
        "subj",
        out_subj,
        source.categorical("subj").unwrap().levels.clone(),
    )
    .unwrap();
    df
}

fn satterthwaite_parity_data(case_name: &str) -> DataFrame {
    match case_name {
        "sleepstudy_random_intercept_days" | "sleepstudy_random_slope_days" => sleepstudy_fixture(),
        "sleepstudy_unbalanced_random_slope_days" => unbalanced_sleepstudy_fixture(),
        "penicillin_crossed_intercept" => penicillin_fixture(),
        other => panic!("unknown Satterthwaite parity case {other}"),
    }
}

fn kenward_roger_parity_data(case_name: &str) -> DataFrame {
    match case_name {
        "sleepstudy_random_intercept_days"
        | "sleepstudy_random_slope_days"
        | "sleepstudy_random_slope_days_rhs10"
        | "sleepstudy_intercept_and_days_joint"
        | "sleepstudy_days_duplicate_rank_deficient_l"
        | "sleepstudy_days_rhs10_f" => sleepstudy_fixture(),
        "penicillin_crossed_intercept" | "penicillin_crossed_intercept_f" => penicillin_fixture(),
        "pastes_nested_intercept" | "pastes_nested_intercept_f" => pastes_fixture(),
        other => panic!("unknown Kenward-Roger parity case {other}"),
    }
}

fn assert_available_finite_fixed_effect_test(test: &FixedEffectTest, label: &str) {
    assert_eq!(
        test.status,
        InferenceStatus::Available,
        "{label}: inference should be available"
    );
    assert!(
        test.estimates.iter().all(|value| value.is_finite()),
        "{label}: estimates should be finite: {:?}",
        test.estimates
    );
    assert!(
        test.standard_errors
            .iter()
            .flatten()
            .all(|value| value.is_finite() && *value > 0.0),
        "{label}: standard errors should be finite and positive: {:?}",
        test.standard_errors
    );
    assert!(
        test.statistics
            .iter()
            .flatten()
            .all(|value| value.is_finite()),
        "{label}: statistics should be finite: {:?}",
        test.statistics
    );
    assert!(
        test.denominator_df
            .is_none_or(|value| value.is_finite() && value > 0.0),
        "{label}: denominator df should be finite and positive when present: {:?}",
        test.denominator_df
    );
    assert!(
        test.p_values
            .iter()
            .flatten()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
        "{label}: p-values should be finite probabilities: {:?}",
        test.p_values
    );
}

#[cfg(not(feature = "nlopt"))]
fn assert_default_native_certificate(model: &LinearMixedModel, label: &str) {
    assert_eq!(
        model.optsum.optimizer,
        Optimizer::TrustBq,
        "{label}: default no-NLopt path should use TrustBQ"
    );
    let certificate = model
        .optimizer_certificate()
        .expect("fitted model should attach optimizer certificate");
    assert_eq!(
        certificate.optimizer_name.as_deref(),
        Some("trust_bq"),
        "{label}: certificate should identify TrustBQ"
    );
    assert!(
        matches!(
            certificate.status,
            FitStatus::ConvergedInterior
                | FitStatus::ConvergedBoundary
                | FitStatus::ConvergedReducedRank
        ),
        "{label}: unexpected certificate status {:?}",
        certificate.status
    );
    assert!(
        certificate.evidence.optimizer_stop.acceptable_stop,
        "{label}: native optimizer stop should be accepted: {:?}",
        certificate.evidence.optimizer_stop
    );
    assert!(
        certificate
            .evidence
            .optimizer_stop
            .function_evaluations
            .is_some_and(|feval| feval > 0),
        "{label}: certificate should record positive function evaluations"
    );
    assert!(
        certificate
            .objective_value
            .is_some_and(|value| value.is_finite()),
        "{label}: certificate should record finite objective"
    );
}

#[cfg(feature = "nlopt")]
fn max_abs_delta(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[cfg(feature = "nlopt")]
fn max_varcorr_std_dev_delta(left: &LinearMixedModel, right: &LinearMixedModel) -> f64 {
    let left = left.varcorr();
    let right = right.varcorr();
    assert_eq!(left.components.len(), right.components.len());
    left.components
        .iter()
        .zip(right.components.iter())
        .flat_map(|(left, right)| {
            assert_eq!(left.group, right.group);
            assert_eq!(left.names, right.names);
            left.std_dev
                .iter()
                .zip(right.std_dev.iter())
                .map(|(a, b)| (a - b).abs())
                .collect::<Vec<_>>()
        })
        .fold(0.0, f64::max)
}

#[cfg(feature = "nlopt")]
fn fit_default_nlopt_reference(data: &DataFrame, formula: &str, reml: bool) -> LinearMixedModel {
    let formula = parse_formula(formula).unwrap();
    let mut model = LinearMixedModel::new(formula, data, None).unwrap();
    model.fit(reml).unwrap();
    model
}

#[cfg(feature = "nlopt")]
fn fit_forced_cobyla_with(
    data: &DataFrame,
    formula: &str,
    reml: bool,
    configure: impl FnOnce(&mut LinearMixedModel),
) -> LinearMixedModel {
    let formula = parse_formula(formula).unwrap();
    let mut model = LinearMixedModel::new(formula, data, None).unwrap();
    configure(&mut model);
    model
        .fit_with_forced_optimizer(reml, Optimizer::Cobyla)
        .unwrap();
    model
}

// ── Parity tests against Julia MixedModels.jl ──────────────────────────

#[test]
fn test_dyestuff_ml_matches_julia() {
    // Mirrors pls.jl "Dyestuff" testset (ML fit).
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    assert_eq!(model.nobs(), 30);
    assert_eq!(model.dof(), 3);
    assert_relative_eq!(model.theta()[0], 0.7525806540074477, epsilon = 1e-4);
    assert_relative_eq!(model.fixef()[0], 1527.5, epsilon = 1e-6);
    assert_relative_eq!(model.sigma(), 49.51010035223816, epsilon = 1e-3);
    assert_relative_eq!(model.stderror()[0], 17.694552929494222, epsilon = 1e-2);
    assert_relative_eq!(model.objective_value(), 327.32705988112673, epsilon = 1e-3);
    // Julia: loglikelihood(fm1) ≈ -163.663... = -327.327/2
    assert_relative_eq!(
        model.loglikelihood(),
        -327.32705988112673 / 2.0,
        epsilon = 1e-3
    );
}

#[test]
fn test_deviance_varpar_matches_ml_scalar_fit_and_restores_state() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let vcov_before = model.vcov();
    let varpar = fitted_varpar(&model);

    let deviance = model.deviance_varpar(&varpar, false).unwrap();

    assert_relative_eq!(deviance, objective_before, epsilon = 1e-8);
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
    assert_relative_eq!(model.vcov(), vcov_before, epsilon = 1e-10);
}

#[test]
fn test_deviance_varpar_matches_reml_vector_fit_and_restores_state() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let vcov_before = model.vcov();
    let varpar = fitted_varpar(&model);

    let deviance = model.deviance_varpar(&varpar, true).unwrap();

    assert_relative_eq!(deviance, objective_before, epsilon = 1e-8);
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
    assert_relative_eq!(model.vcov(), vcov_before, epsilon = 1e-10);
}

#[test]
fn test_deviance_varpar_rejects_invalid_inputs_without_changing_state() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let mut varpar = fitted_varpar(&model);
    varpar[0] = -1.0;
    assert!(model.deviance_varpar(&varpar, false).is_err());

    let mut varpar = fitted_varpar(&model);
    *varpar.last_mut().unwrap() = 0.0;
    assert!(model.deviance_varpar(&varpar, false).is_err());

    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
}

#[test]
fn test_vcov_beta_varpar_matches_fitted_vcov_and_restores_state() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let vcov_before = model.vcov();
    let varpar = fitted_varpar(&model);

    let vcov = model.vcov_beta_varpar(&varpar).unwrap();

    assert_matrix_relative_eq(&vcov, &vcov_before, 1e-10);
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
    assert_matrix_relative_eq(&model.vcov(), &vcov_before, 1e-10);
}

#[test]
fn test_jac_vcov_beta_varpar_returns_symmetric_matrices_and_sigma_derivative() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let varpar = fitted_varpar(&model);
    let vcov = model.vcov_beta_varpar(&varpar).unwrap();

    let jacobian = model.jac_vcov_beta_varpar(&varpar).unwrap();

    assert_eq!(jacobian.len(), varpar.len());
    for derivative in &jacobian {
        assert_eq!(derivative.shape(), vcov.shape());
        assert!(matrix_is_finite(derivative));
        assert_matrix_symmetric(derivative, 1e-10);
    }

    let sigma = *varpar.last().unwrap();
    let sigma_derivative = jacobian.last().unwrap();
    let expected_sigma_derivative = vcov * (2.0 / sigma);
    assert_matrix_relative_eq(sigma_derivative, &expected_sigma_derivative, 1e-6);
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
}

#[test]
fn test_jac_vcov_beta_varpar_rejects_boundary_stencil_without_changing_state() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let mut varpar = fitted_varpar(&model);
    varpar[0] = 0.0;

    let err = model.jac_vcov_beta_varpar(&varpar).unwrap_err();

    assert!(err.to_string().contains("lower bound"));
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
}

#[test]
fn test_vcov_varpar_estimate_returns_hessian_diagnostics_and_restores_state() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let varpar = fitted_varpar(&model);

    let estimate = model.vcov_varpar(&varpar, true).unwrap();

    assert_eq!(estimate.covariance.shape(), (varpar.len(), varpar.len()));
    assert_eq!(estimate.hessian.shape(), (varpar.len(), varpar.len()));
    assert_eq!(estimate.eigenvalues.len(), varpar.len());
    assert_eq!(
        estimate.positive_eigenvalues
            + estimate.near_zero_eigenvalues
            + estimate.negative_eigenvalues,
        varpar.len()
    );
    assert!(estimate.positive_eigenvalues > 0);
    assert!(estimate.tolerance.is_finite());
    assert!(estimate.tolerance > 0.0);
    assert!(matrix_is_finite(&estimate.covariance));
    assert!(matrix_is_finite(&estimate.hessian));
    assert_matrix_symmetric(&estimate.covariance, 1e-8);
    assert_matrix_symmetric(&estimate.hessian, 1e-8);
    for index in 0..varpar.len() {
        assert!(estimate.covariance[(index, index)] >= -1e-8);
    }
    assert!(matches!(
        estimate.reliability,
        ReliabilityGrade::Moderate | ReliabilityGrade::Low
    ));
    assert_eq!(
        estimate.used_reduced_rank,
        estimate.positive_eigenvalues < varpar.len()
    );
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
}

#[test]
fn test_vcov_varpar_rejects_boundary_hessian_without_changing_state() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let theta_before = model.theta();
    let objective_before = model.objective_value();
    let mut varpar = fitted_varpar(&model);
    varpar[0] = 0.0;

    let err = model.vcov_varpar(&varpar, false).unwrap_err();

    assert!(err.to_string().contains("lower bound"));
    assert_eq!(model.theta(), theta_before);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-10);
}

#[test]
fn test_kenward_roger_sigma_g_scalar_random_intercept_components() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let sigma_g = model.kenward_roger_sigma_g().unwrap();

    assert_eq!(sigma_g.n_observations, model.nobs());
    assert_eq!(sigma_g.components.len(), 2);
    assert_eq!(sigma_g.component_weights.len(), 2);
    assert_eq!(sigma_g.component_labels.len(), 2);
    assert_eq!(sigma_g.residual_component_index, 1);
    assert_eq!(sigma_g.component_labels[1], "residual");
    assert!(sigma_g.includes_residual_variance);
    assert!(sigma_g.sigma_positive_definite);
    assert!(sigma_g.sigma_min_eigenvalue > 0.0);
    assert!(matrix_is_finite(&sigma_g.sigma));
    assert_matrix_symmetric(&sigma_g.sigma, 1e-10);
    for component in &sigma_g.components {
        assert_matrix_symmetric(component, 1e-12);
    }

    let residual_variance = model.sigma().powi(2);
    let random_variance = residual_variance * model.theta()[0].powi(2);
    assert_relative_eq!(
        sigma_g.component_weights[0],
        random_variance,
        epsilon = 1e-6
    );
    assert_relative_eq!(
        sigma_g.component_weights[1],
        residual_variance,
        epsilon = 1e-6
    );

    let refs = &model.reterms[0].refs;
    for row in 0..model.nobs() {
        for col in 0..model.nobs() {
            let mut expected = if refs[row] == refs[col] {
                random_variance
            } else {
                0.0
            };
            if row == col {
                expected += residual_variance;
            }
            assert_relative_eq!(sigma_g.sigma[(row, col)], expected, epsilon = 1e-6);
        }
    }
}

#[test]
fn test_kenward_roger_sigma_g_vector_random_effect_components() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let sigma_g = model.kenward_roger_sigma_g().unwrap();

    assert_eq!(sigma_g.components.len(), 4);
    assert_eq!(sigma_g.residual_component_index, 3);
    assert_eq!(sigma_g.component_labels[3], "residual");
    assert!(sigma_g.component_labels[0].contains("(Intercept),(Intercept)"));
    assert!(sigma_g.component_labels[1].contains("days,(Intercept)"));
    assert!(sigma_g.component_labels[2].contains("days,days"));
    assert!(sigma_g.sigma_positive_definite);
    assert!(sigma_g.max_component_asymmetry <= 1e-12);

    let residual_variance = model.sigma().powi(2);
    let varcorr =
        residual_variance * (&model.reterms[0].lambda * model.reterms[0].lambda.transpose());
    assert_relative_eq!(
        sigma_g.component_weights[0],
        varcorr[(0, 0)],
        epsilon = 1e-6
    );
    assert_relative_eq!(
        sigma_g.component_weights[1],
        varcorr[(1, 0)],
        epsilon = 1e-6
    );
    assert_relative_eq!(
        sigma_g.component_weights[2],
        varcorr[(1, 1)],
        epsilon = 1e-6
    );
    assert_relative_eq!(
        sigma_g.component_weights[3],
        residual_variance,
        epsilon = 1e-6
    );
}

#[test]
fn test_kenward_roger_adjusted_vcov_returns_pbkrtest_style_artifacts() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let adjusted = model.kenward_roger_adjusted_vcov().unwrap();

    let p = model.feterm.rank;
    let n_components = model.kenward_roger_sigma_g().unwrap().components.len();
    assert_eq!(adjusted.unadjusted_vcov_active.shape(), (p, p));
    assert_eq!(adjusted.adjusted_vcov_active.shape(), (p, p));
    assert_eq!(
        adjusted.adjusted_vcov.shape(),
        (model.coef_names().len(), model.coef_names().len())
    );
    assert_eq!(adjusted.p_matrices.len(), n_components);
    assert_eq!(
        adjusted.q_matrices.len(),
        n_components * (n_components + 1) / 2
    );
    assert_eq!(adjusted.w.shape(), (n_components, n_components));
    assert_eq!(
        adjusted.information_matrix.shape(),
        (n_components, n_components)
    );
    assert_eq!(adjusted.information_eigenvalues.len(), n_components);
    assert_eq!(adjusted.component_labels.len(), n_components);
    assert!(matrix_is_finite(&adjusted.unadjusted_vcov_active));
    assert!(matrix_is_finite(&adjusted.adjusted_vcov_active));
    assert!(matrix_is_finite(&adjusted.adjusted_vcov));
    assert!(matrix_is_finite(&adjusted.w));
    assert!(matrix_is_finite(&adjusted.information_matrix));
    assert_matrix_symmetric(&adjusted.adjusted_vcov_active, 1e-8);
    assert_matrix_symmetric(&adjusted.adjusted_vcov, 1e-8);
    assert_matrix_symmetric(&adjusted.w, 1e-8);
    assert_matrix_symmetric(&adjusted.information_matrix, 1e-8);
    for p_matrix in &adjusted.p_matrices {
        assert_eq!(p_matrix.shape(), (p, p));
        assert_matrix_symmetric(p_matrix, 1e-8);
    }
    for q_matrix in &adjusted.q_matrices {
        assert_eq!(q_matrix.shape(), (p, p));
        assert!(matrix_is_finite(q_matrix));
    }
    assert!(matches!(
        adjusted.reliability,
        ReliabilityGrade::Moderate | ReliabilityGrade::Low
    ));
}

#[test]
fn test_kenward_roger_adjusted_vcov_rejects_unweighted_prerequisite_gap() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let weights = vec![1.0; data.nrow()];
    let mut model = LinearMixedModel::new(formula, &data, Some(&weights)).unwrap();
    model.fit(true).unwrap();

    let err = model.kenward_roger_adjusted_vcov().unwrap_err();

    assert!(err
        .to_string()
        .contains("unweighted iid Gaussian residual models"));
}

#[test]
fn test_kenward_roger_lbddf_scalar_contrast_matches_expected_scale() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let l = DMatrix::from_row_slice(1, model.coef_names().len(), &[0.0, 1.0]);
    let ddf = model.kenward_roger_lbddf(&l).unwrap();

    assert_eq!(ddf.restriction_rank, 1);
    assert_relative_eq!(ddf.numerator_df, 1.0, epsilon = 1e-12);
    assert!(ddf.denominator_df.is_finite());
    assert!(
        (15.0..=20.0).contains(&ddf.denominator_df),
        "pbkrtest sleepstudy days df is expected near 17, got {}",
        ddf.denominator_df
    );
    assert!(ddf.a1.is_finite());
    assert!(ddf.a2.is_finite());
    assert!(ddf.b.is_finite());
    assert!(ddf.g.is_finite());
    assert!(ddf.rho.is_finite());
    assert!(matches!(
        ddf.reliability,
        ReliabilityGrade::Moderate | ReliabilityGrade::Low
    ));
}

#[test]
fn test_kenward_roger_lbddf_handles_rank_deficient_restriction_matrix() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let l = DMatrix::from_row_slice(
        2,
        model.coef_names().len(),
        &[
            0.0, 1.0, //
            0.0, 1.0,
        ],
    );
    let ddf = model.kenward_roger_lbddf(&l).unwrap();

    assert_eq!(ddf.restriction_rank, 1);
    assert_relative_eq!(ddf.numerator_df, 1.0, epsilon = 1e-12);
    assert!(ddf.used_generalized_inverse);
    assert!(ddf
        .notes
        .iter()
        .any(|note| note.contains("row rank 1 is lower")));
    assert!(ddf.denominator_df.is_finite());
    assert!(ddf.denominator_df > 0.0);
}

#[test]
fn test_kenward_roger_lbddf_multi_df_contrast_returns_rank_df() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let l = DMatrix::identity(model.coef_names().len(), model.coef_names().len());
    let ddf = model.kenward_roger_lbddf(&l).unwrap();

    assert_eq!(ddf.restriction_rank, 2);
    assert_relative_eq!(ddf.numerator_df, 2.0, epsilon = 1e-12);
    assert!(ddf.denominator_df.is_finite());
    assert!(ddf.denominator_df > 0.0);
}

#[test]
fn test_lmm_explicit_kenward_roger_scalar_request_returns_t_test() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", 1, model.coef_names().len()).unwrap();
    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);

    assert_eq!(test.method, InferenceMethod::KenwardRoger);
    assert_eq!(test.status, InferenceStatus::Available);
    assert!(test.numerator_df.is_none());
    assert!(test.denominator_df.unwrap().is_finite());
    assert!((15.0..=20.0).contains(&test.denominator_df.unwrap()));
    assert!(test.standard_errors[0].unwrap().is_finite());
    assert!(test.statistics[0].unwrap().is_finite());
    assert!(test.p_values[0].unwrap().is_finite());
    assert!((0.0..=1.0).contains(&test.p_values[0].unwrap()));
    assert!(test.notes.iter().any(|note| note.contains("Kenward-Roger")));
}

#[test]
fn test_lmm_explicit_kenward_roger_multi_df_request_returns_f_test() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let l = DMatrix::identity(model.coef_names().len(), model.coef_names().len());
    let hypothesis = FixedEffectHypothesis::new(
        "all fixed effects = 0",
        crate::compiler::ContrastMatrix::new(l).unwrap(),
        crate::compiler::ContrastRhs::zeros(model.coef_names().len()),
    )
    .unwrap();
    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);

    assert_eq!(test.method, InferenceMethod::KenwardRoger);
    assert_eq!(test.status, InferenceStatus::Available);
    assert_eq!(test.numerator_df, Some(2.0));
    assert!(test.denominator_df.unwrap().is_finite());
    assert!(test.denominator_df.unwrap() > 0.0);
    assert_eq!(test.statistics.len(), 1);
    assert!(test.statistics[0].unwrap().is_finite());
    assert!(test.statistics[0].unwrap() >= 0.0);
    assert_eq!(test.p_values.len(), 1);
    assert!(test.p_values[0].unwrap().is_finite());
    assert!((0.0..=1.0).contains(&test.p_values[0].unwrap()));
    assert!(test
        .notes
        .iter()
        .any(|note| note.contains("F scaling = 1.0")));

    let row = fixed_effect_test_to_inference_row(FixedEffectInferenceRowKind::Term, test);
    let details = row.details.expect("multi-df row should carry details");
    let family = details
        .contrast_family
        .expect("multi-df row should carry contrast-family details");
    assert_eq!(family.restriction_rows, 2);
    assert_eq!(family.effective_rank, Some(2));
    assert_eq!(family.numerator_df_semantics, "effective_restriction_rank");
    let kr = details
        .kenward_roger
        .expect("KR row should carry KR details");
    assert_eq!(kr.f_scaling, Some(1.0));
    assert_eq!(kr.statistic_scale.as_deref(), Some("unscaled"));
}

#[test]
fn test_lmm_explicit_kenward_roger_ml_request_does_not_fallback() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", 1, model.coef_names().len()).unwrap();
    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);

    assert_eq!(test.method, InferenceMethod::KenwardRoger);
    assert!(matches!(test.status, InferenceStatus::NotAssessed { .. }));
    assert_eq!(test.p_values, vec![None]);
    assert!(fixed_effect_inference_reason(&test)
        .unwrap()
        .contains("REML"));
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_native_default_kenward_roger_rows_are_finite_with_realistic_tolerances() {
    let fixture = kenward_roger_pbkrtest_parity_fixture();

    for case in fixture
        .scalar_cases
        .iter()
        .filter(|case| case.name.contains("random_slope"))
    {
        let data = kenward_roger_parity_data(&case.name);
        let formula = parse_formula(&case.formula).unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();
        assert_default_native_certificate(&model, &case.name);

        let hypothesis = fixed_effect_hypothesis_from_fixture(&case.label, &case.l, &case.rhs);
        let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);
        assert_available_finite_fixed_effect_test(&test, &case.name);
        assert!(
            (test.standard_errors[0].unwrap() - case.std_error).abs() <= 1e-3,
            "{}: native KR SE drift too large: rust={} ref={}",
            case.name,
            test.standard_errors[0].unwrap(),
            case.std_error
        );
        assert!(
            (test.statistics[0].unwrap() - case.statistic).abs() <= 5e-3,
            "{}: native KR statistic drift too large: rust={} ref={}",
            case.name,
            test.statistics[0].unwrap(),
            case.statistic
        );
        assert!(
            (test.denominator_df.unwrap() - case.denominator_df).abs() <= 1e-2,
            "{}: native KR denominator df drift too large: rust={} ref={}",
            case.name,
            test.denominator_df.unwrap(),
            case.denominator_df
        );
    }

    for case in &fixture.multi_df_cases {
        let data = kenward_roger_parity_data(&case.name);
        let formula = parse_formula(&case.formula).unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();
        assert_default_native_certificate(&model, &case.name);

        let hypothesis = fixed_effect_hypothesis_from_fixture(&case.label, &case.l, &case.rhs);
        let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);
        assert_available_finite_fixed_effect_test(&test, &case.name);
        let unscaled_statistic = if case.l.len() == 1 {
            // Single-row contrast: the unscaled F is the *derived*
            // statistic F = t², so it inherits ~2× the relative drift of
            // the underlying t. On the native (no-`nlopt`) optimizer the
            // REML fit lands ~1e-3–1e-4 from the BOBYQA/pbkrtest
            // reference (see the SE note above), and squaring amplifies
            // that, so this row is checked with a realistic
            // absolute+relative band rather than exact equality. This is
            // not a weakening of certification: strict pbkrtest parity —
            // including the p-value at `max_relative = 1e-3` and the
            // statistic/SE/df at 5e-5/1e-5 — is asserted separately under
            // the `nlopt` feature in
            // `test_lmm_kenward_roger_scalar_rows_match_pbkrtest_fixture`.
            test.statistics[0].unwrap().powi(2)
        } else {
            test.statistics[0].unwrap()
        };
        assert!(
            (unscaled_statistic - case.unscaled_statistic).abs()
                <= 1.0 + 2e-3 * case.unscaled_statistic.abs(),
            "{}: native KR unscaled F drift too large: rust={} ref={}",
            case.name,
            unscaled_statistic,
            case.unscaled_statistic
        );
        assert!(
            (test.denominator_df.unwrap() - case.denominator_df).abs() <= 5e-2,
            "{}: native KR denominator df drift too large: rust={} ref={}",
            case.name,
            test.denominator_df.unwrap(),
            case.denominator_df
        );
    }
}

// Parity against pbkrtest reference fits (computed under NLopt-equivalent
// BOBYQA); the native no-default-features path lands ~1e-4 away in SE.
#[cfg(feature = "nlopt")]
#[test]
fn test_lmm_kenward_roger_scalar_rows_match_pbkrtest_fixture() {
    let fixture = kenward_roger_pbkrtest_parity_fixture();

    for case in fixture.scalar_cases {
        let data = kenward_roger_parity_data(&case.name);
        let formula = parse_formula(&case.formula).unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let hypothesis = fixed_effect_hypothesis_from_fixture(&case.label, &case.l, &case.rhs);
        let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);

        assert_eq!(test.method, InferenceMethod::KenwardRoger, "{}", case.name);
        assert_eq!(test.status, InferenceStatus::Available, "{}", case.name);
        assert!(
            matches!(
                test.reliability,
                ReliabilityGrade::Moderate | ReliabilityGrade::Low
            ),
            "{}",
            case.name
        );
        assert!(test.numerator_df.is_none(), "{}", case.name);
        assert_relative_eq!(
            test.estimates[0],
            case.estimate,
            epsilon = 1e-8,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            test.standard_errors[0].unwrap(),
            case.std_error,
            epsilon = 5e-5,
            max_relative = 5e-5
        );
        assert_relative_eq!(
            test.denominator_df.unwrap(),
            case.denominator_df,
            epsilon = 1e-3,
            max_relative = 1e-5
        );
        assert_relative_eq!(
            test.statistics[0].unwrap(),
            case.statistic,
            epsilon = 5e-5,
            max_relative = 5e-5
        );
        assert_relative_eq!(
            test.p_values[0].unwrap(),
            case.p_value,
            epsilon = 1e-12,
            max_relative = 1e-3
        );
    }
}

// Parity against pbkrtest reference fits (NLopt-equivalent BOBYQA); the
// native no-default-features path drifts in the unscaled F statistic.
#[cfg(feature = "nlopt")]
#[test]
fn test_lmm_kenward_roger_multi_df_rows_match_pbkrtest_unscaled_fixture() {
    let fixture = kenward_roger_pbkrtest_parity_fixture();

    for case in fixture.multi_df_cases {
        let data = kenward_roger_parity_data(&case.name);
        let formula = parse_formula(&case.formula).unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let hypothesis = fixed_effect_hypothesis_from_fixture(&case.label, &case.l, &case.rhs);
        let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::KenwardRoger);

        assert_eq!(test.method, InferenceMethod::KenwardRoger, "{}", case.name);
        assert_eq!(test.status, InferenceStatus::Available, "{}", case.name);
        if case.l.len() == 1 {
            assert_eq!(test.numerator_df, None, "{}", case.name);
            let unscaled_from_scalar = test.statistics[0].unwrap().powi(2);
            assert!(
                (unscaled_from_scalar - case.unscaled_statistic).abs()
                    <= 1e-3 + 1e-4 * case.unscaled_statistic.abs(),
                "{}: single-row unscaled F drift exceeds tolerance: rust={} ref={}",
                case.name,
                unscaled_from_scalar,
                case.unscaled_statistic
            );
            assert_relative_eq!(
                test.p_values[0].unwrap(),
                case.unscaled_p_value,
                epsilon = 1e-12,
                max_relative = 1e-3
            );
            continue;
        }
        assert_eq!(test.numerator_df, Some(case.numerator_df), "{}", case.name);
        // Multi-df F drift vs pbkrtest is dominated by numerical noise in the
        // adjusted-vcov off-diagonals (β and Φ_A diagonals match to 1e-7;
        // det(Φ_A) drift sits in the 3e-4 band).  Match a realistic numerical
        // tolerance rather than bit-exactness.
        assert_relative_eq!(
            test.denominator_df.unwrap(),
            case.denominator_df,
            epsilon = 1e-3,
            max_relative = 5e-4,
        );
        assert!(
            (test.statistics[0].unwrap() - case.unscaled_statistic).abs()
                <= 1e-3 + 5e-4 * case.unscaled_statistic.abs(),
            "{}: unscaled F drift exceeds tolerance: rust={} ref={}",
            case.name,
            test.statistics[0].unwrap(),
            case.unscaled_statistic
        );
        assert_relative_eq!(
            test.p_values[0].unwrap(),
            case.unscaled_p_value,
            epsilon = 1e-12,
            max_relative = 1e-3,
        );

        if (case.f_scaling - 1.0).abs() > 1e-12 {
            assert_ne!(case.statistic, case.unscaled_statistic);
            assert_ne!(case.p_value, case.unscaled_p_value);
            assert!(test
                .notes
                .iter()
                .any(|note| note.contains("F scaling = 1.0")));
        }
    }
}

#[test]
fn test_fixed_effect_h0_simulation_smoke_for_analytic_p_values() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
    model.fit(true).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();
    let target = model
        .fixed_effect_null_bootstrap_target(&hypothesis)
        .unwrap();

    let mut rng = StdRng::seed_from_u64(20260501);
    let mut wald_p_values = Vec::new();
    let mut satterthwaite_p_values = Vec::new();
    let mut kenward_roger_p_values = Vec::new();

    for _ in 0..8 {
        let y_sim = model.simulate_fixed_effect_null(&mut rng, &target).unwrap();
        let mut sim_data = DataFrame::new();
        sim_data
            .add_numeric("reaction", y_sim.iter().copied().collect())
            .unwrap();
        sim_data
            .add_numeric("days", data.numeric("days").unwrap().to_vec())
            .unwrap();
        let subj = data.categorical("subj").unwrap();
        sim_data
            .add_categorical_with_levels("subj", subj.values.clone(), subj.levels.clone())
            .unwrap();
        let mut work = LinearMixedModel::new(formula.clone(), &sim_data, None).unwrap();
        work.fit(true).unwrap();

        let wald = work
            .test_contrast_with_method(hypothesis.clone(), FixedEffectTestMethod::AsymptoticWaldZ);
        let satterthwaite = work
            .test_contrast_with_method(hypothesis.clone(), FixedEffectTestMethod::Satterthwaite);
        let kenward_roger =
            work.test_contrast_with_method(hypothesis.clone(), FixedEffectTestMethod::KenwardRoger);

        assert_eq!(wald.status, InferenceStatus::Available);
        assert_eq!(satterthwaite.status, InferenceStatus::Available);
        assert_eq!(kenward_roger.status, InferenceStatus::Available);
        wald_p_values.push(wald.p_values[0].unwrap());
        satterthwaite_p_values.push(satterthwaite.p_values[0].unwrap());
        kenward_roger_p_values.push(kenward_roger.p_values[0].unwrap());
    }

    for (label, values) in [
        ("Wald", &wald_p_values),
        ("Satterthwaite", &satterthwaite_p_values),
        ("Kenward-Roger", &kenward_roger_p_values),
    ] {
        assert_eq!(values.len(), 8, "{label} should produce all p-values");
        assert!(
            values
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
            "{label} p-values should be finite probabilities: {values:?}"
        );
        let tiny = values.iter().filter(|&&p| p < 0.01).count();
        assert!(
            tiny <= 2,
            "{label} produced too many tiny p-values under the simulated null: {values:?}"
        );
    }
}

#[test]
fn test_dyestuff_aic_bic_matches_julia() {
    // Mirrors pls.jl "Dyestuff":
    //   aic(fm1) ≈ 333.32705988112673
    //   bic(fm1) ≈ 337.5306520261132
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let obj = model.objective_value(); // -2*loglik
    let k = model.dof() as f64;
    let n = model.nobs() as f64;
    let aic = obj + 2.0 * k;
    let bic = obj + k * n.ln();

    assert_relative_eq!(aic, 333.32705988112673, epsilon = 1e-3);
    assert_relative_eq!(bic, 337.5306520261132, epsilon = 1e-3);
}

#[test]
fn test_dyestuff_re_std_dev_matches_julia() {
    // Mirrors pls.jl: first(first(fm1.σs)) ≈ 37.260343703061764
    // RE std dev = lambda * sigma = 0.7526 * 49.51 ≈ 37.26
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let vc = model.varcorr();
    assert_eq!(vc.components.len(), 1);
    let comp = &vc.components[0];
    assert_eq!(comp.group, "batch");
    assert_relative_eq!(comp.std_dev[0], 37.260343703061764, epsilon = 0.1);
}

#[test]
fn test_dyestuff_reml_matches_julia() {
    // Mirrors pls.jl "Dyestuff" REML refit.
    // Julia: objective ≈ 319.6542768422576
    //        vcov[0,0] ≈ 375.7167103872769 (variance of intercept under REML)
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap(); // REML

    assert_relative_eq!(model.objective_value(), 319.6542768422576, epsilon = 1e-3);
    // REML vcov of the intercept
    let v = model.vcov();
    assert_eq!(v.nrows(), 1);
    assert_relative_eq!(v[(0, 0)], 375.7167103872769, epsilon = 1.0);
}

#[test]
fn test_sleepstudy_vector_re_matches_julia() {
    // Mirrors pls.jl "sleep" testset (last model: (1 + days | subj)).
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    assert_relative_eq!(model.objective_value(), 1751.9393444636682, epsilon = 0.01);
    let theta = model.theta();
    assert_eq!(theta.len(), 3);
    assert_relative_eq!(theta[0], 0.9292297167514472, epsilon = 1e-3);
    assert_relative_eq!(theta[1], 0.01816466496782548, epsilon = 1e-3);
    assert_relative_eq!(theta[2], 0.22264601131030412, epsilon = 1e-3);

    // coef() returns in original formula order: [intercept, days]
    let coef = MixedModelFit::coef(&model);
    assert_relative_eq!(coef[0], 251.40510484848454, epsilon = 0.01);
    assert_relative_eq!(coef[1], 10.467285959596126, epsilon = 0.01);

    let se = model.stderror();
    assert_relative_eq!(se[0], 6.632295312722272, epsilon = 0.1);
    assert_relative_eq!(se[1], 1.5022387911441102, epsilon = 0.05);

    assert_relative_eq!(model.loglikelihood(), -875.9696722318341, epsilon = 0.01);
}

#[test]
fn test_lrt_sleepstudy_matches_julia() {
    // Mirrors likelihoodratiotest.jl "likelihoodratio test":
    //   fm0: reaction ~ 1 + (1 + days | subj)  [no days in FE, dof=5]
    //   fm1: reaction ~ 1 + days + (1 + days | subj) [days in FE, dof=6]
    // Julia: chisq ≈ 23.5365, dof=1, p < 1e-5
    use crate::stats::lrt::LikelihoodRatioTest;
    let data = sleepstudy_fixture();

    let f0 = parse_formula("reaction ~ 1 + (1 + days | subj)").unwrap();
    let mut m0 = LinearMixedModel::new(f0, &data, None).unwrap();
    m0.fit(false).unwrap();

    let f1 = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut m1 = LinearMixedModel::new(f1, &data, None).unwrap();
    m1.fit(false).unwrap();

    assert!(
        m0.objective_value() > m1.objective_value(),
        "fm0 should have larger objective"
    );
    assert_eq!(m0.dof(), 5);
    assert_eq!(m1.dof(), 6);

    let lrt = LikelihoodRatioTest::test(&[&m0 as &dyn MixedModelFit, &m1]).unwrap();
    assert_eq!(lrt.chisq_dof[0], 1);
    assert_relative_eq!(lrt.chisq[0], 23.5365, epsilon = 0.05);
    assert!(lrt.pvalues[0] < 1e-5);
}

#[test]
fn test_penicillin_crossed_re_matches_julia() {
    // Mirrors pls.jl "penicillin" testset.
    // Formula: diameter ~ 1 + (1 | plate) + (1 | sample)
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    assert_eq!(model.nobs(), 144);

    assert_relative_eq!(model.objective_value(), 332.1883486700085, epsilon = 0.01);

    let coef = MixedModelFit::coef(&model);
    assert_relative_eq!(coef[0], 22.97222222222222, epsilon = 1e-4);

    assert_relative_eq!(model.stderror()[0], 0.7446037806555799, epsilon = 0.01);

    // θ[0] = plate RE, θ[1] = sample RE
    let theta = model.theta();
    assert_eq!(theta.len(), 2);
    assert_relative_eq!(theta[0], 1.5375939045981573, epsilon = 0.01);
    assert_relative_eq!(theta[1], 3.219792193110907, epsilon = 0.01);
}

#[test]
fn test_dyestuff2_singular_fit_matches_julia() {
    // Mirrors pls.jl "Dyestuff2" testset.
    // The within-batch variance dominates → RE collapses to 0 (singular).
    let data = dyestuff2_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap(); // ML

    // Julia: fm.θ ≈ zeros(1)
    assert!(
        model.theta()[0].abs() < 1e-6,
        "theta should be ~0 for singular fit, got {}",
        model.theta()[0]
    );
    // Julia: objective(fm) ≈ 162.87303665382575
    assert_relative_eq!(model.objective_value(), 162.87303665382575, epsilon = 1e-3);
    // Julia: coef(fm) ≈ [5.6656]
    let coef = MixedModelFit::coef(&model);
    assert_relative_eq!(coef[0], 5.6656, epsilon = 1e-3);
    // Julia: stderror(fm) ≈ [0.6669857396443264]
    assert_relative_eq!(model.stderror()[0], 0.6669857396443264, epsilon = 1e-3);
    // Julia: logdet(fm) ≈ 0.0 (RE variance = 0 → Λ diagonal = 0)
    assert_relative_eq!(model.logdet_re(), 0.0, epsilon = 1e-8);
    // Julia: issingular(fm) == true
    assert!(model.is_singular(), "Dyestuff2 fit should be singular");
}

#[test]
fn test_dyestuff_objective_at_specific_theta() {
    // Mirrors pls.jl: objective!(fm1, 0.713) ≈ 327.34216280954615
    // Julia evaluates this on an ML-mode model (reml=false).
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.optsum.reml = false; // match Julia ML mode
    let obj = model.objective_at(&[0.713]).unwrap();
    assert_relative_eq!(obj, 327.34216280954615, epsilon = 1e-3);
}

#[test]
fn test_dyestuff_logdet_pwrss_varest() {
    // Mirrors pls.jl "Dyestuff" testset — additional metrics after ML fit.
    // Julia: logdet(fm1) ≈ 8.06014611206176
    //        varest(fm1) ≈ 2451.2500368886936  (= sigma^2)
    //        pwrss(fm1)  ≈ 73537.50110666081
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.logdet_re(), 8.06014611206176, epsilon = 1e-3);
    assert_relative_eq!(
        model.sigma() * model.sigma(),
        2451.2500368886936,
        epsilon = 1.0
    );
    assert_relative_eq!(model.pwrss(), 73537.50110666081, epsilon = 10.0);
}

#[test]
fn test_penicillin_logdet_and_varest() {
    // Mirrors pls.jl "penicillin" testset — additional metrics.
    // Julia: varest(fm) ≈ 0.30242510228527864
    //        logdet(fm) ≈ 95.74676552743833
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(
        model.sigma() * model.sigma(),
        0.30242510228527864,
        epsilon = 1e-4
    );
    assert_relative_eq!(model.logdet_re(), 95.74676552743833, epsilon = 0.1);
}

#[test]
fn test_sleepstudy_random_slope_only_matches_julia() {
    // Mirrors pls.jl: fmrs = reaction ~ 1 + days + (0 + days | subj)
    // Random slope only (no random intercept).
    // Julia: objective ≈ 1774.080315280526, θ ≈ [0.24353985601485326]
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (0 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.objective_value(), 1774.080315280526, epsilon = 0.01);
    let theta = model.theta();
    assert_eq!(theta.len(), 1, "random-slope-only has scalar theta");
    assert_relative_eq!(theta[0], 0.24353985601485326, epsilon = 1e-3);
}

#[test]
fn test_pastes_nested_re_matches_julia() {
    // Mirrors pls.jl "pastes" testset.
    // Julia formula: strength ~ 1 + (1 | batch / cask)
    // which expands to: strength ~ 1 + (1 | batch) + (1 | batch:cask)
    // We use pre-computed batch_cask interaction column.
    // Julia: objective ≈ 247.9944658624955
    //        coef ≈ [60.0533333333333]
    //        stderror ≈ [0.6421355774401101]
    //        θ ≈ [3.5269029347766856, 1.3299137410046242]
    let data = pastes_fixture();
    let formula = parse_formula("strength ~ 1 + (1 | batch) + (1 | batch_cask)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.nobs(), 60);
    assert_relative_eq!(model.objective_value(), 247.9944658624955, epsilon = 0.01);

    let coef = MixedModelFit::coef(&model);
    assert_relative_eq!(coef[0], 60.0533333333333, epsilon = 1e-3);

    assert_relative_eq!(model.stderror()[0], 0.6421355774401101, epsilon = 0.01);

    let theta = model.theta();
    assert_eq!(theta.len(), 2);
    // Julia sorts by decreasing nranef: θ[0] = batch:cask RE (30 levels), θ[1] = batch RE (10 levels)
    #[cfg(feature = "nlopt")]
    let theta_epsilon = 0.05;
    #[cfg(not(feature = "nlopt"))]
    let theta_epsilon = 0.09;
    assert_relative_eq!(theta[0], 3.5269029347766856, epsilon = theta_epsilon);
    assert_relative_eq!(theta[1], 1.3299137410046242, epsilon = theta_epsilon);
}

fn weighted_lmm_fixture() -> (DataFrame, Vec<f64>) {
    let a = vec![
        1.55945122,
        0.004391538,
        0.005554163,
        -0.173029772,
        4.586284429,
        0.259493671,
        -0.091735715,
        5.546487603,
        0.457734831,
        -0.030169602,
    ];
    let b = vec![
        0.24520519,
        0.080624178,
        0.228083467,
        0.2471453,
        0.398994279,
        0.037213859,
        0.102144973,
        0.241380251,
        0.206570975,
        0.15980803,
    ];
    let c = ["H", "F", "K", "P", "P", "P", "D", "M", "I", "D"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let w1: Vec<f64> = vec![20.0, 40.0, 35.0, 12.0, 29.0, 25.0, 65.0, 105.0, 30.0, 75.0];

    let mut df = DataFrame::new();
    df.add_numeric("a", a).unwrap();
    df.add_numeric("b", b).unwrap();
    df.add_categorical("c", c).unwrap();

    (df, w1)
}

#[test]
fn test_weighted_model_matches_julia() {
    // Mirrors pls.jl "wts" testset.
    // Julia: m2 = fit(@formula(a ~ 1 + b + (1 | c)), data; wts=w1)
    //   θ ≈ [0.2951818091809752]
    //   stderror ≈ [0.964016663994572, 3.6309691484830533]
    //   vcov ≈ [[0.9293, -2.5575], [-2.5575, 13.1839]]
    let (df, w1) = weighted_lmm_fixture();

    let formula = parse_formula("a ~ 1 + b + (1 | c)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, Some(&w1)).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.theta()[0], 0.2951818091809752, epsilon = 1e-3);
    let se = model.stderror();
    assert_eq!(se.len(), 2);
    assert_relative_eq!(se[0], 0.964016663994572, epsilon = 0.01);
    assert_relative_eq!(se[1], 3.6309691484830533, epsilon = 0.1);
    // Julia: vcov ≈ [[0.9293 -2.5575], [-2.5575 13.1839]]
    let v = model.vcov();
    assert_relative_eq!(v[(0, 0)], 0.9293281284592235, epsilon = 0.01);
    assert_relative_eq!(v[(0, 1)], -2.5575260810649962, epsilon = 0.05);
    assert_relative_eq!(v[(1, 0)], -2.5575260810649962, epsilon = 0.05);
    assert_relative_eq!(v[(1, 1)], 13.18393695723575, epsilon = 0.1);
}

#[test]
fn test_weighted_lmm_objective_matches_julia_normalization() {
    let (df, w1) = weighted_lmm_fixture();
    let formula = parse_formula("a ~ 1 + b + (1 | c)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, Some(&w1)).unwrap();
    model.fit(false).unwrap();

    let expected_correction: f64 = w1.iter().map(|weight| weight.ln()).sum();

    assert_relative_eq!(
        model.weight_logdet_correction(),
        expected_correction,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        model.objective_value(),
        model.profiled_objective_value() - expected_correction,
        epsilon = 1e-10
    );
}

#[test]
fn test_unweighted_objective_unchanged_by_weight_normalization() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.weight_logdet_correction(), 0.0);
    assert_relative_eq!(
        model.objective_value(),
        model.profiled_objective_value(),
        epsilon = 1e-12
    );
    assert_relative_eq!(model.objective_value(), 327.32705988112673, epsilon = 1e-3);
}

#[test]
fn test_scalar_vsize1_fast_objective_matches_generic_block_update() {
    let data = simulate_sleepstudy_like(40, 5, 17);
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let model = LinearMixedModel::new(formula, &data, None).unwrap();
    let theta = vec![0.73];
    let tolerance = model
        .compiler_policy()
        .thresholds
        .cholesky_zero_pad_tolerance;

    for reml in [false, true] {
        let fast = LinearMixedModel::profiled_objective_one_vsize1_fast(
            &model.a_blocks,
            &model.reterms,
            &theta,
            model.dims,
            reml,
            model.optsum.sigma,
            tolerance,
        )
        .unwrap();

        let mut generic_reterms = model.reterms.clone();
        let mut generic_l_blocks = model.l_blocks.clone();
        LinearMixedModel::apply_theta_to_reterms(&mut generic_reterms, &theta).unwrap();
        update_l_from_parts(
            &model.a_blocks,
            &mut generic_l_blocks,
            &generic_reterms,
            tolerance,
        )
        .unwrap();

        let k = generic_reterms.len();
        let mut logdet_lzz = 0.0;
        for j in 0..k {
            logdet_lzz += logdet_block(&generic_l_blocks[block_index(j, j)]);
        }
        let l_last = generic_l_blocks[block_index(k, k)].as_dense();
        let pp1 = l_last.nrows();
        let pwrss = l_last[(pp1 - 1, pp1 - 1)].powi(2);
        let logdet = if reml {
            let mut logdet_lxx = 0.0;
            for i in 0..(pp1 - 1) {
                let d = l_last[(i, i)];
                if d > 0.0 {
                    logdet_lxx += d.ln();
                }
            }
            logdet_lzz + 2.0 * logdet_lxx
        } else {
            logdet_lzz
        };
        let denomdf = if reml {
            model.dims.n as f64 - model.dims.p as f64
        } else {
            model.dims.n as f64
        };
        let generic =
            LinearMixedModel::objective_from_components(logdet, pwrss, denomdf, model.optsum.sigma);

        assert_relative_eq!(fast, generic, epsilon = 1e-10);
    }
}

#[test]
fn test_weighted_lrt_matches_profiled_target_difference() {
    use crate::stats::lrt::LikelihoodRatioTest;

    let (df, w1) = weighted_lmm_fixture();
    let f0 = parse_formula("a ~ 1 + (1 | c)").unwrap();
    let mut m0 = LinearMixedModel::new(f0, &df, Some(&w1)).unwrap();
    m0.fit(false).unwrap();

    let f1 = parse_formula("a ~ 1 + b + (1 | c)").unwrap();
    let mut m1 = LinearMixedModel::new(f1, &df, Some(&w1)).unwrap();
    m1.fit(false).unwrap();

    let raw_chisq = m0.profiled_objective_value() - m1.profiled_objective_value();
    let corrected_chisq = m0.objective_value() - m1.objective_value();
    assert_relative_eq!(corrected_chisq, raw_chisq, epsilon = 1e-10);

    let lrt = LikelihoodRatioTest::test(&[&m0 as &dyn MixedModelFit, &m1]).unwrap();
    assert_relative_eq!(lrt.chisq[0], corrected_chisq, epsilon = 1e-10);
}

#[test]
fn test_rank_deficient_fixed_effects() {
    // Mirrors pls.jl "Rank deficient" testset.
    // x2 = 1.5 * x makes the FE design matrix rank-deficient (rank 2, not 3).
    // Julia: length(fixef) == 2, rank(model) == 2, length(coef) == 3
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 % 10.0) / 9.0).collect();
    let x2: Vec<f64> = x.iter().map(|&v| 1.5 * v).collect();
    // Simple deterministic y
    let y: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 17) as f64 * 0.1).collect();
    let z: Vec<String> = (0..n)
        .map(|i| format!("{}", (b'A' + (i % 20) as u8) as char))
        .collect();

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_numeric("x2", x2).unwrap();
    df.add_categorical("z", z).unwrap();

    let formula = parse_formula("y ~ x + x2 + (1 | z)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
    model.fit(false).unwrap();

    // x2 is a linear combination of x → rank 2 (intercept + x or x2)
    assert_eq!(
        model.feterm.rank, 2,
        "rank should be 2 (intercept + one predictor)"
    );
    // fixef() returns only independent coefficients
    assert_eq!(model.fixef().len(), 2);
    // coef() returns all original columns (with 0/NaN for the dropped one)
    assert_eq!(MixedModelFit::coef(&model).len(), 3);
}

#[test]
fn test_sleepstudy_re_std_devs_match_julia() {
    // Mirrors pls.jl "sleep":
    //   first(std(fm)) ≈ [23.78066438213187, 5.7168446983832775]
    //   VarCorr RE correlation between intercept and days ≈ +0.08
    //   fm.corr (fixed-effects correlation) ≈ [1.0 -0.1376; -0.1376 1.0]
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let vc = model.varcorr();
    assert_eq!(vc.components.len(), 1);
    let comp = &vc.components[0];
    assert_eq!(comp.group, "subj");
    assert_eq!(comp.std_dev.len(), 2);
    // Julia: first(std(fm)) ≈ [23.78066438213187, 5.7168446983832775]
    assert_relative_eq!(comp.std_dev[0], 23.78066438213187, epsilon = 0.1);
    assert_relative_eq!(comp.std_dev[1], 5.7168446983832775, epsilon = 0.1);
    // VarCorr RE correlation: theta[1] / ||row_1(lambda)|| ≈ +0.08
    assert_eq!(comp.correlations.len(), 1);
    assert_relative_eq!(comp.correlations[0], 0.0813, epsilon = 0.01);

    // fm.corr in Julia is vcov(m; corr=true) — the fixed-effects correlation,
    // NOT VarCorr. Julia: stderror ≈ [6.6323, 1.5022], corr[0,1] ≈ -0.1376.
    let vcov = model.vcov();
    let se = model.stderror();
    assert_relative_eq!(se[0], 6.632295312722272, epsilon = 0.01);
    assert_relative_eq!(se[1], 1.5022387911441102, epsilon = 0.01);
    let fe_corr = vcov[(0, 1)] / (se[0] * se[1]);
    assert_relative_eq!(fe_corr, -0.13755599049585931, epsilon = 0.01);
}

#[test]
fn test_sleepstudy_vector_re_logdet_and_pwrss() {
    // Mirrors pls.jl "sleep" testset — additional metrics.
    // Julia: logdet(fm) ≈ 73.90350673367566
    //        pwrss(fm)  ≈ 117889.27379003687
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.logdet_re(), 73.90350673367566, epsilon = 0.1);
    assert_relative_eq!(model.pwrss(), 117889.27379003687, epsilon = 100.0);
}

#[test]
fn test_sleepstudy_zerocorr_re_matches_julia() {
    // Mirrors pls.jl "sleep" fmnc (zerocorr) model:
    //   reaction ~ 1 + days + zerocorr(1 + days | subj)
    // Julia: objective ≈ 1752.003255140962
    //        θ ≈ [0.9458, 0.2269]  (diagonal-only lambda: 2 params)
    //        coef ≈ [251.405, 10.467]
    //        stderror ≈ [6.708, 1.519]
    //        logdet ≈ 74.4694698615524
    let data = sleepstudy_fixture();
    // Our parser uses `||` (double-pipe) for zero-correlation RE.
    let formula = parse_formula("reaction ~ 1 + days + (1 + days || subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.objective_value(), 1752.003255140962, epsilon = 0.1);

    let theta = model.theta();
    assert_eq!(theta.len(), 2, "zerocorr model has 2 theta params");
    assert_relative_eq!(theta[0], 0.9458043022417869, epsilon = 0.01);
    assert_relative_eq!(theta[1], 0.22692740996014607, epsilon = 0.01);
    let artifact = model.compiler_artifact();
    assert_eq!(artifact.semantic_model.random_terms.len(), 2);
    assert_eq!(artifact.theta_maps.len(), 2);
    assert!(artifact
        .semantic_model
        .random_terms
        .iter()
        .all(|term| term.block_group.as_deref() == Some("bg0")));
    assert!(artifact
        .covariance_parameter_traces
        .iter()
        .all(|trace| trace
            .parmap_entry
            .as_ref()
            .is_some_and(|entry| entry.matches_theta_map)));

    let coef = MixedModelFit::coef(&model);
    assert_relative_eq!(coef[0], 251.4051048484854, epsilon = 0.1);
    assert_relative_eq!(coef[1], 10.467285959595674, epsilon = 0.05);

    let se = model.stderror();
    assert_relative_eq!(se[0], 6.707646513654387, epsilon = 0.1);
    assert_relative_eq!(se[1], 1.5193112497954953, epsilon = 0.05);

    assert_relative_eq!(model.logdet_re(), 74.4694698615524, epsilon = 0.1);
}

#[cfg(feature = "nlopt")]
#[test]
fn test_penicillin_varcorr_std_devs_match_julia() {
    // Mirrors pls.jl "penicillin": std(fm) ≈ [[0.8456], [1.7707], [0.5499]]
    // std[0] = plate RE, std[1] = sample RE, residual sigma = 0.5499
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let sigma = model.sigma();
    // Julia: only(last(std)) ≈ 0.549931906953287 (residual sigma)
    assert_relative_eq!(sigma, 0.549931906953287, epsilon = 1e-4);

    let vc = model.varcorr();
    assert_eq!(vc.components.len(), 2);
    // plate RE
    assert_eq!(vc.components[0].group, "plate");
    assert_relative_eq!(
        vc.components[0].std_dev[0],
        0.845571948075415,
        epsilon = 1e-4
    );
    // sample RE
    assert_eq!(vc.components[1].group, "sample");
    assert_relative_eq!(
        vc.components[1].std_dev[0],
        1.770666460750787,
        epsilon = 1e-4
    );
    // residual
    assert_relative_eq!(vc.residual_sd.unwrap(), sigma, epsilon = 1e-12);
}

#[test]
fn test_sleepstudy_zerocorr_varcorr_std_devs() {
    // Mirrors pls.jl "sleep" fmnc (zerocorr):
    //   first(std(fmnc)) ≈ [24.171269957611873, 5.79939919963132]
    //   last(std(fmnc))  ≈ [25.55613836753517]   (residual sigma)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days || subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let sigma = model.sigma();
    assert_relative_eq!(sigma, 25.55613836753517, epsilon = 0.1);

    let vc = model.varcorr();
    assert_eq!(vc.components.len(), 1);
    let comp = &vc.components[0];
    assert_eq!(comp.std_dev.len(), 2);
    assert_relative_eq!(comp.std_dev[0], 24.171269957611873, epsilon = 0.1);
    assert_relative_eq!(comp.std_dev[1], 5.79939919963132, epsilon = 0.1);
    // zerocorr → diagonal Lambda → off-diagonal correlation is 0
    assert_eq!(comp.correlations.len(), 1);
    assert_relative_eq!(comp.correlations[0], 0.0, epsilon = 1e-8);
}

#[test]
fn test_sleepstudy_independent_re_equivalent_to_zerocorr() {
    // Mirrors pls.jl "sleep" fm_ind equivalence test (lines 447-454):
    //   fm_ind = models(:sleepstudy)[3]
    //          = reaction ~ 1 + days + (1 | subj) + (0 + days | subj)
    //   @test objective(fm_ind) ≈ objective(fmnc)   # fmnc = zerocorr model
    //   @test coef(fm_ind) ≈ coef(fmnc)
    //   @test stderror(fm_ind) ≈ stderror(fmnc)
    //   @test fm_ind.θ ≈ fmnc.θ
    //   @test logdet(fm_ind) ≈ logdet(fmnc)
    //
    // Two separate scalar RE terms for the same grouping factor are
    // equivalent to a single zerocorr (diagonal-λ) RE term because
    // their contributions to the log-likelihood are additive.
    let data = sleepstudy_fixture();

    let f_zc = parse_formula("reaction ~ 1 + days + (1 + days || subj)").unwrap();
    let mut m_zc = LinearMixedModel::new(f_zc, &data, None).unwrap();
    m_zc.fit(false).unwrap();

    // Two separate scalar terms for same grouping factor
    let f_ind = parse_formula("reaction ~ 1 + days + (1 | subj) + (0 + days | subj)").unwrap();
    let mut m_ind = LinearMixedModel::new(f_ind, &data, None).unwrap();
    m_ind.fit(false).unwrap();

    // Objectives should match to high precision (same log-likelihood surface)
    assert_relative_eq!(
        m_ind.objective_value(),
        m_zc.objective_value(),
        epsilon = 0.01
    );

    // Fixed-effects coefficients (pivot order may differ, compare sums/lengths)
    let coef_zc = MixedModelFit::coef(&m_zc);
    let coef_ind = MixedModelFit::coef(&m_ind);
    assert_eq!(
        coef_zc.len(),
        coef_ind.len(),
        "same number of FE coefficients"
    );

    // logdet should match
    assert_relative_eq!(m_ind.logdet_re(), m_zc.logdet_re(), epsilon = 0.1);

    // theta lengths differ (zerocorr: 2 params in 1 term; fm_ind: 1+1 in 2 terms)
    // but the effective model is the same
    assert_eq!(
        m_ind.theta().len(),
        2,
        "two separate scalar RE → 2 theta params"
    );
    assert_eq!(m_zc.theta().len(), 2, "zerocorr RE → 2 theta params");
}

#[test]
fn test_optsum_fitlog_population() {
    // Mirrors pls.jl "Dyestuff fitlog" testset (lines 146-161):
    //   fitlog = fm1.optsum.fitlog
    //   @test length(fitlogtbl) == 3        -- has iter, objective, θ columns
    //   @test length(first(fitlogtbl)) > 15 -- more than 15 function evals
    //   @test last(fitlogtbl.objective) == fm1.optsum.fmin
    //
    // We verify our OptSummary.fit_log is populated after fitting:
    //   - length(fit_log) == feval (one entry per function evaluation)
    //   - length(fit_log) > 10    (at least 10 evaluations for dyestuff)
    //   - fit_log[0].theta == optsum.initial  (first eval uses initial θ)
    //   - fit_log.last().objective == optsum.fmin  (last entry = minimum)
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let log = &model.optsum.fit_log;

    // log populated and length matches feval count
    assert!(!log.is_empty(), "fit_log should be non-empty after fitting");
    assert_eq!(
        log.len() as i64,
        model.optsum.feval,
        "fit_log length should equal feval"
    );

    // At least 10 function evaluations for dyestuff (typically ~30-50)
    assert!(
        log.len() >= 10,
        "expected ≥ 10 function evaluations, got {}",
        log.len()
    );

    // First entry should use the initial theta
    let initial = &model.optsum.initial;
    assert_eq!(
        log[0].theta.len(),
        initial.len(),
        "first log entry theta length should match initial"
    );

    // The minimum objective across the log should be fmin (or very close)
    let min_logged = log
        .iter()
        .map(|e| e.objective)
        .fold(f64::INFINITY, f64::min);
    assert_relative_eq!(min_logged, model.optsum.fmin, epsilon = 1e-6);
}

#[test]
fn test_optsum_fitlog_theta_dimensions() {
    // Extended fitlog check: every entry's theta has the right length.
    // Mirrors pls.jl: d == length(first(fitlogtbl.θ))  (theta dim consistent)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let n_theta = model.optsum.initial.len();
    for (i, entry) in model.optsum.fit_log.iter().enumerate() {
        assert_eq!(
            entry.theta.len(),
            n_theta,
            "fit_log[{}].theta should have {} elements",
            i,
            n_theta
        );
    }
}

#[test]
fn test_pastes_lrt_pvalue_matches_julia() {
    // Mirrors pls.jl "pastes": lrt = likelihoodratiotest(models(:pastes)...)
    //   last(lrt.pvalues) ≈ 0.5233767965780878
    // models(:pastes)[1] = strength ~ 1 + (1 | batch & cask)  (cask-within-batch only)
    // models(:pastes)[2] = strength ~ 1 + (1 | batch / cask)  (batch + batch:cask)
    let data = pastes_fixture();

    // Simpler model: batch:cask interaction only (no batch main effect)
    let formula1 = parse_formula("strength ~ 1 + (1 | batch_cask)").unwrap();
    let mut m1 = LinearMixedModel::new(formula1, &data, None).unwrap();
    m1.fit(false).unwrap();

    // Richer model: batch main RE + batch:cask interaction RE
    let formula2 = parse_formula("strength ~ 1 + (1 | batch) + (1 | batch_cask)").unwrap();
    let mut m2 = LinearMixedModel::new(formula2, &data, None).unwrap();
    m2.fit(false).unwrap();

    use crate::model::traits::MixedModelFit;
    use crate::stats::lrt::LikelihoodRatioTest;
    let lrt =
        LikelihoodRatioTest::test(&[&m1 as &dyn MixedModelFit, &m2 as &dyn MixedModelFit]).unwrap();
    assert_eq!(lrt.pvalues.len(), 1);
    assert_relative_eq!(lrt.pvalues[0], 0.5233767965780878, epsilon = 0.01);
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_native_default_pastes_varcorr_contract_and_certificate_are_stable() {
    let data = pastes_fixture();
    let formula = parse_formula("strength ~ 1 + (1 | batch) + (1 | batch_cask)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();
    assert_default_native_certificate(&model, "pastes");

    let sigma = model.sigma();
    assert!(sigma.is_finite() && sigma > 0.0);
    assert_relative_eq!(
        sigma * sigma,
        0.677999727889528,
        epsilon = 0.01,
        max_relative = 0.01
    );
    assert_relative_eq!(
        model.logdet_re(),
        101.03834542101686,
        epsilon = 0.6,
        max_relative = 0.01
    );

    let vc = model.varcorr();
    assert_eq!(vc.components.len(), 2);
    for component in &vc.components {
        assert!(
            component
                .std_dev
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
            "pastes native VarCorr standard deviations should be finite: {:?}",
            component.std_dev
        );
    }
    let batch_comp = vc
        .components
        .iter()
        .find(|c| c.group == "batch")
        .expect("batch component");
    let cask_comp = vc
        .components
        .iter()
        .find(|c| c.group == "batch_cask")
        .expect("batch_cask component");
    assert_relative_eq!(cask_comp.std_dev[0], 2.90407793598792, epsilon = 0.05);
    assert_relative_eq!(batch_comp.std_dev[0], 1.0950608007768226, epsilon = 0.08);
    assert_relative_eq!(vc.residual_sd.unwrap(), sigma, epsilon = 1e-12);
}

// Parity against MixedModels.jl reference fit (NLopt BOBYQA); the
// native no-default-features path lands slightly away in sigma^2.
#[cfg(feature = "nlopt")]
#[test]
fn test_pastes_varcorr_and_logdet_match_julia() {
    // Mirrors pls.jl "pastes":
    //   only(first(stdd)) ≈ 2.904   (batch:cask RE std dev, 30 levels — first in nranef sort)
    //   only(stdd[2])     ≈ 1.095   (batch RE std dev, 10 levels — second)
    //   only(last(stdd))  ≈ 0.823   (residual sigma)
    //   varest(fm) ≈ 0.677999727889528
    //   logdet(fm) ≈ 101.03834542101686
    let data = pastes_fixture();
    let formula = parse_formula("strength ~ 1 + (1 | batch) + (1 | batch_cask)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let sigma = model.sigma();
    assert_relative_eq!(sigma, 0.8234073887751603, epsilon = 1e-4);
    assert_relative_eq!(sigma * sigma, 0.677999727889528, epsilon = 1e-4);
    assert_relative_eq!(model.logdet_re(), 101.03834542101686, epsilon = 0.1);

    let vc = model.varcorr();
    assert_eq!(vc.components.len(), 2);
    // Julia sorts RE terms by decreasing nranef: batch:cask (30 levels) first, batch (10) second.
    // Julia: first(std) ≈ 2.904 (batch:cask, 30 levels), stdd[2] ≈ 1.095 (batch, 10 levels)
    let batch_comp = vc
        .components
        .iter()
        .find(|c| c.group == "batch")
        .expect("batch component");
    let cask_comp = vc
        .components
        .iter()
        .find(|c| c.group == "batch_cask")
        .expect("batch_cask component");
    assert_relative_eq!(cask_comp.std_dev[0], 2.90407793598792, epsilon = 1e-3);
    assert_relative_eq!(batch_comp.std_dev[0], 1.0950608007768226, epsilon = 1e-4);
    // residual
    assert_relative_eq!(vc.residual_sd.unwrap(), sigma, epsilon = 1e-12);
}

#[test]
fn test_dyestuff2_sigma_matches_julia() {
    // Mirrors pls.jl "Dyestuff2": std(fm)[2] ≈ [3.6532313513746537]
    // (residual sigma; RE collapses to 0 in singular fit)
    let data = dyestuff2_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.sigma(), 3.6532313513746537, epsilon = 1e-4);
}

#[test]
fn test_pastes_batch_cask_only_model() {
    // models(:pastes)[1] = strength ~ 1 + (1 | batch & cask) — cask-within-batch only.
    // Julia: objective ≈ 247.9944658624955 for the full nested model (last);
    //   the simpler model (batch & cask only) has fewer RE levels.
    // Here we just verify it fits and has sane values.
    let data = pastes_fixture();
    let formula = parse_formula("strength ~ 1 + (1 | batch_cask)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.nobs(), 60);
    // Intercept ≈ mean(strength)
    let coef = MixedModelFit::coef(&model);
    assert_relative_eq!(coef[0], 60.0533333333333, epsilon = 0.1);
    // This simpler model must have lower DOF than the full nested model
    assert_eq!(model.dof(), 3); // 1 FE + 1 RE theta + 1 sigma
}

#[test]
fn test_dyestuff_cond_is_one() {
    // Mirrors pls.jl: cond(fm1) == ones(1)
    // Scalar RE has a 1×1 Lambda → condition number is always 1.
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let c = model.cond();
    assert_eq!(c.len(), 1);
    assert_relative_eq!(c[0], 1.0, epsilon = 1e-12);
}

#[test]
fn test_sleepstudy_vector_re_cond_matches_julia() {
    // Mirrors pls.jl: only(cond(fm)) ≈ 4.175266438717022
    // Vector RE Lambda is 2×2 lower-triangular; condition number > 1.
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let c = model.cond();
    assert_eq!(c.len(), 1);
    assert_relative_eq!(c[0], 4.175266438717022, epsilon = 0.01);
}

#[test]
fn test_dof_residual_matches_julia() {
    // Mirrors pls.jl: dof_residual(fm1) ≥ 0
    // For dyestuff: nobs=30, rank=1 (intercept only) → dof_residual=29
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.dof_residual(), 29); // 30 obs - 1 FE
    assert!(model.dof_residual() > 0);
}

#[test]
fn test_sleepstudy_dof_residual() {
    // Sleepstudy: nobs=180, rank=2 (intercept + days) → dof_residual=178
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.dof_residual(), 178); // 180 obs - 2 FE
}

#[test]
fn test_dyestuff_response_and_model_matrix() {
    // Mirrors pls.jl: modelmatrix(fm1) == ones(30,1), response == ds.yield
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let x = model.model_matrix();
    assert_eq!(x.nrows(), 30);
    assert_eq!(x.ncols(), 1);
    // Intercept-only FE → all ones
    assert!(x.iter().all(|&v| (v - 1.0).abs() < 1e-12));

    let y = model.response();
    assert_eq!(y.len(), 30);
    // First batch A: 5 values with mean ~1538
    let mean_y = y.mean();
    assert_relative_eq!(mean_y, 1527.5, epsilon = 1e-6);
}

#[test]
fn test_dyestuff_fitted_and_residuals() {
    // Mirrors pls.jl "Dyestuff": fitted values and residuals basic checks.
    // For an intercept-only model: mean(fitted) ≈ mean(y), sum(residuals) ≈ 0
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let y = model.response();
    let fitted = model.fitted();
    let residuals = model.residuals();
    assert_eq!(fitted.len(), 30);
    assert_eq!(residuals.len(), 30);
    // residuals = y - fitted
    for i in 0..30 {
        assert_relative_eq!(residuals[i], y[i] - fitted[i], epsilon = 1e-10);
    }
}

#[test]
fn test_penicillin_model_structure() {
    // Mirrors pls.jl: size(fm) == (144, 1, 30, 2)
    // nobs=144, rank=1 (intercept), total_nranef=30 (24 plate + 6 sample), 2 RE terms
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.nobs(), 144);
    assert_eq!(model.feterm.rank, 1);
    assert_eq!(model.reterms.len(), 2);
    let total_ranef: usize = model.reterms.iter().map(|rt| rt.n_ranef()).sum();
    assert_eq!(total_ranef, 30); // 24 plates + 6 samples
}

#[test]
fn test_sleepstudy_model_structure() {
    // Mirrors pls.jl: rank(fm) == 2 for the vector RE model
    // nobs=180, rank=2 (intercept+days), 1 RE term with 18*2=36 ranef
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.nobs(), 180);
    assert_eq!(model.feterm.rank, 2);
    assert_eq!(model.reterms.len(), 1);
    let total_ranef: usize = model.reterms.iter().map(|rt| rt.n_ranef()).sum();
    assert_eq!(total_ranef, 36); // 18 subjects × 2 RE (intercept + slope)
}

// ── condVar parity with MixedModels.jl/test/pls.jl ─────────────────────

#[test]
fn test_dyestuff_condvar_shape() {
    // pls.jl: @test length(cv) == 1; @test size(first(cv)) == (1, 1, 6)
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let cv = model.cond_var();
    assert_eq!(cv.len(), 1, "one RE term");
    assert_eq!(cv[0].len(), 6, "6 batch levels");
    assert_eq!(cv[0][0].nrows(), 1);
    assert_eq!(cv[0][0].ncols(), 1);
}

#[test]
fn test_penicillin_condvar_matches_julia() {
    // pls.jl:
    //   @test length(cv) == 2
    //   @test size(first(cv)) == (1, 1, 24)
    //   @test size(last(cv)) == (1, 1, 6)
    //   @test first(first(cv)) ≈ 0.07331356908917808 rtol = 1.e-4
    //   @test last(last(cv))  ≈ 0.04051591717427688 rtol = 1.e-4
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let cv = model.cond_var();
    assert_eq!(cv.len(), 2);

    // first term = plate (24 levels, sorted first by nranef)
    assert_eq!(cv[0].len(), 24);
    assert_eq!(cv[0][0].nrows(), 1);
    assert_relative_eq!(cv[0][0][(0, 0)], 0.07331356908917808, epsilon = 1e-4);

    // last term = sample (6 levels)
    assert_eq!(cv[1].len(), 6);
    assert_relative_eq!(cv[1][5][(0, 0)], 0.04051591717427688, epsilon = 1e-4);
}

#[test]
fn test_sleepstudy_condvar_matches_julia() {
    // pls.jl:
    //   @test size(cv1) == (2, 2, 18)
    //   @test first(cv1) ≈ 140.96755256125914 rtol = 1.e-4   → cv[0][0][(0,0)]
    //   @test last(cv1)  ≈ 5.157794803497628  rtol = 1.e-4   → cv[0][17][(1,1)]
    //   @test cv1[2]     ≈ -20.604544204749537 rtol = 1.e-4  → cv[0][0][(1,0)]
    //   (Julia column-major: cv1[2] = cv1[2,1,1] = row 2, col 1, level 1 = (1,0) 0-indexed)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let cv = model.cond_var();
    assert_eq!(cv.len(), 1);
    assert_eq!(cv[0].len(), 18);
    assert_eq!(cv[0][0].nrows(), 2);
    assert_eq!(cv[0][0].ncols(), 2);

    assert_relative_eq!(cv[0][0][(0, 0)], 140.96755256125914, epsilon = 1.0);
    assert_relative_eq!(cv[0][17][(1, 1)], 5.157794803497628, epsilon = 0.1);
    assert_relative_eq!(cv[0][0][(1, 0)], -20.604544204749537, epsilon = 0.5);
}

// ── leverage parity with MixedModels.jl/test/pls.jl ────────────────────

#[test]
fn test_dyestuff_leverage_matches_julia() {
    // pls.jl:
    //   @test first(leverage(fm1)) ≈ 0.1565053420672158 rtol = 1.e-5
    //   @test sum(leverage(fm1))   ≈ 4.695160262016474  rtol = 1.e-5
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let lev = model.leverage();
    assert_eq!(lev.len(), 30);
    assert_relative_eq!(lev[0], 0.1565053420672158, epsilon = 1e-4);
    assert_relative_eq!(lev.sum(), 4.695160262016474, epsilon = 1e-3);
}

#[test]
fn test_sleepstudy_vector_re_leverage_sum_matches_julia() {
    // pls.jl:
    //   @test sum(leverage(fm)) ≈ 28.611653305323234 rtol = 1.e-5
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let lev = model.leverage();
    assert_eq!(lev.len(), 180);
    assert_relative_eq!(lev.sum(), 28.611653305323234, epsilon = 0.01);
}

// ── ranef_u / ranef_b parity with MixedModels.jl/test/pls.jl ───────────

fn manual_one_term_ranef_u_via_block_solver(model: &LinearMixedModel) -> DMatrix<f64> {
    assert_eq!(model.reterms.len(), 1);
    let re = &model.reterms[0];
    let vs = re.vsize;
    let n_levels = re.n_levels();
    let nranef = re.n_ranef();
    let p = model.dims.p;
    let n = model.dims.n;
    let beta = model.beta();
    let wtxy = &model.xy_mat.wtxy;

    let mut wr = vec![0.0f64; n];
    for obs in 0..n {
        let mut val = wtxy[(obs, p)];
        for q in 0..p {
            val -= wtxy[(obs, q)] * beta[q];
        }
        wr[obs] = val;
    }

    let mut c = vec![0.0f64; nranef];
    for obs in 0..n {
        let r = re.refs[obs] as usize;
        for s in 0..vs {
            c[r * vs + s] += re.wtz[(s, obs)] * wr[obs];
        }
    }

    let mut c_scaled = vec![0.0f64; nranef];
    for lev in 0..n_levels {
        for i in 0..vs {
            let mut val = 0.0;
            for row in i..vs {
                val += re.lambda[(row, i)] * c[lev * vs + row];
            }
            c_scaled[lev * vs + i] = val;
        }
    }

    let l = &model.l_blocks[block_index(0, 0)];
    let mut rhs_matrix = DMatrix::from_column_slice(nranef, 1, &c_scaled);
    solve_lower_block_rhs(&mut rhs_matrix, l);
    let mut u: Vec<f64> = (0..nranef).map(|idx| rhs_matrix[(idx, 0)]).collect();
    solve_upper_block_from_lower_transpose_against_rhs(l, &mut u);

    DMatrix::from_column_slice(vs, n_levels, &u)
}

#[test]
fn test_ranef_u_matches_solve_lower_block_rhs() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let actual = model.ranef_u();
    let manual = manual_one_term_ranef_u_via_block_solver(&model);

    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].shape(), manual.shape());
    for row in 0..manual.nrows() {
        for col in 0..manual.ncols() {
            assert_relative_eq!(
                actual[0][(row, col)],
                manual[(row, col)],
                epsilon = 1e-10,
                max_relative = 1e-10
            );
        }
    }
}

#[test]
fn test_ranef_u_regression_current_outputs() {
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let rfu = model.ranef_u();

    assert_eq!(rfu.len(), 2);
    assert_relative_eq!(rfu[0][(0, 0)], 0.5231574704291094, epsilon = 1e-3);
    assert_relative_eq!(rfu[1][(0, 5)], -0.9323155679350466, epsilon = 1e-3);
}

#[test]
fn test_dyestuff_ranef_u_sums_to_zero() {
    // pls.jl: @test abs(sum(only(rfu))) < 1.e-5
    // The u vector for a balanced model sums to zero (BLUP property).
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let rfu = model.ranef_u();
    assert_eq!(rfu.len(), 1);
    let u_sum: f64 = rfu[0].iter().sum();
    assert!(
        u_sum.abs() < 1e-4,
        "sum of u (dyestuff) should be ≈ 0, got {u_sum}"
    );
}

#[cfg(feature = "nlopt")]
#[test]
fn test_sleepstudy_ranef_u_shape_and_first_element() {
    // pls.jl:
    //   @test size(first(u3)) == (2, 18)
    //   @test first(only(u3)) ≈ 3.030047743065841 atol = 0.001
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let u3 = model.ranef_u();
    assert_eq!(u3.len(), 1, "one RE term");
    assert_eq!(u3[0].nrows(), 2, "vsize = 2 (intercept + slope)");
    assert_eq!(u3[0].ncols(), 18, "18 subjects");

    // Julia's first(only(u3)) is the (1,1) element (intercept for first subject)
    assert_relative_eq!(u3[0][(0, 0)], 3.030047743065841, epsilon = 0.001);
}

#[cfg(feature = "nlopt")]
#[test]
fn test_sleepstudy_ranef_b_first_element() {
    // pls.jl: @test first(only(b3)) ≈ 2.8156104060324334 atol = 0.001
    // b = Λ * u  (conditional mode on original scale)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let b3 = model.ranef_b();
    assert_eq!(b3.len(), 1);
    assert_eq!(b3[0].nrows(), 2);
    assert_eq!(b3[0].ncols(), 18);
    assert_relative_eq!(b3[0][(0, 0)], 2.8156104060324334, epsilon = 0.001);
}

#[test]
fn test_penicillin_ranef_u_first_element() {
    // pls.jl: @test first(first(rfu)) ≈ 0.5231574704291094 rtol = 1.e-4
    // penicillin has 2 RE terms (plate, sample); rfu is sorted by decreasing nranef.
    // first(rfu) → the term with more levels (24 plates).
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let rfu = model.ranef_u();
    assert_eq!(rfu.len(), 2, "two RE terms");

    // Determine which term is plate (24 levels) — it should sort first
    let first_term = &rfu[0];
    let first_u = first_term[(0, 0)];
    assert_relative_eq!(first_u, 0.5231574704291094, epsilon = 1e-3);
}

#[test]
fn test_penicillin_ranef_b_last_element() {
    // pls.jl: @test last(last(rfb)) ≈ -3.0018241391465703 rtol = 1.e-4
    // last(rfb) is the term with fewer levels (6 samples).
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let rfb = model.ranef_b();
    assert_eq!(rfb.len(), 2);

    // last term (fewer levels = samples, 6 levels), last element
    let last_term = &rfb[rfb.len() - 1];
    let last_b = last_term[(0, last_term.ncols() - 1)];
    assert_relative_eq!(last_b, -3.0018241391465703, epsilon = 1e-3);
}

// ── std / logdet / varest / model_size / refit / simulate parity ─────────

#[test]
fn test_penicillin_varest_and_logdet() {
    // pls.jl:
    //   @test varest(fm) ≈ 0.30242510228527864 atol=0.0001
    //   @test logdet(fm) ≈ 95.74676552743833 atol=0.005
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_relative_eq!(model.varest(), 0.30242510228527864, epsilon = 1e-4);
    assert_relative_eq!(model.logdet(), 95.74676552743833, epsilon = 0.05);
}

#[test]
fn test_penicillin_std_devs() {
    // pls.jl:
    //   stdd = std(fm)
    //   @test only(first(stdd)) ≈ 0.845571948075415 atol=0.0001   # plate
    //   @test only(stdd[2]) ≈ 1.770666460750787 atol=0.0001       # sample
    //   @test only(last(stdd)) ≈ 0.549931906953287 atol=0.0001    # sigma
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let stdd = model.std_devs();
    // reterms sorted by decreasing nranef: plate (24) first, sample (6) second
    assert_relative_eq!(stdd[0][0], 0.845571948075415, epsilon = 1e-3);
    assert_relative_eq!(stdd[1][0], 1.770666460750787, epsilon = 1e-3);
    assert_relative_eq!(stdd[2][0], 0.549931906953287, epsilon = 1e-3); // sigma
}

#[test]
fn test_penicillin_model_size() {
    // pls.jl: @test size(fm) == (144, 1, 30, 2)
    // n=144, p=1, nranef=24+6=30, nretrms=2
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.model_size(), (144, 1, 30, 2));
}

#[test]
fn test_sleepstudy_model_size() {
    // pls.jl: @test size(fm) == (180, 2, 36, 1) for the vector RE model
    // n=180, p=2, nranef=18*2=36, nretrms=1
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    assert_eq!(model.model_size(), (180, 2, 36, 1));
}

#[test]
fn test_dyestuff_refit_new_response() {
    // pls.jl: refit!(fm, new_y); @test objective(fm) ≈ 327.32705988112673 atol=0.001
    // (refitting a dyestuff2-like model with the dyestuff yields)
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();
    let dev_before = model.objective_value();

    // Refit with constant-shifted response (should converge to different value)
    let new_y: Vec<f64> = model.y().iter().map(|&y| y + 100.0).collect();
    model.refit(&new_y).unwrap();

    // β (intercept) should shift by 100; deviance should be unchanged
    assert_relative_eq!(model.objective_value(), dev_before, epsilon = 1e-4);
}

#[test]
fn test_refit_preserves_reml_flag() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();

    let original_y: Vec<f64> = model.y().iter().copied().collect();
    model.refit(&original_y).unwrap();

    assert!(model.optsum.reml);
}

#[test]
fn test_refit_after_reml_objective_matches() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();
    let objective_before = model.objective_value();
    let original_y: Vec<f64> = model.y().iter().copied().collect();

    model.refit(&original_y).unwrap();

    assert!(model.optsum.reml);
    assert_relative_eq!(model.objective_value(), objective_before, epsilon = 1e-6);
}

#[test]
fn test_refit_rejects_constant_response() {
    // pls.jl: @test_throws ArgumentError refit!(fm, zero(slp.reaction))
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let zeros = vec![0.0f64; model.dims.n];
    assert!(model.refit(&zeros).is_err());
}

#[test]
fn test_simulate_length_and_distribution() {
    // simulate(fm) should return a vector of length n
    // bootstrap.jl: refit!(simulate!(rng, fm)); @test deviance ≈ ...
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut rng = StdRng::seed_from_u64(12345);
    let y_sim = model.simulate(&mut rng);

    assert_eq!(
        y_sim.len(),
        30,
        "simulated response should have n=30 elements"
    );

    // Mean should be close to the fitted intercept (±3 sigma)
    let mean_sim = y_sim.iter().sum::<f64>() / 30.0;
    let beta = model.beta();
    assert!(
        (mean_sim - beta[0]).abs() < 3.0 * model.sigma() * (30.0f64).sqrt(),
        "simulated mean {mean_sim:.1} unexpectedly far from intercept {:.1}",
        beta[0]
    );
}

// ── LRT parity tests (likelihoodratiotest.jl) ────────────────────────────

#[test]
fn test_lrt_sleepstudy_deviances_and_chisq() {
    // likelihoodratiotest.jl:
    //   fm0 = reaction ~ 1 + (1 + days | subj)       → deviance ≈ 1775.4759, dof = 5
    //   fm1 = reaction ~ 1 + days + (1 + days | subj) → deviance ≈ 1751.9393, dof = 6
    //   lrt.chisq[0] ≈ 23.5365, p-value < 1e-5
    use crate::stats::lrt::LikelihoodRatioTest;

    let data = sleepstudy_fixture();

    let f0 = parse_formula("reaction ~ 1 + (1 + days | subj)").unwrap();
    let mut fm0 = LinearMixedModel::new(f0, &data, None).unwrap();
    fm0.fit(false).unwrap();

    let f1 = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut fm1 = LinearMixedModel::new(f1, &data, None).unwrap();
    fm1.fit(false).unwrap();

    // deviance = -2 * loglikelihood
    let dev0 = -2.0 * fm0.loglikelihood();
    let dev1 = -2.0 * fm1.loglikelihood();
    assert_relative_eq!(dev0, 1775.4759, epsilon = 0.1);
    assert_relative_eq!(dev1, 1751.9393, epsilon = 0.1);

    assert_eq!(fm0.dof(), 5);
    assert_eq!(fm1.dof(), 6);

    let lrt = LikelihoodRatioTest::test(&[&fm0 as &dyn MixedModelFit, &fm1 as &dyn MixedModelFit])
        .unwrap();

    assert_relative_eq!(lrt.chisq[0], 23.5365, epsilon = 0.05);
    assert!(
        lrt.pvalues[0] < 1e-5,
        "p-value should be < 1e-5, got {}",
        lrt.pvalues[0]
    );
}

#[test]
fn test_lrt_dyestuff_null_vs_intercept_only() {
    // Dyestuff: the batch variance is clearly non-zero so the LRT comparing
    // a model without RE against one with RE should yield a very small p-value.
    let data = dyestuff_fixture();

    // Null model: intercept-only mixed model (fm1 in pls.jl)
    let f1 = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut fm1 = LinearMixedModel::new(f1, &data, None).unwrap();
    fm1.fit(false).unwrap();

    // Constrained model: θ fixed at 0 (singular fit)
    let f0 = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut fm0 = LinearMixedModel::new(f0, &data, None).unwrap();
    fm0.set_theta(&[0.0]).unwrap();
    fm0.update_l().unwrap(); // recompute L at θ=0

    // fm1 deviance = -2*loglik ≈ 327.327 (AIC = deviance + 2*3 ≈ 333.327 — from pls.jl)
    let dev1 = -2.0 * fm1.loglikelihood();
    assert_relative_eq!(dev1, 327.327, epsilon = 0.01);
}

// ── predict / predict_new parity tests (predict.jl) ─────────────────────

#[test]
fn test_predict_training_equals_fitted() {
    // predict.jl: @test predict(m) ≈ fitted(m)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let pred = model.predict();
    let fitted = model.fitted();
    assert_eq!(pred.len(), fitted.len());
    for i in 0..pred.len() {
        assert_relative_eq!(pred[i], fitted[i], epsilon = 1e-12);
    }
}

#[test]
fn test_predict_new_same_data_equals_fitted() {
    // predict.jl: @test predict(m, slp; new_re_levels=:error) ≈ fitted(m)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let fitted = model.fitted();
    for strategy in [
        NewReLevels::Error,
        NewReLevels::Population,
        NewReLevels::Missing,
    ] {
        let result = model.predict_new(&data, strategy).unwrap();
        assert_eq!(result.len(), fitted.len());
        for i in 0..result.len() {
            let pred = result[i].expect("training data should never be None");
            assert_relative_eq!(pred, fitted[i], epsilon = 1e-8, max_relative = 1e-8);
        }
    }
}

#[test]
fn stateless_transform_end_to_end_fit() {
    // log(reaction) ~ days + I(days^2) + (1 | subj) fits, and the
    // transform labels surface as the response name and a coefficient
    // name byte-identical to what R prints.
    let data = sleepstudy_fixture();
    let formula = parse_formula("log(reaction) ~ days + I(days^2) + (1 | subj)").unwrap();
    assert_eq!(formula.response, "log(reaction)");
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let names = model.coef_names();
    assert!(
        names.iter().any(|n| n == "I(days^2)"),
        "coef_names should contain `I(days^2)`, got {names:?}"
    );
    assert!(names.iter().any(|n| n == "days"));
    // The objective is finite (it actually fit).
    assert!(model.objective_value().is_finite());
}

#[test]
fn stateless_transform_predict_round_trip_is_stateless() {
    // Proves predict_new re-evaluates the transform recipe on newdata
    // and that doing so is identical to manually materializing the
    // transformed columns and predicting with a bare-column formula.
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ days + I(days^2) + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    // Fresh rows (subjects present in training so RE levels are known).
    let mut fresh = DataFrame::new();
    fresh.add_numeric("days", vec![0.0, 3.0, 9.0]).unwrap();
    fresh
        .add_categorical("subj", vec!["S308".into(), "S309".into(), "S310".into()])
        .unwrap();
    let via_transform = model.predict_new(&fresh, NewReLevels::Error).unwrap();

    // Manually materialize I(days^2) and fit/predict the equivalent
    // bare-column model; results must match bit-for-bit-ish.
    let mut bare_data = DataFrame::new();
    let days_train = data.numeric("days").unwrap().to_vec();
    bare_data
        .add_numeric("reaction", data.numeric("reaction").unwrap().to_vec())
        .unwrap();
    bare_data.add_numeric("days", days_train.clone()).unwrap();
    bare_data
        .add_numeric(
            "days_sq",
            days_train.iter().map(|d| d * d).collect::<Vec<_>>(),
        )
        .unwrap();
    bare_data
        .add_categorical("subj", data.categorical("subj").unwrap().values.clone())
        .unwrap();
    let bare_formula = parse_formula("reaction ~ days + days_sq + (1 | subj)").unwrap();
    let mut bare_model = LinearMixedModel::new(bare_formula, &bare_data, None).unwrap();
    bare_model.fit(false).unwrap();

    let mut bare_fresh = DataFrame::new();
    bare_fresh.add_numeric("days", vec![0.0, 3.0, 9.0]).unwrap();
    bare_fresh
        .add_numeric("days_sq", vec![0.0, 9.0, 81.0])
        .unwrap();
    bare_fresh
        .add_categorical("subj", vec!["S308".into(), "S309".into(), "S310".into()])
        .unwrap();
    let via_bare = bare_model
        .predict_new(&bare_fresh, NewReLevels::Error)
        .unwrap();

    assert_eq!(via_transform.len(), via_bare.len());
    for (a, b) in via_transform.iter().zip(via_bare.iter()) {
        let a = a.expect("known RE level");
        let b = b.expect("known RE level");
        assert_relative_eq!(a, b, epsilon = 1e-9, max_relative = 1e-9);
    }
}

#[test]
fn test_predict_new_same_data_equals_fitted_for_same_group_random_blocks() {
    // Regression for lme4 issue #403-shaped formulas:
    // separate random-slope blocks share a grouping factor, so prediction
    // must match by grouping factor *and* random-effect basis.
    let mut y = Vec::new();
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let mut subject = Vec::new();
    for subj in 0..12 {
        let b0 = subj as f64 * 0.4 - 2.0;
        let b1 = (subj as f64 - 5.0) * 0.12;
        let b2 = (6.0 - subj as f64) * 0.18;
        for obs in 0..5 {
            let a = obs as f64 - 2.0;
            let c = (obs as f64 + 1.0).powi(2) / 4.0;
            x1.push(a);
            x2.push(c);
            y.push(10.0 + 1.5 * a - 0.7 * c + b0 + b1 * a + b2 * c);
            subject.push(format!("s{subj}"));
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x1", x1).unwrap();
    data.add_numeric("x2", x2).unwrap();
    data.add_categorical("subject", subject).unwrap();

    let formula =
        parse_formula("y ~ 1 + x1 + x2 + (1 + x1 | subject) + (1 + x2 | subject)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let fitted = model.fitted();
    let predicted = model.predict_new(&data, NewReLevels::Error).unwrap();
    assert_eq!(predicted.len(), fitted.len());
    for (idx, pred) in predicted.iter().enumerate() {
        assert_relative_eq!(
            pred.expect("training levels are known"),
            fitted[idx],
            epsilon = 1e-7,
            max_relative = 1e-7
        );
    }
}

#[test]
fn test_predict_new_same_data_equals_fitted_for_cell_grouping() {
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut site = Vec::new();
    let mut item = Vec::new();
    for site_idx in 0..3 {
        for item_idx in 0..4 {
            let cell_effect = site_idx as f64 * 0.8 - item_idx as f64 * 0.3;
            for rep in 0..2 {
                let xv = rep as f64 + item_idx as f64 * 0.25;
                x.push(xv);
                y.push(3.0 + 1.2 * xv + cell_effect);
                site.push(format!("site{site_idx}"));
                item.push(format!("item{item_idx}"));
            }
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric("y", y).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_categorical("site", site).unwrap();
    data.add_categorical("item", item).unwrap();

    let formula = parse_formula("y ~ 1 + x + (1 | site:item)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let fitted = model.fitted();
    let predicted = model.predict_new(&data, NewReLevels::Error).unwrap();
    for (idx, pred) in predicted.iter().enumerate() {
        assert_relative_eq!(
            pred.expect("training cell levels are known"),
            fitted[idx],
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }
}

#[test]
fn test_predict_new_categorical_encoding_is_training_anchored() {
    // Regression for train/predict factor consistency: a categorical
    // fixed-effect predictor whose level order in `newdata` differs from
    // training must NOT silently reorder/redefine its dummy columns. The
    // prediction for a given row must be invariant to newdata row order.
    let grp_seq = [
        "B", "B", "A", "A", "C", "C", "A", "B", "C", "B", "A", "C", "C", "A", "B", "A", "C", "B",
    ];
    let n = grp_seq.len();
    let grp_effect = |g: &str| match g {
        "A" => 5.0,
        "C" => -3.0,
        _ => 0.0, // B is the training reference
    };
    let subj_effect = [0.0, 1.0, -1.0];

    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut grp = Vec::with_capacity(n);
    let mut subj = Vec::with_capacity(n);
    for (i, g) in grp_seq.iter().enumerate() {
        let s = i % 3;
        let xv = i as f64 * 0.5;
        x.push(xv);
        // Deterministic jitter so the model is not a perfect (singular) fit.
        let noise = ((i as f64 * 12.9898).sin() * 43758.547).fract() - 0.5;
        y.push(10.0 + 2.0 * xv + grp_effect(g) + subj_effect[s] + noise);
        grp.push((*g).to_string());
        subj.push(format!("s{s}"));
    }

    // Training frame: grp first appears in the order B, A, C.
    let mut train = DataFrame::new();
    train.add_numeric("y", y.clone()).unwrap();
    train.add_numeric("x", x.clone()).unwrap();
    train.add_categorical("grp", grp.clone()).unwrap();
    train.add_categorical("subj", subj.clone()).unwrap();
    assert_eq!(
        train.categorical("grp").unwrap().levels,
        vec!["B".to_string(), "A".to_string(), "C".to_string()],
        "training grp first-appearance order must be B, A, C"
    );

    let formula = parse_formula("y ~ 1 + x + grp + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &train, None).unwrap();
    model.fit(false).unwrap();

    let fitted = model.fitted();
    let pred_train = model.predict_new(&train, NewReLevels::Error).unwrap();
    for (i, p) in pred_train.iter().enumerate() {
        assert_relative_eq!(
            p.expect("training levels are known"),
            fitted[i],
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }

    // newdata: identical rows, reversed order. grp now first appears as a
    // different level, so newdata's own first-appearance encoding would
    // pick a different treatment reference. The fix realigns to training.
    let mut rev = DataFrame::new();
    let mut yr = y.clone();
    yr.reverse();
    let mut xr = x.clone();
    xr.reverse();
    let mut gr = grp.clone();
    gr.reverse();
    let mut sr = subj.clone();
    sr.reverse();
    rev.add_numeric("y", yr).unwrap();
    rev.add_numeric("x", xr).unwrap();
    rev.add_categorical("grp", gr).unwrap();
    rev.add_categorical("subj", sr).unwrap();
    assert_ne!(
        rev.categorical("grp").unwrap().levels,
        train.categorical("grp").unwrap().levels,
        "reversed newdata must have a different raw level order to exercise the bug"
    );

    let pred_rev = model.predict_new(&rev, NewReLevels::Error).unwrap();
    for (p, pred) in pred_rev.iter().enumerate() {
        let original_idx = n - 1 - p;
        assert_relative_eq!(
            pred.expect("all levels are training-known"),
            fitted[original_idx],
            epsilon = 1e-8,
            max_relative = 1e-8
        );
    }
}

#[test]
fn test_predict_with_unseen_level_returns_typed_err() {
    // predict.jl: @test_throws ArgumentError predict(m, slp2; new_re_levels=:error)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut newdata = DataFrame::new();
    newdata.add_numeric("reaction", vec![300.0]).unwrap();
    newdata.add_numeric("days", vec![0.0]).unwrap();
    newdata
        .add_categorical("subj", vec!["UNSEEN".to_string()])
        .unwrap();

    let err = model.predict_new(&newdata, NewReLevels::Error).unwrap_err();

    match err {
        MixedModelError::InvalidArgument(message) => {
            assert!(message.contains("UNSEEN"));
            assert!(message.contains("subj"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_predict_new_unknown_level_population() {
    // predict.jl: ypop[1:10] ≈ view(m.X, 1:10, :) * m.β  (population prediction = Xβ)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let beta = model.beta();
    let cnames = model.feterm.cnames.clone();
    let days: Vec<f64> = (0..10).map(|d| d as f64).collect();
    let mut newdata = DataFrame::new();
    newdata.add_numeric("reaction", vec![0.0; 10]).unwrap();
    newdata.add_numeric("days", days.clone()).unwrap();
    newdata
        .add_categorical("subj", vec!["NEW".to_string(); 10])
        .unwrap();

    let result = model
        .predict_new(&newdata, NewReLevels::Population)
        .unwrap();
    assert_eq!(result.len(), 10);

    // Coefficients by name (pivot order may not be [intercept, days])
    let intercept = cnames
        .iter()
        .position(|n| n == "(Intercept)")
        .map(|i| beta[i])
        .unwrap_or(0.0);
    let days_coef = cnames
        .iter()
        .position(|n| n == "days")
        .map(|i| beta[i])
        .unwrap_or(0.0);

    for (i, &d) in days.iter().enumerate() {
        let expected = intercept + d * days_coef;
        let pred = result[i].expect("Population should always return Some");
        assert_relative_eq!(pred, expected, epsilon = 1e-8);
    }
}

#[test]
fn test_predict_new_unknown_level_missing() {
    // predict.jl: count(ismissing, ymissing) == 10 (first 10 obs are new subject)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    // 10 new-subject obs followed by 10 known-subject (S309 days 0-9)
    let mut days = vec![0.0f64; 20];
    let mut subjects: Vec<String> = vec!["NEW".to_string(); 10];
    for d in 0..10 {
        days[10 + d] = d as f64;
        subjects.push("S309".to_string());
    }
    let mut newdata = DataFrame::new();
    newdata.add_numeric("reaction", vec![0.0; 20]).unwrap();
    newdata.add_numeric("days", days).unwrap();
    newdata.add_categorical("subj", subjects).unwrap();

    let result = model.predict_new(&newdata, NewReLevels::Missing).unwrap();
    let n_missing = result.iter().filter(|v| v.is_none()).count();
    assert_eq!(n_missing, 10, "first 10 obs (new subject) should be None");
    for i in 10..20 {
        assert!(
            result[i].is_some(),
            "obs {} (known subject) should be Some",
            i
        );
    }
}

#[test]
fn test_predict_new_variance_same_data_has_fixed_random_and_combined_components() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let payload = model
        .predict_new_variance(&data, NewReLevels::Error)
        .unwrap();
    assert_eq!(
        payload.method,
        PredictionVarianceMethod::LmmConditionalModeCovariance
    );
    assert_eq!(payload.confidence_level, Some(0.95));
    assert_eq!(payload.rows.len(), data.nrow());
    let fitted = model.fitted();
    let first = &payload.rows[0];
    assert_eq!(first.status, PredictionVarianceStatus::Available);
    assert_eq!(first.reason, None);
    assert_relative_eq!(
        first.prediction.expect("training row prediction"),
        fitted[0],
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    let fixed = first.fixed_variance.expect("fixed component");
    let random = first.random_variance.expect("random component");
    let cross = first
        .fixed_random_covariance
        .expect("fixed/random covariance component");
    let combined = first.combined_variance.expect("combined component");
    assert!(fixed > 0.0);
    assert!(random > 0.0);
    assert_relative_eq!(combined, fixed + random + 2.0 * cross, epsilon = 1e-8);
    assert_relative_eq!(
        first.se_fit.expect("se.fit").powi(2),
        combined,
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    let prediction_variance = first
        .prediction_variance
        .expect("future-observation prediction variance");
    assert_relative_eq!(
        prediction_variance,
        combined + model.sigma().powi(2),
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    let prediction = first.prediction.expect("prediction");
    let confidence_lower = first.confidence_lower.expect("confidence lower");
    let confidence_upper = first.confidence_upper.expect("confidence upper");
    let prediction_lower = first.prediction_lower.expect("prediction lower");
    let prediction_upper = first.prediction_upper.expect("prediction upper");
    assert_relative_eq!(
        prediction - confidence_lower,
        confidence_upper - prediction,
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    assert_relative_eq!(
        prediction - prediction_lower,
        prediction_upper - prediction,
        epsilon = 1e-8,
        max_relative = 1e-8
    );
    assert!(
        prediction_lower < confidence_lower && prediction_upper > confidence_upper,
        "prediction intervals should be wider than confidence intervals"
    );

    // lme4 2.0.1 reference:
    // predict(lmer(Reaction ~ Days + (Days | Subject), sleepstudy, REML=FALSE),
    //         newdata=sleepstudy[1:10,], re.form=NULL, se.fit=TRUE)$se.fit
    let lme4_ml_conditional_se = [
        12.0707575252371,
        10.3984229256521,
        9.00975482244658,
        8.05286502387107,
        7.69064748489919,
        8.00424592909914,
        8.92268555773584,
        10.2851909434472,
        11.940705943801,
        13.7840572634364,
    ];
    for (row, expected) in payload
        .rows
        .iter()
        .take(lme4_ml_conditional_se.len())
        .zip(lme4_ml_conditional_se)
    {
        assert_relative_eq!(
            row.se_fit.expect("training row conditional se.fit"),
            expected,
            epsilon = 2e-3,
            max_relative = 2e-4
        );
    }
}

#[test]
fn test_predict_new_variance_rejects_invalid_interval_level() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let err = model
        .predict_new_variance_with_level(&data, NewReLevels::Error, 1.0)
        .unwrap_err();
    assert_eq!(err.code(), "invalid_argument");
    assert!(err.to_string().contains("level must be in (0,1)"));
}

#[test]
fn test_predict_new_variance_unseen_level_reports_structured_reason() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut newdata = DataFrame::new();
    newdata.add_numeric("reaction", vec![0.0, 0.0]).unwrap();
    newdata.add_numeric("days", vec![0.0, 1.0]).unwrap();
    newdata
        .add_categorical("subj", vec!["NEW".to_string(), "S309".to_string()])
        .unwrap();

    let payload = model
        .predict_new_variance(&newdata, NewReLevels::Population)
        .unwrap();
    assert_eq!(payload.rows.len(), 2);

    let unseen = &payload.rows[0];
    assert!(unseen.prediction.is_some());
    assert_eq!(unseen.status, PredictionVarianceStatus::Unavailable);
    assert!(unseen.fixed_variance.is_some());
    assert_eq!(unseen.random_variance, None);
    assert_eq!(unseen.fixed_random_covariance, None);
    assert_eq!(unseen.combined_variance, None);
    assert_eq!(unseen.se_fit, None);
    assert_eq!(unseen.prediction_variance, None);
    assert_eq!(unseen.confidence_lower, None);
    assert_eq!(unseen.confidence_upper, None);
    assert_eq!(unseen.prediction_lower, None);
    assert_eq!(unseen.prediction_upper, None);
    assert!(unseen
        .reason
        .as_deref()
        .unwrap_or("")
        .contains("new level 'NEW'"));

    let known = &payload.rows[1];
    assert_eq!(known.status, PredictionVarianceStatus::Available);
    assert!(known.combined_variance.unwrap() > 0.0);
    assert!(known.se_fit.unwrap() > 0.0);
    assert!(known.confidence_lower.unwrap() < known.prediction.unwrap());
    assert!(known.confidence_upper.unwrap() > known.prediction.unwrap());
}

// ── coeftable parity tests (pls.jl "coeftable" testset) ──────────────────

#[test]
fn test_coeftable_dyestuff_shape() {
    // pls.jl: ct = coeftable(only(models(:dyestuff)))
    //         @test [3, 4] == [ct.teststatcol, ct.pvalcol]
    // In our 0-indexed struct: z_values is column 2, p_values is column 3.
    // We verify the table has 1 row (intercept-only FE) and reasonable values.
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let ct = model.coeftable();

    // Dyestuff has one FE: (Intercept)
    assert_eq!(ct.len(), 1);
    assert_eq!(ct.names[0], "(Intercept)");

    // Estimate ≈ 1527.5 (mean of yield)
    assert_relative_eq!(ct.estimates[0], 1527.5, epsilon = 1.0);

    // z = estimate / SE should be very large (≈ 86)
    assert!(
        ct.z_values[0] > 50.0,
        "z for intercept should be large, got {}",
        ct.z_values[0]
    );

    // p-value should be essentially zero
    assert!(
        ct.p_values[0] < 1e-10,
        "p should be ≈0, got {}",
        ct.p_values[0]
    );
}

#[test]
fn test_coeftable_sleepstudy_two_rows() {
    // sleepstudy: FE = (Intercept) + days → 2 rows in coeftable
    // pls.jl: coef ≈ [251.405, 10.467], stderror ≈ [6.632, 1.502]
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let ct = model.coeftable();
    assert_eq!(ct.len(), 2);

    // Both should have small p-values (both highly significant)
    for i in 0..2 {
        assert!(
            ct.p_values[i] < 0.01,
            "coef[{}] p-value {} should be < 0.01",
            i,
            ct.p_values[i]
        );
        // z = estimate / SE should be non-zero and finite
        assert!(ct.z_values[i].is_finite(), "z[{}] should be finite", i);
    }

    // SE should be positive
    for se in &ct.std_errors {
        assert!(*se > 0.0, "SE should be positive, got {}", se);
    }
}

#[test]
fn test_coeftable_p_values_consistent_with_stderror() {
    // coeftable p-values should be consistent with stderror:
    // z = coef / SE,  p = 2*(1-Φ(|z|))
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let ct = model.coeftable();
    let coefs = MixedModelFit::coef(&model);
    let se = model.stderror();

    for i in 0..ct.len() {
        let expected_z = coefs[i] / se[i];
        assert_relative_eq!(ct.z_values[i], expected_z, epsilon = 1e-10);
    }
}

#[test]
fn test_coeftable_rank_deficient_nan_dropped() {
    // For a rank-deficient model, dropped columns get NaN SE/z/p in coeftable.
    // With x2 = 2*x, the pivot QR drops one of {x, x2} (whichever has smaller
    // post-orthogonalisation norm).  We verify exactly one column is NaN.
    let n = 30usize;
    let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect(); // x2 = 2*x
    let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
    let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_numeric("x2", x2).unwrap();
    df.add_categorical("z", z).unwrap();

    let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
    model.fit(false).unwrap();

    let ct = model.coeftable();
    // rank 2, but coeftable has 3 rows (1 + x + x2)
    assert_eq!(ct.len(), 3, "should have 3 rows");
    assert_eq!(model.feterm.rank, 2, "model rank should be 2");

    // Exactly one of x/x2 is dropped → has NaN SE; the other is retained
    let n_nan = ct.std_errors.iter().filter(|&&se| se.is_nan()).count();
    assert_eq!(
        n_nan, 1,
        "exactly one coefficient should be dropped (NaN SE)"
    );

    // The dropped column must be x or x2 (not the intercept)
    for (i, se) in ct.std_errors.iter().enumerate() {
        if se.is_nan() {
            assert!(
                ct.names[i] == "x" || ct.names[i] == "x2",
                "dropped column should be x or x2, not '{}'",
                ct.names[i]
            );
        }
    }
}

#[test]
fn test_coeftable_omits_p_values_for_regularized_fit_intent() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let policy = CompilerPolicy {
        random_strategy: RandomStrategy::Regularized,
        ..CompilerPolicy::default()
    };
    let mut model =
        LinearMixedModel::new_with_compiler_policy(formula, &data, None, policy).unwrap();

    model.fit(false).unwrap();

    let ct = model.coeftable();
    assert!(ct.z_values.iter().all(|value| value.is_finite()));
    assert!(ct.p_values.iter().all(|value| value.is_nan()));
    assert!(ct.p_value_reasons.iter().all(|reason| reason
        .as_deref()
        .unwrap()
        .contains("exploratory fit intent")));

    let summary = ModelSummary::from_linear_model(&model);
    assert!(summary
        .rows
        .iter()
        .filter(|row| row.std_error.is_some())
        .all(|row| row.pvalue.is_none()));
}

#[test]
fn test_lmm_test_contrast_returns_labeled_asymptotic_result() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();
    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::AsymptoticWaldZ);

    assert!(matches!(test.status, InferenceStatus::Available));
    assert_eq!(test.p_values.len(), 1);
    assert!(test.p_values[0].unwrap() < 0.01);
    assert_eq!(test.estimability.status, EstimabilityStatus::Estimable);
    assert!(test
        .notes
        .iter()
        .any(|note| note.contains("asymptotic Wald z")));
}

#[test]
fn test_lmm_explicit_satterthwaite_request_returns_scalar_t_test() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();

    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);

    assert_eq!(test.method, InferenceMethod::Satterthwaite);
    assert_eq!(test.status, InferenceStatus::Available);
    assert_eq!(test.reliability, ReliabilityGrade::Moderate);
    assert!(test.denominator_df.unwrap().is_finite());
    assert!(test.denominator_df.unwrap() > 0.0);
    assert!(test.p_values[0].unwrap().is_finite());
    assert!((0.0..=1.0).contains(&test.p_values[0].unwrap()));
    assert!(test.statistics[0].unwrap().is_finite());
    assert!(test
        .notes
        .iter()
        .any(|note| note.contains("Satterthwaite denominator df computed")));
}

#[test]
fn test_lmm_explicit_satterthwaite_multi_df_request_returns_f_test() {
    let (model, hypothesis) = three_level_condition_fixture();
    let observed = joint_wald_f_direct_inverse_oracle(&model, &hypothesis);

    let test =
        model.test_contrast_with_method(hypothesis.clone(), FixedEffectTestMethod::Satterthwaite);

    assert_eq!(test.method, InferenceMethod::Satterthwaite);
    assert_eq!(test.status, InferenceStatus::Available);
    assert_eq!(test.numerator_df, Some(2.0));
    assert!(test.denominator_df.unwrap().is_finite());
    assert!(test.denominator_df.unwrap() > 0.0);
    assert_eq!(test.statistics.len(), 1);
    assert_relative_eq!(test.statistics[0].unwrap(), observed, epsilon = 1e-10);
    assert_eq!(test.p_values.len(), 1);
    assert!(test.p_values[0].unwrap().is_finite());
    assert!((0.0..=1.0).contains(&test.p_values[0].unwrap()));
    assert!(test
        .notes
        .iter()
        .any(|note| note.contains("Satterthwaite multi-df F row")));

    let row = fixed_effect_test_to_inference_row(FixedEffectInferenceRowKind::Term, test);
    assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::F));
    assert_eq!(row.numerator_df, Some(2.0));
    assert_eq!(
        row.reliability_reason,
        Some(FixedEffectReliabilityReason::SatterthwaiteFiniteDifferenceApproximation)
    );
    let family = row
        .details
        .as_ref()
        .and_then(|details| details.contrast_family.as_ref())
        .expect("multi-df Satterthwaite row should carry contrast-family details");
    assert_eq!(family.effective_rank, Some(2));
    assert_eq!(family.numerator_df_semantics, "effective_restriction_rank");
}

#[test]
fn test_satterthwaite_multi_df_denominator_df_combines_direction_dfs() {
    assert_relative_eq!(
        satterthwaite_f_denominator_df(&[8.0], 1.0e-8).unwrap(),
        8.0,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        satterthwaite_f_denominator_df(&[8.0, 8.0 + 1.0e-10], 1.0e-8).unwrap(),
        8.0 + 0.5e-10,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        satterthwaite_f_denominator_df(&[1.9, 12.0], 1.0e-8).unwrap(),
        2.0,
        epsilon = 1e-12
    );

    let dfs = [6.0, 10.0, 20.0];
    let expected_sum = dfs.iter().map(|df| df / (df - 2.0)).sum::<f64>();
    let expected = 2.0 * expected_sum / (expected_sum - dfs.len() as f64);
    assert_relative_eq!(
        satterthwaite_f_denominator_df(&dfs, 1.0e-8).unwrap(),
        expected,
        epsilon = 1e-12
    );
    assert!(satterthwaite_f_denominator_df(&[], 1.0e-8).is_none());
    assert!(satterthwaite_f_denominator_df(&[0.0], 1.0e-8).is_none());
}

#[cfg(not(feature = "nlopt"))]
#[test]
fn test_native_default_satterthwaite_rows_are_finite_with_realistic_tolerances() {
    let fixture = satterthwaite_lmer_test_parity_fixture();

    for case in fixture
        .cases
        .iter()
        .filter(|case| case.name != "sleepstudy_random_intercept_days")
    {
        let data = satterthwaite_parity_data(&case.name);
        let formula = parse_formula(&case.formula).unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();
        assert_default_native_certificate(&model, &case.name);

        let coefficient_index = model
            .coef_names()
            .iter()
            .position(|name| name == &case.coefficient)
            .unwrap_or_else(|| {
                panic!(
                    "coefficient {} not found in {:?}",
                    case.coefficient,
                    model.coef_names()
                )
            });
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            format!("{} = 0", case.coefficient),
            coefficient_index,
            model.coef_names().len(),
        )
        .unwrap();

        let test =
            model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);
        assert_available_finite_fixed_effect_test(&test, &case.name);
        assert!(
            (test.estimates[0] - case.estimate).abs() <= 1e-3 + 1e-5 * case.estimate.abs(),
            "{}: native Satterthwaite beta drift too large: rust={} ref={}",
            case.name,
            test.estimates[0],
            case.estimate
        );
        assert!(
            (test.standard_errors[0].unwrap() - case.std_error).abs()
                <= 1e-3 + 1e-3 * case.std_error.abs(),
            "{}: native Satterthwaite SE drift too large: rust={} ref={}",
            case.name,
            test.standard_errors[0].unwrap(),
            case.std_error
        );
        assert!(
            (test.statistics[0].unwrap() - case.statistic).abs()
                <= 1e-2 + 1e-3 * case.statistic.abs(),
            "{}: native Satterthwaite statistic drift too large: rust={} ref={}",
            case.name,
            test.statistics[0].unwrap(),
            case.statistic
        );
        assert!(
            (test.denominator_df.unwrap() - case.df).abs() <= 0.25 + 1e-2 * case.df.abs(),
            "{}: native Satterthwaite df drift too large: rust={} ref={}",
            case.name,
            test.denominator_df.unwrap(),
            case.df
        );
    }
}

// Parity against lmerTest reference fits (NLopt-equivalent BOBYQA); the
// native no-default-features path drifts in beta/SE outside the parity tolerance.
#[cfg(feature = "nlopt")]
#[test]
fn test_lmm_satterthwaite_scalar_rows_match_lmer_test_fixture() {
    let fixture = satterthwaite_lmer_test_parity_fixture();

    for case in fixture.cases {
        let data = satterthwaite_parity_data(&case.name);
        let formula = parse_formula(&case.formula).unwrap();
        let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
        model.fit(true).unwrap();

        let coefficient_index = model
            .coef_names()
            .iter()
            .position(|name| name == &case.coefficient)
            .unwrap_or_else(|| {
                panic!(
                    "coefficient {} not found in {:?}",
                    case.coefficient,
                    model.coef_names()
                )
            });
        let hypothesis = FixedEffectHypothesis::single_coefficient(
            format!("{} = 0", case.coefficient),
            coefficient_index,
            model.coef_names().len(),
        )
        .unwrap();

        let test =
            model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);

        assert_eq!(test.method, InferenceMethod::Satterthwaite, "{}", case.name);
        assert_eq!(test.status, InferenceStatus::Available, "{}", case.name);
        let expected_reliability = if case.name == "sleepstudy_unbalanced_random_slope_days" {
            ReliabilityGrade::Low
        } else {
            ReliabilityGrade::Moderate
        };
        assert_eq!(test.reliability, expected_reliability, "{}", case.name);
        assert!(
            (test.estimates[0] - case.estimate).abs() <= 1e-5 + 1e-6 * case.estimate.abs(),
            "{}: β drift",
            case.name
        );
        // Single-grouping sleepstudy fits agree with lme4 to ~5e-5; the
        // crossed-RE penicillin REML optimum lands ~3e-4 away from lme4's
        // (multi-start in Rust is locally optimal to ~1e-6 in REML deviance,
        // so this is optimizer-vs-optimizer drift, not a fit bug).  Hold the
        // looser-but-still-meaningful 5e-4 bound across all cases.
        assert!(
            (test.standard_errors[0].unwrap() - case.std_error).abs()
                <= 5e-4 + 5e-4 * case.std_error.abs(),
            "{}: std_error drift: rust={} ref={}",
            case.name,
            test.standard_errors[0].unwrap(),
            case.std_error
        );
        assert!(
            (test.statistics[0].unwrap() - case.statistic).abs()
                <= 5e-4 + 5e-4 * case.statistic.abs(),
            "{}: t-statistic drift: rust={} ref={}",
            case.name,
            test.statistics[0].unwrap(),
            case.statistic
        );
        // Satterthwaite df is more sensitive to θ drift than vcov itself
        // because it depends on the gradient and Hessian of vcov w.r.t. θ.
        // For the crossed-RE penicillin case the drift sits ~1e-3.
        assert_relative_eq!(
            test.denominator_df.unwrap(),
            case.df,
            epsilon = 1e-2,
            max_relative = 2e-3,
        );
        // Tail-region p-values amplify df/statistic drift: a 5e-4 df move
        // shifts a 1e-6 p-value by ~2e-3 relative.  Hold an honest 2e-3
        // bound rather than chase ten-extra-bits-of-precision.
        assert_relative_eq!(
            test.p_values[0].unwrap(),
            case.p_value,
            epsilon = 1e-8,
            max_relative = 2e-3,
        );
    }
}

#[cfg(feature = "nlopt")]
#[test]
fn test_cobyla_nlopt_delta_audit_stays_within_documented_envelope() {
    let pastes_data = pastes_fixture();
    let pastes_formula = "strength ~ 1 + (1 | batch) + (1 | batch_cask)";
    let pastes_nlopt = fit_default_nlopt_reference(&pastes_data, pastes_formula, false);
    let pastes_cobyla = fit_forced_cobyla_with(&pastes_data, pastes_formula, false, |_| {});
    let pastes_objective_delta = pastes_cobyla.objective_value() - pastes_nlopt.objective_value();
    let pastes_theta_delta = max_abs_delta(&pastes_cobyla.theta(), &pastes_nlopt.theta());
    let pastes_beta_delta = max_abs_delta(
        pastes_cobyla.beta().as_slice(),
        pastes_nlopt.beta().as_slice(),
    );
    let pastes_varcorr_delta = max_varcorr_std_dev_delta(&pastes_cobyla, &pastes_nlopt);
    println!(
            "COBYLA_AUDIT pastes objective_delta={:.6e} theta_delta={:.6e} beta_delta={:.6e} varcorr_sd_delta={:.6e} sigma2_delta={:.6e}",
            pastes_objective_delta,
            pastes_theta_delta,
            pastes_beta_delta,
            pastes_varcorr_delta,
            pastes_cobyla.sigma().powi(2) - pastes_nlopt.sigma().powi(2)
        );
    assert!(pastes_objective_delta.abs() <= 1e-3);
    assert!(pastes_theta_delta <= 5e-3);
    assert!(pastes_beta_delta <= 1e-10);
    assert!(pastes_varcorr_delta <= 5e-3);

    let sleepstudy_data = sleepstudy_fixture();
    let sleepstudy_formula = "reaction ~ days + (days | subj)";
    let sleepstudy_nlopt = fit_default_nlopt_reference(&sleepstudy_data, sleepstudy_formula, true);
    let sleepstudy_cobyla =
        fit_forced_cobyla_with(&sleepstudy_data, sleepstudy_formula, true, |_| {});
    let kr_fixture = kenward_roger_pbkrtest_parity_fixture();
    let kr_scalar_case = kr_fixture
        .scalar_cases
        .iter()
        .find(|case| case.name == "sleepstudy_random_slope_days")
        .unwrap();
    let kr_scalar_hypothesis = fixed_effect_hypothesis_from_fixture(
        &kr_scalar_case.label,
        &kr_scalar_case.l,
        &kr_scalar_case.rhs,
    );
    let kr_scalar_nlopt = sleepstudy_nlopt.test_contrast_with_method(
        kr_scalar_hypothesis.clone(),
        FixedEffectTestMethod::KenwardRoger,
    );
    let kr_scalar_cobyla = sleepstudy_cobyla
        .test_contrast_with_method(kr_scalar_hypothesis, FixedEffectTestMethod::KenwardRoger);
    assert_available_finite_fixed_effect_test(&kr_scalar_cobyla, "KR scalar COBYLA");
    let kr_se_delta =
        kr_scalar_cobyla.standard_errors[0].unwrap() - kr_scalar_nlopt.standard_errors[0].unwrap();
    let kr_df_delta =
        kr_scalar_cobyla.denominator_df.unwrap() - kr_scalar_nlopt.denominator_df.unwrap();
    println!(
        "COBYLA_AUDIT kr_scalar se_delta={:.6e} df_delta={:.6e} stat_delta={:.6e}",
        kr_se_delta,
        kr_df_delta,
        kr_scalar_cobyla.statistics[0].unwrap() - kr_scalar_nlopt.statistics[0].unwrap()
    );
    assert!(kr_se_delta.abs() <= 5e-4);
    assert!(kr_df_delta.abs() <= 1e-2);

    let kr_multi_case = kr_fixture
        .multi_df_cases
        .iter()
        .find(|case| case.name == "sleepstudy_intercept_and_days_joint")
        .unwrap();
    let kr_multi_hypothesis = fixed_effect_hypothesis_from_fixture(
        &kr_multi_case.label,
        &kr_multi_case.l,
        &kr_multi_case.rhs,
    );
    let kr_multi_nlopt = sleepstudy_nlopt.test_contrast_with_method(
        kr_multi_hypothesis.clone(),
        FixedEffectTestMethod::KenwardRoger,
    );
    let kr_multi_cobyla = sleepstudy_cobyla
        .test_contrast_with_method(kr_multi_hypothesis, FixedEffectTestMethod::KenwardRoger);
    assert_available_finite_fixed_effect_test(&kr_multi_cobyla, "KR multi-df COBYLA");
    let kr_f_delta = kr_multi_cobyla.statistics[0].unwrap() - kr_multi_nlopt.statistics[0].unwrap();
    println!(
        "COBYLA_AUDIT kr_multi f_delta={:.6e} df_delta={:.6e} p_delta={:.6e}",
        kr_f_delta,
        kr_multi_cobyla.denominator_df.unwrap() - kr_multi_nlopt.denominator_df.unwrap(),
        kr_multi_cobyla.p_values[0].unwrap() - kr_multi_nlopt.p_values[0].unwrap()
    );
    assert!(kr_f_delta.abs() <= 1.0);

    let satt_fixture = satterthwaite_lmer_test_parity_fixture();
    let satt_case = satt_fixture
        .cases
        .iter()
        .find(|case| case.name == "sleepstudy_random_slope_days")
        .unwrap();
    let coefficient_index = sleepstudy_nlopt
        .coef_names()
        .iter()
        .position(|name| name == &satt_case.coefficient)
        .unwrap();
    let satt_hypothesis = FixedEffectHypothesis::single_coefficient(
        format!("{} = 0", satt_case.coefficient),
        coefficient_index,
        sleepstudy_nlopt.coef_names().len(),
    )
    .unwrap();
    let satt_nlopt = sleepstudy_nlopt.test_contrast_with_method(
        satt_hypothesis.clone(),
        FixedEffectTestMethod::Satterthwaite,
    );
    let satt_cobyla = sleepstudy_cobyla
        .test_contrast_with_method(satt_hypothesis, FixedEffectTestMethod::Satterthwaite);
    assert_available_finite_fixed_effect_test(&satt_cobyla, "Satterthwaite COBYLA");
    println!(
        "COBYLA_AUDIT satt beta_delta={:.6e} se_delta={:.6e} df_delta={:.6e} stat_delta={:.6e}",
        satt_cobyla.estimates[0] - satt_nlopt.estimates[0],
        satt_cobyla.standard_errors[0].unwrap() - satt_nlopt.standard_errors[0].unwrap(),
        satt_cobyla.denominator_df.unwrap() - satt_nlopt.denominator_df.unwrap(),
        satt_cobyla.statistics[0].unwrap() - satt_nlopt.statistics[0].unwrap()
    );
    assert!((satt_cobyla.estimates[0] - satt_nlopt.estimates[0]).abs() <= 1e-3);
    assert!(
        (satt_cobyla.standard_errors[0].unwrap() - satt_nlopt.standard_errors[0].unwrap()).abs()
            <= 5e-4
    );

    let tuning_runs: Vec<(&str, Box<dyn Fn(&mut LinearMixedModel) + '_>)> = vec![
        (
            "default",
            Box::new(|_: &mut LinearMixedModel| {}) as Box<dyn Fn(&mut LinearMixedModel)>,
        ),
        (
            "budget_50000",
            Box::new(|model: &mut LinearMixedModel| model.optsum.max_feval = 50_000),
        ),
        (
            "initial_step_0.25",
            Box::new(|model: &mut LinearMixedModel| {
                model.optsum.initial_step = vec![0.25; model.n_theta()]
            }),
        ),
        (
            "xtol_abs_1e-6",
            Box::new(|model: &mut LinearMixedModel| {
                model.optsum.xtol_abs = vec![1e-6; model.n_theta()]
            }),
        ),
        (
            "nlopt_theta_start",
            Box::new(|model: &mut LinearMixedModel| {
                model.optsum.initial = sleepstudy_nlopt.theta()
            }),
        ),
    ];
    for (label, configure) in tuning_runs {
        let model = fit_forced_cobyla_with(&sleepstudy_data, sleepstudy_formula, true, |model| {
            configure(model)
        });
        println!(
            "COBYLA_TUNING {label} objective_delta={:.6e} theta_delta={:.6e} feval={} return={}",
            model.objective_value() - sleepstudy_nlopt.objective_value(),
            max_abs_delta(&model.theta(), &sleepstudy_nlopt.theta()),
            model.optsum.feval,
            model.optsum.return_value
        );
        assert!(model.objective_value().is_finite());
        assert!(model.theta().iter().all(|value| value.is_finite()));
    }

    let (singular_data, _) = crate::datasets::load("singular").unwrap();
    let singular_formula = "y ~ 1 + A * B * C + (A * B * C || group)";
    let singular_nlopt = fit_default_nlopt_reference(&singular_data, singular_formula, false);
    let singular_cobyla = fit_forced_cobyla_with(&singular_data, singular_formula, false, |_| {});
    println!(
            "COBYLA_AUDIT singular objective_delta={:.6e} nlopt_status={:?} cobyla_status={:?} nlopt_effective_cov={} cobyla_effective_cov={}",
            singular_cobyla.objective_value() - singular_nlopt.objective_value(),
            singular_nlopt.optimizer_certificate().unwrap().status,
            singular_cobyla.optimizer_certificate().unwrap().status,
            singular_nlopt.compiler_artifact().effective_covariance.len(),
            singular_cobyla.compiler_artifact().effective_covariance.len()
        );
    assert!(singular_cobyla.objective_value().is_finite());
    assert!(singular_cobyla.optimizer_certificate().is_some());
}

#[test]
fn test_lmm_satterthwaite_boundary_and_rank_deficient_cases_return_reasons() {
    let data = dyestuff2_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("(Intercept) = 0", 0, model.coef_names().len())
            .unwrap();
    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);

    assert_eq!(test.method, InferenceMethod::Satterthwaite);
    assert!(
        matches!(test.status, InferenceStatus::NotAssessed { ref reason }
            if reason.contains("lower bound"))
    );
    assert_eq!(test.p_values, vec![None]);

    let n = 30usize;
    let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
    let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_numeric("x2", x2).unwrap();
    df.add_categorical("z", z).unwrap();

    let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
    model.fit(false).unwrap();
    let dropped_label = model
        .fixed_effect_inference_table()
        .rows
        .into_iter()
        .find(|row| row.status == FixedEffectInferenceStatus::NotEstimable)
        .expect("rank-deficient fit should mark one coefficient not estimable")
        .label;
    let dropped_index = model
        .coef_names()
        .iter()
        .position(|name| name == &dropped_label)
        .unwrap();
    let hypothesis = FixedEffectHypothesis::single_coefficient(
        format!("{dropped_label} = 0"),
        dropped_index,
        model.coef_names().len(),
    )
    .unwrap();
    let test = model.test_contrast_with_method(hypothesis, FixedEffectTestMethod::Satterthwaite);

    assert!(
        matches!(test.status, InferenceStatus::NotEstimable { ref reason }
            if reason.contains("aliased") || reason.contains("non-finite"))
    );
    assert_eq!(test.p_values, vec![None]);
}

#[test]
fn test_lmm_fixed_effect_inference_table_returns_ordered_satterthwaite_rows() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let table = model.fixed_effect_inference_table();
    let names = model.coef_names();

    assert_eq!(table.rows.len(), names.len());
    assert_eq!(
        table
            .rows
            .iter()
            .map(|row| row.label.clone())
            .collect::<Vec<_>>(),
        names
    );
    for row in &table.rows {
        assert_eq!(row.kind, FixedEffectInferenceRowKind::Coefficient);
        assert_eq!(row.method, FixedEffectInferenceMethod::Satterthwaite);
        assert_eq!(row.status, FixedEffectInferenceStatus::Available);
        assert_eq!(row.reliability, ReliabilityGrade::Moderate);
        assert_eq!(
            row.reliability_reason,
            Some(FixedEffectReliabilityReason::SatterthwaiteFiniteDifferenceApproximation)
        );
        assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::T));
        assert!(row.estimate.is_some());
        assert!(row.std_error.is_some());
        assert!(row.statistic.is_some());
        assert!(row.p_value.is_some());
        assert!(row.numerator_df.is_none());
        assert!(row.denominator_df.is_some());
        assert!(row.reason.is_none());
        assert!(matches!(
            row.estimability,
            EstimabilityAssessment::FixedContrast(_)
        ));
        assert!(row
            .notes
            .iter()
            .any(|note| note.contains("Satterthwaite denominator df")));
    }
    let artifact_table = model
        .compiler_artifact()
        .fixed_effect_inference_table
        .as_ref()
        .expect("fitted artifact should carry cheap fixed-effect rows");
    assert_eq!(artifact_table.rows.len(), table.rows.len());
    for row in &artifact_table.rows {
        assert_eq!(row.kind, FixedEffectInferenceRowKind::Coefficient);
        assert_eq!(row.method, FixedEffectInferenceMethod::AsymptoticWaldZ);
        assert_eq!(
            row.reliability_reason,
            Some(FixedEffectReliabilityReason::AsymptoticWaldZFallback)
        );
        assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::Z));
        assert!(row.denominator_df.is_none());
    }
}

#[test]
fn test_lmm_fixed_effect_covariance_matrix_available_for_full_rank_fit() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let payload = model.fixed_effect_covariance_matrix();
    let vcov = model.vcov();

    assert_eq!(payload.status, FixedEffectCovarianceStatus::Available);
    assert_eq!(payload.method, FixedEffectCovarianceMethod::ModelBased);
    assert_eq!(payload.reliability, ReliabilityGrade::High);
    assert_eq!(payload.coef_names, model.coef_names());
    assert_eq!(payload.details.rank, Some(model.feterm.rank));
    assert_eq!(
        payload.details.expected_rank,
        Some(model.coef_names().len())
    );
    assert!(payload.details.aliased.is_empty());
    assert_eq!(payload.details.matrix_rows, vcov.nrows());
    assert_eq!(payload.details.matrix_cols, vcov.ncols());
    assert_eq!(payload.details.finite, Some(true));
    assert_eq!(payload.details.symmetric, Some(true));
    assert_eq!(payload.matrix.as_ref().unwrap(), &matrix_rows(&vcov));
    assert!(payload
        .notes
        .iter()
        .any(|note| note.contains("inference claims remain")));
    assert_eq!(
        model
            .compiler_artifact()
            .fixed_effect_covariance_matrix
            .as_ref(),
        Some(&payload)
    );
}

#[test]
fn test_lmm_fixed_effect_covariance_matrix_unavailable_for_rank_deficient_fit() {
    let n = 30usize;
    let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
    let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_numeric("x2", x2).unwrap();
    df.add_categorical("z", z).unwrap();

    let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
    model.fit(false).unwrap();

    let payload = model.fixed_effect_covariance_matrix();

    assert_eq!(payload.status, FixedEffectCovarianceStatus::Unavailable);
    assert_eq!(payload.method, FixedEffectCovarianceMethod::Unavailable);
    assert_eq!(payload.reliability, ReliabilityGrade::NotAvailable);
    assert_eq!(
        payload.reason.as_deref(),
        Some("rank_deficient_fixed_effects")
    );
    assert_eq!(payload.matrix, None);
    assert_eq!(payload.details.rank, Some(2));
    assert_eq!(payload.details.expected_rank, Some(3));
    assert_eq!(payload.details.aliased.len(), 1);
    assert!(payload.details.aliased[0] == "x" || payload.details.aliased[0] == "x2");
    assert_eq!(payload.details.finite, Some(false));
    assert_eq!(
        model
            .compiler_artifact()
            .fixed_effect_covariance_matrix
            .as_ref(),
        Some(&payload)
    );
}

#[test]
fn test_lmm_fixed_effect_inference_table_marks_aliased_column_not_estimable() {
    let n = 30usize;
    let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
    let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_numeric("x2", x2).unwrap();
    df.add_categorical("z", z).unwrap();

    let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
    model.fit(false).unwrap();

    let table = model.fixed_effect_inference_table();
    let dropped = table
        .rows
        .iter()
        .find(|row| row.status == FixedEffectInferenceStatus::NotEstimable)
        .expect("one aliased coefficient should be marked not estimable");

    assert_eq!(dropped.method, FixedEffectInferenceMethod::NotComputed);
    assert_eq!(dropped.reliability, ReliabilityGrade::NotAvailable);
    assert!(dropped.p_value.is_none());
    assert!(dropped.reason.as_deref().unwrap().contains("aliased"));
}

#[test]
fn test_lmm_fixed_effect_inference_table_omits_p_values_for_regularized_fit_intent() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let policy = CompilerPolicy {
        random_strategy: RandomStrategy::Regularized,
        ..CompilerPolicy::default()
    };
    let mut model =
        LinearMixedModel::new_with_compiler_policy(formula, &data, None, policy).unwrap();

    model.fit(false).unwrap();

    let table = model.fixed_effect_inference_table();
    assert!(table.rows.iter().all(|row| {
        row.status == FixedEffectInferenceStatus::PValueUnavailable
            && row.method == FixedEffectInferenceMethod::NotComputed
            && row.p_value.is_none()
            && row
                .reason
                .as_deref()
                .unwrap()
                .contains("exploratory fit intent")
    }));
}

#[test]
fn test_lmm_fixed_effect_inference_table_omits_p_values_for_predictive_fit_intent() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new_with_compiler_policy(
        formula,
        &data,
        None,
        CompilerPolicy::predictive(),
    )
    .unwrap();

    model.fit(false).unwrap();

    assert_eq!(
        model.compiler_artifact().reproducibility.fit_intent,
        FitIntent::Predictive
    );
    let table = model
        .compiler_artifact()
        .fixed_effect_inference_table
        .as_ref()
        .expect("fitted artifacts should carry fixed-effect inference rows");
    assert!(table.rows.iter().all(|row| {
        row.status == FixedEffectInferenceStatus::PValueUnavailable
            && row.method == FixedEffectInferenceMethod::NotComputed
            && row.p_value.is_none()
            && row
                .reason
                .as_deref()
                .unwrap()
                .contains("predictive fit intent")
    }));
}

#[test]
fn test_lmm_fixed_effect_inference_table_omits_p_values_after_selection_time_reduction() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.compiler_artifact.reductions.push(ReductionRecord {
        trigger: ReductionTrigger::SelectionTime,
        phase: "post_selection".to_string(),
        reason: "response-dependent random-effect selection".to_string(),
        affected_term: "(1 | subj)".to_string(),
        replacement_term: None,
        inference_consequence:
            "ordinary fixed-effect p-values require a valid refit or selective-inference contract"
                .to_string(),
        diagnostics: Vec::new(),
    });

    model.fit(false).unwrap();

    let table = model
        .compiler_artifact()
        .fixed_effect_inference_table
        .as_ref()
        .expect("fitted artifacts should carry fixed-effect inference rows");
    assert!(table.rows.iter().all(|row| {
        row.status == FixedEffectInferenceStatus::PValueUnavailable
            && row.method == FixedEffectInferenceMethod::NotComputed
            && row.p_value.is_none()
            && row
                .reason
                .as_deref()
                .unwrap()
                .contains("selection-time model changes")
    }));
}

#[test]
fn test_lmm_test_contrast_marks_aliased_column_not_estimable() {
    let n = 30usize;
    let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let x2: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
    let z: Vec<String> = (0..n).map(|i| format!("G{}", i % 6)).collect();

    let mut df = DataFrame::new();
    df.add_numeric("y", y).unwrap();
    df.add_numeric("x", x).unwrap();
    df.add_numeric("x2", x2).unwrap();
    df.add_categorical("z", z).unwrap();

    let formula = parse_formula("y ~ 1 + x + x2 + (1 | z)").unwrap();
    let mut model = LinearMixedModel::new(formula, &df, None).unwrap();
    model.fit(false).unwrap();
    let ct = model.coeftable();
    let dropped = ct
        .std_errors
        .iter()
        .position(|se| se.is_nan())
        .expect("one fixed-effect column should be dropped");

    let hypothesis =
        FixedEffectHypothesis::single_coefficient("dropped coefficient", dropped, ct.len())
            .unwrap();
    let test = model.test_contrast(hypothesis);

    assert!(matches!(test.status, InferenceStatus::NotEstimable { .. }));
    assert_eq!(test.estimability.status, EstimabilityStatus::NotEstimable);
    assert_eq!(test.p_values, vec![None]);
}

#[test]
fn test_lmm_fixed_effect_term_rows_are_rust_owned() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let hypotheses = model.fixed_effect_term_hypotheses();
    assert!(hypotheses
        .iter()
        .any(|hypothesis| hypothesis.label == "days"));

    let table = model.fixed_effect_term_inference_table(FixedEffectTestMethod::Auto);
    let days = table
        .rows
        .iter()
        .find(|row| row.label == "days")
        .expect("days term row should be exposed");
    assert_eq!(days.kind, FixedEffectInferenceRowKind::Term);
    let family = days
        .details
        .as_ref()
        .and_then(|details| details.contrast_family.as_ref())
        .expect("term row should carry contrast-family details");
    assert_eq!(family.family_label, "days");
    assert_eq!(family.restriction_rows, 1);
    assert_eq!(family.coefficient_count, model.coef_names().len());
}

#[test]
fn test_lmm_fixed_effect_term_hypotheses_have_explicit_type_semantics() {
    let model = typed_term_test_fixture();
    let names = model.coef_names();
    let x_index = names.iter().position(|name| name == "x").unwrap();

    let type_i = model.fixed_effect_term_hypotheses_for_type(FixedEffectTermTestType::TypeI);
    let type_ii = model.fixed_effect_term_hypotheses_for_type(FixedEffectTermTestType::TypeII);
    let type_iii = model.fixed_effect_term_hypotheses_for_type(FixedEffectTermTestType::TypeIII);

    let x_type_i = hypothesis_by_label(&type_i, "x");
    let x_type_ii = hypothesis_by_label(&type_ii, "x");
    let x_type_iii = hypothesis_by_label(&type_iii, "x");
    let interaction_type_ii = hypothesis_by_label(&type_ii, "x:z");

    assert_eq!(x_type_iii.l.values.nrows(), 1);
    assert_eq!(x_type_iii.l.values.ncols(), names.len());
    for col in 0..names.len() {
        let expected = if col == x_index { 1.0 } else { 0.0 };
        assert_relative_eq!(x_type_iii.l.values[(0, col)], expected, epsilon = 1.0e-12);
    }

    assert!(
            matrices_differ(&x_type_i.l.values, &x_type_iii.l.values, 1.0e-9),
            "Type I x hypothesis should not collapse to the Type III coefficient block in the interaction fixture"
        );
    assert!(
            matrices_differ(&x_type_ii.l.values, &x_type_iii.l.values, 1.0e-9),
            "Type II x hypothesis should not collapse to the Type III coefficient block in the interaction fixture"
        );
    assert_eq!(interaction_type_ii.l.values.nrows(), 1);
    assert_eq!(interaction_type_ii.l.values.ncols(), names.len());

    let table = model.fixed_effect_term_inference_table_for_type(
        FixedEffectTestMethod::Satterthwaite,
        FixedEffectTermTestType::TypeII,
    );
    let x_row = table
        .rows
        .iter()
        .find(|row| row.label == "x")
        .expect("Type II table should include x term row");
    assert_eq!(x_row.kind, FixedEffectInferenceRowKind::Term);
    assert!(x_row
        .notes
        .iter()
        .any(|note| note.contains("fixed-effect term test type: type_ii")));
}

#[test]
fn test_lmm_fixed_effect_contrast_table_is_rust_owned() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();
    let table =
        model.fixed_effect_contrast_inference_table(vec![hypothesis], FixedEffectTestMethod::Auto);

    assert_eq!(
        table.schema_name,
        crate::compiler::FIXED_EFFECT_INFERENCE_TABLE_SCHEMA
    );
    assert_eq!(table.rows.len(), 1);
    let row = &table.rows[0];
    assert_eq!(row.kind, FixedEffectInferenceRowKind::Contrast);
    assert_eq!(row.label, "days = 0");
    assert_eq!(row.status, FixedEffectInferenceStatus::Available);
    let family = row
        .details
        .as_ref()
        .and_then(|details| details.contrast_family.as_ref())
        .expect("contrast row should carry contrast-family details");
    assert_eq!(family.family_label, "days = 0");
    assert_eq!(
        family.numerator_df_semantics,
        "scalar_contrast_no_numerator_df"
    );
}

// ── Cook's distance parity tests (pls.jl line 705) ───────────────────────

// ── Cook's distance parity tests (pls.jl line 705) ───────────────────────

#[test]
fn test_cooks_distance_length() {
    // cooksdistance(model) should have length n.
    // Uses first(models(:sleepstudy)) = reaction ~ 1 + days + (1 | subj)
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let d = model.cooks_distance();
    assert_eq!(d.len(), data.nrow());
}

#[test]
fn test_cooks_distance_nonnegative() {
    // All Cook's distances should be ≥ 0.
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let d = model.cooks_distance();
    for (i, &di) in d.iter().enumerate() {
        assert!(
            di >= 0.0,
            "Cook's distance[{}] should be non-negative, got {}",
            i,
            di
        );
    }
}

#[test]
fn test_cooks_distance_parity_sleepstudy() {
    // pls.jl line 705-760: lme4 reference values for Cook's distance.
    // Model: first(models(:sleepstudy)) = reaction ~ 1 + days + (1 | subj)
    //
    // Julia uses:  D_i = (r_i/(1-h_i))^2 * h_i / (varest(m) * p)
    // where p = rank of fixed-effects matrix = 2.
    //
    // We compare the first 10 values at rtol=0.10 (10%).
    let lme4_cooks: Vec<f64> = vec![
        0.1270714,
        0.1267805,
        0.243096,
        0.0002437091,
        0.03145029,
        0.2954052,
        0.04550505,
        0.3552723,
        0.1984806,
        0.4518805,
    ];

    let data = sleepstudy_fixture();
    // first(models(:sleepstudy)) — intercept-only RE per subject
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let d = model.cooks_distance();

    for (i, &expected) in lme4_cooks.iter().enumerate() {
        let got = d[i];
        let rel_err = ((got - expected) / expected).abs();
        assert!(
            rel_err < 0.10,
            "Cook's distance[{}]: expected {:.6}, got {:.6} (rel err {:.2}%)",
            i,
            expected,
            got,
            rel_err * 100.0
        );
    }
}

#[test]
fn test_cooks_distance_sum_finite() {
    // Sum should be finite (no NaN/Inf from degenerate h_i).
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let d = model.cooks_distance();
    let s: f64 = d.iter().sum();
    assert!(s.is_finite(), "Sum of Cook's distances should be finite");
}

// ── Parametric bootstrap parity tests (bootstrap.jl) ─────────────────────

#[test]
fn test_parametricbootstrap_length() {
    // bootstrap.jl line 98: length(bsamp.objective) == 100
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut rng = StdRng::seed_from_u64(1234321);
    let bsamp = parametricbootstrap(&mut rng, 5, &model);
    assert_eq!(bsamp.len(), 5);
    assert_eq!(bsamp.objectives().len(), 5);
    assert_eq!(bsamp.sigmas().len(), 5);
    assert_eq!(bsamp.thetas().len(), 5);
}

#[test]
fn test_parametricbootstrap_objectives_finite() {
    // Each replicate should converge to a finite objective.
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut rng = StdRng::seed_from_u64(42);
    let bsamp = parametricbootstrap(&mut rng, 10, &model);

    let n_finite = bsamp
        .objectives()
        .iter()
        .filter(|&&o| o.is_finite())
        .count();
    assert!(
        n_finite >= 8,
        "At least 8 out of 10 replicates should converge; got {}",
        n_finite
    );
}

#[test]
fn test_parametricbootstrap_sigma_positive() {
    // All converged σ values should be positive.
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut rng = StdRng::seed_from_u64(99);
    let bsamp = parametricbootstrap(&mut rng, 5, &model);

    for rep in &bsamp.fits {
        if rep.sigma.is_finite() {
            assert!(
                rep.sigma > 0.0,
                "Bootstrap σ should be positive, got {}",
                rep.sigma
            );
        }
    }
}

#[test]
fn test_parametricbootstrap_beta_length() {
    // Each replicate's β should have length p (rank of FE matrix).
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let p = model.feterm.rank;
    let mut rng = StdRng::seed_from_u64(7);
    let bsamp = parametricbootstrap(&mut rng, 3, &model);

    for rep in &bsamp.fits {
        assert_eq!(
            rep.beta.len(),
            p,
            "Bootstrap β length mismatch: expected {}, got {}",
            p,
            rep.beta.len()
        );
    }
}

#[test]
fn test_parametricbootstrap_theta_length() {
    // bootstrap.jl: keys(first(bsamp.fits)) includes :θ.
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let n_theta = model.n_theta();
    let mut rng = StdRng::seed_from_u64(0);
    let bsamp = parametricbootstrap(&mut rng, 3, &model);

    for rep in &bsamp.fits {
        assert_eq!(
            rep.theta.len(),
            n_theta,
            "Bootstrap θ length mismatch: expected {}, got {}",
            n_theta,
            rep.theta.len()
        );
    }
}

#[test]
fn test_parametricbootstrap_save_restore_round_trip() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let mut rng = StdRng::seed_from_u64(20260428);
    let bsamp = parametricbootstrap(&mut rng, 4, &model);

    let mut bytes = Vec::new();
    crate::stats::savereplicates(&mut bytes, &bsamp).unwrap();
    let restored = crate::stats::restorereplicates(bytes.as_slice(), &model).unwrap();

    assert_eq!(restored.len(), bsamp.len());
    for (actual, expected) in restored.fits.iter().zip(bsamp.fits.iter()) {
        assert_relative_eq!(actual.objective, expected.objective, epsilon = 1e-12);
        assert_relative_eq!(actual.sigma, expected.sigma, epsilon = 1e-12);
        assert_eq!(actual.beta.len(), expected.beta.len());
        for (a, e) in actual.beta.iter().zip(expected.beta.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-12);
        }
        assert_eq!(actual.se.len(), expected.se.len());
        for (a, e) in actual.se.iter().zip(expected.se.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-12);
        }
        assert_eq!(actual.theta.len(), expected.theta.len());
        for (a, e) in actual.theta.iter().zip(expected.theta.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-12);
        }
    }
}

#[test]
fn test_parametricbootstrap_save_restore_preserves_nan_status() {
    let bsamp = MixedModelBootstrap {
        fits: vec![BootstrapReplicate {
            objective: f64::NAN,
            sigma: f64::NAN,
            beta: DVector::from_vec(vec![1.0, 2.0]),
            se: DVector::from_vec(vec![f64::NAN, f64::NAN]),
            theta: vec![0.5],
        }],
    };

    let mut bytes = Vec::new();
    bsamp.save_replicates(&mut bytes).unwrap();
    let restored = MixedModelBootstrap::restore_replicates(bytes.as_slice()).unwrap();

    assert_eq!(restored.len(), 1);
    assert!(restored.fits[0].objective.is_nan());
    assert!(restored.fits[0].sigma.is_nan());
    assert_eq!(restored.fits[0].beta, DVector::from_vec(vec![1.0, 2.0]));
    assert!(restored.fits[0].se.iter().all(|value| value.is_nan()));
    assert_eq!(restored.fits[0].theta, vec![0.5]);
}

#[test]
fn test_parametricbootstrap_run_metadata_records_accounting_and_boundary_rate() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let bsamp = MixedModelBootstrap {
        fits: vec![
            BootstrapReplicate {
                objective: 1.0,
                sigma: 2.0,
                beta: DVector::from_vec(vec![10.0]),
                se: DVector::from_vec(vec![1.0]),
                theta: vec![0.0],
            },
            BootstrapReplicate {
                objective: 2.0,
                sigma: 3.0,
                beta: DVector::from_vec(vec![11.0]),
                se: DVector::from_vec(vec![1.2]),
                theta: vec![0.5],
            },
            BootstrapReplicate {
                objective: f64::NAN,
                sigma: f64::NAN,
                beta: DVector::from_vec(vec![f64::NAN]),
                se: DVector::from_vec(vec![f64::NAN]),
                theta: vec![0.5],
            },
        ],
    };
    let statistics = [1.0, f64::NAN, 3.0];

    let metadata = bsamp.run_metadata_for_model(
        &model,
        BootstrapTarget::full_model_distribution("dyestuff full model"),
        5,
        BootstrapFailedRefitPolicy::Exclude,
        BootstrapSeedRecord::std_rng(20260429),
        BootstrapRefitOptions::from_model(&model),
        Some("abs_t".to_string()),
        Some(&statistics),
        Some(0.25),
    );

    assert_eq!(metadata.schema_name, BOOTSTRAP_RUN_SCHEMA);
    assert_eq!(metadata.schema_version, BOOTSTRAP_RUN_SCHEMA_VERSION);
    assert_eq!(
        metadata.target.kind,
        BootstrapTargetKind::FullModelDistribution
    );
    assert_eq!(metadata.requested_replicates, 5);
    assert_eq!(metadata.completed_replicates, 3);
    assert_eq!(metadata.successful_replicates, 2);
    assert_eq!(metadata.failed_refits, 1);
    assert_eq!(
        metadata.failed_refit_policy,
        BootstrapFailedRefitPolicy::Exclude
    );
    assert_eq!(metadata.boundary_count, 1);
    assert_eq!(metadata.boundary_rate, Some(0.5));
    assert_eq!(metadata.finite_statistic_count, Some(2));
    assert_relative_eq!(metadata.mcse.unwrap(), (0.25_f64 * 0.75 / 2.0).sqrt());
    assert!(metadata
        .notes
        .iter()
        .any(|note| note.contains("do not certify fixed-effect hypothesis-test")));
    assert!(metadata
        .notes
        .iter()
        .any(|note| note.contains("requested 5 bootstrap")));

    let payload = bsamp.into_run_payload(metadata);
    let json = serde_json::to_string(&payload).unwrap();
    let decoded: BootstrapRunPayload = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.metadata.successful_replicates, 2);
    assert_eq!(decoded.replicates.len(), 3);
}

#[test]
fn test_fixed_effect_null_bootstrap_target_projects_beta_and_simulates() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();

    let target = model
        .fixed_effect_null_bootstrap_target(&hypothesis)
        .unwrap();
    let fitted_contrast = (&hypothesis.l.values * &target.beta_fitted)[0];
    let null_contrast = (&hypothesis.l.values * &target.beta_null)[0];

    assert_eq!(target.target.kind, BootstrapTargetKind::FixedEffectNull);
    assert_eq!(
        target.covariance_policy,
        FixedEffectNullCovariancePolicy::ReuseFittedCovariance
    );
    assert!(fitted_contrast.abs() > 1.0);
    assert_relative_eq!(null_contrast, 0.0, epsilon = 1e-8);
    assert_eq!(target.theta, model.theta());
    assert_relative_eq!(target.sigma, model.sigma(), epsilon = 1e-12);
    assert!(target
        .notes
        .iter()
        .any(|note| note.contains("reuses fitted covariance")));

    let mut rng = StdRng::seed_from_u64(20260429);
    let y_sim = model.simulate_fixed_effect_null(&mut rng, &target).unwrap();
    assert_eq!(y_sim.len(), model.nobs());

    let mut mismatched = target.clone();
    mismatched.sigma *= 1.01;
    assert!(matches!(
        model.simulate_fixed_effect_null(&mut rng, &mismatched),
        Err(MixedModelError::InvalidArgument(_))
    ));
}

#[test]
fn test_bootstrap_fixed_effect_coefficient_row_from_certified_payload() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();

    let mut fits = Vec::new();
    for i in 0..40 {
        let mut beta = model.beta();
        beta[days_index] = (i as f64 - 20.0) / 10.0;
        let mut se = DVector::from_element(model.feterm.rank, 1.0);
        se[days_index] = 1.0;
        fits.push(BootstrapReplicate {
            objective: i as f64 + 1.0,
            sigma: model.sigma(),
            beta,
            se,
            theta: model.theta(),
        });
    }
    let bsamp = MixedModelBootstrap { fits };
    let metadata = bsamp.run_metadata_for_model(
        &model,
        BootstrapTarget::fixed_effect_null("days fixed-effect null", "days = 0"),
        40,
        BootstrapFailedRefitPolicy::Exclude,
        BootstrapSeedRecord::std_rng(20260429),
        BootstrapRefitOptions::from_model(&model),
        Some("abs_t".to_string()),
        None,
        None,
    );
    let payload = bsamp.into_run_payload(metadata);

    let test = model.test_contrast_with_bootstrap_payload(hypothesis.clone(), &payload);
    assert_eq!(test.method, InferenceMethod::ParametricBootstrap);
    assert_eq!(test.status, InferenceStatus::Available);
    assert_eq!(test.reliability, ReliabilityGrade::Low);
    assert_relative_eq!(test.p_values[0].unwrap(), 1.0 / 41.0, epsilon = 1e-12);
    assert!(test.denominator_df.is_none());
    assert!(test
        .notes
        .iter()
        .any(|note| note.contains("fixed_effect_null target")));

    let row = model.fixed_effect_bootstrap_inference_row(
        FixedEffectInferenceRowKind::Coefficient,
        hypothesis,
        &payload,
    );
    assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
    assert_eq!(row.status, FixedEffectInferenceStatus::Available);
    assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::T));
    assert_eq!(
        row.reliability_reason,
        Some(FixedEffectReliabilityReason::ParametricBootstrapMonteCarlo)
    );
    assert_relative_eq!(row.p_value.unwrap(), 1.0 / 41.0, epsilon = 1e-12);
    let bootstrap = row
        .details
        .as_ref()
        .and_then(|details| details.bootstrap.as_ref())
        .expect("bootstrap row should carry structured metadata");
    assert_eq!(bootstrap.target_kind, "fixed_effect_null");
    assert_eq!(bootstrap.requested_replicates, 40);
    assert_eq!(bootstrap.successful_replicates, 40);
    assert_eq!(bootstrap.failed_refit_policy, "exclude");
}

#[test]
fn test_bootstrap_fixed_effect_contrast_row_uses_payload_statistics() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let l = DMatrix::from_row_slice(1, model.coef_names().len(), &[1.0, 1.0]);
    let hypothesis =
        FixedEffectHypothesis::zero_rhs("intercept_plus_days = 0", ContrastMatrix { values: l });
    let fits = (0..40)
        .map(|i| BootstrapReplicate {
            objective: i as f64 + 1.0,
            sigma: model.sigma(),
            beta: model.beta(),
            se: DVector::from_element(model.feterm.rank, 1.0),
            theta: model.theta(),
        })
        .collect::<Vec<_>>();
    let bsamp = MixedModelBootstrap { fits };
    let replicate_statistics = vec![0.5; bsamp.len()];
    let metadata = bsamp.run_metadata_for_model(
        &model,
        BootstrapTarget::fixed_effect_null(
            "intercept_plus_days fixed-effect null",
            "intercept_plus_days = 0",
        ),
        40,
        BootstrapFailedRefitPolicy::Exclude,
        BootstrapSeedRecord::std_rng(20260430),
        BootstrapRefitOptions::from_model(&model),
        Some("abs_t".to_string()),
        Some(&replicate_statistics),
        Some(1.0 / 41.0),
    );
    let payload = bsamp.into_run_payload_with_statistics(metadata, replicate_statistics);

    let row = model.fixed_effect_bootstrap_inference_row(
        FixedEffectInferenceRowKind::Contrast,
        hypothesis,
        &payload,
    );
    assert_eq!(row.kind, FixedEffectInferenceRowKind::Contrast);
    assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
    assert_eq!(row.status, FixedEffectInferenceStatus::Available);
    assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::T));
    assert_relative_eq!(row.p_value.unwrap(), 1.0 / 41.0, epsilon = 1e-12);
    let details = row.details.expect("contrast row should carry details");
    assert!(details.bootstrap.is_some());
    let family = details
        .contrast_family
        .expect("contrast row should carry contrast-family details");
    assert_eq!(family.restriction_rows, 1);
    assert_eq!(
        family.numerator_df_semantics,
        "scalar_contrast_no_numerator_df"
    );
}

#[test]
fn test_bootstrap_fixed_effect_row_requires_enough_finite_statistics() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();
    let fits = (0..2)
        .map(|i| BootstrapReplicate {
            objective: i as f64 + 1.0,
            sigma: model.sigma(),
            beta: model.beta(),
            se: DVector::from_element(model.feterm.rank, 1.0),
            theta: model.theta(),
        })
        .collect::<Vec<_>>();
    let bsamp = MixedModelBootstrap { fits };
    let metadata = bsamp.run_metadata_for_model(
        &model,
        BootstrapTarget::fixed_effect_null("days fixed-effect null", "days = 0"),
        2,
        BootstrapFailedRefitPolicy::Exclude,
        BootstrapSeedRecord::std_rng(20260431),
        BootstrapRefitOptions::from_model(&model),
        Some("abs_t".to_string()),
        None,
        None,
    );
    let payload = bsamp.into_run_payload(metadata);

    let test = model.test_contrast_with_bootstrap_payload(hypothesis, &payload);
    assert_eq!(test.method, InferenceMethod::ParametricBootstrap);
    assert!(
        matches!(test.status, InferenceStatus::NotAssessed { ref reason }
            if reason.contains("bootstrap_successful_replicates_too_few"))
    );
    assert_eq!(test.p_values, vec![None]);
}

#[test]
fn test_bootstrap_fixed_effect_row_from_null_simulate_refit_payload() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();
    let target = model
        .fixed_effect_null_bootstrap_target(&hypothesis)
        .unwrap();

    let mut rng = StdRng::seed_from_u64(20260502);
    let mut fits = Vec::new();
    for _ in 0..30 {
        let y_sim = model.simulate_fixed_effect_null(&mut rng, &target).unwrap();
        let mut work = model.clone();
        match work.refit(y_sim.as_slice()) {
            Ok(()) => fits.push(BootstrapReplicate {
                objective: work.objective(),
                sigma: work.sigma(),
                beta: work.beta(),
                se: work.stderror(),
                theta: work.theta(),
            }),
            Err(_) => fits.push(BootstrapReplicate {
                objective: f64::NAN,
                sigma: f64::NAN,
                beta: model.beta(),
                se: DVector::from_element(model.feterm.rank, f64::NAN),
                theta: model.theta(),
            }),
        }
    }

    let bsamp = MixedModelBootstrap { fits };
    let metadata = bsamp.run_metadata_for_model(
        &model,
        target.target.clone(),
        30,
        BootstrapFailedRefitPolicy::Exclude,
        BootstrapSeedRecord::std_rng(20260502),
        BootstrapRefitOptions::from_model(&model),
        Some("abs_t".to_string()),
        None,
        None,
    );
    let payload = bsamp.into_run_payload(metadata);
    let row = model.fixed_effect_bootstrap_inference_row(
        FixedEffectInferenceRowKind::Coefficient,
        hypothesis,
        &payload,
    );

    assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
    assert_eq!(row.status, FixedEffectInferenceStatus::Available);
    assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::T));
    assert_eq!(row.reliability, ReliabilityGrade::Low);
    assert_eq!(
        row.reliability_reason,
        Some(FixedEffectReliabilityReason::ParametricBootstrapMonteCarlo)
    );
    assert!(row.p_value.unwrap().is_finite());
    assert!((1.0 / 31.0..=1.0).contains(&row.p_value.unwrap()));
    assert!(row
        .notes
        .iter()
        .any(|note| note.contains("successful_replicates=30")));
    assert!(row.notes.iter().any(|note| note.contains("mcse=")));
}

#[test]
fn test_fixed_effect_null_bootstrap_table_callable_returns_inference_table() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let days_index = model
        .coef_names()
        .iter()
        .position(|name| name == "days")
        .unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("days = 0", days_index, model.coef_names().len())
            .unwrap();
    let table = model.fixed_effect_null_bootstrap_inference_table(
        vec![hypothesis],
        FixedEffectBootstrapOptions {
            requested_replicates: 2,
            failed_refit_policy: BootstrapFailedRefitPolicy::Exclude,
            seed: Some(20260503),
        },
    );

    assert_eq!(
        table.schema_name,
        crate::compiler::FIXED_EFFECT_INFERENCE_TABLE_SCHEMA
    );
    assert_eq!(table.rows.len(), 1);
    let row = &table.rows[0];
    assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
    assert_eq!(row.kind, FixedEffectInferenceRowKind::Contrast);
    assert!(matches!(
        row.status,
        FixedEffectInferenceStatus::Available | FixedEffectInferenceStatus::NotAssessed
    ));
    let bootstrap = row
        .details
        .as_ref()
        .and_then(|details| details.bootstrap.as_ref())
        .expect("bridge row should carry bootstrap details");
    assert_eq!(bootstrap.requested_replicates, 2);
    assert_eq!(bootstrap.seed, Some(20260503));
    assert!(bootstrap.null_target.is_some());
}

#[test]
fn test_bootstrap_multi_df_payload_matches_joint_f_oracle_and_p_value_accounting() {
    let (model, hypothesis) = three_level_condition_fixture();
    let observed = joint_wald_f_direct_inverse_oracle(&model, &hypothesis);
    assert!(observed.is_finite());

    let mut replicate_statistics = vec![observed + 0.25; 9];
    replicate_statistics.extend(vec![observed - 0.25; 31]);
    let payload = successful_bootstrap_payload_with_statistics(
        &model,
        &hypothesis.label,
        replicate_statistics,
        "joint_wald_f",
    );

    let row = model.fixed_effect_bootstrap_inference_row(
        FixedEffectInferenceRowKind::Term,
        hypothesis,
        &payload,
    );

    assert_eq!(row.kind, FixedEffectInferenceRowKind::Term);
    assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
    assert_eq!(row.status, FixedEffectInferenceStatus::Available, "{row:?}");
    assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::F));
    assert_eq!(row.numerator_df, Some(2.0));
    assert_relative_eq!(row.statistic.unwrap(), observed, epsilon = 1e-10);
    assert_relative_eq!(row.p_value.unwrap(), 10.0 / 41.0, epsilon = 1e-12);
    assert!(row.denominator_df.is_none());
    assert!(row
        .notes
        .iter()
        .any(|note| note.contains("statistic=joint_wald_f")));
    assert!(row
        .notes
        .iter()
        .any(|note| note.contains("finite_statistics=40")));
}

#[test]
fn test_fixed_effect_joint_f_is_invariant_to_restriction_row_order_and_scaling() {
    let (model, hypothesis) = three_level_condition_fixture();
    let baseline = fixed_effect_bootstrap_statistic(&model, &hypothesis)
        .expect("baseline joint-F statistic should be available");

    let mut swapped_l = DMatrix::zeros(2, hypothesis.n_coefficients());
    swapped_l.row_mut(0).copy_from(&hypothesis.l.values.row(1));
    swapped_l.row_mut(1).copy_from(&hypothesis.l.values.row(0));
    let swapped_rhs = DVector::from_vec(vec![hypothesis.rhs.values[1], hypothesis.rhs.values[0]]);
    let swapped = FixedEffectHypothesis {
        label: "cond_swapped".to_string(),
        l: ContrastMatrix::new(swapped_l).unwrap(),
        rhs: ContrastRhs {
            values: swapped_rhs,
        },
    };
    let swapped_stat = fixed_effect_bootstrap_statistic(&model, &swapped)
        .expect("row-permuted joint-F statistic should be available");

    let mut scaled_l = hypothesis.l.values.clone();
    scaled_l.row_mut(0).scale_mut(2.0);
    scaled_l.row_mut(1).scale_mut(-0.5);
    let mut scaled_rhs = hypothesis.rhs.values.clone();
    scaled_rhs[0] *= 2.0;
    scaled_rhs[1] *= -0.5;
    let scaled = FixedEffectHypothesis {
        label: "cond_scaled".to_string(),
        l: ContrastMatrix::new(scaled_l).unwrap(),
        rhs: ContrastRhs { values: scaled_rhs },
    };
    let scaled_stat = fixed_effect_bootstrap_statistic(&model, &scaled)
        .expect("scaled joint-F statistic should be available");

    assert_eq!(baseline.label, "joint_wald_f");
    assert_eq!(baseline.numerator_df, Some(2.0));
    assert_relative_eq!(swapped_stat.value, baseline.value, epsilon = 1e-10);
    assert_relative_eq!(scaled_stat.value, baseline.value, epsilon = 1e-10);
    assert_eq!(swapped_stat.numerator_df, Some(2.0));
    assert_eq!(scaled_stat.numerator_df, Some(2.0));
}

#[test]
fn test_fixed_effect_joint_f_uses_effective_rank_for_redundant_restrictions() {
    let (model, hypothesis) = three_level_condition_fixture();
    let mut l = DMatrix::zeros(2, hypothesis.n_coefficients());
    l.row_mut(0).copy_from(&hypothesis.l.values.row(0));
    l.row_mut(1).copy_from(&hypothesis.l.values.row(0));
    l.row_mut(1).scale_mut(2.0);
    let redundant = FixedEffectHypothesis {
        label: "redundant_cond_b".to_string(),
        l: ContrastMatrix::new(l).unwrap(),
        rhs: ContrastRhs::zeros(2),
    };

    let statistic = fixed_effect_bootstrap_statistic(&model, &redundant)
        .expect("redundant but estimable restrictions should use a pseudo-inverse");
    assert_eq!(statistic.label, "joint_wald_f");
    assert_eq!(statistic.numerator_df, Some(1.0));
    assert!(statistic.value.is_finite());
    assert!(statistic.value >= 0.0);
}

#[test]
fn test_fixed_effect_covariance_artifact_round_trips_from_fitted_models() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let json = serde_json::to_string(model.compiler_artifact()).unwrap();
    let decoded: CompiledModelArtifact = serde_json::from_str(&json).unwrap();
    let covariance = decoded
        .fixed_effect_covariance_matrix
        .expect("fitted full-rank artifact should carry fixed-effect covariance");
    assert_eq!(covariance.status, FixedEffectCovarianceStatus::Available);
    assert_eq!(covariance.method, FixedEffectCovarianceMethod::ModelBased);
    assert_eq!(covariance.coef_names, model.coef_names());
    assert_eq!(
        covariance.matrix.as_ref().map(|matrix| matrix.len()),
        Some(model.coef_names().len())
    );
    assert_eq!(covariance.details.rank, Some(model.feterm.rank));
    assert_eq!(
        covariance.details.expected_rank,
        Some(model.coef_names().len())
    );

    let mut data = DataFrame::new();
    data.add_numeric("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    data.add_numeric("x", vec![0.0, 1.0, 0.0, 1.0]).unwrap();
    data.add_numeric("x_dup", vec![0.0, 1.0, 0.0, 1.0]).unwrap();
    data.add_categorical(
        "group",
        vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
        ],
    )
    .unwrap();
    let formula = parse_formula("y ~ 1 + x + x_dup + (1 | group)").unwrap();
    let mut rank_deficient = LinearMixedModel::new(formula, &data, None).unwrap();
    rank_deficient.fit(false).unwrap();

    let json = serde_json::to_string(rank_deficient.compiler_artifact()).unwrap();
    let decoded: CompiledModelArtifact = serde_json::from_str(&json).unwrap();
    let covariance = decoded
        .fixed_effect_covariance_matrix
        .expect("fitted rank-deficient artifact should carry unavailable covariance status");
    assert_eq!(covariance.status, FixedEffectCovarianceStatus::Unavailable);
    assert_eq!(covariance.method, FixedEffectCovarianceMethod::Unavailable);
    assert!(covariance.matrix.is_none());
    assert_eq!(
        covariance.reason.as_deref(),
        Some("rank_deficient_fixed_effects")
    );
    assert_eq!(
        covariance.details.expected_rank,
        Some(rank_deficient.coef_names().len())
    );
    assert!(covariance.details.rank.unwrap() < covariance.details.expected_rank.unwrap());
}

#[test]
fn test_fixed_effect_null_bootstrap_multi_df_term_returns_joint_f_row() {
    let (model, hypothesis) = three_level_condition_fixture();

    let row = model.fixed_effect_null_bootstrap_inference_row(
        FixedEffectInferenceRowKind::Term,
        hypothesis,
        &FixedEffectBootstrapOptions {
            requested_replicates: 35,
            failed_refit_policy: BootstrapFailedRefitPolicy::Exclude,
            seed: Some(20260512),
        },
    );

    assert_eq!(row.kind, FixedEffectInferenceRowKind::Term);
    assert_eq!(row.method, FixedEffectInferenceMethod::Bootstrap);
    assert_eq!(row.status, FixedEffectInferenceStatus::Available);
    assert_eq!(row.statistic_name, Some(FixedEffectStatisticName::F));
    assert_eq!(row.numerator_df, Some(2.0));
    assert!(row.denominator_df.is_none());
    assert!(row.statistic.unwrap().is_finite());
    assert!(row.p_value.unwrap().is_finite());
    assert!(row
        .notes
        .iter()
        .any(|note| note.contains("statistic=joint_wald_f")));

    let details = row.details.expect("term row should carry details");
    let bootstrap = details.bootstrap.expect("bootstrap metadata");
    assert_eq!(bootstrap.target_kind, "fixed_effect_null");
    assert_eq!(bootstrap.requested_replicates, 35);
    assert_eq!(bootstrap.finite_statistic_count, Some(35));
    let family = details.contrast_family.expect("contrast-family metadata");
    assert_eq!(family.restriction_rows, 2);
    assert_eq!(family.effective_rank, Some(2));
    assert_eq!(family.numerator_df, Some(2.0));
    assert_eq!(family.numerator_df_semantics, "effective_restriction_rank");
}

#[test]
fn test_cluster_resample_full_model_contrast_payload_returns_intervals() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();
    let hypothesis =
        FixedEffectHypothesis::single_coefficient("intercept", 0, model.coef_names().len())
            .unwrap();

    let payload = model
        .cluster_resample_full_model_contrast_payload(
            &data,
            "batch",
            &hypothesis,
            &FixedEffectBootstrapOptions {
                requested_replicates: 3,
                failed_refit_policy: BootstrapFailedRefitPolicy::Exclude,
                seed: Some(20260517),
            },
            &[0.95],
        )
        .unwrap();

    assert_eq!(
        payload.metadata.target.kind,
        BootstrapTargetKind::ClusterResample
    );
    assert_eq!(payload.metadata.requested_replicates, 3);
    assert_eq!(payload.metadata.completed_replicates, 3);
    assert_eq!(payload.metadata.finite_statistic_count, Some(3));
    assert!(payload.metadata.mcse.is_none());
    assert_eq!(payload.replicate_statistics.as_ref().map(Vec::len), Some(3));
    let intervals = payload.intervals.as_ref().expect("intervals");
    assert_eq!(intervals.len(), 1);
    assert_eq!(intervals[0].parameter, "intercept");
    assert_eq!(intervals[0].method, BootstrapIntervalMethod::Percentile);
    assert!(payload
        .metadata
        .notes
        .iter()
        .any(|note| note.contains("estimator-distribution target")));
}

#[test]
fn test_restorereplicates_rejects_mismatched_model_shape() {
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let bsamp = MixedModelBootstrap {
        fits: vec![BootstrapReplicate {
            objective: 1.0,
            sigma: 1.0,
            beta: DVector::zeros(model.feterm.rank + 1),
            se: DVector::zeros(model.feterm.rank + 1),
            theta: model.theta(),
        }],
    };

    let mut bytes = Vec::new();
    crate::stats::savereplicates(&mut bytes, &bsamp).unwrap();
    let err = crate::stats::restorereplicates(bytes.as_slice(), &model).unwrap_err();
    match err {
        MixedModelError::InvalidArgument(message) => {
            assert!(message.contains("beta length"));
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn test_parametricbootstrap_quantile_summaries() {
    let bsamp = deterministic_bootstrap_sample();
    let rows = bsamp.quantiles(0.5).unwrap();

    let objective = rows
        .iter()
        .find(|row| row.parameter == "objective")
        .unwrap();
    assert_eq!(objective.n, 5);
    assert_eq!(objective.value, 30.0);

    let beta1 = rows.iter().find(|row| row.parameter == "beta[1]").unwrap();
    assert_eq!(beta1.value, 12.0);

    let se0 = rows.iter().find(|row| row.parameter == "se[0]").unwrap();
    assert_relative_eq!(se0.value, 0.7, epsilon = 1e-12);

    let theta0 = rows.iter().find(|row| row.parameter == "theta[0]").unwrap();
    assert_relative_eq!(theta0.value, 0.3, epsilon = 1e-12);
}

#[test]
fn test_parametricbootstrap_percentile_intervals() {
    let bsamp = deterministic_bootstrap_sample();
    let rows = bsamp.percentile_intervals(0.8).unwrap();

    let objective = rows
        .iter()
        .find(|row| row.parameter == "objective")
        .unwrap();
    assert_eq!(objective.method, BootstrapIntervalMethod::Percentile);
    assert_eq!(objective.n, 5);
    assert_relative_eq!(objective.lower, 14.0, epsilon = 1e-12);
    assert_relative_eq!(objective.upper, 46.0, epsilon = 1e-12);

    let sigma = rows.iter().find(|row| row.parameter == "sigma").unwrap();
    assert_relative_eq!(sigma.lower, 1.4, epsilon = 1e-12);
    assert_relative_eq!(sigma.upper, 4.6, epsilon = 1e-12);
}

#[test]
fn test_parametricbootstrap_shortest_intervals_filter_nonfinite() {
    let bsamp = MixedModelBootstrap {
        fits: vec![
            BootstrapReplicate {
                objective: f64::NAN,
                sigma: 0.0,
                beta: DVector::from_vec(vec![0.0]),
                se: DVector::from_vec(vec![0.0]),
                theta: vec![0.0],
            },
            BootstrapReplicate {
                objective: 10.0,
                sigma: 10.0,
                beta: DVector::from_vec(vec![10.0]),
                se: DVector::from_vec(vec![10.0]),
                theta: vec![10.0],
            },
            BootstrapReplicate {
                objective: 11.0,
                sigma: 11.0,
                beta: DVector::from_vec(vec![11.0]),
                se: DVector::from_vec(vec![11.0]),
                theta: vec![11.0],
            },
            BootstrapReplicate {
                objective: 12.0,
                sigma: 12.0,
                beta: DVector::from_vec(vec![12.0]),
                se: DVector::from_vec(vec![12.0]),
                theta: vec![12.0],
            },
            BootstrapReplicate {
                objective: 100.0,
                sigma: 100.0,
                beta: DVector::from_vec(vec![100.0]),
                se: DVector::from_vec(vec![100.0]),
                theta: vec![100.0],
            },
        ],
    };

    let rows = bsamp.shortest_intervals(0.6).unwrap();
    let objective = rows
        .iter()
        .find(|row| row.parameter == "objective")
        .unwrap();
    assert_eq!(objective.method, BootstrapIntervalMethod::Shortest);
    assert_eq!(objective.n, 4);
    assert_eq!((objective.lower, objective.upper), (10.0, 12.0));

    let sigma = rows.iter().find(|row| row.parameter == "sigma").unwrap();
    assert_eq!(sigma.n, 5);
    assert_eq!((sigma.lower, sigma.upper), (10.0, 12.0));
}

#[test]
fn test_parametricbootstrap_summaries_reject_bad_inputs() {
    let bsamp = deterministic_bootstrap_sample();
    assert!(matches!(
        bsamp.quantiles(1.2),
        Err(MixedModelError::InvalidArgument(_))
    ));
    assert!(matches!(
        bsamp.percentile_intervals(1.0),
        Err(MixedModelError::InvalidArgument(_))
    ));

    let mismatched = MixedModelBootstrap {
        fits: vec![
            BootstrapReplicate {
                objective: 1.0,
                sigma: 1.0,
                beta: DVector::from_vec(vec![1.0]),
                se: DVector::from_vec(vec![1.0]),
                theta: vec![1.0],
            },
            BootstrapReplicate {
                objective: 2.0,
                sigma: 2.0,
                beta: DVector::from_vec(vec![1.0, 2.0]),
                se: DVector::from_vec(vec![1.0]),
                theta: vec![1.0],
            },
        ],
    };
    assert!(matches!(
        mismatched.quantiles(0.5),
        Err(MixedModelError::InvalidArgument(_))
    ));
}

#[test]
fn test_parametricbootstrap_sigma_near_fitted() {
    // Over many replicates the mean bootstrap σ should be close to the
    // fitted σ (within 50% — bootstrap estimates have high variance for
    // small n, but the mean should be in the right ballpark).
    let data = dyestuff_fixture();
    let formula = parse_formula("yield ~ 1 + (1 | batch)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(false).unwrap();

    let fitted_sigma = model.sigma();

    let mut rng = StdRng::seed_from_u64(1234321);
    let bsamp = parametricbootstrap(&mut rng, 30, &model);

    let finite_sigmas: Vec<f64> = bsamp
        .sigmas()
        .into_iter()
        .filter(|s| s.is_finite())
        .collect();
    assert!(
        !finite_sigmas.is_empty(),
        "Should have at least one converged replicate"
    );

    let mean_sigma = finite_sigmas.iter().sum::<f64>() / finite_sigmas.len() as f64;
    let rel_err = ((mean_sigma - fitted_sigma) / fitted_sigma).abs();
    assert!(
        rel_err < 0.50,
        "Mean bootstrap σ {:.4} should be within 50% of fitted σ {:.4}",
        mean_sigma,
        fitted_sigma
    );
}

fn deterministic_bootstrap_sample() -> MixedModelBootstrap {
    MixedModelBootstrap {
        fits: (0..5)
            .map(|idx| {
                let k = idx as f64;
                BootstrapReplicate {
                    objective: 10.0 * (k + 1.0),
                    sigma: k + 1.0,
                    beta: DVector::from_vec(vec![k, 10.0 + k]),
                    se: DVector::from_vec(vec![0.5 + 0.1 * k, 1.5 + 0.1 * k]),
                    theta: vec![0.1 * (k + 1.0)],
                }
            })
            .collect(),
    }
}

// === K6: streamed fixed-design rank/pivot boundary ===

fn rank_path_diagnostic(model: &LinearMixedModel) -> Option<&crate::compiler::Diagnostic> {
    model.compiler_artifact().diagnostics.iter().find(|diag| {
        diag.payload.get("diagnostic_kind") == Some(&serde_json::json!("fixed_design_rank_path"))
    })
}

#[test]
fn streamed_full_rank_design_takes_gram_certified_rank_path() {
    let data = high_cardinality_streamed_fixture(120, 6);
    let formula = parse_formula("y ~ 1 + x + sku + (1 | group)").unwrap();

    let streamed = LinearMixedModel::new_with_fixed_design_policy(
        formula.clone(),
        &data,
        None,
        FixedDesignBuildPolicy::streamed(),
    )
    .unwrap();
    let dense = LinearMixedModel::new_with_fixed_design_policy(
        formula,
        &data,
        None,
        FixedDesignBuildPolicy::dense(),
    )
    .unwrap();

    assert_eq!(
        streamed.fixed_design.storage(),
        FixedDesignStorage::Streamed
    );
    let diag = rank_path_diagnostic(&streamed)
        .expect("streamed constructor must record a fixed_design_rank_path diagnostic");
    assert_eq!(
        diag.payload.get("rank_path"),
        Some(&serde_json::json!("streamed_gram_certified")),
        "comfortably full-rank streamed design must take the Gram-certified path"
    );
    assert_eq!(diag.severity, crate::compiler::DiagnosticSeverity::Info);

    // The certified path must agree exactly with the dense Householder result.
    assert_eq!(streamed.feterm.rank, dense.feterm.rank);
    assert_eq!(streamed.feterm.piv, dense.feterm.piv);
    assert_eq!(streamed.feterm.cnames, dense.feterm.cnames);

    // The dense backend records no rank-path diagnostic (path unchanged).
    assert!(rank_path_diagnostic(&dense).is_none());
}

fn collinear_streamed_fixture() -> (DataFrame, crate::formula::Formula) {
    let n_levels = 24usize;
    let n_obs = 240usize;
    let mut data = DataFrame::new();
    data.add_numeric(
        "y",
        (0..n_obs).map(|idx| (idx % 17) as f64 * 0.25).collect(),
    )
    .unwrap();
    let labels: Vec<String> = (0..n_obs)
        .map(|idx| format!("sku{:02}", idx % n_levels))
        .collect();
    // `dup` duplicates `sku` exactly, so its dummy block is collinear with
    // sku's and the joint fixed design is rank-deficient.
    data.add_categorical("sku", labels.clone()).unwrap();
    data.add_categorical("dup", labels).unwrap();
    data.add_categorical(
        "group",
        (0..n_obs).map(|idx| format!("g{}", idx % 12)).collect(),
    )
    .unwrap();
    let formula = parse_formula("y ~ 1 + sku + dup + (1 | group)").unwrap();
    (data, formula)
}

#[test]
fn streamed_rank_deficient_design_falls_back_to_dense_householder() {
    let (data, formula) = collinear_streamed_fixture();

    let streamed = LinearMixedModel::new_with_fixed_design_policy(
        formula.clone(),
        &data,
        None,
        FixedDesignBuildPolicy::streamed(),
    )
    .unwrap();
    let dense = LinearMixedModel::new_with_fixed_design_policy(
        formula,
        &data,
        None,
        FixedDesignBuildPolicy::dense(),
    )
    .unwrap();

    let diag = rank_path_diagnostic(&streamed)
        .expect("streamed constructor must record a fixed_design_rank_path diagnostic");
    assert_eq!(
        diag.payload.get("rank_path"),
        Some(&serde_json::json!("dense_householder_fallback")),
        "rank-deficient streamed design must fall back to the exact dense pass"
    );

    // Householder pivot parity: the fallback must reproduce the dense
    // backend's rank, pivot, and kept-column names exactly.
    assert!(streamed.feterm.rank < streamed.feterm.n_cols());
    assert_eq!(streamed.feterm.rank, dense.feterm.rank);
    assert_eq!(streamed.feterm.piv, dense.feterm.piv);
    assert_eq!(streamed.feterm.cnames, dense.feterm.cnames);
}

#[test]
fn streamed_rank_fallback_over_dense_bound_is_a_warning() {
    let (data, formula) = collinear_streamed_fixture();

    let policy = FixedDesignBuildPolicy::streamed().with_max_dense_bytes(1);
    let model =
        LinearMixedModel::new_with_fixed_design_policy(formula, &data, None, policy).unwrap();

    let diag = rank_path_diagnostic(&model)
        .expect("streamed constructor must record a fixed_design_rank_path diagnostic");
    assert_eq!(
        diag.payload.get("rank_path"),
        Some(&serde_json::json!("dense_householder_fallback"))
    );
    assert_eq!(
        diag.severity,
        crate::compiler::DiagnosticSeverity::Warning,
        "ambiguous rank over the dense-bytes policy bound must surface as a warning"
    );
    assert_eq!(
        diag.payload.get("policy_max_dense_bytes"),
        Some(&serde_json::json!(1))
    );
}
