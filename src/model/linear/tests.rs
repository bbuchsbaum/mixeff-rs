use super::*;
use approx::assert_relative_eq;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::compiler::{
    CertificateCheck, CompiledModelArtifact, CompilerPolicy, ContrastMatrix, ContrastRhs,
    ConvergenceLevel, ConvergenceVerdict, DiagnosticCode, EffectiveRankStatus, EvidenceQuality,
    FitIntent, FitStatus, FixedEffectCovarianceMethod, FixedEffectCovarianceStatus,
    FixedEffectHypothesis, InferenceStatus, InformationBudgetStatus, ModelChangeStatus,
    ModelStateStatus, RandomStrategy, ReductionRecord, ReductionTrigger, ThetaMap,
};
use crate::formula::parse_formula;
use crate::model::data::DataFrame;
use crate::model::traits::MixedModelFit;

// Several fixture helpers below (simulate_sleepstudy_like, sleepstudy_fixture,
// dyestuff_fixture, shared_julia_parity_fixture, ...) are intentionally
// duplicated in tests/common/mod.rs for the lmm_engine_* integration suites.
// If you change a generator's constants here, mirror the change there — the
// two copies drift silently otherwise.

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
fn test_explicit_categorical_contrast_basis_respects_non_marginal_interaction_expansion() {
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
    // R's `model.matrix(~ anchor + x:anchor)` uses the explicit contrast for
    // the anchor main effect, but full anchor indicators in the non-marginal
    // interaction so the missing x main-effect component remains spanned.
    assert!(model
        .feterm
        .cnames
        .iter()
        .any(|name| name == "anchor: low:x"));
    assert!(model
        .feterm
        .cnames
        .iter()
        .any(|name| name == "anchor: high:x"));
    assert!(!model
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
fn test_native_auto_crossed_large_recourse_targets_crossed_vector_blocks() {
    let crossed_data = simulate_large_theta_crossed(123);
    let crossed_formula = parse_formula(
        "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
    )
    .unwrap();
    let mut crossed = LinearMixedModel::new(crossed_formula, &crossed_data, None).unwrap();

    assert!(crossed.should_auto_use_native_crossed_large_ladder());
    crossed.configure_native_auto_crossed_large_recourse();
    assert_eq!(
        crossed.trust_bq_start_ladder,
        TrustBqStartLadder::DiagonalFirst
    );
    assert_eq!(
        crossed.optsum.max_feval,
        NATIVE_AUTO_CROSSED_LARGE_MAX_FEVAL
    );

    let mut capped = LinearMixedModel::new(
        parse_formula(
            "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
        )
        .unwrap(),
        &crossed_data,
        None,
    )
    .unwrap();
    capped.optsum.max_feval = 750;
    capped.configure_native_auto_crossed_large_recourse();
    assert_eq!(capped.optsum.max_feval, 750);

    let sleep_data = simulate_sleepstudy_like(24, 10, 42);
    let sleep_formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let sleep = LinearMixedModel::new(sleep_formula, &sleep_data, None).unwrap();
    assert!(!sleep.should_auto_use_native_crossed_large_ladder());

    let scalar_formula =
        parse_formula("reaction ~ 1 + days + (1 | subj) + (1 | item) + (1 | site)").unwrap();
    let scalar = LinearMixedModel::new(scalar_formula, &crossed_data, None).unwrap();
    assert!(!scalar.should_auto_use_native_crossed_large_ladder());

    // Brown-style layout: one full vector block is crossed with covariance
    // directions represented as separate scalar blocks. The diagonal-first
    // ladder still has off-diagonal theta to remove from the full block and
    // must not require a second full-Cholesky vector term.
    let mixed_formula = parse_formula(
        "reaction ~ 1 + days + (1 + days | subj) + \
         (1 | item) + (0 + days | item) + (1 | site) + (0 + days | site)",
    )
    .unwrap();
    let mixed = LinearMixedModel::new(mixed_formula, &crossed_data, None).unwrap();
    assert_eq!(mixed.n_theta(), 7);
    assert!(mixed.should_auto_use_native_crossed_large_ladder());
}

#[test]
fn test_trust_bq_sample_reuse_defaults_to_family_policy() {
    assert_eq!(
        OptimizerControl::default().trust_bq_sample_reuse,
        TrustBqSampleReuse::FamilyPolicy
    );
    assert!(!OptimizerControl::default()
        .caller_set_fields()
        .contains(&"trust_bq_sample_reuse".to_string()));
}

#[test]
fn test_trust_bq_sample_reuse_override_is_audit_recorded() {
    let data = simulate_sleepstudy_like(24, 10, 42);
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model
        .fit_with_options(
            FitOptions::reml().with_optimizer_control(
                OptimizerControl::auto()
                    .with_optimizer(Optimizer::TrustBq)
                    .with_trust_bq_sample_reuse(TrustBqSampleReuse::AllFamilies),
            ),
        )
        .unwrap();

    assert!(model
        .optsum
        .caller_set_fields
        .contains(&"trust_bq_sample_reuse".to_string()));
}

#[test]
fn test_trust_bq_sample_reuse_resolve_maps_every_variant() {
    // FamilyPolicy passes the family default through unchanged in both
    // directions.
    assert!(TrustBqSampleReuse::FamilyPolicy.resolve(true));
    assert!(!TrustBqSampleReuse::FamilyPolicy.resolve(false));
    // Disabled forces reuse off even for a family that would reuse
    // (e.g. crossed/large theta).
    assert!(!TrustBqSampleReuse::Disabled.resolve(true));
    assert!(!TrustBqSampleReuse::Disabled.resolve(false));
    // AllFamilies forces reuse on even for a non-crossed family whose policy
    // default is `false` — the case the audit-recording test does not cover.
    assert!(TrustBqSampleReuse::AllFamilies.resolve(false));
    assert!(TrustBqSampleReuse::AllFamilies.resolve(true));
}

/// Fit the same crossed/large-theta model on the native TrustBQ path with the
/// default family policy (which reuses samples) and with reuse disabled, and
/// confirm the override actually reaches the optimizer: the recorded evaluation
/// count moves while the fitted objective stays identical (exact reuse).
#[test]
fn test_trust_bq_sample_reuse_override_changes_optimizer_trace() {
    let data = simulate_large_theta_crossed(123);
    let formula = parse_formula(
        "reaction ~ 1 + days + (1 + days | subj) + (1 + days | item) + (1 + days | site)",
    )
    .unwrap();

    let fit = |reuse: TrustBqSampleReuse| {
        let mut model = LinearMixedModel::new(formula.clone(), &data, None).unwrap();
        model.optsum.max_feval = 3000;
        model
            .fit_with_options(
                FitOptions::reml().with_optimizer_control(
                    OptimizerControl::auto()
                        .with_optimizer(Optimizer::TrustBq)
                        .with_trust_bq_sample_reuse(reuse),
                ),
            )
            .unwrap();
        model
    };

    // CrossedLarge (n_theta == 9) reuses samples under the family policy, so
    // Disabled is a genuine toggle rather than a no-op.
    let family_policy = fit(TrustBqSampleReuse::FamilyPolicy);
    assert_eq!(family_policy.n_theta(), 9);
    let disabled = fit(TrustBqSampleReuse::Disabled);

    // Exact reuse: turning it off must not move the fitted objective.
    assert!(
        (family_policy.objective_value() - disabled.objective_value()).abs() < 1e-6,
        "reuse is exact: fmin {} (family policy) vs {} (disabled)",
        family_policy.objective_value(),
        disabled.objective_value()
    );
    // Wiring: the override must reach the optimizer, so the evaluation trace
    // differs. If this ever ties, a call site stopped honoring the override.
    assert_ne!(
        family_policy.optsum.feval, disabled.optsum.feval,
        "sample-reuse override did not change the TrustBQ evaluation trace \
         (family policy feval {}, disabled feval {})",
        family_policy.optsum.feval, disabled.optsum.feval
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
    // The optimizer-stop parser must classify by the inner stop code, not the
    // START_LADDER wrapper. The final certificate may still be NotOptimized
    // when its independent derivative checks reject stationarity; that is an
    // intentionally stricter, honest status rather than a wrapper-parsing
    // failure.
    let certificate = laddered.optimizer_certificate().unwrap();
    assert_eq!(
        certificate.evidence.optimizer_stop.acceptable_stop,
        laddered.optsum.converged(),
        "START_LADDER must preserve the inner optimizer stop classification"
    );
    if certificate.status == FitStatus::NotOptimized {
        assert!(
            certificate
                .checks
                .iter()
                .any(|check| matches!(check, CertificateCheck::DerivativeMismatch { .. })),
            "an accepted ladder stop may be demoted only by independent certificate evidence"
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
    let init = DMatrix::from_row_slice(3, 3, &[3.0, 0.8, 0.4, 0.2, 2.5, -0.7, -0.3, -0.1, 1.7]);
    let mut optimized = MatrixBlock::Dense(init.clone());
    let mut expected = init;
    expected.gemm(-1.0, &a, &a.transpose(), 1.0);

    rank_k_downdate(&mut optimized, &a);

    let MatrixBlock::Dense(result) = optimized else {
        panic!("expected dense block");
    };
    for row in 0..3 {
        for col in 0..3 {
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
fn test_rank_k_downdate_vsize2_blockdiag_large_k_matches_gemm() {
    let a = DMatrix::from_fn(4, 520, |row, col| {
        (((row + 5) * (col + 7)) % 23) as f64 / 19.0 - 0.35
    });
    let blocks = vec![
        DMatrix::from_row_slice(2, 2, &[2.0, 0.7, -0.2, 1.5]),
        DMatrix::from_row_slice(2, 2, &[3.0, -0.4, 0.9, 2.2]),
    ];
    let mut optimized = MatrixBlock::BlockDiagonal(blocks.clone());
    let mut expected_blocks = blocks;
    for (block_idx, expected) in expected_blocks.iter_mut().enumerate() {
        let a_block = a.rows(block_idx * 2, 2);
        expected.gemm(-1.0, &a_block, &a_block.transpose(), 1.0);
    }

    rank_k_downdate(&mut optimized, &a);

    let MatrixBlock::BlockDiagonal(result_blocks) = optimized else {
        panic!("expected block-diagonal block");
    };
    for (result, expected) in result_blocks.iter().zip(expected_blocks.iter()) {
        for row in 0..2 {
            for col in 0..2 {
                assert_relative_eq!(
                    result[(row, col)],
                    expected[(row, col)],
                    epsilon = 1e-10,
                    max_relative = 1e-12
                );
            }
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
    if cfg!(windows) {
        // NLopt's Windows timer granularity can allow the tiny fixture to
        // satisfy ftol before the maxtime stop is observed.
        assert!(
            model.optsum.return_value == "MAXTIME_REACHED"
                || model.optsum.return_value == "FTOL_REACHED",
            "unexpected NLopt stop with max_time set: {}",
            model.optsum.return_value
        );
    } else {
        assert_eq!(model.optsum.return_value, "MAXTIME_REACHED");
    }
    assert_eq!(model.optsum.max_time, 1e-9);
    assert!(model.optsum.feval >= 1);
    assert!(model.objective_value().is_finite());
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
    let sigma_abs_tol = 2e-4;
    let sigma_rel_tol = 3e-5;
    assert_relative_eq!(
        model.sigma(),
        julia_sigma,
        epsilon = sigma_abs_tol,
        max_relative = sigma_rel_tol
    );

    let theta = model.theta();
    if cfg!(debug_assertions) {
        // This is a fit-level optimizer smoke test, not the fixed JSON drift
        // gate. Linux NEWUOA can land on slightly different large-theta
        // coordinates while preserving the tight objective/sigma checks above.
        let theta_coord_tol = 1e-3;
        for (actual, expected) in theta.iter().zip(julia_theta.iter()) {
            assert_relative_eq!(
                *actual,
                *expected,
                epsilon = theta_coord_tol,
                max_relative = theta_coord_tol
            );
        }
    } else {
        assert!(
            theta.iter().all(|value| value.is_finite()),
            "release-profile NLopt theta should remain finite: {theta:?}"
        );
    }
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

#[test]
fn trust_bq_propagates_host_interrupt_callback_error() {
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let events = Arc::new(AtomicUsize::new(0));
    let callback_events = Arc::clone(&events);
    let callback = FitProgressCallback::new(move |progress| {
        if progress.phase == FitProgressPhase::LmmOptimizer {
            callback_events.fetch_add(1, Ordering::SeqCst);
            return Err(MixedModelError::Interrupted("test interrupt".to_string()));
        }
        Ok(())
    })
    .with_interval(2);
    let control = OptimizerControl::auto()
        .with_optimizer(Optimizer::TrustBq)
        .with_max_feval(100);
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();

    let error = model
        .fit_with_options(
            FitOptions::reml()
                .with_optimizer_control(control)
                .with_progress_callback(callback),
        )
        .unwrap_err();

    assert_eq!(error.code(), "interrupted");
    assert_eq!(events.load(Ordering::SeqCst), 1);
}
