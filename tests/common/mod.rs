// Shared fixtures for the lmm_engine_* integration suites, migrated from
// src/model/linear/tests.rs (ranked-audit M3). Helpers marked as also used
// by the remaining inline engine tests are intentionally duplicated there —
// keep the two copies in sync when touching generator constants.
//
// The classic datasets (dyestuff, sleepstudy, penicillin, pastes, dyestuff2)
// also exist in `mixeff_rs::datasets`, but that module is only public under
// `unstable-internals`; these hand-coded copies keep the default-feature
// suites self-contained.
#![allow(dead_code)]

use approx::assert_relative_eq;
#[cfg(feature = "unstable-internals")]
use mixeff_rs::compiler::{ContrastMatrix, FixedEffectHypothesis};
use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::{Column, DataFrame};
use mixeff_rs::model::linear::*;
#[cfg(feature = "unstable-internals")]
use mixeff_rs::model::traits::MixedModelFit;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

pub fn assert_matrix_relative_eq(actual: &DMatrix<f64>, expected: &DMatrix<f64>, epsilon: f64) {
    assert_eq!(actual.shape(), expected.shape());
    for row in 0..actual.nrows() {
        for col in 0..actual.ncols() {
            assert_relative_eq!(actual[(row, col)], expected[(row, col)], epsilon = epsilon);
        }
    }
}

pub fn deterministic_bootstrap_sample() -> MixedModelBootstrap {
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

/// Dyestuff2 data — same structure as Dyestuff but within-batch variance
/// dominates, so the RE variance collapses to zero (singular fit).
/// Values decoded from `dyestuff2.arrow` (MixedModelsDatasets.jl).
pub fn dyestuff2_fixture() -> DataFrame {
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

// ── Fixtures from actual Julia MixedModels.jl datasets ─────────────────

/// Dyestuff data (Davies, 1949) — 6 batches × 5 observations.
/// Matches `dataset(:dyestuff)` from MixedModelsDatasets.jl.
pub fn dyestuff_fixture() -> DataFrame {
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

pub fn fitted_varpar(model: &LinearMixedModel) -> Vec<f64> {
    let mut varpar = model.theta();
    varpar.push(model.sigma());
    varpar
}

pub fn grouped_slope_data(n_groups: usize) -> DataFrame {
    grouped_slope_data_with_obs(n_groups, 2)
}

pub fn grouped_slope_data_with_obs(n_groups: usize, obs_per_group: usize) -> DataFrame {
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

#[cfg(feature = "unstable-internals")]
pub fn hypothesis_by_label<'a>(
    hypotheses: &'a [FixedEffectHypothesis],
    label: &str,
) -> &'a FixedEffectHypothesis {
    hypotheses
        .iter()
        .find(|hypothesis| hypothesis.label == label)
        .unwrap_or_else(|| panic!("missing hypothesis {label} in {hypotheses:?}"))
}

pub fn matrices_differ(a: &DMatrix<f64>, b: &DMatrix<f64>, tolerance: f64) -> bool {
    a.shape() != b.shape()
        || a.iter()
            .zip(b.iter())
            .any(|(left, right)| (left - right).abs() > tolerance)
}

/// Pastes data (Davies, 1947) — 10 batches × 3 casks × 2 samples = 60 obs.
/// Matches `dataset(:pastes)` from MixedModelsDatasets.jl.
/// The nested structure `batch / cask` expands to `batch + batch:cask`.
pub fn pastes_fixture() -> DataFrame {
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

/// Penicillin data (Davies, 1967) — 24 plates × 6 samples = 144 observations.
/// Matches `dataset(:penicillin)` from MixedModelsDatasets.jl.
pub fn penicillin_fixture() -> DataFrame {
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

pub fn permute_rows(data: &DataFrame, order: &[usize]) -> DataFrame {
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
            // Column is #[non_exhaustive] outside the crate.
            other => panic!("unsupported column kind in permute_rows: {other:?}"),
        }
    }

    permuted
}

pub fn rank_one_rho_one_random_slope_fixture() -> DataFrame {
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

pub fn shared_julia_parity_fixture() -> DataFrame {
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

pub fn simulate_sleepstudy_like(
    n_subjects: usize,
    n_obs_per_subject: usize,
    seed: u64,
) -> DataFrame {
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

/// Synthetic data where every group mean equals 5.0 (SS_B = 0).
/// The ML estimate of between-group variance is exactly 0 → θ = 0 → singular.
pub fn singular_re_fixture() -> DataFrame {
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

/// Sleepstudy data (Belenky et al., 2003) — 18 subjects × 10 days.
/// Matches `dataset(:sleepstudy)` from MixedModelsDatasets.jl.
pub fn sleepstudy_fixture() -> DataFrame {
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

#[cfg(feature = "unstable-internals")]
pub fn three_level_condition_fixture() -> (LinearMixedModel, FixedEffectHypothesis) {
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

pub fn typed_term_test_fixture() -> LinearMixedModel {
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

pub fn weighted_lmm_fixture() -> (DataFrame, Vec<f64>) {
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
