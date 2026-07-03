// Engine-level prediction tests migrated from src/model/linear/tests.rs
// (ranked-audit M3). Public-API only; internals-bound tests stay inline.

mod common;
#[allow(unused_imports)]
use common::*;

use approx::assert_relative_eq;
#[allow(unused_imports)]
use mixeff_rs::error::*;
use mixeff_rs::formula::parse_formula;
#[allow(unused_imports)]
use mixeff_rs::model::data::{Column, DataFrame};
#[allow(unused_imports)]
use mixeff_rs::model::fixed_design::*;
#[allow(unused_imports)]
use mixeff_rs::model::linear::*;
#[allow(unused_imports)]
use mixeff_rs::model::traits::MixedModelFit;
#[allow(unused_imports)]
use mixeff_rs::stats::*;
#[allow(unused_imports)]
use mixeff_rs::types::*;
use nalgebra::DMatrix;

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
    let helmert = mixeff_rs::model::data::CategoricalContrast::helmert(levels.clone()).unwrap();

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
    #[allow(clippy::needless_range_loop)]
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
