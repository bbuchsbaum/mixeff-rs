//! One-shot calibration probe for the dimensionless weak-identification index.
//! Run with `cargo run --release --example probe_weak_id` to dump the score
//! per fixture for `tests/fixtures/pathology_corpus/calibration.md`.

use nalgebra::dmatrix;

use mixeff_rs::pathology::{
    block_diagonal_crossings, certify, collinear_fe, empty_crossings, extreme_prevalence,
    near_singular_re, pareto_sizes, scale_mismatch, singletons_with_slope, GeneratorSpec,
    WEAK_ID_THRESHOLD,
};

fn main() {
    let mut rows: Vec<(&'static str, f64, usize, bool, &'static str)> = Vec::new();

    let easy = GeneratorSpec::lmm(
        "easy",
        42,
        vec![6; 30],
        vec![1.0, 2.0],
        true,
        1,
        dmatrix![4.0, 0.5; 0.5, 1.0],
    );
    let cert = certify(&easy);
    rows.push((
        "easy",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "balanced 30×6, single predictor, identity C",
    ));

    let boundary = GeneratorSpec::lmm(
        "boundary",
        42,
        vec![6; 30],
        vec![1.0, 2.0],
        true,
        1,
        dmatrix![4.0, 0.0; 0.0, 0.0],
    );
    let cert = certify(&boundary);
    rows.push((
        "boundary_zero_slope",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "same FE as easy, RE slope variance = 0",
    ));

    let mut reduced_rank = GeneratorSpec::lmm(
        "reduced_rank",
        42,
        vec![6; 30],
        vec![1.0, 2.0],
        true,
        1,
        dmatrix![4.0, 0.0; 0.0, 4.0],
    );
    near_singular_re(&mut reduced_rank, 1.0);
    let cert = certify(&reduced_rank);
    rows.push((
        "reduced_rank",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "RE rank-1 (ρ=1 in Σ_truth), FE identity",
    ));

    let refusal = GeneratorSpec::lmm(
        "refusal_singletons",
        42,
        vec![1; 6],
        vec![1.0, 2.0],
        true,
        1,
        dmatrix![4.0, 0.5; 0.5, 1.0],
    );
    let cert = certify(&refusal);
    rows.push((
        "refusal_singletons",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "6 singleton groups; structural refusal",
    ));

    let imbalance = GeneratorSpec::lmm(
        "imbalance_pareto",
        42,
        pareto_sizes(7, 30, 1.5, 6.0),
        vec![1.0, 2.0],
        true,
        1,
        dmatrix![4.0, 0.5; 0.5, 1.0],
    );
    let cert = certify(&imbalance);
    rows.push((
        "imbalance_pareto",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "30 groups, pareto-sized cells",
    ));

    let mut scale_mm = GeneratorSpec::lmm(
        "scale_mismatch_1e3",
        42,
        vec![6; 30],
        vec![1.0, 2.0],
        true,
        1,
        dmatrix![4.0, 0.5; 0.5, 1.0],
    );
    scale_mismatch(&mut scale_mm, vec![1e3]);
    let cert = certify(&scale_mm);
    rows.push((
        "scale_mismatch_1e3",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "FE predictor scale ×1000; identity C",
    ));

    let mut collinear = GeneratorSpec::lmm(
        "collinear_fe_rho_one",
        42,
        vec![6; 30],
        vec![1.0, 2.0, 3.0],
        true,
        1,
        dmatrix![4.0, 0.5; 0.5, 1.0],
    );
    collinear_fe(&mut collinear, 0, 1, 1.0);
    let cert = certify(&collinear);
    rows.push((
        "collinear_fe_rho_one",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "two predictors at ρ=1 (structural)",
    ));

    let mut extreme = GeneratorSpec::lmm(
        "extreme_prevalence_negative_5",
        42,
        vec![20; 30],
        vec![0.0, 0.5],
        true,
        0,
        dmatrix![1.0],
    );
    extreme_prevalence(&mut extreme, -5.0);
    let cert = certify(&extreme);
    rows.push((
        "extreme_prevalence_negative_5",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "Bernoulli/logit, 600 obs, identity C",
    ));

    let mut singletons_t = GeneratorSpec::lmm(
        "singletons_via_transform",
        42,
        vec![6; 8],
        vec![1.0, 2.0],
        true,
        1,
        dmatrix![4.0, 0.5; 0.5, 1.0],
    );
    singletons_with_slope(&mut singletons_t, 8);
    let cert = certify(&singletons_t);
    rows.push((
        "singletons_via_transform",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "transform-built singletons; structural",
    ));

    let mut rs_singletons = GeneratorSpec::lmm(
        "random_slope_singletons",
        7,
        vec![1; 12],
        vec![0.5, 1.5],
        true,
        1,
        dmatrix![2.0, 0.0; 0.0, 1.0],
    );
    singletons_with_slope(&mut rs_singletons, 12);
    let cert = certify(&rs_singletons);
    rows.push((
        "random_slope_singletons",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "12 singletons + slope; structural",
    ));

    let mut crossed_bd = GeneratorSpec::lmm(
        "crossed_block_diagonal_4x4x4",
        42,
        vec![1; 1],
        vec![1.0],
        true,
        0,
        dmatrix![1.5],
    );
    block_diagonal_crossings(&mut crossed_bd, "h", 4, 4, 0.8);
    let cert = certify(&crossed_bd);
    rows.push((
        "crossed_block_diagonal_4x4x4",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "intercept-only, 4 disjoint 4×4 blocks",
    ));

    let mut crossed_sparse = GeneratorSpec::lmm(
        "crossed_sparse_connected",
        42,
        vec![1; 12],
        vec![1.0],
        true,
        0,
        dmatrix![1.5],
    );
    empty_crossings(&mut crossed_sparse, "h", 12, 0.6, 0.5, 11);
    let cert = certify(&crossed_sparse);
    rows.push((
        "crossed_sparse_connected",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "connected sparse 12×12 crossing",
    ));

    let mut weakly_id = GeneratorSpec::lmm(
        "weakly_identified_near_collinear",
        42,
        vec![3; 4],
        vec![1.0, 2.0, 3.0],
        true,
        1,
        dmatrix![4.0, 0.5; 0.5, 1.0],
    );
    collinear_fe(&mut weakly_id, 0, 1, 0.99);
    let cert = certify(&weakly_id);
    rows.push((
        "weakly_identified_near_collinear",
        cert.weak_id_score,
        cert.n_total,
        cert.weak_identification,
        "small n + ρ=0.99 (weak-id)",
    ));

    println!("threshold = {WEAK_ID_THRESHOLD}");
    println!(
        "{:<35}  {:>10}  {:>5}  {:>5}  notes",
        "fixture", "score", "n", "weak"
    );
    for (name, score, n, weak, notes) in rows {
        println!(
            "{:<35}  {:>10.3}  {:>5}  {:>5}  {}",
            name, score, n, weak, notes
        );
    }
}
