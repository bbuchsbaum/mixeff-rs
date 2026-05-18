//! Micro-profile the general PLS evaluation path on synthetic
//! sleepstudy-shaped random-slope data.
//!
//! Run:
//!     cargo run --release --features unstable-internals --example profile_pls_kernel

use std::time::Instant;

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::traits::MixedModelFit;
use mixeff_rs::types::MatrixBlock;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

const FORMULA: &str = "reaction ~ 1 + days + (1 + days | subj)";
const N_SUBJECTS: usize = 10_000;
const N_OBS_PER_SUBJECT: usize = 10;
const WARMUP: usize = 200;
const MEASURED: usize = 1_000;

fn simulate(n_subjects: usize, n_obs_per_subject: usize, seed: u64) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let beta = [250.0, 10.0];
    let sigma_resid = 25.0;
    let l11 = 24.0;
    let l21 = 0.07 * 6.0;
    let l22 = (6.0_f64.powi(2) - l21 * l21).sqrt();

    let total_n = n_subjects * n_obs_per_subject;
    let mut reaction = Vec::with_capacity(total_n);
    let mut days = Vec::with_capacity(total_n);
    let mut subj_labels = Vec::with_capacity(total_n);

    for i in 0..n_subjects {
        let u0: f64 = normal.sample(&mut rng);
        let u1: f64 = normal.sample(&mut rng);
        let b0 = l11 * u0;
        let b1 = l21 * u0 + l22 * u1;
        let label = format!("S{:06}", i + 1);
        for d in 0..n_obs_per_subject {
            let x = d as f64;
            let mu = beta[0] + beta[1] * x + b0 + b1 * x;
            let y = mu + sigma_resid * normal.sample(&mut rng);
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

fn percentile(samples: &mut [f64], q: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples.len() - 1) as f64 * q).round() as usize;
    samples[idx]
}

fn us(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000_000.0
}

fn summarize(label: &str, samples: &[f64]) {
    let min = percentile(&mut samples.to_vec(), 0.0);
    let p50 = percentile(&mut samples.to_vec(), 0.5);
    let p90 = percentile(&mut samples.to_vec(), 0.9);
    println!("{label:<18} min {min:>8.2} us  p50 {p50:>8.2} us  p90 {p90:>8.2} us");
}

#[derive(Default)]
struct StageSamples {
    scale_l00: Vec<f64>,
    scale_l10: Vec<f64>,
    chol_l00: Vec<f64>,
    solve_l10: Vec<f64>,
    downdate_l11_gemm: Vec<f64>,
    downdate_l11_hand: Vec<f64>,
    objective_value: Vec<f64>,
}

fn dense_block(block: &MatrixBlock) -> &nalgebra::DMatrix<f64> {
    match block {
        MatrixBlock::Dense(mat) => mat,
        _ => panic!("expected dense block"),
    }
}

fn dense_block_mut(block: &mut MatrixBlock) -> &mut nalgebra::DMatrix<f64> {
    match block {
        MatrixBlock::Dense(mat) => mat,
        _ => panic!("expected dense block"),
    }
}

fn blockdiag2(block: &MatrixBlock) -> &[nalgebra::DMatrix<f64>] {
    match block {
        MatrixBlock::BlockDiagonal(blocks) => blocks,
        _ => panic!("expected block-diagonal block"),
    }
}

fn blockdiag2_mut(block: &mut MatrixBlock) -> &mut [nalgebra::DMatrix<f64>] {
    match block {
        MatrixBlock::BlockDiagonal(blocks) => blocks,
        _ => panic!("expected block-diagonal block"),
    }
}

fn scale_l00_vsize2(
    dst_blocks: &mut [nalgebra::DMatrix<f64>],
    src_blocks: &[nalgebra::DMatrix<f64>],
    l00: f64,
    l01: f64,
    l10: f64,
    l11: f64,
) {
    for (dst_blk, src_blk) in dst_blocks.iter_mut().zip(src_blocks.iter()) {
        let s00 = src_blk[(0, 0)];
        let s01 = src_blk[(0, 1)];
        let s10 = src_blk[(1, 0)];
        let s11 = src_blk[(1, 1)];

        let t00 = s00 * l00 + s01 * l10;
        let t01 = s00 * l01 + s01 * l11;
        let t10 = s10 * l00 + s11 * l10;
        let t11 = s10 * l01 + s11 * l11;

        dst_blk[(0, 0)] = l00 * t00 + l10 * t10 + 1.0;
        dst_blk[(0, 1)] = l00 * t01 + l10 * t11;
        dst_blk[(1, 0)] = l01 * t00 + l11 * t10;
        dst_blk[(1, 1)] = l01 * t01 + l11 * t11 + 1.0;
    }
}

fn scale_l10_vsize2(
    dst: &mut nalgebra::DMatrix<f64>,
    src: &nalgebra::DMatrix<f64>,
    l00: f64,
    l01: f64,
    l10: f64,
    l11: f64,
) {
    let nblocks = src.ncols() / 2;
    for b in 0..nblocks {
        let col0 = b * 2;
        let col1 = col0 + 1;
        for i in 0..src.nrows() {
            let x0 = src[(i, col0)];
            let x1 = src[(i, col1)];
            dst[(i, col0)] = x0 * l00 + x1 * l10;
            dst[(i, col1)] = x0 * l01 + x1 * l11;
        }
    }
}

fn chol_l00_vsize2(blocks: &mut [nalgebra::DMatrix<f64>]) {
    for blk in blocks.iter_mut() {
        let d00 = blk[(0, 0)];
        if d00 <= 0.0 {
            blk[(0, 0)] = 0.0;
            blk[(1, 0)] = 0.0;
        } else {
            blk[(0, 0)] = d00.sqrt();
            blk[(1, 0)] /= blk[(0, 0)];
        }

        let d11 = blk[(1, 1)] - blk[(1, 0)] * blk[(1, 0)];
        blk[(1, 1)] = if d11 <= 0.0 { 0.0 } else { d11.sqrt() };
        blk[(0, 1)] = 0.0;
    }
}

fn solve_l10_vsize2(a: &mut nalgebra::DMatrix<f64>, blocks: &[nalgebra::DMatrix<f64>]) {
    let mut col_offset = 0;
    for l_blk in blocks {
        let c0 = col_offset;
        let c1 = col_offset + 1;
        let l00 = l_blk[(0, 0)];
        let l10 = l_blk[(1, 0)];
        let l11 = l_blk[(1, 1)];

        for i in 0..a.nrows() {
            let x0 = a[(i, c0)];
            a[(i, c0)] = if l00.abs() < 1e-30 { 0.0 } else { x0 / l00 };
            a[(i, c1)] = if l11.abs() < 1e-30 {
                0.0
            } else {
                (a[(i, c1)] - a[(i, c0)] * l10) / l11
            };
        }
        col_offset += 2;
    }
}

fn downdate_l11_hand(c: &mut nalgebra::DMatrix<f64>, a: &nalgebra::DMatrix<f64>) {
    let rows = a.nrows();
    let cols = a.ncols();
    for row in 0..rows {
        for col in 0..=row {
            let mut sum = 0.0;
            for k in 0..cols {
                sum += a[(row, k)] * a[(col, k)];
            }
            c[(row, col)] -= sum;
        }
    }
}

fn chol_small_dense(mat: &mut nalgebra::DMatrix<f64>) {
    let n = mat.nrows();
    for j in 0..n {
        let mut s = mat[(j, j)];
        for k in 0..j {
            s -= mat[(j, k)] * mat[(j, k)];
        }
        if s <= 0.0 {
            for i in j..n {
                mat[(i, j)] = 0.0;
            }
            continue;
        }
        mat[(j, j)] = s.sqrt();

        for i in (j + 1)..n {
            let mut s = mat[(i, j)];
            for k in 0..j {
                s -= mat[(i, k)] * mat[(j, k)];
            }
            mat[(i, j)] = s / mat[(j, j)];
        }

        for i in 0..j {
            mat[(i, j)] = 0.0;
        }
    }
}

fn profile_update_stages(
    model: &mut LinearMixedModel,
    theta: &[f64],
    measured: usize,
) -> Result<StageSamples, Box<dyn std::error::Error>> {
    let mut samples = StageSamples::default();
    samples.scale_l00.reserve(measured);
    samples.scale_l10.reserve(measured);
    samples.chol_l00.reserve(measured);
    samples.solve_l10.reserve(measured);
    samples.downdate_l11_gemm.reserve(measured);
    samples.downdate_l11_hand.reserve(measured);
    samples.objective_value.reserve(measured);

    for _ in 0..measured {
        model.set_theta(theta)?;
        let l00 = model.reterms()[0].lambda[(0, 0)];
        let l01 = model.reterms()[0].lambda[(0, 1)];
        let l10 = model.reterms()[0].lambda[(1, 0)];
        let l11 = model.reterms()[0].lambda[(1, 1)];

        let (l_blocks, a_blocks) = model.l_blocks_mut_a_blocks();

        let t = Instant::now();
        scale_l00_vsize2(
            blockdiag2_mut(&mut l_blocks[0]),
            blockdiag2(&a_blocks[0]),
            l00,
            l01,
            l10,
            l11,
        );
        samples.scale_l00.push(us(t));

        let t = Instant::now();
        scale_l10_vsize2(
            dense_block_mut(&mut l_blocks[1]),
            dense_block(&a_blocks[1]),
            l00,
            l01,
            l10,
            l11,
        );
        samples.scale_l10.push(us(t));

        dense_block_mut(&mut l_blocks[2]).copy_from(dense_block(&a_blocks[2]));

        let t = Instant::now();
        chol_l00_vsize2(blockdiag2_mut(&mut l_blocks[0]));
        samples.chol_l00.push(us(t));

        let t = Instant::now();
        {
            let (left, right) = l_blocks.split_at_mut(1);
            solve_l10_vsize2(dense_block_mut(&mut right[0]), blockdiag2(&left[0]));
        }
        samples.solve_l10.push(us(t));

        let l10_mat = dense_block(&l_blocks[1]).clone();
        let a2 = dense_block(&a_blocks[2]);

        let t = Instant::now();
        {
            let l11_mat = dense_block_mut(&mut l_blocks[2]);
            l11_mat.copy_from(a2);
            l11_mat.gemm(-1.0, &l10_mat, &l10_mat.transpose(), 1.0);
            chol_small_dense(l11_mat);
        }
        samples.downdate_l11_gemm.push(us(t));

        let t = Instant::now();
        {
            let l11_mat = dense_block_mut(&mut l_blocks[2]);
            l11_mat.copy_from(a2);
            downdate_l11_hand(l11_mat, &l10_mat);
            chol_small_dense(l11_mat);
        }
        samples.downdate_l11_hand.push(us(t));

        let t = Instant::now();
        let _ = model.objective_value();
        samples.objective_value.push(us(t));
    }

    Ok(samples)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = simulate(N_SUBJECTS, N_OBS_PER_SUBJECT, 42);
    let formula = parse_formula(FORMULA)?;
    let mut model = LinearMixedModel::new(formula, &df, None)?;
    model.fit(true)?;
    let theta = model.theta();
    let fevals = model.opt_summary().feval;
    let fmin = model.opt_summary().fmin;

    for _ in 0..WARMUP {
        model.set_theta(&theta)?;
        model.update_l()?;
        let _ = model.objective_value();
    }

    let mut set_theta_samples = Vec::with_capacity(MEASURED);
    let mut update_l_samples = Vec::with_capacity(MEASURED);
    let mut objective_samples = Vec::with_capacity(MEASURED);
    let mut objective_at_samples = Vec::with_capacity(MEASURED);
    let mut total_samples = Vec::with_capacity(MEASURED);
    let mut last_obj = 0.0;

    for _ in 0..MEASURED {
        let t_total = Instant::now();

        let t = Instant::now();
        model.set_theta(&theta)?;
        set_theta_samples.push(us(t));

        let t = Instant::now();
        model.update_l()?;
        update_l_samples.push(us(t));

        let t = Instant::now();
        let _ = model.objective_value();
        objective_samples.push(us(t));

        total_samples.push(us(t_total));

        let t = Instant::now();
        last_obj = model.objective_at(&theta)?;
        objective_at_samples.push(us(t));
    }

    println!(
        "n = {}, levels = {}, formula = {FORMULA}",
        df.nrow(),
        N_SUBJECTS
    );
    println!("fit fevals = {fevals}, fmin = {fmin:.6}, theta = {theta:?}");
    summarize("set_theta", &set_theta_samples);
    summarize("update_l", &update_l_samples);
    summarize("objective_value", &objective_samples);
    summarize("total", &total_samples);
    summarize("objective_at", &objective_at_samples);
    println!("last objective = {last_obj:.6}");

    let stage_samples = profile_update_stages(&mut model, &theta, MEASURED)?;
    println!("\nmanual update_l stage timings:");
    summarize("scale L00", &stage_samples.scale_l00);
    summarize("scale L10", &stage_samples.scale_l10);
    summarize("chol L00", &stage_samples.chol_l00);
    summarize("solve L10", &stage_samples.solve_l10);
    summarize("downdate gemm", &stage_samples.downdate_l11_gemm);
    summarize("downdate hand", &stage_samples.downdate_l11_hand);
    summarize("objective_value", &stage_samples.objective_value);

    Ok(())
}
