//! Diagnostic for the two failing inference parity cases.

use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::LinearMixedModel;
use mixeff_rs::model::traits::MixedModelFit;
use nalgebra::DMatrix;

fn main() {
    diagnose_penicillin();
    println!();
    diagnose_sleepstudy_joint();
}

fn diagnose_penicillin() {
    println!("== Penicillin REML: diameter ~ 1 + (1|plate) + (1|sample) ==");
    let data = penicillin_fixture();
    let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();
    let theta = model.theta();
    let sigma = model.sigma();
    let coef = model.coef();
    let vcov = model.vcov();
    let stderror = vcov[(0, 0)].sqrt();
    println!("  θ                = {theta:?}");
    println!("  σ                = {sigma:.16}");
    println!("  σ²               = {:.16}", sigma * sigma);
    println!("  β                = {:?}", coef.iter().collect::<Vec<_>>());
    println!("  vcov[0,0]        = {:.16}", vcov[(0, 0)]);
    println!("  std_error        = {stderror:.16}");
    println!("  REML obj         = {:.10}", model.objective_value());
    println!(
        "  Δ vs lmerTest std_error 0.80859536 : {:.3e}",
        stderror - 0.808_595_361_658_236_4
    );

    // Probe REML objective.  Use the *fitted* model — deviance_varpar relies on
    // cached block factorizations established during fit().
    println!("  -- objective scan around fitted θ (using already-fitted model) --");
    let theta_fit = model.theta();
    let baseline = model.objective_value();
    for d_plate in [-0.05, -0.01, 0.0, 0.01, 0.05] {
        for d_sample in [-0.5, -0.1, 0.0, 0.1, 0.5] {
            let candidate = vec![
                theta_fit[0] + d_plate,
                theta_fit[1] + d_sample,
                model.sigma(),
            ];
            if candidate[0] <= 0.0 || candidate[1] <= 0.0 {
                continue;
            }
            let dev = match model.deviance_varpar(&candidate, true) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("    err at {candidate:?}: {e}");
                    f64::NAN
                }
            };
            let star = if dev < baseline - 1e-9 { "*" } else { " " };
            println!(
                "    {star} θ=[{:.4}, {:.4}]  REMLobj={:.6}  Δ={:.4e}",
                candidate[0],
                candidate[1],
                dev,
                dev - baseline
            );
        }
    }

    println!("  -- REML obj at lme4-style well-known θ (varpar=[θ..,σ]) --");
    let sigma = model.sigma();
    for (name, mut th) in [
        ("Julia ML θ", vec![1.5375939045981573, 3.219792193110907]),
        ("Rust fitted", theta_fit.clone()),
    ] {
        th.push(sigma);
        let dev = model.deviance_varpar(&th, true).unwrap_or(f64::NAN);
        println!(
            "    {name:18}  θ=[{:.4}, {:.4}]  σ={:.4}  REMLobj={:.6}",
            th[0], th[1], sigma, dev
        );
    }

    // Refit with several starting points to check that optimizer is stable.
    println!("  -- refit from different θ starts --");
    for theta_start in [
        vec![1.0, 1.0],
        vec![2.0, 5.0],
        vec![0.5, 0.5],
        vec![1.5376, 3.2198],
    ] {
        let formula = parse_formula("diameter ~ 1 + (1 | plate) + (1 | sample)").unwrap();
        let mut m = LinearMixedModel::new(formula, &data, None).unwrap();
        m.optsum_mut().initial = theta_start.clone();
        match m.fit(true) {
            Ok(_) => println!(
                "    start={:?}  →  θ=[{:.6}, {:.6}]  σ={:.6}  REMLobj={:.6}  feval={}",
                theta_start,
                m.theta()[0],
                m.theta()[1],
                m.sigma(),
                m.objective_value(),
                m.optsum().feval
            ),
            Err(e) => println!("    start={:?}  →  fit ERROR: {e}", theta_start),
        }
    }
}

fn diagnose_sleepstudy_joint() {
    println!("== Sleepstudy REML: reaction ~ 1 + days + (1 + days | subj) ==");
    let data = sleepstudy_fixture();
    let formula = parse_formula("reaction ~ 1 + days + (1 + days | subj)").unwrap();
    let mut model = LinearMixedModel::new(formula, &data, None).unwrap();
    model.fit(true).unwrap();
    println!("  θ        = {:?}", model.theta());
    println!("  σ        = {:.16}", model.sigma());
    println!("  β        = {:?}", model.coef().iter().collect::<Vec<_>>());
    let vcov = model.vcov();
    println!("  vcov     =");
    for r in 0..vcov.nrows() {
        for c in 0..vcov.ncols() {
            print!("{:22.14} ", vcov[(r, c)]);
        }
        println!();
    }
    let adjusted = model.kenward_roger_adjusted_vcov().unwrap();
    println!("  Φ_A user-order:");
    for r in 0..adjusted.adjusted_vcov.nrows() {
        for c in 0..adjusted.adjusted_vcov.ncols() {
            print!("{:22.14} ", adjusted.adjusted_vcov[(r, c)]);
        }
        println!();
    }
    println!("  Φ active-order (unadjusted):");
    for r in 0..adjusted.unadjusted_vcov_active.nrows() {
        for c in 0..adjusted.unadjusted_vcov_active.ncols() {
            print!("{:22.14} ", adjusted.unadjusted_vcov_active[(r, c)]);
        }
        println!();
    }
    println!("  Φ_A active-order:");
    for r in 0..adjusted.adjusted_vcov_active.nrows() {
        for c in 0..adjusted.adjusted_vcov_active.ncols() {
            print!("{:22.14} ", adjusted.adjusted_vcov_active[(r, c)]);
        }
        println!();
    }

    let beta = model.coef().clone();
    let l = DMatrix::identity(2, 2);
    let l_phia_lt = &l * &adjusted.adjusted_vcov * l.transpose();
    let inv = l_phia_lt.clone().try_inverse().unwrap();
    let q_form = (beta.transpose() * &inv * &beta)[(0, 0)];
    let f = q_form / 2.0;
    println!("  joint F (user-order Φ_A, β) = {f:.12}");
    println!("  pbkrtest unscaled F target  = 749.9504538648124");
    println!(
        "  Δ F = {:.6}  ({:.3e} relative)",
        f - 749.9504538648124,
        (f - 749.9504538648124).abs() / 749.9504538648124
    );

    // Days adjusted SE (scalar test passes against fixture target 1.5458).
    let l_days = DMatrix::from_row_slice(1, 2, &[0.0, 1.0]);
    let cov_days = &l_days * &adjusted.adjusted_vcov * l_days.transpose();
    println!(
        "  days adj SE = {:.16} (target 1.5457896438972805)",
        cov_days[(0, 0)].sqrt()
    );
    println!("  -- β vs lmer REML reference (251.405, 10.467) --");
    println!("    Δβ_intercept = {:.6e}", beta[0] - 251.40510484848);
    println!("    Δβ_days      = {:.6e}", beta[1] - 10.46728595959596);
}

fn sleepstudy_fixture() -> DataFrame {
    let subjects = [
        "S308", "S309", "S310", "S330", "S331", "S332", "S333", "S334", "S335", "S337", "S349",
        "S350", "S351", "S352", "S369", "S370", "S371", "S372",
    ];
    #[rustfmt::skip]
    let reaction: Vec<f64> = vec![
        249.5600, 258.7047, 250.8006, 321.4398, 356.8519,
        414.6901, 382.2038, 290.1486, 430.5853, 466.3535,
        222.7339, 205.2658, 202.9778, 204.7070, 207.7161,
        215.9618, 213.6303, 217.7272, 224.2957, 237.3142,
        199.0539, 194.3322, 234.3200, 232.8416, 229.3074,
        220.4579, 235.4208, 255.7511, 261.0125, 247.5153,
        321.5426, 300.4002, 283.8565, 285.1330, 285.7973,
        297.5855, 280.2396, 318.2613, 305.3495, 354.0487,
        287.6079, 285.0000, 301.8206, 320.1153, 316.2773,
        293.3187, 290.0750, 334.8177, 293.7469, 371.5811,
        234.8606, 242.8118, 272.9613, 309.7688, 317.4629,
        309.9976, 454.1619, 346.8311, 330.3003, 253.8644,
        283.8424, 289.5550, 276.7693, 299.8097, 297.1710,
        338.1665, 332.0265, 348.8399, 333.3600, 362.0428,
        265.4731, 276.2012, 243.3647, 254.6723, 279.0244,
        284.1912, 305.5248, 331.5229, 335.7469, 377.2990,
        241.6083, 273.9472, 254.4907, 270.8021, 251.4519,
        254.6362, 245.4523, 235.3110, 235.7541, 237.2466,
        312.3666, 313.8058, 291.6112, 346.1222, 365.7324,
        391.8385, 404.2601, 416.6923, 455.8643, 458.9167,
        236.1032, 230.3167, 238.9256, 254.9220, 250.7103,
        269.7744, 281.5648, 308.1020, 336.2806, 351.6451,
        256.2968, 243.4543, 256.2046, 255.5271, 268.9165,
        329.7247, 379.4445, 362.9184, 394.4872, 389.0527,
        250.5265, 300.0576, 269.8939, 280.5891, 271.8274,
        304.6336, 287.7466, 266.5955, 321.5418, 347.5655,
        221.6771, 298.1939, 326.8785, 346.8555, 348.7402,
        352.8287, 354.4266, 360.4326, 375.6406, 388.5417,
        271.9235, 268.4369, 257.2424, 277.6566, 314.8222,
        317.2135, 298.1353, 348.1229, 340.2800, 366.5131,
        225.2640, 234.5235, 238.9008, 240.4730, 267.5373,
        344.1937, 281.1481, 347.5855, 365.1630, 372.2288,
        269.8804, 272.4428, 277.8989, 281.7895, 279.1705,
        284.5120, 259.2658, 304.6306, 350.7807, 369.4692,
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

fn penicillin_fixture() -> DataFrame {
    #[rustfmt::skip]
    let diameter: Vec<f64> = vec![
        27.0, 23.0, 26.0, 23.0, 23.0, 21.0,
        27.0, 23.0, 26.0, 23.0, 23.0, 21.0,
        25.0, 21.0, 25.0, 24.0, 24.0, 20.0,
        26.0, 23.0, 25.0, 23.0, 23.0, 20.0,
        25.0, 22.0, 26.0, 22.0, 23.0, 20.0,
        24.0, 22.0, 25.0, 23.0, 22.0, 19.0,
        24.0, 20.0, 23.0, 21.0, 22.0, 19.0,
        26.0, 22.0, 26.0, 24.0, 24.0, 21.0,
        24.0, 21.0, 24.0, 22.0, 22.0, 20.0,
        24.0, 21.0, 24.0, 23.0, 22.0, 19.0,
        26.0, 23.0, 26.0, 24.0, 24.0, 21.0,
        25.0, 22.0, 26.0, 24.0, 24.0, 20.0,
        26.0, 24.0, 26.0, 24.0, 25.0, 22.0,
        26.0, 23.0, 26.0, 23.0, 23.0, 20.0,
        26.0, 23.0, 25.0, 24.0, 24.0, 22.0,
        25.0, 22.0, 25.0, 23.0, 23.0, 20.0,
        25.0, 21.0, 24.0, 23.0, 23.0, 20.0,
        25.0, 22.0, 24.0, 23.0, 23.0, 19.0,
        24.0, 21.0, 23.0, 21.0, 21.0, 19.0,
        26.0, 23.0, 26.0, 24.0, 24.0, 21.0,
        25.0, 21.0, 24.0, 22.0, 22.0, 18.0,
        25.0, 22.0, 25.0, 22.0, 22.0, 20.0,
        24.0, 21.0, 24.0, 22.0, 24.0, 19.0,
        24.0, 21.0, 24.0, 22.0, 21.0, 18.0,
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
