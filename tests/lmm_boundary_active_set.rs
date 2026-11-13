use mixeff_rs::formula::parse_formula;
use mixeff_rs::model::data::DataFrame;
use mixeff_rs::model::linear::{
    BoundaryInferencePolicy, ContrastMatrix, FitOptions, FitToleranceOverrides,
    FixedEffectHypothesis, FixedEffectTestMethod, LinearMixedModel, OptimizerControl,
    ReliabilityGrade,
};
use mixeff_rs::model::traits::MixedModelFit;

// Simulated fixture supplied in https://github.com/bbuchsbaum/mixeff-rs/issues/19.
// R set.seed(24), 34 subjects x 24 trials, y centered within subject.
fn boundary_data() -> DataFrame {
    let mut subject = Vec::new();
    let mut x = Vec::new();
    let mut y = Vec::new();
    for row in csv::Reader::from_reader(include_bytes!("fixtures/boundary_collapse.csv").as_slice())
        .records()
    {
        let row = row.unwrap();
        subject.push(row[0].to_string());
        x.push(row[2].parse::<f64>().unwrap());
        y.push(row[3].parse::<f64>().unwrap());
    }
    let mut data = DataFrame::new();
    data.add_categorical("subject", subject).unwrap();
    data.add_numeric("x", x).unwrap();
    data.add_numeric("y", y).unwrap();
    data
}

#[test]
fn weak_slope_survives_a_zero_intercept_variance() {
    let data = boundary_data();
    let mut reduced = LinearMixedModel::new(
        parse_formula("y ~ x + (0 + x | subject)").unwrap(),
        &data,
        None,
    )
    .unwrap();
    reduced.fit(true).unwrap();
    for formula in [
        "y ~ x + (1 | subject) + (0 + x | subject)",
        "y ~ x + (1 + x || subject)",
        "y ~ x + (0 + x | subject) + (1 | subject)",
    ] {
        let mut model =
            LinearMixedModel::new(parse_formula(formula).unwrap(), &data, None).unwrap();
        model.fit(true).unwrap();
        assert!(
            model.objective() <= reduced.objective() + 1e-6,
            "{formula}: full={}, reduced={}",
            model.objective(),
            reduced.objective()
        );
        // Independently evaluated with lme4 2.0.1. The tolerance
        // allows optimizer termination differences, not the 1.56-unit bug.
        assert!((model.objective() - 2311.4456015819).abs() < 1e-6);
        let kkt = model.scalar_covariance_kkt_certificate().unwrap();
        assert!(kkt
            .blocks
            .iter()
            .filter(|b| b.normalized_variance <= kkt.variance_tolerance)
            .all(|b| b.normalized_score >= -kkt.score_tolerance));
    }
}

#[test]
fn boundary_recovery_survives_predictor_and_response_rescaling() {
    // Isolate inference from objective-relative optimizer termination: the
    // ordinary stop can leave a score that correctly fails strict ActiveSet
    // stationarity after a rescaled refit, despite an almost identical loss.
    let tight_options = || {
        FitOptions::reml().with_optimizer_control(
            OptimizerControl::auto()
                .with_optimizer(mixeff_rs::types::Optimizer::TrustBq)
                .with_tolerances(
                    FitToleranceOverrides::default()
                        .with_ftol_rel(1e-12)
                        .with_ftol_abs(1e-12)
                        .with_xtol_abs(vec![1e-10; 2]),
                ),
        )
    };
    let original = boundary_data();
    let mut reference = LinearMixedModel::new(
        parse_formula("y ~ x + (1 + x || subject)").unwrap(),
        &original,
        None,
    )
    .unwrap();
    reference.fit_with_options(tight_options()).unwrap();
    let methods = [
        FixedEffectTestMethod::Satterthwaite,
        FixedEffectTestMethod::KenwardRoger,
    ];
    let references = methods.map(|method| {
        reference.test_contrast_with_boundary_policy(
            reference.coefficient_hypotheses().remove(1),
            method,
            BoundaryInferencePolicy::ActiveSet,
        )
    });
    let joint = FixedEffectHypothesis::zero_rhs(
        "joint",
        ContrastMatrix::new(nalgebra::DMatrix::identity(2, 2)).unwrap(),
    );
    let joint_reference = reference.test_contrast_with_boundary_policy(
        joint.clone(),
        FixedEffectTestMethod::KenwardRoger,
        BoundaryInferencePolicy::ActiveSet,
    );
    for (xscale, yscale) in [
        (0.1, 10.0),
        (10.0, 0.1),
        (100.0, 1.0),
        (1000.0, 1.0),
        (0.001, 1.0),
        (1.0, 1e-6),
        (1.0, 1e6),
    ] {
        let mut data = original.clone();
        data.add_numeric(
            "x",
            original
                .numeric("x")
                .unwrap()
                .iter()
                .map(|x| x * xscale)
                .collect(),
        )
        .unwrap();
        data.add_numeric(
            "y",
            original
                .numeric("y")
                .unwrap()
                .iter()
                .map(|y| y * yscale)
                .collect(),
        )
        .unwrap();
        let mut reduced = LinearMixedModel::new(
            parse_formula("y ~ x + (0 + x | subject)").unwrap(),
            &data,
            None,
        )
        .unwrap();
        let mut full = LinearMixedModel::new(
            parse_formula("y ~ x + (1 + x || subject)").unwrap(),
            &data,
            None,
        )
        .unwrap();
        reduced.fit(true).unwrap();
        full.fit_with_options(tight_options()).unwrap();
        assert!(
            full.objective() <= reduced.objective() + 1e-6,
            "scales={xscale},{yscale}: {} vs {}",
            full.objective(),
            reduced.objective()
        );
        let expected = 2311.4456015819 + 2.0 * xscale.ln() + 2.0 * 814.0 * yscale.ln();
        assert!(
            (full.objective() - expected).abs() < 1e-5,
            "scales={xscale},{yscale}: {} vs {expected}",
            full.objective()
        );
        for (method, reference) in methods.into_iter().zip(&references) {
            let result = full.test_contrast_with_boundary_policy(
                full.coefficient_hypotheses().remove(1),
                method,
                BoundaryInferencePolicy::ActiveSet,
            );
            assert!(
                result.conditioning_applied,
                "scales={xscale},{yscale}, {method:?}: {result:?}"
            );
            let actual = &result.test;
            let expected = &reference.test;
            assert_eq!(result.fixed_theta_indices, reference.fixed_theta_indices);
            assert_eq!(result.fixed_components, reference.fixed_components);
            let expected_se = expected.standard_errors[0].unwrap() * yscale / xscale;
            assert!((actual.standard_errors[0].unwrap() / expected_se - 1.0).abs() < 1e-5);
            assert!(
                (actual.denominator_df.unwrap() - expected.denominator_df.unwrap()).abs() < 1e-3,
                "scales={xscale},{yscale}, {method:?}: df {:?} vs {:?}",
                actual.denominator_df,
                expected.denominator_df
            );
            assert!(
                (actual.p_values[0].unwrap() - expected.p_values[0].unwrap()).abs() < 1e-6,
                "scales={xscale},{yscale}, {method:?}: p {:?} vs {:?}",
                actual.p_values,
                expected.p_values
            );
        }
        let joint_result = full.test_contrast_with_boundary_policy(
            joint.clone(),
            FixedEffectTestMethod::KenwardRoger,
            BoundaryInferencePolicy::ActiveSet,
        );
        assert!(
            joint_result.conditioning_applied,
            "scales={xscale},{yscale}: {joint_result:?}"
        );
        assert_eq!(joint_result.test.hypothesis, joint);
        assert!(
            (joint_result.test.denominator_df.unwrap()
                - joint_reference.test.denominator_df.unwrap())
            .abs()
                < 1e-3
        );
        assert!(
            (joint_result.test.p_values[0].unwrap() - joint_reference.test.p_values[0].unwrap())
                .abs()
                < 1e-6,
            "joint KR scales={xscale},{yscale}: p {:?} vs {:?}, theta {:?}",
            joint_result.test.p_values,
            joint_reference.test.p_values,
            full.theta()
        );
    }
}

#[test]
fn active_set_matches_explicit_reduced_model_and_preserves_default() {
    let data = boundary_data();
    let mut reduced = LinearMixedModel::new(
        parse_formula("y ~ x + (0 + x | subject)").unwrap(),
        &data,
        None,
    )
    .unwrap();
    reduced.fit(true).unwrap();
    for formula in [
        "y ~ x + (1 | subject) + (0 + x | subject)",
        "y ~ x + (1 + x || subject)",
    ] {
        let mut model =
            LinearMixedModel::new(parse_formula(formula).unwrap(), &data, None).unwrap();
        model.fit(true).unwrap();
        let theta = model.theta();
        let objective = model.objective();
        for method in [
            FixedEffectTestMethod::Satterthwaite,
            FixedEffectTestMethod::KenwardRoger,
        ] {
            let joint = FixedEffectHypothesis::zero_rhs(
                "joint",
                ContrastMatrix::new(nalgebra::DMatrix::identity(2, 2)).unwrap(),
            );
            let mut hypotheses = model.coefficient_hypotheses();
            hypotheses.push(joint.clone());
            let mut references = reduced.coefficient_hypotheses();
            references.push(joint);
            for (h, rh) in hypotheses.into_iter().zip(references) {
                let default = model.test_contrast_with_method(h.clone(), method);
                let reference = reduced.test_contrast_with_method(rh, method);
                let conditional = model.test_contrast_with_boundary_policy(
                    h.clone(),
                    method,
                    BoundaryInferencePolicy::ActiveSet,
                );
                let test = &conditional.test;
                assert!(
                    test.status.p_value_available(),
                    "{formula} {method:?}: {conditional:?}"
                );
                assert_eq!(conditional.fixed_theta_indices, vec![0]);
                assert_eq!(test.reliability, ReliabilityGrade::Low);
                assert_eq!(test.method, reference.method);
                assert!(
                    (test.denominator_df.unwrap() - reference.denominator_df.unwrap()).abs() < 0.02,
                    "{method:?}: df {:?} vs {:?}",
                    test.denominator_df,
                    reference.denominator_df
                );
                assert!(
                    (test.standard_errors[0].unwrap() - reference.standard_errors[0].unwrap())
                        .abs()
                        < 1e-6
                );
                assert!((test.p_values[0].unwrap() - reference.p_values[0].unwrap()).abs() < 1e-5);
                if let (Some(actual), Some(expected)) = (
                    test.kenward_roger_f_scaling,
                    reference.kenward_roger_f_scaling,
                ) {
                    assert!((actual.f_scaling - expected.f_scaling).abs() < 1e-5);
                }
                if test.hypothesis.label == "x" {
                    let (df, se, p) = match method {
                        FixedEffectTestMethod::Satterthwaite => {
                            (37.96719309001693, 0.03930555189007148, 0.01787728419697667)
                        }
                        _ => (31.34118800354958, 0.03941000992923434, 0.01919515610095627),
                    };
                    // Independent lmerTest 3.2.1 / pbkrtest 0.5.5 oracle.
                    assert!((test.denominator_df.unwrap() - df).abs() < 0.02);
                    assert!((test.standard_errors[0].unwrap() - se).abs() < 1e-6);
                    assert!((test.p_values[0].unwrap() - p).abs() < 1e-5);
                }
                assert_eq!(model.test_contrast_with_method(h, method), default);
                let encoded = serde_json::to_value(&conditional).unwrap();
                assert_eq!(encoded["fixed_theta_indices"], serde_json::json!([0]));
            }
        }
        assert_eq!(model.theta(), theta);
        assert_eq!(model.objective(), objective);
    }
}

#[test]
fn unsupported_active_set_refuses_even_if_ordinary_kr_is_available() {
    let data = boundary_data();
    let mut model = LinearMixedModel::new(
        parse_formula("y ~ x + (1 + x | subject)").unwrap(),
        &data,
        None,
    )
    .unwrap();
    model.fit(true).unwrap();
    let hypothesis = model.coefficient_hypotheses().remove(1);
    let baseline =
        model.test_contrast_with_method(hypothesis.clone(), FixedEffectTestMethod::KenwardRoger);
    assert!(baseline.status.p_value_available(), "{baseline:?}");
    let conditional = model.test_contrast_with_boundary_policy(
        hypothesis,
        FixedEffectTestMethod::KenwardRoger,
        BoundaryInferencePolicy::ActiveSet,
    );
    assert!(!conditional.conditioning_applied);
    assert!(
        !conditional.test.status.p_value_available(),
        "{conditional:?}"
    );
    assert!(
        conditional
            .test
            .p_value_unavailable_reason()
            .unwrap()
            .contains("independent scalar or diagonal"),
        "{conditional:?}"
    );
}

#[test]
fn all_zero_random_variances_recover_linear_model_df() {
    let mut data = DataFrame::new();
    let n = 40;
    data.add_numeric("y", (0..n).map(|i| [-2.0, -1.0, 1.0, 2.0][i % 4]).collect())
        .unwrap();
    data.add_categorical("g", (0..n).map(|i| format!("g{}", i / 4)).collect())
        .unwrap();
    let mut model =
        LinearMixedModel::new(parse_formula("y ~ 1 + (1 | g)").unwrap(), &data, None).unwrap();
    model.fit(true).unwrap();
    assert!(model.theta()[0] < 1e-7);
    // Malformed multirow hypotheses retain the ordinary refusal, never panic.
    let invalid = FixedEffectHypothesis::zero_rhs(
        "wrong width",
        ContrastMatrix::new(nalgebra::DMatrix::identity(2, 2)).unwrap(),
    );
    for method in [
        FixedEffectTestMethod::Satterthwaite,
        FixedEffectTestMethod::KenwardRoger,
    ] {
        let default_refusal = model.test_contrast_with_method(invalid.clone(), method);
        let refusal = model.test_contrast_with_boundary_policy(
            invalid.clone(),
            method,
            BoundaryInferencePolicy::ActiveSet,
        );
        assert_eq!(refusal.test.status, default_refusal.status);
        assert_eq!(refusal.test.method, default_refusal.method);
        assert_eq!(refusal.test.p_values, default_refusal.p_values);
        assert!(!refusal.conditioning_applied);
        let result = model.test_contrast_with_boundary_policy(
            model.coefficient_hypotheses().remove(0),
            method,
            BoundaryInferencePolicy::ActiveSet,
        );
        assert!(result.test.status.p_value_available(), "{result:?}");
        assert!((result.test.denominator_df.unwrap() - 39.0).abs() < 1e-4);
        assert!(
            (result.test.standard_errors[0].unwrap() - (100.0_f64 / 39.0 / 40.0).sqrt()).abs()
                < 1e-8
        );
    }
}

fn null_boundary_data(rng: &mut rand::rngs::StdRng, intercept_sd: f64) -> DataFrame {
    use rand_distr::{Distribution, StandardNormal};
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut g = Vec::new();
    for subject in 0..34 {
        let intercept: f64 = StandardNormal.sample(&mut *rng);
        let slope: f64 = StandardNormal.sample(&mut *rng);
        for _ in 0..24 {
            let predictor: f64 = StandardNormal.sample(&mut *rng);
            let residual: f64 = StandardNormal.sample(&mut *rng);
            x.push(predictor);
            y.push(intercept_sd * intercept + 0.15 * slope * predictor + residual);
            g.push(format!("g{subject}"));
        }
    }
    let mut data = DataFrame::new();
    data.add_numeric("x", x).unwrap();
    data.add_numeric("y", y).unwrap();
    data.add_categorical("g", g).unwrap();
    data
}

#[test]
#[cfg(feature = "unstable-internals")]
fn recovery_does_not_leave_a_second_component_at_an_invalid_zero() {
    use rand::{rngs::StdRng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(20261113);
    for replicate in 0..10 {
        let data = null_boundary_data(&mut rng, 0.1);
        let mut model = LinearMixedModel::new(
            parse_formula("y ~ x + (1 | g) + (0 + x | g)").unwrap(),
            &data,
            None,
        )
        .unwrap();
        model.fit(true).unwrap();
        let fitted = model.objective();
        // Independent feasible objective witnesses, including both positive
        // variances. A local stop may never be worse than one of these points.
        for theta in [[0.05, 0.1], [0.1, 0.15], [0.15, 0.2]] {
            let probe = model.objective_at(&theta).unwrap();
            assert!(
                fitted <= probe + 1e-6,
                "replicate={replicate}: fit={fitted}, probe={probe}, theta={theta:?}"
            );
        }
    }
}

/// A reproducible calibration experiment, not a nominal-size assertion.
/// Run explicitly; report refusals and boundary selection along with rejections.
#[test]
#[ignore = "null calibration experiment: 400 simulated fits"]
fn active_set_null_calibration() {
    use rand::{rngs::StdRng, SeedableRng};
    for intercept_sd in [0.0, 0.1] {
        let mut rng = StdRng::seed_from_u64(20261113);
        let mut available = 0;
        let mut selected = 0;
        let mut rejected = 0;
        let mut failures = 0;
        let mut conditional = 0;
        let mut conditional_rejected = 0;
        let mut refusals = std::collections::BTreeMap::new();
        let mut refused_stops = std::collections::BTreeMap::new();
        for _ in 0..200 {
            let data = null_boundary_data(&mut rng, intercept_sd);
            let mut model = LinearMixedModel::new(
                parse_formula("y ~ x + (1 | g) + (0 + x | g)").unwrap(),
                &data,
                None,
            )
            .unwrap();
            if model.fit(true).is_err() {
                failures += 1;
                continue;
            }
            let result = model.test_contrast_with_boundary_policy(
                model.coefficient_hypotheses().remove(1),
                FixedEffectTestMethod::Satterthwaite,
                BoundaryInferencePolicy::ActiveSet,
            );
            selected += usize::from(!result.fixed_theta_indices.is_empty());
            if let Some(p) = result.test.p_values[0] {
                available += 1;
                rejected += usize::from(p < 0.05);
                if result.conditioning_applied {
                    conditional += 1;
                    conditional_rejected += usize::from(p < 0.05);
                }
            } else {
                let reason = result
                    .test
                    .p_value_unavailable_reason()
                    .unwrap_or("unspecified");
                *refusals.entry(reason.to_string()).or_insert(0) += 1;
                if let Some(certificate) = model.optimizer_certificate() {
                    let stop = format!(
                        "{:?}: {:?}",
                        certificate.status, certificate.evidence.optimizer_stop.return_code
                    );
                    *refused_stops.entry(stop).or_insert(0) += 1;
                }
            }
        }
        eprintln!("intercept_sd={intercept_sd}, replicates=200, fit_errors={failures}, available={available}, boundary_selected={selected}, p_below_0.05={rejected}");
        eprintln!("conditional_available={conditional}, conditional_p_below_0.05={conditional_rejected}, refusals={refusals:?}");
        eprintln!("refused_stops={refused_stops:?}");
        assert!(available > 0, "experiment produced no usable tests");
    }
}
