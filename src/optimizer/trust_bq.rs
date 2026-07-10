use nalgebra::{DMatrix, DVector};

use crate::error::{MixedModelError, Result};

const MIN_STEP: f64 = 1e-12;

#[derive(Debug, Clone)]
pub(crate) struct TrustBqOptions {
    pub(crate) initial_radius: f64,
    pub(crate) final_radius: f64,
    pub(crate) max_evaluations: usize,
    pub(crate) ftol_abs: f64,
    pub(crate) ftol_rel: f64,
    /// Require the trust region to contract by at least four halvings before
    /// objective-tolerance or stagnation stops are accepted. Profiled/joint
    /// mixed-model drivers enable this to avoid coarse-radius false FTOL
    /// stops; auxiliary sub-solves keep the historical eager FTOL behavior.
    pub(crate) ftol_requires_local_radius: bool,
    pub(crate) eta_accept: f64,
    pub(crate) eta_expand: f64,
    pub(crate) shrink_factor: f64,
    pub(crate) expand_factor: f64,
    pub(crate) max_cross_terms: usize,
    /// Number of consecutive iterations with no meaningful objective or
    /// parameter progress before the search is declared converged. This is the
    /// early-stop guard for problems (e.g. crossed LMMs) that reach an
    /// objective-clean point but keep rebuilding the interpolation model while
    /// the trust region shrinks, exhausting the evaluation budget without the
    /// accepted-step `ObjectiveTolerance` check ever firing.
    pub(crate) stall_iterations: usize,
    /// Relative tolerance for the early-stop stagnation test. A negative value
    /// (the default) means "inherit `ftol_rel`", which keeps the math unit
    /// tests on exactly their previous behavior; callers that want a looser
    /// statistical convergence band (the profiled-likelihood LMM path) set
    /// this explicitly.
    pub(crate) stall_ftol_rel: f64,
    /// Absolute tolerance for the early-stop stagnation test. A negative value
    /// (the default) means "inherit `ftol_abs`".
    pub(crate) stall_ftol_abs: f64,
    /// When true (the default), the stagnation counter is also reset by
    /// parameter movement, so the search only stops when the objective *and*
    /// the parameters are settled. The profiled-likelihood LMM path sets this
    /// false: on a flat objective ridge (common for crossed designs) theta
    /// keeps drifting while the fit is already converged, so requiring stable
    /// theta would defeat the early stop.
    pub(crate) stall_requires_stable_x: bool,
    /// When true, objective evaluations are memoized within a single
    /// `minimize` call keyed on the exact bit pattern of the input point, so a
    /// point probed in an earlier iteration (e.g. a recurring one-sided
    /// second-difference probe across radius-shrink chains, or a reused trial
    /// point) is not re-evaluated. Because the objective is pure and
    /// deterministic, a cache hit returns the identical value — this is exact
    /// memoization, never a stale approximation, so it cannot perturb the
    /// optimization path; it only lowers the objective-evaluation count.
    /// Default `false` keeps the evaluation count byte-identical to the
    /// previous behavior for callers that do not opt in.
    pub(crate) reuse_samples: bool,
}

impl Default for TrustBqOptions {
    fn default() -> Self {
        Self {
            initial_radius: 0.75,
            final_radius: 1e-6,
            max_evaluations: 1000,
            ftol_abs: 1e-10,
            ftol_rel: 1e-10,
            ftol_requires_local_radius: false,
            eta_accept: 0.05,
            eta_expand: 0.75,
            shrink_factor: 0.5,
            expand_factor: 1.8,
            max_cross_terms: usize::MAX,
            stall_iterations: 4,
            stall_ftol_rel: -1.0,
            stall_ftol_abs: -1.0,
            stall_requires_stable_x: true,
            reuse_samples: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrustBqStopReason {
    RadiusBelowTolerance,
    ObjectiveTolerance,
    MaxEvaluations,
    StepBelowTolerance,
    /// Best objective and parameters have not improved for
    /// `stall_iterations` consecutive iterations while the trust region is
    /// already contracting. Treated as a converged stop.
    ObjectiveStagnation,
    /// A caller-owned convergence certificate accepted the current best point.
    CertifiedConvergence,
}

impl TrustBqStopReason {
    pub(crate) fn trace_classification(self) -> TrustBqTraceClassification {
        match self {
            TrustBqStopReason::RadiusBelowTolerance
            | TrustBqStopReason::ObjectiveTolerance
            | TrustBqStopReason::StepBelowTolerance => {
                TrustBqTraceClassification::SmoothConvergence
            }
            TrustBqStopReason::ObjectiveStagnation => TrustBqTraceClassification::StatisticalStall,
            TrustBqStopReason::CertifiedConvergence => {
                TrustBqTraceClassification::CertificateAccepted
            }
            TrustBqStopReason::MaxEvaluations => TrustBqTraceClassification::BudgetExhaustion,
        }
    }

    pub(crate) fn is_acceptable_convergence(self) -> bool {
        !matches!(
            self.trace_classification(),
            TrustBqTraceClassification::BudgetExhaustion
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrustBqTraceClassification {
    /// Trust-region math reached a standard local convergence stop.
    SmoothConvergence,
    /// Objective movement stayed inside the caller's statistical stall band.
    StatisticalStall,
    /// The caller-owned certificate accepted the current best point.
    CertificateAccepted,
    /// The optimizer exhausted its evaluation budget before convergence.
    BudgetExhaustion,
}

impl TrustBqTraceClassification {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            TrustBqTraceClassification::SmoothConvergence => "smooth_convergence",
            TrustBqTraceClassification::StatisticalStall => "statistical_stall",
            TrustBqTraceClassification::CertificateAccepted => "certificate_accepted",
            TrustBqTraceClassification::BudgetExhaustion => "budget_exhaustion",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TrustBqResult {
    pub(crate) x: Vec<f64>,
    pub(crate) fmin: f64,
    pub(crate) fevals: usize,
    pub(crate) iterations: usize,
    pub(crate) final_radius: f64,
    pub(crate) stop_reason: TrustBqStopReason,
    pub(crate) last_model_sample_count: usize,
}

impl TrustBqResult {
    pub(crate) fn trace_classification(&self) -> TrustBqTraceClassification {
        self.stop_reason.trace_classification()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TrustBqProgress<'a> {
    pub(crate) x: &'a [f64],
    pub(crate) fmin: f64,
    pub(crate) fevals: usize,
    pub(crate) radius: f64,
}

#[derive(Debug, Clone)]
struct SideSample {
    delta: f64,
    value: f64,
}

#[derive(Debug, Clone)]
struct QuadraticInterpolationModel {
    gradient: DVector<f64>,
    hessian: DMatrix<f64>,
    sample_count: usize,
}

impl QuadraticInterpolationModel {
    fn predicted_reduction(&self, step: &[f64]) -> f64 {
        let s = DVector::from_column_slice(step);
        let linear = self.gradient.dot(&s);
        let quadratic = 0.5 * s.dot(&(&self.hessian * &s));
        -(linear + quadratic)
    }
}

#[cfg(test)]
pub(crate) fn minimize<F>(
    initial: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    options: TrustBqOptions,
    mut objective: F,
) -> Result<TrustBqResult>
where
    F: FnMut(&[f64]) -> Result<f64>,
{
    minimize_with_progress(
        initial,
        lower_bounds,
        upper_bounds,
        options,
        &mut objective,
        |_| Ok(false),
    )
}

pub(crate) fn minimize_with_progress<F, P>(
    initial: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    options: TrustBqOptions,
    mut objective: F,
    mut progress: P,
) -> Result<TrustBqResult>
where
    F: FnMut(&[f64]) -> Result<f64>,
    P: FnMut(&TrustBqProgress<'_>) -> Result<bool>,
{
    validate_problem(initial, lower_bounds, upper_bounds, &options)?;

    let mut fevals = 0usize;
    let reuse = options.reuse_samples;
    let mut cache: SampleCache = SampleCache::new();
    let mut x = project_point(initial, lower_bounds, upper_bounds);
    let mut f = evaluate(&mut objective, &x, &mut fevals, &mut cache, reuse)?;
    // Objective at the starting point. The stagnation early-stop is a
    // *plateau-after-descent* detector, so it must not fire until the search
    // has improved on this value at least once (see the gate below).
    let initial_objective = f;
    let mut best_x = x.clone();
    let mut best_f = f;
    let mut radius = options.initial_radius.max(options.final_radius);
    let mut iterations = 0usize;
    let mut last_model_sample_count = 0usize;
    // Early-stop bookkeeping: the snapshot of the best objective/parameters as
    // of the last iteration that made meaningful progress, plus a counter of
    // how many consecutive iterations have failed to beat it.
    let mut stalled = 0usize;
    let mut stall_best_f = best_f;
    let mut stall_best_x = best_x.clone();
    // Stagnation tolerances default to the optimizer's `ftol_*` (preserving
    // prior behavior for callers/tests that do not opt in) but can be set
    // looser by callers whose convergence band is statistical rather than
    // numeric.
    let stall_ftol_rel = if options.stall_ftol_rel >= 0.0 {
        options.stall_ftol_rel
    } else {
        options.ftol_rel
    };
    let stall_ftol_abs = if options.stall_ftol_abs >= 0.0 {
        options.stall_ftol_abs
    } else {
        options.ftol_abs
    };
    // FTOL/stagnation only certify a local model after the trust region has
    // contracted materially from its startup scale. See the accepted-step
    // check below for the production failure this guards.
    let ftol_radius = (options.initial_radius / 16.0).max(options.final_radius);
    loop {
        if fevals >= options.max_evaluations {
            return Ok(TrustBqResult {
                x: best_x,
                fmin: best_f,
                fevals,
                iterations,
                final_radius: radius,
                stop_reason: TrustBqStopReason::MaxEvaluations,
                last_model_sample_count,
            });
        }
        if radius <= options.final_radius {
            return Ok(TrustBqResult {
                x: best_x,
                fmin: best_f,
                fevals,
                iterations,
                final_radius: radius,
                stop_reason: TrustBqStopReason::RadiusBelowTolerance,
                last_model_sample_count,
            });
        }

        if iterations > 0
            && progress(&TrustBqProgress {
                x: &best_x,
                fmin: best_f,
                fevals,
                radius,
            })?
        {
            return Ok(TrustBqResult {
                x: best_x,
                fmin: best_f,
                fevals,
                iterations,
                final_radius: radius,
                stop_reason: TrustBqStopReason::CertifiedConvergence,
                last_model_sample_count,
            });
        }

        // Stagnation-based early stop. Once the trust region has started
        // contracting (`radius < initial_radius`, i.e. we are polishing rather
        // than still descending or expanding), declare convergence if the best
        // objective has not improved beyond the stagnation tolerance — and,
        // when `stall_requires_stable_x` is set, the best parameters have not
        // moved beyond the step tolerance — for `stall_iterations` iterations
        // in a row. This catches the crossed-LMM case where near-optimum steps
        // are repeatedly rejected — shrinking the radius and rebuilding the
        // interpolation model every iteration — so the accepted-step
        // `ObjectiveTolerance` branch never fires and the budget is exhausted.
        let stall_obj_tol = stall_ftol_abs + stall_ftol_rel * stall_best_f.abs().max(1.0);
        let improved_f = (stall_best_f - best_f) > stall_obj_tol;
        let moved_x = options.stall_requires_stable_x
            && best_x
                .iter()
                .zip(stall_best_x.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
                > options.final_radius;
        if improved_f || moved_x {
            stalled = 0;
            stall_best_f = best_f;
            stall_best_x.clone_from(&best_x);
        } else {
            stalled += 1;
        }
        // Only treat a stall as convergence once the search has actually
        // descended below the starting objective. Otherwise — as on the
        // flat high-baseline random-intercept GLMM Laplace surface, where the
        // first trial steps are all rejected because the initial trust radius
        // (sized to repair far-off intercept starts) overshoots a small move —
        // the counter would trip while the radius is still far larger than the
        // move needed, reporting a premature interior convergence at the
        // un-improved start. Withholding the stop lets the radius keep
        // contracting until a step is finally accepted and the descent reaches
        // the true optimum (after which a genuine plateau stalls as intended).
        let has_descended = best_f < initial_objective;
        let stall_radius_is_local = if options.ftol_requires_local_radius {
            radius <= ftol_radius
        } else {
            radius < options.initial_radius
        };
        if has_descended && stalled >= options.stall_iterations && stall_radius_is_local {
            return Ok(TrustBqResult {
                x: best_x,
                fmin: best_f,
                fevals,
                iterations,
                final_radius: radius,
                stop_reason: TrustBqStopReason::ObjectiveStagnation,
                last_model_sample_count,
            });
        }

        iterations += 1;
        let model = build_quadratic_model(
            &x,
            f,
            lower_bounds,
            upper_bounds,
            radius,
            options.max_evaluations,
            options.max_cross_terms,
            &mut fevals,
            &mut objective,
            &mut cache,
            reuse,
        )?;
        last_model_sample_count = model.sample_count;
        if fevals >= options.max_evaluations {
            return Ok(TrustBqResult {
                x: best_x,
                fmin: best_f,
                fevals,
                iterations,
                final_radius: radius,
                stop_reason: TrustBqStopReason::MaxEvaluations,
                last_model_sample_count,
            });
        }

        let step = trust_region_step(&model, &x, lower_bounds, upper_bounds, radius);
        let step_norm = norm(&step);
        if step_norm <= options.final_radius {
            radius *= options.shrink_factor;
            if radius <= options.final_radius {
                return Ok(TrustBqResult {
                    x: best_x,
                    fmin: best_f,
                    fevals,
                    iterations,
                    final_radius: radius,
                    stop_reason: TrustBqStopReason::StepBelowTolerance,
                    last_model_sample_count,
                });
            }
            continue;
        }

        let predicted_reduction = model.predicted_reduction(&step);
        if !predicted_reduction.is_finite() || predicted_reduction <= 0.0 {
            radius *= options.shrink_factor;
            continue;
        }

        let mut trial = x
            .iter()
            .zip(step.iter())
            .map(|(xi, si)| xi + si)
            .collect::<Vec<_>>();
        project_in_place(&mut trial, lower_bounds, upper_bounds);
        if distance(&x, &trial) <= options.final_radius {
            radius *= options.shrink_factor;
            continue;
        }

        let trial_f = evaluate(&mut objective, &trial, &mut fevals, &mut cache, reuse)?;
        if trial_f < best_f {
            best_f = trial_f;
            best_x = trial.clone();
        }

        let actual_reduction = f - trial_f;
        let ratio = actual_reduction / predicted_reduction;
        if ratio >= options.eta_accept && actual_reduction > 0.0 {
            let old_f = f;
            x = trial;
            f = trial_f;

            if ratio >= options.eta_expand && step_norm > 0.8 * radius {
                radius *= options.expand_factor;
            }

            let objective_tol = options.ftol_abs + options.ftol_rel * old_f.abs().max(1.0);
            // A tiny accepted reduction is only meaningful once the
            // interpolation radius is local. At a coarse radius the
            // quadratic model can propose a very short, well-predicted step
            // even when the true objective gradient is still large. Treating
            // that single step as FTOL convergence caused the iamciera
            // two-variance ML fit to stop with radius 8.4e-2 and
            // |gradient|=17.7. Continue contracting until the model has
            // localized by at least four halvings; the existing stagnation
            // and caller-certificate stops remain available in the meantime.
            if actual_reduction.abs() <= objective_tol
                && (!options.ftol_requires_local_radius || radius <= ftol_radius)
            {
                return Ok(TrustBqResult {
                    x: best_x,
                    fmin: best_f,
                    fevals,
                    iterations,
                    final_radius: radius,
                    stop_reason: TrustBqStopReason::ObjectiveTolerance,
                    last_model_sample_count,
                });
            }
        } else {
            radius *= options.shrink_factor;
        }
    }
}

fn validate_problem(
    initial: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    options: &TrustBqOptions,
) -> Result<()> {
    let n = initial.len();
    if n == 0 {
        return Err(MixedModelError::Optimization(
            "TrustBQ requires at least one parameter".to_string(),
        ));
    }
    if lower_bounds.len() != n || upper_bounds.len() != n {
        return Err(MixedModelError::DimensionMismatch(
            "TrustBQ bounds length does not match parameter length".to_string(),
        ));
    }
    if !options.initial_radius.is_finite()
        || options.initial_radius <= 0.0
        || !options.final_radius.is_finite()
        || options.final_radius <= 0.0
        || options.final_radius > options.initial_radius
    {
        return Err(MixedModelError::Optimization(
            "TrustBQ requires 0 < final_radius <= initial_radius".to_string(),
        ));
    }
    if options.max_evaluations == 0 {
        return Err(MixedModelError::Optimization(
            "TrustBQ max_evaluations must be positive".to_string(),
        ));
    }
    if options.stall_iterations == 0 {
        return Err(MixedModelError::Optimization(
            "TrustBQ stall_iterations must be positive".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&options.eta_accept)
        || !(options.eta_accept..=1.0).contains(&options.eta_expand)
        || !(0.0..1.0).contains(&options.shrink_factor)
        || options.expand_factor <= 1.0
        || !options.expand_factor.is_finite()
    {
        return Err(MixedModelError::Optimization(
            "TrustBQ trust-region constants are invalid".to_string(),
        ));
    }
    for i in 0..n {
        if !initial[i].is_finite() {
            return Err(MixedModelError::Optimization(
                "TrustBQ initial point must be finite".to_string(),
            ));
        }
        if lower_bounds[i] > upper_bounds[i] {
            return Err(MixedModelError::Optimization(format!(
                "TrustBQ lower bound exceeds upper bound at index {i}"
            )));
        }
    }
    Ok(())
}

fn build_quadratic_model<F>(
    x: &[f64],
    f0: f64,
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    radius: f64,
    max_evaluations: usize,
    max_cross_terms: usize,
    fevals: &mut usize,
    objective: &mut F,
    cache: &mut SampleCache,
    reuse: bool,
) -> Result<QuadraticInterpolationModel>
where
    F: FnMut(&[f64]) -> Result<f64>,
{
    let n = x.len();
    let mut gradient = DVector::zeros(n);
    let mut hessian = DMatrix::zeros(n, n);
    let mut sample_count = 0usize;
    let mut side_samples: Vec<Option<SideSample>> = vec![None; n];
    let min_step = MIN_STEP * radius.max(1.0);

    for i in 0..n {
        if *fevals >= max_evaluations {
            break;
        }

        let h_plus = feasible_axis_delta(x[i], lower_bounds[i], upper_bounds[i], radius, 1.0);
        let h_minus = feasible_axis_delta(x[i], lower_bounds[i], upper_bounds[i], radius, -1.0);

        let plus = if h_plus.abs() > min_step && *fevals < max_evaluations {
            Some((
                h_plus,
                evaluate_axis(objective, x, i, h_plus, fevals, cache, reuse).inspect(|_| {
                    sample_count += 1;
                })?,
            ))
        } else {
            None
        };
        let minus = if h_minus.abs() > min_step && *fevals < max_evaluations {
            Some((
                h_minus,
                evaluate_axis(objective, x, i, h_minus, fevals, cache, reuse).inspect(|_| {
                    sample_count += 1;
                })?,
            ))
        } else {
            None
        };

        match (plus, minus) {
            (Some((hp, fp)), Some((hm, fm))) => {
                let a = hp.abs();
                let b = hm.abs();
                let fp_delta = fp - f0;
                let fm_delta = fm - f0;
                let denom = a * b * (a + b);
                gradient[i] = (b * b * fp_delta - a * a * fm_delta) / denom;
                hessian[(i, i)] = 2.0 * (b * fp_delta + a * fm_delta) / denom;
                side_samples[i] = Some(select_cross_side(hp, fp, hm, fm, f0));
            }
            (Some((delta, value)), None) | (None, Some((delta, value))) => {
                let mut slope = (value - f0) / delta;
                let mut curvature = 0.0;
                let second_delta = 2.0 * delta;
                if axis_delta_is_feasible(
                    x[i],
                    lower_bounds[i],
                    upper_bounds[i],
                    second_delta,
                    radius * 2.0,
                ) && *fevals < max_evaluations
                {
                    let second =
                        evaluate_axis(objective, x, i, second_delta, fevals, cache, reuse)?;
                    sample_count += 1;
                    curvature = (second - 2.0 * value + f0) / (delta * delta);
                    slope = (value - f0 - 0.5 * curvature * delta * delta) / delta;
                }
                gradient[i] = slope;
                hessian[(i, i)] = curvature;
                side_samples[i] = Some(SideSample { delta, value });
            }
            (None, None) => {}
        }
    }

    let mut cross_terms = 0usize;
    'cross: for i in 0..n {
        for j in (i + 1)..n {
            if *fevals >= max_evaluations {
                break;
            }
            if cross_terms >= max_cross_terms {
                break 'cross;
            }
            let (Some(left), Some(right)) = (&side_samples[i], &side_samples[j]) else {
                continue;
            };
            let mut trial = x.to_vec();
            trial[i] += left.delta;
            trial[j] += right.delta;
            project_in_place(&mut trial, lower_bounds, upper_bounds);
            if distance(x, &trial) <= min_step {
                continue;
            }
            let f_ij = evaluate(objective, &trial, fevals, cache, reuse)?;
            sample_count += 1;
            let cross = (f_ij - left.value - right.value + f0) / (left.delta * right.delta);
            if cross.is_finite() {
                hessian[(i, j)] = cross;
                hessian[(j, i)] = cross;
                cross_terms += 1;
            }
        }
    }

    Ok(QuadraticInterpolationModel {
        gradient,
        hessian,
        sample_count,
    })
}

fn select_cross_side(hp: f64, fp: f64, hm: f64, fm: f64, f0: f64) -> SideSample {
    let plus_improvement = f0 - fp;
    let minus_improvement = f0 - fm;
    if plus_improvement >= minus_improvement {
        SideSample {
            delta: hp,
            value: fp,
        }
    } else {
        SideSample {
            delta: hm,
            value: fm,
        }
    }
}

fn trust_region_step(
    model: &QuadraticInterpolationModel,
    x: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    radius: f64,
) -> Vec<f64> {
    let n = x.len();
    if norm(model.gradient.as_slice()) <= MIN_STEP {
        return vec![0.0; n];
    }

    let identity = DMatrix::identity(n, n);
    let mut shift = 0.0;
    for _ in 0..14 {
        let shifted = if shift == 0.0 {
            model.hessian.clone()
        } else {
            &model.hessian + shift * &identity
        };
        if let Some(cholesky) = shifted.cholesky() {
            let rhs = -&model.gradient;
            let step_vec = cholesky.solve(&rhs);
            let mut step = step_vec.iter().copied().collect::<Vec<_>>();
            bound_and_scale_step(&mut step, x, lower_bounds, upper_bounds, radius);
            if norm(&step) > MIN_STEP && model.predicted_reduction(&step) > 0.0 {
                return step;
            }
        }
        shift = if shift == 0.0 { 1e-8 } else { shift * 10.0 };
    }

    cauchy_step(model, x, lower_bounds, upper_bounds, radius)
}

fn cauchy_step(
    model: &QuadraticInterpolationModel,
    x: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    radius: f64,
) -> Vec<f64> {
    let n = x.len();
    let gnorm = model.gradient.norm();
    if gnorm <= MIN_STEP {
        return vec![0.0; n];
    }

    let direction = model
        .gradient
        .iter()
        .map(|value| -value / gnorm)
        .collect::<Vec<_>>();
    let hd = &model.hessian * DVector::from_column_slice(&direction);
    let curvature = DVector::from_column_slice(&direction).dot(&hd);
    let alpha_model = if curvature > 0.0 {
        (gnorm / curvature).min(radius)
    } else {
        radius
    };
    let alpha_bound = max_bound_step(x, &direction, lower_bounds, upper_bounds, radius);
    let alpha = alpha_model.min(alpha_bound).max(0.0);
    direction
        .into_iter()
        .map(|value| alpha * value)
        .collect::<Vec<_>>()
}

fn max_bound_step(
    x: &[f64],
    direction: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    radius: f64,
) -> f64 {
    let mut alpha = radius;
    for i in 0..x.len() {
        if direction[i] > 0.0 && upper_bounds[i].is_finite() {
            alpha = alpha.min((upper_bounds[i] - x[i]) / direction[i]);
        } else if direction[i] < 0.0 && lower_bounds[i].is_finite() {
            alpha = alpha.min((lower_bounds[i] - x[i]) / direction[i]);
        }
    }
    alpha.max(0.0)
}

fn feasible_axis_delta(value: f64, lower: f64, upper: f64, radius: f64, sign: f64) -> f64 {
    let mut delta = sign.signum() * radius;
    if delta > 0.0 && upper.is_finite() {
        delta = delta.min(upper - value);
    } else if delta < 0.0 && lower.is_finite() {
        delta = delta.max(lower - value);
    }
    delta
}

fn axis_delta_is_feasible(
    value: f64,
    lower: f64,
    upper: f64,
    delta: f64,
    max_abs_delta: f64,
) -> bool {
    delta.abs() <= max_abs_delta
        && value + delta >= lower
        && value + delta <= upper
        && delta.abs() > MIN_STEP
}

/// Within-`minimize` memoization keyed on the exact IEEE-754 bit pattern of
/// each coordinate. An exact key match means the identical point, and the
/// objective is pure, so the cached value equals a fresh evaluation exactly.
type SampleCache = std::collections::HashMap<Box<[u64]>, f64>;

fn evaluate_axis<F>(
    objective: &mut F,
    x: &[f64],
    index: usize,
    delta: f64,
    fevals: &mut usize,
    cache: &mut SampleCache,
    reuse: bool,
) -> Result<f64>
where
    F: FnMut(&[f64]) -> Result<f64>,
{
    let mut trial = x.to_vec();
    trial[index] += delta;
    evaluate(objective, &trial, fevals, cache, reuse)
}

fn evaluate<F>(
    objective: &mut F,
    x: &[f64],
    fevals: &mut usize,
    cache: &mut SampleCache,
    reuse: bool,
) -> Result<f64>
where
    F: FnMut(&[f64]) -> Result<f64>,
{
    if reuse {
        let key: Box<[u64]> = x.iter().map(|v| v.to_bits()).collect();
        if let Some(&cached) = cache.get(&key) {
            return Ok(cached);
        }
        let value = objective(x)?;
        *fevals += 1;
        if value.is_finite() {
            cache.insert(key, value);
            Ok(value)
        } else {
            Err(MixedModelError::Optimization(
                "TrustBQ objective returned a non-finite value".to_string(),
            ))
        }
    } else {
        let value = objective(x)?;
        *fevals += 1;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(MixedModelError::Optimization(
                "TrustBQ objective returned a non-finite value".to_string(),
            ))
        }
    }
}

fn project_point(x: &[f64], lower_bounds: &[f64], upper_bounds: &[f64]) -> Vec<f64> {
    let mut projected = x.to_vec();
    project_in_place(&mut projected, lower_bounds, upper_bounds);
    projected
}

fn project_in_place(x: &mut [f64], lower_bounds: &[f64], upper_bounds: &[f64]) {
    for i in 0..x.len() {
        if lower_bounds[i].is_finite() && x[i] < lower_bounds[i] {
            x[i] = lower_bounds[i];
        }
        if upper_bounds[i].is_finite() && x[i] > upper_bounds[i] {
            x[i] = upper_bounds[i];
        }
    }
}

fn bound_and_scale_step(
    step: &mut [f64],
    x: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    radius: f64,
) {
    for i in 0..step.len() {
        let candidate = x[i] + step[i];
        if lower_bounds[i].is_finite() && candidate < lower_bounds[i] {
            step[i] = lower_bounds[i] - x[i];
        }
        if upper_bounds[i].is_finite() && candidate > upper_bounds[i] {
            step[i] = upper_bounds[i] - x[i];
        }
    }

    let step_norm = norm(step);
    if step_norm > radius && step_norm > 0.0 {
        let scale = radius / step_norm;
        for value in step {
            *value *= scale;
        }
    }
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unbounded(n: usize) -> (Vec<f64>, Vec<f64>) {
        (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n])
    }

    #[test]
    fn trust_bq_trace_classification_labels_are_stable() {
        let cases = [
            (
                TrustBqStopReason::RadiusBelowTolerance,
                TrustBqTraceClassification::SmoothConvergence,
                "smooth_convergence",
                true,
            ),
            (
                TrustBqStopReason::ObjectiveTolerance,
                TrustBqTraceClassification::SmoothConvergence,
                "smooth_convergence",
                true,
            ),
            (
                TrustBqStopReason::StepBelowTolerance,
                TrustBqTraceClassification::SmoothConvergence,
                "smooth_convergence",
                true,
            ),
            (
                TrustBqStopReason::ObjectiveStagnation,
                TrustBqTraceClassification::StatisticalStall,
                "statistical_stall",
                true,
            ),
            (
                TrustBqStopReason::CertifiedConvergence,
                TrustBqTraceClassification::CertificateAccepted,
                "certificate_accepted",
                true,
            ),
            (
                TrustBqStopReason::MaxEvaluations,
                TrustBqTraceClassification::BudgetExhaustion,
                "budget_exhaustion",
                false,
            ),
        ];

        for (reason, classification, label, acceptable) in cases {
            assert_eq!(reason.trace_classification(), classification);
            assert_eq!(classification.as_str(), label);
            assert_eq!(reason.is_acceptable_convergence(), acceptable);
        }
    }

    #[test]
    fn trust_bq_solves_shifted_quadratic() {
        let (lower, upper) = unbounded(2);
        let result = minimize(
            &[4.0, -5.0],
            &lower,
            &upper,
            TrustBqOptions {
                initial_radius: 1.0,
                final_radius: 1e-7,
                max_evaluations: 400,
                ..TrustBqOptions::default()
            },
            |x| Ok((x[0] - 1.25).powi(2) + 2.0 * (x[1] + 2.0).powi(2)),
        )
        .unwrap();

        assert_ne!(result.stop_reason, TrustBqStopReason::MaxEvaluations);
        assert!(result.stop_reason.is_acceptable_convergence());
        assert!(result.fevals > 0);
        assert!(result.last_model_sample_count > 0);
        assert!((result.x[0] - 1.25).abs() < 1e-4, "{result:?}");
        assert!((result.x[1] + 2.0).abs() < 1e-4, "{result:?}");
        assert!(result.fmin < 1e-8, "{result:?}");
    }

    #[test]
    fn trust_bq_handles_rosenbrock_like_valley() {
        let (lower, upper) = (vec![-2.0, -1.0], vec![2.0, 3.0]);
        let result = minimize(
            &[-1.2, 1.0],
            &lower,
            &upper,
            TrustBqOptions {
                initial_radius: 0.5,
                final_radius: 1e-6,
                max_evaluations: 20_000,
                ftol_abs: 1e-12,
                ftol_rel: 1e-12,
                ..TrustBqOptions::default()
            },
            |x| Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)),
        )
        .unwrap();

        assert_ne!(result.stop_reason, TrustBqStopReason::MaxEvaluations);
        assert!(result.fevals > 0);
        assert!(result.iterations > 0);
        assert!(result.final_radius <= 0.5);
        assert!(result.fmin < 1e-5, "{result:?}");
        assert!((result.x[0] - 1.0).abs() < 5e-3, "{result:?}");
        assert!((result.x[1] - 1.0).abs() < 1e-2, "{result:?}");
    }

    #[test]
    fn trust_bq_respects_bounded_boundary_optimum() {
        let result = minimize(
            &[1.5, -0.75],
            &[0.0, -1.0],
            &[4.0, 1.0],
            TrustBqOptions {
                initial_radius: 0.75,
                final_radius: 1e-7,
                max_evaluations: 500,
                ..TrustBqOptions::default()
            },
            |x| Ok((x[0] + 2.0).powi(2) + (x[1] - 0.25).powi(2)),
        )
        .unwrap();

        assert_ne!(result.stop_reason, TrustBqStopReason::MaxEvaluations);
        assert!(result.fevals > 0);
        assert!(result.x[0].abs() < 1e-8, "{result:?}");
        assert!((result.x[1] - 0.25).abs() < 1e-4, "{result:?}");
        assert!((result.fmin - 4.0).abs() < 1e-6, "{result:?}");
    }

    #[test]
    fn trust_bq_early_stops_on_objective_stagnation() {
        // Rosenbrock-like valley with numeric tolerances pinned so small that
        // neither the objective-tolerance nor the radius stop can fire inside
        // the budget. With a *statistical* stall band (loose `stall_ftol_*`)
        // and parameter-stability disabled — the configuration the
        // profiled-likelihood LMM path uses — the search must recognize that
        // further moves are sub-band noise and stop early via
        // `ObjectiveStagnation` rather than burning the whole budget.
        let result = minimize(
            &[-1.2, 1.0],
            &[-2.0, -1.0],
            &[2.0, 3.0],
            TrustBqOptions {
                initial_radius: 0.5,
                final_radius: 1e-10,
                max_evaluations: 20_000,
                ftol_abs: 1e-15,
                ftol_rel: 1e-15,
                stall_iterations: 3,
                stall_ftol_rel: 1e-3,
                stall_ftol_abs: 1e-4,
                stall_requires_stable_x: false,
                ftol_requires_local_radius: true,
                ..TrustBqOptions::default()
            },
            |x| Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)),
        )
        .unwrap();

        assert_eq!(
            result.stop_reason,
            TrustBqStopReason::ObjectiveStagnation,
            "{result:?}"
        );
        assert_eq!(
            result.trace_classification(),
            TrustBqTraceClassification::StatisticalStall
        );
        assert!(
            result.final_radius <= 0.5 / 16.0,
            "a stagnation stop must come from a localized trust region: {result:?}"
        );
        assert!(result.fevals < 20_000, "{result:?}");
        // The loose band trades a little accuracy for far fewer evaluations,
        // but the descent still reached the basin.
        assert!(result.fmin < 1.0, "{result:?}");
    }

    #[test]
    fn trust_bq_progress_callback_can_certify_before_budget() {
        let (lower, upper) = unbounded(2);
        let mut progress_checks = 0usize;
        let result = minimize_with_progress(
            &[4.0, -5.0],
            &lower,
            &upper,
            TrustBqOptions {
                initial_radius: 1.0,
                final_radius: 1e-12,
                max_evaluations: 400,
                ftol_abs: 1e-15,
                ftol_rel: 1e-15,
                stall_iterations: 200,
                ..TrustBqOptions::default()
            },
            |x| Ok((x[0] - 1.25).powi(2) + 2.0 * (x[1] + 2.0).powi(2)),
            |progress| {
                progress_checks += 1;
                Ok(progress.fevals >= 5 && progress.fmin.is_finite())
            },
        )
        .unwrap();

        assert_eq!(
            result.stop_reason,
            TrustBqStopReason::CertifiedConvergence,
            "{result:?}"
        );
        assert_eq!(
            result.trace_classification(),
            TrustBqTraceClassification::CertificateAccepted
        );
        assert!(progress_checks > 0);
        assert!(result.fevals < 400, "{result:?}");
    }

    #[test]
    fn trust_bq_rejects_zero_stall_iterations() {
        let (lower, upper) = unbounded(1);
        let err = minimize(
            &[0.0],
            &lower,
            &upper,
            TrustBqOptions {
                stall_iterations: 0,
                ..TrustBqOptions::default()
            },
            |x| Ok(x[0] * x[0]),
        );
        assert!(err.is_err());
    }

    #[test]
    fn trust_bq_sample_reuse_is_exact_and_cheaper() {
        // A boundary-clamped quadratic: the unconstrained x0 optimum is -2 but
        // x0 is clamped to its lower bound 0, so axis-0 probing is one-sided
        // and the second-difference path re-probes x0+2*delta. Across the
        // radius-shrink chain delta halves, so 2*delta on one iteration equals
        // delta on the next — the same point recurs, which exact memoization
        // can recycle.
        fn quad(x: &[f64]) -> Result<f64> {
            Ok((x[0] + 2.0).powi(2) + (x[1] - 0.25).powi(2))
        }
        let opts = |reuse| TrustBqOptions {
            initial_radius: 0.75,
            final_radius: 1e-7,
            max_evaluations: 2000,
            reuse_samples: reuse,
            ..TrustBqOptions::default()
        };
        let fresh = minimize(&[1.5, -0.75], &[0.0, -1.0], &[4.0, 1.0], opts(false), quad)
            .expect("fresh-rebuild solve");
        let reused = minimize(&[1.5, -0.75], &[0.0, -1.0], &[4.0, 1.0], opts(true), quad)
            .expect("reuse solve");

        // Exact memoization must not perturb the path: identical stop reason,
        // optimum, and objective to the bit.
        assert_eq!(
            reused.stop_reason, fresh.stop_reason,
            "{fresh:?} {reused:?}"
        );
        assert_eq!(reused.iterations, fresh.iterations, "{fresh:?} {reused:?}");
        assert!(
            (reused.fmin - fresh.fmin).abs() < 1e-12,
            "fmin differs: {fresh:?} vs {reused:?}"
        );
        assert!(
            (reused.x[0] - fresh.x[0]).abs() < 1e-12,
            "{fresh:?} {reused:?}"
        );
        assert!(
            (reused.x[1] - fresh.x[1]).abs() < 1e-12,
            "{fresh:?} {reused:?}"
        );
        // ...but it strictly lowers the objective-evaluation count.
        assert!(
            reused.fevals < fresh.fevals,
            "reuse evals {} not fewer than fresh {}",
            reused.fevals,
            fresh.fevals
        );
    }
}
