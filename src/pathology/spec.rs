//! [`GeneratorSpec`]: declarative description of a synthetic mixed-model
//! dataset, plus the [`generate`] entry point.
//!
//! The spec is intentionally coarse: one *primary* grouping factor, optional
//! random intercept plus zero or more random slopes, drawn from a
//! multivariate normal with `re_cov_truth` as covariance. Predictors are
//! i.i.d. standard normal with optional per-column scale factors (for the
//! `scale_mismatch` pathology). The response follows the chosen
//! `(family, link)`; for `(Normal, Identity)` we add Gaussian noise with
//! `residual_sd`, for `(Bernoulli, Logit)` we draw from a Bernoulli with the
//! inverse-logit linear predictor.
//!
//! A *secondary* grouping factor with a scalar intercept-only random effect
//! can be attached via [`CrossedSpec`] to model crossed REs (e.g.
//! `subj × item`). When present, observations are emitted from the explicit
//! cell list, which lets fixtures probe pathological patterns such as
//! structurally empty crossings or a disconnected bipartite cell graph.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use super::certificate::{Certificate, StructuralIssue};
use crate::error::{MixedModelError, Result as ModelResult};
use crate::model::{DataFrame, Family, LinkFunction};

/// Cluster-size threshold below which a Bernoulli/Logit (or Poisson)
/// fixture is treated as exercising the
/// [`PathologyAxis::AdaptiveGaussHermite`] axis.
///
/// Small clusters drive PIRLS into the regime where the Laplace
/// approximation is biased and adaptive Gauss-Hermite quadrature is
/// the conformant fix. The threshold is intentionally conservative —
/// designs with more than three observations per cluster are usually
/// fine under Laplace alone for binary outcomes.
pub const AGQ_SMALL_CLUSTER_THRESHOLD: usize = 3;

/// Pathology subsystem axes used by the GLMM single-axis hygiene
/// policy (`bd-01KQ8FVHD7WCN88RYJX1Y81NEP`).
///
/// Each fixture should exercise at most one of
/// [`PathologyAxis::Separation`], [`PathologyAxis::AdaptiveGaussHermite`],
/// or [`PathologyAxis::Overdispersion`] until the single-axis suite is
/// stable. Combining axes obscures which subsystem (PIRLS, AGQ, link,
/// dispersion) is at fault when a regression hits — see
/// `tests/fixtures/pathology_corpus/README.md` for the rationale.
///
/// [`PathologyAxis::IdentifiabilityCore`] tags the LMM strata
/// (easy / boundary / reduced_rank / refusal / weak_id) and is allowed
/// alongside any axis. [`PathologyAxis::LinkNonlinearity`] is
/// informational — it does not by itself trip the single-axis lint
/// because rare-event prevalence is often the *mechanism* for the
/// `Separation` axis rather than an independent stressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PathologyAxis {
    /// Logistic separation (Konis FE trichotomy or per-group all-zero/
    /// all-one outcomes). Tagged when
    /// [`Certificate::structural_issue`] is
    /// [`StructuralIssue::Separation`].
    Separation,
    /// Adaptive Gauss-Hermite stress: small clusters (`min_group_size
    /// ≤ AGQ_SMALL_CLUSTER_THRESHOLD`) on a binary or count outcome
    /// where the Laplace approximation alone is known to be biased.
    AdaptiveGaussHermite,
    /// Poisson overdispersion (variance exceeding mean). Reserved for
    /// future Poisson fixtures; no axis emitter currently fires this.
    Overdispersion,
    /// Link nonlinearity (extreme prevalence on logit, identity reject
    /// for Normal+Identity, cloglog for rare events). Informational —
    /// not part of the single-axis lint.
    LinkNonlinearity,
    /// LMM identifiability stratum (easy / boundary / reduced_rank /
    /// refusal / weak_id). Always allowed alongside other axes.
    IdentifiabilityCore,
}

/// GLMM-specific axes that participate in the single-axis lint.
///
/// Excludes [`PathologyAxis::LinkNonlinearity`] (informational) and
/// [`PathologyAxis::IdentifiabilityCore`] (LMM strata, orthogonal to
/// the GLMM hygiene policy).
pub const GLMM_LINT_AXES: &[PathologyAxis] = &[
    PathologyAxis::Separation,
    PathologyAxis::AdaptiveGaussHermite,
    PathologyAxis::Overdispersion,
];

/// Infer the pathology axes a `(spec, certificate)` pair exercises.
///
/// Pure inspection: never calls the engine, never re-runs the
/// pathology certificate. Returns a sorted, deduplicated set so
/// callers get a stable axis list per fixture. Empty for designs with
/// no detectable pathology stressor (the `easy` stratum's signature).
pub fn inferred_axes(spec: &GeneratorSpec, certificate: &Certificate) -> Vec<PathologyAxis> {
    let mut axes = std::collections::BTreeSet::new();

    if matches!(
        certificate.structural_issue,
        Some(StructuralIssue::Separation { .. })
    ) {
        axes.insert(PathologyAxis::Separation);
    }

    let glmm_family = !matches!(spec.family, Family::Normal);
    let min_group = spec.group_sizes.iter().copied().min().unwrap_or(usize::MAX);
    if glmm_family && min_group <= AGQ_SMALL_CLUSTER_THRESHOLD {
        axes.insert(PathologyAxis::AdaptiveGaussHermite);
    }

    // Overdispersion: reserved for Poisson fixtures (no emitter today).
    // The placeholder check stays conservative — only fires when a
    // future fixture sets `family == Poisson` with response_sd > 0,
    // which the current GeneratorSpec does not emit. Left in place so
    // the axis machinery is forward-compatible.

    if matches!(spec.family, Family::Bernoulli) && spec.binary_intercept_shift.abs() > 0.0 {
        axes.insert(PathologyAxis::LinkNonlinearity);
    }

    if !glmm_family
        && (certificate.structural_issue.is_some()
            || !certificate.boundary_directions.is_empty()
            || certificate.weak_identification
            || certificate.re_rank_truth < certificate.re_rank_requested)
    {
        axes.insert(PathologyAxis::IdentifiabilityCore);
    }

    axes.into_iter().collect()
}

/// Assert that a `(spec, certificate)` pair exercises at most one of
/// the GLMM-policy axes (separation / AGQ / overdispersion).
///
/// Returns `Ok(())` for compliant fixtures or `Err(violation_message)`
/// listing the offending axes. The `IdentifiabilityCore` and
/// `LinkNonlinearity` tags are deliberately exempt — see
/// [`PathologyAxis`] for the rationale.
pub fn lint_single_axis(spec: &GeneratorSpec, certificate: &Certificate) -> Result<(), String> {
    let axes = inferred_axes(spec, certificate);
    let glmm_axes: Vec<PathologyAxis> = axes
        .iter()
        .copied()
        .filter(|axis| GLMM_LINT_AXES.contains(axis))
        .collect();
    if glmm_axes.len() > 1 {
        Err(format!(
            "fixture '{}' violates single-axis policy: combines {:?}. \
             See tests/fixtures/pathology_corpus/README.md \
             (`bd-01KQ8FVHD7WCN88RYJX1Y81NEP`).",
            spec.label, glmm_axes
        ))
    } else {
        Ok(())
    }
}

/// Declarative description of a synthetic mixed-model dataset.
///
/// The spec captures everything an analytical certificate needs to classify
/// the design's identifiability *without* drawing data, plus the random seed
/// needed for reproducible data generation when the harness wants to actually
/// run the fit engine.
#[derive(Debug, Clone)]
pub struct GeneratorSpec {
    /// Reproducibility seed. Same seed + same spec → same data.
    pub seed: u64,
    /// Observations per group level. Length = number of group levels.
    pub group_sizes: Vec<usize>,
    /// True fixed-effects coefficients. Position 0 is the intercept; the
    /// remainder are slopes for `x1`, `x2`, ... in order.
    pub fe_truth: Vec<f64>,
    /// Per-predictor scale (multiplied into x_ij ~ N(0, scale²)). Position
    /// 0 is for x1, etc. If shorter than `fe_truth.len() - 1`, missing
    /// scales default to 1.0.
    pub fe_scales: Vec<f64>,
    /// Population-level correlation matrix for the fixed-effect predictors.
    /// Default is the identity (uncorrelated). The `collinear_fe` transform
    /// sets off-diagonal entries to drive predictor collinearity. Must be
    /// `n_fe_predictors × n_fe_predictors`; the certificate's `fe_rank_truth`
    /// is derived from this matrix's effective rank.
    pub fe_corr_matrix: nalgebra::DMatrix<f64>,
    /// Residual standard deviation (LMM only). Ignored for Bernoulli.
    pub residual_sd: f64,
    /// True random-effects covariance Σ. Must be q × q where
    /// q = (re_intercept as usize) + n_re_slopes.
    pub re_cov_truth: DMatrix<f64>,
    /// Whether the random-effects structure includes an intercept term.
    pub re_intercept: bool,
    /// Number of random-effect slope terms. Slopes track the first
    /// `n_re_slopes` predictors (x1, x2, ...).
    pub n_re_slopes: usize,
    /// Outcome family.
    pub family: Family,
    /// Link function.
    pub link: LinkFunction,
    /// Additional shift applied to the linear predictor for binary outcomes,
    /// used to push prevalence toward 0 or 1 for the `extreme_prevalence`
    /// pathology. Ignored for non-Bernoulli families.
    pub binary_intercept_shift: f64,
    /// Group factor name in the generated DataFrame. Default: "g".
    pub group_name: String,
    /// Response column name in the generated DataFrame. Default: "y".
    pub response_name: String,
    /// Human label, e.g. "easy" / "boundary_zero_slope". Used in error
    /// messages and the rationale string of [`crate::pathology::Certificate`].
    pub label: String,
    /// Optional secondary grouping factor with a scalar intercept-only
    /// random effect. When set, the design is *crossed* (primary × secondary)
    /// and observations are emitted from [`CrossedSpec::cells`].
    pub crossed: Option<CrossedSpec>,
}

/// Secondary grouping factor for crossed-RE designs.
///
/// Today the secondary RE is restricted to a scalar intercept: the design
/// is `(... | primary) + (1 | secondary)`. This is the simplest crossed
/// structure that meaningfully exercises the multi-`ReMat` fit path and
/// the bipartite cell graph that `empty_crossings`-style pathologies
/// stress. Vector-valued secondary REs are deferred until the crossed
/// pathology corpus needs them.
#[derive(Debug, Clone)]
pub struct CrossedSpec {
    /// Name of the secondary grouping factor (e.g. "item").
    pub name: String,
    /// Number of secondary levels.
    pub n_levels: usize,
    /// Variance of the secondary intercept random effect (truth).
    pub re_var: f64,
    /// Cells to emit observations from, as `(primary_idx, secondary_idx)`
    /// pairs. Each pair contributes one observation. If `None`, the full
    /// Cartesian product `n_primary × n_levels` is used.
    pub cells: Option<Vec<(usize, usize)>>,
}

impl CrossedSpec {
    /// Convenience constructor for a fully-crossed (Cartesian product) design.
    pub fn full_cross(name: impl Into<String>, n_levels: usize, re_var: f64) -> Self {
        Self {
            name: name.into(),
            n_levels,
            re_var,
            cells: None,
        }
    }

    /// Convenience constructor for an explicit cell list.
    pub fn from_cells(
        name: impl Into<String>,
        n_levels: usize,
        re_var: f64,
        cells: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            name: name.into(),
            n_levels,
            re_var,
            cells: Some(cells),
        }
    }
}

impl GeneratorSpec {
    /// Convenience constructor for an LMM spec with sane defaults.
    ///
    /// The caller still owns `re_cov_truth` and `fe_truth`; everything else
    /// defaults to a balanced LMM with no scale mismatch and Gaussian noise.
    pub fn lmm(
        label: impl Into<String>,
        seed: u64,
        group_sizes: Vec<usize>,
        fe_truth: Vec<f64>,
        re_intercept: bool,
        n_re_slopes: usize,
        re_cov_truth: DMatrix<f64>,
    ) -> Self {
        let n_predictors = fe_truth.len().saturating_sub(1);
        Self {
            seed,
            group_sizes,
            fe_scales: vec![1.0; n_predictors],
            fe_corr_matrix: DMatrix::identity(n_predictors, n_predictors),
            fe_truth,
            residual_sd: 1.0,
            re_cov_truth,
            re_intercept,
            n_re_slopes,
            family: Family::Normal,
            link: LinkFunction::Identity,
            binary_intercept_shift: 0.0,
            group_name: "g".into(),
            response_name: "y".into(),
            label: label.into(),
            crossed: None,
        }
    }

    /// Total observations across all groups.
    ///
    /// For crossed designs (`self.crossed.is_some()`) this is the number of
    /// emitted cells, not the sum of `group_sizes`.
    pub fn n_total(&self) -> usize {
        match self.crossed.as_ref() {
            Some(c) => match &c.cells {
                Some(cells) => cells.len(),
                None => self.group_sizes.len() * c.n_levels,
            },
            None => self.group_sizes.iter().sum(),
        }
    }

    /// Materialised cell list for the crossed design.
    ///
    /// Returns `None` when [`Self::crossed`] is `None`. Otherwise returns the
    /// explicit cell list if one was supplied, or the full Cartesian product
    /// expanded against `group_sizes.len()` primary levels.
    pub fn crossed_cells(&self) -> Option<Vec<(usize, usize)>> {
        let crossed = self.crossed.as_ref()?;
        if let Some(cells) = &crossed.cells {
            return Some(cells.clone());
        }
        let n_primary = self.group_sizes.len();
        let mut out = Vec::with_capacity(n_primary * crossed.n_levels);
        for i in 0..n_primary {
            for j in 0..crossed.n_levels {
                out.push((i, j));
            }
        }
        Some(out)
    }

    /// Number of fixed-effect predictors (excluding intercept).
    pub fn n_fe_predictors(&self) -> usize {
        self.fe_truth.len().saturating_sub(1)
    }

    /// Random-effects dimension q = re_intercept + n_re_slopes.
    pub fn re_dim(&self) -> usize {
        (self.re_intercept as usize) + self.n_re_slopes
    }
}

/// Output of [`generate`]: the generated DataFrame and the formula string
/// matching the spec's design.
#[derive(Debug, Clone)]
pub struct GeneratorOutput {
    pub data: DataFrame,
    pub formula: String,
}

/// Draw a synthetic dataset deterministically from a spec.
///
/// Identifiability certification ([`crate::pathology::certify`]) is
/// pure linear algebra over the spec for non-binomial families. For
/// `Family::Bernoulli`, separation detection is intrinsically
/// data-dependent (Konis 2007 LP plus a per-group response scan), so
/// certify drives [`crate::pathology::separation::detect_separation`]
/// — which calls this function with `spec.seed` — to classify the
/// separation tier. The realised draw is therefore deterministic
/// given the spec and used only by the separation tier; all other
/// strata stay seed-independent.
pub fn generate(spec: &GeneratorSpec) -> ModelResult<GeneratorOutput> {
    let q = spec.re_dim();
    // A malformed spec is invalid input, not an assertion failure: return an
    // error so callers (notably `detect_separation`, which already handles
    // `Err`, and through it the contractually-total `certify`) refuse cleanly
    // instead of panicking.
    if spec.re_cov_truth.nrows() != q || spec.re_cov_truth.ncols() != q {
        return Err(MixedModelError::InvalidArgument(format!(
            "re_cov_truth dim mismatch in spec '{}': expected {q}×{q} got {}×{}",
            spec.label,
            spec.re_cov_truth.nrows(),
            spec.re_cov_truth.ncols()
        )));
    }

    let n_predictors = spec.n_fe_predictors();
    if spec.n_re_slopes > n_predictors {
        return Err(MixedModelError::InvalidArgument(format!(
            "spec '{}' requests {} random slopes but only {} fixed-effect \
             predictors exist",
            spec.label, spec.n_re_slopes, n_predictors
        )));
    }

    let mut rng = StdRng::seed_from_u64(spec.seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let sqrt_sigma = sqrt_psd(&spec.re_cov_truth);
    let sqrt_fe_cov = build_fe_covariance_sqrt(spec, n_predictors);

    // Pre-draw primary random effects u_g for every primary level once.
    let n_primary = spec.group_sizes.len();
    let mut primary_re: Vec<DVector<f64>> = Vec::with_capacity(n_primary);
    for _ in 0..n_primary {
        let z = DVector::from_iterator(q, (0..q).map(|_| normal.sample(&mut rng)));
        let u_g: DVector<f64> = if q == 0 {
            DVector::zeros(0)
        } else {
            &sqrt_sigma * &z
        };
        primary_re.push(u_g);
    }

    // Pre-draw secondary intercept REs for every secondary level when the
    // spec is crossed. Variance is `crossed.re_var`; the truth is a scalar
    // intercept-only RE.
    let secondary_re: Vec<f64> = if let Some(crossed) = &spec.crossed {
        let sd = crossed.re_var.max(0.0).sqrt();
        (0..crossed.n_levels)
            .map(|_| sd * normal.sample(&mut rng))
            .collect()
    } else {
        Vec::new()
    };

    let n_total = spec.n_total();
    let mut response = Vec::with_capacity(n_total);
    let mut groups = Vec::with_capacity(n_total);
    let mut secondary_groups = Vec::with_capacity(n_total);
    let mut predictors: Vec<Vec<f64>> = (0..n_predictors)
        .map(|_| Vec::with_capacity(n_total))
        .collect();

    let mut emit = |g_idx: usize, h_idx: Option<usize>, rng: &mut StdRng| -> ModelResult<()> {
        let x: Vec<f64> = if n_predictors == 0 {
            Vec::new()
        } else {
            let z_x =
                DVector::from_iterator(n_predictors, (0..n_predictors).map(|_| normal.sample(rng)));
            let x_vec: DVector<f64> = &sqrt_fe_cov * z_x;
            x_vec.iter().copied().collect()
        };

        let mut eta = spec.fe_truth.first().copied().unwrap_or(0.0);
        for (j, x_j) in x.iter().enumerate() {
            eta += spec.fe_truth.get(j + 1).copied().unwrap_or(0.0) * x_j;
        }

        let u_g = &primary_re[g_idx];
        let mut re_pos = 0;
        if spec.re_intercept {
            eta += u_g[re_pos];
            re_pos += 1;
        }
        for j in 0..spec.n_re_slopes {
            eta += u_g[re_pos] * x[j];
            re_pos += 1;
        }

        if let Some(h) = h_idx {
            eta += secondary_re[h];
        }

        let y = sample_response(spec, eta, rng)?;
        response.push(y);
        groups.push(format!("g{:03}", g_idx + 1));
        if let Some(h) = h_idx {
            secondary_groups.push(format!("h{:03}", h + 1));
        }
        for (j, val) in x.iter().enumerate() {
            predictors[j].push(*val);
        }
        Ok(())
    };

    if let Some(crossed) = &spec.crossed {
        let cells = spec.crossed_cells().unwrap();
        for &(g_idx, h_idx) in &cells {
            assert!(
                g_idx < n_primary,
                "spec '{}': crossed cell primary index {} out of range (n_primary = {})",
                spec.label,
                g_idx,
                n_primary
            );
            assert!(
                h_idx < crossed.n_levels,
                "spec '{}': crossed cell secondary index {} out of range (n_levels = {})",
                spec.label,
                h_idx,
                crossed.n_levels
            );
            emit(g_idx, Some(h_idx), &mut rng)?;
        }
    } else {
        for (g_idx, &group_n) in spec.group_sizes.iter().enumerate() {
            for _ in 0..group_n {
                emit(g_idx, None, &mut rng)?;
            }
        }
    }

    let mut data = DataFrame::new();
    data.add_numeric(&spec.response_name, response).unwrap();
    for (j, col) in predictors.into_iter().enumerate() {
        data.add_numeric(&format!("x{}", j + 1), col).unwrap();
    }
    data.add_categorical(&spec.group_name, groups).unwrap();
    if let Some(crossed) = &spec.crossed {
        data.add_categorical(&crossed.name, secondary_groups)
            .unwrap();
    }

    Ok(GeneratorOutput {
        data,
        formula: build_formula(spec),
    })
}

fn sample_response(spec: &GeneratorSpec, eta: f64, rng: &mut StdRng) -> ModelResult<f64> {
    match (spec.family, spec.link) {
        (Family::Normal, LinkFunction::Identity) => {
            let noise: f64 = Normal::new(0.0, spec.residual_sd).unwrap().sample(rng);
            Ok(eta + noise)
        }
        (Family::Bernoulli, LinkFunction::Logit) => {
            let z = eta + spec.binary_intercept_shift;
            let p = 1.0 / (1.0 + (-z).exp());
            if rng.gen::<f64>() < p {
                Ok(1.0)
            } else {
                Ok(0.0)
            }
        }
        _ => Err(MixedModelError::UnsupportedFamilyLink {
            family: format!("{:?}", spec.family),
            link: format!("{:?}", spec.link),
        }),
    }
}

fn build_formula(spec: &GeneratorSpec) -> String {
    let n_pred = spec.n_fe_predictors();
    let fe_terms: Vec<String> = (0..n_pred).map(|j| format!("x{}", j + 1)).collect();
    let fe_part = if fe_terms.is_empty() {
        "1".to_string()
    } else {
        format!("1 + {}", fe_terms.join(" + "))
    };

    let slope_terms: Vec<String> = (0..spec.n_re_slopes)
        .map(|j| format!("x{}", j + 1))
        .collect();
    let re_inner = match (spec.re_intercept, slope_terms.is_empty()) {
        (true, true) => "1".to_string(),
        (true, false) => format!("1 + {}", slope_terms.join(" + ")),
        (false, false) => format!("0 + {}", slope_terms.join(" + ")),
        (false, true) => "1".to_string(),
    };

    let primary_block = format!("({} | {})", re_inner, spec.group_name);
    let secondary_block = match &spec.crossed {
        Some(c) => format!(" + (1 | {})", c.name),
        None => String::new(),
    };
    format!(
        "{} ~ {} + {}{}",
        spec.response_name, fe_part, primary_block, secondary_block
    )
}

/// Build the symmetric square root of the population fixed-effect predictor
/// covariance matrix `S = diag(scales) · corr · diag(scales)`.
///
/// `corr` is `spec.fe_corr_matrix` (defaults to identity in the standard LMM
/// constructor). The `collinear_fe` transform sets non-trivial off-diagonals.
/// If the spec's correlation matrix has the wrong dimension we fall back to
/// the identity, defensively, so a misconfigured spec produces uncorrelated
/// predictors rather than panicking inside the data generator.
fn build_fe_covariance_sqrt(spec: &GeneratorSpec, n_predictors: usize) -> DMatrix<f64> {
    if n_predictors == 0 {
        return DMatrix::zeros(0, 0);
    }
    let corr = if spec.fe_corr_matrix.nrows() == n_predictors
        && spec.fe_corr_matrix.ncols() == n_predictors
    {
        spec.fe_corr_matrix.clone()
    } else {
        DMatrix::identity(n_predictors, n_predictors)
    };
    let scale_diag = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
        n_predictors,
        (0..n_predictors).map(|j| spec.fe_scales.get(j).copied().unwrap_or(1.0)),
    ));
    let cov = &scale_diag * &corr * &scale_diag;
    sqrt_psd(&cov)
}

/// Symmetric square root of a PSD matrix via eigendecomposition.
///
/// Cholesky would suffice for strictly positive-definite Σ, but pathologies
/// in the reduced-rank stratum deliberately produce singular Σ (eigenvalue
/// near zero). Eigendecomposition handles both cases — we floor negative
/// eigenvalues at zero to absorb floating-point noise on a true PSD matrix.
fn sqrt_psd(sigma: &DMatrix<f64>) -> DMatrix<f64> {
    let q = sigma.nrows();
    if q == 0 {
        return DMatrix::zeros(0, 0);
    }
    let eig = SymmetricEigen::new(sigma.clone());
    let sqrt_evs = eig.eigenvalues.map(|v| v.max(0.0).sqrt());
    &eig.eigenvectors * DMatrix::from_diagonal(&sqrt_evs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_spec() -> GeneratorSpec {
        GeneratorSpec::lmm(
            "generator-test",
            123,
            vec![2, 2],
            vec![0.0, 0.25],
            true,
            0,
            DMatrix::from_element(1, 1, 0.5),
        )
    }

    #[test]
    fn test_pathology_generate_unsupported_family_returns_err() {
        for (family, link) in [
            (Family::Bernoulli, LinkFunction::Probit),
            (Family::Poisson, LinkFunction::Log),
            (Family::Gamma, LinkFunction::Inverse),
        ] {
            let mut spec = base_spec();
            spec.family = family;
            spec.link = link;

            let err = generate(&spec).unwrap_err();

            match err {
                MixedModelError::UnsupportedFamilyLink {
                    family: got_family,
                    link: got_link,
                } => {
                    assert_eq!(got_family, format!("{family:?}"));
                    assert_eq!(got_link, format!("{link:?}"));
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn test_pathology_generate_supported_pairs_succeed() {
        generate(&base_spec()).unwrap();

        let mut bernoulli = base_spec();
        bernoulli.family = Family::Bernoulli;
        bernoulli.link = LinkFunction::Logit;

        generate(&bernoulli).unwrap();
    }
}
