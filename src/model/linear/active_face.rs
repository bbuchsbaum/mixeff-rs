//! Experimental active-face refit for singular vector random-effect blocks.
//!
//! After the primary optimizer stops, an over-specified random-slope block
//! often sits on (or near) a lower-rank face of the covariance cone: some
//! eigenvalues of the fitted per-term covariance `ΛΛ'` are numerically zero
//! while the optimizer keeps burning its budget wandering along the
//! degenerate directions of the full `k(k+1)/2`-dimensional theta space.
//! This module implements the "certified lower-rank face" continuation
//! documented in `docs/optimizer_profiles.md` and
//! `docs/mmtrust_psd_lmm_prototype.md`: hold the detected active eigenbasis
//! `U` (k×r) fixed and re-optimize only the face covariance `S = C C'`
//! (`r(r+1)/2` coordinates), expanding every trial to full theta through
//! `G = U S U'` and the rank-revealing Cholesky, so evaluation still runs
//! through the ordinary profiled PLS objective.
//!
//! The path is opt-in ([`ActiveFaceRefit::Experimental`], default `Off`) and
//! best-effort: the refit is accepted only when it strictly improves the
//! objective, and the dropped directions are probed by finite differences at
//! the final point so the face claim is recorded as `certified` or
//! `uncertified` in the audit-visible return value
//! (`ACTIVE_FACE(<label>): <stop reason>`).

use nalgebra::{DMatrix, SymmetricEigen};

use super::{
    minimize_trust_bq_with_progress, trust_bq_final_radius, trust_bq_initial_radius,
    trust_bq_model_family_policy, ActiveFaceRefit, LinearMixedModel, TrustBqOptions,
};
use crate::error::Result;
use crate::types::FitLogEntry;

/// Maximum number of detect → refit rounds. Each round can only shrink a
/// term's active rank (a round that leaves every rank unchanged terminates
/// the loop), so this is a hard backstop, not a tuning knob.
const MAX_FACE_ROUNDS: usize = 3;

/// Relative step used for the forward-difference probe of dropped
/// directions, as a fraction of the largest active eigenvalue.
const FACE_PROBE_RELATIVE_STEP: f64 = 1e-4;

/// A dropped direction is unsupported when the probe objective drops by more
/// than this fraction of `1 + |f|` — the same order as the family ftol bands,
/// loose enough to ignore forward-difference noise.
const FACE_PROBE_DESCENT_TOLERANCE: f64 = 1e-7;

/// One detected lower-rank face of a vector random-effect term's fitted
/// relative covariance `ΛΛ'`.
struct DetectedFace {
    term_index: usize,
    requested_rank: usize,
    active_rank: usize,
    /// `k × r` active eigenvectors (columns, descending eigenvalue order).
    basis: DMatrix<f64>,
    /// `k × (k − r)` dropped eigenvectors.
    dropped: DMatrix<f64>,
    /// Active eigenvalues (relative covariance scale, descending).
    active_eigenvalues: Vec<f64>,
}

/// Per-term slice of the combined face-refit parameter vector.
enum TermPlan {
    /// Full-rank term: its raw theta coordinates pass through unchanged.
    Raw,
    /// Reduced-rank term: `r(r+1)/2` coordinates parameterize the lower
    /// Cholesky factor `C` of the face covariance `S = C C'`.
    Face(DetectedFace),
}

struct FaceSpace {
    plans: Vec<TermPlan>,
    /// Per-term `(row, col)` positions of that term's theta slots, in global
    /// theta order (column-major lower triangle).
    term_positions: Vec<Vec<(usize, usize)>>,
    /// Per-term starting offset into the full theta vector.
    term_theta_offsets: Vec<usize>,
    initial: Vec<f64>,
    lower: Vec<f64>,
    n_theta_full: usize,
}

impl FaceSpace {
    /// Expand a combined face-space vector to a full theta vector.
    ///
    /// A face point `C` represents the trial covariance `G = U (C C') U'`.
    /// Its non-triangular factor `W = U C` (which satisfies `W W' = G`
    /// exactly, including the singular directions) is re-triangularized by
    /// an LQ decomposition, so no covariance matrix is ever formed.
    fn expand(&self, combined: &[f64]) -> Vec<f64> {
        let mut theta = vec![0.0_f64; self.n_theta_full];
        let mut offset = 0usize;
        for (term_index, plan) in self.plans.iter().enumerate() {
            let positions = &self.term_positions[term_index];
            let theta_offset = self.term_theta_offsets[term_index];
            match plan {
                TermPlan::Raw => {
                    for (slot, _) in positions.iter().enumerate() {
                        theta[theta_offset + slot] = combined[offset + slot];
                    }
                    offset += positions.len();
                }
                TermPlan::Face(face) => {
                    let r = face.active_rank;
                    let k = face.requested_rank;
                    let dim = vech_dim(r);
                    let mut c = DMatrix::<f64>::zeros(r, r);
                    for (slot, (row, col)) in vech_lower_positions(r).into_iter().enumerate() {
                        c[(row, col)] = combined[offset + slot];
                    }
                    let w = &face.basis * c;
                    let entries = theta_entries_from_factor(&w, positions, k);
                    for (slot, value) in entries.into_iter().enumerate() {
                        theta[theta_offset + slot] = value;
                    }
                    offset += dim;
                }
            }
        }
        theta
    }
}

fn vech_dim(r: usize) -> usize {
    r * (r + 1) / 2
}

/// Column-major lower-triangle `(row, col)` positions for an `r × r` factor,
/// matching the per-term theta layout used by `parmap`.
fn vech_lower_positions(r: usize) -> Vec<(usize, usize)> {
    let mut positions = Vec::with_capacity(vech_dim(r));
    for col in 0..r {
        for row in col..r {
            positions.push((row, col));
        }
    }
    positions
}

/// Read the term's theta entries off the lower-triangular factor of
/// `G = W W'`, where `W` is any `k × m` factor of the trial covariance.
///
/// The LQ decomposition `W = L Q` (computed as the QR of `Wᵀ`) yields a
/// lower-trapezoidal `L` with `L L' = W W'` exactly — including the singular
/// directions a rank-deficient face produces, which a zero-pivot Cholesky
/// cannot reproduce. Column signs are flipped so the diagonal is
/// nonnegative, matching the theta lower bounds.
fn theta_entries_from_factor(w: &DMatrix<f64>, positions: &[(usize, usize)], k: usize) -> Vec<f64> {
    let qr = w.transpose().qr();
    let r = qr.r();
    let mut factor = DMatrix::<f64>::zeros(k, k);
    for col in 0..r.nrows().min(k) {
        for row in col..k {
            factor[(row, col)] = r[(col, row)];
        }
    }
    for col in 0..k {
        if factor[(col, col)] < 0.0 {
            for row in col..k {
                factor[(row, col)] = -factor[(row, col)];
            }
        }
    }
    positions
        .iter()
        .map(|&(row, col)| factor[(row, col)])
        .collect()
}

impl LinearMixedModel {
    /// Detect lower-rank faces of the fitted per-term relative covariances.
    ///
    /// Only fully parameterized vector terms (`n_theta == k(k+1)/2`) are
    /// eligible: patterned terms (e.g. zero-correlation `||` blocks) have
    /// theta layouts a dense face factor cannot map back into. The rank cut
    /// uses the same `effective_rank_tolerance` (on the σ²-scaled
    /// eigenvalues) as the effective-covariance summaries, so the refit
    /// trigger and the reported diagnostics cannot disagree.
    fn detect_active_faces(&self) -> Vec<DetectedFace> {
        let thresholds = &self.compiler_artifact.compiler_policy.thresholds;
        let sigma_sq = self.sigma().powi(2);
        let mut faces = Vec::new();
        for (term_index, reterm) in self.reterms.iter().enumerate() {
            let k = reterm.vsize;
            if k < 2 || reterm.n_theta() != vech_dim(k) {
                continue;
            }
            let g_rel = &reterm.lambda * reterm.lambda.transpose();
            let eig = SymmetricEigen::new(g_rel);
            let mut pairs: Vec<(f64, Vec<f64>)> = (0..k)
                .map(|idx| {
                    (
                        eig.eigenvalues[idx],
                        eig.eigenvectors.column(idx).iter().copied().collect(),
                    )
                })
                .collect();
            pairs.sort_by(|left, right| {
                right
                    .0
                    .partial_cmp(&left.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let max_eigenvalue = pairs.first().map(|(v, _)| v.max(0.0)).unwrap_or(0.0);
            let rank_tolerance = thresholds.effective_rank_tolerance(sigma_sq * max_eigenvalue);
            let active_rank = pairs
                .iter()
                .filter(|(value, _)| sigma_sq * value.max(0.0) > rank_tolerance)
                .count();
            if active_rank == k {
                continue;
            }
            let column_from = |vector: &[f64]| DMatrix::from_column_slice(k, 1, vector);
            let mut basis = DMatrix::<f64>::zeros(k, active_rank);
            let mut dropped = DMatrix::<f64>::zeros(k, k - active_rank);
            let mut active_eigenvalues = Vec::with_capacity(active_rank);
            for (index, (value, vector)) in pairs.iter().enumerate() {
                if index < active_rank {
                    basis.set_column(index, &column_from(vector).column(0));
                    active_eigenvalues.push(value.max(0.0));
                } else {
                    dropped.set_column(index - active_rank, &column_from(vector).column(0));
                }
            }
            faces.push(DetectedFace {
                term_index,
                requested_rank: k,
                active_rank,
                basis,
                dropped,
                active_eigenvalues,
            });
        }
        faces
    }

    /// Build the combined face-refit parameter space from the current theta
    /// and the detected faces. Returns `None` when the space is empty (every
    /// coordinate dropped).
    fn build_face_space(&self, faces: &[DetectedFace]) -> Option<FaceSpace> {
        let theta = self.theta();
        let n_terms = self.reterms.len();
        let mut term_positions: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_terms];
        let mut term_theta_offsets = vec![0usize; n_terms];
        let mut seen = vec![false; n_terms];
        for (index, &(term, row, col)) in self.parmap.iter().enumerate() {
            if !seen[term] {
                term_theta_offsets[term] = index;
                seen[term] = true;
            }
            term_positions[term].push((row, col));
        }

        let mut plans: Vec<TermPlan> = (0..n_terms).map(|_| TermPlan::Raw).collect();
        let mut face_by_term: Vec<Option<&DetectedFace>> = vec![None; n_terms];
        for face in faces {
            face_by_term[face.term_index] = Some(face);
        }

        let mut initial = Vec::new();
        let mut lower = Vec::new();
        for term_index in 0..n_terms {
            match face_by_term[term_index] {
                Some(face) => {
                    for (row, col) in vech_lower_positions(face.active_rank) {
                        initial.push(if row == col {
                            face.active_eigenvalues[row].max(0.0).sqrt()
                        } else {
                            0.0
                        });
                        lower.push(if row == col { 0.0 } else { f64::NEG_INFINITY });
                    }
                    plans[term_index] = TermPlan::Face(DetectedFace {
                        term_index: face.term_index,
                        requested_rank: face.requested_rank,
                        active_rank: face.active_rank,
                        basis: face.basis.clone(),
                        dropped: face.dropped.clone(),
                        active_eigenvalues: face.active_eigenvalues.clone(),
                    });
                }
                None => {
                    let offset = term_theta_offsets[term_index];
                    for (slot, &(row, col)) in term_positions[term_index].iter().enumerate() {
                        initial.push(theta[offset + slot]);
                        lower.push(if row == col { 0.0 } else { f64::NEG_INFINITY });
                    }
                }
            }
        }
        if initial.is_empty() {
            return None;
        }
        Some(FaceSpace {
            plans,
            term_positions,
            term_theta_offsets,
            initial,
            lower,
            n_theta_full: theta.len(),
        })
    }

    /// Post-fit active-face continuation. Returns `Ok(true)` when a refit
    /// was applied (objective strictly improved on a detected lower-rank
    /// face); `Ok(false)` leaves the fit untouched.
    pub(crate) fn apply_active_face_refit(&mut self) -> Result<bool> {
        if self.active_face_refit != ActiveFaceRefit::Experimental {
            return Ok(false);
        }
        if !self.optsum.is_fitted() || self.reterms.is_empty() {
            return Ok(false);
        }

        let mut rounds = 0usize;
        let mut total_fevals = 0usize;
        let mut face_fit_log: Vec<FitLogEntry> = Vec::new();
        let mut last_status: Option<String> = None;
        let mut last_ranks: Option<Vec<(usize, usize, usize)>> = None;
        let mut final_faces: Vec<DetectedFace> = Vec::new();

        while rounds < MAX_FACE_ROUNDS {
            let faces = self.detect_active_faces();
            if faces.is_empty() {
                break;
            }
            let ranks: Vec<(usize, usize, usize)> = faces
                .iter()
                .map(|face| (face.term_index, face.active_rank, face.requested_rank))
                .collect();
            if last_ranks.as_ref() == Some(&ranks) {
                break;
            }
            let Some(space) = self.build_face_space(&faces) else {
                break;
            };

            let baseline = self.optsum.fmin;
            let invalid_objective = baseline.abs().max(1.0) + 1.0e6 * (1.0 + baseline.abs());
            let mut evaluator = self.clone();
            let mut stage_fevals = 0usize;
            let mut stage_log: Vec<FitLogEntry> = Vec::new();
            let mut best_theta: Option<Vec<f64>> = None;
            let mut best_fmin = baseline;

            let policy = trust_bq_model_family_policy(
                space.initial.len(),
                None,
                &[],
                &[],
                -1,
                self.optsum.ftol_abs,
                self.optsum.ftol_rel,
            );
            // Honor the opt-in exact-sample-reuse override on the active-face
            // refit sub-solve too, so a diagnostic A/B run is faithful across
            // every native-TrustBQ path, not just the main optimization.
            let face_reuse_samples = self.trust_bq_sample_reuse.resolve(policy.reuse_samples);
            let result = {
                let mut objective_fn = |combined: &[f64]| -> Result<f64> {
                    stage_fevals += 1;
                    let theta = space.expand(combined);
                    let objective = evaluator
                        .objective_at_fast_or_generic(&theta)
                        .unwrap_or(invalid_objective);
                    let objective = if objective.is_finite() {
                        objective
                    } else {
                        invalid_objective
                    };
                    if objective + 1e-12 < best_fmin {
                        best_fmin = objective;
                        best_theta = Some(theta.clone());
                    }
                    stage_log.push(FitLogEntry { theta, objective });
                    Ok(objective)
                };
                minimize_trust_bq_with_progress(
                    &space.initial,
                    &space.lower,
                    &vec![f64::INFINITY; space.initial.len()],
                    TrustBqOptions {
                        initial_radius: trust_bq_initial_radius(&[], space.initial.len()),
                        final_radius: trust_bq_final_radius(&[], space.initial.len()),
                        max_evaluations: policy.max_evaluations,
                        ftol_abs: policy.ftol_abs,
                        ftol_rel: policy.ftol_rel,
                        max_cross_terms: policy.max_cross_terms,
                        reuse_samples: face_reuse_samples,
                        stall_iterations: policy.stall_iterations,
                        stall_ftol_rel: policy.stall_ftol_rel,
                        stall_ftol_abs: policy.stall_ftol_abs,
                        stall_requires_stable_x: policy.stall_requires_stable_x,
                        ..TrustBqOptions::default()
                    },
                    &mut objective_fn,
                    |_| Ok(false),
                )
            };
            total_fevals += stage_fevals;
            face_fit_log.append(&mut stage_log);
            let Ok(result) = result else {
                break;
            };
            let Some(best_theta) = best_theta else {
                // No trial improved on the incumbent fit: stop without
                // touching the installed optimum.
                break;
            };

            self.set_theta(&best_theta)?;
            self.update_l()?;
            self.optsum.fmin = best_fmin;
            self.optsum.final_params = best_theta;
            last_status = Some(Self::trust_bq_status_label(result.stop_reason));
            last_ranks = Some(ranks);
            final_faces = faces;
            rounds += 1;
        }

        if rounds == 0 {
            return Ok(false);
        }

        // Certify the face: forward-difference probe of every dropped
        // direction at the final point. A material descent along a dropped
        // direction means the lower-rank face does not support the optimum,
        // and the label must say so.
        let (certified, mut probe_log) = self.probe_dropped_directions(&final_faces)?;
        total_fevals += probe_log.len();
        face_fit_log.append(&mut probe_log);

        let mut rank_label = final_faces
            .iter()
            .map(|face| format!("rank{}of{}", face.active_rank, face.requested_rank))
            .collect::<Vec<_>>()
            .join("+");
        // Preserve the audit trail of a KKT-guided restart that ran before
        // the face refit replaced its status.
        if self
            .optsum
            .return_value
            .starts_with("KKT_BOUNDARY_RESTART(")
        {
            rank_label.push_str(":after_kkt_restart");
        }
        let support_label = if certified {
            "certified"
        } else {
            "uncertified"
        };
        let status = last_status.unwrap_or_else(|| "FTOL_REACHED".to_string());
        self.optsum.feval += total_fevals as i64;
        self.optsum.fit_log.append(&mut face_fit_log);
        self.optsum.return_value =
            format!("ACTIVE_FACE({rank_label}:{total_fevals} evals:{support_label}): {status}");
        Ok(true)
    }

    /// Probe the dropped eigendirections of every faced term at the current
    /// (final) theta. Returns whether the face is supported plus one fit-log
    /// entry per objective evaluation spent, so the caller can keep
    /// `feval == fit_log.len()` accounting exact.
    fn probe_dropped_directions(&self, faces: &[DetectedFace]) -> Result<(bool, Vec<FitLogEntry>)> {
        let theta = self.theta();
        let f0 = self.optsum.fmin;
        let descent_tolerance = FACE_PROBE_DESCENT_TOLERANCE * (1.0 + f0.abs());

        let mut term_positions: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.reterms.len()];
        let mut term_theta_offsets = vec![0usize; self.reterms.len()];
        let mut seen = vec![false; self.reterms.len()];
        for (index, &(term, row, col)) in self.parmap.iter().enumerate() {
            if !seen[term] {
                term_theta_offsets[term] = index;
                seen[term] = true;
            }
            term_positions[term].push((row, col));
        }

        let mut evaluator = self.clone();
        let mut probe_log: Vec<FitLogEntry> = Vec::new();
        let mut certified = true;
        for face in faces {
            let reterm = &self.reterms[face.term_index];
            let scale = face
                .active_eigenvalues
                .first()
                .copied()
                .unwrap_or(0.0)
                .max(1e-8);
            let step = FACE_PROBE_RELATIVE_STEP * scale;
            for dropped_index in 0..face.dropped.ncols() {
                // Probe covariance `Λ Λ' + step d d'` via its augmented
                // factor `[Λ | sqrt(step) d]`, so the perturbed matrix is
                // factored exactly rather than formed.
                let k = face.requested_rank;
                let mut w = DMatrix::<f64>::zeros(k, k + 1);
                w.view_mut((0, 0), (k, k)).copy_from(&reterm.lambda);
                let direction = face.dropped.column(dropped_index);
                for row in 0..k {
                    w[(row, k)] = step.sqrt() * direction[row];
                }
                let entries = theta_entries_from_factor(&w, &term_positions[face.term_index], k);
                let mut probe_theta = theta.clone();
                let offset = term_theta_offsets[face.term_index];
                for (slot, value) in entries.into_iter().enumerate() {
                    probe_theta[offset + slot] = value;
                }
                let Ok(f_probe) = evaluator.objective_at_fast_or_generic(&probe_theta) else {
                    continue;
                };
                probe_log.push(FitLogEntry {
                    theta: probe_theta,
                    objective: f_probe,
                });
                if f_probe.is_finite() && f_probe < f0 - descent_tolerance {
                    certified = false;
                }
            }
        }
        Ok((certified, probe_log))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A face point must expand to a theta whose implied covariance
    /// reproduces `U S U'` exactly (up to factorization roundoff), including
    /// the singular directions.
    #[test]
    fn face_expansion_reproduces_rank_deficient_covariance() {
        let k = 4;
        let r = 2;
        // Orthonormal basis spanning the active face.
        let basis = DMatrix::from_column_slice(k, r, &[0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5]);
        let c_entries = [1.3, 0.4, 0.7]; // vech of a 2x2 lower factor
        let mut c = DMatrix::<f64>::zeros(r, r);
        for (slot, (row, col)) in vech_lower_positions(r).into_iter().enumerate() {
            c[(row, col)] = c_entries[slot];
        }
        let s = &c * c.transpose();
        let g = &basis * &s * basis.transpose();
        let w = &basis * &c;

        let positions = vech_lower_positions(k);
        let entries = theta_entries_from_factor(&w, &positions, k);
        let mut factor = DMatrix::<f64>::zeros(k, k);
        for (slot, (row, col)) in positions.iter().enumerate() {
            factor[(*row, *col)] = entries[slot];
        }
        // The expanded factor is lower-triangular with a nonnegative
        // diagonal, matching the theta layout and bounds.
        for col in 0..k {
            assert!(factor[(col, col)] >= 0.0);
        }
        let reconstructed = &factor * factor.transpose();
        for row in 0..k {
            for col in 0..k {
                assert!(
                    (reconstructed[(row, col)] - g[(row, col)]).abs() < 1e-12,
                    "G mismatch at ({row}, {col}): {} vs {}",
                    reconstructed[(row, col)],
                    g[(row, col)]
                );
            }
        }
        // The dropped directions stay exactly null.
        let eig = SymmetricEigen::new(reconstructed);
        let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!(eigenvalues[r..].iter().all(|value| value.abs() < 1e-12));
    }
}
