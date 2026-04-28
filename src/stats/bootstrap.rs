//! Parametric bootstrap helpers.
//!
//! The fitted-model side of the parametric bootstrap lives in
//! [`crate::model::linear::parametricbootstrap`]; this module re-exports the
//! result types and exposes the `shortest_cov_int` utility used to summarize
//! replicate distributions.

use std::io::{Read, Write};

use crate::error::{MixedModelError, Result};
use crate::model::linear::LinearMixedModel;
pub use crate::model::linear::{
    parametricbootstrap, BootstrapInterval, BootstrapIntervalMethod, BootstrapQuantile,
    BootstrapReplicate, MixedModelBootstrap,
};

/// Save bootstrap replicates as JSON.
///
/// Mirrors the Julia `savereplicates(io, pb)` surface while using a portable
/// JSON representation in Rust.
pub fn savereplicates<W: Write>(
    writer: W,
    bootstrap: &MixedModelBootstrap,
) -> std::result::Result<(), serde_json::Error> {
    bootstrap.save_replicates(writer)
}

/// Rust-style alias for [`savereplicates`].
pub fn save_replicates<W: Write>(
    writer: W,
    bootstrap: &MixedModelBootstrap,
) -> std::result::Result<(), serde_json::Error> {
    savereplicates(writer, bootstrap)
}

/// Restore bootstrap replicates from JSON and validate dimensions against `model`.
///
/// Mirrors Julia's `restorereplicates(io, model)` shape: the model is used as a
/// template guard so stale or mismatched replicate files are rejected up front.
pub fn restorereplicates<R: Read>(
    reader: R,
    model: &LinearMixedModel,
) -> Result<MixedModelBootstrap> {
    let bootstrap = MixedModelBootstrap::restore_replicates(reader).map_err(|err| {
        MixedModelError::InvalidArgument(format!("could not restore bootstrap replicates: {err}"))
    })?;
    bootstrap.validate_for_model(model)?;
    Ok(bootstrap)
}

/// Rust-style alias for [`restorereplicates`].
pub fn restore_replicates<R: Read>(
    reader: R,
    model: &LinearMixedModel,
) -> Result<MixedModelBootstrap> {
    restorereplicates(reader, model)
}

/// Shortest coverage interval containing `level` proportion of values.
///
/// Sorts `v` ascending in place, then scans every contiguous window of size
/// `ceil(n * level)` and returns the narrowest one. Mirrors the
/// `shortestcovint` summary helper used by the Julia bootstrap surface.
pub fn shortest_cov_int(v: &mut [f64], level: f64) -> (f64, f64) {
    assert!((0.0..1.0).contains(&level), "level must be in (0, 1)");
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let ilen = ((n as f64) * level).ceil() as usize;
    if ilen >= n {
        return (v[0], v[n - 1]);
    }
    let mut min_len = f64::INFINITY;
    let mut best_i = 0;
    for i in 0..=(n - ilen) {
        let len = v[i + ilen - 1] - v[i];
        if len < min_len {
            min_len = len;
            best_i = i;
        }
    }
    (v[best_i], v[best_i + ilen - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortest_cov_int_narrow_window() {
        let mut v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (lo, hi) = shortest_cov_int(&mut v, 0.95);
        // 95-element window over uniformly-spaced integers spans at most 94.
        assert!(hi - lo <= 95.0);
        assert!(lo >= 0.0 && hi <= 99.0);
    }

    #[test]
    fn test_shortest_cov_int_full_coverage() {
        let mut v = vec![1.0, 5.0, 7.0];
        let (lo, hi) = shortest_cov_int(&mut v, 0.99);
        // ceil(3 * 0.99) == 3, so the only window is the full vector.
        assert_eq!((lo, hi), (1.0, 7.0));
    }

    #[test]
    fn test_shortest_cov_int_picks_tightest_cluster() {
        // Tight cluster at [10, 11, 12] vs. spread elsewhere: with level=0.6
        // (ceil(5 * 0.6) = 3) the tight cluster wins.
        let mut v = vec![0.0, 10.0, 11.0, 12.0, 100.0];
        let (lo, hi) = shortest_cov_int(&mut v, 0.6);
        assert_eq!((lo, hi), (10.0, 12.0));
    }
}
