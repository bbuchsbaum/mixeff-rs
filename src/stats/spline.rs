//! Natural cubic spline interpolation.
//!
//! Used by the profile-likelihood machinery to build forward (ζ as a
//! function of a parameter) and reverse (parameter as a function of ζ)
//! maps from a small table of evaluations.
//!
//! The "natural" boundary conditions force the second derivative to zero
//! at both endpoints, so extrapolation outside the knot range degrades to
//! a linear tail. This matches the `BSplineOrder(4) + Natural()` choice
//! used by MixedModels.jl.

use crate::error::{MixedModelError, Result};

/// A natural cubic spline through a set of (x, y) knots.
///
/// The spline is `C²`-continuous between knots with zero second derivative
/// at the first and last knots.
#[derive(Debug, Clone)]
pub struct NaturalCubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Second derivatives at each knot, length `x.len()`.
    m: Vec<f64>,
}

impl NaturalCubicSpline {
    /// Fit a natural cubic spline through the given knots.
    ///
    /// Requires `x` strictly increasing with at least two points.
    pub fn fit(x: &[f64], y: &[f64]) -> Result<Self> {
        if x.len() != y.len() {
            return Err(MixedModelError::InvalidArgument(format!(
                "NaturalCubicSpline: x and y must have same length ({} vs {})",
                x.len(),
                y.len()
            )));
        }
        let n = x.len();
        if n < 2 {
            return Err(MixedModelError::InvalidArgument(
                "NaturalCubicSpline: need at least 2 knots".into(),
            ));
        }
        for i in 1..n {
            if !(x[i] > x[i - 1]) {
                return Err(MixedModelError::InvalidArgument(format!(
                    "NaturalCubicSpline: x must be strictly increasing (x[{}]={} x[{}]={})",
                    i - 1,
                    x[i - 1],
                    i,
                    x[i]
                )));
            }
        }

        let mut m = vec![0.0_f64; n];
        if n == 2 {
            // Linear segment; both second derivatives stay zero.
            return Ok(NaturalCubicSpline {
                x: x.to_vec(),
                y: y.to_vec(),
                m,
            });
        }

        // Solve the tridiagonal system for interior second derivatives
        // (m_0 = m_{n-1} = 0 by the natural boundary condition).
        //
        //   h_i (m_i) + 2 (h_i + h_{i+1}) m_{i+1} + h_{i+1} m_{i+2}
        //     = 6 ((y_{i+2}-y_{i+1})/h_{i+1} - (y_{i+1}-y_i)/h_i)
        //
        // Thomas algorithm on interior unknowns m[1..n-1].
        let interior = n - 2;
        let mut a = vec![0.0_f64; interior]; // sub-diagonal
        let mut b = vec![0.0_f64; interior]; // diagonal
        let mut c = vec![0.0_f64; interior]; // super-diagonal
        let mut d = vec![0.0_f64; interior]; // rhs

        let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
        for i in 0..interior {
            let hi = h[i];
            let hi1 = h[i + 1];
            b[i] = 2.0 * (hi + hi1);
            if i > 0 {
                a[i] = hi;
            }
            if i < interior - 1 {
                c[i] = hi1;
            }
            d[i] = 6.0 * ((y[i + 2] - y[i + 1]) / hi1 - (y[i + 1] - y[i]) / hi);
        }

        // Forward sweep.
        for i in 1..interior {
            let w = a[i] / b[i - 1];
            b[i] -= w * c[i - 1];
            d[i] -= w * d[i - 1];
        }
        // Back substitution.
        let mut sol = vec![0.0_f64; interior];
        sol[interior - 1] = d[interior - 1] / b[interior - 1];
        for i in (0..interior - 1).rev() {
            sol[i] = (d[i] - c[i] * sol[i + 1]) / b[i];
        }
        m[1..interior + 1].copy_from_slice(&sol);

        Ok(NaturalCubicSpline {
            x: x.to_vec(),
            y: y.to_vec(),
            m,
        })
    }

    /// Evaluate the spline at `xi`.
    ///
    /// Outside the knot range the spline extrapolates linearly using the
    /// slope at the nearest endpoint (consistent with natural boundary
    /// conditions).
    pub fn eval(&self, xi: f64) -> f64 {
        let n = self.x.len();
        if n == 1 {
            return self.y[0];
        }
        if xi <= self.x[0] {
            // Linear extrapolation off the left end using the slope at x[0].
            let h = self.x[1] - self.x[0];
            let slope = (self.y[1] - self.y[0]) / h - h * (2.0 * self.m[0] + self.m[1]) / 6.0;
            return self.y[0] + slope * (xi - self.x[0]);
        }
        if xi >= self.x[n - 1] {
            let h = self.x[n - 1] - self.x[n - 2];
            let slope = (self.y[n - 1] - self.y[n - 2]) / h
                + h * (2.0 * self.m[n - 1] + self.m[n - 2]) / 6.0;
            return self.y[n - 1] + slope * (xi - self.x[n - 1]);
        }

        // Binary search for the interval containing xi.
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.x[mid] > xi {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let h = self.x[hi] - self.x[lo];
        let a = (self.x[hi] - xi) / h;
        let b = (xi - self.x[lo]) / h;
        a * self.y[lo]
            + b * self.y[hi]
            + ((a.powi(3) - a) * self.m[lo] + (b.powi(3) - b) * self.m[hi]) * (h * h) / 6.0
    }

    /// Knot x-coordinates (for diagnostics or inverting via a second spline).
    pub fn knots_x(&self) -> &[f64] {
        &self.x
    }

    /// Knot y-coordinates.
    pub fn knots_y(&self) -> &[f64] {
        &self.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn interpolates_at_knots() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 1.0, 4.0, 9.0, 16.0];
        let s = NaturalCubicSpline::fit(&x, &y).unwrap();
        for (xi, yi) in x.iter().zip(y.iter()) {
            assert!(approx_eq(s.eval(*xi), *yi, 1e-12));
        }
    }

    #[test]
    fn linear_data_stays_linear() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0]; // y = 2x + 1
        let s = NaturalCubicSpline::fit(&x, &y).unwrap();
        for xi in [0.5, 1.5, 2.5] {
            assert!(approx_eq(s.eval(xi), 2.0 * xi + 1.0, 1e-12));
        }
        // Linear extrapolation on both sides.
        assert!(approx_eq(s.eval(-1.0), -1.0, 1e-12));
        assert!(approx_eq(s.eval(5.0), 11.0, 1e-12));
    }

    #[test]
    fn cubic_recovery_moderate() {
        // A smooth function — natural splines are exact on cubics only when
        // the natural boundary conditions happen to match. We just check
        // reasonable error.
        let x: Vec<f64> = (0..11).map(|i| i as f64 * 0.2).collect();
        let f = |t: f64| (t * std::f64::consts::PI / 2.0).sin();
        let y: Vec<f64> = x.iter().map(|&xi| f(xi)).collect();
        let s = NaturalCubicSpline::fit(&x, &y).unwrap();
        for q in 0..20 {
            let xi = q as f64 * 0.1;
            if xi >= x[0] && xi <= *x.last().unwrap() {
                let err = (s.eval(xi) - f(xi)).abs();
                assert!(err < 5e-4, "sin spline err {err} at {xi}");
            }
        }
    }

    #[test]
    fn rejects_non_increasing_x() {
        let x = [0.0, 1.0, 1.0, 2.0];
        let y = [0.0, 1.0, 2.0, 3.0];
        assert!(NaturalCubicSpline::fit(&x, &y).is_err());
    }

    #[test]
    fn two_point_is_linear() {
        let s = NaturalCubicSpline::fit(&[0.0, 2.0], &[1.0, 5.0]).unwrap();
        assert!(approx_eq(s.eval(1.0), 3.0, 1e-12));
        assert!(approx_eq(s.eval(0.0), 1.0, 1e-12));
        assert!(approx_eq(s.eval(2.0), 5.0, 1e-12));
    }
}
