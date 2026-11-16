//! Platform shims for wall-clock time and worker threads.
//!
//! `wasm32-unknown-unknown` has neither a monotonic clock nor threads in
//! `std`: `std::time::Instant::now()` panics with "time not implemented on
//! this platform" and `std::thread::spawn`/`scope` panic as well. This
//! crate conservatively treats every `wasm` target that way. The fit
//! paths only use the clock for diagnostic phase timings, never for a time
//! budget or a stopping rule, so on those targets timing is simply absent
//! (`None`) and results are unchanged. Threaded helpers consult
//! [`THREADS_AVAILABLE`] and run serially instead.
//!
//! Every use of `Instant` or `std::thread` on a non-test code path must go
//! through this module; `tests/wasm_crossed_fit.rs` (run in CI under
//! `wasm32-unknown-unknown`) guards the LMM fit path.

use std::time::Duration;

/// Whether `std::thread` can spawn workers on this target.
pub(crate) const THREADS_AVAILABLE: bool = !cfg!(target_family = "wasm");

/// A monotonic stopwatch that is inert on targets without a clock.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Stopwatch {
    #[cfg(not(target_family = "wasm"))]
    start: std::time::Instant,
}

impl Stopwatch {
    /// Start timing now (a no-op on targets without a clock).
    pub(crate) fn start() -> Self {
        Self {
            #[cfg(not(target_family = "wasm"))]
            start: std::time::Instant::now(),
        }
    }

    /// Time elapsed since [`start`](Self::start), or `None` when the target
    /// has no clock.
    pub(crate) fn elapsed(&self) -> Option<Duration> {
        #[cfg(not(target_family = "wasm"))]
        {
            Some(self.start.elapsed())
        }
        #[cfg(target_family = "wasm")]
        {
            None
        }
    }

    /// Time between `earlier` and `self`, or `None` when the target has no
    /// clock.
    pub(crate) fn duration_since(&self, earlier: Stopwatch) -> Option<Duration> {
        #[cfg(not(target_family = "wasm"))]
        {
            Some(self.start.saturating_duration_since(earlier.start))
        }
        #[cfg(target_family = "wasm")]
        {
            let _ = earlier;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_stopwatch_reports_monotonic_elapsed_time() {
        let first = Stopwatch::start();
        let second = Stopwatch::start();
        let since = second.duration_since(first).expect("native has a clock");
        let elapsed = first.elapsed().expect("native has a clock");
        assert!(elapsed >= since);
    }
}
