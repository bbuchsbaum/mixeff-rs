//! Deterministic, dependency-free work distribution for independent refits
//! (parametric bootstrap replicates, profile-likelihood parameters).
//!
//! Callers keep every source of randomness on the calling thread (responses
//! are simulated serially in the serial RNG order) and hand the worker
//! threads only pure refit jobs, so results are bit-identical for every
//! thread count. Workers never call host callbacks: the R bridge requires
//! every R API call to happen on the main thread, so templates handed to
//! workers must have their host progress callback cleared (or replaced by a
//! pure-Rust one, such as a cancellation-flag check).

use crate::error::{MixedModelError, Result};

/// The calling thread's floating-point control state, re-applied on every
/// worker thread so a refit computes bit-identically wherever it runs.
///
/// On x86/x86_64 there are two per-thread control registers:
///
/// * the x87 FPU control word (precision, rounding, exception masks). Rust
///   code itself never uses x87 on x86_64, but mingw-w64's libm (`exp`,
///   `log`, `pow`, ... as linked by the `*-windows-gnu` targets) evaluates
///   with x87 instructions, so its last bits depend on the precision-control
///   field. R on Windows runs `fninit` on its main thread (`Rwin_fpset`),
///   selecting 64-bit (extended) precision, control word `0x037F`, whereas a
///   thread created by `CreateThread` starts with the Windows x64 default
///   `0x027F` (53-bit precision). Serial refits (on R's thread) and
///   threaded ones therefore differed in the last ulps.
/// * the SSE `MXCSR` (rounding, flush-to-zero, denormals-are-zero,
///   exception masks), which governs all ordinary `f64` arithmetic.
///
/// Both are read with `fnstcw`/`stmxcsr` before spawning and loaded with
/// `fldcw`/`ldmxcsr` in each worker. Going through the CRT's
/// `_controlfp`/`_control87` does not work: on x64 they ignore the
/// precision-control field (`_MCW_PC` is unsupported there) and report an
/// abstraction of `MXCSR`, so the x87 precision never reached the workers.
/// The registers are propagated on every x86 target (harmless where the
/// OS already copies them to new threads, as Linux does); other
/// architectures carry no state, since this crate never changes it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FloatingPointEnv {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    x87_control_word: u16,
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse")
    ))]
    mxcsr: u32,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_control {
    use std::arch::asm;

    /// `MXCSR` sticky exception-status flags (IE, DE, ZE, OE, UE, PE): not
    /// control state, so they are not copied to workers.
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse")
    ))]
    pub(super) const MXCSR_STATUS_FLAGS: u32 = 0x3F;

    /// Read this thread's x87 control word (`fnstcw`).
    pub(super) fn x87_control_word() -> u16 {
        let mut word: u16 = 0;
        // SAFETY: `fnstcw` stores the 16-bit control word to the given,
        // valid and writable location; it changes no other state.
        unsafe {
            asm!(
                "fnstcw word ptr [{0}]",
                in(reg) &mut word as *mut u16,
                options(nostack, preserves_flags)
            );
        }
        word
    }

    /// Load `word` into this thread's x87 control word (`fldcw`).
    pub(super) fn set_x87_control_word(word: u16) {
        // SAFETY: `fldcw` only reads the 16-bit location and replaces this
        // thread's x87 control word; Rust code does not rely on its value
        // (SSE2 arithmetic), and the value comes from `fnstcw`.
        unsafe {
            asm!(
                "fldcw word ptr [{0}]",
                in(reg) &word as *const u16,
                options(nostack, preserves_flags, readonly)
            );
        }
    }

    /// Read this thread's `MXCSR` (`stmxcsr`).
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse")
    ))]
    pub(super) fn mxcsr() -> u32 {
        let mut csr: u32 = 0;
        // SAFETY: `stmxcsr` stores the 32-bit register to the given, valid
        // and writable location; it changes no other state.
        unsafe {
            asm!(
                "stmxcsr dword ptr [{0}]",
                in(reg) &mut csr as *mut u32,
                options(nostack, preserves_flags)
            );
        }
        csr
    }

    /// Load `csr` into this thread's `MXCSR` (`ldmxcsr`).
    #[cfg(any(
        target_arch = "x86_64",
        all(target_arch = "x86", target_feature = "sse")
    ))]
    pub(super) fn set_mxcsr(csr: u32) {
        // SAFETY: `ldmxcsr` only reads the 32-bit location; the value comes
        // from `stmxcsr` (reserved bits zero), so it cannot fault.
        unsafe {
            asm!(
                "ldmxcsr dword ptr [{0}]",
                in(reg) &csr as *const u32,
                options(nostack, preserves_flags, readonly)
            );
        }
    }
}

impl FloatingPointEnv {
    /// Capture the calling thread's control state.
    pub(crate) fn capture() -> Self {
        Self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            x87_control_word: x86_control::x87_control_word(),
            #[cfg(any(
                target_arch = "x86_64",
                all(target_arch = "x86", target_feature = "sse")
            ))]
            mxcsr: x86_control::mxcsr() & !x86_control::MXCSR_STATUS_FLAGS,
        }
    }

    /// Apply the captured state to the current (worker) thread.
    pub(crate) fn apply(self) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        x86_control::set_x87_control_word(self.x87_control_word);
        #[cfg(any(
            target_arch = "x86_64",
            all(target_arch = "x86", target_feature = "sse")
        ))]
        x86_control::set_mxcsr(
            self.mxcsr | (x86_control::mxcsr() & x86_control::MXCSR_STATUS_FLAGS),
        );
    }
}

/// Spawn a scoped worker thread that first adopts `fp_env` (captured on the
/// calling thread). Every worker spawn site goes through this helper, so no
/// worker can run a job under a different floating-point control state.
fn spawn_worker<'scope, T, F>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    fp_env: FloatingPointEnv,
    f: F,
) -> std::thread::ScopedJoinHandle<'scope, T>
where
    T: Send + 'scope,
    F: FnOnce() -> T + Send + 'scope,
{
    scope.spawn(move || {
        fp_env.apply();
        f()
    })
}

/// Run `f` with this thread's x87 control word set to `0x037F` (all
/// exceptions masked, 64-bit extended precision, round to nearest): the
/// state R on Windows establishes on its main thread with `fninit`. The
/// previous control word is restored afterwards (also on panic). Tests use
/// it to reproduce the R host situation, in which worker threads start with
/// a different x87 precision than the calling thread.
#[cfg(test)]
pub(crate) fn with_r_host_x87_precision<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        struct Restore(u16);
        impl Drop for Restore {
            fn drop(&mut self) {
                x86_control::set_x87_control_word(self.0);
            }
        }
        let _restore = Restore(x86_control::x87_control_word());
        x86_control::set_x87_control_word(R_HOST_X87_CONTROL_WORD);
        f()
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        f()
    }
}

/// x87 control word after `fninit` (R's `Rwin_fpset`).
#[cfg(test)]
const R_HOST_X87_CONTROL_WORD: u16 = 0x037F;

/// Validate a caller-supplied thread count (`0` is rejected; `1` is serial).
pub(crate) fn validate_threads(threads: usize) -> Result<()> {
    if threads == 0 {
        return Err(MixedModelError::InvalidArgument(
            "threads must be at least 1 (1 runs serially)".to_string(),
        ));
    }
    Ok(())
}

/// Map `run` over `items` with up to `threads` scoped worker threads, for
/// long jobs that must stay interruptible.
///
/// Each worker builds its own state with `init` and pulls jobs from a
/// shared queue; results are returned in item order regardless of which
/// worker produced them. While the workers run, the calling thread invokes `poll` about every
/// `POLL_INTERVAL` (this is where a host interrupt check belongs, since it
/// must run on the calling thread). When `poll` fails, `cancel` is raised,
/// workers take no further jobs (and jobs that watch `cancel` can stop
/// early), and the poll error is returned once every worker has stopped.
pub(crate) fn map_with_workers_polled<T, S, R, I, F>(
    threads: usize,
    items: Vec<T>,
    init: I,
    run: F,
    cancel: &std::sync::atomic::AtomicBool,
    poll: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<R>>
where
    T: Send,
    R: Send,
    I: Fn() -> S + Sync,
    F: Fn(&mut S, T) -> R + Sync,
{
    use std::sync::atomic::{AtomicUsize, Ordering};
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(25);

    let n = items.len();
    let workers = threads.min(n).max(1);
    let queue = std::sync::Mutex::new(items.into_iter().enumerate());
    let running = AtomicUsize::new(workers);
    let mut poll_error = None;
    let fp_env = FloatingPointEnv::capture();
    let produced: Vec<Vec<(usize, R)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                spawn_worker(scope, fp_env, || {
                    // Decrement `running` even if a job panics.
                    struct Finished<'a>(&'a AtomicUsize);
                    impl Drop for Finished<'_> {
                        fn drop(&mut self) {
                            self.0.fetch_sub(1, Ordering::AcqRel);
                        }
                    }
                    let _finished = Finished(&running);
                    let mut state = init();
                    let mut out = Vec::new();
                    while !cancel.load(Ordering::Acquire) {
                        let next = queue
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .next();
                        let Some((index, item)) = next else {
                            break;
                        };
                        out.push((index, run(&mut state, item)));
                    }
                    out
                })
            })
            .collect();
        while running.load(Ordering::Acquire) > 0 {
            std::thread::sleep(POLL_INTERVAL);
            if poll_error.is_none() {
                if let Err(error) = poll() {
                    cancel.store(true, Ordering::Release);
                    poll_error = Some(error);
                }
            }
        }
        handles
            .into_iter()
            .map(|handle| match handle.join() {
                Ok(out) => out,
                Err(payload) => std::panic::resume_unwind(payload),
            })
            .collect()
    });
    if let Some(error) = poll_error {
        return Err(error);
    }
    let mut slots: Vec<Option<R>> = (0..n).map(|_| None).collect();
    for (index, result) in produced.into_iter().flatten() {
        slots[index] = Some(result);
    }
    slots
        .into_iter()
        .map(|slot| {
            slot.ok_or_else(|| MixedModelError::Interrupted("parallel job cancelled".to_string()))
        })
        .collect()
}

/// Streaming producer/worker pipeline for long runs of independent jobs.
///
/// `produce(i)` builds job `i` on the calling thread, strictly in order
/// `0, 1, ...` (this is where serial RNG draws belong); `run` executes jobs
/// on `threads` scoped workers, each with its own `init()` state that
/// persists across its jobs; results come back to the calling thread,
/// which calls `on_result(i, result)` strictly in job order `0, 1, ...` (a
/// good place for host progress/interrupt callbacks, and for an early stop:
/// returning `Ok(false)` stops the run there). At most a bounded number of
/// produced jobs wait in flight, so memory stays O(threads).
///
/// An error from `produce` or `on_result` stops the run: workers take no
/// further jobs and the error is returned after they finish.
pub(crate) fn pipeline<T, S, R, I, F>(
    threads: usize,
    total: usize,
    produce: &mut dyn FnMut(usize) -> Result<T>,
    init: I,
    run: F,
    on_result: &mut dyn FnMut(usize, R) -> Result<bool>,
) -> Result<()>
where
    T: Send,
    R: Send,
    I: Fn() -> S + Sync,
    F: Fn(&mut S, T) -> R + Sync,
{
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;

    if threads <= 1 || total <= 1 {
        let mut state = init();
        for index in 0..total {
            let job = produce(index)?;
            if !on_result(index, run(&mut state, job))? {
                break;
            }
        }
        return Ok(());
    }

    let workers = threads.min(total);
    let in_flight = workers * 2;
    let stop = AtomicBool::new(false);
    let (job_tx, job_rx) = mpsc::sync_channel::<(usize, T)>(in_flight);
    let job_rx = std::sync::Mutex::new(job_rx);
    let (result_tx, result_rx) = mpsc::channel::<(usize, R)>();
    // A panicking job must not leave the calling thread waiting for its
    // result: the worker records the payload, raises `stop`, and the panic
    // is resumed on the calling thread once every worker has finished.
    let panic_payload: std::sync::Mutex<Option<Box<dyn std::any::Any + Send>>> =
        std::sync::Mutex::new(None);

    let fp_env = FloatingPointEnv::capture();
    let outcome = std::thread::scope(|scope| -> Result<()> {
        for _ in 0..workers {
            let result_tx = result_tx.clone();
            let (job_rx, stop, init, run, panic_payload) =
                (&job_rx, &stop, &init, &run, &panic_payload);
            spawn_worker(scope, fp_env, move || {
                let worker = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut state = init();
                    loop {
                        let next = job_rx
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .recv();
                        let Ok((index, job)) = next else {
                            break;
                        };
                        if stop.load(Ordering::Acquire) {
                            continue;
                        }
                        if result_tx.send((index, run(&mut state, job))).is_err() {
                            break;
                        }
                    }
                }));
                if let Err(payload) = worker {
                    stop.store(true, Ordering::Release);
                    *panic_payload
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(payload);
                }
            });
        }
        drop(result_tx);
        let panicked = || stop.load(Ordering::Acquire);

        // Hand results to `on_result` in job order.
        let mut pending: BTreeMap<usize, R> = BTreeMap::new();
        let mut next_result = 0usize;
        let mut outcome: Result<()> = Ok(());
        let mut deliver =
            |pending: &mut BTreeMap<usize, R>, next_result: &mut usize| -> Result<bool> {
                while let Some(result) = pending.remove(next_result) {
                    let index = *next_result;
                    *next_result += 1;
                    if !on_result(index, result)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            };

        let mut job_tx = Some(job_tx);
        let mut produced = 0usize;
        // A produced job the bounded queue had no room for yet (production
        // consumes serial RNG draws, so a job is never produced twice).
        let mut held: Option<(usize, T)> = None;
        while next_result < total {
            // Keep the queue full while there is work to produce.
            let mut progressed = false;
            if let Some(tx) = job_tx.as_ref() {
                if held.is_none()
                    && produced < total
                    && produced - next_result < in_flight + workers
                {
                    match produce(produced) {
                        Ok(job) => {
                            held = Some((produced, job));
                            produced += 1;
                        }
                        Err(error) => {
                            outcome = Err(error);
                            break;
                        }
                    }
                }
                if let Some(job) = held.take() {
                    match tx.try_send(job) {
                        Ok(()) => progressed = true,
                        Err(mpsc::TrySendError::Full(job)) => held = Some(job),
                        Err(mpsc::TrySendError::Disconnected(_)) => break,
                    }
                }
                if held.is_none() && produced == total {
                    job_tx = None;
                }
            }
            // Deliver anything already finished without blocking.
            while let Ok((index, result)) = result_rx.try_recv() {
                pending.insert(index, result);
            }
            match deliver(&mut pending, &mut next_result) {
                Ok(true) => {}
                Ok(false) => break,
                Err(error) => {
                    outcome = Err(error);
                    break;
                }
            }
            if progressed || next_result >= total {
                continue;
            }
            match result_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok((index, result)) => {
                    pending.insert(index, result);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if panicked() {
                        break;
                    }
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            match deliver(&mut pending, &mut next_result) {
                Ok(true) => {}
                Ok(false) => break,
                Err(error) => {
                    outcome = Err(error);
                    break;
                }
            }
        }
        // Stop the workers: no more jobs, and queued ones are skipped.
        stop.store(true, Ordering::Release);
        drop(job_tx);
        while result_rx.recv().is_ok() {}
        outcome
    });
    if let Some(payload) = panic_payload
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        std::panic::resume_unwind(payload);
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polled_map_keeps_item_order_for_every_thread_count() {
        let items: Vec<u64> = (0..37).collect();
        let expected: Vec<u64> = items.iter().map(|x| x * x + 1).collect();
        for threads in [1, 2, 3, 8, 64] {
            let cancel = std::sync::atomic::AtomicBool::new(false);
            let polled = map_with_workers_polled(
                threads,
                items.clone(),
                || (),
                |_, x| x * x + 1,
                &cancel,
                &mut || Ok(()),
            )
            .unwrap();
            assert_eq!(polled, expected, "threads={threads}");
        }
        assert!(validate_threads(0).is_err());
        assert!(validate_threads(1).is_ok());
    }

    /// libm-heavy work (the exp/log/lgamma-style kernels of GLMM refits) is
    /// bit-identical on worker threads and on the calling thread, through
    /// both mappers (see `worker_floating_point_matches_extended_precision_caller`
    /// for the R-host x87 state).
    #[test]
    fn worker_floating_point_matches_calling_thread() {
        fn kernel(x: f64) -> u64 {
            let a = (x * 0.37).exp() + (1.0 + x).ln() + (x * 0.11).sin();
            let b = (2.5 + x).powf(0.73) * (x / 7.0).tanh();
            let c = (x * 1e-3).ln_1p() + (x * 1e-2).exp_m1() + x.atan2(3.0) + (x + 0.5).log10();
            let d = statrs::function::gamma::ln_gamma(x + 1.0) - (-x * 0.05).exp() * x.cosh().ln();
            (a / 3.0 + b + c * 0.25 + d).to_bits()
        }
        let items: Vec<f64> = (0..2000).map(|i| i as f64 * 0.0731 + 0.013).collect();
        let serial: Vec<u64> = items.iter().map(|&x| kernel(x)).collect();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let mapped = map_with_workers_polled(
            4,
            items.clone(),
            || (),
            |_, x| kernel(x),
            &cancel,
            &mut || Ok(()),
        )
        .unwrap();
        assert_eq!(mapped, serial);
        let mut piped = Vec::new();
        pipeline(
            4,
            items.len(),
            &mut |i| Ok(items[i]),
            || (),
            |_, x| kernel(x),
            &mut |_, bits| {
                piped.push(bits);
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(piped, serial);
    }

    /// The caller's x87 control word and MXCSR reach every worker, also
    /// when they differ from the state a fresh thread starts with: here the
    /// caller runs with R's extended x87 precision (`fninit`, `0x037F`)
    /// and flush-to-zero set, while a new Windows thread starts at `0x027F`
    /// with FTZ clear. Fails on Windows without the propagation.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn worker_threads_adopt_extended_precision_caller_control_state() {
        with_r_host_x87_precision(|| {
            #[cfg(any(
                target_arch = "x86_64",
                all(target_arch = "x86", target_feature = "sse")
            ))]
            let saved_mxcsr = x86_control::mxcsr();
            #[cfg(any(
                target_arch = "x86_64",
                all(target_arch = "x86", target_feature = "sse")
            ))]
            x86_control::set_mxcsr(saved_mxcsr | 0x8000); // FTZ
            let caller = FloatingPointEnv::capture();
            assert_eq!(caller.x87_control_word, R_HOST_X87_CONTROL_WORD);
            let read = |_: &mut (), _: usize| FloatingPointEnv::capture();
            let cancel = std::sync::atomic::AtomicBool::new(false);
            let mapped =
                map_with_workers_polled(3, vec![0; 6], || (), read, &cancel, &mut || Ok(()))
                    .unwrap();
            let mut piped = Vec::new();
            pipeline(3, 6, &mut |i| Ok(i), || (), read, &mut |_, env| {
                piped.push(env);
                Ok(true)
            })
            .unwrap();
            #[cfg(any(
                target_arch = "x86_64",
                all(target_arch = "x86", target_feature = "sse")
            ))]
            x86_control::set_mxcsr(saved_mxcsr);
            for seen in mapped.iter().chain(&piped) {
                assert_eq!(seen.x87_control_word, caller.x87_control_word);
                #[cfg(any(
                    target_arch = "x86_64",
                    all(target_arch = "x86", target_feature = "sse")
                ))]
                assert_eq!(seen.mxcsr, caller.mxcsr);
            }
        });
    }

    /// The libm kernel stays bit-identical on workers when the caller runs
    /// with R's extended x87 precision. With mingw-w64's x87-based libm
    /// (windows-gnu) the results depend on the precision control, so this
    /// fails there without the propagation.
    #[test]
    fn worker_floating_point_matches_extended_precision_caller() {
        with_r_host_x87_precision(worker_floating_point_matches_calling_thread);
    }
}

#[cfg(test)]
mod cancel_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn poll_error_cancels_remaining_jobs() {
        let cancel = AtomicBool::new(false);
        let items: Vec<u64> = (0..1000).collect();
        let result = map_with_workers_polled(
            2,
            items,
            || (),
            |_, x| {
                std::thread::sleep(std::time::Duration::from_millis(2));
                x
            },
            &cancel,
            &mut || Err(MixedModelError::Interrupted("host interrupt".to_string())),
        );
        assert!(matches!(result, Err(MixedModelError::Interrupted(_))));
        assert!(cancel.load(Ordering::Acquire));
    }
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;

    fn collect(threads: usize, total: usize, stop_at: Option<usize>) -> Result<Vec<(usize, u64)>> {
        let mut seen = Vec::new();
        let mut rng_state = 17u64;
        pipeline(
            threads,
            total,
            &mut |index| {
                // A serial "RNG" draw per job, in job order.
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(index as u64);
                Ok(rng_state)
            },
            || 0u64,
            |calls, draw| {
                *calls += 1;
                draw.rotate_left(7) ^ 0x9E37
            },
            &mut |index, result| {
                seen.push((index, result));
                Ok(stop_at.is_none_or(|stop| index + 1 < stop))
            },
        )?;
        Ok(seen)
    }

    #[test]
    fn pipeline_delivers_in_order_identically_for_every_thread_count() {
        let serial = collect(1, 101, None).unwrap();
        assert_eq!(serial.len(), 101);
        assert!(serial.iter().enumerate().all(|(i, (index, _))| *index == i));
        for threads in [2, 3, 8] {
            assert_eq!(collect(threads, 101, None).unwrap(), serial);
            assert_eq!(
                collect(threads, 101, Some(40)).unwrap(),
                serial[..40].to_vec()
            );
        }
        assert!(collect(4, 0, None).unwrap().is_empty());
    }

    #[test]
    fn pipeline_propagates_producer_and_consumer_errors() {
        let failing_produce = pipeline(
            3,
            50,
            &mut |index| {
                if index == 20 {
                    Err(MixedModelError::InvalidArgument("draw failed".to_string()))
                } else {
                    Ok(index)
                }
            },
            || (),
            |_, job| job,
            &mut |_, _| Ok(true),
        );
        assert!(matches!(
            failing_produce,
            Err(MixedModelError::InvalidArgument(_))
        ));
        let failing_consume = pipeline(
            3,
            50,
            &mut |index| Ok(index),
            || (),
            |_, job| job,
            &mut |index, _| {
                if index == 10 {
                    Err(MixedModelError::Interrupted("host".to_string()))
                } else {
                    Ok(true)
                }
            },
        );
        assert!(matches!(
            failing_consume,
            Err(MixedModelError::Interrupted(_))
        ));
    }

    #[test]
    fn pipeline_resumes_a_worker_panic_instead_of_hanging() {
        let result = std::panic::catch_unwind(|| {
            pipeline(
                2,
                40,
                &mut |index| Ok(index),
                || (),
                |_, job| {
                    assert!(job != 7, "job seven fails");
                    job
                },
                &mut |_, _| Ok(true),
            )
        });
        assert!(result.is_err());
    }
}
