use std::ffi::{c_double, c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use crate::error::{MixedModelError, Result};

type PrimaObj = Option<extern "C" fn(*const c_double, *mut c_double, *const c_void)>;

extern "C" {
    fn prima_bobyqa(
        calfun: PrimaObj,
        data: *const c_void,
        n: c_int,
        x: *mut c_double,
        f: *mut c_double,
        xl: *const c_double,
        xu: *const c_double,
        nf: *mut c_int,
        rhobeg: c_double,
        rhoend: c_double,
        ftarget: c_double,
        maxfun: c_int,
        npt: c_int,
        iprint: c_int,
    ) -> c_int;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrimaBobyqaOptions {
    pub(crate) rhobeg: f64,
    pub(crate) rhoend: f64,
    pub(crate) maxfun: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct PrimaBobyqaResult {
    pub(crate) x: Vec<f64>,
    pub(crate) fmin: f64,
    pub(crate) feval: i64,
    pub(crate) return_code: String,
}

struct ObjectiveState<'a, F>
where
    F: FnMut(&[f64]) -> f64,
{
    n: usize,
    objective: &'a mut F,
    panicked: bool,
}

extern "C" fn objective_trampoline<F>(x: *const c_double, f: *mut c_double, data: *const c_void)
where
    F: FnMut(&[f64]) -> f64,
{
    if x.is_null() || f.is_null() || data.is_null() {
        return;
    }

    let state = unsafe { &mut *(data as *mut ObjectiveState<F>) };
    let theta = unsafe { slice::from_raw_parts(x, state.n) };
    let value = catch_unwind(AssertUnwindSafe(|| (state.objective)(theta)));
    unsafe {
        *f = match value {
            Ok(value) => value,
            Err(_) => {
                state.panicked = true;
                f64::NAN
            }
        };
    }
}

pub(crate) fn minimize_bobyqa<F>(
    initial: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    options: PrimaBobyqaOptions,
    mut objective: F,
) -> Result<PrimaBobyqaResult>
where
    F: FnMut(&[f64]) -> f64,
{
    let n = initial.len();
    if n == 0 {
        return Err(MixedModelError::Optimization(
            "PRIMA BOBYQA requires at least one theta parameter".to_string(),
        ));
    }
    if lower_bounds.len() != n || upper_bounds.len() != n {
        return Err(MixedModelError::DimensionMismatch(
            "PRIMA BOBYQA bounds length does not match theta length".to_string(),
        ));
    }
    if n > c_int::MAX as usize || options.maxfun > c_int::MAX as usize {
        return Err(MixedModelError::Optimization(
            "PRIMA BOBYQA problem size exceeds C API limits".to_string(),
        ));
    }

    let mut x0 = initial.to_vec();
    let xl = lower_bounds.to_vec();
    let xu = upper_bounds.to_vec();
    let npt = n.checked_mul(2).and_then(|value| value.checked_add(1));
    let Some(npt) = npt else {
        return Err(MixedModelError::Optimization(
            "PRIMA BOBYQA problem size exceeds C API limits".to_string(),
        ));
    };
    if npt > c_int::MAX as usize {
        return Err(MixedModelError::Optimization(
            "PRIMA BOBYQA problem size exceeds C API limits".to_string(),
        ));
    }

    let mut state = ObjectiveState {
        n,
        objective: &mut objective,
        panicked: false,
    };

    let mut fmin = f64::NAN;
    let mut feval = 0;
    let status = unsafe {
        prima_bobyqa(
            Some(objective_trampoline::<F>),
            (&mut state as *mut ObjectiveState<F>).cast::<c_void>(),
            n as c_int,
            x0.as_mut_ptr(),
            &mut fmin,
            xl.as_ptr(),
            xu.as_ptr(),
            &mut feval,
            options.rhobeg,
            options.rhoend,
            f64::NEG_INFINITY,
            options.maxfun as c_int,
            npt as c_int,
            0,
        )
    };

    if state.panicked {
        return Err(MixedModelError::Optimization(
            "PRIMA objective callback panicked".to_string(),
        ));
    }
    // PRIMA status codes in [0, 100) are normal stop reasons; <0 or >=100 are
    // hard failures (NaN/Inf, invalid input, …).
    if !(0..100).contains(&status) {
        return Err(MixedModelError::Optimization(format!(
            "PRIMA BOBYQA failed: {}",
            prima_status_label(status),
        )));
    }
    if x0.len() != n || !fmin.is_finite() {
        return Err(MixedModelError::Optimization(
            "PRIMA BOBYQA did not return a usable optimum".to_string(),
        ));
    }

    Ok(PrimaBobyqaResult {
        x: x0,
        fmin,
        feval: i64::from(feval),
        return_code: prima_status_label(status).to_string(),
    })
}

fn prima_status_label(code: c_int) -> &'static str {
    match code {
        0 => "SMALL_TR_RADIUS",
        1 => "FTARGET_ACHIEVED",
        2 => "TRSUBP_FAILED",
        3 => "MAXFUN_REACHED",
        20 => "MAXTR_REACHED",
        30 => "CALLBACK_TERMINATE",
        -1 => "NAN_INF_X",
        -2 => "NAN_INF_F",
        -3 => "NAN_INF_MODEL",
        6 => "NO_SPACE_BETWEEN_BOUNDS",
        7 => "DAMAGING_ROUNDING",
        8 => "ZERO_LINEAR_CONSTRAINT",
        100 => "INVALID_INPUT",
        101 => "ASSERTION_FAILS",
        102 => "VALIDATION_FAILS",
        103 => "MEMORY_ALLOCATION_FAILS",
        110 => "NULL_OPTIONS",
        111 => "NULL_PROBLEM",
        112 => "NULL_X0",
        113 => "NULL_RESULT",
        114 => "NULL_FUNCTION",
        115 => "RESULT_INITIALIZED",
        _ => "UNKNOWN",
    }
}
