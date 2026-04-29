use std::ffi::{c_char, c_double, c_int, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use crate::error::{MixedModelError, Result};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimaAlgorithm {
    Bobyqa = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimaMessage {
    None = 0,
}

type PrimaObj = Option<extern "C" fn(*const c_double, *mut c_double, *const c_void)>;
type PrimaObjCon =
    Option<extern "C" fn(*const c_double, *mut c_double, *mut c_double, *const c_void)>;
type PrimaCallback = Option<
    extern "C" fn(
        c_int,
        *const c_double,
        c_double,
        c_int,
        c_int,
        c_double,
        c_int,
        *const c_double,
        *mut bool,
    ),
>;

#[repr(C)]
struct PrimaProblem {
    n: c_int,
    calfun: PrimaObj,
    calcfc: PrimaObjCon,
    x0: *mut c_double,
    xl: *mut c_double,
    xu: *mut c_double,
    m_ineq: c_int,
    aineq: *mut c_double,
    bineq: *mut c_double,
    m_eq: c_int,
    aeq: *mut c_double,
    beq: *mut c_double,
    m_nlcon: c_int,
    f0: c_double,
    nlconstr0: *mut c_double,
}

#[repr(C)]
struct PrimaOptions {
    rhobeg: c_double,
    rhoend: c_double,
    maxfun: c_int,
    iprint: PrimaMessage,
    ftarget: c_double,
    npt: c_int,
    ctol: c_double,
    data: *mut c_void,
    callback: PrimaCallback,
}

#[repr(C)]
struct PrimaResult {
    x: *mut c_double,
    f: c_double,
    cstrv: c_double,
    nlconstr: *mut c_double,
    nf: c_int,
    status: c_int,
    success: bool,
    message: *const c_char,
}

extern "C" {
    fn prima_init_problem(problem: *mut PrimaProblem, n: c_int) -> c_int;
    fn prima_init_options(options: *mut PrimaOptions) -> c_int;
    fn prima_minimize(
        algorithm: PrimaAlgorithm,
        problem: PrimaProblem,
        options: PrimaOptions,
        result: *mut PrimaResult,
    ) -> c_int;
    fn prima_free_result(result: *mut PrimaResult) -> c_int;
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
    let mut xl = lower_bounds.to_vec();
    let mut xu = upper_bounds.to_vec();

    let mut problem = unsafe { std::mem::MaybeUninit::<PrimaProblem>::zeroed().assume_init() };
    let init_problem_rc = unsafe { prima_init_problem(&mut problem, n as c_int) };
    if init_problem_rc != 0 {
        return Err(MixedModelError::Optimization(format!(
            "PRIMA init problem failed: {}",
            prima_status_label(init_problem_rc)
        )));
    }
    problem.calfun = Some(objective_trampoline::<F>);
    problem.x0 = x0.as_mut_ptr();
    problem.xl = xl.as_mut_ptr();
    problem.xu = xu.as_mut_ptr();

    let mut state = ObjectiveState {
        n,
        objective: &mut objective,
        panicked: false,
    };

    let mut prima_options =
        unsafe { std::mem::MaybeUninit::<PrimaOptions>::zeroed().assume_init() };
    let init_options_rc = unsafe { prima_init_options(&mut prima_options) };
    if init_options_rc != 0 {
        return Err(MixedModelError::Optimization(format!(
            "PRIMA init options failed: {}",
            prima_status_label(init_options_rc)
        )));
    }
    prima_options.rhobeg = options.rhobeg;
    prima_options.rhoend = options.rhoend;
    prima_options.maxfun = options.maxfun as c_int;
    prima_options.iprint = PrimaMessage::None;
    prima_options.data = (&mut state as *mut ObjectiveState<F>).cast::<c_void>();

    let mut result = unsafe { std::mem::MaybeUninit::<PrimaResult>::zeroed().assume_init() };
    let minimize_rc =
        unsafe { prima_minimize(PrimaAlgorithm::Bobyqa, problem, prima_options, &mut result) };

    let x = if !result.x.is_null() {
        unsafe { slice::from_raw_parts(result.x, n).to_vec() }
    } else {
        Vec::new()
    };
    let fmin = result.f;
    let feval = i64::from(result.nf);
    let result_status = result.status;
    let success = result.success;
    let message = if result.message.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(result.message) }
                .to_string_lossy()
                .into_owned(),
        )
    };

    unsafe {
        let _ = prima_free_result(&mut result);
    }

    if state.panicked {
        return Err(MixedModelError::Optimization(
            "PRIMA objective callback panicked".to_string(),
        ));
    }
    if minimize_rc >= 100 || minimize_rc < 0 {
        let detail = message
            .as_deref()
            .map(|msg| format!(" ({msg})"))
            .unwrap_or_default();
        return Err(MixedModelError::Optimization(format!(
            "PRIMA BOBYQA failed: {}{}",
            prima_status_label(minimize_rc),
            detail
        )));
    }
    if x.len() != n || !fmin.is_finite() {
        return Err(MixedModelError::Optimization(
            "PRIMA BOBYQA did not return a usable optimum".to_string(),
        ));
    }
    if !success && matches!(result_status, -3..=-1 | 6..=8 | 100..=i32::MAX) {
        let detail = message
            .as_deref()
            .map(|msg| format!(" ({msg})"))
            .unwrap_or_default();
        return Err(MixedModelError::Optimization(format!(
            "PRIMA BOBYQA ended abnormally: {}{}",
            prima_status_label(result_status),
            detail
        )));
    }

    Ok(PrimaBobyqaResult {
        x,
        fmin,
        feval,
        return_code: prima_status_label(result_status).to_string(),
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
