//! Statistical methods: VarCorr, bootstrap, likelihood ratio tests.

pub mod block_description;
pub mod bootstrap;
pub mod coeftable;
pub mod lrt;
pub mod model_summary;
pub mod profile;
pub mod spline;
pub mod varcorr;

pub use block_description::*;
pub use bootstrap::*;
pub use coeftable::*;
pub use lrt::*;
pub use model_summary::*;
pub use profile::*;
pub use spline::NaturalCubicSpline;
pub use varcorr::*;
