//! Mixed model types and fitting algorithms.

pub mod traits;
pub mod linear;
pub mod generalized;
pub mod data;

pub use traits::*;
pub use linear::*;
pub use generalized::*;
pub use data::*;
