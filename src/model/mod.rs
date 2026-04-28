//! Mixed model types and fitting algorithms.

pub mod data;
pub mod generalized;
pub mod linear;
pub mod traits;

pub use data::*;
pub use generalized::*;
pub use linear::*;
pub use traits::*;
