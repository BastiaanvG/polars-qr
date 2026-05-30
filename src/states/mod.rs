//! Mergeable state for the operations that have one.
//!
//! Some operations can be computed from a summary of the data that is much smaller than the
//! data itself, and that summary can be merged with another one. That is what makes them
//! work over partitions: each partition summarises its own rows, the summaries are merged in
//! any order, and the result is finalised once.

pub mod codec;
pub mod covariance;
pub mod least_squares;
