//! Dense numerical operations for Polars, backed by faer.
//!
//! The crate is compiled as a Polars expression plugin. Every public operation is reached
//! from Python as an expression, which keeps the numerics inside the query plan.

mod covariance;
mod dense;
mod expressions;
mod least_squares;
mod result;
mod weights;
