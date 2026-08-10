//! Autoregressive models, and the structure that makes them cheap.
//!
//! An autoregression is a least squares problem, and this crate already solves those. What
//! it is not is a *general* least squares problem: the matrix of a series against its own
//! lags is Toeplitz, every diagonal of it constant, and that structure is worth using. The
//! Levinson–Durbin recursion solves such a system in `O(p²)` rather than `O(p³)`, reads the
//! series once into `p + 1` autocovariances rather than materialising `p` lag columns, and
//! passes through every lower order on its way, so the partial autocorrelations and the
//! per-order variances that order selection needs are already in hand when it finishes.
//!
//! A lag here is a row and not a duration. The model assumes one step of the sequence is
//! like any other, which is why none of this lives in [`crate::timeseries`], where the
//! distance between two observations is a number the caller supplies.

pub mod burg;
pub mod fit;
pub mod input;
pub mod levinson;
pub mod moments;
pub mod options;
