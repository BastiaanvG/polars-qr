//! The expression entry points that Polars calls into.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

/// The version of the compiled plugin, so a stale build is easy to spot from Python.
#[polars_expr(output_type=String)]
fn plugin_version(_inputs: &[Series]) -> PolarsResult<Series> {
    let version = env!("CARGO_PKG_VERSION");
    Ok(Series::new("version".into(), [version]))
}
