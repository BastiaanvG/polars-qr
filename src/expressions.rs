//! The expression entry points that Polars calls into.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::dense::{column_names, DenseFrame, NullPolicy};
use crate::least_squares::solve_qr;
use crate::result;

/// The version of the compiled plugin, so a stale build is easy to spot from Python.
#[polars_expr(output_type=String)]
fn plugin_version(_inputs: &[Series]) -> PolarsResult<Series> {
    let version = env!("CARGO_PKG_VERSION");
    Ok(Series::new("version".into(), [version]))
}

#[derive(Deserialize)]
struct LeastSquaresKwargs {
    null_policy: NullPolicy,
}

fn least_squares_dtype(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "least_squares".into(),
        DataType::Struct(vec![
            Field::new("features".into(), result::string_list_dtype()),
            Field::new("targets".into(), result::string_list_dtype()),
            Field::new("coefficients".into(), result::matrix_dtype()),
            Field::new("n_observations".into(), DataType::UInt32),
            Field::new(
                "residual_sum_of_squares".into(),
                result::float_list_dtype(),
            ),
        ]),
    ))
}

/// Fit one target against a matrix of features.
///
/// The first input is the target and the rest are the features. Rows are read jointly, so
/// the null policy applies to the target and the features together.
#[polars_expr(output_type_func=least_squares_dtype)]
fn least_squares(inputs: &[Series], kwargs: LeastSquaresKwargs) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        polars_bail!(InvalidOperation: "least_squares needs a target and at least one feature");
    }

    let dense = DenseFrame::from_series(inputs, kwargs.null_policy)?;
    let matrix = dense.matrix();
    let targets = matrix.subcols(0, 1);
    let features = matrix.subcols(1, dense.n_cols() - 1);

    let fit = solve_qr(features, targets)?;
    let names = column_names(inputs);

    result::struct_row(
        "least_squares",
        &[
            result::string_list("features", &names[1..]),
            result::string_list("targets", &names[..1]),
            result::matrix_rows("coefficients", fit.coefficients.transpose()),
            result::count("n_observations", fit.n_observations),
            result::float_list("residual_sum_of_squares", &fit.residual_sum_of_squares),
        ],
    )
}
