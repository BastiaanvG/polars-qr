//! The expression entry points that Polars calls into.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::dense::{column_names, DenseFrame, NullPolicy};
use crate::least_squares::{self, Solver};
use crate::result;
use crate::weights::Weights;

/// The version of the compiled plugin, so a stale build is easy to spot from Python.
#[polars_expr(output_type=String)]
fn plugin_version(_inputs: &[Series]) -> PolarsResult<Series> {
    let version = env!("CARGO_PKG_VERSION");
    Ok(Series::new("version".into(), [version]))
}

#[derive(Deserialize)]
struct LeastSquaresKwargs {
    n_targets: usize,
    weighted: bool,
    intercept: bool,
    solver: Solver,
    null_policy: NullPolicy,
}

fn least_squares_dtype(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "least_squares".into(),
        DataType::Struct(vec![
            Field::new("features".into(), result::string_list_dtype()),
            Field::new("targets".into(), result::string_list_dtype()),
            Field::new("coefficients".into(), result::matrix_dtype()),
            Field::new("intercept".into(), result::float_list_dtype()),
            Field::new("n_observations".into(), DataType::UInt32),
            Field::new("rank".into(), DataType::UInt32),
            Field::new("residual_sum_of_squares".into(), result::float_list_dtype()),
            Field::new("singular_values".into(), result::float_list_dtype()),
            Field::new("condition".into(), DataType::Float64),
            Field::new("solver".into(), DataType::String),
        ]),
    ))
}

/// Fit one or several targets against a shared matrix of features.
///
/// The inputs arrive as the targets, then the features, then the weight column when there
/// is one. Rows are read jointly, so the null policy applies to all of them together and
/// every target is fitted on the same sample.
#[polars_expr(output_type_func=least_squares_dtype)]
fn least_squares(inputs: &[Series], kwargs: LeastSquaresKwargs) -> PolarsResult<Series> {
    let n_targets = kwargs.n_targets;
    let n_weights = usize::from(kwargs.weighted);
    if n_targets == 0 {
        polars_bail!(InvalidOperation: "least_squares needs at least one target");
    }
    if inputs.len() <= n_targets + n_weights {
        polars_bail!(InvalidOperation: "least_squares needs at least one feature");
    }

    let dense = DenseFrame::from_series(inputs, kwargs.null_policy)?;
    let matrix = dense.matrix();
    let n_features = dense.n_cols() - n_targets - n_weights;
    let targets = matrix.subcols(0, n_targets);
    let features = matrix.subcols(n_targets, n_features);
    let names = column_names(inputs);

    let weights = kwargs
        .weighted
        .then(|| {
            Weights::new(
                matrix.subcols(dense.n_cols() - 1, 1),
                &names[names.len() - 1],
            )
        })
        .transpose()?;
    let options = least_squares::Options {
        intercept: kwargs.intercept,
        solver: kwargs.solver,
    };
    let fit = least_squares::fit(features, targets, weights.as_ref(), &options)?;

    result::struct_row(
        "least_squares",
        &[
            result::string_list("features", &names[n_targets..n_targets + n_features]),
            result::string_list("targets", &names[..n_targets]),
            result::matrix_rows("coefficients", fit.coefficients.transpose()),
            result::optional_float_list("intercept", fit.intercept.as_deref()),
            result::count("n_observations", fit.n_observations),
            result::count("rank", fit.rank),
            result::float_list("residual_sum_of_squares", &fit.residual_sum_of_squares),
            result::optional_float_list("singular_values", fit.singular_values.as_deref()),
            result::number("condition", fit.condition),
            result::text("solver", fit.solver.name()),
        ],
    )
}
