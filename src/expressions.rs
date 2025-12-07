//! The expression entry points that Polars calls into.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::covariance;
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
    l2_penalty: f64,
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
        l2_penalty: kwargs.l2_penalty,
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

#[derive(Deserialize)]
struct CovarianceKwargs {
    ddof: f64,
    normalise: bool,
    null_policy: NullPolicy,
}

/// The result schema of both second-moment operations, which differ only in what the
/// matrix field is called.
fn second_moment_dtype(name: &str) -> PolarsResult<Field> {
    Ok(Field::new(
        name.into(),
        DataType::Struct(vec![
            Field::new("features".into(), result::string_list_dtype()),
            Field::new("means".into(), result::float_list_dtype()),
            Field::new("standard_deviations".into(), result::float_list_dtype()),
            Field::new(name.into(), result::matrix_dtype()),
            Field::new("n_observations".into(), DataType::UInt32),
            Field::new("size".into(), DataType::UInt32),
        ]),
    ))
}

fn covariance_dtype(_: &[Field]) -> PolarsResult<Field> {
    second_moment_dtype("covariance")
}

fn correlation_dtype(_: &[Field]) -> PolarsResult<Field> {
    second_moment_dtype("correlation")
}

/// Estimate the second moments of the input columns and label the result.
fn second_moments(
    inputs: &[Series],
    kwargs: &CovarianceKwargs,
    name: &str,
) -> PolarsResult<Series> {
    let dense = DenseFrame::from_series(inputs, kwargs.null_policy)?;
    let options = covariance::Options {
        ddof: kwargs.ddof,
        normalise: kwargs.normalise,
    };
    let estimate = covariance::covariance(dense.matrix(), &options)?;
    let names = column_names(inputs);

    result::struct_row(
        name,
        &[
            result::string_list("features", &names),
            result::float_list("means", &estimate.means),
            result::float_list("standard_deviations", &estimate.standard_deviations),
            result::matrix_rows(name, estimate.values.as_ref()),
            result::count("n_observations", estimate.n_observations),
            result::count("size", dense.n_cols()),
        ],
    )
}

/// Estimate the covariance of the input columns.
#[polars_expr(output_type_func=covariance_dtype)]
fn covariance(inputs: &[Series], kwargs: CovarianceKwargs) -> PolarsResult<Series> {
    second_moments(inputs, &kwargs, "covariance")
}

/// Estimate the correlation of the input columns.
#[polars_expr(output_type_func=correlation_dtype)]
fn correlation(inputs: &[Series], kwargs: CovarianceKwargs) -> PolarsResult<Series> {
    second_moments(inputs, &kwargs, "correlation")
}
