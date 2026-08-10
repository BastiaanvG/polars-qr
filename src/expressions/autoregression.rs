//! The autoregression expression entry points that Polars calls into.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::autoregression::fit::{self, Autoregression};
use crate::autoregression::input::Observations;
use crate::autoregression::levinson;
use crate::autoregression::moments;
use crate::autoregression::options::{Criterion, Method, NullPolicy, Options, Output};
use crate::expressions::{row_order, with_names};
use crate::result;

#[derive(Deserialize)]
struct SequenceKwargs {
    names: Vec<String>,
    max_lag: usize,
    demean: bool,
    unbiased: bool,
    null_policy: NullPolicy,
}

/// The result schema of both autocovariance operations, which differ only in what the
/// sequence field is called.
fn sequence_dtype(name: &str) -> PolarsResult<Field> {
    Ok(Field::new(
        name.into(),
        DataType::Struct(vec![
            Field::new("series".into(), DataType::String),
            Field::new("lags".into(), result::index_list_dtype()),
            Field::new(name.into(), result::float_list_dtype()),
            Field::new("mean".into(), DataType::Float64),
            Field::new("n_observations".into(), DataType::UInt32),
        ]),
    ))
}

fn autocovariance_dtype(_: &[Field]) -> PolarsResult<Field> {
    sequence_dtype("autocovariance")
}

fn autocorrelation_dtype(_: &[Field]) -> PolarsResult<Field> {
    sequence_dtype("autocorrelation")
}

/// Check that exactly one column arrived, so that reading the first one is safe.
///
/// The Python side always sends one, but a plugin is reachable by name and the crate does
/// not panic on anything a caller can send it.
fn one_input(inputs: &[Series], names: &[String], operation: &str) -> PolarsResult<Series> {
    let inputs = with_names(inputs, names)?;
    let [series] = inputs.as_slice() else {
        polars_bail!(
            InvalidOperation:
            "{} reads one column, but it was given {}", operation, inputs.len(),
        );
    };
    Ok(series.clone())
}

/// Estimate the autocovariance sequence, and normalise it when asked to.
fn sequence(inputs: &[Series], kwargs: &SequenceKwargs, name: &str) -> PolarsResult<Series> {
    let input = one_input(inputs, &kwargs.names, name)?;
    let read = Observations::from_series(&input, kwargs.null_policy, kwargs.demean, name)?;
    moments::check_max_lag(name, kwargs.max_lag, read.len())?;

    let estimated = moments::autocovariance(read.centred(), kwargs.max_lag, kwargs.unbiased);
    let reported = if name == "autocorrelation" {
        moments::normalise(&estimated, name)?
    } else {
        estimated
    };

    result::struct_row(
        name,
        &[
            result::text("series", &kwargs.names[0]),
            result::index_list("lags", 0..=kwargs.max_lag),
            result::float_list(name, &reported),
            result::number("mean", read.mean()),
            result::count("n_observations", read.len()),
        ],
    )
}

/// Estimate the autocovariance of a series against its own lags.
#[polars_expr(output_type_func=autocovariance_dtype)]
fn autocovariance(inputs: &[Series], kwargs: SequenceKwargs) -> PolarsResult<Series> {
    sequence(inputs, &kwargs, "autocovariance")
}

/// Estimate the autocorrelation of a series against its own lags.
#[polars_expr(output_type_func=autocorrelation_dtype)]
fn autocorrelation(inputs: &[Series], kwargs: SequenceKwargs) -> PolarsResult<Series> {
    sequence(inputs, &kwargs, "autocorrelation")
}

#[derive(Deserialize)]
struct AutoregressionKwargs {
    names: Vec<String>,
    order: Option<usize>,
    criterion: Option<Criterion>,
    max_order: usize,
    method: Method,
    demean: bool,
    null_policy: NullPolicy,
}

impl AutoregressionKwargs {
    /// What the caller asked the recursion for.
    fn options(&self) -> Options {
        Options {
            order: self.order,
            criterion: self.criterion,
            max_order: self.max_order,
            method: self.method,
        }
    }
}

fn partial_autocorrelation_dtype(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "partial_autocorrelation".into(),
        DataType::Struct(vec![
            Field::new("series".into(), DataType::String),
            Field::new("lags".into(), result::index_list_dtype()),
            Field::new("partial_autocorrelation".into(), result::float_list_dtype()),
            Field::new("n_observations".into(), DataType::UInt32),
        ]),
    ))
}

/// Report the reflection coefficient at every order up to the one asked for.
#[polars_expr(output_type_func=partial_autocorrelation_dtype)]
fn partial_autocorrelation(
    inputs: &[Series],
    kwargs: AutoregressionKwargs,
) -> PolarsResult<Series> {
    let name = "partial_autocorrelation";
    let fitted = fitted(inputs, &kwargs, name)?;

    result::struct_row(
        name,
        &[
            result::text("series", &kwargs.names[0]),
            result::index_list("lags", 1..=kwargs.max_order),
            result::float_list(name, &fitted.partial_autocorrelations),
            result::count("n_observations", fitted.n_observations),
        ],
    )
}

fn autoregression_dtype(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "autoregression".into(),
        DataType::Struct(vec![
            Field::new("series".into(), DataType::String),
            Field::new("coefficients".into(), result::float_list_dtype()),
            Field::new("mean".into(), DataType::Float64),
            Field::new("variance".into(), DataType::Float64),
            Field::new("order".into(), DataType::UInt32),
            Field::new(
                "partial_autocorrelations".into(),
                result::float_list_dtype(),
            ),
            Field::new("order_variance".into(), result::float_list_dtype()),
            Field::new("criterion".into(), result::float_list_dtype()),
            Field::new("n_observations".into(), DataType::UInt32),
            Field::new("stationary".into(), DataType::Boolean),
            Field::new("method".into(), DataType::String),
        ]),
    ))
}

/// Fit an autoregression to a series.
#[polars_expr(output_type_func=autoregression_dtype)]
fn autoregression(inputs: &[Series], kwargs: AutoregressionKwargs) -> PolarsResult<Series> {
    let fitted = fitted(inputs, &kwargs, "autoregression")?;

    result::struct_row(
        "autoregression",
        &[
            result::text("series", &kwargs.names[0]),
            result::float_list("coefficients", &fitted.coefficients),
            result::number("mean", fitted.mean),
            result::number("variance", fitted.variance),
            result::count("order", fitted.order),
            result::float_list("partial_autocorrelations", &fitted.partial_autocorrelations),
            result::float_list("order_variance", &fitted.order_variance),
            result::optional_float_list("criterion", fitted.criterion.as_deref()),
            result::count("n_observations", fitted.n_observations),
            result::flag("stationary", fitted.stationary),
            result::text("method", fitted.method.name()),
        ],
    )
}

/// Read the series and fit it, which every operation here starts by doing.
fn fitted(
    inputs: &[Series],
    kwargs: &AutoregressionKwargs,
    operation: &str,
) -> PolarsResult<Autoregression> {
    let input = one_input(inputs, &kwargs.names, operation)?;
    let read = Observations::from_series(&input, kwargs.null_policy, kwargs.demean, operation)?;
    moments::check_max_lag(operation, kwargs.max_order, read.len())?;
    fit::fit(&read, &kwargs.options(), operation)
}

#[derive(Deserialize)]
struct TransformKwargs {
    names: Vec<String>,
    order: Option<usize>,
    criterion: Option<Criterion>,
    max_order: usize,
    method: Method,
    demean: bool,
    output: Output,
    null_policy: NullPolicy,
}

/// Fit a series and report what the fitted filter does to the same rows.
#[polars_expr(output_type=Float64)]
fn autoregression_transform(inputs: &[Series], kwargs: TransformKwargs) -> PolarsResult<Series> {
    let operation = "autoregression_transform";
    let input = one_input(inputs, &kwargs.names, operation)?;
    let read = Observations::from_series(&input, kwargs.null_policy, kwargs.demean, operation)?;
    moments::check_max_lag(operation, kwargs.max_order, read.len())?;

    let options = Options {
        order: kwargs.order,
        criterion: kwargs.criterion,
        max_order: kwargs.max_order,
        method: kwargs.method,
    };
    let fitted = fit::fit(&read, &options, operation)?;
    let values = fit::apply(&fitted, &read, kwargs.output);

    Ok(Series::new(kwargs.names[0].as_str().into(), values))
}

#[derive(Deserialize)]
struct SolveToeplitzKwargs {
    names: Vec<String>,
    n_rhs: usize,
    diagonal_shift: f64,
}

fn solve_toeplitz_dtype(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "solve_toeplitz".into(),
        DataType::Struct(vec![
            Field::new("rhs".into(), result::string_list_dtype()),
            Field::new("solution".into(), result::matrix_dtype()),
            Field::new("size".into(), DataType::UInt32),
            Field::new("diagonal_shift".into(), DataType::Float64),
        ]),
    ))
}

/// Solve a symmetric positive-definite Toeplitz system given by its first column.
///
/// The inputs arrive as the first column, then the right-hand sides, then the row index.
/// The rows are put in the order of that index before anything is read, so the matrix does
/// not depend on the order the frame happens to be in.
#[polars_expr(output_type_func=solve_toeplitz_dtype)]
fn solve_toeplitz(inputs: &[Series], kwargs: SolveToeplitzKwargs) -> PolarsResult<Series> {
    let operation = "solve_toeplitz";
    if kwargs.n_rhs == 0 {
        polars_bail!(InvalidOperation: "{} needs at least one right-hand side", operation);
    }

    let inputs = with_names(inputs, &kwargs.names)?;
    // The first column, then the right-hand sides, then the row index.
    let expected = 2 + kwargs.n_rhs;
    if inputs.len() != expected {
        polars_bail!(
            InvalidOperation:
            "{} reads {} columns for {} right-hand sides, but it was given {}",
            operation, expected, kwargs.n_rhs, inputs.len(),
        );
    }

    let order = row_order(&inputs[expected - 1])?;
    let ordered: Vec<Series> = inputs[..expected - 1]
        .iter()
        .map(|series| series.take(&order))
        .collect::<PolarsResult<_>>()?;

    let columns: Vec<Vec<f64>> = ordered
        .iter()
        .map(|series| read_column(series, operation))
        .collect::<PolarsResult<_>>()?;
    let Some((first_column, rhs)) = columns.split_first() else {
        polars_bail!(InvalidOperation: "{} needs a first column", operation);
    };

    let solutions = levinson::solve(first_column, rhs, kwargs.diagonal_shift, operation)?;
    let solution = faer::Mat::from_fn(solutions.len(), first_column.len(), |i, j| solutions[i][j]);

    result::struct_row(
        operation,
        &[
            result::string_list("rhs", &kwargs.names[1..1 + kwargs.n_rhs]),
            result::matrix_rows("solution", solution.as_ref()),
            result::count("size", first_column.len()),
            result::number("diagonal_shift", kwargs.diagonal_shift),
        ],
    )
}

/// Read one column of a system that has no room for a missing value.
fn read_column(series: &Series, operation: &str) -> PolarsResult<Vec<f64>> {
    let cast = series.cast(&DataType::Float64)?;
    let column = cast.f64()?;
    column
        .iter()
        .enumerate()
        .map(|(row, value)| match value {
            None => polars_bail!(
                ComputeError:
                "{} received a null.\n\nInput:\n{}\n\nRow:\n{}\n\nA row cannot be dropped \
                 without changing the size of the system.",
                operation, series.name(), row,
            ),
            Some(value) if !value.is_finite() => polars_bail!(
                ComputeError:
                "{} received a value that is not finite.\n\nInput:\n{}\n\nRow:\n{}\n\n\
                 Value:\n{}",
                operation, series.name(), row, value,
            ),
            Some(value) => Ok(value),
        })
        .collect()
}
