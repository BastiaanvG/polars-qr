//! The expression entry points that Polars calls into.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::covariance;
use crate::dense::{DenseFrame, NullPolicy};
use crate::least_squares::{self, Solver};
use crate::pca;
use crate::result::{self, binary_row, concatenate_rows};
use crate::spd;
use crate::states::covariance::CovarianceState;
use crate::states::least_squares::LeastSquaresState;
use crate::weights::Weights;

/// Label the inputs with the names their expressions carry.
///
/// Inside a grouped aggregation Polars passes a plugin its inputs without names, so the
/// names travel as keyword arguments instead. Renaming here means everything downstream —
/// results and error messages alike — sees the columns the caller asked for.
fn with_names(inputs: &[Series], names: &[String]) -> PolarsResult<Vec<Series>> {
    if names.len() != inputs.len() {
        polars_bail!(
            ShapeMismatch:
            "{} inputs were given {} names", inputs.len(), names.len(),
        );
    }
    Ok(inputs
        .iter()
        .zip(names)
        .map(|(series, name)| series.clone().with_name(name.as_str().into()))
        .collect())
}

/// The version of the compiled plugin, so a stale build is easy to spot from Python.
#[polars_expr(output_type=String)]
fn plugin_version(_inputs: &[Series]) -> PolarsResult<Series> {
    let version = env!("CARGO_PKG_VERSION");
    Ok(Series::new("version".into(), [version]))
}

#[derive(Deserialize)]
struct LeastSquaresKwargs {
    names: Vec<String>,
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

    let inputs = with_names(inputs, &kwargs.names)?;
    let dense = DenseFrame::from_series(&inputs, kwargs.null_policy)?;
    let matrix = dense.matrix();
    let n_features = dense.n_cols() - n_targets - n_weights;
    let targets = matrix.subcols(0, n_targets);
    let features = matrix.subcols(n_targets, n_features);
    let names = &kwargs.names;

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

    least_squares_row(
        &names[n_targets..n_targets + n_features],
        &names[..n_targets],
        &fit,
    )
}

/// The one struct row a fit is reported as, however it was arrived at.
fn least_squares_row(
    features: &[String],
    targets: &[String],
    fit: &least_squares::LeastSquaresFit,
) -> PolarsResult<Series> {
    result::struct_row(
        "least_squares",
        &[
            result::string_list("features", features),
            result::string_list("targets", targets),
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
    names: Vec<String>,
    ddof: f64,
    weighted: bool,
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
            Field::new("sum_weights".into(), DataType::Float64),
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
    let inputs = with_names(inputs, &kwargs.names)?;
    let dense = DenseFrame::from_series(&inputs, kwargs.null_policy)?;
    let names = &kwargs.names;
    let matrix = dense.matrix();
    let n_features = dense.n_cols() - usize::from(kwargs.weighted);
    if n_features == 0 {
        polars_bail!(InvalidOperation: "at least one feature is required");
    }

    let weights = kwargs
        .weighted
        .then(|| Weights::new(matrix.subcols(n_features, 1), &names[n_features]))
        .transpose()?;
    let options = covariance::Options {
        ddof: kwargs.ddof,
        normalise: kwargs.normalise,
    };
    let estimate =
        covariance::covariance(matrix.subcols(0, n_features), weights.as_ref(), &options)?;

    second_moment_row(&names[..n_features], &estimate, name)
}

/// The one struct row a second-moment estimate is reported as, however it was arrived at.
fn second_moment_row(
    features: &[String],
    estimate: &covariance::Covariance,
    name: &str,
) -> PolarsResult<Series> {
    result::struct_row(
        name,
        &[
            result::string_list("features", features),
            result::float_list("means", &estimate.means),
            result::float_list("standard_deviations", &estimate.standard_deviations),
            result::matrix_rows(name, estimate.values.as_ref()),
            result::count("n_observations", estimate.n_observations),
            result::number("sum_weights", estimate.sum_weights),
            result::count("size", features.len()),
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

#[derive(Deserialize)]
struct PcaKwargs {
    names: Vec<String>,
    n_components: Option<usize>,
    centre: bool,
    scale: bool,
    null_policy: NullPolicy,
}

fn pca_dtype(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "pca".into(),
        DataType::Struct(vec![
            Field::new("features".into(), result::string_list_dtype()),
            Field::new("means".into(), result::float_list_dtype()),
            Field::new("scales".into(), result::float_list_dtype()),
            Field::new("components".into(), result::matrix_dtype()),
            Field::new("singular_values".into(), result::float_list_dtype()),
            Field::new("explained_variance".into(), result::float_list_dtype()),
            Field::new(
                "explained_variance_ratio".into(),
                result::float_list_dtype(),
            ),
            Field::new("rank".into(), DataType::UInt32),
            Field::new("n_observations".into(), DataType::UInt32),
        ]),
    ))
}

/// Find the principal components of the input columns.
#[polars_expr(output_type_func=pca_dtype)]
fn pca(inputs: &[Series], kwargs: PcaKwargs) -> PolarsResult<Series> {
    let inputs = with_names(inputs, &kwargs.names)?;
    let dense = DenseFrame::from_series(&inputs, kwargs.null_policy)?;
    let options = pca::Options {
        n_components: kwargs.n_components,
        centre: kwargs.centre,
        scale: kwargs.scale,
    };
    let found = pca::pca(dense.matrix(), &options)?;
    let names = &kwargs.names;

    result::struct_row(
        "pca",
        &[
            result::string_list("features", &names),
            result::float_list("means", &found.means),
            result::float_list("scales", &found.scales),
            result::matrix_rows("components", found.components.as_ref()),
            result::float_list("singular_values", &found.singular_values),
            result::float_list("explained_variance", &found.explained_variance),
            result::float_list("explained_variance_ratio", &found.explained_variance_ratio),
            result::count("rank", found.rank),
            result::count("n_observations", found.n_observations),
        ],
    )
}

#[derive(Deserialize)]
struct PcaTransformKwargs {
    names: Vec<String>,
    n_components: usize,
    centre: bool,
    scale: bool,
    null_policy: NullPolicy,
}

/// The names of the score columns, which are fixed by the number of components asked for.
fn component_names(n_components: usize) -> Vec<String> {
    (1..=n_components)
        .map(|index| format!("component_{index}"))
        .collect()
}

fn pca_transform_dtype(_: &[Field], kwargs: PcaTransformKwargs) -> PolarsResult<Field> {
    let fields = component_names(kwargs.n_components)
        .into_iter()
        .map(|name| Field::new(name.into(), DataType::Float64))
        .collect();
    Ok(Field::new("pca_transform".into(), DataType::Struct(fields)))
}

/// Score every row on the components found from the rows it was read with.
///
/// The result has one row for every row that was given, so it can sit next to the input in
/// the same frame. Rows the null policy dropped score as null.
#[polars_expr(output_type_func_with_kwargs=pca_transform_dtype)]
fn pca_transform(inputs: &[Series], kwargs: PcaTransformKwargs) -> PolarsResult<Series> {
    let inputs = with_names(inputs, &kwargs.names)?;
    let dense = DenseFrame::from_series(&inputs, kwargs.null_policy)?;
    let options = pca::Options {
        n_components: Some(kwargs.n_components),
        centre: kwargs.centre,
        scale: kwargs.scale,
    };
    let found = pca::pca(dense.matrix(), &options)?;
    let scores = pca::scores(dense.matrix(), &found);

    let fields: Vec<Series> = component_names(kwargs.n_components)
        .iter()
        .enumerate()
        .map(|(index, name)| {
            result::scattered(
                name,
                dense.valid(),
                (0..scores.nrows()).map(|i| scores[(i, index)]),
            )
        })
        .collect();
    result::struct_rows("pca_transform", dense.height(), &fields)
}

#[derive(Deserialize)]
struct SolveSpdKwargs {
    names: Vec<String>,
    n_matrix: usize,
    n_rhs: usize,
    diagonal_shift: f64,
}

fn solve_spd_dtype(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "solve_spd".into(),
        DataType::Struct(vec![
            Field::new("rows".into(), result::string_list_dtype()),
            Field::new("rhs".into(), result::string_list_dtype()),
            Field::new("solution".into(), result::matrix_dtype()),
            Field::new("size".into(), DataType::UInt32),
            Field::new("symmetry_error".into(), DataType::Float64),
            Field::new("diagonal_shift".into(), DataType::Float64),
        ]),
    ))
}

/// Solve a positive-definite system given as a wide frame.
///
/// The inputs arrive as the matrix columns, then the right-hand sides, then the row index.
/// The rows are put in the order of that index before anything is read, so the matrix does
/// not depend on the order the frame happens to be in.
#[polars_expr(output_type_func=solve_spd_dtype)]
fn solve_spd(inputs: &[Series], kwargs: SolveSpdKwargs) -> PolarsResult<Series> {
    let (n_matrix, n_rhs) = (kwargs.n_matrix, kwargs.n_rhs);
    if n_matrix == 0 || n_rhs == 0 {
        polars_bail!(InvalidOperation: "solve_spd needs a matrix and at least one right-hand side");
    }

    let inputs = with_names(inputs, &kwargs.names)?;
    let order = row_order(&inputs[n_matrix + n_rhs])?;
    let ordered: Vec<Series> = inputs[..n_matrix + n_rhs]
        .iter()
        .map(|series| series.take(&order))
        .collect::<PolarsResult<_>>()?;

    // Dropping rows would change the shape of the matrix, so the only policy that makes
    // sense here is to insist on complete input.
    let dense = DenseFrame::from_series(&ordered, NullPolicy::Raise)?;
    let matrix = dense.matrix();
    let solved = spd::solve_spd(
        matrix.subcols(0, n_matrix),
        matrix.subcols(n_matrix, n_rhs),
        &spd::Options {
            diagonal_shift: kwargs.diagonal_shift,
        },
    )?;
    let names = &kwargs.names;

    result::struct_row(
        "solve_spd",
        &[
            result::string_list("rows", &names[..n_matrix]),
            result::string_list("rhs", &names[n_matrix..n_matrix + n_rhs]),
            result::matrix_rows("solution", solved.solution.transpose()),
            result::count("size", n_matrix),
            result::number("symmetry_error", solved.symmetry_error),
            result::number("diagonal_shift", kwargs.diagonal_shift),
        ],
    )
}

/// The order the rows have to be read in, taken from an integer index column.
fn row_order(index: &Series) -> PolarsResult<IdxCa> {
    if !index.dtype().is_integer() {
        polars_bail!(
            InvalidOperation:
            "the row index '{}' has dtype {}, which is not an integer",
            index.name(), index.dtype(),
        );
    }
    if index.null_count() > 0 {
        polars_bail!(ComputeError: "the row index '{}' has nulls", index.name());
    }
    if index.n_unique()? != index.len() {
        polars_bail!(
            ComputeError:
            "the row index '{}' repeats a value, so the rows have no single order",
            index.name(),
        );
    }
    Ok(index.arg_sort(SortOptions::default().with_maintain_order(true)))
}

#[derive(Deserialize)]
struct StateKwargs {
    names: Vec<String>,
    n_targets: usize,
    weighted: bool,
    intercept: bool,
    null_policy: NullPolicy,
}

/// Summarise a partition of a least-squares problem into a state that can be merged.
#[polars_expr(output_type=Binary)]
fn least_squares_state(inputs: &[Series], kwargs: StateKwargs) -> PolarsResult<Series> {
    let n_targets = kwargs.n_targets;
    let n_weights = usize::from(kwargs.weighted);
    if n_targets == 0 {
        polars_bail!(InvalidOperation: "least_squares_state needs at least one target");
    }
    if inputs.len() <= n_targets + n_weights {
        polars_bail!(InvalidOperation: "least_squares_state needs at least one feature");
    }

    let inputs = with_names(inputs, &kwargs.names)?;
    let dense = DenseFrame::from_series(&inputs, kwargs.null_policy)?;
    let matrix = dense.matrix();
    let n_features = dense.n_cols() - n_targets - n_weights;
    let names = &kwargs.names;
    let weights = kwargs
        .weighted
        .then(|| {
            Weights::new(
                matrix.subcols(dense.n_cols() - 1, 1),
                &names[names.len() - 1],
            )
        })
        .transpose()?;

    let state = LeastSquaresState::accumulate(
        matrix.subcols(n_targets, n_features),
        matrix.subcols(0, n_targets),
        weights.as_ref(),
        names[n_targets..n_targets + n_features].to_vec(),
        names[..n_targets].to_vec(),
        kwargs.intercept,
    )?;
    Ok(binary_row("least_squares_state", state.encode()))
}

/// Merge every least-squares state in the input into one.
#[polars_expr(output_type=Binary)]
fn merge_least_squares_states(inputs: &[Series]) -> PolarsResult<Series> {
    let states = inputs[0].binary()?;
    let mut merged: Option<LeastSquaresState> = None;
    for bytes in states.into_iter().flatten() {
        let state = LeastSquaresState::decode(bytes)?;
        merged = Some(match merged {
            Some(existing) => existing.merge(&state)?,
            None => state,
        });
    }
    let merged = merged
        .ok_or_else(|| polars_err!(ComputeError: "there are no least-squares states to merge"))?;
    Ok(binary_row("least_squares_state", merged.encode()))
}

#[derive(Deserialize)]
struct FinaliseLeastSquaresKwargs {
    solver: Solver,
    l2_penalty: f64,
}

/// Solve the problem a merged least-squares state summarises.
#[polars_expr(output_type_func=least_squares_dtype)]
fn finalise_least_squares(
    inputs: &[Series],
    kwargs: FinaliseLeastSquaresKwargs,
) -> PolarsResult<Series> {
    let states = inputs[0].binary()?;
    let mut rows = Vec::with_capacity(states.len());
    for bytes in states.into_iter() {
        let bytes =
            bytes.ok_or_else(|| polars_err!(ComputeError: "a least-squares state is null"))?;
        let state = LeastSquaresState::decode(bytes)?;
        let fit = state.finalise(kwargs.solver, kwargs.l2_penalty)?;
        rows.push(least_squares_row(&state.features, &state.targets, &fit)?);
    }
    concatenate_rows("least_squares", rows, least_squares_dtype(&[])?)
}

#[derive(Deserialize)]
struct CovarianceStateKwargs {
    names: Vec<String>,
    weighted: bool,
    null_policy: NullPolicy,
}

/// Summarise a partition of a set of columns into a state that can be merged.
#[polars_expr(output_type=Binary)]
fn covariance_state(inputs: &[Series], kwargs: CovarianceStateKwargs) -> PolarsResult<Series> {
    let inputs = with_names(inputs, &kwargs.names)?;
    let dense = DenseFrame::from_series(&inputs, kwargs.null_policy)?;
    let names = &kwargs.names;
    let matrix = dense.matrix();
    let n_features = dense.n_cols() - usize::from(kwargs.weighted);
    if n_features == 0 {
        polars_bail!(InvalidOperation: "at least one feature is required");
    }

    let weights = kwargs
        .weighted
        .then(|| Weights::new(matrix.subcols(n_features, 1), &names[n_features]))
        .transpose()?;
    let state = CovarianceState::accumulate(
        matrix.subcols(0, n_features),
        weights.as_ref(),
        names[..n_features].to_vec(),
    )?;
    Ok(binary_row("covariance_state", state.encode()))
}

/// Merge every covariance state in the input into one.
#[polars_expr(output_type=Binary)]
fn merge_covariance_states(inputs: &[Series]) -> PolarsResult<Series> {
    let states = inputs[0].binary()?;
    let mut merged: Option<CovarianceState> = None;
    for bytes in states.into_iter().flatten() {
        let state = CovarianceState::decode(bytes)?;
        merged = Some(match merged {
            Some(existing) => existing.merge(&state)?,
            None => state,
        });
    }
    let merged = merged
        .ok_or_else(|| polars_err!(ComputeError: "there are no covariance states to merge"))?;
    Ok(binary_row("covariance_state", merged.encode()))
}

#[derive(Deserialize)]
struct FinaliseCovarianceKwargs {
    ddof: f64,
    normalise: bool,
}

/// Turn a merged covariance state into the matrix it summarises.
fn finalise_second_moments(
    inputs: &[Series],
    kwargs: &FinaliseCovarianceKwargs,
    name: &str,
) -> PolarsResult<Series> {
    let states = inputs[0].binary()?;
    let options = covariance::Options {
        ddof: kwargs.ddof,
        normalise: kwargs.normalise,
    };
    let mut rows = Vec::with_capacity(states.len());
    for bytes in states.into_iter() {
        let bytes = bytes.ok_or_else(|| polars_err!(ComputeError: "a covariance state is null"))?;
        let state = CovarianceState::decode(bytes)?;
        let estimate = state.finalise(&options)?;
        rows.push(second_moment_row(&state.features, &estimate, name)?);
    }
    concatenate_rows(name, rows, second_moment_dtype(name)?)
}

/// Turn a merged covariance state into the covariance it summarises.
#[polars_expr(output_type_func=covariance_dtype)]
fn finalise_covariance(
    inputs: &[Series],
    kwargs: FinaliseCovarianceKwargs,
) -> PolarsResult<Series> {
    finalise_second_moments(inputs, &kwargs, "covariance")
}

/// Turn a merged covariance state into the correlation it summarises.
#[polars_expr(output_type_func=correlation_dtype)]
fn finalise_correlation(
    inputs: &[Series],
    kwargs: FinaliseCovarianceKwargs,
) -> PolarsResult<Series> {
    finalise_second_moments(inputs, &kwargs, "correlation")
}
