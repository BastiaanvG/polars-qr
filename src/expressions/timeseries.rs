//! The timeseries expression entry points that Polars calls into.
//!
//! Each one decodes its options, reads the clock and the values, runs a state machine over
//! the rows and hands back one number per row. Nothing numerical lives here.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::timeseries::clock::{Clock, SpanSpec};
use crate::timeseries::options::{
    self, Bivariate, EwmOptions, NullPolicy, RollingOptions, Univariate, Values,
};
use crate::timeseries::{ewm, rolling};

/// What an exponentially weighted statistic is asked for.
#[derive(Deserialize)]
struct EwmKwargs {
    operation: String,
    half_life: SpanSpec,
    min_samples: u64,
    #[serde(default)]
    bias: bool,
    null_policy: NullPolicy,
}

/// What a hard-window statistic is asked for.
#[derive(Deserialize)]
struct RollingKwargs {
    operation: String,
    window: SpanSpec,
    min_samples: u64,
    min_clock_span: Option<SpanSpec>,
    #[serde(default = "one")]
    ddof: u32,
    null_policy: NullPolicy,
}

fn one() -> u32 {
    1
}

/// Read the clock and one column of values, checking they line up.
fn read_univariate(
    inputs: &[Series],
    operation: &str,
    policy: NullPolicy,
) -> PolarsResult<(Values, Clock)> {
    let values = Values::from_series(&inputs[0], policy, operation)?;
    let clock = Clock::from_series(&inputs[1], operation)?;
    options::check_lengths(
        operation,
        (inputs[0].name().as_str(), values.len()),
        (inputs[1].name().as_str(), clock.len()),
    )?;
    Ok((values, clock))
}

/// Read the clock and a pair of columns, checking they all line up.
fn read_bivariate(
    inputs: &[Series],
    operation: &str,
    policy: NullPolicy,
) -> PolarsResult<(Values, Values, Clock)> {
    let x = Values::from_series(&inputs[0], policy, operation)?;
    let y = Values::from_series(&inputs[1], policy, operation)?;
    let clock = Clock::from_series(&inputs[2], operation)?;
    options::check_lengths(
        operation,
        (inputs[0].name().as_str(), x.len()),
        (inputs[1].name().as_str(), y.len()),
    )?;
    options::check_lengths(
        operation,
        (inputs[0].name().as_str(), x.len()),
        (inputs[2].name().as_str(), clock.len()),
    )?;
    Ok((x, y, clock))
}

/// Which univariate statistic a name asks for.
fn univariate_statistic(operation: &str) -> PolarsResult<Univariate> {
    match operation.rsplit('_').next() {
        Some("sum") => Ok(Univariate::Sum),
        Some("mean") => Ok(Univariate::Mean),
        Some("variance") => Ok(Univariate::Variance),
        _ => polars_bail!(InvalidOperation: "unknown timeseries statistic '{}'", operation),
    }
}

/// Which bivariate statistic a name asks for.
fn bivariate_statistic(operation: &str) -> PolarsResult<Bivariate> {
    match operation.rsplit('_').next() {
        Some("covariance") => Ok(Bivariate::Covariance),
        Some("correlation") => Ok(Bivariate::Correlation),
        _ => polars_bail!(InvalidOperation: "unknown timeseries statistic '{}'", operation),
    }
}

/// An exponentially weighted statistic of one column over a clock.
#[polars_expr(output_type=Float64)]
fn ewm_univariate(inputs: &[Series], kwargs: EwmKwargs) -> PolarsResult<Series> {
    let operation = kwargs.operation.as_str();
    let (values, clock) = read_univariate(inputs, operation, kwargs.null_policy)?;
    let half_life = clock.resolve(kwargs.half_life, operation, "half_life")?;

    let options = EwmOptions {
        half_life,
        min_samples: kwargs.min_samples,
        bias: kwargs.bias,
    };
    let output = ewm::univariate(
        &values,
        &clock,
        &options,
        univariate_statistic(operation)?,
        operation,
    )?;
    Ok(options::output(inputs[0].name().as_str(), output))
}

/// An exponentially weighted statistic of a pair of columns over a clock.
#[polars_expr(output_type=Float64)]
fn ewm_bivariate(inputs: &[Series], kwargs: EwmKwargs) -> PolarsResult<Series> {
    let operation = kwargs.operation.as_str();
    let (x, y, clock) = read_bivariate(inputs, operation, kwargs.null_policy)?;
    let half_life = clock.resolve(kwargs.half_life, operation, "half_life")?;

    let options = EwmOptions {
        half_life,
        min_samples: kwargs.min_samples,
        bias: kwargs.bias,
    };
    let output = ewm::bivariate(
        &x,
        &y,
        &clock,
        &options,
        bivariate_statistic(operation)?,
        operation,
    )?;
    Ok(options::output(inputs[0].name().as_str(), output))
}

/// A hard-window statistic of one column over a clock.
#[polars_expr(output_type=Float64)]
fn rolling_univariate(inputs: &[Series], kwargs: RollingKwargs) -> PolarsResult<Series> {
    let operation = kwargs.operation.as_str();
    let (values, clock) = read_univariate(inputs, operation, kwargs.null_policy)?;
    let window = clock.resolve(kwargs.window, operation, "window")?;
    let min_clock_span = kwargs
        .min_clock_span
        .map(|span| clock.resolve(span, operation, "min_clock_span"))
        .transpose()?;

    let options = RollingOptions {
        window,
        min_samples: kwargs.min_samples,
        min_clock_span,
        ddof: kwargs.ddof,
    };
    let output = rolling::univariate(
        &values,
        &clock,
        &options,
        univariate_statistic(operation)?,
        operation,
    )?;
    Ok(options::output(inputs[0].name().as_str(), output))
}

/// A hard-window statistic of a pair of columns over a clock.
#[polars_expr(output_type=Float64)]
fn rolling_bivariate(inputs: &[Series], kwargs: RollingKwargs) -> PolarsResult<Series> {
    let operation = kwargs.operation.as_str();
    let (x, y, clock) = read_bivariate(inputs, operation, kwargs.null_policy)?;
    let window = clock.resolve(kwargs.window, operation, "window")?;
    let min_clock_span = kwargs
        .min_clock_span
        .map(|span| clock.resolve(span, operation, "min_clock_span"))
        .transpose()?;

    let options = RollingOptions {
        window,
        min_samples: kwargs.min_samples,
        min_clock_span,
        ddof: kwargs.ddof,
    };
    let output = rolling::bivariate(
        &x,
        &y,
        &clock,
        &options,
        bivariate_statistic(operation)?,
        operation,
    )?;
    Ok(options::output(inputs[0].name().as_str(), output))
}
