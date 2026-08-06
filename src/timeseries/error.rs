//! The errors the timeseries operations raise.
//!
//! Every one of them names the operation, what was wrong and what would fix it. A caller
//! sees these through a Polars compute error, with no stack to read, so the message is the
//! whole of the diagnosis.

use polars::prelude::*;

/// The clock went backwards.
pub fn clock_decreased(operation: &str, name: &str, previous: &str, current: &str) -> PolarsError {
    polars_err!(
        ComputeError:
        "{} requires a non-decreasing clock.\n\nClock:\n{}\n\nPrevious clock:\n{}\n\n\
         Current clock:\n{}\n\nSort the frame by the clock, and include whatever the clock \
         restarts on in the grouping.",
        operation, name, previous, current,
    )
}

/// The clock has a null in it.
pub fn clock_null(operation: &str, name: &str, row: usize) -> PolarsError {
    polars_err!(
        ComputeError:
        "{} received a null clock value.\n\nClock:\n{}\n\nRow:\n{}\n\nA null cannot place an \
         observation in time, so there is nothing sensible to skip to.",
        operation, name, row,
    )
}

/// The clock has a value that is not a finite number.
pub fn clock_not_finite(operation: &str, name: &str, row: usize, value: f64) -> PolarsError {
    polars_err!(
        ComputeError:
        "{} received a clock value that is not finite.\n\nClock:\n{}\n\nRow:\n{}\n\nValue:\n{}",
        operation, name, row, value,
    )
}

/// The clock is of a dtype that cannot be read as a clock.
pub fn clock_dtype(operation: &str, name: &str, dtype: &DataType) -> PolarsError {
    polars_err!(
        InvalidOperation:
        "{} received a clock of dtype {}.\n\nClock:\n{}\n\nA clock must be an integer, a \
         float, a Date or a Datetime.",
        operation, dtype, name,
    )
}

/// A numeric clock was given a span meant for a temporal one, or the other way round.
pub fn span_mismatch(
    operation: &str,
    parameter: &str,
    dtype: &DataType,
    received: &str,
    wanted: &str,
) -> PolarsError {
    polars_err!(
        InvalidOperation:
        "{} received a {} clock and a {} {}.\n\nClock dtype:\n{}\n\n{}",
        operation,
        if dtype.is_temporal() { "temporal" } else { "numeric" },
        received,
        parameter,
        dtype,
        wanted,
    )
}

/// A span that has to be positive was not.
pub fn span_not_positive(operation: &str, parameter: &str, value: f64) -> PolarsError {
    polars_err!(
        InvalidOperation:
        "{} requires {} > 0.\n\nReceived:\n{}={}",
        operation, parameter, parameter, value,
    )
}

/// Two inputs that have to line up row for row did not.
pub fn length_mismatch(
    operation: &str,
    first: (&str, usize),
    second: (&str, usize),
) -> PolarsError {
    polars_err!(
        ShapeMismatch:
        "{} requires matching input lengths.\n\n{} length:\n{}\n\n{} length:\n{}",
        operation, first.0, first.1, second.0, second.1,
    )
}

/// A value is not a finite number.
pub fn value_not_finite(operation: &str, name: &str, row: usize, value: f64) -> PolarsError {
    polars_err!(
        ComputeError:
        "{} received a value that is not finite.\n\nInput:\n{}\n\nRow:\n{}\n\nValue:\n{}\n\n\
         Filter or replace it before the statistic reads it.",
        operation, name, row, value,
    )
}

/// A null turned up under `null_policy="raise"`.
pub fn value_null(operation: &str, name: &str, row: usize) -> PolarsError {
    polars_err!(
        ComputeError:
        "{} received a null value.\n\nInput:\n{}\n\nRow:\n{}\n\nPass null_policy='skip' to \
         let the state decay past it instead.",
        operation, name, row,
    )
}

/// `min_clock_span` asked for more than the window holds.
pub fn span_exceeds_window(operation: &str, min_clock_span: f64, window: f64) -> PolarsError {
    polars_err!(
        InvalidOperation:
        "{} requires min_clock_span <= window.\n\nReceived:\nmin_clock_span={}\nwindow={}\n\n\
         A window that short can never cover the span, so every row would be null.",
        operation, min_clock_span, window,
    )
}
