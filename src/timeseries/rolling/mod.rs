//! Hard-window statistics over a clock.
//!
//! The window at each row covers the clock interval `(c - window, c]`: everything the clock
//! has passed within a window's width, up to and including the current row, and nothing
//! ahead of it. Observations are whole. A trade that straddles the far edge of a volume
//! window is either in or out; it is never split to make the window hold exactly the width
//! asked for.

pub mod bivariate;
pub mod univariate;

use std::collections::VecDeque;

use polars::prelude::*;

use crate::timeseries::clock::Clock;
use crate::timeseries::error;
use crate::timeseries::options::{Bivariate, RollingOptions, Univariate, Values};
use bivariate::RollingBivariateState;
use univariate::RollingUnivariateState;

/// How many removals to allow before the state is rebuilt from the window it holds.
///
/// Each add and remove leaves a little rounding behind. Over a long run it accumulates, and
/// the window is already in memory, so clearing it costs one pass over what is there. The
/// rebuild also picks a fresh origin, which is what keeps the state accurate when the level
/// of the data wanders away from where the window started.
const REBUILD_AFTER: u64 = 1024;

/// Round a centred second moment that rounding has pushed just below zero back up to it.
fn clamp_to_zero(moment: f64) -> f64 {
    if moment < 0.0 && moment > -1e-12 {
        0.0
    } else {
        moment
    }
}

/// Check a window before anything is computed with it.
fn check_window(window: f64, min_clock_span: Option<f64>, operation: &str) -> PolarsResult<()> {
    if window.is_nan() || window <= 0.0 || !window.is_finite() {
        return Err(error::span_not_positive(operation, "window", window));
    }
    if let Some(span) = min_clock_span {
        if span > window {
            return Err(error::span_exceeds_window(operation, span, window));
        }
    }
    Ok(())
}

/// Whether the window covers enough clock to report on.
fn spans_enough(
    clock: &Clock,
    oldest: Option<usize>,
    row: usize,
    min_clock_span: Option<f64>,
) -> bool {
    match (min_clock_span, oldest) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(span), Some(oldest)) => clock.distance(oldest, row) >= span,
    }
}

/// Run a hard-window statistic over one column.
pub fn univariate(
    values: &Values,
    clock: &Clock,
    options: &RollingOptions,
    statistic: Univariate,
    operation: &str,
) -> PolarsResult<Vec<Option<f64>>> {
    check_window(options.window, options.min_clock_span, operation)?;

    let mut state = RollingUnivariateState::default();
    let mut active: VecDeque<(usize, f64)> = VecDeque::new();
    let mut removals = 0u64;
    let mut output = Vec::with_capacity(values.len());

    for row in 0..values.len() {
        if let Some(value) = values.at(row) {
            state.add(value);
            active.push_back((row, value));
        }

        // Everything the clock has left behind by a full window drops out.
        while let Some((oldest, value)) = active.front().copied() {
            if clock.distance(oldest, row) < options.window {
                break;
            }
            state.remove(value);
            active.pop_front();
            removals += 1;
        }

        if removals >= REBUILD_AFTER {
            state.rebuild(active.iter().map(|(_, value)| *value));
            removals = 0;
        }

        let oldest = active.front().map(|(row, _)| *row);
        output.push(
            if state.count() < options.min_samples.max(1)
                || !spans_enough(clock, oldest, row, options.min_clock_span)
            {
                None
            } else {
                match statistic {
                    Univariate::Sum => Some(state.sum()),
                    Univariate::Mean => state.mean(),
                    Univariate::Variance => state.variance(options.ddof),
                }
            },
        );
    }
    Ok(output)
}

/// Run a hard-window statistic over a pair of columns.
pub fn bivariate(
    x: &Values,
    y: &Values,
    clock: &Clock,
    options: &RollingOptions,
    statistic: Bivariate,
    operation: &str,
) -> PolarsResult<Vec<Option<f64>>> {
    check_window(options.window, options.min_clock_span, operation)?;

    let mut state = RollingBivariateState::default();
    let mut active: VecDeque<(usize, f64, f64)> = VecDeque::new();
    let mut removals = 0u64;
    let mut output = Vec::with_capacity(x.len());

    for row in 0..x.len() {
        if let (Some(x), Some(y)) = (x.at(row), y.at(row)) {
            state.add(x, y);
            active.push_back((row, x, y));
        }

        while let Some((oldest, x, y)) = active.front().copied() {
            if clock.distance(oldest, row) < options.window {
                break;
            }
            state.remove(x, y);
            active.pop_front();
            removals += 1;
        }

        if removals >= REBUILD_AFTER {
            state.rebuild(active.iter().map(|(_, x, y)| (*x, *y)));
            removals = 0;
        }

        let oldest = active.front().map(|(row, _, _)| *row);
        output.push(
            if state.count() < options.min_samples.max(1)
                || !spans_enough(clock, oldest, row, options.min_clock_span)
            {
                None
            } else {
                match statistic {
                    Bivariate::Covariance => state.covariance(options.ddof),
                    Bivariate::Correlation => state.correlation(),
                }
            },
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_must_be_positive_and_finite() {
        assert!(check_window(0.0, None, "test").is_err());
        assert!(check_window(-1.0, None, "test").is_err());
        assert!(check_window(f64::INFINITY, None, "test").is_err());
        assert!(check_window(10.0, None, "test").is_ok());
    }

    #[test]
    fn a_minimum_span_cannot_exceed_the_window_that_holds_it() {
        assert!(check_window(10.0, Some(20.0), "test").is_err());
        assert!(check_window(10.0, Some(10.0), "test").is_ok());
    }
}
