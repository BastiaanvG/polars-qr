//! Exponentially weighted statistics over a clock.

pub mod bivariate;
pub mod univariate;

use polars::prelude::*;

use crate::timeseries::clock::Clock;
use crate::timeseries::error;
use crate::timeseries::options::{Bivariate, EwmOptions, Univariate, Values};
use bivariate::EwmBivariateState;
use univariate::EwmUnivariateState;

/// How far a statistic has decayed after one step of the clock.
///
/// A half-life is the distance over which what is already held loses half its weight, so
/// the factor is two to the power of minus the distance in half-lives. Standing still
/// leaves the state untouched; a gap wide enough underflows to zero, which forgets the past
/// entirely and is the right answer rather than a failure.
fn decay(distance: f64, half_life: f64) -> f64 {
    if distance <= 0.0 {
        1.0
    } else {
        (-std::f64::consts::LN_2 * distance / half_life).exp()
    }
}

/// Check a half-life before anything is computed with it.
fn check_half_life(half_life: f64, operation: &str) -> PolarsResult<()> {
    if half_life.is_nan() || half_life <= 0.0 || !half_life.is_finite() {
        return Err(error::span_not_positive(operation, "half_life", half_life));
    }
    Ok(())
}

/// Run an exponentially weighted statistic over one column.
pub fn univariate(
    values: &Values,
    clock: &Clock,
    options: &EwmOptions,
    statistic: Univariate,
    operation: &str,
) -> PolarsResult<Vec<Option<f64>>> {
    check_half_life(options.half_life, operation)?;

    let mut state = EwmUnivariateState::default();
    let mut output = Vec::with_capacity(values.len());

    for row in 0..values.len() {
        if row > 0 {
            state.decay(decay(clock.distance(row - 1, row), options.half_life));
        }
        if let Some(value) = values.at(row) {
            state.add(value);
        }

        output.push(if state.valid_count() < options.min_samples {
            None
        } else {
            match statistic {
                Univariate::Sum => Some(state.sum()),
                Univariate::Mean => state.mean(),
                Univariate::Variance => state.variance(options.bias),
            }
        });
    }
    Ok(output)
}

/// Run an exponentially weighted statistic over a pair of columns.
pub fn bivariate(
    x: &Values,
    y: &Values,
    clock: &Clock,
    options: &EwmOptions,
    statistic: Bivariate,
    operation: &str,
) -> PolarsResult<Vec<Option<f64>>> {
    check_half_life(options.half_life, operation)?;

    let mut state = EwmBivariateState::default();
    let mut output = Vec::with_capacity(x.len());

    for row in 0..x.len() {
        if row > 0 {
            state.decay(decay(clock.distance(row - 1, row), options.half_life));
        }
        if let (Some(x), Some(y)) = (x.at(row), y.at(row)) {
            state.add(x, y);
        }

        output.push(if state.valid_count() < options.min_samples {
            None
        } else {
            match statistic {
                Bivariate::Covariance => state.covariance(options.bias),
                Bivariate::Correlation => state.correlation(),
            }
        });
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standing_still_does_not_decay() {
        assert_eq!(decay(0.0, 10.0), 1.0);
    }

    #[test]
    fn a_half_life_of_clock_halves_the_weight() {
        assert!((decay(10.0, 10.0) - 0.5).abs() < 1e-15);
        assert!((decay(20.0, 10.0) - 0.25).abs() < 1e-15);
    }

    #[test]
    fn a_gap_wide_enough_forgets_everything() {
        assert_eq!(decay(1e9, 1.0), 0.0);
    }

    #[test]
    fn a_half_life_must_be_positive_and_finite() {
        assert!(check_half_life(0.0, "test").is_err());
        assert!(check_half_life(-1.0, "test").is_err());
        assert!(check_half_life(f64::INFINITY, "test").is_err());
        assert!(check_half_life(f64::NAN, "test").is_err());
        assert!(check_half_life(1e-9, "test").is_ok());
    }
}
