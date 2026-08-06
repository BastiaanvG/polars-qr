//! What the caller asks for, and the values a statistic reads.

use polars::prelude::*;
use serde::Deserialize;

use crate::timeseries::error;

/// What to do about a null value.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NullPolicy {
    /// Let the state decay past the row without adding anything to it.
    #[default]
    Skip,
    /// Fail on it.
    Raise,
}

/// One column of observations, checked against the null policy.
///
/// A value that is not finite is always an error. Reading it as missing would be a guess
/// about what the caller meant, and the two possible guesses — that it is a gap, or that it
/// is a real infinity — lead to different numbers.
pub struct Values {
    values: Vec<Option<f64>>,
}

impl Values {
    /// Read a column of observations.
    pub fn from_series(series: &Series, policy: NullPolicy, operation: &str) -> PolarsResult<Self> {
        let name = series.name().to_string();
        let cast = series.cast(&DataType::Float64)?;
        let column = cast.f64()?;

        let mut values = Vec::with_capacity(column.len());
        for (row, value) in column.iter().enumerate() {
            match value {
                None if policy == NullPolicy::Raise => {
                    return Err(error::value_null(operation, &name, row))
                }
                None => values.push(None),
                Some(value) if !value.is_finite() => {
                    return Err(error::value_not_finite(operation, &name, row, value))
                }
                Some(value) => values.push(Some(value)),
            }
        }
        Ok(Self { values })
    }

    /// The value at `row`, if there is one.
    pub fn at(&self, row: usize) -> Option<f64> {
        self.values[row]
    }

    /// How many rows there are.
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

/// How an exponentially weighted statistic is set up.
pub struct EwmOptions {
    /// The clock distance over which held weight halves.
    pub half_life: f64,
    /// How many valid observations are needed before a number is reported.
    pub min_samples: u64,
    /// Whether to divide by the total weight as it stands.
    pub bias: bool,
}

/// How a hard-window statistic is set up.
pub struct RollingOptions {
    /// The width of the window, in clock units.
    pub window: f64,
    /// How many valid observations are needed before a number is reported.
    pub min_samples: u64,
    /// How much clock the window must cover before a number is reported.
    pub min_clock_span: Option<f64>,
    /// The delta degrees of freedom subtracted from the divisor.
    pub ddof: u32,
}

/// Which univariate statistic to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Univariate {
    /// The decaying or windowed sum.
    Sum,
    /// The weighted mean.
    Mean,
    /// The weighted variance.
    Variance,
}

/// Which statistic of a pair of columns to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bivariate {
    /// The weighted covariance.
    Covariance,
    /// The weighted correlation.
    Correlation,
}

/// Check that two inputs line up row for row.
pub fn check_lengths(
    operation: &str,
    first: (&str, usize),
    second: (&str, usize),
) -> PolarsResult<()> {
    if first.1 != second.1 {
        return Err(error::length_mismatch(operation, first, second));
    }
    Ok(())
}

/// Build the output column a timeseries statistic returns.
pub fn output(name: &str, values: Vec<Option<f64>>) -> Series {
    Series::new(name.into(), values)
}
