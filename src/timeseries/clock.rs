//! The clock the statistics age against.
//!
//! A clock is any column that only ever moves forward: wall time, cumulative volume, a
//! trade count, cumulative squared return. What matters to a statistic is never the
//! position of the clock but the distance between two of its values, so that is the only
//! thing this module hands out.
//!
//! Integer and temporal clocks are kept as integers rather than widened to `f64`. A
//! microsecond timestamp is already past the point where `f64` spaces its values one apart,
//! and a cumulative-volume clock can get there too. Differences are taken first and widened
//! second, which keeps them exact whatever the clock has counted up to.

use polars::prelude::*;
use serde::Deserialize;

use crate::timeseries::error;

/// A window width or a half-life, as it arrives from Python.
///
/// Numeric clocks take a number in their own units; temporal clocks take a `timedelta`,
/// which crosses over as a count of nanoseconds. Mixing the two is an error rather than a
/// guess, because both guesses are wrong: a number is not a duration, and a duration says
/// nothing about how much volume has traded.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanSpec {
    /// A number in the units of a numeric clock.
    Numeric(f64),
    /// A `timedelta`, in nanoseconds, for a temporal clock.
    Nanoseconds(i64),
}

/// The values of a clock, in whatever form keeps its differences exact.
enum Values {
    /// Integer and temporal clocks.
    Integral(Vec<i64>),
    /// Floating-point clocks.
    Real(Vec<f64>),
}

/// A validated clock: no nulls, nothing infinite, and never going backwards.
pub struct Clock {
    values: Values,
    dtype: DataType,
    name: String,
}

impl Clock {
    /// Read and validate a clock column.
    ///
    /// The whole column is checked before any statistic reads it, so a clock that turns
    /// around halfway through a group fails the query rather than quietly producing numbers
    /// for the rows before the turn.
    pub fn from_series(series: &Series, operation: &str) -> PolarsResult<Self> {
        let name = series.name().to_string();
        let dtype = series.dtype().clone();

        if let Some(row) = first_null(series) {
            return Err(error::clock_null(operation, &name, row));
        }

        let values = match &dtype {
            DataType::Float32 | DataType::Float64 => {
                let cast = series.cast(&DataType::Float64)?;
                let values: Vec<f64> = cast.f64()?.into_no_null_iter().collect();
                for (row, value) in values.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(error::clock_not_finite(operation, &name, row, *value));
                    }
                }
                Values::Real(values)
            }
            dtype if is_integral(dtype) => {
                let cast = series.cast(&DataType::Int64)?;
                Values::Integral(cast.i64()?.into_no_null_iter().collect())
            }
            dtype => return Err(error::clock_dtype(operation, &name, dtype)),
        };

        let clock = Self {
            values,
            dtype,
            name,
        };
        clock.check_monotone(operation)?;
        Ok(clock)
    }

    /// How many observations the clock covers.
    pub fn len(&self) -> usize {
        match &self.values {
            Values::Integral(values) => values.len(),
            Values::Real(values) => values.len(),
        }
    }

    /// The distance from the clock at `earlier` to the clock at `later`.
    ///
    /// The subtraction happens before the widening, so the result is exact even when the
    /// clock itself has counted past what a float can represent one unit at a time.
    pub fn distance(&self, earlier: usize, later: usize) -> f64 {
        match &self.values {
            Values::Integral(values) => {
                (i128::from(values[later]) - i128::from(values[earlier])) as f64
            }
            Values::Real(values) => values[later] - values[earlier],
        }
    }

    /// Turn a span from Python into a distance in the units of this clock.
    pub fn resolve(&self, span: SpanSpec, operation: &str, parameter: &str) -> PolarsResult<f64> {
        match (span, self.dtype.is_temporal()) {
            (SpanSpec::Numeric(value), false) => Ok(value),
            (SpanSpec::Nanoseconds(nanoseconds), true) => {
                Ok(nanoseconds as f64 / self.nanoseconds_per_unit())
            }
            (SpanSpec::Numeric(_), true) => Err(error::span_mismatch(
                operation,
                parameter,
                &self.dtype,
                "numeric",
                "Use datetime.timedelta for temporal clocks.",
            )),
            (SpanSpec::Nanoseconds(_), false) => Err(error::span_mismatch(
                operation,
                parameter,
                &self.dtype,
                "timedelta",
                "Use a number in the units of the clock for numeric clocks.",
            )),
        }
    }

    /// How many nanoseconds one step of a temporal clock covers.
    fn nanoseconds_per_unit(&self) -> f64 {
        match &self.dtype {
            DataType::Date => 86_400_000_000_000.0,
            DataType::Datetime(TimeUnit::Milliseconds, _) => 1_000_000.0,
            DataType::Datetime(TimeUnit::Microseconds, _) => 1_000.0,
            _ => 1.0,
        }
    }

    /// Fail if the clock ever goes backwards.
    fn check_monotone(&self, operation: &str) -> PolarsResult<()> {
        for row in 1..self.len() {
            if self.distance(row - 1, row) < 0.0 {
                return Err(error::clock_decreased(
                    operation,
                    &self.name,
                    &self.render(row - 1),
                    &self.render(row),
                ));
            }
        }
        Ok(())
    }

    /// One clock value, for an error message.
    fn render(&self, row: usize) -> String {
        match &self.values {
            Values::Integral(values) => values[row].to_string(),
            Values::Real(values) => values[row].to_string(),
        }
    }
}

/// Whether a dtype counts in whole steps, so its differences stay exact as integers.
fn is_integral(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Date
            | DataType::Datetime(_, _)
    )
}

/// The first row of `series` that is null, if it has one.
fn first_null(series: &Series) -> Option<usize> {
    if series.null_count() == 0 {
        return None;
    }
    series
        .is_null()
        .iter()
        .position(|is_null| is_null.unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(series: Series) -> PolarsResult<Clock> {
        Clock::from_series(&series, "test")
    }

    #[test]
    fn reads_an_integer_clock() {
        let read = clock(Series::new("c".into(), [10i64, 20, 45])).unwrap();

        assert_eq!(read.len(), 3);
        assert_eq!(read.distance(0, 1), 10.0);
        assert_eq!(read.distance(0, 2), 35.0);
        assert_eq!(read.distance(1, 1), 0.0);
    }

    #[test]
    fn reads_a_float_clock() {
        let read = clock(Series::new("c".into(), [0.5f64, 1.25])).unwrap();

        assert_eq!(read.distance(0, 1), 0.75);
    }

    #[test]
    fn a_large_integer_clock_keeps_its_differences_exact() {
        // Past 2^53 an f64 cannot hold every integer, but the difference is small enough
        // to survive the widening whatever the clock has counted up to.
        let far = 1i64 << 60;
        let read = clock(Series::new("c".into(), [far, far + 7])).unwrap();

        assert_eq!(read.distance(0, 1), 7.0);
    }

    #[test]
    fn refuses_a_clock_that_goes_backwards() {
        assert!(clock(Series::new("c".into(), [10i64, 9])).is_err());
    }

    #[test]
    fn accepts_a_clock_that_stands_still() {
        assert!(clock(Series::new("c".into(), [10i64, 10, 10])).is_ok());
    }

    #[test]
    fn refuses_a_null_clock() {
        assert!(clock(Series::new("c".into(), [Some(1i64), None])).is_err());
    }

    #[test]
    fn refuses_a_clock_that_is_not_finite() {
        assert!(clock(Series::new("c".into(), [1.0f64, f64::NAN])).is_err());
        assert!(clock(Series::new("c".into(), [1.0f64, f64::INFINITY])).is_err());
    }

    #[test]
    fn refuses_a_clock_that_is_not_a_number_or_a_time() {
        assert!(clock(Series::new("c".into(), ["a", "b"])).is_err());
    }

    #[test]
    fn a_numeric_clock_takes_a_number() {
        let read = clock(Series::new("c".into(), [1i64, 2])).unwrap();

        assert_eq!(
            read.resolve(SpanSpec::Numeric(5.0), "test", "window")
                .unwrap(),
            5.0
        );
        assert!(read
            .resolve(SpanSpec::Nanoseconds(5), "test", "window")
            .is_err());
    }

    #[test]
    fn a_temporal_clock_takes_a_duration_in_its_own_unit() {
        let series = Series::new("c".into(), [1i64, 2])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let read = Clock::from_series(&series, "test").unwrap();

        // Half an hour, as microseconds.
        let span = read
            .resolve(
                SpanSpec::Nanoseconds(1_800_000_000_000),
                "test",
                "half_life",
            )
            .unwrap();
        assert_eq!(span, 1_800_000_000.0);
        assert!(read
            .resolve(SpanSpec::Numeric(1.0), "test", "half_life")
            .is_err());
    }

    #[test]
    fn a_date_clock_counts_in_days() {
        let series = Series::new("c".into(), [1i32, 2])
            .cast(&DataType::Date)
            .unwrap();
        let read = Clock::from_series(&series, "test").unwrap();

        let span = read
            .resolve(SpanSpec::Nanoseconds(86_400_000_000_000), "test", "window")
            .unwrap();
        assert_eq!(span, 1.0);
        assert_eq!(read.distance(0, 1), 1.0);
    }
}
