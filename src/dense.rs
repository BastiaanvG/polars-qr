//! Turning a set of Polars columns into a dense matrix.
//!
//! Every operation in this crate reads the same shape of input: a handful of numeric columns
//! that together form one dense matrix, one row per observation. Collecting that matrix is
//! the only place where Polars data is copied, so the conversion is kept in one module.

use faer::Mat;
use polars::prelude::*;
use serde::Deserialize;

/// What to do with a row that cannot be used.
///
/// A row is unusable when any of the columns it spans is null, or holds a value that is not
/// finite. Both cases are treated the same way: a missing observation and an infinite one
/// are equally unusable as input to a factorisation.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NullPolicy {
    /// Fail when any row cannot be used.
    #[default]
    Raise,
    /// Drop the rows that cannot be used, keeping the rest.
    Drop,
}

/// A dense matrix read out of a wide frame, one column per input series.
pub struct DenseFrame {
    values: Mat<f64>,
    valid: Vec<bool>,
    height: usize,
}

impl DenseFrame {
    /// Read `inputs` into a single dense matrix, applying `policy` row by row.
    ///
    /// The columns are taken in the order they are given; that order is what every result
    /// labels itself with. Rows are dropped jointly: a row is either used by every column or
    /// by none of them, which keeps the columns of the matrix estimated from one sample.
    pub fn from_series(inputs: &[Series], policy: NullPolicy) -> PolarsResult<Self> {
        let n_cols = inputs.len();
        if n_cols == 0 {
            polars_bail!(InvalidOperation: "at least one column is required");
        }

        let height = inputs[0].len();
        let mut columns = Vec::with_capacity(n_cols);
        for series in inputs {
            if series.len() != height {
                polars_bail!(
                    ShapeMismatch:
                    "column '{}' has {} rows, but '{}' has {}",
                    series.name(), series.len(), inputs[0].name(), height,
                );
            }
            check_numeric(series)?;
            columns.push(series.cast(&DataType::Float64)?);
        }

        let mut valid = vec![true; height];
        for (series, column) in inputs.iter().zip(&columns) {
            let column = column.f64()?;
            let mut unusable = 0usize;
            for (i, value) in column.iter().enumerate() {
                let usable = value.is_some_and(f64::is_finite);
                if !usable {
                    unusable += 1;
                    valid[i] = false;
                }
            }
            if unusable > 0 && policy == NullPolicy::Raise {
                polars_bail!(
                    ComputeError:
                    "column '{}' has {} null or non-finite values; pass null_policy='drop' \
                     to drop those rows",
                    series.name(), unusable,
                );
            }
        }

        let n_rows = valid.iter().filter(|kept| **kept).count();
        let mut values = Mat::<f64>::zeros(n_rows, n_cols);
        for (j, column) in columns.iter().enumerate() {
            let column = column.f64()?;
            let mut row = 0usize;
            for (i, value) in column.iter().enumerate() {
                if valid[i] {
                    values[(row, j)] = value.unwrap_or(f64::NAN);
                    row += 1;
                }
            }
        }

        Ok(Self {
            values,
            valid,
            height,
        })
    }

    /// The dense matrix, with one row per usable observation.
    pub fn matrix(&self) -> faer::MatRef<'_, f64> {
        self.values.as_ref()
    }

    /// The number of usable observations in the matrix.
    pub fn n_rows(&self) -> usize {
        self.values.nrows()
    }

    /// The number of columns in the matrix.
    pub fn n_cols(&self) -> usize {
        self.values.ncols()
    }

    /// The number of rows that were read, before any row was dropped.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Which of the rows that were read ended up in the matrix.
    ///
    /// Row-preserving operations use this to scatter their results back over the rows they
    /// were given, leaving a null where a row was dropped.
    pub fn valid(&self) -> &[bool] {
        &self.valid
    }
}

/// Reject the dtypes that cannot be read as a real number.
///
/// Strings are the reason this is an explicit check rather than a cast: casting a string
/// column to `Float64` parses it, which would quietly accept a column that is not numeric.
fn check_numeric(series: &Series) -> PolarsResult<()> {
    match series.dtype() {
        DataType::Float32
        | DataType::Float64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Ok(()),
        dtype => polars_bail!(
            InvalidOperation:
            "column '{}' has dtype {}, which is not numeric", series.name(), dtype,
        ),
    }
}

/// Collect the names of the input columns, in order.
pub fn column_names(inputs: &[Series]) -> Vec<String> {
    inputs.iter().map(|s| s.name().to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(name: &str, values: &[f64]) -> Series {
        Series::new(name.into(), values)
    }

    #[test]
    fn reads_columns_in_the_order_they_are_given() {
        let inputs = [series("a", &[1.0, 2.0]), series("b", &[3.0, 4.0])];
        let dense = DenseFrame::from_series(&inputs, NullPolicy::Raise).unwrap();

        assert_eq!(dense.n_rows(), 2);
        assert_eq!(dense.n_cols(), 2);
        assert_eq!(dense.matrix()[(0, 0)], 1.0);
        assert_eq!(dense.matrix()[(1, 0)], 2.0);
        assert_eq!(dense.matrix()[(0, 1)], 3.0);
        assert_eq!(column_names(&inputs), ["a", "b"]);
    }

    #[test]
    fn widens_integers_to_double_precision() {
        let inputs = [Series::new("a".into(), [1i32, 2, 3])];
        let dense = DenseFrame::from_series(&inputs, NullPolicy::Raise).unwrap();

        assert_eq!(dense.matrix()[(2, 0)], 3.0);
    }

    #[test]
    fn rejects_columns_of_different_lengths() {
        let inputs = [series("a", &[1.0, 2.0]), series("b", &[3.0])];

        assert!(DenseFrame::from_series(&inputs, NullPolicy::Raise).is_err());
    }

    #[test]
    fn rejects_columns_that_are_not_numeric() {
        let inputs = [Series::new("a".into(), ["1.0", "2.0"])];

        assert!(DenseFrame::from_series(&inputs, NullPolicy::Raise).is_err());
    }

    #[test]
    fn rejects_an_empty_input() {
        assert!(DenseFrame::from_series(&[], NullPolicy::Raise).is_err());
    }

    #[test]
    fn raises_on_nulls_and_on_values_that_are_not_finite() {
        let nulls = [Series::new("a".into(), [Some(1.0), None])];
        let infinite = [series("a", &[1.0, f64::INFINITY])];
        let missing = [series("a", &[1.0, f64::NAN])];

        assert!(DenseFrame::from_series(&nulls, NullPolicy::Raise).is_err());
        assert!(DenseFrame::from_series(&infinite, NullPolicy::Raise).is_err());
        assert!(DenseFrame::from_series(&missing, NullPolicy::Raise).is_err());
    }

    #[test]
    fn drops_unusable_rows_across_every_column() {
        let inputs = [
            Series::new("a".into(), [Some(1.0), None, Some(3.0), Some(4.0)]),
            Series::new("b".into(), [Some(5.0), Some(6.0), Some(f64::NAN), Some(8.0)]),
        ];
        let dense = DenseFrame::from_series(&inputs, NullPolicy::Drop).unwrap();

        assert_eq!(dense.height(), 4);
        assert_eq!(dense.n_rows(), 2);
        assert_eq!(dense.valid(), [true, false, false, true]);
        assert_eq!(dense.matrix()[(0, 0)], 1.0);
        assert_eq!(dense.matrix()[(1, 0)], 4.0);
        assert_eq!(dense.matrix()[(1, 1)], 8.0);
    }
}
