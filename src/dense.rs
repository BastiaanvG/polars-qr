//! Turning a set of Polars columns into a dense matrix.
//!
//! Every operation in this crate reads the same shape of input: a handful of numeric columns
//! that together form one dense matrix, one row per observation. Collecting that matrix is
//! the only place where Polars data is copied, so the conversion is kept in one module.

use faer::Mat;
use polars::prelude::*;

/// A dense matrix read out of a wide frame, one column per input series.
pub struct DenseFrame {
    values: Mat<f64>,
    height: usize,
}

impl DenseFrame {
    /// Read `inputs` into a single dense matrix.
    ///
    /// The columns are taken in the order they are given; that order is what every result
    /// labels itself with.
    pub fn from_series(inputs: &[Series]) -> PolarsResult<Self> {
        let n_cols = inputs.len();
        if n_cols == 0 {
            polars_bail!(InvalidOperation: "at least one column is required");
        }

        let height = inputs[0].len();
        for series in inputs {
            if series.len() != height {
                polars_bail!(
                    ShapeMismatch:
                    "column '{}' has {} rows, but '{}' has {}",
                    series.name(), series.len(), inputs[0].name(), height,
                );
            }
            check_numeric(series)?;
        }

        let mut values = Mat::<f64>::zeros(height, n_cols);
        for (j, series) in inputs.iter().enumerate() {
            let column = series.cast(&DataType::Float64)?;
            let column = column.f64()?;
            if column.null_count() > 0 {
                polars_bail!(
                    ComputeError:
                    "column '{}' contains nulls", series.name(),
                );
            }
            for (i, value) in column.into_no_null_iter().enumerate() {
                values[(i, j)] = value;
            }
        }

        Ok(Self { values, height })
    }

    /// The dense matrix, with one row per observation.
    pub fn matrix(&self) -> faer::MatRef<'_, f64> {
        self.values.as_ref()
    }

    /// The number of observations in the matrix.
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
        let dense = DenseFrame::from_series(&inputs).unwrap();

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
        let dense = DenseFrame::from_series(&inputs).unwrap();

        assert_eq!(dense.matrix()[(2, 0)], 3.0);
    }

    #[test]
    fn rejects_columns_of_different_lengths() {
        let inputs = [series("a", &[1.0, 2.0]), series("b", &[3.0])];

        assert!(DenseFrame::from_series(&inputs).is_err());
    }

    #[test]
    fn rejects_columns_that_are_not_numeric() {
        let inputs = [Series::new("a".into(), ["1.0", "2.0"])];

        assert!(DenseFrame::from_series(&inputs).is_err());
    }

    #[test]
    fn rejects_nulls() {
        let inputs = [Series::new("a".into(), [Some(1.0), None])];

        assert!(DenseFrame::from_series(&inputs).is_err());
    }

    #[test]
    fn rejects_an_empty_input() {
        assert!(DenseFrame::from_series(&[]).is_err());
    }
}
