//! Observation weights.
//!
//! Weights are validated the same way wherever they are accepted: they must be finite and
//! non-negative, and they are read under the same row policy as the values they weight. A
//! weight of zero is legal and leaves the observation out of the result without dropping it
//! from the sample.

use faer::{Mat, MatRef};
use polars::prelude::*;

/// A validated column of observation weights.
pub struct Weights {
    values: Vec<f64>,
    total: f64,
}

impl Weights {
    /// Validate a column of weights read from a dense frame.
    pub fn new(column: MatRef<'_, f64>, name: &str) -> PolarsResult<Self> {
        let values: Vec<f64> = (0..column.nrows()).map(|i| column[(i, 0)]).collect();
        if let Some(negative) = values.iter().find(|value| **value < 0.0) {
            polars_bail!(
                ComputeError:
                "weight column '{}' has a negative weight ({})", name, negative,
            );
        }
        let total: f64 = values.iter().sum();
        if total <= 0.0 {
            polars_bail!(
                ComputeError:
                "weight column '{}' sums to {}, so no observation carries any weight",
                name, total,
            );
        }
        Ok(Self { values, total })
    }

    /// The sum of the weights.
    pub fn total(&self) -> f64 {
        self.total
    }

    /// The weights themselves, one per observation.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Scale every row of `matrix` by the square root of its weight.
    ///
    /// Solving the scaled system in the ordinary least-squares sense minimises the weighted
    /// sum of squared residuals of the original one.
    pub fn scale_rows(&self, matrix: MatRef<'_, f64>) -> Mat<f64> {
        let roots: Vec<f64> = self.values.iter().map(|weight| weight.sqrt()).collect();
        Mat::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
            roots[i] * matrix[(i, j)]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(values: &[f64]) -> Mat<f64> {
        Mat::from_fn(values.len(), 1, |i, _| values[i])
    }

    #[test]
    fn sums_the_weights() {
        let weights = Weights::new(column(&[1.0, 2.0, 3.0]).as_ref(), "w").unwrap();

        assert_eq!(weights.total(), 6.0);
    }

    #[test]
    fn accepts_a_weight_of_zero() {
        let weights = Weights::new(column(&[0.0, 2.0]).as_ref(), "w").unwrap();

        assert_eq!(weights.total(), 2.0);
    }

    #[test]
    fn rejects_a_negative_weight() {
        assert!(Weights::new(column(&[1.0, -0.5]).as_ref(), "w").is_err());
    }

    #[test]
    fn rejects_weights_that_are_all_zero() {
        assert!(Weights::new(column(&[0.0, 0.0]).as_ref(), "w").is_err());
    }

    #[test]
    fn scales_rows_by_the_square_root_of_the_weight() {
        let weights = Weights::new(column(&[4.0, 9.0]).as_ref(), "w").unwrap();
        let matrix = Mat::from_fn(2, 2, |i, j| (2 * i + j) as f64 + 1.0);

        let scaled = weights.scale_rows(matrix.as_ref());

        assert_eq!(scaled[(0, 0)], 2.0);
        assert_eq!(scaled[(0, 1)], 4.0);
        assert_eq!(scaled[(1, 0)], 9.0);
        assert_eq!(scaled[(1, 1)], 12.0);
    }
}
