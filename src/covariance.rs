//! Covariance from centred second moments.

use faer::{Mat, MatRef};
use polars::prelude::*;

/// How a covariance matrix is estimated.
pub struct Options {
    /// The delta degrees of freedom subtracted from the divisor.
    pub ddof: f64,
}

/// A covariance matrix and the moments it was built from.
pub struct Covariance {
    /// The column means.
    pub means: Vec<f64>,
    /// The square roots of the diagonal of the covariance matrix.
    pub standard_deviations: Vec<f64>,
    /// The covariance matrix itself.
    pub values: Mat<f64>,
    /// The number of observations that went into it.
    pub n_observations: usize,
}

/// Estimate the covariance of the columns of `x`.
pub fn covariance(x: MatRef<'_, f64>, options: &Options) -> PolarsResult<Covariance> {
    let (n, p) = (x.nrows(), x.ncols());
    let divisor = n as f64 - options.ddof;
    if divisor <= 0.0 {
        polars_bail!(
            ComputeError:
            "{} observations with ddof={} leaves nothing to divide by", n, options.ddof,
        );
    }

    let means: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64)
        .collect();
    let centred = Mat::from_fn(n, p, |i, j| x[(i, j)] - means[j]);

    let mut values = centred.transpose() * &centred;
    for value in values.col_iter_mut().flat_map(|column| column.iter_mut()) {
        *value /= divisor;
    }
    // The matrix is symmetric by construction; make it so to the last bit as well, so that
    // a caller can hand the result straight to a factorisation that checks.
    symmetrise(&mut values);

    let standard_deviations = (0..p).map(|j| values[(j, j)].max(0.0).sqrt()).collect();

    Ok(Covariance {
        means,
        standard_deviations,
        values,
        n_observations: n,
    })
}

/// Average the two triangles of a matrix that should be symmetric.
fn symmetrise(matrix: &mut Mat<f64>) {
    let n = matrix.nrows();
    for i in 0..n {
        for j in 0..i {
            let averaged = 0.5 * (matrix[(i, j)] + matrix[(j, i)]);
            matrix[(i, j)] = averaged;
            matrix[(j, i)] = averaged;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[&[f64]]) -> Mat<f64> {
        Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
    }

    #[test]
    fn matches_a_variance_computed_by_hand() {
        // Values 1, 2, 3, 4: mean 2.5, squared deviations 2.25 + 0.25 + 0.25 + 2.25 = 5.
        let x = matrix(&[&[1.0], &[2.0], &[3.0], &[4.0]]);

        let sample = covariance(x.as_ref(), &Options { ddof: 1.0 }).unwrap();
        let population = covariance(x.as_ref(), &Options { ddof: 0.0 }).unwrap();

        assert!((sample.means[0] - 2.5).abs() < 1e-12);
        assert!((sample.values[(0, 0)] - 5.0 / 3.0).abs() < 1e-12);
        assert!((population.values[(0, 0)] - 1.25).abs() < 1e-12);
        assert!((sample.standard_deviations[0] - (5.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(sample.n_observations, 4);
    }

    #[test]
    fn finds_the_covariance_between_two_columns() {
        // b is exactly -2 * a, so the covariance is -2 times the variance of a.
        let x = matrix(&[&[1.0, -2.0], &[2.0, -4.0], &[3.0, -6.0]]);

        let result = covariance(x.as_ref(), &Options { ddof: 1.0 }).unwrap();

        assert!((result.values[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((result.values[(0, 1)] + 2.0).abs() < 1e-12);
        assert!((result.values[(1, 1)] - 4.0).abs() < 1e-12);
        assert_eq!(result.values[(0, 1)], result.values[(1, 0)]);
    }

    #[test]
    fn a_constant_column_has_no_variance() {
        let x = matrix(&[&[1.0, 7.0], &[2.0, 7.0], &[3.0, 7.0]]);

        let result = covariance(x.as_ref(), &Options { ddof: 1.0 }).unwrap();

        assert!(result.values[(1, 1)].abs() < 1e-24);
        assert_eq!(result.standard_deviations[1], 0.0);
    }

    #[test]
    fn shifting_a_column_leaves_the_covariance_alone() {
        let x = matrix(&[&[1.0, 2.0], &[2.0, 5.0], &[3.0, 1.0]]);
        let shifted = Mat::from_fn(3, 2, |i, j| x[(i, j)] + 1e6);

        let plain = covariance(x.as_ref(), &Options { ddof: 1.0 }).unwrap();
        let moved = covariance(shifted.as_ref(), &Options { ddof: 1.0 }).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((plain.values[(i, j)] - moved.values[(i, j)]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn rejects_a_sample_too_small_for_the_degrees_of_freedom() {
        let x = matrix(&[&[1.0]]);

        assert!(covariance(x.as_ref(), &Options { ddof: 1.0 }).is_err());
    }
}
