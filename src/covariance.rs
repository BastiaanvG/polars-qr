//! Covariance from centred second moments.

use faer::{Mat, MatRef};
use polars::prelude::*;

use crate::weights::Weights;

/// How a covariance matrix is estimated.
pub struct Options {
    /// The delta degrees of freedom subtracted from the divisor.
    pub ddof: f64,
    /// Whether to divide the matrix through by the standard deviations, turning it into a
    /// correlation matrix.
    pub normalise: bool,
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
    /// The sum of the weights, which is the observation count when there are none.
    pub sum_weights: f64,
}

/// Estimate the covariance of the columns of `x`.
///
/// Weights are treated as reliability weights: they say how precisely each observation was
/// measured rather than how many times it was seen. The divisor is corrected for that, so
/// scaling every weight by the same factor leaves the estimate alone.
pub fn covariance(
    x: MatRef<'_, f64>,
    weights: Option<&Weights>,
    options: &Options,
) -> PolarsResult<Covariance> {
    let (n, p) = (x.nrows(), x.ncols());
    let (sum_weights, sum_squared_weights) = match weights {
        Some(weights) => (
            weights.total(),
            weights.values().iter().map(|w| w * w).sum::<f64>(),
        ),
        None => (n as f64, n as f64),
    };

    let weight_of = |i: usize| weights.map_or(1.0, |weights| weights.values()[i]);
    let means: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| weight_of(i) * x[(i, j)]).sum::<f64>() / sum_weights)
        .collect();
    let centred = Mat::from_fn(n, p, |i, j| x[(i, j)] - means[j]);
    let weighted = Mat::from_fn(n, p, |i, j| weight_of(i) * centred[(i, j)]);
    let cross_products = weighted.transpose() * &centred;

    from_moments(
        &means,
        cross_products.as_ref(),
        n,
        sum_weights,
        sum_squared_weights,
        options,
    )
}

/// Divide accumulated moments through into a covariance matrix.
///
/// The moments can come from one pass over the rows or from a merged summary of several, so
/// this is where the divisor, the symmetry and the normalisation are decided once for both.
pub fn from_moments(
    means: &[f64],
    cross_products: MatRef<'_, f64>,
    n_observations: usize,
    sum_weights: f64,
    sum_squared_weights: f64,
    options: &Options,
) -> PolarsResult<Covariance> {
    let p = means.len();
    let divisor = sum_weights - options.ddof * sum_squared_weights / sum_weights;
    // Written to catch a divisor that is not a number as well as one that is too small: an
    // empty sample makes the correction term 0/0, and NaN passes every ordinary comparison.
    if !(divisor > 0.0) {
        polars_bail!(
            ComputeError:
            "{} observations with ddof={} leaves nothing to divide by",
            n_observations, options.ddof,
        );
    }

    let mut values = Mat::from_fn(p, p, |i, j| cross_products[(i, j)] / divisor);
    // The matrix is symmetric by construction; make it so to the last bit as well, so that
    // a caller can hand the result straight to a factorisation that checks.
    symmetrise(&mut values);

    let standard_deviations: Vec<f64> = (0..p).map(|j| values[(j, j)].max(0.0).sqrt()).collect();
    if options.normalise {
        normalise(&mut values, &standard_deviations);
    }

    Ok(Covariance {
        means: means.to_vec(),
        standard_deviations,
        values,
        n_observations,
        sum_weights,
    })
}

/// Divide a covariance matrix through by its standard deviations.
///
/// A column that does not vary has no scale to divide by, and its correlations with the
/// other columns are undefined rather than zero. Those entries come out as NaN, which keeps
/// the undefined ones visible instead of quietly reading as "uncorrelated".
fn normalise(matrix: &mut Mat<f64>, standard_deviations: &[f64]) {
    let n = matrix.nrows();
    for i in 0..n {
        for j in 0..n {
            let scale = standard_deviations[i] * standard_deviations[j];
            matrix[(i, j)] = if scale > 0.0 {
                if i == j {
                    1.0
                } else {
                    (matrix[(i, j)] / scale).clamp(-1.0, 1.0)
                }
            } else {
                f64::NAN
            };
        }
    }
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

    fn sample() -> Options {
        Options {
            ddof: 1.0,
            normalise: false,
        }
    }

    fn correlation() -> Options {
        Options {
            normalise: true,
            ..sample()
        }
    }

    #[test]
    fn matches_a_variance_computed_by_hand() {
        // Values 1, 2, 3, 4: mean 2.5, squared deviations 2.25 + 0.25 + 0.25 + 2.25 = 5.
        let x = matrix(&[&[1.0], &[2.0], &[3.0], &[4.0]]);

        let over_sample = covariance(x.as_ref(), None, &sample()).unwrap();
        let over_population = covariance(
            x.as_ref(),
            None,
            &Options {
                ddof: 0.0,
                ..sample()
            },
        )
        .unwrap();

        assert!((over_sample.means[0] - 2.5).abs() < 1e-12);
        assert!((over_sample.values[(0, 0)] - 5.0 / 3.0).abs() < 1e-12);
        assert!((over_population.values[(0, 0)] - 1.25).abs() < 1e-12);
        assert!((over_sample.standard_deviations[0] - (5.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(over_sample.n_observations, 4);
    }

    #[test]
    fn finds_the_covariance_between_two_columns() {
        // b is exactly -2 * a, so the covariance is -2 times the variance of a.
        let x = matrix(&[&[1.0, -2.0], &[2.0, -4.0], &[3.0, -6.0]]);

        let result = covariance(x.as_ref(), None, &sample()).unwrap();

        assert!((result.values[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((result.values[(0, 1)] + 2.0).abs() < 1e-12);
        assert!((result.values[(1, 1)] - 4.0).abs() < 1e-12);
        assert_eq!(result.values[(0, 1)], result.values[(1, 0)]);
    }

    #[test]
    fn a_constant_column_has_no_variance() {
        let x = matrix(&[&[1.0, 7.0], &[2.0, 7.0], &[3.0, 7.0]]);

        let result = covariance(x.as_ref(), None, &sample()).unwrap();

        assert!(result.values[(1, 1)].abs() < 1e-24);
        assert_eq!(result.standard_deviations[1], 0.0);
    }

    #[test]
    fn shifting_a_column_leaves_the_covariance_alone() {
        let x = matrix(&[&[1.0, 2.0], &[2.0, 5.0], &[3.0, 1.0]]);
        let shifted = Mat::from_fn(3, 2, |i, j| x[(i, j)] + 1e6);

        let plain = covariance(x.as_ref(), None, &sample()).unwrap();
        let moved = covariance(shifted.as_ref(), None, &sample()).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((plain.values[(i, j)] - moved.values[(i, j)]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn a_correlation_matrix_has_a_unit_diagonal() {
        let x = matrix(&[&[1.0, 2.0], &[2.0, 5.0], &[3.0, 1.0], &[4.0, 8.0]]);

        let result = covariance(x.as_ref(), None, &correlation()).unwrap();

        assert_eq!(result.values[(0, 0)], 1.0);
        assert_eq!(result.values[(1, 1)], 1.0);
        assert!(result.values[(0, 1)].abs() <= 1.0);
        assert_eq!(result.values[(0, 1)], result.values[(1, 0)]);
    }

    #[test]
    fn perfectly_dependent_columns_correlate_exactly() {
        let up = matrix(&[&[1.0, 3.0], &[2.0, 6.0], &[3.0, 9.0]]);
        let down = matrix(&[&[1.0, -3.0], &[2.0, -6.0], &[3.0, -9.0]]);

        assert_eq!(
            covariance(up.as_ref(), None, &correlation())
                .unwrap()
                .values[(0, 1)],
            1.0
        );
        assert_eq!(
            covariance(down.as_ref(), None, &correlation())
                .unwrap()
                .values[(0, 1)],
            -1.0
        );
    }

    #[test]
    fn correlation_keeps_the_unnormalised_standard_deviations() {
        let x = matrix(&[&[1.0, 2.0], &[2.0, 5.0], &[3.0, 1.0]]);

        let plain = covariance(x.as_ref(), None, &sample()).unwrap();
        let normalised = covariance(x.as_ref(), None, &correlation()).unwrap();

        assert_eq!(plain.standard_deviations, normalised.standard_deviations);
        assert_eq!(plain.means, normalised.means);
    }

    #[test]
    fn a_column_that_does_not_vary_correlates_with_nothing() {
        let x = matrix(&[&[1.0, 7.0], &[2.0, 7.0], &[3.0, 7.0]]);

        let result = covariance(x.as_ref(), None, &correlation()).unwrap();

        assert!(result.values[(1, 1)].is_nan());
        assert!(result.values[(0, 1)].is_nan());
        assert_eq!(result.values[(0, 0)], 1.0);
    }

    #[test]
    fn equal_weights_leave_the_estimate_alone() {
        let x = matrix(&[&[1.0, 2.0], &[2.0, 5.0], &[3.0, 1.0], &[4.0, 8.0]]);
        let w = matrix(&[&[3.0], &[3.0], &[3.0], &[3.0]]);
        let weights = Weights::new(w.as_ref(), "w").unwrap();

        let plain = covariance(x.as_ref(), None, &sample()).unwrap();
        let weighted = covariance(x.as_ref(), Some(&weights), &sample()).unwrap();

        for i in 0..2 {
            assert!((plain.means[i] - weighted.means[i]).abs() < 1e-12);
            for j in 0..2 {
                assert!((plain.values[(i, j)] - weighted.values[(i, j)]).abs() < 1e-12);
            }
        }
        assert_eq!(weighted.sum_weights, 12.0);
    }

    #[test]
    fn a_zero_weight_leaves_an_observation_out() {
        let x = matrix(&[&[1.0, 2.0], &[2.0, 5.0], &[3.0, 1.0], &[99.0, -99.0]]);
        let kept = matrix(&[&[1.0, 2.0], &[2.0, 5.0], &[3.0, 1.0]]);
        let w = matrix(&[&[1.0], &[1.0], &[1.0], &[0.0]]);
        let weights = Weights::new(w.as_ref(), "w").unwrap();

        let weighted = covariance(x.as_ref(), Some(&weights), &sample()).unwrap();
        let without = covariance(kept.as_ref(), None, &sample()).unwrap();

        for i in 0..2 {
            assert!((weighted.means[i] - without.means[i]).abs() < 1e-12);
            for j in 0..2 {
                assert!((weighted.values[(i, j)] - without.values[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn scaling_every_weight_leaves_the_estimate_alone() {
        let x = matrix(&[&[1.0, 2.0], &[2.0, 5.0], &[3.0, 1.0], &[4.0, 8.0]]);
        let small = matrix(&[&[0.5], &[1.0], &[2.0], &[1.5]]);
        let large = Mat::from_fn(4, 1, |i, _| 1000.0 * small[(i, 0)]);

        let one = covariance(
            x.as_ref(),
            Some(&Weights::new(small.as_ref(), "w").unwrap()),
            &sample(),
        )
        .unwrap();
        let other = covariance(
            x.as_ref(),
            Some(&Weights::new(large.as_ref(), "w").unwrap()),
            &sample(),
        )
        .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((one.values[(i, j)] - other.values[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn rejects_a_sample_too_small_for_the_degrees_of_freedom() {
        let x = matrix(&[&[1.0]]);

        assert!(covariance(x.as_ref(), None, &sample()).is_err());
    }
}
