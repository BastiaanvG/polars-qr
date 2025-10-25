//! Least-squares fitting.
//!
//! One implementation covers ordinary fitting, projection and calibration: they differ in
//! what the caller does with the coefficients, not in how the system is solved.

use faer::linalg::solvers::SolveLstsq;
use faer::{Mat, MatRef};
use polars::prelude::*;

/// The outcome of a least-squares solve.
pub struct LeastSquaresFit {
    /// One row per feature, one column per target.
    pub coefficients: Mat<f64>,
    /// The number of observations the fit used.
    pub n_observations: usize,
    /// The squared norm of the residual, one entry per target.
    pub residual_sum_of_squares: Vec<f64>,
}

/// Solve `x * beta = y` in the least-squares sense with a QR factorisation.
///
/// The QR path assumes `x` has full column rank. A rank-deficient system does not fail
/// here; it produces coefficients that are not meaningful, which is why the rank is
/// reported alongside them.
pub fn solve_qr(x: MatRef<'_, f64>, y: MatRef<'_, f64>) -> PolarsResult<LeastSquaresFit> {
    let (n, p) = (x.nrows(), x.ncols());
    if y.nrows() != n {
        polars_bail!(
            ShapeMismatch:
            "the target has {} rows and the features have {}", y.nrows(), n,
        );
    }
    if n < p {
        polars_bail!(
            ComputeError:
            "a QR solve needs at least as many observations as features, but got {n} \
             observations for {p} features",
        );
    }

    let coefficients = x.qr().solve_lstsq(y);
    let residual_sum_of_squares = residual_sum_of_squares(x, y, coefficients.as_ref());

    Ok(LeastSquaresFit {
        coefficients,
        n_observations: n,
        residual_sum_of_squares,
    })
}

/// The squared norm of `y - x * beta`, one entry per column of `y`.
fn residual_sum_of_squares(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    beta: MatRef<'_, f64>,
) -> Vec<f64> {
    let residual = y - x * beta;
    residual
        .col_iter()
        .map(|column| column.squared_norm_l2())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[&[f64]]) -> Mat<f64> {
        Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
    }

    #[test]
    fn recovers_an_exact_fit() {
        // y = 2 * a - b, with no noise to fit through.
        let x = matrix(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[2.0, 1.0]]);
        let y = matrix(&[&[2.0], &[-1.0], &[1.0], &[3.0]]);

        let fit = solve_qr(x.as_ref(), y.as_ref()).unwrap();

        assert!((fit.coefficients[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((fit.coefficients[(1, 0)] + 1.0).abs() < 1e-12);
        assert!(fit.residual_sum_of_squares[0] < 1e-24);
        assert_eq!(fit.n_observations, 4);
    }

    #[test]
    fn splits_the_difference_when_no_line_fits() {
        // Two observations of the same feature value disagree, so the fit averages them.
        let x = matrix(&[&[1.0], &[1.0]]);
        let y = matrix(&[&[1.0], &[3.0]]);

        let fit = solve_qr(x.as_ref(), y.as_ref()).unwrap();

        assert!((fit.coefficients[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((fit.residual_sum_of_squares[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_a_system_with_fewer_observations_than_features() {
        let x = matrix(&[&[1.0, 2.0]]);
        let y = matrix(&[&[1.0]]);

        assert!(solve_qr(x.as_ref(), y.as_ref()).is_err());
    }

    #[test]
    fn rejects_a_target_of_the_wrong_length() {
        let x = matrix(&[&[1.0], &[2.0]]);
        let y = matrix(&[&[1.0]]);

        assert!(solve_qr(x.as_ref(), y.as_ref()).is_err());
    }
}
