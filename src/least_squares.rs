//! Least-squares fitting.
//!
//! One implementation covers ordinary fitting, projection and calibration: they differ in
//! what the caller does with the coefficients, not in how the system is solved.

use faer::linalg::solvers::SolveLstsq;
use faer::{Mat, MatRef};
use polars::prelude::*;

use crate::weights::Weights;

/// How a fit is set up.
pub struct Options {
    /// Whether to fit a constant term alongside the features.
    pub intercept: bool,
}

/// The outcome of a least-squares solve.
pub struct LeastSquaresFit {
    /// One row per feature, one column per target.
    pub coefficients: Mat<f64>,
    /// The constant term, one entry per target, when one was fitted.
    pub intercept: Option<Vec<f64>>,
    /// The number of observations the fit used.
    pub n_observations: usize,
    /// The squared norm of the residual, one entry per target.
    pub residual_sum_of_squares: Vec<f64>,
}

/// Fit `targets` against `features` in the least-squares sense.
///
/// The design matrix is assembled first and weighted second, so a fitted constant term is
/// weighted like every other column.
pub fn fit(
    features: MatRef<'_, f64>,
    targets: MatRef<'_, f64>,
    weights: Option<&Weights>,
    options: &Options,
) -> PolarsResult<LeastSquaresFit> {
    let (n, p) = (features.nrows(), features.ncols());
    if targets.nrows() != n {
        polars_bail!(
            ShapeMismatch:
            "the targets have {} rows and the features have {}", targets.nrows(), n,
        );
    }

    let design = design_matrix(features, options.intercept);
    let columns = design.ncols();
    if n < columns {
        polars_bail!(
            ComputeError:
            "a QR solve needs at least as many observations as columns, but got {n} \
             observations for {columns} columns",
        );
    }

    let (design, scaled_targets) = match weights {
        Some(weights) => (
            weights.scale_rows(design.as_ref()),
            weights.scale_rows(targets),
        ),
        None => (design, targets.to_owned()),
    };

    let solution = design.qr().solve_lstsq(&scaled_targets);
    let residual_sum_of_squares =
        residual_sum_of_squares(design.as_ref(), scaled_targets.as_ref(), solution.as_ref());

    let (intercept, coefficients) = split_intercept(solution.as_ref(), options.intercept);
    debug_assert_eq!(coefficients.nrows(), p);

    Ok(LeastSquaresFit {
        coefficients,
        intercept,
        n_observations: n,
        residual_sum_of_squares,
    })
}

/// The features, with a leading column of ones when a constant term is wanted.
fn design_matrix(features: MatRef<'_, f64>, intercept: bool) -> Mat<f64> {
    if !intercept {
        return features.to_owned();
    }
    Mat::from_fn(features.nrows(), features.ncols() + 1, |i, j| {
        if j == 0 {
            1.0
        } else {
            features[(i, j - 1)]
        }
    })
}

/// Split the constant term off the top of the solution.
fn split_intercept(solution: MatRef<'_, f64>, intercept: bool) -> (Option<Vec<f64>>, Mat<f64>) {
    if !intercept {
        return (None, solution.to_owned());
    }
    let constant = (0..solution.ncols()).map(|j| solution[(0, j)]).collect();
    let coefficients = solution.subrows(1, solution.nrows() - 1).to_owned();
    (Some(constant), coefficients)
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

    fn plain() -> Options {
        Options { intercept: false }
    }

    #[test]
    fn recovers_an_exact_fit() {
        // y = 2 * a - b, with no noise to fit through.
        let x = matrix(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[2.0, 1.0]]);
        let y = matrix(&[&[2.0], &[-1.0], &[1.0], &[3.0]]);

        let fit = fit(x.as_ref(), y.as_ref(), None, &plain()).unwrap();

        assert!((fit.coefficients[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((fit.coefficients[(1, 0)] + 1.0).abs() < 1e-12);
        assert!(fit.residual_sum_of_squares[0] < 1e-24);
        assert_eq!(fit.n_observations, 4);
        assert!(fit.intercept.is_none());
    }

    #[test]
    fn splits_the_difference_when_no_line_fits() {
        // Two observations of the same feature value disagree, so the fit averages them.
        let x = matrix(&[&[1.0], &[1.0]]);
        let y = matrix(&[&[1.0], &[3.0]]);

        let fit = fit(x.as_ref(), y.as_ref(), None, &plain()).unwrap();

        assert!((fit.coefficients[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((fit.residual_sum_of_squares[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn recovers_a_constant_term() {
        // y = 3 + 2 * a.
        let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
        let y = matrix(&[&[3.0], &[5.0], &[7.0], &[9.0]]);

        let fit = fit(
            x.as_ref(),
            y.as_ref(),
            None,
            &Options { intercept: true },
        )
        .unwrap();

        let intercept = fit.intercept.unwrap();
        assert!((intercept[0] - 3.0).abs() < 1e-12);
        assert!((fit.coefficients[(0, 0)] - 2.0).abs() < 1e-12);
        assert_eq!(fit.coefficients.nrows(), 1);
    }

    #[test]
    fn a_constant_term_is_weighted_like_any_other_column() {
        // The zero-weighted row disagrees with the line through the other three.
        let x = matrix(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
        let y = matrix(&[&[3.0], &[5.0], &[7.0], &[100.0]]);
        let w = matrix(&[&[1.0], &[1.0], &[1.0], &[0.0]]);
        let weights = Weights::new(w.as_ref(), "w").unwrap();

        let fit = fit(
            x.as_ref(),
            y.as_ref(),
            Some(&weights),
            &Options { intercept: true },
        )
        .unwrap();

        let intercept = fit.intercept.unwrap();
        assert!((intercept[0] - 3.0).abs() < 1e-12);
        assert!((fit.coefficients[(0, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_a_system_with_fewer_observations_than_columns() {
        let x = matrix(&[&[1.0, 2.0]]);
        let y = matrix(&[&[1.0]]);

        assert!(fit(x.as_ref(), y.as_ref(), None, &plain()).is_err());
    }

    #[test]
    fn counts_the_constant_term_as_a_column() {
        let x = matrix(&[&[1.0], &[2.0]]);
        let y = matrix(&[&[1.0], &[2.0]]);

        assert!(fit(x.as_ref(), y.as_ref(), None, &plain()).is_ok());
        assert!(fit(x.as_ref(), y.as_ref(), None, &Options { intercept: true }).is_ok());

        let one_row = matrix(&[&[1.0]]);
        let one_target = matrix(&[&[1.0]]);
        assert!(fit(one_row.as_ref(), one_target.as_ref(), None, &plain()).is_ok());
        assert!(fit(
            one_row.as_ref(),
            one_target.as_ref(),
            None,
            &Options { intercept: true }
        )
        .is_err());
    }
}
