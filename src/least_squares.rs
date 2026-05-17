//! Least-squares fitting.
//!
//! One implementation covers ordinary fitting, projection and calibration: they differ in
//! what the caller does with the coefficients, not in how the system is solved.

use faer::linalg::solvers::SolveLstsq;
use faer::{Mat, MatRef};
use polars::prelude::*;
use serde::Deserialize;

use crate::weights::Weights;

/// Which factorisation solves the system.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Solver {
    /// A QR factorisation: the fastest route for a system of full column rank.
    #[default]
    Qr,
    /// A thin SVD: slower, but it also solves rank-deficient and underdetermined systems,
    /// where it returns the solution of smallest norm.
    Svd,
}

impl Solver {
    /// The name reported alongside a fit.
    pub fn name(self) -> &'static str {
        match self {
            Self::Qr => "qr",
            Self::Svd => "svd",
        }
    }
}

/// How a fit is set up.
pub struct Options {
    /// Whether to fit a constant term alongside the features.
    pub intercept: bool,
    /// Which factorisation to solve with.
    pub solver: Solver,
    /// The ridge penalty on the coefficients. A constant term is never penalised.
    pub l2_penalty: f64,
}

/// The outcome of a least-squares solve.
pub struct LeastSquaresFit {
    /// One row per feature, one column per target.
    pub coefficients: Mat<f64>,
    /// The constant term, one entry per target, when one was fitted.
    pub intercept: Option<Vec<f64>>,
    /// The number of observations the fit used.
    pub n_observations: usize,
    /// The numerical rank of the design matrix.
    pub rank: usize,
    /// The squared norm of the residual, one entry per target.
    pub residual_sum_of_squares: Vec<f64>,
    /// The singular values of the design matrix, when the solver computed them.
    pub singular_values: Option<Vec<f64>>,
    /// An estimate of the condition number of the design matrix.
    pub condition: f64,
    /// Which factorisation produced the fit.
    pub solver: Solver,
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
    if options.solver == Solver::Qr && n < columns {
        polars_bail!(
            ComputeError:
            "a QR solve needs at least as many observations as columns, but got {n} \
             observations for {columns} columns; solver='svd' solves an underdetermined \
             system",
        );
    }

    let (design, scaled_targets) = match weights {
        Some(weights) => (
            weights.scale_rows(design.as_ref()),
            weights.scale_rows(targets),
        ),
        None => (design, targets.to_owned()),
    };

    let mut fit = fit_design(design.as_ref(), scaled_targets.as_ref(), options)?;
    debug_assert_eq!(fit.coefficients.nrows(), p);
    fit.n_observations = n;
    Ok(fit)
}

/// Fit a design matrix that has already been assembled and weighted.
///
/// The constant term, when there is one, is the first column of `design` rather than
/// something added here. A summary of a fit is a design matrix too — a much shorter one —
/// so this is the entry point a mergeable state finalises through.
pub fn fit_design(
    design: MatRef<'_, f64>,
    targets: MatRef<'_, f64>,
    options: &Options,
) -> PolarsResult<LeastSquaresFit> {
    let (n, columns) = (design.nrows(), design.ncols());
    if targets.nrows() != n {
        polars_bail!(
            ShapeMismatch:
            "the targets have {} rows and the design has {}", targets.nrows(), n,
        );
    }

    let (penalised, penalised_targets) =
        penalise(design, targets, options.l2_penalty, options.intercept);

    let (solution, diagnostics) = match options.solver {
        Solver::Qr => {
            let qr = penalised.qr();
            (qr.solve_lstsq(&penalised_targets), from_qr(qr.thin_R()))
        }
        Solver::Svd => solve_svd(penalised.as_ref(), penalised_targets.as_ref())?,
    };
    // The residual is reported for the system that was asked about, not for the padded one
    // the penalty is expressed through.
    let residual_sum_of_squares = residual_sum_of_squares(design, targets, solution.as_ref());

    let (intercept, coefficients) = split_intercept(solution.as_ref(), options.intercept);
    debug_assert_eq!(
        coefficients.nrows(),
        columns - usize::from(options.intercept)
    );

    Ok(LeastSquaresFit {
        coefficients,
        intercept,
        n_observations: n,
        rank: diagnostics.rank,
        residual_sum_of_squares,
        singular_values: diagnostics.singular_values,
        condition: diagnostics.condition,
        solver: options.solver,
    })
}

/// Express a ridge penalty as extra rows on the system.
///
/// Appending `sqrt(lambda) * I` under the design and zeros under the targets makes the
/// ordinary least-squares solution of the padded system the ridge solution of the original
/// one, which keeps both solvers unchanged. The row belonging to a constant term is left
/// out, so the penalty never pulls the intercept towards zero.
fn penalise(
    design: MatRef<'_, f64>,
    targets: MatRef<'_, f64>,
    l2_penalty: f64,
    intercept: bool,
) -> (Mat<f64>, Mat<f64>) {
    if l2_penalty <= 0.0 {
        return (design.to_owned(), targets.to_owned());
    }

    let (n, columns) = (design.nrows(), design.ncols());
    let first_penalised = usize::from(intercept);
    let extra = columns - first_penalised;
    let root = l2_penalty.sqrt();

    let padded = Mat::from_fn(n + extra, columns, |i, j| {
        if i < n {
            design[(i, j)]
        } else if j == i - n + first_penalised {
            root
        } else {
            0.0
        }
    });
    let padded_targets = Mat::from_fn(n + extra, targets.ncols(), |i, j| {
        if i < n {
            targets[(i, j)]
        } else {
            0.0
        }
    });
    (padded, padded_targets)
}

/// What a factorisation says about the conditioning of the design matrix.
struct Diagnostics {
    rank: usize,
    singular_values: Option<Vec<f64>>,
    condition: f64,
}

/// Read the rank and conditioning off the diagonal of a QR factor.
///
/// The ratio of the largest to the smallest diagonal entry of `R` is only an estimate of
/// the condition number, but it costs nothing on top of a factorisation that has already
/// been computed.
fn from_qr(r: MatRef<'_, f64>) -> Diagnostics {
    let diagonal: Vec<f64> = (0..r.ncols()).map(|i| r[(i, i)].abs()).collect();
    let largest = diagonal.iter().copied().fold(0.0, f64::max);
    let smallest = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
    let threshold = singular_value_threshold(r.nrows(), r.ncols(), largest);
    Diagnostics {
        rank: diagonal.iter().filter(|value| **value > threshold).count(),
        singular_values: None,
        condition: if smallest > 0.0 {
            largest / smallest
        } else {
            f64::INFINITY
        },
    }
}

/// Solve through a thin SVD, discarding the directions that carry no signal.
///
/// Singular values below the threshold are treated as zero rather than inverted, which is
/// what makes this the solution of smallest norm when the system does not pin one down.
fn solve_svd(
    design: MatRef<'_, f64>,
    targets: MatRef<'_, f64>,
) -> PolarsResult<(Mat<f64>, Diagnostics)> {
    let svd = design
        .thin_svd()
        .map_err(|error| polars_err!(ComputeError: "the SVD did not converge: {:?}", error))?;
    let singular = svd.S().column_vector();
    let values: Vec<f64> = (0..singular.nrows()).map(|i| singular[i]).collect();
    let threshold = singular_value_threshold(design.nrows(), design.ncols(), values[0]);

    // beta = V * diag(1 / s) * U^T * y, over the directions above the threshold only.
    let projected = svd.U().transpose() * targets;
    let scaled = Mat::from_fn(projected.nrows(), projected.ncols(), |i, j| {
        if values[i] > threshold {
            projected[(i, j)] / values[i]
        } else {
            0.0
        }
    });

    let smallest = *values.last().unwrap_or(&0.0);
    let diagnostics = Diagnostics {
        rank: values.iter().filter(|value| **value > threshold).count(),
        condition: if smallest > 0.0 {
            values[0] / smallest
        } else {
            f64::INFINITY
        },
        singular_values: Some(values),
    };
    Ok((svd.V() * scaled, diagnostics))
}

/// The size below which a singular value is treated as zero.
pub fn singular_value_threshold(n_rows: usize, n_cols: usize, largest: f64) -> f64 {
    n_rows.max(n_cols) as f64 * f64::EPSILON * largest
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
        Options {
            intercept: false,
            solver: Solver::Qr,
            l2_penalty: 0.0,
        }
    }

    fn with_svd() -> Options {
        Options {
            solver: Solver::Svd,
            ..plain()
        }
    }

    fn with_intercept() -> Options {
        Options {
            intercept: true,
            ..plain()
        }
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

        let fit = fit(x.as_ref(), y.as_ref(), None, &with_intercept()).unwrap();

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

        let fit = fit(x.as_ref(), y.as_ref(), Some(&weights), &with_intercept()).unwrap();

        let intercept = fit.intercept.unwrap();
        assert!((intercept[0] - 3.0).abs() < 1e-12);
        assert!((fit.coefficients[(0, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn an_svd_solve_agrees_with_a_qr_solve_on_a_full_rank_system() {
        let x = matrix(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[2.0, 1.0]]);
        let y = matrix(&[&[2.0], &[-1.0], &[1.5], &[3.0]]);

        let by_qr = fit(x.as_ref(), y.as_ref(), None, &plain()).unwrap();
        let by_svd = fit(x.as_ref(), y.as_ref(), None, &with_svd()).unwrap();

        for i in 0..2 {
            assert!((by_qr.coefficients[(i, 0)] - by_svd.coefficients[(i, 0)]).abs() < 1e-12);
        }
        assert!(
            (by_qr.residual_sum_of_squares[0] - by_svd.residual_sum_of_squares[0]).abs() < 1e-12
        );
    }

    #[test]
    fn both_solvers_see_the_full_rank_of_a_well_posed_system() {
        let x = matrix(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[2.0, 1.0]]);
        let y = matrix(&[&[2.0], &[-1.0], &[1.5], &[3.0]]);

        let by_qr = fit(x.as_ref(), y.as_ref(), None, &plain()).unwrap();
        let by_svd = fit(x.as_ref(), y.as_ref(), None, &with_svd()).unwrap();

        assert_eq!(by_qr.rank, 2);
        assert_eq!(by_svd.rank, 2);
        assert!(by_qr.singular_values.is_none());
        assert_eq!(by_svd.singular_values.as_ref().unwrap().len(), 2);
        assert!(by_qr.condition.is_finite());
        assert!(by_svd.condition > 1.0);
    }

    #[test]
    fn a_duplicated_feature_costs_a_rank() {
        let x = matrix(&[&[1.0, 1.0], &[2.0, 2.0], &[3.0, 3.0]]);
        let y = matrix(&[&[2.0], &[4.0], &[6.0]]);

        let by_svd = fit(x.as_ref(), y.as_ref(), None, &with_svd()).unwrap();
        let by_qr = fit(x.as_ref(), y.as_ref(), None, &plain()).unwrap();

        assert_eq!(by_svd.rank, 1);
        assert_eq!(by_qr.rank, 1);
        // The dependent direction does not come out of the factorisation as exactly zero,
        // so the condition estimate is enormous rather than infinite. The rank is what
        // says the system is deficient.
        assert!(by_svd.condition > 1e12);
    }

    #[test]
    fn the_singular_values_come_back_in_decreasing_order() {
        let x = matrix(&[&[3.0, 0.0], &[0.0, 1.0], &[0.0, 0.0]]);
        let y = matrix(&[&[1.0], &[1.0], &[1.0]]);

        let fit = fit(x.as_ref(), y.as_ref(), None, &with_svd()).unwrap();

        let values = fit.singular_values.unwrap();
        assert!((values[0] - 3.0).abs() < 1e-12);
        assert!((values[1] - 1.0).abs() < 1e-12);
        assert!((fit.condition - 3.0).abs() < 1e-12);
    }

    #[test]
    fn an_svd_solve_spreads_a_duplicated_feature_evenly() {
        // The second feature repeats the first, so the system does not pin down a single
        // answer. The smallest-norm solution splits the slope between the two columns.
        let x = matrix(&[&[1.0, 1.0], &[2.0, 2.0], &[3.0, 3.0]]);
        let y = matrix(&[&[2.0], &[4.0], &[6.0]]);

        let fit = fit(x.as_ref(), y.as_ref(), None, &with_svd()).unwrap();

        assert!((fit.coefficients[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((fit.coefficients[(1, 0)] - 1.0).abs() < 1e-10);
        assert!(fit.residual_sum_of_squares[0] < 1e-20);
    }

    #[test]
    fn an_svd_solve_handles_more_features_than_observations() {
        let x = matrix(&[&[1.0, 1.0, 0.0], &[0.0, 1.0, 1.0]]);
        let y = matrix(&[&[2.0], &[2.0]]);

        let fit = fit(x.as_ref(), y.as_ref(), None, &with_svd()).unwrap();

        // Any solution reproduces the targets; this one has the smallest norm of those.
        assert!(fit.residual_sum_of_squares[0] < 1e-20);
        let norm: f64 = (0..3).map(|i| fit.coefficients[(i, 0)].powi(2)).sum();
        assert!(norm < 8.0 / 3.0 + 1e-9);
    }

    #[test]
    fn a_penalty_shrinks_the_coefficients_towards_zero() {
        let x = matrix(&[&[1.0], &[2.0], &[3.0]]);
        let y = matrix(&[&[2.0], &[4.0], &[6.0]]);

        let plain_fit = fit(x.as_ref(), y.as_ref(), None, &plain()).unwrap();
        let ridge = fit(
            x.as_ref(),
            y.as_ref(),
            None,
            &Options {
                l2_penalty: 14.0,
                ..plain()
            },
        )
        .unwrap();

        // beta = x'y / (x'x + lambda) = 28 / (14 + 14).
        assert!((plain_fit.coefficients[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((ridge.coefficients[(0, 0)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_penalty_leaves_the_constant_term_alone() {
        // A feature that is always zero cannot explain anything, so the constant term has
        // to carry the mean whatever the penalty is.
        let x = matrix(&[&[0.0], &[0.0], &[0.0]]);
        let y = matrix(&[&[4.0], &[4.0], &[4.0]]);

        let ridge = fit(
            x.as_ref(),
            y.as_ref(),
            None,
            &Options {
                l2_penalty: 100.0,
                ..with_intercept()
            },
        )
        .unwrap();

        assert!((ridge.intercept.unwrap()[0] - 4.0).abs() < 1e-12);
        assert!(ridge.coefficients[(0, 0)].abs() < 1e-12);
    }

    #[test]
    fn the_residual_ignores_the_rows_the_penalty_adds() {
        let x = matrix(&[&[1.0], &[2.0], &[3.0]]);
        let y = matrix(&[&[2.0], &[4.0], &[6.0]]);

        let ridge = fit(
            x.as_ref(),
            y.as_ref(),
            None,
            &Options {
                l2_penalty: 14.0,
                ..plain()
            },
        )
        .unwrap();

        // With beta = 1 the residuals are 1, 2 and 3.
        assert!((ridge.residual_sum_of_squares[0] - 14.0).abs() < 1e-12);
    }

    #[test]
    fn both_solvers_agree_under_a_penalty() {
        let x = matrix(&[&[1.0, 0.5], &[2.0, -1.0], &[3.0, 0.25]]);
        let y = matrix(&[&[2.0], &[4.0], &[6.0]]);

        let by_qr = fit(
            x.as_ref(),
            y.as_ref(),
            None,
            &Options {
                l2_penalty: 3.0,
                ..plain()
            },
        )
        .unwrap();
        let by_svd = fit(
            x.as_ref(),
            y.as_ref(),
            None,
            &Options {
                l2_penalty: 3.0,
                ..with_svd()
            },
        )
        .unwrap();

        for i in 0..2 {
            assert!((by_qr.coefficients[(i, 0)] - by_svd.coefficients[(i, 0)]).abs() < 1e-10);
        }
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
        assert!(fit(x.as_ref(), y.as_ref(), None, &with_intercept()).is_ok());

        let one_row = matrix(&[&[1.0]]);
        let one_target = matrix(&[&[1.0]]);
        assert!(fit(one_row.as_ref(), one_target.as_ref(), None, &plain()).is_ok());
        assert!(fit(
            one_row.as_ref(),
            one_target.as_ref(),
            None,
            &with_intercept()
        )
        .is_err());
    }
}
