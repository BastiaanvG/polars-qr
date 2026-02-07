//! Solving systems whose matrix is symmetric and positive definite.

use faer::linalg::solvers::Solve;
use faer::{Mat, MatRef, Side};
use polars::prelude::*;

/// How far the two triangles of the matrix may disagree before it is rejected, relative to
/// the size of its largest entry.
const SYMMETRY_TOLERANCE: f64 = 1e-8;

/// How the system is set up.
pub struct Options {
    /// A value added to every diagonal entry before factorising.
    pub diagonal_shift: f64,
}

/// The solution of a positive-definite system.
pub struct SpdSolution {
    /// One column per right-hand side, one row per row of the matrix.
    pub solution: Mat<f64>,
    /// How far the two triangles of the matrix disagreed, relative to its largest entry.
    pub symmetry_error: f64,
}

/// Solve `matrix * x = rhs` through a Cholesky factorisation.
///
/// The matrix is factorised once and every right-hand side is solved against that one
/// factorisation.
pub fn solve_spd(
    matrix: MatRef<'_, f64>,
    rhs: MatRef<'_, f64>,
    options: &Options,
) -> PolarsResult<SpdSolution> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        polars_bail!(
            ShapeMismatch:
            "the matrix has {} rows and {} columns, so it is not square", n, matrix.ncols(),
        );
    }
    if rhs.nrows() != n {
        polars_bail!(
            ShapeMismatch:
            "the right-hand side has {} rows and the matrix has {}", rhs.nrows(), n,
        );
    }

    let symmetry_error = symmetry_error(matrix);
    if symmetry_error > SYMMETRY_TOLERANCE {
        polars_bail!(
            ComputeError:
            "the matrix is not symmetric: its two triangles differ by {:.3e} relative to \
             its largest entry, which is more than the {:.0e} allowed",
            symmetry_error, SYMMETRY_TOLERANCE,
        );
    }

    // Work from the average of the two triangles: they agree to within the tolerance
    // already, and averaging them makes the input exactly symmetric.
    let prepared = Mat::from_fn(n, n, |i, j| {
        let averaged = 0.5 * (matrix[(i, j)] + matrix[(j, i)]);
        if i == j {
            averaged + options.diagonal_shift
        } else {
            averaged
        }
    });

    let llt = prepared.llt(Side::Lower).map_err(|_| {
        polars_err!(
            ComputeError:
            "the matrix is not positive definite; a larger diagonal_shift than {} may make \
             it so",
            options.diagonal_shift,
        )
    })?;

    Ok(SpdSolution {
        solution: llt.solve(rhs),
        symmetry_error,
    })
}

/// How far the two triangles of `matrix` disagree, relative to its largest entry.
fn symmetry_error(matrix: MatRef<'_, f64>) -> f64 {
    let n = matrix.nrows();
    let mut largest = 0.0f64;
    let mut difference = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            largest = largest.max(matrix[(i, j)].abs());
            if j < i {
                difference = difference.max((matrix[(i, j)] - matrix[(j, i)]).abs());
            }
        }
    }
    if largest > 0.0 {
        difference / largest
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[&[f64]]) -> Mat<f64> {
        Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
    }

    fn unshifted() -> Options {
        Options {
            diagonal_shift: 0.0,
        }
    }

    #[test]
    fn solves_a_small_system() {
        // [[4, 1], [1, 3]] x = [1, 2] has the solution [1/11, 7/11].
        let a = matrix(&[&[4.0, 1.0], &[1.0, 3.0]]);
        let b = matrix(&[&[1.0], &[2.0]]);

        let solved = solve_spd(a.as_ref(), b.as_ref(), &unshifted()).unwrap();

        assert!((solved.solution[(0, 0)] - 1.0 / 11.0).abs() < 1e-12);
        assert!((solved.solution[(1, 0)] - 7.0 / 11.0).abs() < 1e-12);
        assert_eq!(solved.symmetry_error, 0.0);
    }

    #[test]
    fn every_right_hand_side_is_solved_against_one_factorisation() {
        let a = matrix(&[&[4.0, 1.0], &[1.0, 3.0]]);
        let both = matrix(&[&[1.0, 0.0], &[2.0, 1.0]]);
        let first = matrix(&[&[1.0], &[2.0]]);

        let together = solve_spd(a.as_ref(), both.as_ref(), &unshifted()).unwrap();
        let apart = solve_spd(a.as_ref(), first.as_ref(), &unshifted()).unwrap();

        assert_eq!(together.solution.ncols(), 2);
        assert!((together.solution[(0, 0)] - apart.solution[(0, 0)]).abs() < 1e-12);
        assert!((together.solution[(1, 0)] - apart.solution[(1, 0)]).abs() < 1e-12);
    }

    #[test]
    fn the_solution_reproduces_the_right_hand_side() {
        let a = matrix(&[&[2.0, -1.0, 0.0], &[-1.0, 2.0, -1.0], &[0.0, -1.0, 2.0]]);
        let b = matrix(&[&[1.0], &[0.0], &[3.0]]);

        let solved = solve_spd(a.as_ref(), b.as_ref(), &unshifted()).unwrap();

        let reproduced = &a * &solved.solution;
        for i in 0..3 {
            assert!((reproduced[(i, 0)] - b[(i, 0)]).abs() < 1e-12);
        }
    }

    #[test]
    fn a_shift_makes_a_singular_matrix_solvable() {
        // The second row repeats the first, so the matrix is singular as it stands.
        let a = matrix(&[&[1.0, 1.0], &[1.0, 1.0]]);
        let b = matrix(&[&[1.0], &[1.0]]);

        assert!(solve_spd(a.as_ref(), b.as_ref(), &unshifted()).is_err());
        assert!(solve_spd(
            a.as_ref(),
            b.as_ref(),
            &Options {
                diagonal_shift: 1e-6
            }
        )
        .is_ok());
    }

    #[test]
    fn a_matrix_that_is_not_positive_definite_is_rejected() {
        let a = matrix(&[&[1.0, 2.0], &[2.0, 1.0]]);
        let b = matrix(&[&[1.0], &[1.0]]);

        assert!(solve_spd(a.as_ref(), b.as_ref(), &unshifted()).is_err());
    }

    #[test]
    fn a_matrix_that_is_not_symmetric_is_rejected() {
        let a = matrix(&[&[4.0, 1.0], &[2.0, 3.0]]);
        let b = matrix(&[&[1.0], &[1.0]]);

        assert!(solve_spd(a.as_ref(), b.as_ref(), &unshifted()).is_err());
    }

    #[test]
    fn a_rounding_sized_asymmetry_is_tolerated_and_reported() {
        let a = matrix(&[&[4.0, 1.0], &[1.0 + 1e-12, 3.0]]);
        let b = matrix(&[&[1.0], &[2.0]]);

        let solved = solve_spd(a.as_ref(), b.as_ref(), &unshifted()).unwrap();

        assert!(solved.symmetry_error > 0.0);
        assert!(solved.symmetry_error < SYMMETRY_TOLERANCE);
        assert!((solved.solution[(0, 0)] - 1.0 / 11.0).abs() < 1e-9);
    }

    #[test]
    fn a_matrix_that_is_not_square_is_rejected() {
        let a = matrix(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]]);
        let b = matrix(&[&[1.0], &[1.0]]);

        assert!(solve_spd(a.as_ref(), b.as_ref(), &unshifted()).is_err());
    }
}
