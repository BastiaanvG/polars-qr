//! The Levinson–Durbin recursion, and the Toeplitz solve it generalises to.
//!
//! Durbin's half solves the Yule–Walker system, whose right-hand side is the autocovariance
//! sequence shifted by one. Levinson's half takes the prediction vectors that produces and
//! solves the same matrix against any right-hand side. Both climb one order at a time, and
//! both are `O(p²)` where forming the matrix and factorising it would be `O(p³)`.

use polars::prelude::*;

/// What the recursion produces on its way up through the orders.
#[derive(Clone, Debug)]
pub struct Recursion {
    /// The reflection coefficient at each order. Index 0 is unused and zero.
    ///
    /// These are the partial autocorrelations: the reflection coefficient at order `k` is
    /// the last coefficient of the order-`k` fit, which is what a partial autocorrelation
    /// is defined to be.
    pub reflection: Vec<f64>,
    /// The prediction-error variance at each order, index 0 being the variance of the
    /// series itself.
    pub variance: Vec<f64>,
}

/// Extend the coefficients of one order to the next.
///
/// The new order's coefficients are the previous ones corrected by the reflection
/// coefficient acting on their own reverse. That reversal is the whole trick: it is what
/// the constant diagonals of a Toeplitz matrix buy, and what a general factorisation cannot
/// exploit.
///
/// Each entry needs the one mirroring it, so the two are updated together and the vector
/// never has to be copied to be read from while it is written to.
pub fn extend(coefficients: &mut Vec<f64>, reflection: f64) {
    let length = coefficients.len();
    for index in 0..length / 2 {
        let mirror = length - 1 - index;
        let (front, back) = (coefficients[index], coefficients[mirror]);
        coefficients[index] = front - reflection * back;
        coefficients[mirror] = back - reflection * front;
    }
    if length % 2 == 1 {
        // The middle entry mirrors itself.
        coefficients[length / 2] *= 1.0 - reflection;
    }
    coefficients.push(reflection);
}

/// The coefficients of the fit of `order`, from the reflection coefficients up to it.
pub fn coefficients(reflection: &[f64], order: usize) -> Vec<f64> {
    let mut coefficients = Vec::with_capacity(order);
    for step in &reflection[1..=order] {
        extend(&mut coefficients, *step);
    }
    coefficients
}

/// Run Durbin's recursion over an autocovariance sequence.
///
/// The sequence must be positive definite, which the biased estimator of a real series
/// gives. What that looks like from inside the recursion is a prediction-error variance
/// that stays above zero.
pub fn durbin(sequence: &[f64], operation: &str) -> PolarsResult<Recursion> {
    let max_order = sequence.len() - 1;
    let mut reflection = vec![0.0; max_order + 1];
    let mut variance = vec![0.0; max_order + 1];
    variance[0] = sequence[0];
    if !variance[0].is_finite() {
        return Err(overflowed(operation));
    }
    if variance[0] <= 0.0 {
        return Err(no_variation(operation, variance[0]));
    }

    let mut prediction: Vec<f64> = Vec::with_capacity(max_order);
    for order in 1..=max_order {
        if variance[order - 1] <= 0.0 {
            return Err(exactly_predictable(operation, order - 1));
        }
        let mut cross = sequence[order];
        for (index, coefficient) in prediction.iter().enumerate() {
            cross -= coefficient * sequence[order - 1 - index];
        }

        // A reflection coefficient that is not finite would carry NaN into every
        // coefficient above it, and NaN compares false against every bound, so nothing
        // further would notice. Stopping here is what keeps the guards above meaningful.
        let step = cross / variance[order - 1];
        if !step.is_finite() {
            return Err(overflowed(operation));
        }
        extend(&mut prediction, step);
        reflection[order] = step;
        // Rounding can carry this a little below zero when the fit is all but exact, and a
        // negative variance is not a smaller variance. Clamping leaves the next order to
        // report that the recursion has nothing further to find.
        variance[order] = (variance[order - 1] * (1.0 - step * step)).max(0.0);
    }

    Ok(Recursion {
        reflection,
        variance,
    })
}

/// Solve a symmetric positive-definite Toeplitz system against every right-hand side.
///
/// The matrix is given by its first column, since that is all a Toeplitz matrix has. The
/// prediction vectors climb alongside the solutions, because extending a solution by one
/// row is exactly the prediction vector of that order applied to the residual left over.
pub fn solve(
    first_column: &[f64],
    rhs: &[Vec<f64>],
    diagonal_shift: f64,
    operation: &str,
) -> PolarsResult<Vec<Vec<f64>>> {
    let n = first_column.len();
    let mut row = first_column.to_vec();
    row[0] += diagonal_shift;
    if !row[0].is_finite() || row[0] <= 0.0 {
        return Err(not_positive_definite(operation, 0, diagonal_shift));
    }

    let mut error = row[0];
    let mut prediction: Vec<f64> = Vec::with_capacity(n);
    let mut solutions: Vec<Vec<f64>> = rhs.iter().map(|b| vec![b[0] / row[0]]).collect();

    for order in 1..n {
        let mut cross = row[order];
        for (index, coefficient) in prediction.iter().enumerate() {
            cross -= coefficient * row[order - 1 - index];
        }
        let step = cross / error;
        if !step.is_finite() {
            return Err(overflowed(operation));
        }
        extend(&mut prediction, step);
        error *= 1.0 - step * step;
        if error <= 0.0 {
            return Err(not_positive_definite(operation, order, diagonal_shift));
        }

        // Every solution holds the leading system of this order, and the update reads the
        // prediction vector rather than the solution, so nothing has to be copied to be
        // read while it is written to.
        let length = order;
        for (solution, b) in solutions.iter_mut().zip(rhs) {
            let mut residual = b[order];
            for (index, value) in solution.iter().enumerate() {
                residual -= row[order - index] * value;
            }
            let weight = residual / error;
            for (index, value) in solution.iter_mut().enumerate() {
                *value -= weight * prediction[length - 1 - index];
            }
            solution.push(weight);
        }
    }

    Ok(solutions)
}

/// The series has no spread for the recursion to work with.
fn no_variation(operation: &str, variance: f64) -> PolarsError {
    polars_err!(
        ComputeError:
        "{} received a series with variance {}.\n\nA series that does not move has no \
         autocorrelation to model.",
        operation, variance,
    )
}

/// The recursion ran out of anything left to predict.
fn exactly_predictable(operation: &str, order: usize) -> PolarsError {
    polars_err!(
        ComputeError:
        "{} drove the prediction error to zero at order {}.\n\nThe series is exactly \
         determined by that many lags, so no higher order can be fitted. Ask for an order \
         of at most {}.",
        operation, order, order,
    )
}

/// The arithmetic left the range a double can hold.
///
/// Worth its own message because the alternative is a result full of NaN: every comparison
/// against NaN is false, so a NaN that is allowed to spread passes every bound the
/// recursion checks and is reported as an answer.
fn overflowed(operation: &str) -> PolarsError {
    polars_err!(
        ComputeError:
        "{} overflowed the range of a double.\n\nThe values are large enough that their \
         products do not fit. Scale the series down before fitting it.",
        operation,
    )
}

/// The matrix the caller gave is not what the recursion needs.
fn not_positive_definite(operation: &str, order: usize, diagonal_shift: f64) -> PolarsError {
    polars_err!(
        ComputeError:
        "{} met a non-positive prediction error at order {}, which is what a matrix that is \
         not positive definite looks like from inside the recursion.\n\nA larger \
         diagonal_shift than {} may make it so.",
        operation, order, diagonal_shift,
    )
}

#[cfg(test)]
mod tests {
    use faer::linalg::solvers::Solve;
    use faer::Mat;

    use super::*;

    /// Build the Toeplitz matrix a first column stands for.
    fn toeplitz(first_column: &[f64]) -> Mat<f64> {
        let n = first_column.len();
        Mat::from_fn(n, n, |i, j| first_column[i.abs_diff(j)])
    }

    /// Solve the same system the long way round, as the oracle.
    fn dense_solve(first_column: &[f64], rhs: &[f64]) -> Vec<f64> {
        let matrix = toeplitz(first_column);
        let b = Mat::from_fn(rhs.len(), 1, |i, _| rhs[i]);
        let solved = matrix.partial_piv_lu().solve(&b);
        (0..rhs.len()).map(|i| solved[(i, 0)]).collect()
    }

    /// An autocovariance sequence of an AR(1) with the given coefficient.
    fn ar1_sequence(phi: f64, max_lag: usize) -> Vec<f64> {
        (0..=max_lag).map(|lag| phi.powi(lag as i32)).collect()
    }

    /// The update written the obvious way, to check the in-place one against.
    fn extended(coefficients: &[f64], reflection: f64) -> Vec<f64> {
        let mut out: Vec<f64> = coefficients
            .iter()
            .enumerate()
            .map(|(index, value)| value - reflection * coefficients[coefficients.len() - 1 - index])
            .collect();
        out.push(reflection);
        out
    }

    #[test]
    fn extending_in_place_matches_extending_by_copy() {
        // An odd length is the case with an entry that mirrors itself.
        for start in [vec![], vec![0.5], vec![0.5, -0.25], vec![0.5, -0.25, 0.125]] {
            let mut in_place = start.clone();
            extend(&mut in_place, 0.3);

            assert_eq!(in_place, extended(&start, 0.3), "starting from {start:?}");
        }
    }

    #[test]
    fn an_ar1_sequence_recovers_its_own_coefficient() {
        let recursion = durbin(&ar1_sequence(0.7, 4), "").unwrap();

        let fitted = coefficients(&recursion.reflection, 1);
        assert!((fitted[0] - 0.7).abs() < 1e-12);
    }

    #[test]
    fn an_ar1_sequence_has_no_partial_autocorrelation_beyond_lag_one() {
        let recursion = durbin(&ar1_sequence(0.7, 5), "").unwrap();

        assert!((recursion.reflection[1] - 0.7).abs() < 1e-12);
        for order in 2..=5 {
            assert!(recursion.reflection[order].abs() < 1e-12);
        }
    }

    #[test]
    fn the_recursion_agrees_with_solving_the_system_the_long_way() {
        let sequence = [1.0, 0.6, 0.1, -0.2, -0.3];

        let recursion = durbin(&sequence, "").unwrap();

        for order in 1..sequence.len() {
            let fitted = coefficients(&recursion.reflection, order);
            let dense = dense_solve(&sequence[..order], &sequence[1..=order]);
            for (ours, theirs) in fitted.iter().zip(&dense) {
                assert!((ours - theirs).abs() < 1e-12, "{ours} vs {theirs}");
            }
        }
    }

    #[test]
    fn the_reflection_coefficient_is_the_last_coefficient_of_its_own_order() {
        let recursion = durbin(&[1.0, 0.6, 0.1, -0.2, -0.3], "").unwrap();

        for order in 1..=4 {
            let fitted = coefficients(&recursion.reflection, order);
            assert!((fitted[order - 1] - recursion.reflection[order]).abs() < 1e-14);
        }
    }

    #[test]
    fn the_prediction_error_falls_with_every_order() {
        let recursion = durbin(&[1.0, 0.6, 0.1, -0.2, -0.3], "").unwrap();

        for order in 1..recursion.variance.len() {
            assert!(recursion.variance[order] <= recursion.variance[order - 1]);
        }
    }

    #[test]
    fn a_positive_definite_sequence_stays_inside_the_unit_circle() {
        let recursion = durbin(&[1.0, 0.6, 0.1, -0.2, -0.3], "").unwrap();

        assert!(recursion.reflection[1..].iter().all(|k| k.abs() < 1.0));
    }

    #[test]
    fn a_series_with_no_spread_is_rejected() {
        assert!(durbin(&[0.0, 0.0], "ar").is_err());
    }

    #[test]
    fn a_sequence_that_predicts_itself_exactly_is_reported_rather_than_divided_by() {
        // Correlation one at lag one leaves nothing for the second order to explain.
        let error = durbin(&[1.0, 1.0, 1.0], "ar").unwrap_err();

        assert!(error.to_string().contains("order 1"));
    }

    #[test]
    fn solving_agrees_with_solving_the_long_way() {
        let first_column = [4.0, 1.0, 0.5, 0.25];
        let rhs = vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.0, -1.0, 1.0, 0.0]];

        let solved = solve(&first_column, &rhs, 0.0, "").unwrap();

        for (ours, b) in solved.iter().zip(&rhs) {
            let dense = dense_solve(&first_column, b);
            for (value, expected) in ours.iter().zip(&dense) {
                assert!((value - expected).abs() < 1e-10, "{value} vs {expected}");
            }
        }
    }

    #[test]
    fn a_solution_reproduces_its_right_hand_side() {
        let first_column = [3.0, 1.0, 0.25];
        let rhs = vec![vec![1.0, 0.0, 2.0]];

        let solved = solve(&first_column, &rhs, 0.0, "").unwrap();

        let matrix = toeplitz(&first_column);
        for row in 0..3 {
            let reproduced: f64 = (0..3).map(|j| matrix[(row, j)] * solved[0][j]).sum();
            assert!((reproduced - rhs[0][row]).abs() < 1e-10);
        }
    }

    #[test]
    fn a_matrix_that_is_not_positive_definite_is_rejected() {
        // The off-diagonal is larger than the diagonal, so the matrix is indefinite.
        let solved = solve(&[1.0, 2.0, 0.0], &[vec![1.0, 1.0, 1.0]], 0.0, "ar");

        assert!(solved.is_err());
    }

    #[test]
    fn a_shift_makes_a_singular_matrix_solvable() {
        let first_column = [1.0, 1.0, 1.0];
        let rhs = vec![vec![1.0, 1.0, 1.0]];

        assert!(solve(&first_column, &rhs, 0.0, "ar").is_err());
        assert!(solve(&first_column, &rhs, 1e-6, "ar").is_ok());
    }
}
