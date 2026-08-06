//! Mergeable state for a least-squares problem.
//!
//! The state is the upper-triangular factor `R` of a QR factorisation of the design matrix
//! with the targets appended to it. That one matrix carries everything a fit needs: the
//! cross-products of the features with each other and with the targets, and the residual
//! that is left once the features have explained what they can.
//!
//! Merging two states stacks their factors and factorises again. That is what makes the
//! result independent of how the rows were partitioned, and it keeps the numerics of a QR
//! rather than falling back on normal equations, which square the condition number.

use faer::{Mat, MatRef};
use polars::prelude::*;

use crate::least_squares::{self, LeastSquaresFit, Solver};
use crate::states::codec::{Kind, Reader, SchemaHash, Writer};
use crate::weights::Weights;

/// The summary of a least-squares problem over some set of rows.
pub struct LeastSquaresState {
    /// The names of the feature columns, in order.
    pub features: Vec<String>,
    /// The names of the target columns, in order.
    pub targets: Vec<String>,
    /// Whether the design matrix carries a constant term.
    pub intercept: bool,
    /// The upper-triangular factor of `[design | targets]`.
    pub factor: Mat<f64>,
    /// How many observations went into it.
    pub n_observations: u64,
    /// The sum of their weights, which is the count when there are none.
    pub sum_weights: f64,
}

impl LeastSquaresState {
    /// Summarise the rows of one partition.
    pub fn accumulate(
        features: MatRef<'_, f64>,
        targets: MatRef<'_, f64>,
        weights: Option<&Weights>,
        feature_names: Vec<String>,
        target_names: Vec<String>,
        intercept: bool,
    ) -> PolarsResult<Self> {
        let n = features.nrows();
        if targets.nrows() != n {
            polars_bail!(
                ShapeMismatch:
                "the targets have {} rows and the features have {}", targets.nrows(), n,
            );
        }

        let columns = features.ncols() + usize::from(intercept);
        let augmented = Mat::from_fn(n, columns + targets.ncols(), |i, j| {
            if intercept && j == 0 {
                1.0
            } else if j < columns {
                features[(i, j - usize::from(intercept))]
            } else {
                targets[(i, j - columns)]
            }
        });
        let augmented = match weights {
            Some(weights) => weights.scale_rows(augmented.as_ref()),
            None => augmented,
        };

        Ok(Self {
            features: feature_names,
            targets: target_names,
            intercept,
            factor: triangular_factor(augmented.as_ref()),
            n_observations: n as u64,
            sum_weights: weights.map_or(n as f64, Weights::total),
        })
    }

    /// Combine two summaries into one covering both sets of rows.
    pub fn merge(&self, other: &Self) -> PolarsResult<Self> {
        if self.features != other.features
            || self.targets != other.targets
            || self.intercept != other.intercept
        {
            polars_bail!(
                ComputeError:
                "these states were built for different columns, so they cannot be merged",
            );
        }

        let size = self.factor.ncols();
        let stacked = Mat::from_fn(2 * size, size, |i, j| {
            if i < size {
                self.factor[(i, j)]
            } else {
                other.factor[(i - size, j)]
            }
        });

        Ok(Self {
            features: self.features.clone(),
            targets: self.targets.clone(),
            intercept: self.intercept,
            factor: triangular_factor(stacked.as_ref()),
            n_observations: self.n_observations + other.n_observations,
            sum_weights: self.sum_weights + other.sum_weights,
        })
    }

    /// Solve the problem the state summarises.
    ///
    /// The factor stands in for the data: the least-squares problem over `R` has the same
    /// solution as the one over the rows it was built from.
    pub fn finalise(&self, solver: Solver, l2_penalty: f64) -> PolarsResult<LeastSquaresFit> {
        let columns = self.features.len() + usize::from(self.intercept);
        let design = self.factor.subcols(0, columns).to_owned();
        let targets = self.factor.subcols(columns, self.targets.len()).to_owned();

        // The factor is fitted as it stands: its first column is the constant term when
        // there is one, and the rows below the design block hold the residual the
        // factorisation set aside, so the solve over the factor reports the residual of the
        // data it was built from.
        let mut fit = least_squares::fit_design(
            design.as_ref(),
            targets.as_ref(),
            &least_squares::Options {
                intercept: self.intercept,
                solver,
                l2_penalty,
            },
        )?;

        // The rows of the factor are not observations, so the count comes from the state.
        fit.n_observations = self.n_observations as usize;
        Ok(fit)
    }

    /// The schema two states have to share before they can be merged.
    fn schema(&self) -> SchemaHash {
        SchemaHash::new()
            .names(&self.features)
            .names(&self.targets)
            .byte(u8::from(self.intercept))
    }

    /// Write the state out.
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = Writer::new(Kind::LeastSquares, self.schema());
        writer
            .u32(self.features.len() as u32)
            .u32(self.targets.len() as u32)
            .flag(self.intercept)
            .u64(self.n_observations)
            .f64(self.sum_weights);
        let size = self.factor.ncols();
        writer.u32(size as u32);
        for i in 0..size {
            writer.numbers((0..size).map(|j| self.factor[(i, j)]));
        }
        writer.names(&self.features);
        writer.names(&self.targets);
        writer.finish()
    }

    /// Read a state back.
    pub fn decode(bytes: &[u8]) -> PolarsResult<Self> {
        let mut reader = Reader::new(bytes, Kind::LeastSquares)?;
        let n_features = reader.u32()? as usize;
        let n_targets = reader.u32()? as usize;
        let intercept = reader.flag()?;
        let n_observations = reader.u64()?;
        let sum_weights = reader.f64()?;

        let size = reader.u32()? as usize;
        if size != n_features + usize::from(intercept) + n_targets {
            polars_bail!(ComputeError: "this least-squares state has an inconsistent size");
        }
        let values = reader.numbers(size * size)?;
        let factor = Mat::from_fn(size, size, |i, j| values[i * size + j]);

        let features = reader.names()?;
        let targets = reader.names()?;
        if features.len() != n_features || targets.len() != n_targets {
            polars_bail!(ComputeError: "this least-squares state names the wrong columns");
        }

        let state = Self {
            features,
            targets,
            intercept,
            factor,
            n_observations,
            sum_weights,
        };
        if state.schema() != reader.schema() {
            polars_bail!(
                ComputeError:
                "this least-squares state does not match the schema it was written with",
            );
        }
        Ok(state)
    }
}

/// The upper-triangular factor of `matrix`, padded to be square.
///
/// Only `R` is kept: the orthogonal factor is what the rows looked like, and the point of a
/// state is to forget that while keeping what it implies.
fn triangular_factor(matrix: MatRef<'_, f64>) -> Mat<f64> {
    let size = matrix.ncols();
    let qr = matrix.qr();
    let r = qr.R();
    Mat::from_fn(size, size, |i, j| {
        if i <= j && i < r.nrows() {
            r[(i, j)]
        } else {
            0.0
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[&[f64]]) -> Mat<f64> {
        Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
    }

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn state(x: &Mat<f64>, y: &Mat<f64>, intercept: bool) -> LeastSquaresState {
        LeastSquaresState::accumulate(
            x.as_ref(),
            y.as_ref(),
            None,
            names(&["a", "b"]),
            names(&["y"]),
            intercept,
        )
        .unwrap()
    }

    fn design() -> (Mat<f64>, Mat<f64>) {
        let x = matrix(&[
            &[1.0, 0.5],
            &[2.0, -1.0],
            &[3.0, 2.0],
            &[-1.0, 0.0],
            &[0.5, 1.5],
            &[4.0, -2.0],
        ]);
        let y = matrix(&[&[2.0], &[3.0], &[8.0], &[-1.5], &[2.0], &[5.0]]);
        (x, y)
    }

    fn split(matrix: &Mat<f64>, rows: &[usize]) -> Mat<f64> {
        Mat::from_fn(rows.len(), matrix.ncols(), |i, j| matrix[(rows[i], j)])
    }

    #[test]
    fn a_state_over_every_row_fits_what_the_rows_fit() {
        let (x, y) = design();
        let direct = least_squares::fit(
            x.as_ref(),
            y.as_ref(),
            None,
            &least_squares::Options {
                intercept: false,
                solver: Solver::Qr,
                l2_penalty: 0.0,
            },
        )
        .unwrap();

        let fit = state(&x, &y, false).finalise(Solver::Qr, 0.0).unwrap();

        for i in 0..2 {
            assert!((fit.coefficients[(i, 0)] - direct.coefficients[(i, 0)]).abs() < 1e-10);
        }
        assert!((fit.residual_sum_of_squares[0] - direct.residual_sum_of_squares[0]).abs() < 1e-10);
        assert_eq!(fit.n_observations, 6);
    }

    #[test]
    fn merging_two_partitions_fits_what_all_the_rows_fit() {
        let (x, y) = design();
        let whole = state(&x, &y, false).finalise(Solver::Qr, 0.0).unwrap();

        let first = state(&split(&x, &[0, 1, 2]), &split(&y, &[0, 1, 2]), false);
        let second = state(&split(&x, &[3, 4, 5]), &split(&y, &[3, 4, 5]), false);
        let merged = first
            .merge(&second)
            .unwrap()
            .finalise(Solver::Qr, 0.0)
            .unwrap();

        for i in 0..2 {
            assert!((merged.coefficients[(i, 0)] - whole.coefficients[(i, 0)]).abs() < 1e-10);
        }
        assert!(
            (merged.residual_sum_of_squares[0] - whole.residual_sum_of_squares[0]).abs() < 1e-10
        );
        assert_eq!(merged.n_observations, 6);
    }

    #[test]
    fn the_order_the_partitions_are_merged_in_does_not_matter() {
        let (x, y) = design();
        let parts: Vec<LeastSquaresState> = [&[0usize, 1][..], &[2, 3][..], &[4, 5][..]]
            .iter()
            .map(|rows| state(&split(&x, rows), &split(&y, rows), true))
            .collect();

        let forwards = parts[0]
            .merge(&parts[1])
            .unwrap()
            .merge(&parts[2])
            .unwrap()
            .finalise(Solver::Qr, 0.0)
            .unwrap();
        let backwards = parts[2]
            .merge(&parts[1])
            .unwrap()
            .merge(&parts[0])
            .unwrap()
            .finalise(Solver::Qr, 0.0)
            .unwrap();

        assert!((forwards.coefficients[(0, 0)] - backwards.coefficients[(0, 0)]).abs() < 1e-10);
        assert!((forwards.intercept.unwrap()[0] - backwards.intercept.unwrap()[0]).abs() < 1e-10);
    }

    #[test]
    fn a_state_carries_an_intercept_through() {
        let (x, y) = design();
        let direct = least_squares::fit(
            x.as_ref(),
            y.as_ref(),
            None,
            &least_squares::Options {
                intercept: true,
                solver: Solver::Qr,
                l2_penalty: 0.0,
            },
        )
        .unwrap();

        let fit = state(&x, &y, true).finalise(Solver::Qr, 0.0).unwrap();

        assert!((fit.intercept.unwrap()[0] - direct.intercept.unwrap()[0]).abs() < 1e-10);
        for i in 0..2 {
            assert!((fit.coefficients[(i, 0)] - direct.coefficients[(i, 0)]).abs() < 1e-10);
        }
    }

    #[test]
    fn a_penalty_is_applied_once_when_the_state_is_finalised() {
        let (x, y) = design();
        let direct = least_squares::fit(
            x.as_ref(),
            y.as_ref(),
            None,
            &least_squares::Options {
                intercept: false,
                solver: Solver::Qr,
                l2_penalty: 4.0,
            },
        )
        .unwrap();

        let first = state(&split(&x, &[0, 1, 2]), &split(&y, &[0, 1, 2]), false);
        let second = state(&split(&x, &[3, 4, 5]), &split(&y, &[3, 4, 5]), false);
        let merged = first
            .merge(&second)
            .unwrap()
            .finalise(Solver::Qr, 4.0)
            .unwrap();

        for i in 0..2 {
            assert!((merged.coefficients[(i, 0)] - direct.coefficients[(i, 0)]).abs() < 1e-10);
        }
    }

    #[test]
    fn a_state_survives_a_round_trip_through_its_bytes() {
        let (x, y) = design();
        let original = state(&x, &y, true);

        let restored = LeastSquaresState::decode(&original.encode()).unwrap();

        assert_eq!(restored.features, original.features);
        assert_eq!(restored.targets, original.targets);
        assert_eq!(restored.intercept, original.intercept);
        assert_eq!(restored.n_observations, original.n_observations);
        assert_eq!(restored.sum_weights, original.sum_weights);
        for i in 0..original.factor.nrows() {
            for j in 0..original.factor.ncols() {
                assert_eq!(restored.factor[(i, j)], original.factor[(i, j)]);
            }
        }
    }

    #[test]
    fn states_for_different_columns_refuse_to_merge() {
        let (x, y) = design();
        let one = state(&x, &y, false);
        let other = LeastSquaresState::accumulate(
            x.as_ref(),
            y.as_ref(),
            None,
            names(&["a", "c"]),
            names(&["y"]),
            false,
        )
        .unwrap();

        assert!(one.merge(&other).is_err());
    }

    #[test]
    fn weights_are_carried_into_the_state() {
        let (x, y) = design();
        let w = Mat::from_fn(6, 1, |i, _| if i < 3 { 1.0 } else { 2.0 });
        let weights = Weights::new(w.as_ref(), "w").unwrap();

        let direct = least_squares::fit(
            x.as_ref(),
            y.as_ref(),
            Some(&weights),
            &least_squares::Options {
                intercept: false,
                solver: Solver::Qr,
                l2_penalty: 0.0,
            },
        )
        .unwrap();
        let state = LeastSquaresState::accumulate(
            x.as_ref(),
            y.as_ref(),
            Some(&weights),
            names(&["a", "b"]),
            names(&["y"]),
            false,
        )
        .unwrap();
        let fit = state.finalise(Solver::Qr, 0.0).unwrap();

        assert_eq!(state.sum_weights, 9.0);
        for i in 0..2 {
            assert!((fit.coefficients[(i, 0)] - direct.coefficients[(i, 0)]).abs() < 1e-10);
        }
        assert!((fit.residual_sum_of_squares[0] - direct.residual_sum_of_squares[0]).abs() < 1e-10);
    }
}
