//! Mergeable state for second moments.
//!
//! The state is the count, the weight, the mean vector and the centred cross-product
//! matrix. Merging two of them shifts both means onto the combined mean and corrects the
//! cross-products for the shift, which is what keeps the result independent of how the rows
//! were partitioned. Accumulating the raw sums instead would be smaller still, but centred
//! moments lose far less precision when the values are large next to their spread.

use faer::{Mat, MatRef};
use polars::prelude::*;

use crate::covariance::{self, Covariance};
use crate::states::codec::{Kind, Reader, SchemaHash, Writer};
use crate::weights::Weights;

/// The summary of a set of columns over some rows.
pub struct CovarianceState {
    /// The names of the columns, in order.
    pub features: Vec<String>,
    /// How many observations went into it.
    pub n_observations: u64,
    /// The sum of their weights, which is the count when there are none.
    pub sum_weights: f64,
    /// The sum of the squares of their weights, which the divisor corrects for.
    pub sum_squared_weights: f64,
    /// The weighted mean of each column.
    pub means: Vec<f64>,
    /// The weighted cross-products of the columns about those means.
    pub cross_products: Mat<f64>,
}

impl CovarianceState {
    /// Summarise the rows of one partition.
    pub fn accumulate(
        x: MatRef<'_, f64>,
        weights: Option<&Weights>,
        features: Vec<String>,
    ) -> PolarsResult<Self> {
        let (n, p) = (x.nrows(), x.ncols());
        if features.len() != p {
            polars_bail!(
                ShapeMismatch:
                "{} columns were given {} names", p, features.len(),
            );
        }

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

        Ok(Self {
            features,
            n_observations: n as u64,
            sum_weights,
            sum_squared_weights,
            means,
            cross_products: weighted.transpose() * &centred,
        })
    }

    /// Combine two summaries into one covering both sets of rows.
    pub fn merge(&self, other: &Self) -> PolarsResult<Self> {
        if self.features != other.features {
            polars_bail!(
                ComputeError:
                "these states were built for different columns, so they cannot be merged",
            );
        }

        let total = self.sum_weights + other.sum_weights;
        let share = other.sum_weights / total;
        let shift: Vec<f64> = self
            .means
            .iter()
            .zip(&other.means)
            .map(|(here, there)| there - here)
            .collect();

        let means = self
            .means
            .iter()
            .zip(&shift)
            .map(|(here, delta)| here + delta * share)
            .collect();

        // Each side's cross-products are about its own mean; the correction moves both onto
        // the mean of the two together.
        let correction = self.sum_weights * other.sum_weights / total;
        let p = self.features.len();
        let cross_products = Mat::from_fn(p, p, |i, j| {
            self.cross_products[(i, j)]
                + other.cross_products[(i, j)]
                + correction * shift[i] * shift[j]
        });

        Ok(Self {
            features: self.features.clone(),
            n_observations: self.n_observations + other.n_observations,
            sum_weights: total,
            sum_squared_weights: self.sum_squared_weights + other.sum_squared_weights,
            means,
            cross_products,
        })
    }

    /// Turn the state into the covariance or correlation it summarises.
    pub fn finalise(&self, options: &covariance::Options) -> PolarsResult<Covariance> {
        covariance::from_moments(
            &self.means,
            self.cross_products.as_ref(),
            self.n_observations as usize,
            self.sum_weights,
            self.sum_squared_weights,
            options,
        )
    }

    /// The schema two states have to share before they can be merged.
    fn schema(&self) -> SchemaHash {
        SchemaHash::new().names(&self.features)
    }

    /// Write the state out.
    pub fn encode(&self) -> Vec<u8> {
        let mut writer = Writer::new(Kind::Covariance, self.schema());
        let p = self.features.len();
        writer
            .u32(p as u32)
            .u64(self.n_observations)
            .f64(self.sum_weights)
            .f64(self.sum_squared_weights)
            .numbers(self.means.iter().copied());
        for i in 0..p {
            writer.numbers((0..p).map(|j| self.cross_products[(i, j)]));
        }
        writer.names(&self.features);
        writer.finish()
    }

    /// Read a state back.
    pub fn decode(bytes: &[u8]) -> PolarsResult<Self> {
        let mut reader = Reader::new(bytes, Kind::Covariance)?;
        let p = reader.u32()? as usize;
        let n_observations = reader.u64()?;
        let sum_weights = reader.f64()?;
        let sum_squared_weights = reader.f64()?;
        let means = reader.numbers(p)?;
        let values = reader.numbers(p * p)?;
        let cross_products = Mat::from_fn(p, p, |i, j| values[i * p + j]);
        let features = reader.names()?;
        if features.len() != p {
            polars_bail!(ComputeError: "this covariance state names the wrong columns");
        }

        let state = Self {
            features,
            n_observations,
            sum_weights,
            sum_squared_weights,
            means,
            cross_products,
        };
        if state.schema() != reader.schema() {
            polars_bail!(
                ComputeError:
                "this covariance state does not match the schema it was written with",
            );
        }
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: &[&[f64]]) -> Mat<f64> {
        Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
    }

    fn names() -> Vec<String> {
        vec!["a".to_string(), "b".to_string()]
    }

    fn sample() -> covariance::Options {
        covariance::Options {
            ddof: 1.0,
            normalise: false,
        }
    }

    fn data() -> Mat<f64> {
        matrix(&[
            &[1.0, 0.5],
            &[2.0, -1.0],
            &[3.0, 2.0],
            &[-1.0, 0.0],
            &[0.5, 1.5],
            &[4.0, -2.0],
        ])
    }

    fn split(matrix: &Mat<f64>, rows: &[usize]) -> Mat<f64> {
        Mat::from_fn(rows.len(), matrix.ncols(), |i, j| matrix[(rows[i], j)])
    }

    fn state(x: &Mat<f64>) -> CovarianceState {
        CovarianceState::accumulate(x.as_ref(), None, names()).unwrap()
    }

    #[test]
    fn a_state_over_every_row_estimates_what_the_rows_estimate() {
        let x = data();
        let direct = covariance::covariance(x.as_ref(), None, &sample()).unwrap();

        let through = state(&x).finalise(&sample()).unwrap();

        for i in 0..2 {
            assert!((through.means[i] - direct.means[i]).abs() < 1e-12);
            for j in 0..2 {
                assert!((through.values[(i, j)] - direct.values[(i, j)]).abs() < 1e-12);
            }
        }
        assert_eq!(through.n_observations, 6);
    }

    #[test]
    fn merging_two_partitions_estimates_what_all_the_rows_estimate() {
        let x = data();
        let direct = covariance::covariance(x.as_ref(), None, &sample()).unwrap();

        let merged = state(&split(&x, &[0, 1, 2]))
            .merge(&state(&split(&x, &[3, 4, 5])))
            .unwrap()
            .finalise(&sample())
            .unwrap();

        for i in 0..2 {
            assert!((merged.means[i] - direct.means[i]).abs() < 1e-12);
            for j in 0..2 {
                assert!((merged.values[(i, j)] - direct.values[(i, j)]).abs() < 1e-12);
            }
        }
        assert_eq!(merged.n_observations, 6);
    }

    #[test]
    fn uneven_partitions_make_no_difference() {
        let x = data();
        let evenly = state(&split(&x, &[0, 1, 2]))
            .merge(&state(&split(&x, &[3, 4, 5])))
            .unwrap()
            .finalise(&sample())
            .unwrap();

        let unevenly = state(&split(&x, &[0]))
            .merge(&state(&split(&x, &[1, 2, 3, 4])))
            .unwrap()
            .merge(&state(&split(&x, &[5])))
            .unwrap()
            .finalise(&sample())
            .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((evenly.values[(i, j)] - unevenly.values[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn the_order_the_partitions_are_merged_in_does_not_matter() {
        let x = data();
        let parts: Vec<CovarianceState> = [&[0usize, 1][..], &[2, 3][..], &[4, 5][..]]
            .iter()
            .map(|rows| state(&split(&x, rows)))
            .collect();

        let forwards = parts[0]
            .merge(&parts[1])
            .unwrap()
            .merge(&parts[2])
            .unwrap()
            .finalise(&sample())
            .unwrap();
        let backwards = parts[2]
            .merge(&parts[1])
            .unwrap()
            .merge(&parts[0])
            .unwrap()
            .finalise(&sample())
            .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((forwards.values[(i, j)] - backwards.values[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn weights_are_carried_into_the_state() {
        let x = data();
        let w = Mat::from_fn(6, 1, |i, _| 1.0 + i as f64);
        let weights = Weights::new(w.as_ref(), "w").unwrap();
        let direct = covariance::covariance(x.as_ref(), Some(&weights), &sample()).unwrap();

        let through = CovarianceState::accumulate(x.as_ref(), Some(&weights), names())
            .unwrap()
            .finalise(&sample())
            .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((through.values[(i, j)] - direct.values[(i, j)]).abs() < 1e-12);
            }
        }
        assert_eq!(through.sum_weights, 21.0);
    }

    #[test]
    fn weighted_partitions_merge_into_the_weighted_whole() {
        let x = data();
        let w = Mat::from_fn(6, 1, |i, _| 1.0 + i as f64);
        let weights = Weights::new(w.as_ref(), "w").unwrap();
        let direct = covariance::covariance(x.as_ref(), Some(&weights), &sample()).unwrap();

        let part = |rows: &[usize]| {
            let values = split(&x, rows);
            let sub = Mat::from_fn(rows.len(), 1, |i, _| w[(rows[i], 0)]);
            let weights = Weights::new(sub.as_ref(), "w").unwrap();
            CovarianceState::accumulate(values.as_ref(), Some(&weights), names()).unwrap()
        };
        let merged = part(&[0, 1])
            .merge(&part(&[2, 3, 4]))
            .unwrap()
            .merge(&part(&[5]))
            .unwrap()
            .finalise(&sample())
            .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((merged.values[(i, j)] - direct.values[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn a_state_can_be_finalised_as_a_correlation() {
        let x = data();
        let options = covariance::Options {
            normalise: true,
            ..sample()
        };
        let direct = covariance::covariance(x.as_ref(), None, &options).unwrap();

        let through = state(&x).finalise(&options).unwrap();

        assert_eq!(through.values[(0, 0)], 1.0);
        assert!((through.values[(0, 1)] - direct.values[(0, 1)]).abs() < 1e-12);
    }

    #[test]
    fn a_state_survives_a_round_trip_through_its_bytes() {
        let original = state(&data());

        let restored = CovarianceState::decode(&original.encode()).unwrap();

        assert_eq!(restored.features, original.features);
        assert_eq!(restored.n_observations, original.n_observations);
        assert_eq!(restored.sum_weights, original.sum_weights);
        assert_eq!(restored.sum_squared_weights, original.sum_squared_weights);
        assert_eq!(restored.means, original.means);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    restored.cross_products[(i, j)],
                    original.cross_products[(i, j)]
                );
            }
        }
    }

    #[test]
    fn states_for_different_columns_refuse_to_merge() {
        let x = data();
        let other =
            CovarianceState::accumulate(x.as_ref(), None, vec!["a".to_string(), "c".to_string()])
                .unwrap();

        assert!(state(&x).merge(&other).is_err());
    }

    #[test]
    fn centred_moments_hold_up_when_the_values_dwarf_their_spread() {
        // The spread is a billionth of the values, which is where accumulating raw sums of
        // squares would lose most of its precision.
        let x = Mat::from_fn(6, 2, |i, j| 1e9 + data()[(i, j)]);
        let direct = covariance::covariance(x.as_ref(), None, &sample()).unwrap();

        let merged = CovarianceState::accumulate(split(&x, &[0, 1, 2]).as_ref(), None, names())
            .unwrap()
            .merge(
                &CovarianceState::accumulate(split(&x, &[3, 4, 5]).as_ref(), None, names())
                    .unwrap(),
            )
            .unwrap()
            .finalise(&sample())
            .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((merged.values[(i, j)] - direct.values[(i, j)]).abs() < 1e-6);
            }
        }
    }
}
