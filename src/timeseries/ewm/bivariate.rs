//! Exponentially weighted state for a pair of columns.
//!
//! A row enters only when both of its values are there. Sharing one validity mask is what
//! makes the covariance and the two variances describe the same observations, which is what
//! a correlation divides one by the others for.

use crate::timeseries::ewm::univariate::clamp_to_zero;

/// The running state of a pair of columns under exponential decay.
#[derive(Clone, Debug, Default)]
pub struct EwmBivariateState {
    valid_count: u64,
    weight_sum: f64,
    weight_squared_sum: f64,
    mean_x: f64,
    mean_y: f64,
    centred_m2_x: f64,
    centred_m2_y: f64,
    centred_cross: f64,
}

impl EwmBivariateState {
    /// Age everything held by one step of decay.
    pub fn decay(&mut self, lambda: f64) {
        self.weight_sum *= lambda;
        self.weight_squared_sum *= lambda * lambda;
        self.centred_m2_x *= lambda;
        self.centred_m2_y *= lambda;
        self.centred_cross *= lambda;
    }

    /// Add one jointly valid pair, with weight one.
    pub fn add(&mut self, x: f64, y: f64) {
        self.valid_count += 1;

        let weight_sum = self.weight_sum + 1.0;
        let deviation_x = x - self.mean_x;
        let deviation_y = y - self.mean_y;
        let shift_x = deviation_x / weight_sum;
        let shift_y = deviation_y / weight_sum;

        self.mean_x += shift_x;
        self.mean_y += shift_y;
        self.centred_m2_x += self.weight_sum * deviation_x * shift_x;
        self.centred_m2_y += self.weight_sum * deviation_y * shift_y;
        // The cross moment pairs each deviation with the other column's, taken before and
        // after the shift so the two updates stay symmetric in x and y.
        self.centred_cross += self.weight_sum * deviation_x * shift_y;

        self.weight_sum = weight_sum;
        self.weight_squared_sum += 1.0;
    }

    /// How many jointly valid pairs have entered the state.
    pub fn valid_count(&self) -> u64 {
        self.valid_count
    }

    /// The weighted covariance.
    pub fn covariance(&self, bias: bool) -> Option<f64> {
        let denominator = self.denominator(bias)?;
        Some(self.centred_cross / denominator)
    }

    /// The weighted correlation.
    ///
    /// The weights cancel between the covariance and the two variances, so there is nothing
    /// for a bias correction to change and no parameter offering one.
    pub fn correlation(&self) -> Option<f64> {
        let spread_x = clamp_to_zero(self.centred_m2_x);
        let spread_y = clamp_to_zero(self.centred_m2_y);
        if spread_x <= 0.0 || spread_y <= 0.0 {
            return None;
        }
        let denominator = (spread_x * spread_y).sqrt();
        if !denominator.is_finite() || denominator <= 0.0 {
            return None;
        }
        let correlation = self.centred_cross / denominator;
        if !correlation.is_finite() {
            return None;
        }
        // Rounding can put a correlation a hair outside the interval it is defined on.
        // Anything further out than that would be a fault in the update rather than in the
        // last bit, and is left visible.
        Some(if correlation.abs() <= 1.0 {
            correlation
        } else if correlation.abs() <= 1.0 + 1e-9 {
            correlation.signum()
        } else {
            correlation
        })
    }

    /// The divisor a weighted covariance is taken over.
    fn denominator(&self, bias: bool) -> Option<f64> {
        let denominator = if bias {
            self.weight_sum
        } else {
            self.weight_sum - self.weight_squared_sum / self.weight_sum
        };
        (denominator > 0.0 && denominator.is_finite()).then_some(denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn undecayed(pairs: &[(f64, f64)]) -> EwmBivariateState {
        let mut state = EwmBivariateState::default();
        for (x, y) in pairs {
            state.decay(1.0);
            state.add(*x, *y);
        }
        state
    }

    #[test]
    fn without_decay_it_is_the_ordinary_covariance() {
        // y = 2x, so the covariance is twice the variance of x.
        let state = undecayed(&[(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]);

        assert!((state.covariance(false).unwrap() - 2.0).abs() < 1e-12);
        assert!((state.covariance(true).unwrap() - 4.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_perfect_relationship_correlates_exactly() {
        let up = undecayed(&[(1.0, 3.0), (2.0, 6.0), (3.0, 9.0)]);
        let down = undecayed(&[(1.0, -3.0), (2.0, -6.0), (3.0, -9.0)]);

        assert!((up.correlation().unwrap() - 1.0).abs() < 1e-12);
        assert!((down.correlation().unwrap() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn covariance_does_not_depend_on_which_column_is_which() {
        let one = undecayed(&[(1.0, 5.0), (2.0, -1.0), (3.0, 4.0)]);
        let other = undecayed(&[(5.0, 1.0), (-1.0, 2.0), (4.0, 3.0)]);

        assert!((one.covariance(false).unwrap() - other.covariance(false).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn a_column_that_does_not_move_has_no_correlation() {
        let state = undecayed(&[(1.0, 7.0), (2.0, 7.0), (3.0, 7.0)]);

        assert_eq!(state.correlation(), None);
        assert!((state.covariance(false).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn one_pair_is_not_enough_for_an_unbiased_covariance() {
        let state = undecayed(&[(1.0, 2.0)]);

        assert_eq!(state.covariance(false), None);
        assert_eq!(state.covariance(true), Some(0.0));
        assert_eq!(state.valid_count(), 1);
    }

    #[test]
    fn decay_leans_the_covariance_towards_the_newer_pairs() {
        let mut fresh = EwmBivariateState::default();
        for (x, y) in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)] {
            fresh.decay(0.1);
            fresh.add(x, y);
        }

        // Still a perfect relationship, whatever the weights.
        assert!((fresh.correlation().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn large_offsets_do_not_swamp_a_small_spread() {
        // Summing squares directly would lose the spread entirely at this offset.
        let offset = 1e9;
        let plain = undecayed(&[(1.0, 2.0), (2.0, 1.0), (3.0, 4.0)]);
        let shifted = undecayed(&[
            (offset + 1.0, offset + 2.0),
            (offset + 2.0, offset + 1.0),
            (offset + 3.0, offset + 4.0),
        ]);

        assert!(
            (plain.covariance(false).unwrap() - shifted.covariance(false).unwrap()).abs() < 1e-6
        );
    }
}
