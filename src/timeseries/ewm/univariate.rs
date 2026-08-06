//! Exponentially weighted state for one column.
//!
//! Every valid observation enters with weight one and every weight already held is
//! multiplied by the decay of the clock distance just travelled. The weights are therefore
//! normalised by their own sum rather than assumed to add to one, which is what lets an
//! observation arriving at zero clock distance count in full instead of vanishing.

/// The running state of one column under exponential decay.
#[derive(Clone, Debug, Default)]
pub struct EwmUnivariateState {
    /// How many valid observations have been seen.
    valid_count: u64,
    /// The sum of the weights currently held.
    weight_sum: f64,
    /// The sum of their squares, which the bias correction needs.
    weight_squared_sum: f64,
    /// The decaying sum of the values themselves.
    weighted_sum: f64,
    /// The weighted mean.
    mean: f64,
    /// The weighted sum of squared deviations from that mean.
    centred_m2: f64,
}

impl EwmUnivariateState {
    /// Age everything held by one step of decay.
    ///
    /// The mean is a ratio of two quantities that decay together, so decay leaves it alone.
    pub fn decay(&mut self, lambda: f64) {
        self.weight_sum *= lambda;
        self.weight_squared_sum *= lambda * lambda;
        self.weighted_sum *= lambda;
        self.centred_m2 *= lambda;
    }

    /// Add one observation, with weight one.
    pub fn add(&mut self, value: f64) {
        self.valid_count += 1;
        self.weighted_sum += value;

        let weight_sum = self.weight_sum + 1.0;
        let deviation = value - self.mean;
        let shift = deviation / weight_sum;
        self.mean += shift;
        // The deviation before the shift times the one after it: the weighted form of the
        // usual online update, and stable in the way summing squares directly is not.
        self.centred_m2 += self.weight_sum * deviation * shift;
        self.weight_sum = weight_sum;
        self.weight_squared_sum += 1.0;
    }

    /// How many valid observations have entered the state.
    pub fn valid_count(&self) -> u64 {
        self.valid_count
    }

    /// The decaying sum.
    pub fn sum(&self) -> f64 {
        self.weighted_sum
    }

    /// The weighted mean.
    pub fn mean(&self) -> Option<f64> {
        (self.weight_sum > 0.0).then_some(self.mean)
    }

    /// The weighted variance.
    pub fn variance(&self, bias: bool) -> Option<f64> {
        let denominator = self.variance_denominator(bias)?;
        Some(clamp_to_zero(self.centred_m2) / denominator)
    }

    /// The divisor a weighted variance is taken over.
    ///
    /// Unbiased weights lose the part of the spread that estimating the mean from the same
    /// observations already accounted for. With every weight equal to one this is the usual
    /// `n - 1`.
    fn variance_denominator(&self, bias: bool) -> Option<f64> {
        let denominator = if bias {
            self.weight_sum
        } else {
            self.weight_sum - self.weight_squared_sum / self.weight_sum
        };
        (denominator > 0.0 && denominator.is_finite()).then_some(denominator)
    }
}

/// Round a centred second moment that rounding has pushed just below zero back up to it.
///
/// A moment that is materially negative would be an error in the update itself, and is left
/// alone so that it shows up rather than being covered over.
pub fn clamp_to_zero(moment: f64) -> f64 {
    if moment < 0.0 && moment > -1e-12 {
        0.0
    } else {
        moment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_observation_has_no_spread() {
        let mut state = EwmUnivariateState::default();
        state.add(3.0);

        assert_eq!(state.valid_count(), 1);
        assert_eq!(state.sum(), 3.0);
        assert_eq!(state.mean(), Some(3.0));
        // One observation cannot say anything about spread once the mean is taken from it.
        assert_eq!(state.variance(false), None);
        assert_eq!(state.variance(true), Some(0.0));
    }

    #[test]
    fn without_decay_it_is_the_ordinary_mean_and_variance() {
        let mut state = EwmUnivariateState::default();
        for value in [1.0, 2.0, 3.0, 4.0] {
            state.decay(1.0);
            state.add(value);
        }

        assert_eq!(state.sum(), 10.0);
        assert_eq!(state.mean(), Some(2.5));
        // Squared deviations 2.25 + 0.25 + 0.25 + 2.25 = 5, over n - 1.
        assert!((state.variance(false).unwrap() - 5.0 / 3.0).abs() < 1e-12);
        assert!((state.variance(true).unwrap() - 1.25).abs() < 1e-12);
    }

    #[test]
    fn decay_halves_the_weight_of_what_came_before() {
        let mut state = EwmUnivariateState::default();
        state.add(1.0);
        state.decay(0.5);
        state.add(3.0);

        // Weights 0.5 and 1: the mean leans towards the newer observation.
        assert_eq!(state.sum(), 3.5);
        assert!((state.mean().unwrap() - 7.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn decay_leaves_the_mean_where_it_was() {
        let mut state = EwmUnivariateState::default();
        state.add(2.0);
        state.add(4.0);
        let before = state.mean().unwrap();

        state.decay(0.25);

        assert_eq!(state.mean().unwrap(), before);
    }

    #[test]
    fn a_full_decay_forgets_everything_but_the_count() {
        let mut state = EwmUnivariateState::default();
        state.add(5.0);
        state.decay(0.0);
        state.add(1.0);

        assert_eq!(state.sum(), 1.0);
        assert_eq!(state.mean(), Some(1.0));
        assert_eq!(state.valid_count(), 2);
    }

    #[test]
    fn the_sum_follows_its_own_recursion() {
        let mut state = EwmUnivariateState::default();
        let mut expected = 0.0;
        for (value, lambda) in [(1.0, 1.0), (2.0, 0.5), (3.0, 0.25)] {
            state.decay(lambda);
            state.add(value);
            expected = value + lambda * expected;
        }

        assert!((state.sum() - expected).abs() < 1e-12);
    }

    #[test]
    fn the_mean_is_the_sum_over_the_weight() {
        let mut state = EwmUnivariateState::default();
        for (value, lambda) in [(1.5, 1.0), (-2.0, 0.8), (4.0, 0.3)] {
            state.decay(lambda);
            state.add(value);
        }

        assert!((state.mean().unwrap() - state.sum() / state.weight_sum).abs() < 1e-12);
    }
}
