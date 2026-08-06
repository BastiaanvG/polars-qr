//! Hard-window state for one column.
//!
//! An observation joins the window when the clock reaches it and leaves when the clock has
//! travelled a window's width past it, so the state has to be able to take a value back out
//! again.
//!
//! Everything is held relative to an origin, taken from the first observation to enter an
//! empty window. Without it a deviation is the difference of two numbers that are nearly
//! equal: at a level of a billion with a spread of one, `value - mean` agrees in ten digits
//! before it disagrees in any, and only the last six carry the answer. Subtracting the
//! origin first makes both sides small, so the deviation keeps every digit it has. The
//! shift changes nothing else: a variance does not know where its values sit.

/// The running state of one column over a window.
#[derive(Clone, Debug, Default)]
pub struct RollingUnivariateState {
    count: u64,
    origin: f64,
    /// The sum of the values, less the origin: added back when the sum is asked for.
    shifted_sum: f64,
    /// The mean of the values, less the origin.
    shifted_mean: f64,
    centred_m2: f64,
}

impl RollingUnivariateState {
    /// Add one observation.
    pub fn add(&mut self, value: f64) {
        if self.count == 0 {
            self.origin = value;
        }
        let value = value - self.origin;

        self.count += 1;
        self.shifted_sum += value;

        let deviation = value - self.shifted_mean;
        self.shifted_mean += deviation / self.count as f64;
        self.centred_m2 += deviation * (value - self.shifted_mean);
    }

    /// Take one observation back out.
    pub fn remove(&mut self, value: f64) {
        let value = value - self.origin;

        self.count -= 1;
        if self.count == 0 {
            let origin = self.origin;
            *self = Self::default();
            self.origin = origin;
            return;
        }

        self.shifted_sum -= value;
        let previous_mean = self.shifted_mean;
        self.shifted_mean -= (value - self.shifted_mean) / self.count as f64;
        self.centred_m2 -= (value - previous_mean) * (value - self.shifted_mean);
    }

    /// Rebuild the state from the observations it should hold.
    ///
    /// Adding and removing in turn leaves a little rounding behind each time. Recomputing
    /// from what is still in the window clears it, and picks a fresh origin along the way,
    /// which matters when the level of the data has wandered away from the old one.
    pub fn rebuild(&mut self, values: impl Iterator<Item = f64>) {
        *self = Self::default();
        for value in values {
            self.add(value);
        }
    }

    /// How many observations are in the window.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The sum over the window.
    pub fn sum(&self) -> f64 {
        self.shifted_sum + self.count as f64 * self.origin
    }

    /// The mean over the window.
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.origin + self.shifted_mean)
    }

    /// The variance over the window.
    pub fn variance(&self, ddof: u32) -> Option<f64> {
        let denominator = self.count as f64 - f64::from(ddof);
        if denominator <= 0.0 {
            return None;
        }
        Some(super::clamp_to_zero(self.centred_m2) / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn over(values: &[f64]) -> RollingUnivariateState {
        let mut state = RollingUnivariateState::default();
        for value in values {
            state.add(*value);
        }
        state
    }

    #[test]
    fn adds_up_what_it_is_given() {
        let state = over(&[1.0, 2.0, 3.0, 4.0]);

        assert_eq!(state.count(), 4);
        assert_eq!(state.sum(), 10.0);
        assert_eq!(state.mean(), Some(2.5));
        assert!((state.variance(1).unwrap() - 5.0 / 3.0).abs() < 1e-12);
        assert!((state.variance(0).unwrap() - 1.25).abs() < 1e-12);
    }

    #[test]
    fn removing_leaves_what_adding_alone_would_have_built() {
        let mut rolled = over(&[10.0, 1.0, 2.0, 3.0]);
        rolled.remove(10.0);
        let fresh = over(&[1.0, 2.0, 3.0]);

        assert_eq!(rolled.count(), fresh.count());
        assert!((rolled.sum() - fresh.sum()).abs() < 1e-12);
        assert!((rolled.mean().unwrap() - fresh.mean().unwrap()).abs() < 1e-12);
        assert!((rolled.variance(1).unwrap() - fresh.variance(1).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn emptying_the_window_clears_it() {
        let mut state = over(&[1.0, 2.0]);
        state.remove(1.0);
        state.remove(2.0);

        assert_eq!(state.count(), 0);
        assert_eq!(state.sum(), 0.0);
        assert_eq!(state.mean(), None);
        assert_eq!(state.variance(1), None);
    }

    #[test]
    fn one_observation_has_no_sample_variance() {
        let state = over(&[3.0]);

        assert_eq!(state.variance(1), None);
        assert_eq!(state.variance(0), Some(0.0));
    }

    #[test]
    fn a_window_that_does_not_move_has_no_variance() {
        let state = over(&[7.0, 7.0, 7.0]);

        assert!(state.variance(1).unwrap().abs() < 1e-12);
        assert!(state.variance(1).unwrap() >= 0.0);
    }

    #[test]
    fn a_long_run_of_updates_does_not_drift() {
        let values: Vec<f64> = (0..1000).map(|i| 1e6 + (i % 7) as f64).collect();
        let mut state = RollingUnivariateState::default();
        for value in &values[..3] {
            state.add(*value);
        }
        for index in 3..values.len() {
            state.add(values[index]);
            state.remove(values[index - 3]);
        }

        let fresh = over(&values[values.len() - 3..]);
        assert!((state.mean().unwrap() - fresh.mean().unwrap()).abs() < 1e-9);
        assert!((state.variance(1).unwrap() - fresh.variance(1).unwrap()).abs() < 1e-9);
    }

    #[test]
    fn a_large_offset_does_not_cost_the_spread_its_digits() {
        // The values sit a billion away from zero and vary by one. Held as they are, the
        // deviations would lose ten of their sixteen digits to the offset.
        let offset = 1e9;
        let sample = [1.0, 2.5, -0.5, 3.25, 0.75];
        let plain = over(&sample);
        let shifted = over(&sample.map(|value| value + offset));

        assert!((shifted.variance(1).unwrap() - plain.variance(1).unwrap()).abs() < 1e-12);
        assert!((shifted.mean().unwrap() - plain.mean().unwrap() - offset).abs() < 1e-6);
    }

    #[test]
    fn rebuilding_matches_building() {
        let mut state = over(&[5.0, 1.0, 9.0]);
        state.rebuild([1.0, 2.0, 3.0].into_iter());
        let fresh = over(&[1.0, 2.0, 3.0]);

        assert_eq!(state.count(), fresh.count());
        assert!((state.variance(1).unwrap() - fresh.variance(1).unwrap()).abs() < 1e-12);
    }
}
