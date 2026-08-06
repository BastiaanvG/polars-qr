//! Hard-window state for a pair of columns.
//!
//! Both columns are held relative to an origin, for the reason set out in the univariate
//! state: a deviation taken from values that sit far from zero loses most of its digits
//! before the arithmetic starts.

/// The running state of a pair of columns over a window.
#[derive(Clone, Debug, Default)]
pub struct RollingBivariateState {
    count: u64,
    origin_x: f64,
    origin_y: f64,
    mean_x: f64,
    mean_y: f64,
    centred_m2_x: f64,
    centred_m2_y: f64,
    centred_cross: f64,
}

impl RollingBivariateState {
    /// Add one jointly valid pair.
    pub fn add(&mut self, x: f64, y: f64) {
        if self.count == 0 {
            self.origin_x = x;
            self.origin_y = y;
        }
        let x = x - self.origin_x;
        let y = y - self.origin_y;

        self.count += 1;
        let count = self.count as f64;

        let deviation_x = x - self.mean_x;
        let deviation_y = y - self.mean_y;
        self.mean_x += deviation_x / count;
        self.mean_y += deviation_y / count;
        self.centred_m2_x += deviation_x * (x - self.mean_x);
        self.centred_m2_y += deviation_y * (y - self.mean_y);
        // One deviation from before the shift, one from after: the cross moment stays
        // symmetric in the two columns whichever is called x.
        self.centred_cross += deviation_x * (y - self.mean_y);
    }

    /// Take one jointly valid pair back out.
    pub fn remove(&mut self, x: f64, y: f64) {
        let x = x - self.origin_x;
        let y = y - self.origin_y;

        self.count -= 1;
        if self.count == 0 {
            let (origin_x, origin_y) = (self.origin_x, self.origin_y);
            *self = Self::default();
            self.origin_x = origin_x;
            self.origin_y = origin_y;
            return;
        }

        let count = self.count as f64;
        let previous_x = self.mean_x;
        let previous_y = self.mean_y;
        self.mean_x -= (x - self.mean_x) / count;
        self.mean_y -= (y - self.mean_y) / count;
        self.centred_m2_x -= (x - previous_x) * (x - self.mean_x);
        self.centred_m2_y -= (y - previous_y) * (y - self.mean_y);
        self.centred_cross -= (x - previous_x) * (y - self.mean_y);
    }

    /// Rebuild the state from the pairs it should hold.
    pub fn rebuild(&mut self, pairs: impl Iterator<Item = (f64, f64)>) {
        *self = Self::default();
        for (x, y) in pairs {
            self.add(x, y);
        }
    }

    /// How many pairs are in the window.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The covariance over the window.
    pub fn covariance(&self, ddof: u32) -> Option<f64> {
        let denominator = self.count as f64 - f64::from(ddof);
        if denominator <= 0.0 {
            return None;
        }
        Some(self.centred_cross / denominator)
    }

    /// The correlation over the window.
    pub fn correlation(&self) -> Option<f64> {
        let spread_x = super::clamp_to_zero(self.centred_m2_x);
        let spread_y = super::clamp_to_zero(self.centred_m2_y);
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
        Some(if correlation.abs() <= 1.0 {
            correlation
        } else if correlation.abs() <= 1.0 + 1e-9 {
            correlation.signum()
        } else {
            correlation
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn over(pairs: &[(f64, f64)]) -> RollingBivariateState {
        let mut state = RollingBivariateState::default();
        for (x, y) in pairs {
            state.add(*x, *y);
        }
        state
    }

    #[test]
    fn measures_what_it_is_given() {
        // y = 2x, so the covariance is twice the variance of x.
        let state = over(&[(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]);

        assert_eq!(state.count(), 3);
        assert!((state.covariance(1).unwrap() - 2.0).abs() < 1e-12);
        assert!((state.correlation().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn removing_leaves_what_adding_alone_would_have_built() {
        let mut rolled = over(&[(9.0, -4.0), (1.0, 2.0), (2.0, 1.0), (3.0, 4.0)]);
        rolled.remove(9.0, -4.0);
        let fresh = over(&[(1.0, 2.0), (2.0, 1.0), (3.0, 4.0)]);

        assert_eq!(rolled.count(), fresh.count());
        assert!((rolled.covariance(1).unwrap() - fresh.covariance(1).unwrap()).abs() < 1e-12);
        assert!((rolled.correlation().unwrap() - fresh.correlation().unwrap()).abs() < 1e-12);
    }

    #[test]
    fn rolling_a_window_across_a_series_tracks_a_fresh_state() {
        let pairs: Vec<(f64, f64)> = (0..200)
            .map(|i| (i as f64 * 0.5, ((i % 5) as f64) - 2.0))
            .collect();
        let width = 4;

        let mut state = RollingBivariateState::default();
        for (x, y) in &pairs[..width] {
            state.add(*x, *y);
        }
        for index in width..pairs.len() {
            state.add(pairs[index].0, pairs[index].1);
            state.remove(pairs[index - width].0, pairs[index - width].1);

            let fresh = over(&pairs[index - width + 1..=index]);
            assert!((state.covariance(1).unwrap() - fresh.covariance(1).unwrap()).abs() < 1e-9);
        }
    }

    #[test]
    fn covariance_does_not_depend_on_which_column_is_which() {
        let one = over(&[(1.0, 5.0), (2.0, -1.0), (3.0, 4.0)]);
        let other = over(&[(5.0, 1.0), (-1.0, 2.0), (4.0, 3.0)]);

        assert!((one.covariance(1).unwrap() - other.covariance(1).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn a_column_that_does_not_move_has_no_correlation() {
        let state = over(&[(1.0, 7.0), (2.0, 7.0), (3.0, 7.0)]);

        assert_eq!(state.correlation(), None);
    }

    #[test]
    fn emptying_the_window_clears_it() {
        let mut state = over(&[(1.0, 2.0)]);
        state.remove(1.0, 2.0);

        assert_eq!(state.count(), 0);
        assert_eq!(state.covariance(1), None);
        assert_eq!(state.correlation(), None);
    }
}
