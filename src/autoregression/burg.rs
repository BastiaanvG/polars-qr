//! Burg's estimator.
//!
//! Yule–Walker builds an autocovariance sequence first and fits whatever that sequence
//! implies, which on a short series means fitting an estimate of an estimate. Burg goes
//! straight at the thing being minimised — the forward and backward prediction errors,
//! together — and climbs the orders on the same recursion, so it keeps the guarantee that
//! matters: every reflection coefficient it produces is inside the unit circle, and the fit
//! is therefore stationary.

use polars::prelude::*;

use crate::autoregression::levinson::{extend, Recursion};

/// Fit every order up to `max_order` by minimising the forward and backward errors.
pub fn burg(centred: &[f64], max_order: usize, operation: &str) -> PolarsResult<Recursion> {
    let n = centred.len();
    let mut reflection = vec![0.0; max_order + 1];
    let mut variance = vec![0.0; max_order + 1];
    variance[0] = centred.iter().map(|value| value * value).sum::<f64>() / n as f64;
    if variance[0] <= 0.0 {
        polars_bail!(
            ComputeError:
            "{} received a series with variance {}.\n\nA series that does not move has no \
             autocorrelation to model.",
            operation, variance[0],
        );
    }

    // What the fit of the current order leaves unexplained, looking forwards and backwards
    // from every row it reaches.
    let mut forward = centred.to_vec();
    let mut backward = centred.to_vec();
    let mut prediction: Vec<f64> = Vec::with_capacity(max_order);

    for order in 1..=max_order {
        let mut cross = 0.0;
        let mut squares = 0.0;
        for row in order..n {
            cross += forward[row] * backward[row - 1];
            squares += forward[row] * forward[row] + backward[row - 1] * backward[row - 1];
        }
        if squares <= 0.0 {
            polars_bail!(
                ComputeError:
                "{} drove the prediction error to zero at order {}.\n\nThe series is exactly \
                 determined by that many lags, so no higher order can be fitted. Ask for an \
                 order of at most {}.",
                operation, order - 1, order - 1,
            );
        }

        // Twice the cross term over the sum of both squares: the ratio that minimises the
        // two errors at once. It cannot leave the unit interval, which is where the
        // stationarity guarantee comes from.
        let step = 2.0 * cross / squares;
        for row in (order..n).rev() {
            let (ahead, behind) = (forward[row], backward[row - 1]);
            forward[row] = ahead - step * behind;
            backward[row] = behind - step * ahead;
        }

        extend(&mut prediction, step);
        reflection[order] = step;
        variance[order] = (variance[order - 1] * (1.0 - step * step)).max(0.0);
    }

    Ok(Recursion {
        reflection,
        variance,
    })
}

#[cfg(test)]
mod tests {
    use crate::autoregression::levinson::coefficients;
    use crate::autoregression::moments::autocovariance;

    use super::*;

    /// A deterministic AR(1) path: enough structure to fit, no randomness to explain.
    fn ar1(phi: f64, n: usize) -> Vec<f64> {
        let mut series = Vec::with_capacity(n);
        let mut value = 1.0;
        for row in 0..n {
            // A shock that changes sign keeps the series from decaying to nothing.
            value = phi * value + if row % 7 == 0 { 1.0 } else { -0.15 };
            series.push(value);
        }
        let mean = series.iter().sum::<f64>() / n as f64;
        series.iter().map(|value| value - mean).collect()
    }

    #[test]
    fn a_reflection_coefficient_stays_inside_the_unit_circle() {
        let recursion = burg(&ar1(0.6, 200), 12, "").unwrap();

        assert!(recursion.reflection[1..].iter().all(|k| k.abs() < 1.0));
    }

    #[test]
    fn the_prediction_error_falls_with_every_order() {
        let recursion = burg(&ar1(0.6, 200), 8, "").unwrap();

        for order in 1..recursion.variance.len() {
            assert!(recursion.variance[order] <= recursion.variance[order - 1]);
        }
    }

    #[test]
    fn it_agrees_with_yule_walker_on_a_long_series() {
        let series = ar1(0.6, 4000);

        let ours = coefficients(&burg(&series, 2, "").unwrap().reflection, 2);
        let sequence = autocovariance(&series, 2, false);
        let theirs = coefficients(
            &crate::autoregression::levinson::durbin(&sequence, "")
                .unwrap()
                .reflection,
            2,
        );

        for (ours, theirs) in ours.iter().zip(&theirs) {
            assert!((ours - theirs).abs() < 1e-2, "{ours} vs {theirs}");
        }
    }

    #[test]
    fn the_sign_of_the_first_coefficient_follows_the_series() {
        let rising = ar1(0.8, 500);
        let alternating: Vec<f64> = (0..500)
            .map(|row| if row % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        assert!(burg(&rising, 1, "").unwrap().reflection[1] > 0.0);
        assert!(burg(&alternating, 1, "").unwrap().reflection[1] < 0.0);
    }

    #[test]
    fn a_series_with_no_spread_is_rejected() {
        assert!(burg(&[0.0, 0.0, 0.0], 1, "ar").is_err());
    }
}
