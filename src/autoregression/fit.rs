//! Fitting an autoregression, and applying the fit to the rows it came from.

use polars::prelude::*;

use crate::autoregression::burg::burg;
use crate::autoregression::input::Observations;
use crate::autoregression::levinson::{coefficients, durbin, Recursion};
use crate::autoregression::moments::autocovariance;
use crate::autoregression::options::{Method, Options, Output};

/// A fitted autoregression, and everything the recursion saw on the way to it.
pub struct Autoregression {
    /// The coefficients of the chosen order, lag one first.
    pub coefficients: Vec<f64>,
    /// What was subtracted before fitting.
    pub mean: f64,
    /// The prediction-error variance of the chosen order.
    pub variance: f64,
    /// The order that was fitted.
    pub order: usize,
    /// The reflection coefficient at each order from one upwards.
    pub partial_autocorrelations: Vec<f64>,
    /// The prediction-error variance at each order, index 0 being the series itself.
    pub order_variance: Vec<f64>,
    /// The selection criterion at each order, when an order was chosen rather than given.
    pub criterion: Option<Vec<f64>>,
    /// How many rows the fit read.
    pub n_observations: usize,
    /// Whether every reflection coefficient came out inside the unit circle.
    pub stationary: bool,
    /// Which estimator produced it.
    pub method: Method,
}

/// Fit an autoregression to a series.
pub fn fit(
    observations: &Observations,
    options: &Options,
    operation: &str,
) -> PolarsResult<Autoregression> {
    let n = observations.len();
    let recursion = match options.method {
        Method::YuleWalker => {
            let sequence = autocovariance(observations.centred(), options.max_order, false);
            durbin(&sequence, operation)?
        }
        Method::Burg => burg(observations.centred(), options.max_order, operation)?,
    };

    // A criterion the caller overrode with an order of their own is not reported, because
    // there would be nothing to read from it: the fit did not come from its argument.
    let criterion = match (options.order, options.criterion) {
        (None, Some(criterion)) => Some(
            (0..=options.max_order)
                .map(|order| {
                    n as f64 * recursion.variance[order].ln() + criterion.penalty(order, n)
                })
                .collect::<Vec<f64>>(),
        ),
        _ => None,
    };
    let order = match (options.order, &criterion) {
        (Some(order), _) => order,
        (None, Some(values)) => argmin(values),
        (None, None) => options.max_order,
    };

    Ok(Autoregression {
        coefficients: coefficients(&recursion.reflection, order),
        mean: observations.mean(),
        variance: recursion.variance[order],
        order,
        stationary: is_stationary(&recursion, order),
        partial_autocorrelations: recursion.reflection[1..].to_vec(),
        order_variance: recursion.variance,
        criterion,
        n_observations: n,
        method: options.method,
    })
}

/// Where the smallest value sits, the earliest one winning a tie.
///
/// The earliest is the smallest order, so a criterion that cannot separate two orders
/// takes the simpler of them.
fn argmin(values: &[f64]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if value < &values[best] {
            best = index;
        }
    }
    best
}

/// Whether the fit of `order` describes a series that stays where it is.
///
/// Read off the reflection coefficients rather than from the roots of the characteristic
/// polynomial. It is cheaper, it is exactly what the recursion already produced, and it
/// cannot be got backwards: the polynomial's roots must lie outside the unit circle, but
/// the reversed polynomial that most root-finders are given has the reciprocals, which must
/// lie inside. Testing one form against the other's boundary inverts the answer.
fn is_stationary(recursion: &Recursion, order: usize) -> bool {
    recursion.reflection[1..=order]
        .iter()
        .all(|reflection| reflection.abs() < 1.0)
}

/// Apply a fit to the rows it was fitted on.
///
/// In-sample by construction: the filter is the one the same rows produced. That is the
/// honest thing for a diagnostic and the wrong thing for a forecast.
pub fn apply(
    fit: &Autoregression,
    observations: &Observations,
    output: Output,
) -> Vec<Option<f64>> {
    let centred = observations.centred();
    let valid = observations.valid();
    let order = fit.order;

    (0..observations.len())
        .map(|row| {
            if row < order || !valid[row] || !valid[row - order..row].iter().all(|kept| *kept) {
                return None;
            }
            let prediction: f64 = fit
                .coefficients
                .iter()
                .enumerate()
                .map(|(lag, coefficient)| coefficient * centred[row - lag - 1])
                .sum();
            Some(match output {
                Output::Prediction => fit.mean + prediction,
                Output::Residual => centred[row] - prediction,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::autoregression::options::{Criterion, NullPolicy};

    use super::*;

    fn observations(values: &[f64]) -> Observations {
        let series = Series::new("y".into(), values);
        Observations::from_series(&series, NullPolicy::Raise, true, "").unwrap()
    }

    /// A deterministic AR(1) path with a repeating shock.
    fn ar1(phi: f64, n: usize) -> Vec<f64> {
        let mut series = Vec::with_capacity(n);
        let mut value = 1.0;
        for row in 0..n {
            value = phi * value + if row % 7 == 0 { 1.0 } else { -0.15 };
            series.push(value);
        }
        series
    }

    fn options(order: Option<usize>, criterion: Option<Criterion>, max_order: usize) -> Options {
        Options {
            order,
            criterion,
            max_order,
            method: Method::YuleWalker,
        }
    }

    #[test]
    fn a_fixed_order_reports_that_many_coefficients() {
        let read = observations(&ar1(0.6, 200));

        let fitted = fit(&read, &options(Some(3), None, 3), "").unwrap();

        assert_eq!(fitted.order, 3);
        assert_eq!(fitted.coefficients.len(), 3);
        assert!(fitted.criterion.is_none());
        assert_eq!(fitted.n_observations, 200);
    }

    #[test]
    fn a_criterion_reports_one_value_per_order_it_considered() {
        let read = observations(&ar1(0.6, 200));

        let fitted = fit(&read, &options(None, Some(Criterion::Bic), 8), "").unwrap();

        let criterion = fitted.criterion.unwrap();
        assert_eq!(criterion.len(), 9);
        assert_eq!(fitted.order_variance.len(), 9);
        assert_eq!(fitted.partial_autocorrelations.len(), 8);
    }

    #[test]
    fn the_chosen_order_is_where_the_criterion_is_smallest() {
        let read = observations(&ar1(0.6, 200));

        let fitted = fit(&read, &options(None, Some(Criterion::Aic), 8), "").unwrap();

        let criterion = fitted.criterion.clone().unwrap();
        assert_eq!(fitted.order, argmin(&criterion));
        assert_eq!(fitted.coefficients.len(), fitted.order);
    }

    #[test]
    fn a_criterion_is_ignored_when_an_order_is_given() {
        let read = observations(&ar1(0.6, 200));

        let fitted = fit(&read, &options(Some(2), Some(Criterion::Bic), 6), "").unwrap();

        assert_eq!(fitted.order, 2);
        assert!(fitted.criterion.is_none());
    }

    #[test]
    fn a_yule_walker_fit_comes_out_stationary() {
        let read = observations(&ar1(0.95, 500));

        let fitted = fit(&read, &options(Some(10), None, 10), "").unwrap();

        assert!(fitted.stationary);
    }

    #[test]
    fn the_residual_of_the_fit_is_what_the_filter_left_behind() {
        let read = observations(&ar1(0.6, 100));
        let fitted = fit(&read, &options(Some(2), None, 2), "").unwrap();

        let residual = apply(&fitted, &read, Output::Residual);
        let prediction = apply(&fitted, &read, Output::Prediction);

        assert!(residual[0].is_none() && residual[1].is_none());
        for row in 2..100 {
            let value = read.centred()[row] + fitted.mean;
            assert!((residual[row].unwrap() - (value - prediction[row].unwrap())).abs() < 1e-12);
        }
    }

    #[test]
    fn the_residual_is_smaller_than_what_it_was_taken_from() {
        let read = observations(&ar1(0.9, 300));
        let fitted = fit(&read, &options(Some(1), None, 1), "").unwrap();

        let residual = apply(&fitted, &read, Output::Residual);

        let before: f64 = read.centred()[1..].iter().map(|value| value * value).sum();
        let after: f64 = residual.iter().flatten().map(|value| value * value).sum();
        assert!(after < before);
    }

    #[test]
    fn order_zero_predicts_the_mean_and_leaves_the_deviation() {
        let read = observations(&ar1(0.6, 50));
        let fitted = fit(&read, &options(Some(0), None, 0), "").unwrap();

        let prediction = apply(&fitted, &read, Output::Prediction);
        let residual = apply(&fitted, &read, Output::Residual);

        assert!(fitted.coefficients.is_empty());
        assert_eq!(prediction[0], Some(fitted.mean));
        assert!((residual[0].unwrap() - read.centred()[0]).abs() < 1e-12);
    }

    #[test]
    fn a_row_whose_history_is_missing_has_no_answer() {
        let series = Series::new("y".into(), &[Some(1.0), None, Some(3.0), Some(2.0)]);
        let read = Observations::from_series(&series, NullPolicy::Zero, true, "").unwrap();
        let fitted = Autoregression {
            coefficients: vec![0.5],
            mean: read.mean(),
            variance: 1.0,
            order: 1,
            partial_autocorrelations: vec![0.5],
            order_variance: vec![1.0, 1.0],
            criterion: None,
            n_observations: 4,
            stationary: true,
            method: Method::YuleWalker,
        };

        let residual = apply(&fitted, &read, Output::Residual);

        assert_eq!(residual[0], None, "no history at all");
        assert_eq!(residual[1], None, "the row itself is missing");
        assert_eq!(residual[2], None, "the lag it needs is missing");
        assert!(residual[3].is_some());
    }

    #[test]
    fn the_earliest_order_wins_a_tie() {
        assert_eq!(argmin(&[1.0, 1.0, 2.0]), 0);
        assert_eq!(argmin(&[2.0, 1.0, 1.0]), 1);
    }
}
