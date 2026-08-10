//! The autocovariance sequence of a series against its own lags.

use polars::prelude::*;

/// Estimate the autocovariance at every lag from zero up to `max_lag`.
///
/// The divisor is the length of the series and not the number of products at that lag. The
/// unbiased choice can produce a sequence that is not positive semi-definite, and such a
/// sequence has no valid Levinson–Durbin recursion and can yield an explosive fit. It is
/// offered for reading the sequence itself, and never used to build a model.
pub fn autocovariance(centred: &[f64], max_lag: usize, unbiased: bool) -> Vec<f64> {
    let n = centred.len();
    (0..=max_lag)
        .map(|lag| {
            let cross: f64 = centred[lag..]
                .iter()
                .zip(centred)
                .map(|(later, earlier)| later * earlier)
                .sum();
            let divisor = if unbiased { n - lag } else { n };
            cross / divisor as f64
        })
        .collect()
}

/// Normalise an autocovariance sequence by its value at lag zero.
pub fn normalise(sequence: &[f64], operation: &str) -> PolarsResult<Vec<f64>> {
    let variance = sequence[0];
    if variance <= 0.0 {
        polars_bail!(
            ComputeError:
            "{} cannot normalise a series that does not vary: its variance is {}.",
            operation, variance,
        );
    }
    Ok(sequence.iter().map(|value| value / variance).collect())
}

/// Check that a lag can be estimated at all from the rows there are.
pub fn check_max_lag(operation: &str, max_lag: usize, n: usize) -> PolarsResult<()> {
    if max_lag >= n {
        polars_bail!(
            ComputeError:
            "{} was asked for lag {} from {} observations.\n\nNo pair of observations is that \
             far apart, so the lag cannot be estimated. Ask for a lag below the number of rows \
             in the group.",
            operation, max_lag, n,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lag_zero_is_the_variance_about_the_mean() {
        let centred = [-1.5, -0.5, 0.5, 1.5];

        let sequence = autocovariance(&centred, 0, false);

        assert!((sequence[0] - 1.25).abs() < 1e-12);
    }

    #[test]
    fn the_biased_divisor_is_the_length_at_every_lag() {
        let centred = [1.0, 1.0, 1.0, 1.0];

        let sequence = autocovariance(&centred, 2, false);

        assert!((sequence[1] - 3.0 / 4.0).abs() < 1e-12);
        assert!((sequence[2] - 2.0 / 4.0).abs() < 1e-12);
    }

    #[test]
    fn the_unbiased_divisor_counts_the_products_it_has() {
        let centred = [1.0, 1.0, 1.0, 1.0];

        let sequence = autocovariance(&centred, 2, true);

        assert!((sequence[1] - 1.0).abs() < 1e-12);
        assert!((sequence[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_alternating_series_is_negatively_correlated_at_lag_one() {
        let centred = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

        let sequence = normalise(&autocovariance(&centred, 2, false), "").unwrap();

        assert_eq!(sequence[0], 1.0);
        assert!(sequence[1] < 0.0);
        assert!(sequence[2] > 0.0);
    }

    #[test]
    fn a_correlation_never_leaves_the_unit_interval_under_the_biased_divisor() {
        let centred = [0.5, -1.25, 2.0, -0.75, 0.25, 1.5, -2.0];

        let sequence = normalise(&autocovariance(&centred, 6, false), "").unwrap();

        assert!(sequence.iter().all(|value| value.abs() <= 1.0));
    }

    #[test]
    fn a_series_that_does_not_vary_cannot_be_normalised() {
        assert!(normalise(&[0.0, 0.0], "ar").is_err());
    }

    #[test]
    fn a_lag_no_pair_of_rows_reaches_is_rejected() {
        assert!(check_max_lag("ar", 4, 4).is_err());
        assert!(check_max_lag("ar", 3, 4).is_ok());
    }
}
