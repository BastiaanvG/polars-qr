//! Reading a column as the observations an autoregression is fitted to.

use polars::prelude::*;

use crate::autoregression::options::NullPolicy;

/// The observations, held as deviations from their own mean.
///
/// Everything downstream works on the deviations: the autocovariance sequence is a sum of
/// their products, and the filter is applied to them and the mean added back at the end.
/// Centring once here is what keeps the level of the data out of every product.
pub struct Observations {
    centred: Vec<f64>,
    valid: Vec<bool>,
    mean: f64,
}

impl Observations {
    /// Read a column of observations.
    pub fn from_series(
        series: &Series,
        policy: NullPolicy,
        demean: bool,
        operation: &str,
    ) -> PolarsResult<Self> {
        let name = series.name().to_string();
        let cast = series.cast(&DataType::Float64)?;
        let column = cast.f64()?;

        let mut values = Vec::with_capacity(column.len());
        for (row, value) in column.iter().enumerate() {
            match value {
                None if policy == NullPolicy::Raise => {
                    polars_bail!(
                        ComputeError:
                        "{} received a null value.\n\nInput:\n{}\n\nRow:\n{}\n\nA lag counts \
                         rows, so dropping this one would close the gap and quietly change \
                         what every lag across it means. Pass null_policy='zero' to treat it \
                         as an observation at the mean instead, or decide yourself what \
                         belongs there.",
                        operation, name, row,
                    )
                }
                None => values.push(None),
                Some(value) if !value.is_finite() => {
                    polars_bail!(
                        ComputeError:
                        "{} received a value that is not finite.\n\nInput:\n{}\n\nRow:\n{}\n\n\
                         Value:\n{}",
                        operation, name, row, value,
                    )
                }
                Some(value) => values.push(Some(value)),
            }
        }

        let n_valid = values.iter().flatten().count();
        if n_valid == 0 {
            polars_bail!(
                ComputeError:
                "{} received no observations.\n\nInput:\n{}", operation, name,
            );
        }
        let mean = if demean {
            values.iter().flatten().sum::<f64>() / n_valid as f64
        } else {
            0.0
        };

        Ok(Self {
            centred: values
                .iter()
                .map(|value| value.map_or(0.0, |value| value - mean))
                .collect(),
            valid: values.iter().map(Option::is_some).collect(),
            mean,
        })
    }

    /// The observations as deviations from the mean, a null reading as no deviation at all.
    pub fn centred(&self) -> &[f64] {
        &self.centred
    }

    /// What was subtracted, or zero when the caller asked for no centring.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Which rows held a value.
    pub fn valid(&self) -> &[bool] {
        &self.valid
    }

    /// How many rows there are, whether or not they held a value.
    pub fn len(&self) -> usize {
        self.centred.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(values: &[Option<f64>]) -> Series {
        Series::new("y".into(), values)
    }

    #[test]
    fn centring_subtracts_the_mean_of_what_is_there() {
        let read = Observations::from_series(
            &series(&[Some(1.0), Some(3.0)]),
            NullPolicy::Raise,
            true,
            "",
        )
        .unwrap();

        assert_eq!(read.mean(), 2.0);
        assert_eq!(read.centred(), [-1.0, 1.0]);
        assert_eq!(read.len(), 2);
    }

    #[test]
    fn without_centring_the_values_are_left_where_they_are() {
        let read = Observations::from_series(
            &series(&[Some(1.0), Some(3.0)]),
            NullPolicy::Raise,
            false,
            "",
        )
        .unwrap();

        assert_eq!(read.mean(), 0.0);
        assert_eq!(read.centred(), [1.0, 3.0]);
    }

    #[test]
    fn a_null_contributes_nothing_and_is_remembered() {
        let read = Observations::from_series(
            &series(&[Some(1.0), None, Some(3.0)]),
            NullPolicy::Zero,
            true,
            "",
        )
        .unwrap();

        assert_eq!(read.mean(), 2.0);
        assert_eq!(read.centred(), [-1.0, 0.0, 1.0]);
        assert_eq!(read.valid(), [true, false, true]);
    }

    #[test]
    fn a_null_is_an_error_by_default() {
        let read =
            Observations::from_series(&series(&[Some(1.0), None]), NullPolicy::Raise, true, "ar");

        assert!(read.is_err());
    }

    #[test]
    fn a_value_that_is_not_finite_is_an_error_under_either_policy() {
        for policy in [NullPolicy::Raise, NullPolicy::Zero] {
            let read = Observations::from_series(
                &series(&[Some(1.0), Some(f64::INFINITY)]),
                policy,
                true,
                "ar",
            );

            assert!(read.is_err());
        }
    }

    #[test]
    fn a_column_with_nothing_in_it_is_an_error() {
        let read = Observations::from_series(&series(&[None, None]), NullPolicy::Zero, true, "ar");

        assert!(read.is_err());
    }
}
