//! What the caller asks an autoregression for.

use serde::Deserialize;

/// What to do about a null value.
///
/// The default is stricter than elsewhere in the crate, and deliberately. Dropping a row
/// from a matrix removes an observation; dropping a row from a *sequence* redefines every
/// lag that spans it, because the gap closes and rows that were three apart become two
/// apart without saying so.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NullPolicy {
    /// Fail on it.
    #[default]
    Raise,
    /// Treat it as an observation at the mean, contributing nothing to any lagged product.
    Zero,
}

/// Which estimator produces the coefficients.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    /// Solve the Yule–Walker system built from the biased autocovariance sequence.
    #[default]
    YuleWalker,
    /// Minimise the forward and backward prediction errors together.
    Burg,
}

impl Method {
    /// What the result reports this method as.
    pub fn name(self) -> &'static str {
        match self {
            Self::YuleWalker => "yule_walker",
            Self::Burg => "burg",
        }
    }
}

/// How an order is chosen when the caller does not name one.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Criterion {
    /// Akaike's criterion, which is not order-consistent and over-selects on long series.
    Aic,
    /// The Bayesian criterion, the conservative one.
    Bic,
    /// The Hannan–Quinn criterion, which sits between the two.
    Hqic,
}

impl Criterion {
    /// What a fit of `order` costs beyond its fit to the data, given `n` observations.
    ///
    /// Additive constants are dropped, consistently, so these compare within one fit and
    /// not against another library's.
    pub fn penalty(self, order: usize, n: usize) -> f64 {
        let order = order as f64;
        let n = n as f64;
        match self {
            Self::Aic => 2.0 * order,
            Self::Bic => order * n.ln(),
            Self::Hqic => 2.0 * order * n.ln().ln(),
        }
    }
}

/// How a fit is set up.
pub struct Options {
    /// The order to fit, when the caller named one.
    pub order: Option<usize>,
    /// The criterion to choose an order by, when the caller did not.
    pub criterion: Option<Criterion>,
    /// The highest order the recursion runs to.
    pub max_order: usize,
    /// Which estimator to use.
    pub method: Method,
}

/// Which row-aligned series a transform reports.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Output {
    /// What the fitted filter left behind: the series whitened by its own model.
    #[default]
    Residual,
    /// What the fitted filter expected, one step ahead.
    Prediction,
}
