//! Rolling and exponentially weighted statistics over a generalised clock.
//!
//! A clock is any column that only moves forward. It may be wall time, but it may just as
//! well be cumulative volume, a trade count or cumulative squared return: whatever the
//! caller thinks an observation should age against. The statistics here take the clock as
//! given and never build one, because what a clock means is a modelling decision and not a
//! numerical one.
//!
//! A clock is not an observation weight. It says how far apart two observations are, which
//! is what decides how much the older one has decayed by the time the newer one arrives;
//! it does not say that one observation counts for more than another at the moment it
//! arrives. Every valid observation enters with weight one.

pub mod clock;
pub mod error;
pub mod ewm;
pub mod options;
pub mod rolling;
