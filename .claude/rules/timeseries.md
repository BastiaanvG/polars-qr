---
paths:
  - "src/timeseries/**"
  - "src/expressions/timeseries.rs"
  - "python/polars_qr/timeseries.py"
  - "tests/test_timeseries.py"
  - "tests/reference.py"
---

# Clocked statistics

`docs/guide/timeseries.md` is the user-facing contract. These are the decisions behind it
that are easy to undo by accident.

## Settled decisions

- **A clock is not a weight.** It sets how far apart observations are, which decides decay.
  Every valid observation still enters with weight one. Anyone asking for per-observation
  weights is asking for a separate feature, not a reinterpretation of the clock.
- **Weights are normalised by their own sum**, so there is no `adjust` parameter. This is
  what keeps `ewm_sum` and `ewm_mean` consistent and gives an observation arriving at zero
  clock distance its full weight. It deliberately differs from the Polars `ewm_mean_by`
  recursion, so a difference from that function is not automatically a fault.
- **The window is `(c - window, c]`** and there is no `closed` parameter. Right-closed is the
  causal case and the one that behaves predictably when rows share a clock value.
- **Correlation takes no `bias` argument.** The correction cancels between the covariance and
  the two variances.
- **A decreasing clock is an error and the frame is never sorted silently.** Sorting would
  change which row each answer belongs to and hide a mistake made upstream.
- **NaN and infinity are always errors**, under either null policy, for values and clocks
  alike. Reading them as missing is a guess, and both guesses change the answer.

## Numerical traps

- Rolling state is held **relative to an origin** taken from the first observation in the
  window. Without it, `value - mean` at a level of 1e9 with a spread of 1 loses ten of its
  sixteen digits. Removing the origin shift silently costs about eight digits on offset data
  and no test of unshifted data will notice; `a_large_offset_does_not_cost_the_spread_its_digits`
  is the one that does.
- Clock differences are taken **before** widening to `f64`, so an integer or microsecond
  clock keeps them exact however far it has counted. Do not convert a clock column to `f64`
  on the way in.
- Rolling state is rebuilt from its window every `REBUILD_AFTER` removals, which clears
  accumulated rounding and refreshes the origin. It is internal and must not surface in the
  API.
- A centred second moment that rounding pushes just below zero reads as zero; one that is
  materially negative is left visible, because it means the update is wrong.

## Testing

`tests/reference.py` is the oracle: a from-scratch implementation, written to be read rather
than to be fast, that every compiled statistic is compared against row by row. When you
change the numerics, change the reference first and let it say whether the new behaviour is
what you meant.

Comparisons on ill-conditioned data belong in relative terms, and the sharper test is the
residual: an answer that is genuinely worse fits the data worse, whatever its coefficients
look like.
