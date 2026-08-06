# Generalized-clock timeseries statistics

This is the specification for `polars_faer.timeseries`. It is what the implementation is
checked against; where the two disagree, one of them is wrong and this file says which.

## What a clock is

A clock is any column that is monotonically non-decreasing within each execution group. It
does not have to represent wall time. It may be cumulative traded volume, cumulative
notional, a trade count, a quote-update count, cumulative absolute return, cumulative
squared return, or any other numeric measure of activity.

```python
volume_clock = pl.col("quantity").cum_sum()
notional_clock = (pl.col("price") * pl.col("quantity")).cum_sum()
trade_clock = pl.lit(1).cum_sum()
variance_clock = pl.col("return").pow(2).cum_sum()
```

The package never builds a clock. What a clock means is a modelling decision, so the caller
makes it with ordinary Polars expressions.

**A clock is not an observation weight.** It says how far apart two observations are, which
decides how much the older one has decayed by the time the newer one arrives. It does not
say that one observation counts for more than another at the moment it arrives: every valid
observation enters with weight one.

## Public API

The canonical API is the functional one. It is explicitly typed, easy to discover and does
not depend on a type checker understanding a namespace registered at runtime.

```python
import polars_faer as pf

pf.timeseries.rolling_sum(value, *, clock, window, min_samples=1, min_clock_span=None, null_policy="skip")
pf.timeseries.rolling_mean(value, *, clock, window, ...)
pf.timeseries.rolling_variance(value, *, clock, window, ..., ddof=1)
pf.timeseries.rolling_covariance(x, y, *, clock, window, ..., ddof=1)
pf.timeseries.rolling_correlation(x, y, *, clock, window, ...)

pf.timeseries.ewm_sum(value, *, clock, half_life, min_samples=1, null_policy="skip")
pf.timeseries.ewm_mean(value, *, clock, half_life, ...)
pf.timeseries.ewm_variance(value, *, clock, half_life, ..., bias=False)
pf.timeseries.ewm_covariance(x, y, *, clock, half_life, ..., bias=False)
pf.timeseries.ewm_correlation(x, y, *, clock, half_life, ...)
```

The `Expr.faer` namespace is a convenience over exactly those functions, with the same
parameters, defaults, numbers, dtypes and errors. It holds no numerical logic.

```python
pl.col("signed_quantity").faer.ewm_sum(clock="cumulative_volume", half_life=5_000_000)
pl.col("asset_return").faer.ewm_covariance(
    "market_return", clock="cumulative_notional", half_life=100_000_000
)
```

There is no frame or Series namespace. Every operation is already a row-preserving
expression, so `select`, `with_columns`, `.over(...)` and lazy queries all reach it.

## Spans

```python
ClockSpan = int | float | timedelta
```

A numeric clock takes a number in its own units. A temporal clock takes a `timedelta`.
Mixing them is an error rather than a guess. String durations such as `"10000i"` or `"5m"`
are not accepted: the units of a clock belong to the caller, and encoding them in a string
hides them from the type checker.

| Clock | Span | Result |
| --- | --- | --- |
| numeric | number | the number, in clock units |
| temporal | `timedelta` | converted to the clock's own unit |
| numeric | `timedelta` | error |
| temporal | number | error |

Supported clock dtypes are the signed and unsigned integers, `Float32`, `Float64`, `Date`
and `Datetime`. `Date` counts in days and `Datetime` in its own time unit.

## Execution semantics

Every operation returns one value per input row, in the order it was given, computed from
the current row and the ones before it and never from the ones after. State resets at each
Polars window group, so a clock that restarts each session needs the session in `.over(...)`.

```python
result = (
    trades.lazy()
    .sort(["symbol", "session", "cumulative_volume"])
    .with_columns(
        decayed_flow=pf.timeseries.ewm_sum(
            "signed_quantity",
            clock="cumulative_volume",
            half_life=5_000_000,
        ).over(["symbol", "session"])
    )
)
```

The output dtype is always `Float64`, whatever the inputs are: the decay is floating point,
and one predictable dtype makes the schema knowable without running the query.

## Clock validation

- A clock that decreases within a group is an error. The frame is never sorted silently,
  because that would change which row each answer belongs to and hide the mistake upstream.
- A null clock value is an error. A null cannot place an observation in time.
- A clock value that is not finite is an error.
- Equal adjacent clock values are legal. Nothing decays between them, and the newer
  observation still enters with weight one. For a hard window, earlier rows sharing a clock
  value are visible and later ones are not, so row order breaks the tie.

## Hard windows

At each row the window covers the clock interval `(c - window, c]`. A prior observation is
in it when `c_current - c_prior < window`, and only rows at or before the current one are
ever visible.

There is no `closed` parameter: this is the causal case, and it is the one that behaves
predictably when several rows share a clock value. Other closures can be added later without
changing this default.

Observations are indivisible. On a volume clock, a trade that straddles the far edge of the
window is either in or out; it is never split to make the window hold exactly the width
asked for.

- `min_samples` is the minimum number of valid observations in the window. For a pair, a row
  counts only when both values are valid.
- `min_clock_span` holds the result back until the window covers that much clock, measured
  from the oldest observation still in it. It may not exceed `window`.

## Exponential weighting

For consecutive clock values the distance is `Δc = c_i - c_{i-1}`, and with half-life `h`
the decay applied to everything already held is

```
λ = 2 ** (-Δc / h)
```

so `Δc = 0` leaves the state untouched, `Δc = h` halves it, and a gap wide enough underflows
to zero, which forgets the past entirely.

The state carries the decaying sum `S`, the total weight `W`, the total squared weight `W²`,
the weighted mean and the weighted centred moments. Each valid observation enters with
weight one:

```
S = x + λ·S      W = 1 + λ·W      W² = 1 + λ²·W²
```

- `ewm_sum` reports `S`.
- `ewm_mean` reports `S / W`.
- `ewm_variance` reports `M₂ / W` when `bias=True`, and `M₂ / (W - W²/W)` otherwise.
- `ewm_covariance` reports `C / W` or `C / (W - W²/W)` the same way.
- `ewm_correlation` reports `C / sqrt(M₂ₓ · M₂ᵧ)`. There is no `bias` parameter: the same
  correction applied to the covariance and to both variances cancels in the division.

Weights are normalised by their own sum rather than assumed to add to one. There is no
`adjust` parameter. This is what keeps `ewm_sum` and `ewm_mean` consistent with each other,
lets the whole family share one weighting model, and gives an observation arriving at zero
clock distance its full weight instead of none.

This is deliberately not a wrapper around the Polars `ewm_mean_by` recursion, whose new
observation carries a coefficient that depends on the clock gap. A difference from it is
therefore not automatically a fault.

## Nulls and non-finite values

Under the default `null_policy="skip"`, a null value still advances the clock, still decays
the state and still moves the window; it simply adds nothing. The result is null only when
what remains does not meet `min_samples` or `min_clock_span`. This makes the statistic a
state estimate that survives a row with nothing new in it.

`null_policy="raise"` makes a null an error instead.

For a pair, `x` and `y` share one validity mask: a row enters only when both are there, so
the covariance and the two variances behind a correlation describe the same observations.

A value that is `NaN` or infinite is always an error, under either policy. Treating it as
missing would be a guess, and both available guesses change the answer.

## Numerical notes

- Rolling state is held relative to an origin taken from the first observation in the
  window. Without it, `value - mean` at a level of a billion with a spread of one loses ten
  of its sixteen digits before the arithmetic starts.
- Rolling state is rebuilt from the window it holds every 1024 removals, which clears the
  rounding that add-and-remove leaves behind and picks a fresh origin as the level of the
  data moves. This is internal and does not appear in the public API.
- A centred second moment that rounding pushes just below zero is read as zero. One that is
  materially negative is left visible, because it would mean the update itself is wrong.
- A correlation a hair outside `[-1, 1]` is clamped. One materially outside is not.

## Complexity

| Family | Time | Memory |
| --- | --- | --- |
| exponentially weighted | `O(n)` | `O(1)` |
| hard window | `O(n)` | `O(observations in the window)` |

## Not in this release

Hayashi–Yoshida and other estimators for series that are not observed together; automatic
resampling; as-of joins; volume, dollar, imbalance and run bars; fractional splitting of an
observation at a window boundary; rolling regression or PCA; full covariance matrices per
row; observation weights separate from the clock; lead–lag estimation; microstructure-noise
correction; state that continues across separate query executions; user-defined callbacks.

Series that are not observed together can be resampled or aligned before these statistics
are applied.

## Changing any of this

A change to documented behaviour updates, together: the Python signature, its docstring,
this file, the numerical tests, and at least one runnable example.
