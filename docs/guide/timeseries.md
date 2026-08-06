# Statistics against a clock

A clock is any column that only moves forward. It may be wall time, but it may just as well
be cumulative traded volume, a trade count, a quote-update count, or cumulative squared
return — whatever you think an observation should age against.

```python
result = trades.with_columns(
    decayed_flow=pq.timeseries.ewm_sum(
        "signed_quantity",
        clock="cumulative_volume",
        half_life=5_000_000,
    ).over(["symbol", "session"])
)
```

`5_000_000` is five million units of volume. Not five million rows, not a duration, and not
a string that has to be decoded to find out which.

## Building a clock

polars-qr never builds one. What a clock means is a modelling decision, so you make it with
ordinary Polars expressions:

```python
volume_clock = pl.col("quantity").cum_sum()
notional_clock = (pl.col("price") * pl.col("quantity")).cum_sum()
trade_clock = pl.lit(1).cum_sum()
variance_clock = pl.col("return").pow(2).cum_sum()
```

!!! note "A clock is not a weight"
    A clock says how far apart two observations are, which decides how much the older one
    has decayed by the time the newer one arrives. It does not say that one observation
    counts for more than another at the moment it arrives: every valid observation enters
    with weight one.

## Windows and half-lives

A numeric clock takes a number in its own units. A temporal clock takes a `timedelta`.
Mixing them is an error rather than a guess, because both guesses are wrong: a number is not
a duration, and a duration says nothing about how much volume has traded.

| Clock | Span | |
| --- | --- | --- |
| numeric | number | the number, in clock units |
| temporal | `timedelta` | converted to the clock's own unit |
| numeric | `timedelta` | error |
| temporal | number | error |

Integer, float, `Date` and `Datetime` clocks are all supported. `Date` counts in days and
`Datetime` in its own time unit.

## Hard windows

At each row the window covers the clock interval `(c - window, c]`: everything the clock has
passed within a window's width, up to and including the current row, and nothing ahead of
it.

Observations are indivisible. On a volume clock, a trade that straddles the far edge is
either in or out; it is never split to make the window hold exactly the width asked for.

`min_samples` holds the result back until the window has that many valid observations.
`min_clock_span` holds it back until the window covers that much clock, measured from the
oldest observation still in it.

## Exponential weighting

Everything already held is halved for every `half_life` of clock that passes, and each valid
observation enters with weight one. A clock that stands still decays nothing; a gap wide
enough forgets the past entirely.

- `ewm_sum` is the decaying total.
- `ewm_mean` is that total over the weight behind it.
- `ewm_variance`, `ewm_covariance` and `ewm_correlation` are the weighted second moments.

`bias=False`, the default, corrects for the spread lost to estimating the mean from the same
observations; with equal weights it is the usual step from `n` to `n - 1`. Correlation takes
no `bias` argument, because the same correction applied to the covariance and to both
variances cancels in the division.

## Nulls

Under the default `null_policy="skip"` a null value still advances the clock, still decays
the state and still moves the window; it simply adds nothing. That makes the statistic a
state estimate that survives a row with nothing new in it. `null_policy="raise"` makes a
null an error instead.

For a pair, both columns share one validity mask: a row enters only when both values are
there, so a correlation divides a covariance by two variances taken over the same
observations.

A value that is NaN or infinite is always an error. Reading it as missing would be a guess,
and both available guesses change the answer.

## Grouping

State resets at each Polars window group, so a clock that restarts each session needs the
session in `.over(...)`:

```python
result = (
    trades.lazy()
    .sort(["symbol", "session", "cumulative_volume"])
    .with_columns(
        decayed=pq.timeseries.ewm_mean("signal", clock="cumulative_volume", half_life=1e6).over(
            ["symbol", "session"]
        )
    )
)
```

A clock that decreases within a group is an error. The frame is never sorted silently:
sorting it would change which row each answer belongs to, and would hide a mistake made
further upstream.

## On the column itself

The `.qr` namespace puts the column in front, which reads better when the statistic is about
that column:

```python
pl.col("signed_quantity").qr.ewm_sum(clock="cumulative_volume", half_life=5_000_000)
pl.col("asset_return").qr.ewm_covariance("market_return", clock="volume", half_life=1e8)
```

Every namespace method hands straight over to the function of the same name, with the same
arguments, defaults and numbers.

## What is not here

Estimators for series that are not observed together, such as Hayashi–Yoshida; automatic
resampling; as-of joins; volume, dollar or imbalance bars; rolling regression; observation
weights separate from the clock. Series that are not observed together can be resampled or
aligned before these statistics are applied.
