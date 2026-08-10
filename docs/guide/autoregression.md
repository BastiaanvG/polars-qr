# Autoregression

Fitting a series against its own lags, the autocorrelation diagnostics that go with it, and
the structured solve underneath.

```python
fit = (
    prices.lazy()
    .group_by("symbol")
    .agg(pq.autoregression.fit("log_return", order="bic", max_order=20).alias("ar"))
    .unnest("ar")
    .collect()
)
```

## A lag is a row

The model assumes one step of the sequence is like any other. That is why this is not in
`pq.timeseries`, where the distance between two observations is a number you supply: a
clocked statistic would take a `clock` argument, and an autoregression has nowhere to put
one.

!!! warning "Unevenly spaced observations"
    A fit over rows that arrive at irregular intervals describes the sequence, not the
    process in time. Resample before fitting.

Row order within the group is the sequence, and the frame is never sorted silently.

## Why not lag columns and least squares

Shifting the column and fitting is three lines, and `pq.least_squares` will do it:

```python
lags = [pl.col("y").shift(k).alias(f"lag_{k}") for k in range(1, p + 1)]
frame.with_columns(lags).drop_nulls().select(pq.least_squares("y", [f"lag_{k}" for k in ...]))
```

The matrix that produces is Toeplitz — constant along each of its diagonals — and a general
factorisation cannot use that. What the structure buys:

- **Memory.** The lag route materialises `p` columns of `n` values. Fitting 200 lags on ten
  million rows builds two billion doubles. Here the series is read into `p + 1`
  autocovariances and nothing else.
- **Cost.** The Levinson–Durbin recursion is `O(p²)` rather than `O(p³)`.
- **Every order at once.** The recursion climbs through the orders, so the fits below the
  one you asked for, and their prediction-error variances, are already computed. Order
  selection costs nothing beyond the fit.
- **Stationarity.** A fit from a positive-definite autocovariance sequence has every
  reflection coefficient inside the unit circle, so it cannot come out explosive. Least
  squares on lag columns offers no such guarantee, which is a real failure mode on short or
  near-unit-root series.

## Choosing an order

`order` takes a number, or the name of a criterion to choose one by.

```python
pq.autoregression.fit("y", order=3)
pq.autoregression.fit("y", order="bic", max_order=20)
```

With `σ̂²_k` the prediction-error variance at order `k` and `n` the observation count:

```
AIC(k)  = n·ln(σ̂²_k) + 2k
BIC(k)  = n·ln(σ̂²_k) + k·ln(n)
HQIC(k) = n·ln(σ̂²_k) + 2k·ln(ln(n))
```

Additive constants are dropped, consistently, so the values compare within one fit and not
against another library's. Order 0 is a legal outcome and means the series is white noise as
far as the criterion can tell.

The three are not interchangeable. BIC is the conservative one and is order-consistent. AIC
is not, and over-selects on long series — on a simulated AR(2) with 200,000 observations BIC
picks 2 and AIC picks 4. That is AIC behaving as designed rather than misbehaving.

The `criterion` and `order_variance` fields carry the value at every order, so an
order-selection plot needs no second query.

## Estimators

`method="yule_walker"`, the default, builds the autocovariance sequence and solves the
system it describes. The divisor is `n` and not `n − k`, deliberately: the unbiased estimate
can produce a sequence that is not positive semi-definite, and such a sequence has no valid
recursion. On a short near-unit-root series the biased sequence gave a matrix whose smallest
eigenvalue was `+0.129` where the unbiased one gave `−0.034`.

`autocovariance` exposes `unbiased=True` for reading the sequence. `fit` does not, and will
not.

`method="burg"` minimises the forward and backward prediction errors together, recursing on
order without forming the sequence at all. It is better on short series, and gives the same
stationarity guarantee and the same result fields.

## Diagnostics

```python
frame.select(pq.autoregression.autocorrelation("y", max_lag=40).alias("acf")).unnest("acf")
frame.select(pq.autoregression.partial_autocorrelation("y", max_lag=40).alias("pacf")).unnest(
    "pacf"
)
```

The partial autocorrelations *are* the reflection coefficients the recursion produces, so
they cost nothing beyond the fit. The value at lag `k` equals the last coefficient of a fit
of order `k`, which is what a partial autocorrelation is defined to be.

`stationary` is read off those coefficients — every one inside the unit circle — and not
from the roots of the characteristic polynomial. It is cheaper, and it cannot be got
backwards: the polynomial's roots must lie outside the unit circle, but the reversed
polynomial that most root-finders are handed has the reciprocals, which must lie inside.

## Whitening a series

```python
frame.with_columns(
    residual=pq.autoregression.transform("y", order="bic", max_order=10),
    prediction=pq.autoregression.transform("y", order="bic", max_order=10, output="prediction"),
)
```

One value per input row. The first `order` rows of a group have no complete history and are
null. `output="residual"` gives the series whitened by its own fitted model, which is what
you want before cross-correlating two series; `output="prediction"` gives the one-step
fitted value.

!!! note "In-sample"
    The fit and the rows it is applied to are the same rows. That is the honest default for
    a diagnostic and the wrong tool for a forecast.

## Solving a Toeplitz system

A symmetric Toeplitz matrix is determined by its first column, so unlike
[`solve_spd`](spd.md), which reads a square wide frame, this reads one column of length `n`
plus the right-hand sides.

```python
solution = frame.select(
    pq.autoregression.solve_toeplitz("sequence", ["rhs_a", "rhs_b"], row_index="lag").alias(
        "solved"
    )
).unnest("solved")
```

The matrix is never formed and the solve costs `O(n²)`. A matrix that is not positive
definite shows up as a non-positive prediction error partway through the recursion and is
reported as that, with `diagonal_shift` available exactly as in `solve_spd`.

## Nulls

`null_policy="raise"` is the default here, which is stricter than the rest of the package.
Dropping a row from a matrix removes an observation; dropping a row from a sequence
redefines every lag that spans it, because the gap closes and rows that were three apart
become two apart without saying so.

`null_policy="zero"` treats a null as an observation at the mean, so it contributes nothing
to any lagged product. It is an approximation, for series with scattered gaps, and a null
row still counts towards the divisor. Under `transform`, a row is null when its own value or
any lag it needs was missing.

There is no `"drop"`. Anyone who wants it can drop the rows themselves and accept what that
means for their lags.

## On the column itself

```python
pl.col("log_return").ar.fit(order="bic", max_order=20)
pl.col("log_return").ar.partial_autocorrelation(max_lag=40)
```

`.ar` is a separate namespace from `.qr` because a clock and a lag are different ideas.
Every method hands straight over to the function of the same name.

## What is not here

Moving-average and ARMA terms, which need nonlinear optimisation. Seasonal terms.
Differencing and unit-root testing. Vector autoregression, which needs block-Toeplitz
machinery. State-space models and the Kalman filter. Forecasting beyond one step, confidence
intervals and impulse responses. Spectral estimation. Any general Toeplitz or circulant
matrix API beyond the one solve. Mergeable partitioned state: an autocovariance sequence
looks like it should merge, but the pooled estimate includes the cross terms spanning the
boundary between two partitions, and a summary of one partition's interior cannot reproduce
them.
