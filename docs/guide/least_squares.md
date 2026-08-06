# Least squares

One implementation covers ordinary fitting, weighted fitting, several targets at once,
ridge-penalised fitting, and systems that do not pin down a single answer. It is named after
the problem it solves rather than after one use of it, because the same solve serves a
regression, a projection, a hedge ratio and a calibration.

```python
fit = frame.select(
    pq.least_squares(
        ["return_1h", "return_6h"],  # several targets share one factorisation
        feature_columns,
        weights="liquidity",  # optional, finite and non-negative
        intercept=True,
        solver="qr",
        l2_penalty=0.0,
    ).alias("fit")
).unnest("fit")
```

## Several targets

Targets that share a feature matrix share the factorisation of it, so fitting five targets
costs one decomposition rather than five. They also share the sample: a row that is unusable
for one target is unusable for all of them, which is what makes the coefficients comparable
across targets.

## Choosing a solver

`solver="qr"` is the fast route and needs the features to have full column rank.

`solver="svd"` also solves rank-deficient and underdetermined systems. Where the data does
not pin down a single answer it returns the one of smallest norm, and it reports the
singular values it used.

The reported `rank` is what tells you which situation you were in. A rank below the number
of columns means the features are collinear and the individual coefficients are not
determined by the data, whatever the solver returned.

```python
out = frame.select(pq.least_squares("y", columns, solver="svd").alias("fit")).unnest("fit")
out["rank"][0]  # 3, where columns has 4 entries: one is redundant
out["singular_values"][0]  # what the decomposition saw
out["condition"][0]  # the ratio of the largest to the smallest
```

Under `solver="qr"` the condition estimate is read off the diagonal of the QR factor. It is
cheaper and rougher than the ratio of singular values, and it is an estimate rather than the
number itself.

## The intercept

`intercept=True` fits a constant term and reports it separately, so `features` and
`coefficients` keep lining up. Under weights it is weighted like every other column, and
under a penalty it is never shrunk: pulling the constant towards zero would move the fit
rather than regularise it.

## The penalty

`l2_penalty` shrinks the coefficients towards zero. The residual sums of squares that come
back are those of the unpenalised system, so they still say how well the fit describes the
data rather than how well it describes the data plus the penalty.

## Reading the result

| Field | |
| --- | --- |
| `features`, `targets` | the names, in the order they were given |
| `coefficients` | one list per target, ordered like `features` |
| `intercept` | one entry per target, or null when none was fitted |
| `n_observations` | how many rows the fit used |
| `rank` | the numerical rank of the design, the constant term included |
| `residual_sum_of_squares` | one entry per target |
| `singular_values` | under `solver="svd"`, else null |
| `condition` | an estimate of the condition number |
| `solver` | which factorisation produced the fit |
