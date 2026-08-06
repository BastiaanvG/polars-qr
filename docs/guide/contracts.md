# Reading and returning data

Every operation here reads the same shape of input and returns the same shape of output, so
what you learn from one carries to the rest.

## What an operation reads

A set of numeric columns that together form one dense matrix, one row per observation. The
columns are taken in the order they are given, and that order is what the result labels
itself with.

- Integers are widened to double precision. A non-numeric column is rejected rather than
  parsed, so a string column of numbers is an error and not a silent success.
- A row is unusable when any column it spans is null or holds a value that is not finite.
  `null_policy="raise"` fails on such a row; `null_policy="drop"` leaves it out.
- Rows are dropped jointly: a row is used by every column or by none of them.

!!! note "Why rows are dropped jointly"
    Dropping per column would estimate each entry of a covariance matrix from a different
    sample. The result can fail to be a covariance matrix at all — it may have a negative
    eigenvalue — and then a Cholesky factorisation of it fails for reasons that have nothing
    to do with the data.

Weights, where an operation takes them, must be finite and non-negative. A weight of zero is
legal: it leaves an observation out of the result without dropping it from the sample.

## What an operation returns

One struct per group, carrying the numbers together with the labels needed to read them, so
nothing depends on remembering the column order.

- A vector is a list of floats, ordered like the input columns.
- A matrix is a list of its rows.
- Diagnostics — the numerical rank, a condition estimate, the sum of the weights — are
  fields of the same struct rather than separate operations to call.

`unnest` turns the struct into ordinary columns:

```python
fit = frame.select(pq.least_squares("y", ["a", "b"]).alias("fit")).unnest("fit")
fit["features"][0].to_list()  # ['a', 'b']
fit["coefficients"][0].to_list()  # [[0.51, -0.24]]  one row per target
```

The row-preserving operations are the exception: they return one value per input row instead
of one per group, so they can sit next to the data they were computed from.

## Names

The names in a result come from the expressions you passed, not from the data.

```python
frame.select(pq.least_squares("y", [pl.col("a").alias("renamed"), "b"]).alias("fit"))
# features: ['renamed', 'b']
```

!!! warning "Why this matters inside `group_by`"
    Polars hands a plugin its inputs without names inside an aggregation. Reading the name
    off the data would label every grouped result with empty strings, so the names travel
    separately.
