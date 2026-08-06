# Partitioned data

Least squares and second moments can be computed from a summary of the rows that is far
smaller than the rows themselves and can be merged with another summary. Each partition
summarises what it holds, the summaries are merged in any order, and the result is finalised
once.

```python
states = (
    frame.lazy()
    .group_by("partition")
    .agg(pq.least_squares_state("return", feature_columns).alias("state"))
)

fit = (
    states.select(pq.merge_least_squares_states("state").alias("state"))
    .select(pq.finalise_least_squares("state", solver="qr").alias("fit"))
    .unnest("fit")
    .collect()
)
```

A state is a binary value. It can be written to a file, sent between processes, or stored in
a table and merged with tomorrow's. Its size is quadratic in the number of columns and does
not depend on how many rows went into it.

## The two states

| State | Holds | Finalises to |
| --- | --- | --- |
| `least_squares_state` | the triangular factor of the design with the targets appended | `finalise_least_squares` |
| `covariance_state` | counts, weights, means and centred cross-products | `finalise_covariance`, `finalise_correlation`, `finalise_pca` |

They are not interchangeable. Least squares goes through the QR factor because solving it
from second moments squares the conditioning of the data; second moments go through the
covariance state because that is what they are.

## What is decided when

What the state holds is fixed when it is built: which columns, in which order, whether there
is a constant term, and what weights the observations carried. Everything else is chosen at
the end, so one set of summaries can be finalised several ways.

```python
merged = states.select(pq.merge_least_squares_states("state").alias("state"))
plain = merged.select(pq.finalise_least_squares("state").alias("fit"))
ridge = merged.select(pq.finalise_least_squares("state", l2_penalty=500.0).alias("fit"))
```

A penalty is applied once, at the end, however many partitions the state was merged from.

## Merging is safe or it fails

A state carries a format version and a hash of the columns it was built for. Merging two
that were built for different columns is an error rather than a number, and so is reading a
state written by a build with a different format.

```python
pl.concat([one, other]).select(pq.merge_least_squares_states("state"))
# ComputeError: these states were built for different columns, so they cannot be merged
```

Merging is associative and commutative up to floating point, so the result does not depend
on how the rows were partitioned or on the order the partitions arrive in. Only the last
bits differ, and they differ in the way any two orders of summing do.
