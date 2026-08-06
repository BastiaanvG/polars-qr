---
paths:
  - "python/**/*.py"
  - "tests/**/*.py"
  - "examples/**/*.py"
  - "docs/**/*.md"
---

# The public API

Three ways in, all reaching the same code. The functional API is canonical: it is typed,
discoverable, and does not depend on a type checker understanding a namespace registered at
runtime. The namespaces are wrappers with no logic of their own.

```python
import polars_qr as pq

pq.least_squares(...)  # module-level functions
pq.timeseries.ewm_sum(...)  # the clocked statistics
pl.col("x").qr.ewm_sum(...)  # the same, with the column in front
frame.lazy().qr.pca_transform(...)  # row-preserving operations on a frame
```

## Whole-group operations

These aggregate a group into one struct. Use them inside `select`, or inside
`group_by().agg()` for one result per group, then `unnest`.

| Function | Reach for it when |
| --- | --- |
| `pq.least_squares(targets, features, ...)` | fitting, projecting, hedging or calibrating; several targets share one factorisation |
| `pq.pca(features, ...)` | extracting factors, measuring effective dimensionality, finding redundant columns |
| `pq.covariance(features, ...)` | second moments, risk estimates, input to a solve |
| `pq.correlation(features, ...)` | the same, scale-free |
| `pq.solve_spd(matrix, rhs, row_index=...)` | solving against a covariance-like matrix, precision weighting, whitening by hand |

## Row-preserving operations

These return one value per input row, so they sit next to the data.

| Function | Reach for it when |
| --- | --- |
| `pq.pca_transform(features, n_components=k)` | scoring rows on components; each becomes `component_i` |
| `pq.timeseries.rolling_*` | a statistic over a trailing window of a clock |
| `pq.timeseries.ewm_*` | a statistic that decays as a clock advances |

`frame.qr.pca_transform(...)` and `frame.lazy().qr.pca_transform(...)` do the `with_columns`,
the `over` and the `unnest` in one call. The expression form is the same thing written out.

## Partitioned data

`pq.least_squares_state` and `pq.covariance_state` summarise a partition into a binary value;
`pq.merge_*_states` combines them; `pq.finalise_*` turns the result into the same struct the
direct operation returns. Reach for these when the rows do not fit in one place, or when one
pass should serve several finalisations.

## Typical shapes

Fitting per group inside a lazy query:

```python
(
    frame.lazy()
    .group_by("date")
    .agg(pq.least_squares("return", signals, intercept=True).alias("fit"))
    .unnest("fit")
)
```

Scoring rows on components fitted within their own group:

```python
frame.lazy().qr.pca_transform(signals, n_components=5, by="date")
```

A statistic that ages by activity rather than by rows:

```python
frame.with_columns(
    pq.timeseries.ewm_sum("flow", clock="cumulative_volume", half_life=5_000_000).over("symbol")
)
```

Assembling a fit from partitions:

```python
(
    frame.lazy()
    .group_by("partition")
    .agg(pq.least_squares_state("y", features).alias("state"))
    .select(pq.merge_least_squares_states("state").alias("state"))
    .select(pq.finalise_least_squares("state").alias("fit"))
    .unnest("fit")
)
```

## When adding to the API

A namespace method takes the same parameters, defaults and errors as the function it wraps,
and does nothing else. If you add a function, add the wrapper, and add a test asserting the
two produce identical frames — `tests/test_timeseries.py` has that test parametrised over
every method.

Anything new is exported from `polars_qr/__init__.py` and listed in `__all__`, and gets an
API page under `docs/api/`.
