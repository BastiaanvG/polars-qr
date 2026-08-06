# polars-qr

Dense numerical operations for [Polars](https://pola.rs), backed by
[faer](https://github.com/sarah-quinones/faer-rs).

Some numerical work does not fit an elementwise expression: fitting several targets against
one feature matrix, decomposing a block of columns, solving a covariance system. Writing
those in Polars means leaving the query plan, materialising to NumPy, and putting the result
back. polars-qr keeps them inside the plan.

The package is an expression plugin. Every operation is a Polars expression, so it composes
with grouping, lazy execution and the rest of a query plan, and the numerics run in Rust.

```python
import polars as pl
import polars_qr as pq

fit = (
    frame.lazy()
    .group_by("date")
    .agg(pq.least_squares("return", ["signal_a", "signal_b"], intercept=True).alias("fit"))
    .unnest("fit")
    .collect()
)
```

## Installing

```bash
pip install polars-qr
```

## Operations

| Operation | Returns |
| --- | --- |
| `pq.least_squares` | Coefficients per target, with rank, residuals and a condition estimate |
| `pq.pca` | Loadings, singular values and explained variance |
| `pq.pca_transform` | One score column per component, aligned with the input rows |
| `pq.covariance` | A labelled covariance matrix with means and standard deviations |
| `pq.correlation` | The same, divided through by the standard deviations |
| `pq.solve_spd` | The solution of a positive-definite system, one per right-hand side |

Statistics measured against a clock of your own — cumulative volume, a trade count,
cumulative squared return — live in `pq.timeseries` and are documented in
[docs/timeseries.md](docs/timeseries.md):

| Operation | Returns |
| --- | --- |
| `pq.timeseries.rolling_sum`, `_mean`, `_variance` | One value per row over a trailing window of the clock |
| `pq.timeseries.rolling_covariance`, `_correlation` | The same, for a pair of columns |
| `pq.timeseries.ewm_sum`, `_mean`, `_variance` | One value per row, weighted by how far the clock has moved |
| `pq.timeseries.ewm_covariance`, `_correlation` | The same, for a pair of columns |

```python
result = trades.with_columns(
    decayed_flow=pq.timeseries.ewm_sum(
        "signed_quantity",
        clock="cumulative_volume",
        half_life=5_000_000,  # five million units of volume, not five million rows
    ).over(["symbol", "session"])
)
```

### Least squares

One implementation covers ordinary fitting, weighted fitting, several targets at once,
ridge-penalised fitting and underdetermined systems. It is named after the problem it
solves rather than after one use of it.

```python
fit = frame.select(
    pq.least_squares(
        ["return_1h", "return_6h"],  # several targets share one factorisation
        feature_columns,
        weights="liquidity",  # optional, finite and non-negative
        intercept=True,
        solver="qr",  # "svd" for rank-deficient or underdetermined systems
        l2_penalty=0.0,
    ).alias("fit")
).unnest("fit")
```

`solver="qr"` is the fast route and needs the features to have full column rank.
`solver="svd"` also solves rank-deficient and underdetermined systems, returning the
solution of smallest norm, and reports the singular values it used.

### Principal components

```python
components = frame.select(pq.pca(feature_columns, n_components=10).alias("pca")).unnest("pca")

scored = frame.lazy().qr.pca_transform(feature_columns, n_components=10, by="date").collect()
```

`pca` reports the loadings, the singular values, the explained variance and its ratio.
`pca_transform` keeps its rows, so the scores land next to the data they came from. The sign
of each component is fixed so that its largest loading is positive, which keeps two runs
over the same data comparable.

### Covariance and correlation

```python
risk = frame.select(pq.covariance(feature_columns, weights="recency").alias("cov")).unnest("cov")
```

Weights are read as reliability weights: scaling all of them by the same factor leaves the
estimate alone, and the divisor is corrected for them the way `numpy.cov(aweights=...)`
corrects it.

### Positive-definite systems

```python
solution = wide.select(
    pq.solve_spd(
        matrix_columns,
        ["expected", "exposure"],
        row_index="asset_index",
        diagonal_shift=1e-8,
    ).alias("solved")
).unnest("solved")
```

The matrix is read from a wide frame, one column per matrix column and one row per matrix
row, with `row_index` fixing which row is which. It is checked for symmetry, shifted along
the diagonal if asked, factorised once, and every right-hand side is solved against that
factorisation.

## Partitioned data

Least squares and second moments can be computed from a summary of the rows that is much
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

A state is a binary value, so it can be written to a file, sent between processes or stored
in a table and merged later. It carries a format version and a hash of the columns it was
built for, and refuses to merge into a state that does not match.

| State | Holds | Finalises to |
| --- | --- | --- |
| `pq.least_squares_state` | The triangular factor of the design with the targets appended | `pq.finalise_least_squares` |
| `pq.covariance_state` | Counts, weights, means and centred cross-products | `pq.finalise_covariance`, `pq.finalise_correlation`, `pq.finalise_pca` |

The two states are not interchangeable. Least squares goes through the QR factor because
solving it from second moments would square the conditioning of the data; second moments go
through the covariance state because that is what they are.

## The input contract

An operation reads a set of numeric columns that together form one dense matrix, one row per
observation. The columns are taken in the order they are given, and that order is what the
result labels itself with.

- Integers are widened to double precision. Non-numeric columns are rejected rather than
  parsed.
- A row is unusable when any column it spans is null or holds a value that is not finite.
  `null_policy="raise"` fails on such a row and `null_policy="drop"` leaves it out.
- Rows are dropped jointly: a row is used by every column or by none of them. This is what
  keeps a covariance matrix estimated from a single sample, and so still usable as the input
  to a factorisation.
- Weights, where an operation takes them, must be finite and non-negative. A weight of zero
  is legal and leaves an observation out of the result without dropping it from the sample.

## The result contract

An aggregate returns one struct per group. The struct carries the numbers together with the
labels needed to read them, so nothing depends on the caller remembering the column order.

- A vector is a list of floats, ordered like the input columns.
- A matrix is a list of its rows.
- Diagnostics such as the numerical rank, a condition estimate or the sum of the weights are
  fields of the same struct, not separate operations.

`unnest` turns the struct into ordinary columns.

## What this is not

polars-qr is a small set of dense operations, not a regression library, a statistics
package or a wrapper around faer. There is no matrix object crossing into Python, no formula
parsing, no sparse support, and no raw factorisations: an operation is exposed when it is
useful in itself, not because a decomposition can compute it.

## Examples

The `examples/` directory holds runnable scripts covering each operation; the test suite
runs them, so they cannot drift from the code.

```bash
uv run python examples/01_least_squares.py
```

## Development

The extension is built with [maturin](https://www.maturin.rs) and needs a Rust toolchain.

```bash
uv sync
uv run maturin develop --release --uv
uv run pytest
cargo test
```

## Licence

BSD 3-Clause.
