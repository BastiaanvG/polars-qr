# polars-faer

Dense numerical operations for [Polars](https://pola.rs), backed by
[faer](https://github.com/sarah-quinones/faer-rs).

The package is an expression plugin. Every operation is a Polars expression, so it composes
with grouping, lazy execution and the rest of a query plan, and the numerics run in Rust
without a round trip through Python.

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

`unnest` turns the struct into ordinary columns:

```python
import polars as pl
import polars_faer as pf

fit = (
    prices.lazy()
    .group_by("date")
    .agg(pf.least_squares("excess_return", ["factor_a", "factor_b"]).alias("fit"))
    .unnest("fit")
    .collect()
)
```

## Operations

| Operation | Returns |
| --- | --- |
| `pf.least_squares` | Coefficients per target, with rank, residuals and a condition estimate |
| `pf.covariance` | A labelled covariance matrix with means and standard deviations |
| `pf.correlation` | The same, divided through by the standard deviations |
| `pf.pca` | Loadings, singular values and explained variance |
| `pf.pca_transform` | One score column per component, aligned with the input rows |
| `pf.solve_spd` | The solution of a positive-definite system, one per right-hand side |

Row-preserving operations are also reachable from a frame, where the grouping and the
unnesting are part of the call:

```python
scored = frame.lazy().faer.pca_transform(signals, n_components=10, by="date").collect()
```

## Development

The extension is built with [maturin](https://www.maturin.rs) and needs a Rust toolchain.

```bash
uv sync
uv run maturin develop --release --uv
uv run pytest
cargo test
```
