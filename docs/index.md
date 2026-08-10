# polars-qr

Dense numerical operations for [Polars](https://pola.rs), backed by
[faer](https://github.com/sarah-quinones/faer-rs).

Some numerical work does not fit an elementwise expression: fitting several targets against
one feature matrix, decomposing a block of columns, solving a covariance system, measuring a
statistic against something other than wall time. Writing those in Polars usually means
leaving the query plan, materialising to NumPy, and putting the result back. polars-qr keeps
them inside the plan.

Every operation is a Polars expression. It composes with `select`, `with_columns`,
`group_by`, `.over(...)` and lazy execution, and the numbers are computed in Rust.

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

## Install

```bash
pip install polars-qr
```

Polars 1.43 or newer, Python 3.11 or newer. There is nothing else to install: the extension
ships compiled.

## What is here

| | |
| --- | --- |
| [Least squares](guide/least_squares.md) | Fitting, projection and calibration, weighted or penalised, one target or several |
| [Components and second moments](guide/moments.md) | Principal components, scores, covariance and correlation |
| [Positive-definite systems](guide/spd.md) | One Cholesky factorisation, any number of right-hand sides |
| [Partitioned data](guide/states.md) | Summaries that merge, so a fit can be assembled from partitions |
| [Statistics against a clock](guide/timeseries.md) | Rolling and decaying statistics measured against cumulative volume, a trade count, or anything else that only moves forward |
| [Autoregression](guide/autoregression.md) | Fitting a series against its own lags through the structure that makes it cheap, with the diagnostics that fall out |

## What this is not

polars-qr is a small set of dense operations, not a regression library, a statistics package
or a wrapper around faer. There is no matrix object crossing into Python, no formula parsing,
no sparse support, and no raw factorisations: an operation is exposed when it is useful in
itself, not because a decomposition happens to compute it.
