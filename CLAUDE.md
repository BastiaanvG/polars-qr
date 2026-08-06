# polars-faer

A Polars expression plugin: dense numerical operations in Rust, backed by faer, reached from
Python as ordinary expressions.

## Building

The extension needs a Rust toolchain.

```bash
uv sync
uv run maturin develop --release --uv
uv run pytest
cargo test
```

## Generalized-clock timeseries statistics

The canonical specification is `docs/timeseries.md`. Read it before editing anything under:

- `src/timeseries/`
- `src/expressions/timeseries.rs`
- `python/polars_faer/timeseries.py`
- the timeseries namespace methods in `python/polars_faer/namespaces.py`
- `tests/test_timeseries.py` and `tests/reference.py`

Public API:

- `polars_faer.timeseries.rolling_sum`
- `polars_faer.timeseries.rolling_mean`
- `polars_faer.timeseries.rolling_variance`
- `polars_faer.timeseries.rolling_covariance`
- `polars_faer.timeseries.rolling_correlation`
- `polars_faer.timeseries.ewm_sum`
- `polars_faer.timeseries.ewm_mean`
- `polars_faer.timeseries.ewm_variance`
- `polars_faer.timeseries.ewm_covariance`
- `polars_faer.timeseries.ewm_correlation`

Convenience namespace: `Expr.faer.<operation>`, which wraps the functions above and holds no
numerical logic of its own.

Invariants:

- One output per input row, in the order it was given.
- Clocks are validated as non-null, finite and non-decreasing, and are never sorted silently.
- Numeric spans use the units of the numeric clock; temporal spans use `datetime.timedelta`.
- String durations such as `"10000i"` are not accepted.
- Exponential weighting uses normalised decaying-observation weights, so every valid
  observation enters with weight one.
- Pairwise statistics use one joint validity mask.
- The output dtype is always `Float64`.
- User input never causes a Rust panic: it returns a `PolarsResult` with a message that says
  what was wrong and what would fix it.
- Do not change numerical semantics without updating `docs/timeseries.md`.

## The rest of the package

Dense operations over a whole group: `least_squares`, `pca`, `pca_transform`, `covariance`,
`correlation`, `solve_spd`, and mergeable states for partitioned data. The README documents
their input and result contracts.

Two things worth knowing before changing them:

- Inside `group_by().agg()` Polars passes a plugin its inputs without names, so every
  operation sends the caller's column names as a keyword argument (`_typing.output_names`)
  rather than reading `series.name()`.
- Lockfiles are deliberately not committed.
