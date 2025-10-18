# polars-faer

Dense numerical operations for [Polars](https://pola.rs), backed by
[faer](https://github.com/sarah-quinones/faer-rs).

The package is an expression plugin: every operation is a Polars expression, so it composes
with grouping, lazy execution and the rest of a query plan.

## Development

The extension is built with [maturin](https://www.maturin.rs) and needs a Rust toolchain.

```bash
uv sync
uv run maturin develop --uv
uv run pytest
```
