# polars-qr

A Polars expression plugin: dense numerical operations written in Rust, backed by the faer
crate, reached from Python as ordinary expressions.

## Commands

```bash
uv sync
uv run maturin develop --release --uv   # rebuild after any change under src/
uv run pytest
cargo test
uv run mkdocs serve
```

The Rust side is only rebuilt by `maturin develop`. A Python-only change needs no rebuild; a
change under `src/` that is not rebuilt will appear to have done nothing.

## Conventions that are not the tool defaults

- **Prose and identifiers are British**: `finalise`, `centred`, `normalise`, `summarise`.
  This reaches the public API, so `finalise_covariance` is the function name.
- **Errors never panic on user input.** Anything reachable from Python returns a
  `PolarsResult` whose message says what was wrong, what was received and what would fix it.
  No `unwrap`, `expect` or unchecked indexing on user data.
- **Column names come from the expressions, not the data.** Polars hands a plugin its inputs
  without names inside `group_by().agg()`, so names travel as a keyword argument built by
  `_typing.output_names`. Reading `series.name()` gives empty strings in a grouped query.
- **Lockfiles are deliberately not committed**, so a build resolves against the floors in
  `pyproject.toml` and `Cargo.toml`.
- **Comments say why, not what.** A comment that restates the line above it is noise; one
  that explains a numerical choice, a boundary or a trap is the point.
- **Tests are named as sentences** that state the behaviour: `a_null_lets_the_state_decay_past_it`.

## Where things live

Guides and the API reference are in `docs/`, published with MkDocs. The public contracts —
what an operation reads, what it returns, how nulls and weights are treated — are in
`docs/guide/contracts.md`, and each operation family has a guide page beside it.

The Python package is one module per operation family under `python/polars_qr/`, each one
thin: it normalises its arguments, works out the output names, and hands over to a plugin
function of the same name in `src/expressions/`. The numerics live under `src/`, in a
module that knows nothing about Polars beyond its error type.

`pq.timeseries` and `pq.autoregression` are separate namespaces on purpose. A clock measures
how far apart two observations are and a lag counts rows; neither argument makes sense in
the other's family.

## Before changing numerical behaviour

Update, together: the Rust implementation, its unit tests, the Python docstring, the guide
page under `docs/guide/`, and at least one example. A change that alters a documented result
without touching the docs is a bug in the change.
