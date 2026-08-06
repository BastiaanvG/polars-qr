---
paths:
  - "src/**/*.rs"
---

# The Rust side

## Shape of a plugin entry point

Everything under `src/expressions/` decodes options, reads inputs, calls one thing and builds
a result. No arithmetic lives there. The numerics live in their own module with their own
unit tests, so they can be tested without Python in the loop.

```rust
#[polars_expr(output_type=Float64)]
fn ewm_univariate(inputs: &[Series], kwargs: EwmKwargs) -> PolarsResult<Series> {
    let (values, clock) = read_univariate(inputs, operation, kwargs.null_policy)?;
    let half_life = clock.resolve(kwargs.half_life, operation, "half_life")?;
    ...
}
```

Options arrive as typed structs deserialised from the keyword arguments. Strings are not
parsed into meaning on the Rust side: an enum with `#[serde(rename_all = "snake_case")]`
does that, so an unknown value fails at the boundary.

## Errors

User data never reaches `unwrap`, `expect`, unchecked indexing or an assertion that can take
the interpreter down with it. Every failure is a `PolarsResult` whose message names the
operation, the input, the offending value and the way out. `src/timeseries/error.rs` is the
pattern: constructors that build the message once, so the same fault reads the same way
wherever it is raised.

Polars reports every error raised inside a plugin as a `ComputeError` regardless of the
variant chosen, so tests match on the message rather than the type.

## Numerics

- Prefer centred, incremental updates to sums of squares. `sum(x²) - mean·sum(x)` cancels
  catastrophically on data that sits far from zero.
- Anything that can be checked once, before the loop, is checked once: a half-life that is
  not positive, a window narrower than its own minimum span, a clock that goes backwards.
- Say in a comment why a tolerance is the size it is. A bare `1e-12` tells the next reader
  nothing.

## Tests

Unit tests sit in the module they test, in a `#[cfg(test)] mod tests`. They are the fast
feedback loop — `cargo test` needs no Python build — so a numerical rule should have one
here as well as an end-to-end test on the Python side.

Small analytic cases beat generated data: values whose mean, variance and covariance can be
worked out by hand make a failure legible.
