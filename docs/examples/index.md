# Examples

Every script here runs on data it generates itself, prints what it found, and is executed
by the test suite, so none of it can drift away from the code. Each one is complete: copy
it, run it, and change the numbers.

```bash
uv run python examples/01_least_squares.py
```

## Least squares

Fitting one target, then several against the same feature matrix, then weighting the
observations and shrinking the coefficients.

```python title="examples/01_least_squares.py"
--8<-- "examples/01_least_squares.py"
```

## Grouped and lazy

A fit per date inside one lazy query, and scores that land next to the rows they came from.

```python title="examples/02_grouped_and_lazy.py"
--8<-- "examples/02_grouped_and_lazy.py"
```

## Principal components

Eight columns driven by three underlying factors, decomposed with and without standardising
first.

```python title="examples/03_principal_components.py"
--8<-- "examples/03_principal_components.py"
```

## Covariance and solving

Estimating a covariance matrix, weighting the observations, and solving a system against
the result.

```python title="examples/04_covariance_and_solve.py"
--8<-- "examples/04_covariance_and_solve.py"
```

## Partitioned data

Summarising each partition, merging the summaries and finalising once.

```python title="examples/05_partitioned_data.py"
--8<-- "examples/05_partitioned_data.py"
```

## Statistics against a clock

A volume clock and a variance clock, with decaying and windowed statistics measured against
each.

```python title="examples/06_timeseries.py"
--8<-- "examples/06_timeseries.py"
```

## Autoregression

Order selection by BIC, the diagnostics that come with the fit, and whitening a series with
its own model.

```python title="examples/07_autoregression.py"
--8<-- "examples/07_autoregression.py"
```
