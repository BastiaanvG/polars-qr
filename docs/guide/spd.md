# Positive-definite systems

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

The matrix is read from a wide frame: one column per matrix column, one row per matrix row.
`row_index` says which row is which, and the rows are put in its order before anything is
read, so the answer does not depend on the order the frame happens to be in. The index must
be an integer column and may not repeat a value.

The matrix is checked for symmetry, shifted along the diagonal if asked, factorised once,
and every right-hand side is solved against that one factorisation.

## The diagonal shift

A sample covariance matrix is often only just positive definite, or just short of it. A
small `diagonal_shift` is what makes it solvable, and the shift that was applied comes back
in the result so it is visible in whatever you do with the answer.

```python
frame.select(pq.solve_spd(columns, "rhs", row_index="row"))
# ComputeError: the matrix is not positive definite; a larger diagonal_shift than 0 may make it so
```

## Symmetry

`symmetry_error` reports how far the two triangles disagreed, relative to the largest entry.
A matrix whose triangles differ by more than 1e-8 is rejected rather than solved, because at
that point it is not clear which half was meant. Within the tolerance the two are averaged.

## Nulls

There is no null policy here. Dropping a row would change the shape of the matrix and leave
it no longer square, so a null anywhere in the matrix or the right-hand sides is an error.
