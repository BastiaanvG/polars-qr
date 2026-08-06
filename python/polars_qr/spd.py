import polars as pl

from polars_qr._plugin import plugin_expr
from polars_qr._typing import IntoExpr, IntoExprColumns, as_expressions, output_names

__all__ = ["solve_spd"]


def solve_spd(
    matrix: IntoExprColumns,
    rhs: IntoExprColumns,
    *,
    row_index: IntoExpr,
    diagonal_shift: float = 0.0,
) -> pl.Expr:
    """Solve `matrix @ x = rhs` through one Cholesky factorisation.

    The matrix is read from a wide frame: one column per matrix column, one row per matrix
    row. `row_index` says which row is which, and the rows are put in its order before the
    matrix is read, so the result does not depend on the order the frame happens to be in.

    Every right-hand side is solved against the same factorisation.

    Parameters
    ----------
    matrix
        The columns holding the matrix. Their number is the size of both axes.
    rhs
        The columns holding the right-hand sides, one column each.
    row_index
        An integer column giving the order of the rows. It may not repeat a value.
    diagonal_shift
        A value added to every diagonal entry before factorising. A small shift is what
        makes a matrix that is only just short of positive definite solvable.

    Returns
    -------
    An expression producing one struct per group, with the fields:

    `rows`
        The matrix column names, which name both axes.
    `rhs`
        The right-hand side names.
    `solution`
        One solution per right-hand side, each a list over `rows`.
    `size`
        The size of the matrix.
    `symmetry_error`
        How far the two triangles of the matrix disagreed, relative to its largest entry.
        A matrix whose triangles differ by more than 1e-8 is rejected rather than solved.
    `diagonal_shift`
        The shift that was applied.

    Notes
    -----
    Rows cannot be dropped without changing the shape of the matrix, so this operation has
    no null policy: a null anywhere in the matrix or the right-hand sides is an error.
    """
    matrix_columns = as_expressions(matrix)
    rhs_columns = as_expressions(rhs)
    index_column = as_expressions(row_index)
    return plugin_expr(
        "solve_spd",
        [*matrix_columns, *rhs_columns, *index_column],
        {
            "names": output_names([*matrix_columns, *rhs_columns, *index_column]),
            "n_matrix": len(matrix_columns),
            "n_rhs": len(rhs_columns),
            "diagonal_shift": diagonal_shift,
        },
        returns_scalar=True,
    )
