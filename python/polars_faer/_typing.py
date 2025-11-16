from collections.abc import Sequence
from typing import Literal

import polars as pl

IntoExpr = str | pl.Expr

IntoExprColumns = IntoExpr | Sequence[IntoExpr]

NullPolicy = Literal["raise", "drop"]

Solver = Literal["qr", "svd"]


def as_expressions(columns: IntoExprColumns) -> list[pl.Expr]:
    """Normalise a column argument into a list of expressions.

    Parameters
    ----------
    columns
        A single column, or a sequence of them. Strings name a column; expressions are
        taken as they are.

    Raises
    ------
    ValueError
        If no column is given.
    """
    if isinstance(columns, (str, pl.Expr)):
        columns = [columns]
    expressions = [pl.col(column) if isinstance(column, str) else column for column in columns]
    if not expressions:
        message = "at least one column is required"
        raise ValueError(message)
    return expressions
