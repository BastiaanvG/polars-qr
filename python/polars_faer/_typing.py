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


def output_names(expressions: Sequence[pl.Expr]) -> list[str]:
    """Name each expression the way its output column will be named.

    The names have to be worked out here rather than read off the data: inside a grouped
    aggregation Polars hands a plugin its inputs without names, and a result that labels
    itself would have nothing to label itself with.

    Parameters
    ----------
    expressions
        The expressions whose output names are wanted.
    """
    names = []
    for position, expression in enumerate(expressions):
        name = expression.meta.output_name(raise_if_undetermined=False)
        names.append(name if name else f"column_{position}")
    return names
