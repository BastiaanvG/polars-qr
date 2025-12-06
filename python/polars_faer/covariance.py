import polars as pl

from polars_faer._plugin import plugin_expr
from polars_faer._typing import IntoExprColumns, NullPolicy, as_expressions

__all__ = ["covariance"]


def covariance(
    features: IntoExprColumns,
    *,
    ddof: float = 1.0,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """Estimate the covariance of `features`.

    Parameters
    ----------
    features
        The columns to estimate over. Their order is the order of both axes of the matrix.
    ddof
        The delta degrees of freedom: the divisor is the number of observations minus this.
        The default of 1 gives the usual sample covariance.
    null_policy
        `"raise"` to fail on a row that is null or not finite, `"drop"` to leave it out.
        Rows are dropped jointly, so every entry of the matrix is estimated from the same
        sample and the result stays usable as input to a factorisation.

    Returns
    -------
    An expression producing one struct per group, with the fields:

    `features`
        The feature names, in the order they were given.
    `means`
        The column means.
    `standard_deviations`
        The square roots of the diagonal of the matrix.
    `covariance`
        The matrix, as a list of its rows.
    `n_observations`
        The number of rows the estimate used.
    `size`
        The number of features, which is the size of both axes.
    """
    columns = as_expressions(features)
    return plugin_expr(
        "covariance",
        columns,
        {"ddof": ddof, "null_policy": null_policy},
        returns_scalar=True,
    )
