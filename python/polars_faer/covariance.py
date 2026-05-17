import polars as pl

from polars_faer._plugin import plugin_expr
from polars_faer._typing import IntoExpr, IntoExprColumns, NullPolicy, as_expressions, output_names

__all__ = ["correlation", "covariance"]


def covariance(
    features: IntoExprColumns,
    *,
    weights: IntoExpr | None = None,
    ddof: float = 1.0,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """Estimate the covariance of `features`.

    Parameters
    ----------
    features
        The columns to estimate over. Their order is the order of both axes of the matrix.
    weights
        An optional column of observation weights, which must be finite and non-negative.
        They are read as reliability weights: scaling all of them by the same factor leaves
        the estimate unchanged, and the divisor is corrected accordingly.
    ddof
        The delta degrees of freedom: the divisor is the number of observations minus this.
        The default of 1 gives the usual sample covariance. Under weights the divisor is
        the sum of the weights minus `ddof` times the sum of their squares over their sum.
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
    `sum_weights`
        The sum of the weights, which is the observation count when there are none.
    `size`
        The number of features, which is the size of both axes.
    """
    columns = as_expressions(features)
    weight_columns = [] if weights is None else as_expressions(weights)
    return plugin_expr(
        "covariance",
        [*columns, *weight_columns],
        {
            "names": output_names([*columns, *weight_columns]),
            "ddof": ddof,
            "weighted": weights is not None,
            "normalise": False,
            "null_policy": null_policy,
        },
        returns_scalar=True,
    )


def correlation(
    features: IntoExprColumns,
    *,
    weights: IntoExpr | None = None,
    ddof: float = 1.0,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """Estimate the correlation of `features`.

    The estimate is the covariance divided through by the standard deviations, so it reads
    the same inputs and follows the same policies as :func:`covariance`.

    Parameters
    ----------
    features
        The columns to estimate over. Their order is the order of both axes of the matrix.
    weights
        An optional column of observation weights, read the same way as in :func:`covariance`.
    ddof
        The delta degrees of freedom used for the underlying covariance. It cancels out of
        the correlations themselves, but it is still what the reported standard deviations
        are computed with.
    null_policy
        `"raise"` to fail on a row that is null or not finite, `"drop"` to leave it out.
        Rows are dropped jointly, so every entry of the matrix is estimated from the same
        sample.

    Returns
    -------
    An expression producing one struct per group, with the same fields as
    :func:`covariance`, except that the matrix is called `correlation`. Its diagonal is
    exactly one. A column that does not vary has no correlation to report, and its row and
    column come back as NaN.
    """
    columns = as_expressions(features)
    weight_columns = [] if weights is None else as_expressions(weights)
    return plugin_expr(
        "correlation",
        [*columns, *weight_columns],
        {
            "names": output_names([*columns, *weight_columns]),
            "ddof": ddof,
            "weighted": weights is not None,
            "normalise": True,
            "null_policy": null_policy,
        },
        returns_scalar=True,
    )
