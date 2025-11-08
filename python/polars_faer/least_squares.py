import polars as pl

from polars_faer._plugin import plugin_expr
from polars_faer._typing import IntoExprColumns, NullPolicy, as_expressions

__all__ = ["least_squares"]


def least_squares(
    targets: IntoExprColumns,
    features: IntoExprColumns,
    *,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """Fit `targets` against `features` in the least-squares sense.

    Parameters
    ----------
    targets
        The columns to fit. Several targets share one feature matrix and one sample, so
        they are fitted in a single factorisation.
    features
        The columns to fit them against. Their order is the order of the coefficients.
    null_policy
        `"raise"` to fail on a row that is null or not finite, `"drop"` to leave it out.
        Rows are read jointly, so a row is either used by every column or by none.

    Returns
    -------
    An expression producing one struct per group, with the fields:

    `features`
        The feature names, in the order they were given.
    `targets`
        The target names.
    `coefficients`
        One list of coefficients per target, ordered like `features`.
    `n_observations`
        The number of rows the fit used.
    `residual_sum_of_squares`
        The squared norm of the residual, one entry per target.
    """
    target_columns = as_expressions(targets)
    feature_columns = as_expressions(features)
    return plugin_expr(
        "least_squares",
        [*target_columns, *feature_columns],
        {"n_targets": len(target_columns), "null_policy": null_policy},
        returns_scalar=True,
    )
