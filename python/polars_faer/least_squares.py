import polars as pl

from polars_faer._plugin import plugin_expr
from polars_faer._typing import IntoExpr, IntoExprColumns, NullPolicy, Solver, as_expressions

__all__ = ["least_squares"]


def least_squares(
    targets: IntoExprColumns,
    features: IntoExprColumns,
    *,
    weights: IntoExpr | None = None,
    intercept: bool = False,
    solver: Solver = "qr",
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
    weights
        An optional column of observation weights, which must be finite and non-negative.
        The fit minimises the weighted sum of squared residuals, and the reported residual
        sums of squares are weighted too.
    intercept
        Whether to fit a constant term alongside the features. It is reported on its own,
        so `features` and `coefficients` keep lining up.
    solver
        `"qr"` for a QR factorisation, which is the faster route and needs the features to
        have full column rank, or `"svd"` for a thin SVD, which also solves rank-deficient
        and underdetermined systems and returns the solution of smallest norm.
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
    `intercept`
        The constant term, one entry per target, or null when none was fitted.
    `n_observations`
        The number of rows the fit used.
    `residual_sum_of_squares`
        The squared norm of the residual, one entry per target.
    """
    target_columns = as_expressions(targets)
    feature_columns = as_expressions(features)
    weight_columns = [] if weights is None else as_expressions(weights)
    return plugin_expr(
        "least_squares",
        [*target_columns, *feature_columns, *weight_columns],
        {
            "n_targets": len(target_columns),
            "weighted": weights is not None,
            "intercept": intercept,
            "solver": solver,
            "null_policy": null_policy,
        },
        returns_scalar=True,
    )
