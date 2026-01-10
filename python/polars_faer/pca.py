import polars as pl

from polars_faer._plugin import plugin_expr
from polars_faer._typing import IntoExprColumns, NullPolicy, as_expressions

__all__ = ["pca"]


def pca(
    features: IntoExprColumns,
    *,
    n_components: int | None = None,
    centre: bool = True,
    scale: bool = False,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """Find the principal components of `features`.

    Parameters
    ----------
    features
        The columns to decompose. Their order is the order of the loadings.
    n_components
        How many components to keep. The default keeps every component the data supports,
        which is the smaller of the number of observations and the number of features.
    centre
        Whether to subtract the column means before decomposing.
    scale
        Whether to divide the columns through by their standard deviations, computed with
        one degree of freedom. A column that does not vary is left alone and its reported
        scale is one.
    null_policy
        `"raise"` to fail on a row that is null or not finite, `"drop"` to leave it out.

    Returns
    -------
    An expression producing one struct per group, with the fields:

    `features`
        The feature names, in the order they were given.
    `means`
        The means that were subtracted, or zeros when `centre` is false.
    `scales`
        The scales the columns were divided by, or ones when `scale` is false.
    `components`
        The loadings, as a list of components; each one is a list over `features`.
    `singular_values`
        The singular values of the centred and scaled matrix, one per component.
    `rank`
        The numerical rank of that matrix.
    `n_observations`
        The number of rows the components were found from.
    """
    columns = as_expressions(features)
    return plugin_expr(
        "pca",
        columns,
        {
            "n_components": n_components,
            "centre": centre,
            "scale": scale,
            "null_policy": null_policy,
        },
        returns_scalar=True,
    )
