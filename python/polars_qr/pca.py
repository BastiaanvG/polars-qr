import polars as pl

from polars_qr._plugin import plugin_expr
from polars_qr._typing import IntoExprColumns, NullPolicy, as_expressions, output_names

__all__ = ["pca", "pca_transform"]


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
        The loadings, as a list of components; each one is a list over `features`. The sign
        of a component is fixed so that its largest entry is positive, which keeps two runs
        over the same data comparable.
    `singular_values`
        The singular values of the centred and scaled matrix, one per component.
    `explained_variance`
        The variance along each component: its singular value squared, over one degree of
        freedom less than the number of observations.
    `explained_variance_ratio`
        The share of the total variance each component carries. The total counts every
        direction the data spans, so keeping fewer components does not inflate the shares.
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
            "names": output_names(columns),
            "n_components": n_components,
            "centre": centre,
            "scale": scale,
            "null_policy": null_policy,
        },
        returns_scalar=True,
    )


def pca_transform(
    features: IntoExprColumns,
    *,
    n_components: int,
    centre: bool = True,
    scale: bool = False,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """Score every row on the components found from the rows it is read with.

    The expression keeps its rows, so it can sit next to the input in the same frame. Use
    `.over(...)` to score each group on its own components.

    Parameters
    ----------
    features
        The columns to decompose and project.
    n_components
        How many components to score on. It is required here, unlike in :func:`pca`,
        because it fixes the number of output columns.
    centre
        Whether to subtract the column means before decomposing.
    scale
        Whether to divide the columns through by their standard deviations first.
    null_policy
        `"raise"` to fail on a row that is null or not finite, `"drop"` to leave it out of
        the decomposition. A dropped row still appears in the result, scoring null.

    Returns
    -------
    An expression producing a struct with one `component_i` field per component, aligned
    with the rows it was given.
    """
    columns = as_expressions(features)
    return plugin_expr(
        "pca_transform",
        columns,
        {
            "names": output_names(columns),
            "n_components": n_components,
            "centre": centre,
            "scale": scale,
            "null_policy": null_policy,
        },
    )
