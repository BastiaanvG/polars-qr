"""Mergeable state for the operations that have one.

A state is a summary of some rows that is much smaller than the rows themselves and can be
merged with another summary. Each partition summarises what it holds, the summaries are
merged in any order, and the result is finalised once. States travel as binary values, so
they can be written to a file, sent between processes or stored in a table.
"""

import polars as pl

from polars_faer._plugin import plugin_expr
from polars_faer._typing import (
    IntoExpr,
    IntoExprColumns,
    NullPolicy,
    Solver,
    as_expressions,
    output_names,
)

__all__ = [
    "covariance_state",
    "finalise_correlation",
    "finalise_covariance",
    "finalise_least_squares",
    "finalise_pca",
    "least_squares_state",
    "merge_covariance_states",
    "merge_least_squares_states",
]


def least_squares_state(
    targets: IntoExprColumns,
    features: IntoExprColumns,
    *,
    weights: IntoExpr | None = None,
    intercept: bool = False,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """Summarise a least-squares problem over the rows of one partition.

    The summary is the triangular factor of a QR factorisation of the features with the
    targets appended, which is what lets a fit be assembled from partitions without ever
    forming the normal equations, whose conditioning is the square of the data's.

    Its size is quadratic in the number of features plus targets, and does not depend on
    the number of rows.

    Parameters
    ----------
    targets
        The columns to fit.
    features
        The columns to fit them against. Their order is fixed in the state.
    weights
        An optional column of observation weights, read as in :func:`least_squares`.
    intercept
        Whether the summarised design carries a constant term.
    null_policy
        `"raise"` to fail on a row that is null or not finite, `"drop"` to leave it out.

    Returns
    -------
    An expression producing one binary state per group.
    """
    target_columns = as_expressions(targets)
    feature_columns = as_expressions(features)
    weight_columns = [] if weights is None else as_expressions(weights)
    return plugin_expr(
        "least_squares_state",
        [*target_columns, *feature_columns, *weight_columns],
        {
            "names": output_names([*target_columns, *feature_columns, *weight_columns]),
            "n_targets": len(target_columns),
            "weighted": weights is not None,
            "intercept": intercept,
            "null_policy": null_policy,
        },
        returns_scalar=True,
    )


def merge_least_squares_states(state: IntoExpr) -> pl.Expr:
    """Merge every state in `state` into one.

    Merging is associative and commutative up to floating point, so the result does not
    depend on how the rows were partitioned or on the order the partitions arrive in.

    Parameters
    ----------
    state
        A column of states built by :func:`least_squares_state`, or by an earlier merge.

    Returns
    -------
    An expression producing one binary state per group.
    """
    return plugin_expr(
        "merge_least_squares_states",
        as_expressions(state),
        returns_scalar=True,
    )


def finalise_least_squares(
    state: IntoExpr,
    *,
    solver: Solver = "qr",
    l2_penalty: float = 0.0,
) -> pl.Expr:
    """Solve the problem a state summarises.

    The solver and the penalty are chosen here rather than when the state was built, so one
    set of summaries can be finalised several ways. A penalty is applied once, at this
    point, however many partitions the state was merged from.

    Parameters
    ----------
    state
        A column of states, usually the output of :func:`merge_least_squares_states`.
    solver
        `"qr"` or `"svd"`, as in :func:`least_squares`.
    l2_penalty
        A ridge penalty on the coefficients, applied once here.

    Returns
    -------
    An expression producing the same struct as :func:`least_squares`, one per state.
    """
    return plugin_expr(
        "finalise_least_squares",
        as_expressions(state),
        {"solver": solver, "l2_penalty": l2_penalty},
        is_elementwise=True,
    )


def covariance_state(
    features: IntoExprColumns,
    *,
    weights: IntoExpr | None = None,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """Summarise the second moments of `features` over the rows of one partition.

    The summary is the count, the weight, the mean of each column and the cross-products of
    the columns about those means. Its size is quadratic in the number of columns and does
    not depend on the number of rows.

    Parameters
    ----------
    features
        The columns to summarise. Their order is fixed in the state.
    weights
        An optional column of observation weights, read as in :func:`covariance`.
    null_policy
        `"raise"` to fail on a row that is null or not finite, `"drop"` to leave it out.

    Returns
    -------
    An expression producing one binary state per group.
    """
    columns = as_expressions(features)
    weight_columns = [] if weights is None else as_expressions(weights)
    return plugin_expr(
        "covariance_state",
        [*columns, *weight_columns],
        {
            "names": output_names([*columns, *weight_columns]),
            "weighted": weights is not None,
            "null_policy": null_policy,
        },
        returns_scalar=True,
    )


def merge_covariance_states(state: IntoExpr) -> pl.Expr:
    """Merge every state in `state` into one.

    Merging shifts both means onto the mean of the two together and corrects the
    cross-products for the shift, so the result does not depend on how the rows were
    partitioned or on the order the partitions arrive in.

    Parameters
    ----------
    state
        A column of states built by :func:`covariance_state`, or by an earlier merge.

    Returns
    -------
    An expression producing one binary state per group.
    """
    return plugin_expr(
        "merge_covariance_states",
        as_expressions(state),
        returns_scalar=True,
    )


def finalise_covariance(state: IntoExpr, *, ddof: float = 1.0) -> pl.Expr:
    """Turn a state into the covariance it summarises.

    Parameters
    ----------
    state
        A column of states, usually the output of :func:`merge_covariance_states`.
    ddof
        The delta degrees of freedom, chosen here rather than when the state was built.

    Returns
    -------
    An expression producing the same struct as :func:`covariance`, one per state.
    """
    return plugin_expr(
        "finalise_covariance",
        as_expressions(state),
        {"ddof": ddof, "normalise": False},
        is_elementwise=True,
    )


def finalise_correlation(state: IntoExpr, *, ddof: float = 1.0) -> pl.Expr:
    """Turn a state into the correlation it summarises.

    Parameters
    ----------
    state
        A column of states, usually the output of :func:`merge_covariance_states`.
    ddof
        The delta degrees of freedom used for the underlying covariance.

    Returns
    -------
    An expression producing the same struct as :func:`correlation`, one per state.
    """
    return plugin_expr(
        "finalise_correlation",
        as_expressions(state),
        {"ddof": ddof, "normalise": True},
        is_elementwise=True,
    )


def finalise_pca(
    state: IntoExpr,
    *,
    n_components: int | None = None,
    scale: bool = False,
) -> pl.Expr:
    """Find the principal components a covariance state implies.

    The components of a set of columns are the eigenvectors of their covariance, so a
    summary that carries the covariance carries the components too. This is the route a
    partitioned decomposition has to take, and it is a little less precise than decomposing
    the rows themselves: forming the covariance squares the conditioning of the data.

    Parameters
    ----------
    state
        A column of states, usually the output of :func:`merge_covariance_states`.
    n_components
        How many components to keep. The default keeps one per feature.
    scale
        Whether to standardise the columns first, which is the same as decomposing their
        correlation instead of their covariance.

    Returns
    -------
    An expression producing the same struct as :func:`pca`, one per state.
    """
    return plugin_expr(
        "finalise_pca",
        as_expressions(state),
        {"n_components": n_components, "scale": scale},
        is_elementwise=True,
    )
