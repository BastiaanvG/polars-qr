"""Autoregressive models, and the structure that makes them cheap.

An autoregression is a least squares problem, and :func:`polars_qr.least_squares` already
solves those. What it is not is a *general* least squares problem: the matrix of a series
against its own lags is Toeplitz, every diagonal of it constant. The Levinson-Durbin
recursion uses that, solving in `O(p²)` rather than `O(p³)`, reading the series into `p + 1`
autocovariances rather than materialising `p` lag columns, and passing through every lower
order on the way, so the partial autocorrelations and the per-order variances that order
selection needs are already in hand when it finishes.

A lag here is a row and not a duration. The model assumes one step of the sequence is like
any other, which is why none of this lives in `polars_qr.timeseries`, where the distance
between two observations is a number the caller supplies. If your observations are unevenly
spaced, a fit over them describes the sequence and not the process in time.
"""

import polars as pl

from polars_qr._plugin import plugin_expr
from polars_qr._typing import (
    AutoregressionMethod,
    AutoregressionNullPolicy,
    AutoregressionOrder,
    AutoregressionOutput,
    IntoExpr,
    IntoExprColumns,
    as_expressions,
    output_names,
)

CRITERIA = ("aic", "bic", "hqic")

__all__ = [
    "autocorrelation",
    "autocovariance",
    "fit",
    "partial_autocorrelation",
    "solve_toeplitz",
    "transform",
]


def _one(column: IntoExpr) -> pl.Expr:
    """Read a single column argument."""
    return as_expressions(column)[0]


def _order(order: AutoregressionOrder, max_order: int | None) -> dict[str, object]:
    """Say either which order to fit or which criterion to choose one by.

    Raises
    ------
    TypeError
        If `order` is neither an integer nor the name of a criterion.
    ValueError
        If an integer order is negative, or a criterion is given without a `max_order` to
        search up to.
    """
    if isinstance(order, bool) or not isinstance(order, (int, str)):
        message = f"order must be an integer or one of {CRITERIA}, not {type(order).__name__}"
        raise TypeError(message)
    if isinstance(order, int):
        if order < 0:
            message = f"order must not be negative, but it is {order}"
            raise ValueError(message)
        return {"order": order, "criterion": None, "max_order": order}
    if order not in CRITERIA:
        message = f"order must be an integer or one of {CRITERIA}, but it is {order!r}"
        raise ValueError(message)
    if max_order is None:
        message = f"order={order!r} chooses an order, so it needs a max_order to choose from"
        raise ValueError(message)
    if max_order < 1:
        message = f"max_order must be at least 1, but it is {max_order}"
        raise ValueError(message)
    return {"order": None, "criterion": order, "max_order": max_order}


def autocovariance(
    value: IntoExpr,
    *,
    max_lag: int,
    demean: bool = True,
    unbiased: bool = False,
    null_policy: AutoregressionNullPolicy = "raise",
) -> pl.Expr:
    """Estimate the covariance of a series with itself at every lag up to `max_lag`.

    Parameters
    ----------
    value
        The column to measure. Its row order is the sequence.
    max_lag
        The largest lag to report. It must be below the number of rows in the group.
    demean
        Whether to subtract the mean of the series before measuring.
    unbiased
        Whether to divide each lag by the number of products at that lag rather than by the
        length of the series. The unbiased estimate can produce a sequence that is not
        positive semi-definite, which is why it is offered here for reading and never used
        to build a model.
    null_policy
        `"raise"` to fail on a null, `"zero"` to treat it as an observation at the mean, so
        that it contributes nothing to any lagged product. A null row still counts towards
        the divisor.

    Returns
    -------
    An expression producing one struct per group, with the fields:

    `series`
        The name of the column that was measured.
    `lags`
        The lags, from 0 to `max_lag`.
    `autocovariance`
        The estimate at each of those lags.
    `mean`
        What was subtracted, or 0.0 when `demean=False`.
    `n_observations`
        How many rows were read.
    """
    return _sequence(
        "autocovariance",
        value,
        max_lag=max_lag,
        demean=demean,
        unbiased=unbiased,
        null_policy=null_policy,
    )


def autocorrelation(
    value: IntoExpr,
    *,
    max_lag: int,
    demean: bool = True,
    unbiased: bool = False,
    null_policy: AutoregressionNullPolicy = "raise",
) -> pl.Expr:
    """Estimate the correlation of a series with itself at every lag up to `max_lag`.

    This is :func:`autocovariance` divided through by its value at lag zero, so the field
    holding it is called `autocorrelation` and the value at lag zero is 1.

    The arguments are those of :func:`autocovariance`.
    """
    return _sequence(
        "autocorrelation",
        value,
        max_lag=max_lag,
        demean=demean,
        unbiased=unbiased,
        null_policy=null_policy,
    )


def _sequence(
    operation: str,
    value: IntoExpr,
    *,
    max_lag: int,
    demean: bool,
    unbiased: bool,
    null_policy: AutoregressionNullPolicy,
) -> pl.Expr:
    column = _one(value)
    return plugin_expr(
        operation,
        [column],
        {
            "names": output_names([column]),
            "max_lag": max_lag,
            "demean": demean,
            "unbiased": unbiased,
            "null_policy": null_policy,
        },
        returns_scalar=True,
    )


def partial_autocorrelation(
    value: IntoExpr,
    *,
    max_lag: int,
    method: AutoregressionMethod = "yule_walker",
    demean: bool = True,
    null_policy: AutoregressionNullPolicy = "raise",
) -> pl.Expr:
    """Measure how much of a lag is left once the shorter lags are accounted for.

    These are the reflection coefficients the recursion produces on its way up, so they
    cost nothing beyond the fit itself. The value at lag `k` is the last coefficient of a
    fit of order `k`, which is what a partial autocorrelation is defined to be.

    Parameters
    ----------
    value
        The column to measure.
    max_lag
        The largest lag to report.
    method
        `"yule_walker"` or `"burg"`; see :func:`fit`.
    demean
        Whether to subtract the mean of the series before measuring.
    null_policy
        As in :func:`autocovariance`.

    Returns
    -------
    An expression producing one struct per group, with the fields:

    `series`
        The name of the column that was measured.
    `lags`
        The lags, from 1 to `max_lag`.
    `partial_autocorrelation`
        The estimate at each of those lags.
    `n_observations`
        How many rows were read.
    """
    column = _one(value)
    return plugin_expr(
        "partial_autocorrelation",
        [column],
        {
            "names": output_names([column]),
            **_order(max_lag, None),
            "method": method,
            "demean": demean,
            "null_policy": null_policy,
        },
        returns_scalar=True,
    )


def fit(
    value: IntoExpr,
    *,
    order: AutoregressionOrder,
    max_order: int | None = None,
    method: AutoregressionMethod = "yule_walker",
    demean: bool = True,
    null_policy: AutoregressionNullPolicy = "raise",
) -> pl.Expr:
    """Fit an autoregression to a series.

    Parameters
    ----------
    value
        The column to fit. Its row order is the sequence, and the frame is never sorted
        silently: sorting it would change which row each lag refers to.
    order
        The order to fit, or `"aic"`, `"bic"` or `"hqic"` to choose one. A criterion needs
        a `max_order` to choose from, and order 0 is a legal outcome, meaning the series is
        white noise as far as the criterion can tell. `"bic"` is the conservative choice;
        `"aic"` is not order-consistent and over-selects on long series.
    max_order
        The highest order to consider. Ignored when `order` is an integer.
    method
        `"yule_walker"` solves the system built from the biased autocovariance sequence.
        `"burg"` minimises the forward and backward prediction errors together, which is
        better on short series. Both give a stationary fit.
    demean
        Whether to subtract the mean of the series before fitting.
    null_policy
        As in :func:`autocovariance`.

    Returns
    -------
    An expression producing one struct per group, with the fields:

    `series`
        The name of the column that was fitted.
    `coefficients`
        The coefficients of the chosen order, lag one first.
    `mean`
        What was subtracted, or 0.0 when `demean=False`.
    `variance`
        The white-noise variance the chosen order leaves behind.
    `order`
        The order that was fitted.
    `partial_autocorrelations`
        The reflection coefficient at every order from 1 to `max_order`.
    `order_variance`
        The prediction-error variance at every order, index 0 being the variance of the
        series itself. An order-selection plot needs no second query.
    `criterion`
        The criterion at every order, or null when an order was given rather than chosen.
    `n_observations`
        How many rows were read.
    `stationary`
        Whether every reflection coefficient came out inside the unit circle.
    `method`
        Which estimator produced it.

    Notes
    -----
    Fitting the same series through lag columns and :func:`polars_qr.least_squares` is a
    different estimator: it materialises `p` columns of `n` values, costs `O(p³)`, and
    offers no guarantee that the fit it returns is stationary.
    """
    column = _one(value)
    return plugin_expr(
        "autoregression",
        [column],
        {
            "names": output_names([column]),
            **_order(order, max_order),
            "method": method,
            "demean": demean,
            "null_policy": null_policy,
        },
        returns_scalar=True,
    )


def transform(
    value: IntoExpr,
    *,
    order: AutoregressionOrder,
    max_order: int | None = None,
    method: AutoregressionMethod = "yule_walker",
    output: AutoregressionOutput = "residual",
    demean: bool = True,
    null_policy: AutoregressionNullPolicy = "raise",
) -> pl.Expr:
    """Fit a series and apply the fitted filter to the same rows.

    `output="residual"` gives what the filter left behind, which is the series whitened by
    its own model. `output="prediction"` gives what the filter expected, one step ahead.

    The fit and the rows it is applied to are the same rows, so the result is in-sample.
    That is the honest default for a diagnostic and the wrong tool for a forecast.

    The remaining arguments are those of :func:`fit`.

    Returns
    -------
    A `Float64` expression with one value per input row. The first `order` rows of a group
    have no complete history and are null, as is any row whose own value or whose lags were
    not there.
    """
    column = _one(value)
    return plugin_expr(
        "autoregression_transform",
        [column],
        {
            "names": output_names([column]),
            **_order(order, max_order),
            "method": method,
            "output": output,
            "demean": demean,
            "null_policy": null_policy,
        },
    )


def solve_toeplitz(
    first_column: IntoExpr,
    rhs: IntoExprColumns,
    *,
    row_index: IntoExpr,
    diagonal_shift: float = 0.0,
) -> pl.Expr:
    """Solve `matrix @ x = rhs` for a symmetric positive-definite Toeplitz matrix.

    A Toeplitz matrix is constant along each of its diagonals, so it is determined by its
    first column alone. Unlike :func:`polars_qr.solve_spd`, which reads a square wide
    frame, this reads one column of length `n`. That difference is the point of the
    operation: the matrix is never formed, and the solve costs `O(n²)` rather than `O(n³)`.

    Parameters
    ----------
    first_column
        The first column of the matrix, whose length is the size of both axes.
    rhs
        The columns holding the right-hand sides, one column each.
    row_index
        An integer column giving the order of the rows. It may not repeat a value.
    diagonal_shift
        A value added to every diagonal entry before solving. A small shift is what makes a
        matrix that is only just short of positive definite solvable.

    Returns
    -------
    An expression producing one struct per group, with the fields:

    `rhs`
        The right-hand side names.
    `solution`
        One solution per right-hand side, each a list over the rows.
    `size`
        The size of the matrix.
    `diagonal_shift`
        The shift that was applied.

    Notes
    -----
    A matrix that is not positive definite shows up as a non-positive prediction error
    partway through the recursion, and is reported as that rather than solved.

    Rows cannot be dropped without changing the size of the system, so this operation has
    no null policy: a null anywhere in the input is an error.
    """
    matrix_column = as_expressions(first_column)
    rhs_columns = as_expressions(rhs)
    index_column = as_expressions(row_index)
    return plugin_expr(
        "solve_toeplitz",
        [*matrix_column, *rhs_columns, *index_column],
        {
            "names": output_names([*matrix_column, *rhs_columns, *index_column]),
            "n_rhs": len(rhs_columns),
            "diagonal_shift": diagonal_shift,
        },
        returns_scalar=True,
    )
