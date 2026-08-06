"""Rolling and exponentially weighted statistics over a generalised clock.

A clock is any column that only moves forward. It may be wall time, but it may just as
well be cumulative volume, a trade count or cumulative squared return: whatever the caller
thinks an observation should age against. A window width or a half-life is then given in
the units of that clock, so `half_life=5_000_000` on a cumulative-volume clock means what
it says — five million units of volume, not five million rows and not a duration.

A clock says how far apart two observations are, which decides how much the older one has
decayed by the time the newer one arrives. It is not an observation weight: every valid
observation enters with weight one.
"""

from datetime import timedelta

import polars as pl

from polars_qr._plugin import plugin_expr
from polars_qr._typing import ClockSpan, IntoExpr, TimeseriesNullPolicy, as_expressions

# How many value columns a statistic of a pair reads.
PAIR = 2

__all__ = [
    "ewm_correlation",
    "ewm_covariance",
    "ewm_mean",
    "ewm_sum",
    "ewm_variance",
    "rolling_correlation",
    "rolling_covariance",
    "rolling_mean",
    "rolling_sum",
    "rolling_variance",
]


def _span(value: ClockSpan) -> dict[str, float | int]:
    """Describe a span in a way that says which kind of clock it belongs to.

    A number belongs to a numeric clock and a `timedelta` to a temporal one. Which clock it
    actually meets is only known once the query runs, so the pairing is checked there.
    """
    if isinstance(value, timedelta):
        return {"nanoseconds": value // timedelta(microseconds=1) * 1_000}
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"a clock span must be a number or a timedelta, not {type(value).__name__}"
        raise TypeError(message)
    return {"numeric": float(value)}


def _one(column: IntoExpr) -> pl.Expr:
    """Read a single column argument."""
    return as_expressions(column)[0]


def _ewm(
    operation: str,
    values: list[pl.Expr],
    clock: IntoExpr,
    half_life: ClockSpan,
    min_samples: int,
    *,
    bias: bool,
    null_policy: TimeseriesNullPolicy,
) -> pl.Expr:
    return plugin_expr(
        "ewm_bivariate" if len(values) == PAIR else "ewm_univariate",
        [*values, _one(clock)],
        {
            "operation": operation,
            "half_life": _span(half_life),
            "min_samples": min_samples,
            "bias": bias,
            "null_policy": null_policy,
        },
    )


def _rolling(
    operation: str,
    values: list[pl.Expr],
    clock: IntoExpr,
    window: ClockSpan,
    min_samples: int,
    min_clock_span: ClockSpan | None,
    ddof: int,
    *,
    null_policy: TimeseriesNullPolicy,
) -> pl.Expr:
    return plugin_expr(
        "rolling_bivariate" if len(values) == PAIR else "rolling_univariate",
        [*values, _one(clock)],
        {
            "operation": operation,
            "window": _span(window),
            "min_samples": min_samples,
            "min_clock_span": None if min_clock_span is None else _span(min_clock_span),
            "ddof": ddof,
            "null_policy": null_policy,
        },
    )


def rolling_sum(
    value: IntoExpr,
    *,
    clock: IntoExpr,
    window: ClockSpan,
    min_samples: int = 1,
    min_clock_span: ClockSpan | None = None,
    null_policy: TimeseriesNullPolicy = "skip",
) -> pl.Expr:
    """Sum `value` over a trailing window of the clock.

    Parameters
    ----------
    value
        The column to sum.
    clock
        A non-decreasing column the window is measured along.
    window
        The width of the window, in the units of `clock`. The window covers
        `(current clock - window, current clock]`: everything the clock has passed within
        that width, up to and including the current row, and nothing ahead of it.
    min_samples
        How many valid observations the window needs before a number is reported.
    min_clock_span
        How much clock the window must cover before a number is reported, measured from the
        oldest observation still in it. It may not exceed `window`.
    null_policy
        `"skip"` to let the window move past a null, `"raise"` to fail on it.

    Returns
    -------
    A `Float64` expression with one value per input row, null where the requirements are
    not met.
    """
    return _rolling(
        "rolling_sum",
        [_one(value)],
        clock,
        window,
        min_samples,
        min_clock_span,
        1,
        null_policy=null_policy,
    )


def rolling_mean(
    value: IntoExpr,
    *,
    clock: IntoExpr,
    window: ClockSpan,
    min_samples: int = 1,
    min_clock_span: ClockSpan | None = None,
    null_policy: TimeseriesNullPolicy = "skip",
) -> pl.Expr:
    """Average `value` over a trailing window of the clock.

    The arguments are those of :func:`rolling_sum`.
    """
    return _rolling(
        "rolling_mean",
        [_one(value)],
        clock,
        window,
        min_samples,
        min_clock_span,
        1,
        null_policy=null_policy,
    )


def rolling_variance(
    value: IntoExpr,
    *,
    clock: IntoExpr,
    window: ClockSpan,
    min_samples: int = 1,
    min_clock_span: ClockSpan | None = None,
    ddof: int = 1,
    null_policy: TimeseriesNullPolicy = "skip",
) -> pl.Expr:
    """Measure the variance of `value` over a trailing window of the clock.

    The arguments are those of :func:`rolling_sum`, plus:

    Parameters
    ----------
    ddof
        The delta degrees of freedom: the divisor is the number of observations in the
        window minus this. The default of 1 gives the usual sample variance.
    """
    return _rolling(
        "rolling_variance",
        [_one(value)],
        clock,
        window,
        min_samples,
        min_clock_span,
        ddof,
        null_policy=null_policy,
    )


def rolling_covariance(
    x: IntoExpr,
    y: IntoExpr,
    *,
    clock: IntoExpr,
    window: ClockSpan,
    min_samples: int = 1,
    min_clock_span: ClockSpan | None = None,
    ddof: int = 1,
    null_policy: TimeseriesNullPolicy = "skip",
) -> pl.Expr:
    """Measure the covariance of `x` and `y` over a trailing window of the clock.

    A row enters the window only when both values are there, so the covariance and the two
    variances behind it describe the same observations.

    The remaining arguments are those of :func:`rolling_variance`.
    """
    return _rolling(
        "rolling_covariance",
        [_one(x), _one(y)],
        clock,
        window,
        min_samples,
        min_clock_span,
        ddof,
        null_policy=null_policy,
    )


def rolling_correlation(
    x: IntoExpr,
    y: IntoExpr,
    *,
    clock: IntoExpr,
    window: ClockSpan,
    min_samples: int = 1,
    min_clock_span: ClockSpan | None = None,
    null_policy: TimeseriesNullPolicy = "skip",
) -> pl.Expr:
    """Measure the correlation of `x` and `y` over a trailing window of the clock.

    The result is null when either column does not move within the window, because a
    correlation with something that does not vary is undefined rather than zero.

    The remaining arguments are those of :func:`rolling_covariance`.
    """
    return _rolling(
        "rolling_correlation",
        [_one(x), _one(y)],
        clock,
        window,
        min_samples,
        min_clock_span,
        1,
        null_policy=null_policy,
    )


def ewm_sum(
    value: IntoExpr,
    *,
    clock: IntoExpr,
    half_life: ClockSpan,
    min_samples: int = 1,
    null_policy: TimeseriesNullPolicy = "skip",
) -> pl.Expr:
    """Sum `value` with a weight that decays along the clock.

    Each valid observation enters with weight one, and everything already held is halved
    for every `half_life` of clock that passes.

    Parameters
    ----------
    value
        The column to sum.
    clock
        A non-decreasing column the decay is measured along.
    half_life
        The clock distance over which what is already held loses half its weight, in the
        units of `clock`.
    min_samples
        How many valid observations must have been seen before a number is reported.
    null_policy
        `"skip"` to let the state decay past a null without adding to it, `"raise"` to fail
        on it.

    Returns
    -------
    A `Float64` expression with one value per input row.
    """
    return _ewm(
        "ewm_sum",
        [_one(value)],
        clock,
        half_life,
        min_samples,
        bias=False,
        null_policy=null_policy,
    )


def ewm_mean(
    value: IntoExpr,
    *,
    clock: IntoExpr,
    half_life: ClockSpan,
    min_samples: int = 1,
    null_policy: TimeseriesNullPolicy = "skip",
) -> pl.Expr:
    """Average `value` with a weight that decays along the clock.

    The mean is the decaying sum over the decaying total weight, so it is the sum reported
    by :func:`ewm_sum` divided by what that sum was weighted with.

    The arguments are those of :func:`ewm_sum`.
    """
    return _ewm(
        "ewm_mean",
        [_one(value)],
        clock,
        half_life,
        min_samples,
        bias=False,
        null_policy=null_policy,
    )


def ewm_variance(
    value: IntoExpr,
    *,
    clock: IntoExpr,
    half_life: ClockSpan,
    min_samples: int = 1,
    bias: bool = False,
    null_policy: TimeseriesNullPolicy = "skip",
) -> pl.Expr:
    """Measure the variance of `value` with a weight that decays along the clock.

    The arguments are those of :func:`ewm_sum`, plus:

    Parameters
    ----------
    bias
        Whether to divide by the total weight as it stands. The default corrects for the
        spread lost to estimating the mean from the same observations, which with equal
        weights is the usual step from `n` to `n - 1`.
    """
    return _ewm(
        "ewm_variance",
        [_one(value)],
        clock,
        half_life,
        min_samples,
        bias=bias,
        null_policy=null_policy,
    )


def ewm_covariance(
    x: IntoExpr,
    y: IntoExpr,
    *,
    clock: IntoExpr,
    half_life: ClockSpan,
    min_samples: int = 1,
    bias: bool = False,
    null_policy: TimeseriesNullPolicy = "skip",
) -> pl.Expr:
    """Measure the covariance of `x` and `y` with a weight that decays along the clock.

    A row enters the state only when both values are there.

    The remaining arguments are those of :func:`ewm_variance`.
    """
    return _ewm(
        "ewm_covariance",
        [_one(x), _one(y)],
        clock,
        half_life,
        min_samples,
        bias=bias,
        null_policy=null_policy,
    )


def ewm_correlation(
    x: IntoExpr,
    y: IntoExpr,
    *,
    clock: IntoExpr,
    half_life: ClockSpan,
    min_samples: int = 1,
    null_policy: TimeseriesNullPolicy = "skip",
) -> pl.Expr:
    """Measure the correlation of `x` and `y` with a weight that decays along the clock.

    There is no `bias` argument: the same correction applied to the covariance and to both
    variances cancels when one is divided by the others.

    The remaining arguments are those of :func:`ewm_covariance`.
    """
    return _ewm(
        "ewm_correlation",
        [_one(x), _one(y)],
        clock,
        half_life,
        min_samples,
        bias=False,
        null_policy=null_policy,
    )
