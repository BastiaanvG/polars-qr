"""The `.qr` namespaces.

On a frame, row-preserving operations read better as methods than as expressions: the
grouping column, the window and the unnesting are all part of one call instead of three.

On an expression, the timeseries statistics read better with the column they measure in
front of them. Those methods are wrappers and nothing else — every one of them hands
straight over to the function of the same name in `polars_qr.timeseries`, which is where
the arguments, the defaults and the numbers are decided.
"""

import polars as pl

from polars_qr import timeseries
from polars_qr._typing import (
    ClockSpan,
    IntoExpr,
    IntoExprColumns,
    NullPolicy,
    TimeseriesNullPolicy,
)
from polars_qr.pca import pca_transform

__all__ = ["QrExprNamespace", "QrFrame", "QrLazyFrame"]


def _scores(
    frame: pl.LazyFrame,
    features: IntoExprColumns,
    n_components: int,
    by: IntoExprColumns | None,
    *,
    centre: bool,
    scale: bool,
    null_policy: NullPolicy,
) -> pl.LazyFrame:
    scores = pca_transform(
        features,
        n_components=n_components,
        centre=centre,
        scale=scale,
        null_policy=null_policy,
    )
    if by is not None:
        scores = scores.over(by)
    return frame.with_columns(scores.alias("__faer_scores")).unnest("__faer_scores")


@pl.api.register_lazyframe_namespace("qr")
class QrLazyFrame:
    """Row-preserving dense operations on a lazy frame."""

    def __init__(self, frame: pl.LazyFrame) -> None:
        self._frame = frame

    def pca_transform(
        self,
        features: IntoExprColumns,
        *,
        n_components: int,
        by: IntoExprColumns | None = None,
        centre: bool = True,
        scale: bool = False,
        null_policy: NullPolicy = "raise",
    ) -> pl.LazyFrame:
        """Add one score column per component to the frame.

        Parameters
        ----------
        features
            The columns to decompose and project.
        n_components
            How many components to score on. Each one becomes a `component_i` column.
        by
            Columns to group by. Each group is decomposed on its own rows; without this the
            components come from the whole frame.
        centre
            Whether to subtract the column means before decomposing.
        scale
            Whether to divide the columns through by their standard deviations first.
        null_policy
            `"raise"` to fail on a row that is null or not finite, `"drop"` to leave it out
            of the decomposition. A dropped row keeps its place and scores null.
        """
        return _scores(
            self._frame,
            features,
            n_components,
            by,
            centre=centre,
            scale=scale,
            null_policy=null_policy,
        )


@pl.api.register_dataframe_namespace("qr")
class QrFrame:
    """Row-preserving dense operations on an eager frame."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def pca_transform(
        self,
        features: IntoExprColumns,
        *,
        n_components: int,
        by: IntoExprColumns | None = None,
        centre: bool = True,
        scale: bool = False,
        null_policy: NullPolicy = "raise",
    ) -> pl.DataFrame:
        """Add one score column per component to the frame.

        The arguments are those of :meth:`QrLazyFrame.pca_transform`.
        """
        return _scores(
            self._frame.lazy(),
            features,
            n_components,
            by,
            centre=centre,
            scale=scale,
            null_policy=null_policy,
        ).collect()


@pl.api.register_expr_namespace("qr")
class QrExprNamespace:
    """Timeseries statistics on the expression they measure."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def rolling_sum(
        self,
        *,
        clock: IntoExpr,
        window: ClockSpan,
        min_samples: int = 1,
        min_clock_span: ClockSpan | None = None,
        null_policy: TimeseriesNullPolicy = "skip",
    ) -> pl.Expr:
        """See :func:`polars_qr.timeseries.rolling_sum`."""
        return timeseries.rolling_sum(
            self._expr,
            clock=clock,
            window=window,
            min_samples=min_samples,
            min_clock_span=min_clock_span,
            null_policy=null_policy,
        )

    def rolling_mean(
        self,
        *,
        clock: IntoExpr,
        window: ClockSpan,
        min_samples: int = 1,
        min_clock_span: ClockSpan | None = None,
        null_policy: TimeseriesNullPolicy = "skip",
    ) -> pl.Expr:
        """See :func:`polars_qr.timeseries.rolling_mean`."""
        return timeseries.rolling_mean(
            self._expr,
            clock=clock,
            window=window,
            min_samples=min_samples,
            min_clock_span=min_clock_span,
            null_policy=null_policy,
        )

    def rolling_variance(
        self,
        *,
        clock: IntoExpr,
        window: ClockSpan,
        min_samples: int = 1,
        min_clock_span: ClockSpan | None = None,
        ddof: int = 1,
        null_policy: TimeseriesNullPolicy = "skip",
    ) -> pl.Expr:
        """See :func:`polars_qr.timeseries.rolling_variance`."""
        return timeseries.rolling_variance(
            self._expr,
            clock=clock,
            window=window,
            min_samples=min_samples,
            min_clock_span=min_clock_span,
            ddof=ddof,
            null_policy=null_policy,
        )

    def rolling_covariance(
        self,
        other: IntoExpr,
        *,
        clock: IntoExpr,
        window: ClockSpan,
        min_samples: int = 1,
        min_clock_span: ClockSpan | None = None,
        ddof: int = 1,
        null_policy: TimeseriesNullPolicy = "skip",
    ) -> pl.Expr:
        """See :func:`polars_qr.timeseries.rolling_covariance`."""
        return timeseries.rolling_covariance(
            self._expr,
            other,
            clock=clock,
            window=window,
            min_samples=min_samples,
            min_clock_span=min_clock_span,
            ddof=ddof,
            null_policy=null_policy,
        )

    def rolling_correlation(
        self,
        other: IntoExpr,
        *,
        clock: IntoExpr,
        window: ClockSpan,
        min_samples: int = 1,
        min_clock_span: ClockSpan | None = None,
        null_policy: TimeseriesNullPolicy = "skip",
    ) -> pl.Expr:
        """See :func:`polars_qr.timeseries.rolling_correlation`."""
        return timeseries.rolling_correlation(
            self._expr,
            other,
            clock=clock,
            window=window,
            min_samples=min_samples,
            min_clock_span=min_clock_span,
            null_policy=null_policy,
        )

    def ewm_sum(
        self,
        *,
        clock: IntoExpr,
        half_life: ClockSpan,
        min_samples: int = 1,
        null_policy: TimeseriesNullPolicy = "skip",
    ) -> pl.Expr:
        """See :func:`polars_qr.timeseries.ewm_sum`."""
        return timeseries.ewm_sum(
            self._expr,
            clock=clock,
            half_life=half_life,
            min_samples=min_samples,
            null_policy=null_policy,
        )

    def ewm_mean(
        self,
        *,
        clock: IntoExpr,
        half_life: ClockSpan,
        min_samples: int = 1,
        null_policy: TimeseriesNullPolicy = "skip",
    ) -> pl.Expr:
        """See :func:`polars_qr.timeseries.ewm_mean`."""
        return timeseries.ewm_mean(
            self._expr,
            clock=clock,
            half_life=half_life,
            min_samples=min_samples,
            null_policy=null_policy,
        )

    def ewm_variance(
        self,
        *,
        clock: IntoExpr,
        half_life: ClockSpan,
        min_samples: int = 1,
        bias: bool = False,
        null_policy: TimeseriesNullPolicy = "skip",
    ) -> pl.Expr:
        """See :func:`polars_qr.timeseries.ewm_variance`."""
        return timeseries.ewm_variance(
            self._expr,
            clock=clock,
            half_life=half_life,
            min_samples=min_samples,
            bias=bias,
            null_policy=null_policy,
        )

    def ewm_covariance(
        self,
        other: IntoExpr,
        *,
        clock: IntoExpr,
        half_life: ClockSpan,
        min_samples: int = 1,
        bias: bool = False,
        null_policy: TimeseriesNullPolicy = "skip",
    ) -> pl.Expr:
        """See :func:`polars_qr.timeseries.ewm_covariance`."""
        return timeseries.ewm_covariance(
            self._expr,
            other,
            clock=clock,
            half_life=half_life,
            min_samples=min_samples,
            bias=bias,
            null_policy=null_policy,
        )

    def ewm_correlation(
        self,
        other: IntoExpr,
        *,
        clock: IntoExpr,
        half_life: ClockSpan,
        min_samples: int = 1,
        null_policy: TimeseriesNullPolicy = "skip",
    ) -> pl.Expr:
        """See :func:`polars_qr.timeseries.ewm_correlation`."""
        return timeseries.ewm_correlation(
            self._expr,
            other,
            clock=clock,
            half_life=half_life,
            min_samples=min_samples,
            null_policy=null_policy,
        )
