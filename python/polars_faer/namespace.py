"""The `.faer` namespace on a frame.

Row-preserving operations read better as frame methods than as expressions: the grouping
column, the window and the unnesting are all part of one call instead of three.
"""

import polars as pl

from polars_faer._typing import IntoExprColumns, NullPolicy
from polars_faer.pca import pca_transform

__all__ = ["FaerFrame", "FaerLazyFrame"]


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


@pl.api.register_lazyframe_namespace("faer")
class FaerLazyFrame:
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


@pl.api.register_dataframe_namespace("faer")
class FaerFrame:
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

        The arguments are those of :meth:`FaerLazyFrame.pca_transform`.
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
