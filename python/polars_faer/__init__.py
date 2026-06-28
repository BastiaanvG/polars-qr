"""Dense numerical operations for Polars, backed by faer."""

from importlib.metadata import PackageNotFoundError, version

from polars_faer.covariance import correlation, covariance
from polars_faer.least_squares import least_squares
from polars_faer.namespace import FaerFrame, FaerLazyFrame
from polars_faer.pca import pca, pca_transform
from polars_faer.spd import solve_spd
from polars_faer.states import (
    covariance_state,
    finalise_correlation,
    finalise_covariance,
    finalise_least_squares,
    finalise_pca,
    least_squares_state,
    merge_covariance_states,
    merge_least_squares_states,
)

try:
    __version__ = version("polars-faer")
except PackageNotFoundError:  # pragma: no cover - only hit in a source tree
    __version__ = "0.0.0"

__all__ = [
    "FaerFrame",
    "FaerLazyFrame",
    "__version__",
    "correlation",
    "covariance",
    "covariance_state",
    "finalise_correlation",
    "finalise_covariance",
    "finalise_least_squares",
    "finalise_pca",
    "least_squares",
    "least_squares_state",
    "merge_covariance_states",
    "merge_least_squares_states",
    "pca",
    "pca_transform",
    "solve_spd",
]
