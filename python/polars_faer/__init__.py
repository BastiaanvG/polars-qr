"""Dense numerical operations for Polars, backed by faer."""

from importlib.metadata import PackageNotFoundError, version

from polars_faer.covariance import correlation, covariance
from polars_faer.least_squares import least_squares
from polars_faer.namespace import FaerFrame, FaerLazyFrame
from polars_faer.pca import pca, pca_transform
from polars_faer.spd import solve_spd

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
    "least_squares",
    "pca",
    "pca_transform",
    "solve_spd",
]
