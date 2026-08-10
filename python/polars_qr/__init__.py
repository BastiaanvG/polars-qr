"""Dense numerical operations for Polars, backed by faer."""

from importlib.metadata import PackageNotFoundError, version

from polars_qr import autoregression, timeseries
from polars_qr.covariance import correlation, covariance
from polars_qr.least_squares import least_squares
from polars_qr.namespaces import ArExprNamespace, QrExprNamespace, QrFrame, QrLazyFrame
from polars_qr.pca import pca, pca_transform
from polars_qr.spd import solve_spd
from polars_qr.states import (
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
    __version__ = version("polars-qr")
except PackageNotFoundError:  # pragma: no cover - only hit in a source tree
    __version__ = "0.0.0"

__all__ = [
    "ArExprNamespace",
    "QrExprNamespace",
    "QrFrame",
    "QrLazyFrame",
    "__version__",
    "autoregression",
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
    "timeseries",
]
