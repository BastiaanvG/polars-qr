"""Dense numerical operations for Polars, backed by faer."""

from importlib.metadata import PackageNotFoundError, version

from polars_faer.least_squares import least_squares

try:
    __version__ = version("polars-faer")
except PackageNotFoundError:  # pragma: no cover - only hit in a source tree
    __version__ = "0.0.0"

__all__ = ["__version__", "least_squares"]
