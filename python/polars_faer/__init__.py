"""Dense numerical operations for Polars, backed by faer."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("polars-faer")
except PackageNotFoundError:  # pragma: no cover - only hit in a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
