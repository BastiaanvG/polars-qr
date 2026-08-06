from pathlib import Path
from typing import Any

import polars as pl
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parent


def plugin_expr(
    function_name: str,
    args: list[pl.Expr],
    kwargs: dict[str, Any] | None = None,
    *,
    returns_scalar: bool = False,
    is_elementwise: bool = False,
) -> pl.Expr:
    """Call a plugin function on `args`.

    Parameters
    ----------
    function_name
        Name of the exported Rust function.
    args
        Expressions passed as the positional inputs of the plugin function.
    kwargs
        Keyword arguments serialised for the plugin function.
    returns_scalar
        Whether the function aggregates its inputs into a single row.
    is_elementwise
        Whether the function may be applied to arbitrary slices of its inputs.
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name=function_name,
        args=args,
        kwargs=kwargs,
        returns_scalar=returns_scalar,
        is_elementwise=is_elementwise,
    )
