import polars as pl

import polars_faer as pf
from polars_faer._plugin import plugin_expr


def test_version_is_exposed():
    assert pf.__version__


def test_plugin_reports_the_same_version_as_the_package():
    out = pl.select(plugin_expr("plugin_version", [pl.lit(0)]).alias("v"))
    assert out["v"][0] == pf.__version__
