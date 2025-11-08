import numpy as np
import polars as pl
import pytest

import polars_faer as pf

FEATURES = ["a", "b", "c"]


def reference(frame, features=FEATURES, target="y"):
    x = frame.select(features).to_numpy()
    y = frame[target].to_numpy()
    return np.linalg.lstsq(x, y, rcond=None)


def test_coefficients_match_a_reference_solve(frame):
    out = frame.select(pf.least_squares("y", FEATURES).alias("fit")).unnest("fit")

    coefficients, residuals, _, _ = reference(frame)
    assert out["features"][0].to_list() == FEATURES
    assert out["targets"][0].to_list() == ["y"]
    assert np.allclose(out["coefficients"][0].to_list()[0], coefficients)
    assert out["n_observations"][0] == frame.height
    assert np.allclose(out["residual_sum_of_squares"][0].to_list(), residuals)


def test_a_single_feature_is_accepted(frame):
    out = frame.select(pf.least_squares("y", "a").alias("fit")).unnest("fit")

    coefficients, _, _, _ = reference(frame, features=["a"])
    assert out["features"][0].to_list() == ["a"]
    assert np.allclose(out["coefficients"][0].to_list()[0], coefficients)


def test_expressions_may_be_passed_instead_of_names(frame):
    out = frame.select(
        pf.least_squares(pl.col("y"), [pl.col("a"), pl.col("b"), pl.col("c")]).alias("fit")
    ).unnest("fit")

    coefficients, _, _, _ = reference(frame)
    assert np.allclose(out["coefficients"][0].to_list()[0], coefficients)


def test_each_group_is_fitted_on_its_own_rows(frame):
    out = (
        frame.lazy()
        .group_by("group")
        .agg(pf.least_squares("y", FEATURES).alias("fit"))
        .unnest("fit")
        .collect()
    )

    assert out.height == frame["group"].n_unique()
    for row in out.iter_rows(named=True):
        group = frame.filter(pl.col("group") == row["group"])
        coefficients, _, _, _ = reference(group)
        assert np.allclose(row["coefficients"][0], coefficients)
        assert row["n_observations"] == group.height


def test_a_lazy_frame_gives_the_same_answer(frame):
    eager = frame.select(pf.least_squares("y", FEATURES).alias("fit")).unnest("fit")
    lazy = frame.lazy().select(pf.least_squares("y", FEATURES).alias("fit")).unnest("fit").collect()

    assert eager.equals(lazy)


def test_nulls_are_rejected_by_default(frame):
    with_null = frame.with_columns(
        pl.when(pl.arange(0, frame.height) == 3).then(None).otherwise(pl.col("a")).alias("a")
    )

    with pytest.raises(pl.exceptions.ComputeError, match="null or non-finite"):
        with_null.select(pf.least_squares("y", FEATURES))


def test_dropping_nulls_fits_the_remaining_rows(frame):
    with_null = frame.with_columns(
        pl.when(pl.arange(0, frame.height) == 3).then(None).otherwise(pl.col("a")).alias("a")
    )

    out = with_null.select(pf.least_squares("y", FEATURES, null_policy="drop").alias("fit")).unnest(
        "fit"
    )

    kept = frame.filter(pl.arange(0, frame.height) != 3)
    coefficients, _, _, _ = reference(kept)
    assert out["n_observations"][0] == frame.height - 1
    assert np.allclose(out["coefficients"][0].to_list()[0], coefficients)


def test_a_non_numeric_column_is_rejected(frame):
    labelled = frame.with_columns(pl.lit("x").alias("label"))

    # Polars reports every failure raised inside a plugin as a compute error.
    with pytest.raises(pl.exceptions.ComputeError, match="not numeric"):
        labelled.select(pf.least_squares("y", ["a", "label"]))


def test_several_targets_share_one_feature_matrix(frame):
    out = frame.select(pf.least_squares(["y", "z"], FEATURES).alias("fit")).unnest("fit")

    x = frame.select(FEATURES).to_numpy()
    expected = np.linalg.lstsq(x, frame.select(["y", "z"]).to_numpy(), rcond=None)[0]
    assert out["targets"][0].to_list() == ["y", "z"]
    assert np.allclose(out["coefficients"][0].to_list(), expected.T)


def test_fitting_targets_together_matches_fitting_them_apart(frame):
    together = frame.select(pf.least_squares(["y", "z"], FEATURES).alias("fit")).unnest("fit")
    apart = [
        frame.select(pf.least_squares(target, FEATURES).alias("fit")).unnest("fit")
        for target in ("y", "z")
    ]

    for index, single in enumerate(apart):
        assert np.allclose(
            together["coefficients"][0].to_list()[index],
            single["coefficients"][0].to_list()[0],
        )
        assert np.allclose(
            together["residual_sum_of_squares"][0].to_list()[index],
            single["residual_sum_of_squares"][0].to_list()[0],
        )
