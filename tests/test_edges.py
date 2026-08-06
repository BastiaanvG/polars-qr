import numpy as np
import polars as pl
import pytest

import polars_qr as pq

FEATURES = ["a", "b", "c"]

OPERATIONS = [
    pytest.param(lambda: pq.least_squares("y", FEATURES), id="least_squares"),
    pytest.param(lambda: pq.covariance(FEATURES), id="covariance"),
    pytest.param(lambda: pq.correlation(FEATURES), id="correlation"),
    pytest.param(lambda: pq.pca(FEATURES, n_components=2), id="pca"),
    pytest.param(lambda: pq.pca_transform(FEATURES, n_components=2), id="pca_transform"),
]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_an_empty_frame_is_rejected_rather_than_answered(frame, operation):
    with pytest.raises(pl.exceptions.ComputeError, match="no rows"):
        frame.head(0).select(operation())


DROPPING_OPERATIONS = [
    pytest.param(lambda: pq.least_squares("y", FEATURES, null_policy="drop"), id="least_squares"),
    pytest.param(lambda: pq.covariance(FEATURES, null_policy="drop"), id="covariance"),
    pytest.param(lambda: pq.correlation(FEATURES, null_policy="drop"), id="correlation"),
    pytest.param(lambda: pq.pca(FEATURES, n_components=2, null_policy="drop"), id="pca"),
    pytest.param(
        lambda: pq.pca_transform(FEATURES, n_components=2, null_policy="drop"),
        id="pca_transform",
    ),
]


@pytest.mark.parametrize("operation", DROPPING_OPERATIONS)
def test_dropping_every_row_is_rejected(frame, operation):
    holed = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("a"))

    with pytest.raises(pl.exceptions.ComputeError, match="leaving nothing to compute from"):
        holed.select(operation())


def test_an_empty_covariance_does_not_come_back_as_nan(frame):
    # An empty sample used to make the divisor 0/0, which slipped past the check on it.
    with pytest.raises(pl.exceptions.ComputeError):
        frame.head(0).select(pq.covariance(FEATURES))


def test_a_filter_that_matches_nothing_is_rejected(frame):
    with pytest.raises(pl.exceptions.ComputeError, match="no rows"):
        frame.lazy().filter(pl.col("a") > 1e9).select(pq.covariance(FEATURES)).collect()


def test_a_group_too_small_for_the_features_says_so(frame):
    small = frame.head(2)

    with pytest.raises(pl.exceptions.ComputeError, match="at least as many observations"):
        small.select(pq.least_squares("y", FEATURES))


def test_a_single_row_still_supports_a_population_covariance(frame):
    out = frame.head(1).select(pq.covariance(FEATURES, ddof=0).alias("cov")).unnest("cov")

    assert out["n_observations"][0] == 1
    assert np.allclose(out["covariance"][0].to_list(), np.zeros((3, 3)))


def test_a_grouped_result_is_still_labelled(frame):
    # Polars hands a plugin its inputs without names inside an aggregation, so the names
    # have to come from the expressions instead.
    out = (
        frame.lazy()
        .group_by("group")
        .agg(pq.least_squares("y", FEATURES).alias("fit"))
        .unnest("fit")
        .collect()
    )

    assert out["features"][0].to_list() == FEATURES
    assert out["targets"][0].to_list() == ["y"]


def test_a_grouped_covariance_is_still_labelled(frame):
    out = (
        frame.lazy()
        .group_by("group")
        .agg(pq.covariance(FEATURES).alias("cov"))
        .unnest("cov")
        .collect()
    )

    assert out["features"][0].to_list() == FEATURES


def test_a_renamed_column_is_reported_under_its_new_name(frame):
    out = frame.select(
        pq.least_squares(pl.col("y"), [pl.col("a").alias("renamed"), "b", "c"]).alias("fit")
    ).unnest("fit")

    assert out["features"][0].to_list() == ["renamed", "b", "c"]


def test_a_derived_column_is_named_after_what_it_was_derived_from(frame):
    out = frame.select(
        pq.least_squares(pl.col("y"), [pl.col("a") + pl.col("b"), pl.col("c") * 2]).alias("fit")
    ).unnest("fit")

    assert out["features"][0].to_list() == ["a", "c"]
