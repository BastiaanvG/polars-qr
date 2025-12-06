import numpy as np
import polars as pl
import pytest

import polars_faer as pf

FEATURES = ["a", "b", "c"]


def test_covariance_matches_a_reference_estimate(frame):
    out = frame.select(pf.covariance(FEATURES).alias("cov")).unnest("cov")

    x = frame.select(FEATURES).to_numpy()
    assert out["features"][0].to_list() == FEATURES
    assert np.allclose(out["covariance"][0].to_list(), np.cov(x, rowvar=False, ddof=1))
    assert np.allclose(out["means"][0].to_list(), x.mean(axis=0))
    assert np.allclose(out["standard_deviations"][0].to_list(), x.std(axis=0, ddof=1))
    assert out["n_observations"][0] == frame.height
    assert out["size"][0] == len(FEATURES)


def test_the_degrees_of_freedom_change_the_divisor(frame):
    out = frame.select(pf.covariance(FEATURES, ddof=0).alias("cov")).unnest("cov")

    x = frame.select(FEATURES).to_numpy()
    assert np.allclose(out["covariance"][0].to_list(), np.cov(x, rowvar=False, ddof=0))


def test_the_matrix_is_symmetric(frame):
    out = frame.select(pf.covariance(FEATURES).alias("cov")).unnest("cov")

    values = np.array(out["covariance"][0].to_list())
    assert np.array_equal(values, values.T)


def test_a_single_feature_gives_its_variance(frame):
    out = frame.select(pf.covariance("a").alias("cov")).unnest("cov")

    assert out["size"][0] == 1
    assert np.allclose(out["covariance"][0].to_list()[0][0], frame["a"].var(ddof=1))


def test_each_group_is_estimated_on_its_own_rows(frame):
    out = (
        frame.lazy()
        .group_by("group")
        .agg(pf.covariance(FEATURES).alias("cov"))
        .unnest("cov")
        .collect()
    )

    for row in out.iter_rows(named=True):
        group = frame.filter(pl.col("group") == row["group"]).select(FEATURES).to_numpy()
        assert np.allclose(row["covariance"], np.cov(group, rowvar=False, ddof=1))


def test_dropping_nulls_estimates_on_the_remaining_rows(frame):
    with_null = frame.with_columns(
        pl.when(pl.arange(0, frame.height) == 7).then(None).otherwise(pl.col("b")).alias("b")
    )

    out = with_null.select(pf.covariance(FEATURES, null_policy="drop").alias("cov")).unnest("cov")

    kept = frame.filter(pl.arange(0, frame.height) != 7).select(FEATURES).to_numpy()
    assert out["n_observations"][0] == frame.height - 1
    assert np.allclose(out["covariance"][0].to_list(), np.cov(kept, rowvar=False, ddof=1))


def test_too_few_observations_for_the_degrees_of_freedom_is_rejected(frame):
    single = frame.head(1)

    with pytest.raises(pl.exceptions.ComputeError, match="nothing to divide by"):
        single.select(pf.covariance(FEATURES))
