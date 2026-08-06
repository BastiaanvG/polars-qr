import numpy as np
import polars as pl
import pytest

import polars_qr as pq

FEATURES = ["a", "b", "c"]


def test_covariance_matches_a_reference_estimate(frame):
    out = frame.select(pq.covariance(FEATURES).alias("cov")).unnest("cov")

    x = frame.select(FEATURES).to_numpy()
    assert out["features"][0].to_list() == FEATURES
    assert np.allclose(out["covariance"][0].to_list(), np.cov(x, rowvar=False, ddof=1))
    assert np.allclose(out["means"][0].to_list(), x.mean(axis=0))
    assert np.allclose(out["standard_deviations"][0].to_list(), x.std(axis=0, ddof=1))
    assert out["n_observations"][0] == frame.height
    assert out["size"][0] == len(FEATURES)


def test_the_degrees_of_freedom_change_the_divisor(frame):
    out = frame.select(pq.covariance(FEATURES, ddof=0).alias("cov")).unnest("cov")

    x = frame.select(FEATURES).to_numpy()
    assert np.allclose(out["covariance"][0].to_list(), np.cov(x, rowvar=False, ddof=0))


def test_the_matrix_is_symmetric(frame):
    out = frame.select(pq.covariance(FEATURES).alias("cov")).unnest("cov")

    values = np.array(out["covariance"][0].to_list())
    assert np.array_equal(values, values.T)


def test_a_single_feature_gives_its_variance(frame):
    out = frame.select(pq.covariance("a").alias("cov")).unnest("cov")

    assert out["size"][0] == 1
    assert np.allclose(out["covariance"][0].to_list()[0][0], frame["a"].var(ddof=1))


def test_each_group_is_estimated_on_its_own_rows(frame):
    out = (
        frame.lazy()
        .group_by("group")
        .agg(pq.covariance(FEATURES).alias("cov"))
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

    out = with_null.select(pq.covariance(FEATURES, null_policy="drop").alias("cov")).unnest("cov")

    kept = frame.filter(pl.arange(0, frame.height) != 7).select(FEATURES).to_numpy()
    assert out["n_observations"][0] == frame.height - 1
    assert np.allclose(out["covariance"][0].to_list(), np.cov(kept, rowvar=False, ddof=1))


def test_too_few_observations_for_the_degrees_of_freedom_is_rejected(frame):
    single = frame.head(1)

    with pytest.raises(pl.exceptions.ComputeError, match="nothing to divide by"):
        single.select(pq.covariance(FEATURES))


def test_correlation_matches_a_reference_estimate(frame):
    out = frame.select(pq.correlation(FEATURES).alias("corr")).unnest("corr")

    x = frame.select(FEATURES).to_numpy()
    assert out["features"][0].to_list() == FEATURES
    assert np.allclose(out["correlation"][0].to_list(), np.corrcoef(x, rowvar=False))
    assert out["size"][0] == len(FEATURES)


def test_correlation_has_an_exact_unit_diagonal(frame):
    out = frame.select(pq.correlation(FEATURES).alias("corr")).unnest("corr")

    values = np.array(out["correlation"][0].to_list())
    assert np.array_equal(np.diag(values), np.ones(len(FEATURES)))


def test_correlation_is_unchanged_by_rescaling_a_column(frame):
    plain = frame.select(pq.correlation(FEATURES).alias("corr")).unnest("corr")
    rescaled = (
        frame.with_columns((pl.col("a") * 1000.0 + 5.0).alias("a"))
        .select(pq.correlation(FEATURES).alias("corr"))
        .unnest("corr")
    )

    assert np.allclose(plain["correlation"][0].to_list(), rescaled["correlation"][0].to_list())


def test_correlation_reports_the_same_moments_as_covariance(frame):
    cov = frame.select(pq.covariance(FEATURES).alias("cov")).unnest("cov")
    corr = frame.select(pq.correlation(FEATURES).alias("corr")).unnest("corr")

    assert np.allclose(cov["means"][0].to_list(), corr["means"][0].to_list())
    assert np.allclose(
        cov["standard_deviations"][0].to_list(), corr["standard_deviations"][0].to_list()
    )


def test_a_column_that_does_not_vary_correlates_with_nothing(frame):
    flat = frame.with_columns(pl.lit(3.0).alias("flat"))

    out = flat.select(pq.correlation([*FEATURES, "flat"]).alias("corr")).unnest("corr")

    values = np.array(out["correlation"][0].to_list())
    assert np.isnan(values[-1]).all()
    assert not np.isnan(values[:-1, :-1]).any()


def test_weighted_covariance_matches_a_reference_estimate(frame):
    out = frame.select(pq.covariance(FEATURES, weights="weight").alias("cov")).unnest("cov")

    x = frame.select(FEATURES).to_numpy()
    w = frame["weight"].to_numpy()
    assert np.allclose(out["covariance"][0].to_list(), np.cov(x, rowvar=False, aweights=w, ddof=1))
    assert np.allclose(out["means"][0].to_list(), np.average(x, axis=0, weights=w))
    assert np.isclose(out["sum_weights"][0], w.sum())


def test_weighted_correlation_matches_a_reference_estimate(frame):
    out = frame.select(pq.correlation(FEATURES, weights="weight").alias("corr")).unnest("corr")

    x = frame.select(FEATURES).to_numpy()
    w = frame["weight"].to_numpy()
    cov = np.cov(x, rowvar=False, aweights=w, ddof=1)
    scale = np.sqrt(np.diag(cov))
    assert np.allclose(out["correlation"][0].to_list(), cov / np.outer(scale, scale))


def test_the_sum_of_weights_is_the_row_count_without_weights(frame):
    out = frame.select(pq.covariance(FEATURES).alias("cov")).unnest("cov")

    assert out["sum_weights"][0] == frame.height


def test_scaling_every_weight_leaves_the_estimate_alone(frame):
    one = frame.select(pq.covariance(FEATURES, weights="weight").alias("cov")).unnest("cov")
    other = (
        frame.with_columns((pl.col("weight") * 1000.0).alias("weight"))
        .select(pq.covariance(FEATURES, weights="weight").alias("cov"))
        .unnest("cov")
    )

    assert np.allclose(one["covariance"][0].to_list(), other["covariance"][0].to_list())


def test_a_negative_weight_is_rejected_by_covariance(frame):
    negative = frame.with_columns(
        pl.when(pl.arange(0, frame.height) == 1).then(-1.0).otherwise(1.0).alias("w")
    )

    with pytest.raises(pl.exceptions.ComputeError, match="negative weight"):
        negative.select(pq.covariance(FEATURES, weights="w"))
