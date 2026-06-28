import numpy as np
import polars as pl
import pytest

import polars_faer as pf

FEATURES = ["a", "b", "c"]


def partitioned(frame, rng, n_parts=5):
    """Label every row with a partition, unevenly."""
    return frame.with_columns(
        pl.Series("part", rng.integers(0, n_parts, size=frame.height)),
    )


def fit_through_states(frame, **kwargs):
    finalise = {key: kwargs.pop(key) for key in ("solver", "l2_penalty") if key in kwargs}
    return (
        frame.lazy()
        .group_by("part")
        .agg(pf.least_squares_state("y", FEATURES, **kwargs).alias("state"))
        .select(pf.merge_least_squares_states("state").alias("state"))
        .select(pf.finalise_least_squares("state", **finalise).alias("fit"))
        .unnest("fit")
        .collect()
    )


def test_a_fit_through_states_matches_a_fit_in_one_pass(frame, rng):
    direct = frame.select(pf.least_squares("y", FEATURES).alias("fit")).unnest("fit")

    through = fit_through_states(partitioned(frame, rng))

    assert np.allclose(through["coefficients"][0].to_list(), direct["coefficients"][0].to_list())
    assert np.isclose(
        through["residual_sum_of_squares"][0].to_list()[0],
        direct["residual_sum_of_squares"][0].to_list()[0],
    )
    assert through["n_observations"][0] == frame.height
    assert through["features"][0].to_list() == FEATURES
    assert through["rank"][0] == direct["rank"][0]


def test_the_partitioning_does_not_change_the_answer(frame, rng):
    one = fit_through_states(partitioned(frame, rng, n_parts=2))
    another = fit_through_states(partitioned(frame, rng, n_parts=17))

    assert np.allclose(one["coefficients"][0].to_list(), another["coefficients"][0].to_list())
    assert np.isclose(
        one["residual_sum_of_squares"][0].to_list()[0],
        another["residual_sum_of_squares"][0].to_list()[0],
    )


def test_a_state_carries_an_intercept_weights_and_several_targets(frame, rng):
    parted = partitioned(frame, rng)
    direct = frame.select(
        pf.least_squares(["y", "z"], FEATURES, weights="weight", intercept=True).alias("fit")
    ).unnest("fit")

    through = (
        parted.lazy()
        .group_by("part")
        .agg(
            pf.least_squares_state(["y", "z"], FEATURES, weights="weight", intercept=True).alias(
                "state"
            )
        )
        .select(pf.merge_least_squares_states("state").alias("state"))
        .select(pf.finalise_least_squares("state").alias("fit"))
        .unnest("fit")
        .collect()
    )

    assert np.allclose(through["coefficients"][0].to_list(), direct["coefficients"][0].to_list())
    assert np.allclose(through["intercept"][0].to_list(), direct["intercept"][0].to_list())
    assert through["targets"][0].to_list() == ["y", "z"]


def test_the_penalty_is_applied_once_at_the_end(frame, rng):
    direct = frame.select(pf.least_squares("y", FEATURES, l2_penalty=25.0).alias("fit")).unnest(
        "fit"
    )

    through = fit_through_states(partitioned(frame, rng), l2_penalty=25.0)

    assert np.allclose(through["coefficients"][0].to_list(), direct["coefficients"][0].to_list())


def test_a_state_can_be_finalised_with_either_solver(frame, rng):
    parted = partitioned(frame, rng)

    by_qr = fit_through_states(parted)
    by_svd = fit_through_states(parted, solver="svd")

    assert np.allclose(by_qr["coefficients"][0].to_list(), by_svd["coefficients"][0].to_list())
    assert by_svd["solver"][0] == "svd"
    assert by_svd["singular_values"][0] is not None


def test_a_state_is_smaller_than_the_rows_it_summarises(frame, rng):
    states = (
        partitioned(frame, rng, n_parts=2)
        .lazy()
        .group_by("part")
        .agg(pf.least_squares_state("y", FEATURES).alias("state"))
        .collect()
    )

    # Four columns and one target: a five by five factor, plus its header and names.
    assert states["state"].dtype == pl.Binary
    assert all(len(value) < 400 for value in states["state"].to_list())


def test_states_can_be_merged_a_second_time(frame, rng):
    parted = partitioned(frame, rng, n_parts=6)

    once = (
        parted.lazy()
        .group_by("part")
        .agg(pf.least_squares_state("y", FEATURES).alias("state"))
        .select(pf.merge_least_squares_states("state").alias("state"))
        .collect()
    )
    twice = (
        once.select(pf.merge_least_squares_states("state").alias("state"))
        .select(pf.finalise_least_squares("state").alias("fit"))
        .unnest("fit")
    )
    direct = frame.select(pf.least_squares("y", FEATURES).alias("fit")).unnest("fit")

    assert np.allclose(twice["coefficients"][0].to_list(), direct["coefficients"][0].to_list())


def test_states_built_for_different_columns_refuse_to_merge(frame):
    states = pl.concat(
        [
            frame.select(pf.least_squares_state("y", FEATURES).alias("state")),
            frame.select(pf.least_squares_state("y", ["a", "b"]).alias("state")),
        ],
        how="vertical",
    )

    with pytest.raises(pl.exceptions.ComputeError, match="different columns"):
        states.select(pf.merge_least_squares_states("state"))


def test_a_blob_that_is_not_a_state_is_rejected():
    frame = pl.DataFrame({"state": [b"certainly not a state"]})

    with pytest.raises(pl.exceptions.ComputeError, match="not a polars-faer state"):
        frame.select(pf.finalise_least_squares("state"))


def covariance_through_states(frame, finaliser=None, **kwargs):
    finaliser = finaliser or pf.finalise_covariance
    return (
        frame.lazy()
        .group_by("part")
        .agg(pf.covariance_state(FEATURES, **kwargs).alias("state"))
        .select(pf.merge_covariance_states("state").alias("state"))
        .select(finaliser("state").alias("cov"))
        .unnest("cov")
        .collect()
    )


def test_a_covariance_through_states_matches_one_in_one_pass(frame, rng):
    direct = frame.select(pf.covariance(FEATURES).alias("cov")).unnest("cov")

    through = covariance_through_states(partitioned(frame, rng))

    assert np.allclose(through["covariance"][0].to_list(), direct["covariance"][0].to_list())
    assert np.allclose(through["means"][0].to_list(), direct["means"][0].to_list())
    assert through["n_observations"][0] == frame.height
    assert through["features"][0].to_list() == FEATURES


def test_the_covariance_partitioning_does_not_change_the_answer(frame, rng):
    one = covariance_through_states(partitioned(frame, rng, n_parts=2))
    another = covariance_through_states(partitioned(frame, rng, n_parts=23))

    assert np.allclose(one["covariance"][0].to_list(), another["covariance"][0].to_list())
    assert np.allclose(one["means"][0].to_list(), another["means"][0].to_list())


def test_a_covariance_state_carries_weights(frame, rng):
    direct = frame.select(pf.covariance(FEATURES, weights="weight").alias("cov")).unnest("cov")

    through = covariance_through_states(partitioned(frame, rng), weights="weight")

    assert np.allclose(through["covariance"][0].to_list(), direct["covariance"][0].to_list())
    assert np.isclose(through["sum_weights"][0], direct["sum_weights"][0])


def test_a_covariance_state_can_be_finalised_as_a_correlation(frame, rng):
    direct = frame.select(pf.correlation(FEATURES).alias("corr")).unnest("corr")

    through = covariance_through_states(partitioned(frame, rng), finaliser=pf.finalise_correlation)

    assert np.allclose(through["correlation"][0].to_list(), direct["correlation"][0].to_list())


def test_the_degrees_of_freedom_are_chosen_when_the_state_is_finalised(frame, rng):
    parted = partitioned(frame, rng)
    direct = frame.select(pf.covariance(FEATURES, ddof=0).alias("cov")).unnest("cov")

    through = (
        parted.lazy()
        .group_by("part")
        .agg(pf.covariance_state(FEATURES).alias("state"))
        .select(pf.merge_covariance_states("state").alias("state"))
        .select(pf.finalise_covariance("state", ddof=0).alias("cov"))
        .unnest("cov")
        .collect()
    )

    assert np.allclose(through["covariance"][0].to_list(), direct["covariance"][0].to_list())


def test_a_least_squares_state_is_not_a_covariance_state(frame):
    states = frame.select(pf.least_squares_state("y", FEATURES).alias("state"))

    with pytest.raises(pl.exceptions.ComputeError, match="least-squares state, but a covariance"):
        states.select(pf.finalise_covariance("state"))


def test_components_from_a_state_match_ones_from_the_rows(frame, rng):
    direct = frame.select(pf.pca(FEATURES).alias("pca")).unnest("pca")

    through = (
        partitioned(frame, rng)
        .lazy()
        .group_by("part")
        .agg(pf.covariance_state(FEATURES).alias("state"))
        .select(pf.merge_covariance_states("state").alias("state"))
        .select(pf.finalise_pca("state").alias("pca"))
        .unnest("pca")
        .collect()
    )

    assert np.allclose(
        through["components"][0].to_list(), direct["components"][0].to_list(), atol=1e-8
    )
    assert np.allclose(
        through["explained_variance"][0].to_list(),
        direct["explained_variance"][0].to_list(),
    )
    assert np.allclose(
        through["singular_values"][0].to_list(), direct["singular_values"][0].to_list()
    )
    assert through["rank"][0] == direct["rank"][0]
    assert through["n_observations"][0] == frame.height
    # Merged means agree to within the last bit or two, not exactly: summing a partition at
    # a time rounds differently from summing every row in one pass.
    assert np.allclose(through["means"][0].to_list(), direct["means"][0].to_list())


def test_a_standardised_decomposition_from_a_state(frame, rng):
    direct = frame.select(pf.pca(FEATURES, scale=True).alias("pca")).unnest("pca")

    through = (
        partitioned(frame, rng)
        .lazy()
        .group_by("part")
        .agg(pf.covariance_state(FEATURES).alias("state"))
        .select(pf.merge_covariance_states("state").alias("state"))
        .select(pf.finalise_pca("state", scale=True).alias("pca"))
        .unnest("pca")
        .collect()
    )

    assert np.allclose(through["scales"][0].to_list(), direct["scales"][0].to_list())
    assert np.allclose(
        through["explained_variance_ratio"][0].to_list(),
        direct["explained_variance_ratio"][0].to_list(),
    )


def test_only_the_requested_components_come_back_from_a_state(frame, rng):
    through = (
        partitioned(frame, rng)
        .lazy()
        .group_by("part")
        .agg(pf.covariance_state(FEATURES).alias("state"))
        .select(pf.merge_covariance_states("state").alias("state"))
        .select(pf.finalise_pca("state", n_components=2).alias("pca"))
        .unnest("pca")
        .collect()
    )

    assert len(through["components"][0].to_list()) == 2
    assert sum(through["explained_variance_ratio"][0].to_list()) < 1.0
