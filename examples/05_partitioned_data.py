"""Fitting and estimating over partitions, through mergeable states."""

import numpy as np
import polars as pl

import polars_faer as pf

rng = np.random.default_rng(4)
n = 2000
columns = ["signal", "control"]
features = rng.normal(size=(n, 2))
frame = pl.DataFrame(
    {
        "partition": rng.integers(0, 8, size=n),
        "signal": features[:, 0],
        "control": features[:, 1],
    }
).with_columns(
    pl.Series("target", features @ np.array([1.5, -0.5]) + rng.normal(scale=0.2, size=n))
)

# Each partition summarises its own rows. A state does not grow with the rows it saw.
states = (
    frame.lazy()
    .group_by("partition")
    .agg(pf.least_squares_state("target", columns).alias("state"))
    .collect()
)
print("partitions:", states.height)
print("state size in bytes:", [len(value) for value in states["state"].to_list()][:3], "...")
print("rows behind each:", frame.group_by("partition").len()["len"].to_list())

# Merging them and finalising gives what a fit over every row gives.
through = (
    states.lazy()
    .select(pf.merge_least_squares_states("state").alias("state"))
    .select(pf.finalise_least_squares("state").alias("fit"))
    .unnest("fit")
    .collect()
)
direct = frame.select(pf.least_squares("target", columns).alias("fit")).unnest("fit")
print("\nthrough states:", np.round(through["coefficients"][0].to_list()[0], 6))
print("in one pass:   ", np.round(direct["coefficients"][0].to_list()[0], 6))
print("observations:  ", through["n_observations"][0])

# The same summaries can be finalised more than one way.
ridge = (
    states.lazy()
    .select(pf.merge_least_squares_states("state").alias("state"))
    .select(pf.finalise_least_squares("state", l2_penalty=500.0).alias("fit"))
    .unnest("fit")
    .collect()
)
print("under a penalty:", np.round(ridge["coefficients"][0].to_list()[0], 6))

# Second moments have a state of their own, which finalises three ways.
moments = (
    frame.lazy()
    .group_by("partition")
    .agg(pf.covariance_state(columns).alias("state"))
    .select(pf.merge_covariance_states("state").alias("state"))
)
covariance = moments.select(pf.finalise_covariance("state").alias("cov")).unnest("cov").collect()
correlation = (
    moments.select(pf.finalise_correlation("state").alias("corr")).unnest("corr").collect()
)
components = moments.select(pf.finalise_pca("state").alias("pca")).unnest("pca").collect()

print("\ncovariance:\n", np.round(np.array(covariance["covariance"][0].to_list()), 4))
print("correlation:\n", np.round(np.array(correlation["correlation"][0].to_list()), 4))
print(
    "explained variance ratio:",
    np.round(components["explained_variance_ratio"][0].to_list(), 4),
)
