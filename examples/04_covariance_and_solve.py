"""Estimating a covariance matrix and solving against it."""

import numpy as np
import polars as pl

import polars_faer as pf

rng = np.random.default_rng(3)
n = 600
columns = ["alpha", "beta", "gamma"]
factor = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.3, 0.2, 0.93]])
observed = rng.normal(size=(n, 3)) @ factor.T

frame = pl.DataFrame({name: observed[:, index] for index, name in enumerate(columns)}).with_columns(
    pl.Series("recency", np.linspace(0.2, 2.0, n))
)

estimate = frame.select(pf.covariance(columns).alias("cov")).unnest("cov")
values = np.array(estimate["covariance"][0].to_list())
print("features:", estimate["features"][0].to_list())
print("covariance:\n", np.round(values, 3))

# Weighting recent observations more heavily gives a different estimate of the same thing.
weighted = frame.select(pf.covariance(columns, weights="recency").alias("cov")).unnest("cov")
print("\nweighted covariance:\n", np.round(np.array(weighted["covariance"][0].to_list()), 3))
print("sum of weights:", round(weighted["sum_weights"][0], 2))

correlation = frame.select(pf.correlation(columns).alias("corr")).unnest("corr")
print("\ncorrelation:\n", np.round(np.array(correlation["correlation"][0].to_list()), 3))

# The covariance matrix is laid out as a wide frame, then solved against.
wide = pl.DataFrame(
    {
        "row": np.arange(len(columns)),
        **{name: values[:, index] for index, name in enumerate(columns)},
        "view": [1.0, 0.0, -0.5],
    }
)
solved = wide.select(
    pf.solve_spd(columns, "view", row_index="row", diagonal_shift=1e-10).alias("solved")
).unnest("solved")
solution = np.array(solved["solution"][0].to_list()[0])
print("\nsolution:", np.round(solution, 4))
print("symmetry error:", f"{solved['symmetry_error'][0]:.2e}")
print("reproduces the right-hand side:", np.round(values @ solution, 4))
