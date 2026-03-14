"""Fitting one and several targets against a shared feature matrix."""

import numpy as np
import polars as pl

import polars_faer as pf

rng = np.random.default_rng(0)
n = 500
features = rng.normal(size=(n, 3))
frame = pl.DataFrame(
    {
        "size": features[:, 0],
        "value": features[:, 1],
        "momentum": features[:, 2],
        "weight": rng.uniform(0.2, 2.0, size=n),
    }
).with_columns(
    pl.Series("target", features @ np.array([1.5, -0.75, 0.25]) + rng.normal(scale=0.1, size=n)),
    pl.Series("other", features @ np.array([0.0, 0.5, 0.5]) + rng.normal(scale=0.1, size=n)),
)

columns = ["size", "value", "momentum"]

fit = frame.select(pf.least_squares("target", columns, intercept=True).alias("fit")).unnest("fit")
print("features:  ", fit["features"][0].to_list())
print("coefficients:", np.round(fit["coefficients"][0].to_list()[0], 3))
print("intercept: ", round(fit["intercept"][0].to_list()[0], 3))
print("rank:      ", fit["rank"][0], "of", len(columns) + 1)
print("condition: ", round(fit["condition"][0], 2))

# Several targets share one factorisation, so they are fitted on one pass over the data.
both = frame.select(pf.least_squares(["target", "other"], columns).alias("fit")).unnest("fit")
print("\ntargets:   ", both["targets"][0].to_list())
print("coefficients:")
for name, row in zip(both["targets"][0].to_list(), both["coefficients"][0].to_list(), strict=True):
    print(f"  {name:8} {np.round(row, 3)}")

# Weighting the observations, and shrinking the coefficients towards zero.
weighted = frame.select(
    pf.least_squares("target", columns, weights="weight", l2_penalty=10.0).alias("fit")
).unnest("fit")
print("\nweighted and penalised:", np.round(weighted["coefficients"][0].to_list()[0], 3))
