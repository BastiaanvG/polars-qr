"""Fitting within groups, inside a lazy query."""

import numpy as np
import polars as pl

import polars_faer as pf

rng = np.random.default_rng(1)
n = 900
dates = np.repeat(np.arange(30), n // 30)
features = rng.normal(size=(n, 2))
# Each date has its own relationship between the features and the target.
slope = np.repeat(rng.normal(loc=1.0, scale=0.4, size=30), n // 30)
frame = pl.DataFrame(
    {
        "date": dates,
        "signal": features[:, 0],
        "control": features[:, 1],
        "target": slope * features[:, 0] - 0.5 * features[:, 1] + rng.normal(scale=0.1, size=n),
    }
)

daily = (
    frame.lazy()
    .group_by("date")
    .agg(pf.least_squares("target", ["signal", "control"]).alias("fit"))
    .unnest("fit")
    .select(
        "date",
        pl.col("coefficients").list.first().list.first().alias("signal_beta"),
        pl.col("n_observations"),
        pl.col("residual_sum_of_squares").list.first().alias("rss"),
    )
    .sort("date")
    .collect()
)
print(daily.head())
print("\nfitted betas against the ones the data was built with:")
print("  correlation:", round(np.corrcoef(daily["signal_beta"].to_numpy(), slope[::30])[0, 1], 4))

# Scores are row-preserving, so they land next to the rows they came from.
scored = (
    frame.lazy()
    .faer.pca_transform(["signal", "control"], n_components=1, by="date")  # type: ignore[attr-defined]
    .select("date", "signal", "component_1")
    .collect()
)
print("\nscored rows:", scored.height)
print(scored.head())
