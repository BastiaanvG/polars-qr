"""Decomposing a block of correlated columns."""

import numpy as np
import polars as pl

import polars_qr as pq

rng = np.random.default_rng(2)
n = 400
# Three underlying drivers, observed through eight noisy columns.
drivers = rng.normal(size=(n, 3))
loadings = rng.normal(size=(3, 8))
observed = drivers @ loadings + rng.normal(scale=0.05, size=(n, 8))

columns = [f"x{index}" for index in range(8)]
frame = pl.DataFrame({name: observed[:, index] for index, name in enumerate(columns)})

found = frame.select(pq.pca(columns).alias("pca")).unnest("pca")
ratios = np.array(found["explained_variance_ratio"][0].to_list())
print("explained variance ratio:", np.round(ratios, 4))
print("carried by the first three components:", round(ratios[:3].sum(), 4))
print("rank:", found["rank"][0], "of", len(columns))

# Standardising first puts the columns on one footing before decomposing.
standardised = frame.select(pq.pca(columns, scale=True, n_components=3).alias("pca")).unnest("pca")
print("\nscales:", np.round(standardised["scales"][0].to_list(), 3))
print("first component:", np.round(standardised["components"][0].to_list()[0], 3))

# Polars registers the namespace at runtime, so a type checker cannot see it.
scores = frame.qr.pca_transform(columns, n_components=3)  # type: ignore[attr-defined]
print("\nscore columns:", scores.columns[-3:])
print("score standard deviations:", np.round(scores[:, -3:].to_numpy().std(axis=0, ddof=1), 3))
print(
    "which are the square roots of the explained variance:",
    np.round(np.sqrt(found["explained_variance"][0].to_list()[:3]), 3),
)
