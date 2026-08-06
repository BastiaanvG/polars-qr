# Components and second moments

## Covariance and correlation

```python
risk = frame.select(pq.covariance(feature_columns, weights="recency").alias("cov")).unnest("cov")
```

The result carries the matrix as a list of its rows, the column means, the standard
deviations, the observation count and the sum of the weights.

Weights are read as reliability weights: they say how precisely an observation was measured
rather than how many times it was seen. Scaling all of them by the same factor therefore
leaves the estimate alone, and the divisor is corrected the way `numpy.cov(aweights=...)`
corrects it.

`correlation` is the same estimate divided through by the standard deviations. Its diagonal
is exactly one. A column that does not vary has no scale to divide by, so its row and column
come back as NaN rather than zero — undefined is not the same as uncorrelated.

The matrix is made exactly symmetric before it is returned, so it can be handed straight to
[`solve_spd`](spd.md) without a factorisation failing on a difference in the last bit.

## Principal components

```python
components = frame.select(pq.pca(feature_columns, n_components=10).alias("pca")).unnest("pca")
```

`pca` reports the loadings, the singular values, the explained variance and its ratio, along
with the means and scales it applied. `centre=True` subtracts the column means;
`scale=True` also divides by the standard deviations, which is the same as decomposing the
correlation instead of the covariance.

The share each component carries is taken against every direction the data spans, not only
the retained ones, so asking for fewer components does not inflate their shares.

!!! note "Signs are pinned down"
    A component and its negation describe the same direction, and which one a decomposition
    returns can differ between machines or library versions. Each component here is flipped
    so that its largest loading is positive, which is what makes two runs over the same data
    comparable.

## Scores

`pca_transform` keeps its rows, so the scores land next to the data they came from.

```python
scored = frame.lazy().qr.pca_transform(feature_columns, n_components=10, by="date").collect()
```

Each component becomes a `component_i` column. `by=` decomposes each group on its own rows;
without it the components come from the whole frame. A row dropped by the null policy keeps
its place and scores null.

The frame method is a convenience: it does `with_columns` of the expression, `over` the
grouping, and `unnest` of the result. The expression `pq.pca_transform(...)` is the same
thing with those three steps written out.
