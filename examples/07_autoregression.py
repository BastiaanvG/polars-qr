"""Fitting a series against its own lags, choosing the order, and whitening it."""

import numpy as np
import numpy.typing as npt
import polars as pl

import polars_qr as pq

rng = np.random.default_rng(11)
n = 20_000
truth = np.array([0.6, -0.3])


def simulate(
    coefficients: npt.NDArray[np.float64], rows: int, burn: int = 500
) -> npt.NDArray[np.float64]:
    """A path from an autoregression with the coefficients given."""
    order = len(coefficients)
    series = np.zeros(rows + burn)
    noise = rng.normal(size=rows + burn)
    for row in range(order, rows + burn):
        series[row] = coefficients @ series[row - order : row][::-1] + noise[row]
    return series[burn:]


frame = pl.DataFrame(
    {
        "symbol": ["AAA"] * n + ["BBB"] * n,
        "y": np.concatenate([simulate(truth, n), simulate(np.array([0.85]), n)]),
    }
)

# One fit per symbol. BIC chooses the order; the fit reports what it chose and why.
fits = (
    frame.lazy()
    .group_by("symbol", maintain_order=True)
    .agg(pq.autoregression.fit("y", order="bic", max_order=12).alias("ar"))
    .unnest("ar")
    .collect()
)

for row in fits.iter_rows(named=True):
    print(f"{row['symbol']}: order {row['order']} chosen from 0-12 by bic")
    print(f"  coefficients: {np.round(row['coefficients'], 4)}")
    print(f"  variance:     {row['variance']:.4f}")
    print(f"  stationary:   {row['stationary']}   method: {row['method']}")

print(f"\nAAA was built from {truth}, so the fit should be close to it.")

# The criterion at every order comes back with the fit, so the choice can be inspected
# without running the query again.
criterion = fits.filter(pl.col("symbol") == "AAA")["criterion"][0].to_list()
best = int(np.argmin(criterion))
print("\nbic by order for AAA (relative to the best):")
for order, value in enumerate(criterion[:6]):
    chosen = "  <- chosen" if order == best else ""
    print(f"  order {order}: {value - criterion[best]:+10.2f}{chosen}")

# The partial autocorrelations are the reflection coefficients the recursion already
# produced, so they cost nothing beyond the fit itself.
diagnostics = (
    frame.filter(pl.col("symbol") == "AAA")
    .select(pq.autoregression.partial_autocorrelation("y", max_lag=6).alias("pacf"))
    .unnest("pacf")
)
print("\npacf at lags 1-6:", np.round(diagnostics["partial_autocorrelation"][0].to_list(), 4))
print("an AR(2) should have nothing beyond lag 2.")

# Whitening: apply the fitted filter to the rows it was fitted on.
whitened = (
    frame.lazy()
    .with_columns(
        residual=pq.autoregression.transform("y", order="bic", max_order=12).over("symbol")
    )
    .collect()
)


def autocorrelation(values: npt.NDArray[np.float64], max_lag: int) -> list[float]:
    """The correlation of a series with its own lags, for reading the result."""
    centred = values - values.mean()
    return [
        float(np.corrcoef(centred[:-lag], centred[lag:])[0, 1]) for lag in range(1, max_lag + 1)
    ]


series = whitened.filter(pl.col("symbol") == "AAA")
residual = series["residual"].drop_nulls().to_numpy()
print("\nautocorrelation at lags 1-4")
print("  before:", np.round(autocorrelation(series["y"].to_numpy(), 4), 4))
print("  after: ", np.round(autocorrelation(residual, 4), 4))
print("nulls in the residual:", series["residual"].null_count(), "one per lag with no history")

# The same fit reads more naturally from the column it is about.
through_namespace = (
    frame.filter(pl.col("symbol") == "AAA")
    .select(pl.col("y").ar.fit(order="bic", max_order=12).alias("ar"))  # type: ignore[attr-defined]
    .unnest("ar")
)
print(
    "\nnamespace agrees with the function:",
    through_namespace["coefficients"][0].to_list()
    == fits.filter(pl.col("symbol") == "AAA")["coefficients"][0].to_list(),
)

# The solve underneath, on its own: a symmetric Toeplitz system given by its first column.
sequence = (
    frame.filter(pl.col("symbol") == "AAA")
    .select(pq.autoregression.autocovariance("y", max_lag=5).alias("g"))
    .unnest("g")
)
system = pl.DataFrame(
    {
        "lag": np.arange(5),
        "sequence": sequence["autocovariance"][0].to_list()[:5],
        "rhs": sequence["autocovariance"][0].to_list()[1:6],
    }
)
solved = system.select(
    pq.autoregression.solve_toeplitz("sequence", "rhs", row_index="lag").alias("solved")
).unnest("solved")
print("\nsolving the Yule-Walker system directly:")
print("  size:    ", solved["size"][0])
print("  solution:", np.round(solved["solution"][0][0].to_list(), 4))
print("  which is the order-5 fit, arrived at through the same recursion.")
