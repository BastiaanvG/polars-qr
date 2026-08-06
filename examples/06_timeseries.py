"""Rolling and decaying statistics measured against a clock that is not wall time."""

import numpy as np
import polars as pl

import polars_faer as pf

rng = np.random.default_rng(5)
n = 4000

# Trades arriving unevenly: quantity is what the clock will count, not the row number.
quantity = rng.integers(1, 500, size=n).astype(np.float64)
side = rng.choice([-1.0, 1.0], size=n)
asset = rng.normal(scale=0.001, size=n)
market = 0.7 * asset + rng.normal(scale=0.001, size=n)

trades = pl.DataFrame(
    {
        "symbol": rng.choice(["AAA", "BBB"], size=n),
        "quantity": quantity,
        "signed_quantity": side * quantity,
        "asset_return": asset,
        "market_return": market,
    }
)

# The clock is cumulative volume within a symbol, built with ordinary Polars expressions.
trades = trades.with_columns(
    cumulative_volume=pl.col("quantity").cum_sum().over("symbol"),
)

result = (
    trades.lazy()
    .sort(["symbol", "cumulative_volume"])
    .with_columns(
        decayed_flow=pf.timeseries.ewm_sum(
            "signed_quantity",
            clock="cumulative_volume",
            half_life=50_000,
        ).over("symbol"),
        rolling_corr=pf.timeseries.rolling_correlation(
            "asset_return",
            "market_return",
            clock="cumulative_volume",
            window=100_000,
            min_samples=50,
        ).over("symbol"),
        decayed_corr=pf.timeseries.ewm_correlation(
            "asset_return",
            "market_return",
            clock="cumulative_volume",
            half_life=50_000,
            min_samples=30,
        ).over("symbol"),
    )
    .collect()
)

print(result.select("symbol", "cumulative_volume", "decayed_flow", "rolling_corr").tail(5))
print("\nrows in, rows out:", trades.height, result.height)

# The half-life is in units of volume, so it means the same thing however the trades arrive.
for symbol in ("AAA", "BBB"):
    rows = result.filter(pl.col("symbol") == symbol)
    volume = float(rows["cumulative_volume"].max())  # type: ignore[arg-type]
    print(
        f"\n{symbol}: {rows.height} trades over {volume:,.0f} units of volume,"
        f" so ~{volume / 50_000:.0f} half-lives"
    )
    print(f"  decayed flow, last:  {rows['decayed_flow'][-1]:,.1f}")
    print(f"  rolling correlation: {rows['rolling_corr'][-1]:.4f}")
    print(f"  decayed correlation: {rows['decayed_corr'][-1]:.4f}")

# The same statistic reads more naturally from the column it measures.
through_namespace = trades.sort(["symbol", "cumulative_volume"]).with_columns(
    decayed_flow=pl.col("signed_quantity")
    .faer.ewm_sum(clock="cumulative_volume", half_life=50_000)  # type: ignore[attr-defined]
    .over("symbol")
)
print(
    "\nnamespace agrees with the function:",
    np.allclose(
        through_namespace["decayed_flow"].to_numpy(),
        result.sort(["symbol", "cumulative_volume"])["decayed_flow"].to_numpy(),
    ),
)

# A clock does not have to count volume. Here it counts realised variance, so the statistic
# ages by how much the market has moved rather than by how many trades have printed.
variance_clocked = (
    trades.lazy()
    .sort(["symbol", "cumulative_volume"])
    .with_columns(variance_clock=pl.col("market_return").pow(2).cum_sum().over("symbol"))
    .with_columns(
        signal_state=pf.timeseries.ewm_mean(
            "asset_return",
            clock="variance_clock",
            half_life=1e-5,
        ).over("symbol")
    )
    .collect()
)
print("\nagainst a variance clock:")
print(variance_clocked.select("symbol", "variance_clock", "signal_state").tail(3))
