import numpy as np
import polars as pl
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20251026)


@pytest.fixture
def frame(rng):
    n = 240
    features = rng.normal(size=(n, 3))
    noise = rng.normal(scale=0.1, size=n)
    target = features @ np.array([1.5, -0.75, 0.25]) + noise
    return pl.DataFrame(
        {
            "a": features[:, 0],
            "b": features[:, 1],
            "c": features[:, 2],
            "y": target,
            "z": target * 0.5 - features[:, 1],
            "group": rng.integers(0, 3, size=n),
            "weight": rng.uniform(0.1, 2.0, size=n),
        }
    )
