import numpy as np
import polars as pl
import pytest

import polars_faer as pf

FEATURES = ["a", "b", "c"]


def reference(frame, features=FEATURES, *, centre=True, scale=False):
    x = frame.select(features).to_numpy()
    means = x.mean(axis=0) if centre else np.zeros(x.shape[1])
    scales = x.std(axis=0, ddof=1) if scale else np.ones(x.shape[1])
    return np.linalg.svd((x - means) / scales, full_matrices=False)


def test_components_match_a_reference_decomposition(frame):
    out = frame.select(pf.pca(FEATURES).alias("pca")).unnest("pca")

    _, singular_values, right = reference(frame)
    assert out["features"][0].to_list() == FEATURES
    assert np.allclose(out["singular_values"][0].to_list(), singular_values)
    assert np.allclose(np.abs(out["components"][0].to_list()), np.abs(right))
    assert out["rank"][0] == len(FEATURES)
    assert out["n_observations"][0] == frame.height


def test_the_components_are_orthonormal(frame):
    out = frame.select(pf.pca(FEATURES).alias("pca")).unnest("pca")

    components = np.array(out["components"][0].to_list())
    assert np.allclose(components @ components.T, np.eye(len(FEATURES)))


def test_centring_is_reported_and_can_be_turned_off(frame):
    centred = frame.select(pf.pca(FEATURES).alias("pca")).unnest("pca")
    raw = frame.select(pf.pca(FEATURES, centre=False).alias("pca")).unnest("pca")

    x = frame.select(FEATURES).to_numpy()
    assert np.allclose(centred["means"][0].to_list(), x.mean(axis=0))
    assert raw["means"][0].to_list() == [0.0] * len(FEATURES)
    assert np.allclose(raw["singular_values"][0].to_list(), np.linalg.svd(x, compute_uv=False))


def test_scaling_matches_a_standardised_decomposition(frame):
    out = frame.select(pf.pca(FEATURES, scale=True).alias("pca")).unnest("pca")

    _, singular_values, _ = reference(frame, scale=True)
    x = frame.select(FEATURES).to_numpy()
    assert np.allclose(out["scales"][0].to_list(), x.std(axis=0, ddof=1))
    assert np.allclose(out["singular_values"][0].to_list(), singular_values)


def test_only_the_requested_components_come_back(frame):
    out = frame.select(pf.pca(FEATURES, n_components=2).alias("pca")).unnest("pca")

    assert len(out["components"][0].to_list()) == 2
    assert len(out["singular_values"][0].to_list()) == 2
    assert out["rank"][0] == len(FEATURES)


def test_each_group_is_decomposed_on_its_own_rows(frame):
    out = (
        frame.lazy()
        .group_by("group")
        .agg(pf.pca(FEATURES, n_components=2).alias("pca"))
        .unnest("pca")
        .collect()
    )

    for row in out.iter_rows(named=True):
        group = frame.filter(pl.col("group") == row["group"])
        _, singular_values, _ = reference(group)
        assert np.allclose(row["singular_values"], singular_values[:2])


def test_more_components_than_the_data_supports_is_rejected(frame):
    with pytest.raises(pl.exceptions.ComputeError, match="support at most"):
        frame.select(pf.pca(FEATURES, n_components=4))


def test_the_component_signs_do_not_depend_on_the_data_sign(frame):
    plain = frame.select(pf.pca(FEATURES).alias("pca")).unnest("pca")
    negated = (
        frame.with_columns([(-pl.col(name)).alias(name) for name in FEATURES])
        .select(pf.pca(FEATURES).alias("pca"))
        .unnest("pca")
    )

    assert np.allclose(plain["components"][0].to_list(), negated["components"][0].to_list())


def test_every_component_leads_with_a_positive_entry(frame):
    out = frame.select(pf.pca(FEATURES).alias("pca")).unnest("pca")

    components = np.array(out["components"][0].to_list())
    leading = np.take_along_axis(components, np.abs(components).argmax(axis=1)[:, None], axis=1)
    assert (leading > 0).all()
