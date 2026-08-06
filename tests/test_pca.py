import numpy as np
import polars as pl
import pytest

import polars_qr as pq

FEATURES = ["a", "b", "c"]


def reference(frame, features=FEATURES, *, centre=True, scale=False):
    x = frame.select(features).to_numpy()
    means = x.mean(axis=0) if centre else np.zeros(x.shape[1])
    scales = x.std(axis=0, ddof=1) if scale else np.ones(x.shape[1])
    return np.linalg.svd((x - means) / scales, full_matrices=False)


def test_components_match_a_reference_decomposition(frame):
    out = frame.select(pq.pca(FEATURES).alias("pca")).unnest("pca")

    _, singular_values, right = reference(frame)
    assert out["features"][0].to_list() == FEATURES
    assert np.allclose(out["singular_values"][0].to_list(), singular_values)
    assert np.allclose(np.abs(out["components"][0].to_list()), np.abs(right))
    assert out["rank"][0] == len(FEATURES)
    assert out["n_observations"][0] == frame.height


def test_the_components_are_orthonormal(frame):
    out = frame.select(pq.pca(FEATURES).alias("pca")).unnest("pca")

    components = np.array(out["components"][0].to_list())
    assert np.allclose(components @ components.T, np.eye(len(FEATURES)))


def test_centring_is_reported_and_can_be_turned_off(frame):
    centred = frame.select(pq.pca(FEATURES).alias("pca")).unnest("pca")
    raw = frame.select(pq.pca(FEATURES, centre=False).alias("pca")).unnest("pca")

    x = frame.select(FEATURES).to_numpy()
    assert np.allclose(centred["means"][0].to_list(), x.mean(axis=0))
    assert raw["means"][0].to_list() == [0.0] * len(FEATURES)
    assert np.allclose(raw["singular_values"][0].to_list(), np.linalg.svd(x, compute_uv=False))


def test_scaling_matches_a_standardised_decomposition(frame):
    out = frame.select(pq.pca(FEATURES, scale=True).alias("pca")).unnest("pca")

    _, singular_values, _ = reference(frame, scale=True)
    x = frame.select(FEATURES).to_numpy()
    assert np.allclose(out["scales"][0].to_list(), x.std(axis=0, ddof=1))
    assert np.allclose(out["singular_values"][0].to_list(), singular_values)


def test_only_the_requested_components_come_back(frame):
    out = frame.select(pq.pca(FEATURES, n_components=2).alias("pca")).unnest("pca")

    assert len(out["components"][0].to_list()) == 2
    assert len(out["singular_values"][0].to_list()) == 2
    assert out["rank"][0] == len(FEATURES)


def test_each_group_is_decomposed_on_its_own_rows(frame):
    out = (
        frame.lazy()
        .group_by("group")
        .agg(pq.pca(FEATURES, n_components=2).alias("pca"))
        .unnest("pca")
        .collect()
    )

    for row in out.iter_rows(named=True):
        group = frame.filter(pl.col("group") == row["group"])
        _, singular_values, _ = reference(group)
        assert np.allclose(row["singular_values"], singular_values[:2])


def test_more_components_than_the_data_supports_is_rejected(frame):
    with pytest.raises(pl.exceptions.ComputeError, match="support at most"):
        frame.select(pq.pca(FEATURES, n_components=4))


def test_the_component_signs_do_not_depend_on_the_data_sign(frame):
    plain = frame.select(pq.pca(FEATURES).alias("pca")).unnest("pca")
    negated = (
        frame.with_columns([(-pl.col(name)).alias(name) for name in FEATURES])
        .select(pq.pca(FEATURES).alias("pca"))
        .unnest("pca")
    )

    assert np.allclose(plain["components"][0].to_list(), negated["components"][0].to_list())


def test_every_component_leads_with_a_positive_entry(frame):
    out = frame.select(pq.pca(FEATURES).alias("pca")).unnest("pca")

    components = np.array(out["components"][0].to_list())
    leading = np.take_along_axis(components, np.abs(components).argmax(axis=1)[:, None], axis=1)
    assert (leading > 0).all()


def test_the_explained_variance_matches_the_column_variance(frame):
    out = frame.select(pq.pca(FEATURES).alias("pca")).unnest("pca")

    x = frame.select(FEATURES).to_numpy()
    variance = np.array(out["explained_variance"][0].to_list())
    assert np.isclose(variance.sum(), x.var(axis=0, ddof=1).sum())
    assert np.allclose(out["explained_variance_ratio"][0].to_list(), variance / variance.sum())
    assert np.isclose(sum(out["explained_variance_ratio"][0].to_list()), 1.0)


def test_keeping_fewer_components_does_not_inflate_the_ratios(frame):
    everything = frame.select(pq.pca(FEATURES).alias("pca")).unnest("pca")
    first = frame.select(pq.pca(FEATURES, n_components=1).alias("pca")).unnest("pca")

    assert np.isclose(
        everything["explained_variance_ratio"][0].to_list()[0],
        first["explained_variance_ratio"][0].to_list()[0],
    )
    assert sum(first["explained_variance_ratio"][0].to_list()) < 1.0


def test_the_variance_decreases_from_one_component_to_the_next(frame):
    out = frame.select(pq.pca(FEATURES).alias("pca")).unnest("pca")

    variance = out["explained_variance"][0].to_list()
    assert variance == sorted(variance, reverse=True)


def test_scores_project_the_rows_onto_the_components(frame):
    out = frame.select(pq.pca_transform(FEATURES, n_components=2).alias("scores")).unnest("scores")

    fit = frame.select(pq.pca(FEATURES, n_components=2).alias("pca")).unnest("pca")
    x = frame.select(FEATURES).to_numpy()
    components = np.array(fit["components"][0].to_list())
    expected = (x - x.mean(axis=0)) @ components.T
    assert out.columns == ["component_1", "component_2"]
    assert np.allclose(out.to_numpy(), expected)


def test_scores_keep_one_row_per_input_row(frame):
    out = frame.select(pq.pca_transform(FEATURES, n_components=1).alias("scores"))

    assert out.height == frame.height


def test_a_dropped_row_scores_null(frame):
    with_null = frame.with_columns(
        pl.when(pl.arange(0, frame.height) == 4).then(None).otherwise(pl.col("c")).alias("c")
    )

    out = with_null.select(
        pq.pca_transform(FEATURES, n_components=2, null_policy="drop").alias("scores")
    ).unnest("scores")

    assert out.height == frame.height
    assert out["component_1"][4] is None
    assert out["component_1"].null_count() == 1


def test_scores_can_be_taken_within_a_group(frame):
    out = frame.with_columns(
        pq.pca_transform(FEATURES, n_components=1).over("group").alias("scores")
    ).unnest("scores")

    for group in frame["group"].unique():
        rows = frame.filter(pl.col("group") == group)
        x = rows.select(FEATURES).to_numpy()
        fit = rows.select(pq.pca(FEATURES, n_components=1).alias("pca")).unnest("pca")
        components = np.array(fit["components"][0].to_list())
        expected = (x - x.mean(axis=0)) @ components.T
        got = out.filter(pl.col("group") == group)["component_1"].to_numpy()
        assert np.allclose(got, expected.ravel())


def test_the_frame_namespace_adds_the_score_columns(frame):
    out = frame.qr.pca_transform(FEATURES, n_components=2)

    expected = frame.select(pq.pca_transform(FEATURES, n_components=2).alias("scores")).unnest(
        "scores"
    )
    assert out.columns == [*frame.columns, "component_1", "component_2"]
    assert np.allclose(out.select(["component_1", "component_2"]).to_numpy(), expected.to_numpy())


def test_the_frame_namespace_can_group(frame):
    out = frame.qr.pca_transform(FEATURES, n_components=1, by="group")

    expected = frame.with_columns(
        pq.pca_transform(FEATURES, n_components=1).over("group").alias("scores")
    ).unnest("scores")
    assert np.allclose(out["component_1"].to_numpy(), expected["component_1"].to_numpy())


def test_the_lazy_namespace_matches_the_eager_one(frame):
    lazy = frame.lazy().qr.pca_transform(FEATURES, n_components=2, by="group").collect()
    eager = frame.qr.pca_transform(FEATURES, n_components=2, by="group")

    assert lazy.equals(eager)
