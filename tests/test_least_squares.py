import numpy as np
import polars as pl
import pytest

import polars_faer as pf

FEATURES = ["a", "b", "c"]


def reference(frame, features=FEATURES, target="y"):
    x = frame.select(features).to_numpy()
    y = frame[target].to_numpy()
    return np.linalg.lstsq(x, y, rcond=None)


def test_coefficients_match_a_reference_solve(frame):
    out = frame.select(pf.least_squares("y", FEATURES).alias("fit")).unnest("fit")

    coefficients, residuals, _, _ = reference(frame)
    assert out["features"][0].to_list() == FEATURES
    assert out["targets"][0].to_list() == ["y"]
    assert np.allclose(out["coefficients"][0].to_list()[0], coefficients)
    assert out["n_observations"][0] == frame.height
    assert np.allclose(out["residual_sum_of_squares"][0].to_list(), residuals)


def test_a_single_feature_is_accepted(frame):
    out = frame.select(pf.least_squares("y", "a").alias("fit")).unnest("fit")

    coefficients, _, _, _ = reference(frame, features=["a"])
    assert out["features"][0].to_list() == ["a"]
    assert np.allclose(out["coefficients"][0].to_list()[0], coefficients)


def test_expressions_may_be_passed_instead_of_names(frame):
    out = frame.select(
        pf.least_squares(pl.col("y"), [pl.col("a"), pl.col("b"), pl.col("c")]).alias("fit")
    ).unnest("fit")

    coefficients, _, _, _ = reference(frame)
    assert np.allclose(out["coefficients"][0].to_list()[0], coefficients)


def test_each_group_is_fitted_on_its_own_rows(frame):
    out = (
        frame.lazy()
        .group_by("group")
        .agg(pf.least_squares("y", FEATURES).alias("fit"))
        .unnest("fit")
        .collect()
    )

    assert out.height == frame["group"].n_unique()
    for row in out.iter_rows(named=True):
        group = frame.filter(pl.col("group") == row["group"])
        coefficients, _, _, _ = reference(group)
        assert np.allclose(row["coefficients"][0], coefficients)
        assert row["n_observations"] == group.height


def test_a_lazy_frame_gives_the_same_answer(frame):
    eager = frame.select(pf.least_squares("y", FEATURES).alias("fit")).unnest("fit")
    lazy = frame.lazy().select(pf.least_squares("y", FEATURES).alias("fit")).unnest("fit").collect()

    assert eager.equals(lazy)


def test_nulls_are_rejected_by_default(frame):
    with_null = frame.with_columns(
        pl.when(pl.arange(0, frame.height) == 3).then(None).otherwise(pl.col("a")).alias("a")
    )

    with pytest.raises(pl.exceptions.ComputeError, match="null or non-finite"):
        with_null.select(pf.least_squares("y", FEATURES))


def test_dropping_nulls_fits_the_remaining_rows(frame):
    with_null = frame.with_columns(
        pl.when(pl.arange(0, frame.height) == 3).then(None).otherwise(pl.col("a")).alias("a")
    )

    out = with_null.select(pf.least_squares("y", FEATURES, null_policy="drop").alias("fit")).unnest(
        "fit"
    )

    kept = frame.filter(pl.arange(0, frame.height) != 3)
    coefficients, _, _, _ = reference(kept)
    assert out["n_observations"][0] == frame.height - 1
    assert np.allclose(out["coefficients"][0].to_list()[0], coefficients)


def test_a_non_numeric_column_is_rejected(frame):
    labelled = frame.with_columns(pl.lit("x").alias("label"))

    # Polars reports every failure raised inside a plugin as a compute error.
    with pytest.raises(pl.exceptions.ComputeError, match="not numeric"):
        labelled.select(pf.least_squares("y", ["a", "label"]))


def test_several_targets_share_one_feature_matrix(frame):
    out = frame.select(pf.least_squares(["y", "z"], FEATURES).alias("fit")).unnest("fit")

    x = frame.select(FEATURES).to_numpy()
    expected = np.linalg.lstsq(x, frame.select(["y", "z"]).to_numpy(), rcond=None)[0]
    assert out["targets"][0].to_list() == ["y", "z"]
    assert np.allclose(out["coefficients"][0].to_list(), expected.T)


def test_fitting_targets_together_matches_fitting_them_apart(frame):
    together = frame.select(pf.least_squares(["y", "z"], FEATURES).alias("fit")).unnest("fit")
    apart = [
        frame.select(pf.least_squares(target, FEATURES).alias("fit")).unnest("fit")
        for target in ("y", "z")
    ]

    for index, single in enumerate(apart):
        assert np.allclose(
            together["coefficients"][0].to_list()[index],
            single["coefficients"][0].to_list()[0],
        )
        assert np.allclose(
            together["residual_sum_of_squares"][0].to_list()[index],
            single["residual_sum_of_squares"][0].to_list()[0],
        )


def weighted_reference(frame, features=FEATURES, target="y", weight="weight"):
    x = frame.select(features).to_numpy()
    y = frame[target].to_numpy()
    w = frame[weight].to_numpy()
    root = np.sqrt(w)
    return np.linalg.lstsq(x * root[:, None], y * root, rcond=None)


def test_weights_reproduce_a_scaled_solve(frame):
    out = frame.select(pf.least_squares("y", FEATURES, weights="weight").alias("fit")).unnest("fit")

    coefficients, residuals, _, _ = weighted_reference(frame)
    assert np.allclose(out["coefficients"][0].to_list()[0], coefficients)
    assert np.allclose(out["residual_sum_of_squares"][0].to_list(), residuals)


def test_equal_weights_leave_the_fit_unchanged(frame):
    unweighted = frame.select(pf.least_squares("y", FEATURES).alias("fit")).unnest("fit")
    weighted = (
        frame.with_columns(pl.lit(2.0).alias("w"))
        .select(pf.least_squares("y", FEATURES, weights="w").alias("fit"))
        .unnest("fit")
    )

    assert np.allclose(
        unweighted["coefficients"][0].to_list()[0],
        weighted["coefficients"][0].to_list()[0],
    )


def test_a_zero_weight_drops_an_observation_from_the_fit(frame):
    zeroed = frame.with_columns(
        pl.when(pl.arange(0, frame.height) < 5).then(0.0).otherwise(1.0).alias("w")
    )

    out = zeroed.select(pf.least_squares("y", FEATURES, weights="w").alias("fit")).unnest("fit")

    kept = frame.filter(pl.arange(0, frame.height) >= 5)
    coefficients, _, _, _ = reference(kept)
    assert np.allclose(out["coefficients"][0].to_list()[0], coefficients)
    assert out["n_observations"][0] == frame.height


def test_a_negative_weight_is_rejected(frame):
    negative = frame.with_columns(
        pl.when(pl.arange(0, frame.height) == 2).then(-1.0).otherwise(1.0).alias("w")
    )

    with pytest.raises(pl.exceptions.ComputeError, match="negative weight"):
        negative.select(pf.least_squares("y", FEATURES, weights="w"))


def test_weights_that_are_all_zero_are_rejected(frame):
    zeroed = frame.with_columns(pl.lit(0.0).alias("w"))

    with pytest.raises(pl.exceptions.ComputeError, match="no observation carries any weight"):
        zeroed.select(pf.least_squares("y", FEATURES, weights="w"))


def test_no_intercept_is_fitted_by_default(frame):
    out = frame.select(pf.least_squares("y", FEATURES).alias("fit")).unnest("fit")

    assert out["intercept"][0] is None


def test_an_intercept_matches_an_explicit_constant_column(frame):
    out = frame.select(pf.least_squares("y", FEATURES, intercept=True).alias("fit")).unnest("fit")

    x = np.column_stack([np.ones(frame.height), frame.select(FEATURES).to_numpy()])
    expected = np.linalg.lstsq(x, frame["y"].to_numpy(), rcond=None)[0]
    assert np.allclose(out["intercept"][0].to_list(), expected[0])
    assert np.allclose(out["coefficients"][0].to_list()[0], expected[1:])
    assert out["features"][0].to_list() == FEATURES


def test_an_intercept_absorbs_a_shifted_target(frame):
    shifted = frame.with_columns((pl.col("y") + 10.0).alias("shifted"))

    plain = shifted.select(pf.least_squares("y", FEATURES, intercept=True).alias("fit")).unnest(
        "fit"
    )
    moved = shifted.select(
        pf.least_squares("shifted", FEATURES, intercept=True).alias("fit")
    ).unnest("fit")

    assert np.allclose(plain["coefficients"][0].to_list(), moved["coefficients"][0].to_list())
    assert np.allclose(
        plain["intercept"][0].to_list()[0] + 10.0, moved["intercept"][0].to_list()[0]
    )


def test_a_weighted_intercept_matches_a_scaled_solve(frame):
    out = frame.select(
        pf.least_squares("y", FEATURES, weights="weight", intercept=True).alias("fit")
    ).unnest("fit")

    root = np.sqrt(frame["weight"].to_numpy())
    x = np.column_stack([np.ones(frame.height), frame.select(FEATURES).to_numpy()])
    expected = np.linalg.lstsq(x * root[:, None], frame["y"].to_numpy() * root, rcond=None)[0]
    assert np.allclose(out["intercept"][0].to_list(), expected[0])
    assert np.allclose(out["coefficients"][0].to_list()[0], expected[1:])


def test_the_two_solvers_agree_on_a_well_posed_system(frame):
    by_qr = frame.select(pf.least_squares("y", FEATURES).alias("fit")).unnest("fit")
    by_svd = frame.select(pf.least_squares("y", FEATURES, solver="svd").alias("fit")).unnest("fit")

    assert np.allclose(by_qr["coefficients"][0].to_list(), by_svd["coefficients"][0].to_list())
    assert np.allclose(
        by_qr["residual_sum_of_squares"][0].to_list(),
        by_svd["residual_sum_of_squares"][0].to_list(),
    )


def test_an_svd_solve_matches_the_reference_on_a_duplicated_feature(frame):
    duplicated = frame.with_columns(pl.col("a").alias("a_again"))
    columns = [*FEATURES, "a_again"]

    out = duplicated.select(pf.least_squares("y", columns, solver="svd").alias("fit")).unnest("fit")

    x = duplicated.select(columns).to_numpy()
    expected = np.linalg.lstsq(x, duplicated["y"].to_numpy(), rcond=None)[0]
    assert np.allclose(out["coefficients"][0].to_list()[0], expected)


def test_an_svd_solve_handles_more_features_than_rows(rng):
    narrow = pl.DataFrame({name: rng.normal(size=3) for name in ("y", "a", "b", "c", "d", "e")})

    out = narrow.select(
        pf.least_squares("y", ["a", "b", "c", "d", "e"], solver="svd").alias("fit")
    ).unnest("fit")

    x = narrow.select(["a", "b", "c", "d", "e"]).to_numpy()
    expected = np.linalg.lstsq(x, narrow["y"].to_numpy(), rcond=None)[0]
    assert np.allclose(out["coefficients"][0].to_list()[0], expected)
    assert out["residual_sum_of_squares"][0].to_list()[0] < 1e-20


def test_a_qr_solve_refuses_an_underdetermined_system(rng):
    narrow = pl.DataFrame({name: rng.normal(size=3) for name in ("y", "a", "b", "c", "d")})

    with pytest.raises(pl.exceptions.ComputeError, match="at least as many observations"):
        narrow.select(pf.least_squares("y", ["a", "b", "c", "d"]))
