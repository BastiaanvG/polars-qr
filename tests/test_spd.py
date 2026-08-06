import numpy as np
import polars as pl
import pytest

import polars_qr as pq

COLUMNS = ["m0", "m1", "m2"]


@pytest.fixture
def system(rng):
    factor = rng.normal(size=(3, 3))
    values = factor @ factor.T + 3.0 * np.eye(3)
    frame = pl.DataFrame(
        {
            "asset": [0, 1, 2],
            "m0": values[:, 0],
            "m1": values[:, 1],
            "m2": values[:, 2],
            "expected": [1.0, -2.0, 0.5],
            "exposure": [0.0, 1.0, 1.0],
        }
    )
    return frame, values


def test_the_solution_matches_a_reference_solve(system):
    frame, values = system

    out = frame.select(pq.solve_spd(COLUMNS, "expected", row_index="asset").alias("solved")).unnest(
        "solved"
    )

    expected = np.linalg.solve(values, frame["expected"].to_numpy())
    assert out["rows"][0].to_list() == COLUMNS
    assert out["rhs"][0].to_list() == ["expected"]
    assert np.allclose(out["solution"][0].to_list()[0], expected)
    assert out["size"][0] == 3
    assert out["symmetry_error"][0] < 1e-12


def test_several_right_hand_sides_are_solved_at_once(system):
    frame, values = system

    out = frame.select(
        pq.solve_spd(COLUMNS, ["expected", "exposure"], row_index="asset").alias("solved")
    ).unnest("solved")

    expected = np.linalg.solve(values, frame.select(["expected", "exposure"]).to_numpy())
    assert out["rhs"][0].to_list() == ["expected", "exposure"]
    assert np.allclose(out["solution"][0].to_list(), expected.T)


def test_the_row_index_fixes_the_order(system):
    frame, _ = system
    shuffled = frame.sort("expected")

    ordered = frame.select(
        pq.solve_spd(COLUMNS, "expected", row_index="asset").alias("solved")
    ).unnest("solved")
    out = shuffled.select(
        pq.solve_spd(COLUMNS, "expected", row_index="asset").alias("solved")
    ).unnest("solved")

    assert np.allclose(ordered["solution"][0].to_list(), out["solution"][0].to_list())


def test_a_shift_makes_a_singular_matrix_solvable():
    frame = pl.DataFrame(
        {
            "asset": [0, 1],
            "m0": [1.0, 1.0],
            "m1": [1.0, 1.0],
            "rhs": [1.0, 1.0],
        }
    )

    with pytest.raises(pl.exceptions.ComputeError, match="not positive definite"):
        frame.select(pq.solve_spd(["m0", "m1"], "rhs", row_index="asset"))

    out = frame.select(
        pq.solve_spd(["m0", "m1"], "rhs", row_index="asset", diagonal_shift=1e-6).alias("solved")
    ).unnest("solved")
    assert out["diagonal_shift"][0] == 1e-6


def test_an_asymmetric_matrix_is_rejected():
    frame = pl.DataFrame(
        {
            "asset": [0, 1],
            "m0": [4.0, 2.0],
            "m1": [1.0, 3.0],
            "rhs": [1.0, 1.0],
        }
    )

    with pytest.raises(pl.exceptions.ComputeError, match="not symmetric"):
        frame.select(pq.solve_spd(["m0", "m1"], "rhs", row_index="asset"))


def test_a_repeated_row_index_is_rejected(system):
    frame, _ = system
    repeated = frame.with_columns(pl.lit(0).alias("asset"))

    with pytest.raises(pl.exceptions.ComputeError, match="repeats a value"):
        repeated.select(pq.solve_spd(COLUMNS, "expected", row_index="asset"))


def test_a_non_integer_row_index_is_rejected(system):
    frame, _ = system
    labelled = frame.with_columns(pl.col("asset").cast(pl.Float64).alias("asset"))

    with pytest.raises(pl.exceptions.ComputeError, match="not an integer"):
        labelled.select(pq.solve_spd(COLUMNS, "expected", row_index="asset"))


def test_a_null_in_the_matrix_is_rejected(system):
    frame, _ = system
    holed = frame.with_columns(
        pl.when(pl.col("asset") == 1).then(None).otherwise(pl.col("m0")).alias("m0")
    )

    with pytest.raises(pl.exceptions.ComputeError, match="null or non-finite"):
        holed.select(pq.solve_spd(COLUMNS, "expected", row_index="asset"))


def test_a_covariance_matrix_can_be_solved_against(frame):
    features = ["a", "b", "c"]
    covariance = frame.select(pq.covariance(features).alias("cov")).unnest("cov")
    values = np.array(covariance["covariance"][0].to_list())

    wide = pl.DataFrame(
        {
            "row": [0, 1, 2],
            "a": values[:, 0],
            "b": values[:, 1],
            "c": values[:, 2],
            "signal": [1.0, 0.0, -1.0],
        }
    )

    out = wide.select(
        pq.solve_spd(features, "signal", row_index="row", diagonal_shift=1e-10).alias("solved")
    ).unnest("solved")

    expected = np.linalg.solve(values + 1e-10 * np.eye(3), wide["signal"].to_numpy())
    assert np.allclose(out["solution"][0].to_list()[0], expected)
