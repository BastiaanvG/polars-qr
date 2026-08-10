import numpy as np
import polars as pl
import pytest

import polars_qr as pq
import reference

MAX_LAG = 8


def simulate(rng, coefficients, n, scale=1.0, burn=200):
    """A path from an autoregression with the coefficients given."""
    order = len(coefficients)
    series = np.zeros(n + burn)
    noise = rng.normal(scale=scale, size=n + burn)
    for row in range(order, n + burn):
        series[row] = coefficients @ series[row - order : row][::-1] + noise[row]
    return series[burn:]


@pytest.fixture
def series(rng):
    """An AR(2) path long enough for the estimates to settle."""
    return pl.DataFrame({"y": simulate(rng, np.array([0.6, -0.3]), 4000)})


def values_of(frame, column="y"):
    return frame[column].to_list()


def fit_of(frame, **kwargs):
    return frame.select(pq.autoregression.fit("y", **kwargs).alias("ar")).unnest("ar")


def test_the_sequence_matches_the_reference(series):
    out = series.select(pq.autoregression.autocovariance("y", max_lag=MAX_LAG).alias("g")).unnest(
        "g"
    )

    want = reference.autocovariance(values_of(series), MAX_LAG)
    assert out["autocovariance"][0].to_list() == pytest.approx(want, rel=1e-12)
    assert out["lags"][0].to_list() == list(range(MAX_LAG + 1))
    assert out["series"][0] == "y"
    assert out["n_observations"][0] == series.height
    assert out["mean"][0] == pytest.approx(np.mean(values_of(series)))


def test_the_correlation_is_the_covariance_over_its_first_value(series):
    out = series.select(pq.autoregression.autocorrelation("y", max_lag=MAX_LAG).alias("r")).unnest(
        "r"
    )

    want = reference.autocorrelation(values_of(series), MAX_LAG)
    assert out["autocorrelation"][0][0] == pytest.approx(1.0)
    assert out["autocorrelation"][0].to_list() == pytest.approx(want, rel=1e-12)


def test_the_unbiased_divisor_counts_only_the_products_it_has(series):
    biased = series.select(pq.autoregression.autocovariance("y", max_lag=MAX_LAG).alias("g"))
    unbiased = series.select(
        pq.autoregression.autocovariance("y", max_lag=MAX_LAG, unbiased=True).alias("g")
    )

    want = reference.autocovariance(values_of(series), MAX_LAG, unbiased=True)
    assert unbiased.unnest("g")["autocovariance"][0].to_list() == pytest.approx(want, rel=1e-12)
    # The two differ by the ratio of their divisors, so the unbiased one is the larger.
    for lag in range(1, MAX_LAG + 1):
        assert abs(unbiased.unnest("g")["autocovariance"][0][lag]) >= abs(
            biased.unnest("g")["autocovariance"][0][lag]
        )


def test_not_centring_leaves_the_level_of_the_series_in_the_sequence(series):
    shifted = series.with_columns(pl.col("y") + 100.0)

    out = shifted.select(
        pq.autoregression.autocovariance("y", max_lag=2, demean=False).alias("g")
    ).unnest("g")

    want = reference.autocovariance(values_of(shifted), 2, demean=False)
    assert out["mean"][0] == 0.0
    assert out["autocovariance"][0].to_list() == pytest.approx(want, rel=1e-9)
    assert out["autocovariance"][0][0] > 1e4


def test_the_recursion_agrees_with_solving_the_system_the_long_way(series):
    for order in (1, 2, 5, MAX_LAG):
        out = fit_of(series, order=order)

        want = reference.yule_walker(values_of(series), order)
        assert out["coefficients"][0].to_list() == pytest.approx(want, rel=1e-9, abs=1e-12)
        assert out["order"][0] == order


def test_a_long_enough_series_recovers_the_coefficients_it_was_built_from(rng):
    frame = pl.DataFrame({"y": simulate(rng, np.array([0.7]), 200_000)})

    out = fit_of(frame, order=1)

    assert out["coefficients"][0][0] == pytest.approx(0.7, abs=0.01)
    assert out["variance"][0] == pytest.approx(1.0, abs=0.05)


def test_the_partial_autocorrelation_is_the_last_coefficient_of_its_own_order(series):
    out = series.select(
        pq.autoregression.partial_autocorrelation("y", max_lag=MAX_LAG).alias("p")
    ).unnest("p")

    want = reference.partial_autocorrelation(values_of(series), MAX_LAG)
    assert out["partial_autocorrelation"][0].to_list() == pytest.approx(want, rel=1e-8, abs=1e-12)
    assert out["lags"][0].to_list() == list(range(1, MAX_LAG + 1))


def test_a_fit_carries_the_same_partial_autocorrelations(series):
    fit = fit_of(series, order="bic", max_order=MAX_LAG)
    apart = series.select(
        pq.autoregression.partial_autocorrelation("y", max_lag=MAX_LAG).alias("p")
    ).unnest("p")

    assert fit["partial_autocorrelations"][0].to_list() == pytest.approx(
        apart["partial_autocorrelation"][0].to_list()
    )


def test_every_reflection_coefficient_stays_inside_the_unit_circle(rng):
    for coefficients in ([0.95], [0.6, -0.3], [0.3, 0.3, 0.3], [-0.9]):
        frame = pl.DataFrame({"y": simulate(rng, np.array(coefficients), 400)})

        out = fit_of(frame, order=12)

        assert all(abs(value) < 1.0 for value in out["partial_autocorrelations"][0])
        assert out["stationary"][0]


def test_the_prediction_error_falls_with_every_order(series):
    out = fit_of(series, order="aic", max_order=MAX_LAG)

    variances = out["order_variance"][0].to_list()
    assert len(variances) == MAX_LAG + 1
    assert variances == sorted(variances, reverse=True)
    assert out["variance"][0] == variances[out["order"][0]]


def test_the_criterion_matches_the_reference_and_picks_its_own_smallest(series):
    for name in ("aic", "bic", "hqic"):
        out = fit_of(series, order=name, max_order=MAX_LAG)

        want = reference.criterion(values_of(series), MAX_LAG, name)
        got = out["criterion"][0].to_list()
        assert got == pytest.approx(want, rel=1e-9)
        assert out["order"][0] == int(np.argmin(got))


def test_a_given_order_reports_no_criterion(series):
    out = fit_of(series, order=3)

    assert out["criterion"][0] is None
    assert out["order"][0] == 3
    assert len(out["coefficients"][0]) == 3


def test_white_noise_is_chosen_as_white_noise(rng):
    frame = pl.DataFrame({"y": rng.normal(size=20_000)})

    out = fit_of(frame, order="bic", max_order=10)

    assert out["order"][0] == 0
    assert out["coefficients"][0].to_list() == []
    assert out["variance"][0] == pytest.approx(1.0, abs=0.05)


def test_burg_agrees_with_yule_walker_on_a_long_series(series):
    walker = fit_of(series, order=2)
    burg = fit_of(series, order=2, method="burg")

    assert burg["method"][0] == "burg"
    assert walker["method"][0] == "yule_walker"
    assert burg["coefficients"][0].to_list() == pytest.approx(
        walker["coefficients"][0].to_list(), abs=0.02
    )
    assert burg["stationary"][0]


def test_the_biased_divisor_is_what_keeps_a_short_series_fittable(rng):
    """The reason `fit` never uses the unbiased sequence, checked rather than asserted.

    On a short series close to a unit root the unbiased estimate is not positive
    semi-definite, so the system it describes has no valid recursion. The biased one is,
    which is what makes the stationarity guarantee true.
    """
    frame = pl.DataFrame({"y": simulate(rng, np.array([0.95]), 40)})
    sequences = {
        unbiased: frame.select(
            pq.autoregression.autocovariance("y", max_lag=20, unbiased=unbiased).alias("g")
        ).unnest("g")["autocovariance"][0]
        for unbiased in (False, True)
    }
    system = pl.DataFrame(
        {
            "lag": np.arange(21),
            "biased": sequences[False],
            "unbiased": sequences[True],
            "rhs": np.ones(21),
        }
    )

    solved = system.select(
        pq.autoregression.solve_toeplitz("biased", "rhs", row_index="lag").alias("s")
    )
    assert solved.height == 1
    with pytest.raises(pl.exceptions.ComputeError, match="positive definite"):
        system.select(pq.autoregression.solve_toeplitz("unbiased", "rhs", row_index="lag"))

    assert fit_of(frame, order=20)["stationary"][0]


def test_a_ramp_is_still_fittable_at_almost_as_many_lags_as_it_has_rows():
    frame = pl.DataFrame({"y": [float(row) for row in range(50)]})

    out = fit_of(frame, order=45)

    assert out["variance"][0] > 0.0
    assert out["stationary"][0]


def test_a_series_that_does_not_move_has_nothing_to_model():
    frame = pl.DataFrame({"y": [2.0] * 20})

    with pytest.raises(pl.exceptions.ComputeError, match="does not move"):
        fit_of(frame, order=2)


def test_a_lag_no_pair_of_rows_reaches_is_refused():
    frame = pl.DataFrame({"y": [1.0, 2.0, 3.0]})

    with pytest.raises(pl.exceptions.ComputeError, match="No pair of observations"):
        fit_of(frame, order=3)


def test_the_transform_matches_the_reference(series):
    out = series.with_columns(
        residual=pq.autoregression.transform("y", order=3),
        prediction=pq.autoregression.transform("y", order=3, output="prediction"),
    )

    values = values_of(series)
    assert out["residual"].to_list() == pytest.approx(
        reference.autoregression_transform(values, 3), rel=1e-8, abs=1e-12, nan_ok=True
    )
    assert out["prediction"].to_list() == pytest.approx(
        reference.autoregression_transform(values, 3, output="prediction"),
        rel=1e-8,
        abs=1e-12,
        nan_ok=True,
    )


def test_the_residual_is_what_the_prediction_left_over(series):
    out = series.with_columns(
        residual=pq.autoregression.transform("y", order=2),
        prediction=pq.autoregression.transform("y", order=2, output="prediction"),
    ).drop_nulls()

    left_over = out["y"] - out["prediction"] - out["residual"]
    assert left_over.abs().max() < 1e-9


def test_the_first_rows_of_a_fit_have_no_history_to_use(series):
    out = series.select(pq.autoregression.transform("y", order=4).alias("residual"))

    assert out["residual"].head(4).to_list() == [None] * 4
    assert out["residual"].null_count() == 4
    assert out.height == series.height


def test_the_residual_has_lost_the_autocorrelation_the_series_had(series):
    whitened = series.with_columns(residual=pq.autoregression.transform("y", order=2)).drop_nulls()

    before = reference.autocorrelation(values_of(series), 4)[1:]
    after = reference.autocorrelation(values_of(whitened, "residual"), 4)[1:]
    assert max(abs(value) for value in before) > 0.2
    assert max(abs(value) for value in after) < 0.05


def test_a_null_is_an_error_unless_it_is_asked_to_stand_in():
    frame = pl.DataFrame({"y": [1.0, None, 3.0, 2.0, 4.0, 1.5, 2.5, 3.5]})

    with pytest.raises(pl.exceptions.ComputeError, match="null value"):
        fit_of(frame, order=2)

    out = fit_of(frame, order=2, null_policy="zero")
    assert out["n_observations"][0] == frame.height


def test_a_row_the_filter_could_not_reach_has_no_answer():
    frame = pl.DataFrame({"y": [1.0, 4.0, None, 2.0, 5.0, 1.5, 2.5, 3.5]})

    out = frame.select(
        pq.autoregression.transform("y", order=1, null_policy="zero").alias("residual")
    )

    # Row 0 has no history, row 2 is missing, and row 3 needs row 2 as its lag.
    assert [value is None for value in out["residual"].to_list()] == [
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_a_value_that_is_not_finite_is_always_an_error():
    frame = pl.DataFrame({"y": [1.0, float("inf"), 3.0, 2.0, 4.0, 1.5]})

    with pytest.raises(pl.exceptions.ComputeError, match="not finite"):
        fit_of(frame, order=2, null_policy="zero")


def test_a_criterion_needs_something_to_choose_from():
    with pytest.raises(ValueError, match="max_order"):
        pq.autoregression.fit("y", order="bic")


def test_an_order_that_is_neither_a_number_nor_a_criterion_is_refused():
    with pytest.raises(ValueError, match="aic"):
        pq.autoregression.fit("y", order="best", max_order=4)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="integer"):
        pq.autoregression.fit("y", order=2.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="negative"):
        pq.autoregression.fit("y", order=-1)


def test_solving_a_toeplitz_system_agrees_with_solving_it_as_a_dense_one(rng):
    size = 10
    first_column = np.array([1.0 / (1.0 + lag) for lag in range(size)])
    frame = pl.DataFrame(
        {
            "lag": np.arange(size),
            "g": first_column,
            "a": rng.normal(size=size),
            "b": rng.normal(size=size),
        }
    )

    out = frame.select(
        pq.autoregression.solve_toeplitz("g", ["a", "b"], row_index="lag").alias("s")
    ).unnest("s")

    matrix = reference.toeplitz(first_column.tolist())
    for index, name in enumerate(["a", "b"]):
        want = reference.dense_solve(matrix, frame[name].to_list())
        assert out["solution"][0][index].to_list() == pytest.approx(want, rel=1e-8)
    assert out["rhs"][0].to_list() == ["a", "b"]
    assert out["size"][0] == size
    assert out["diagonal_shift"][0] == 0.0


def test_the_toeplitz_solve_agrees_with_the_dense_one_in_the_package(rng):
    size = 6
    first_column = np.array([2.0**-lag for lag in range(size)])
    dense = np.array(reference.toeplitz(first_column.tolist()))
    frame = pl.DataFrame(
        {"lag": np.arange(size), "g": first_column, "rhs": rng.normal(size=size)}
    ).with_columns(**{f"m{column}": pl.Series(dense[:, column]) for column in range(size)})

    fast = frame.select(
        pq.autoregression.solve_toeplitz("g", "rhs", row_index="lag").alias("s")
    ).unnest("s")
    slow = frame.select(
        pq.solve_spd([f"m{column}" for column in range(size)], "rhs", row_index="lag").alias("s")
    ).unnest("s")

    assert fast["solution"][0][0].to_list() == pytest.approx(slow["solution"][0][0].to_list())


def test_the_rows_are_read_in_the_order_the_index_gives(rng):
    size = 6
    first_column = np.array([2.0**-lag for lag in range(size)])
    frame = pl.DataFrame({"lag": np.arange(size), "g": first_column, "rhs": rng.normal(size=size)})

    ordered = frame.select(
        pq.autoregression.solve_toeplitz("g", "rhs", row_index="lag").alias("s")
    ).unnest("s")
    shuffled = (
        frame.sample(fraction=1.0, shuffle=True, seed=1)
        .select(pq.autoregression.solve_toeplitz("g", "rhs", row_index="lag").alias("s"))
        .unnest("s")
    )

    assert ordered["solution"][0][0].to_list() == pytest.approx(
        shuffled["solution"][0][0].to_list()
    )


def test_a_toeplitz_matrix_that_is_not_positive_definite_is_reported():
    frame = pl.DataFrame({"lag": [0, 1, 2], "g": [1.0, 2.0, 0.0], "rhs": [1.0, 1.0, 1.0]})

    with pytest.raises(pl.exceptions.ComputeError, match="positive definite"):
        frame.select(pq.autoregression.solve_toeplitz("g", "rhs", row_index="lag"))


def test_a_shift_makes_a_singular_toeplitz_system_solvable():
    frame = pl.DataFrame({"lag": [0, 1, 2], "g": [1.0, 1.0, 1.0], "rhs": [1.0, 1.0, 1.0]})

    with pytest.raises(pl.exceptions.ComputeError, match="positive definite"):
        frame.select(pq.autoregression.solve_toeplitz("g", "rhs", row_index="lag"))

    out = frame.select(
        pq.autoregression.solve_toeplitz("g", "rhs", row_index="lag", diagonal_shift=1e-6).alias(
            "s"
        )
    ).unnest("s")
    assert out["diagonal_shift"][0] == 1e-6


def test_a_grouped_fit_is_the_per_group_fits_side_by_side(rng):
    frame = pl.concat(
        [
            pl.DataFrame({"g": [name] * 500, "y": simulate(rng, np.array(phi), 500)})
            for name, phi in (("a", [0.7]), ("b", [-0.4, 0.2]))
        ]
    )

    together = (
        frame.lazy()
        .group_by("g", maintain_order=True)
        .agg(pq.autoregression.fit("y", order=2).alias("ar"))
        .unnest("ar")
        .collect()
    )

    for row, name in enumerate(["a", "b"]):
        apart = fit_of(frame.filter(pl.col("g") == name), order=2)
        assert together["coefficients"][row].to_list() == pytest.approx(
            apart["coefficients"][0].to_list()
        )
        assert together["series"][row] == "y"


def test_chunked_input_gives_the_same_answer(series):
    chunked = pl.concat(
        [series.head(1000), series.slice(1000, 1500), series.tail(1500)], rechunk=False
    )

    out = fit_of(chunked, order=4)

    want = fit_of(series.rechunk(), order=4)
    assert out["coefficients"][0].to_list() == pytest.approx(want["coefficients"][0].to_list())


def test_the_namespace_hands_over_to_the_function(series):
    for through_namespace, through_function in (
        (pl.col("y").ar.fit(order=2), pq.autoregression.fit("y", order=2)),  # type: ignore[attr-defined]
        (
            pl.col("y").ar.autocovariance(max_lag=3),  # type: ignore[attr-defined]
            pq.autoregression.autocovariance("y", max_lag=3),
        ),
        (
            pl.col("y").ar.autocorrelation(max_lag=3),  # type: ignore[attr-defined]
            pq.autoregression.autocorrelation("y", max_lag=3),
        ),
        (
            pl.col("y").ar.partial_autocorrelation(max_lag=3),  # type: ignore[attr-defined]
            pq.autoregression.partial_autocorrelation("y", max_lag=3),
        ),
        (
            pl.col("y").ar.transform(order=2),  # type: ignore[attr-defined]
            pq.autoregression.transform("y", order=2),
        ),
    ):
        assert series.select(through_namespace.alias("out")).equals(
            series.select(through_function.alias("out"))
        )
