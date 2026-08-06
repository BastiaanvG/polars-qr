from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

import polars_faer as pf
import reference

HALF_LIFE = 40.0
WINDOW = 100.0


@pytest.fixture
def series(rng):
    """A clock that moves in uneven steps, with values and a couple of gaps."""
    n = 300
    steps = rng.integers(0, 40, size=n)
    clock = np.cumsum(steps).astype(np.int64)
    x = rng.normal(size=n)
    y = 0.6 * x + rng.normal(scale=0.5, size=n)
    return pl.DataFrame({"clock": clock, "x": x, "y": y})


def values_of(frame, column):
    return frame[column].to_list()


def assert_matches(got, want, tolerance=1e-9):
    assert len(got) == len(want)
    for row, (a, b) in enumerate(zip(got, want, strict=True)):
        if b is None:
            assert a is None, f"row {row}: expected null, got {a}"
        else:
            assert a is not None, f"row {row}: expected {b}, got null"
            assert abs(a - b) <= tolerance * max(1.0, abs(b)), f"row {row}: {a} != {b}"


# --------------------------------------------------------------------- EWM


def test_ewm_sum_matches_the_reference(series):
    out = series.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=HALF_LIFE).alias("out"))

    want = reference.ewm_sum(values_of(series, "x"), values_of(series, "clock"), HALF_LIFE)
    assert_matches(out["out"].to_list(), want)


def test_ewm_sum_follows_its_recursion(series):
    out = series.select(
        pf.timeseries.ewm_sum("x", clock="clock", half_life=HALF_LIFE).alias("out")
    )["out"].to_list()

    clock = values_of(series, "clock")
    x = values_of(series, "x")
    for row in range(1, len(x)):
        lam = reference.decay(clock[row] - clock[row - 1], HALF_LIFE)
        assert out[row] == pytest.approx(x[row] + lam * out[row - 1], rel=1e-9, abs=1e-12)


def test_ewm_mean_is_the_sum_over_its_weight(series):
    out = series.select(
        pf.timeseries.ewm_mean("x", clock="clock", half_life=HALF_LIFE).alias("out")
    )

    want = reference.ewm_mean(values_of(series, "x"), values_of(series, "clock"), HALF_LIFE)
    assert_matches(out["out"].to_list(), want)


def test_ewm_variance_matches_the_reference(series):
    for bias in (False, True):
        out = series.select(
            pf.timeseries.ewm_variance("x", clock="clock", half_life=HALF_LIFE, bias=bias).alias(
                "out"
            )
        )

        want = reference.ewm_variance(
            values_of(series, "x"), values_of(series, "clock"), HALF_LIFE, bias=bias
        )
        assert_matches(out["out"].to_list(), want)


def test_ewm_covariance_matches_the_reference(series):
    for bias in (False, True):
        out = series.select(
            pf.timeseries.ewm_covariance(
                "x", "y", clock="clock", half_life=HALF_LIFE, bias=bias
            ).alias("out")
        )

        want = reference.ewm_covariance(
            values_of(series, "x"),
            values_of(series, "y"),
            values_of(series, "clock"),
            HALF_LIFE,
            bias=bias,
        )
        assert_matches(out["out"].to_list(), want)


def test_ewm_correlation_matches_the_reference(series):
    out = series.select(
        pf.timeseries.ewm_correlation("x", "y", clock="clock", half_life=HALF_LIFE).alias("out")
    )

    want = reference.ewm_correlation(
        values_of(series, "x"),
        values_of(series, "y"),
        values_of(series, "clock"),
        HALF_LIFE,
    )
    assert_matches(out["out"].to_list(), want)


def test_ewm_correlation_stays_within_its_range(series):
    out = series.select(
        pf.timeseries.ewm_correlation("x", "y", clock="clock", half_life=HALF_LIFE).alias("out")
    )["out"].to_list()

    assert all(-1.0 <= value <= 1.0 for value in out if value is not None)


def test_a_variance_clock_of_zero_distance_does_not_decay():
    frame = pl.DataFrame({"clock": [5, 5, 5], "x": [1.0, 2.0, 3.0]})

    out = frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=10.0).alias("out"))

    # Nothing decays when the clock does not move, so this is the plain running sum.
    assert out["out"].to_list() == [1.0, 3.0, 6.0]


def test_a_gap_wide_enough_forgets_what_came_before():
    frame = pl.DataFrame({"clock": [0, 10_000], "x": [100.0, 1.0]})

    out = frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=1.0).alias("out"))

    assert out["out"].to_list() == [100.0, 1.0]


def test_one_half_life_halves_what_came_before():
    frame = pl.DataFrame({"clock": [0, 10], "x": [8.0, 0.0]})

    out = frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=10.0).alias("out"))

    assert out["out"].to_list() == pytest.approx([8.0, 4.0])


# ----------------------------------------------------------------- rolling


def test_rolling_sum_matches_the_reference(series):
    out = series.select(pf.timeseries.rolling_sum("x", clock="clock", window=WINDOW).alias("out"))

    want = reference.rolling_sum(values_of(series, "x"), values_of(series, "clock"), WINDOW)
    assert_matches(out["out"].to_list(), want)


def test_rolling_mean_matches_the_reference(series):
    out = series.select(pf.timeseries.rolling_mean("x", clock="clock", window=WINDOW).alias("out"))

    want = reference.rolling_mean(values_of(series, "x"), values_of(series, "clock"), WINDOW)
    assert_matches(out["out"].to_list(), want)


def test_rolling_variance_matches_the_reference(series):
    for ddof in (0, 1):
        out = series.select(
            pf.timeseries.rolling_variance("x", clock="clock", window=WINDOW, ddof=ddof).alias(
                "out"
            )
        )

        want = reference.rolling_variance(
            values_of(series, "x"), values_of(series, "clock"), WINDOW, ddof=ddof
        )
        assert_matches(out["out"].to_list(), want)


def test_rolling_covariance_matches_the_reference(series):
    out = series.select(
        pf.timeseries.rolling_covariance("x", "y", clock="clock", window=WINDOW).alias("out")
    )

    want = reference.rolling_covariance(
        values_of(series, "x"), values_of(series, "y"), values_of(series, "clock"), WINDOW
    )
    assert_matches(out["out"].to_list(), want)


def test_rolling_correlation_matches_the_reference(series):
    out = series.select(
        pf.timeseries.rolling_correlation("x", "y", clock="clock", window=WINDOW).alias("out")
    )

    want = reference.rolling_correlation(
        values_of(series, "x"), values_of(series, "y"), values_of(series, "clock"), WINDOW
    )
    assert_matches(out["out"].to_list(), want)


def test_the_window_is_open_at_the_far_end_and_closed_at_the_near_one():
    # With a window of 10 the row at clock 10 sees clock 1 through 10, not clock 0.
    frame = pl.DataFrame({"clock": [0, 1, 10], "x": [1.0, 2.0, 4.0]})

    out = frame.select(pf.timeseries.rolling_sum("x", clock="clock", window=10).alias("out"))

    assert out["out"].to_list() == [1.0, 3.0, 6.0]


def test_rows_sharing_a_clock_value_are_all_visible():
    frame = pl.DataFrame({"clock": [0, 0, 0], "x": [1.0, 2.0, 3.0]})

    out = frame.select(pf.timeseries.rolling_sum("x", clock="clock", window=1).alias("out"))

    # Earlier rows at the same clock are in; later ones are not yet.
    assert out["out"].to_list() == [1.0, 3.0, 6.0]


def test_min_samples_holds_the_result_back(series):
    out = series.select(
        pf.timeseries.rolling_mean("x", clock="clock", window=WINDOW, min_samples=5).alias("out")
    )

    want = reference.rolling_mean(
        values_of(series, "x"), values_of(series, "clock"), WINDOW, min_samples=5
    )
    assert_matches(out["out"].to_list(), want)
    assert out["out"][0] is None


def test_min_clock_span_holds_the_result_back(series):
    out = series.select(
        pf.timeseries.rolling_variance("x", clock="clock", window=WINDOW, min_clock_span=60).alias(
            "out"
        )
    )

    want = reference.rolling_variance(
        values_of(series, "x"), values_of(series, "clock"), WINDOW, min_clock_span=60
    )
    assert_matches(out["out"].to_list(), want)


def test_a_minimum_span_wider_than_the_window_is_rejected(series):
    with pytest.raises(pl.exceptions.ComputeError, match=r"min_clock_span <= window"):
        series.select(pf.timeseries.rolling_sum("x", clock="clock", window=10, min_clock_span=20))


# ------------------------------------------------------------------ clocks


def test_a_float_clock_is_read_in_its_own_units():
    frame = pl.DataFrame({"clock": [0.0, 0.5, 1.5], "x": [1.0, 1.0, 1.0]})

    out = frame.select(pf.timeseries.rolling_sum("x", clock="clock", window=1.0).alias("out"))

    assert out["out"].to_list() == [1.0, 2.0, 1.0]


def test_a_datetime_clock_takes_a_duration():
    frame = pl.DataFrame(
        {
            "clock": [
                datetime(2026, 8, 6, 9, 0),
                datetime(2026, 8, 6, 9, 30),
                datetime(2026, 8, 6, 11, 0),
            ],
            "x": [1.0, 1.0, 1.0],
        }
    )

    out = frame.select(
        pf.timeseries.rolling_sum("x", clock="clock", window=timedelta(hours=1)).alias("out")
    )

    assert out["out"].to_list() == [1.0, 2.0, 1.0]


def test_a_date_clock_counts_in_days():
    frame = pl.DataFrame(
        {"clock": [date(2026, 8, 1), date(2026, 8, 2), date(2026, 8, 9)], "x": [1.0, 1.0, 1.0]}
    )

    out = frame.select(
        pf.timeseries.rolling_sum("x", clock="clock", window=timedelta(days=3)).alias("out")
    )

    assert out["out"].to_list() == [1.0, 2.0, 1.0]


def test_a_temporal_clock_refuses_a_number():
    frame = pl.DataFrame({"clock": [datetime(2026, 8, 6, 9, 0)], "x": [1.0]})

    with pytest.raises(pl.exceptions.ComputeError, match=r"Use datetime\.timedelta"):
        frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=5))


def test_a_numeric_clock_refuses_a_duration():
    frame = pl.DataFrame({"clock": [1, 2], "x": [1.0, 1.0]})

    with pytest.raises(pl.exceptions.ComputeError, match="Use a number in the units"):
        frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=timedelta(hours=1)))


def test_a_clock_that_goes_backwards_is_rejected():
    frame = pl.DataFrame({"clock": [10, 20, 15], "x": [1.0, 1.0, 1.0]})

    with pytest.raises(pl.exceptions.ComputeError, match="non-decreasing clock"):
        frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=5))


def test_a_null_clock_is_rejected():
    frame = pl.DataFrame({"clock": [1, None, 3], "x": [1.0, 1.0, 1.0]})

    with pytest.raises(pl.exceptions.ComputeError, match="null clock"):
        frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=5))


def test_a_clock_that_is_not_finite_is_rejected():
    frame = pl.DataFrame({"clock": [1.0, float("nan")], "x": [1.0, 1.0]})

    with pytest.raises(pl.exceptions.ComputeError, match="not finite"):
        frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=5))


def test_a_clock_that_is_not_a_number_or_a_time_is_rejected():
    frame = pl.DataFrame({"clock": ["a", "b"], "x": [1.0, 1.0]})

    with pytest.raises(pl.exceptions.ComputeError, match=r"not numeric|must be an integer"):
        frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=5))


def test_a_half_life_must_be_positive():
    frame = pl.DataFrame({"clock": [1, 2], "x": [1.0, 1.0]})

    with pytest.raises(pl.exceptions.ComputeError, match=r"half_life > 0"):
        frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=0))


# ------------------------------------------------------------------- nulls


def test_a_null_lets_the_state_decay_past_it():
    frame = pl.DataFrame({"clock": [0, 10, 20], "x": [8.0, None, 0.0]})

    out = frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=10.0).alias("out"))

    # The clock still advances, so the first value has halved twice by the last row.
    assert out["out"].to_list() == pytest.approx([8.0, 4.0, 2.0])


def test_a_null_can_be_made_an_error():
    frame = pl.DataFrame({"clock": [0, 10], "x": [1.0, None]})

    with pytest.raises(pl.exceptions.ComputeError, match="null value"):
        frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=10.0, null_policy="raise"))


def test_a_value_that_is_not_finite_is_always_an_error():
    frame = pl.DataFrame({"clock": [0, 10], "x": [1.0, float("inf")]})

    with pytest.raises(pl.exceptions.ComputeError, match="not finite"):
        frame.select(pf.timeseries.ewm_sum("x", clock="clock", half_life=10.0))


def test_a_pair_enters_only_when_both_values_are_there(series):
    holed = series.with_columns(
        pl.when(pl.arange(0, series.height) % 7 == 0).then(None).otherwise(pl.col("y")).alias("y")
    )

    out = holed.select(
        pf.timeseries.ewm_covariance("x", "y", clock="clock", half_life=HALF_LIFE).alias("out")
    )

    want = reference.ewm_covariance(
        values_of(holed, "x"),
        values_of(holed, "y"),
        values_of(holed, "clock"),
        HALF_LIFE,
    )
    assert_matches(out["out"].to_list(), want)


def test_nulls_in_a_rolling_window_are_skipped(series):
    holed = series.with_columns(
        pl.when(pl.arange(0, series.height) % 5 == 0).then(None).otherwise(pl.col("x")).alias("x")
    )

    out = holed.select(pf.timeseries.rolling_mean("x", clock="clock", window=WINDOW).alias("out"))

    want = reference.rolling_mean(values_of(holed, "x"), values_of(holed, "clock"), WINDOW)
    assert_matches(out["out"].to_list(), want)


# ------------------------------------------------------------- integration


def test_every_operation_keeps_one_row_per_input_row(series):
    out = series.select(
        pf.timeseries.ewm_sum("x", clock="clock", half_life=HALF_LIFE).alias("a"),
        pf.timeseries.ewm_correlation("x", "y", clock="clock", half_life=HALF_LIFE).alias("b"),
        pf.timeseries.rolling_variance("x", clock="clock", window=WINDOW).alias("c"),
        pf.timeseries.rolling_covariance("x", "y", clock="clock", window=WINDOW).alias("d"),
    )

    assert out.height == series.height
    assert out.dtypes == [pl.Float64] * 4


def test_a_group_is_computed_on_its_own_rows(series, rng):
    grouped = series.with_columns(pl.Series("symbol", rng.integers(0, 3, size=series.height))).sort(
        "symbol", "clock"
    )

    out = grouped.with_columns(
        pf.timeseries.ewm_sum("x", clock="clock", half_life=HALF_LIFE).over("symbol").alias("out")
    )

    for symbol in out["symbol"].unique():
        rows = out.filter(pl.col("symbol") == symbol)
        want = reference.ewm_sum(values_of(rows, "x"), values_of(rows, "clock"), HALF_LIFE)
        assert_matches(rows["out"].to_list(), want)


def test_a_group_matches_running_that_group_on_its_own(series, rng):
    grouped = series.with_columns(pl.Series("symbol", rng.integers(0, 3, size=series.height))).sort(
        "symbol", "clock"
    )

    together = grouped.with_columns(
        pf.timeseries.rolling_correlation("x", "y", clock="clock", window=WINDOW)
        .over("symbol")
        .alias("out")
    )
    apart = pl.concat(
        [
            grouped.filter(pl.col("symbol") == symbol).with_columns(
                pf.timeseries.rolling_correlation("x", "y", clock="clock", window=WINDOW).alias(
                    "out"
                )
            )
            for symbol in sorted(grouped["symbol"].unique())
        ]
    )

    assert_matches(together["out"].to_list(), apart["out"].to_list())


def test_a_lazy_query_gives_the_same_answer(series):
    eager = series.select(
        pf.timeseries.ewm_variance("x", clock="clock", half_life=HALF_LIFE).alias("out")
    )
    lazy = (
        series.lazy()
        .select(pf.timeseries.ewm_variance("x", clock="clock", half_life=HALF_LIFE).alias("out"))
        .collect()
    )

    assert eager.equals(lazy)


def test_the_schema_is_known_without_collecting(series):
    schema = (
        series.lazy()
        .select(pf.timeseries.ewm_sum("x", clock="clock", half_life=HALF_LIFE).alias("out"))
        .collect_schema()
    )

    assert schema["out"] == pl.Float64


def test_chunked_input_gives_the_same_answer(series):
    chunked = pl.concat([series.head(100), series.slice(100, 100), series.tail(100)], rechunk=False)

    out = chunked.select(
        pf.timeseries.ewm_mean("x", clock="clock", half_life=HALF_LIFE).alias("out")
    )
    want = series.rechunk().select(
        pf.timeseries.ewm_mean("x", clock="clock", half_life=HALF_LIFE).alias("out")
    )

    assert_matches(out["out"].to_list(), want["out"].to_list())


# -------------------------------------------------------------- properties


def test_shifting_the_values_shifts_the_mean_and_leaves_the_spread(series):
    shifted = series.with_columns((pl.col("x") + 100.0).alias("x"))

    plain = series.select(
        pf.timeseries.ewm_mean("x", clock="clock", half_life=HALF_LIFE).alias("mean"),
        pf.timeseries.ewm_variance("x", clock="clock", half_life=HALF_LIFE).alias("variance"),
    )
    moved = shifted.select(
        pf.timeseries.ewm_mean("x", clock="clock", half_life=HALF_LIFE).alias("mean"),
        pf.timeseries.ewm_variance("x", clock="clock", half_life=HALF_LIFE).alias("variance"),
    )

    assert_matches(
        moved["mean"].to_list(), [value + 100.0 for value in plain["mean"].to_list()], 1e-10
    )
    assert_matches(moved["variance"].to_list(), plain["variance"].to_list(), 1e-8)


def test_scaling_the_values_scales_the_moments(series):
    scaled = series.with_columns((pl.col("x") * 3.0).alias("x"))

    plain = series.select(
        pf.timeseries.ewm_variance("x", clock="clock", half_life=HALF_LIFE).alias("variance"),
        pf.timeseries.ewm_covariance("x", "y", clock="clock", half_life=HALF_LIFE).alias("cov"),
        pf.timeseries.ewm_correlation("x", "y", clock="clock", half_life=HALF_LIFE).alias("corr"),
    )
    bigger = scaled.select(
        pf.timeseries.ewm_variance("x", clock="clock", half_life=HALF_LIFE).alias("variance"),
        pf.timeseries.ewm_covariance("x", "y", clock="clock", half_life=HALF_LIFE).alias("cov"),
        pf.timeseries.ewm_correlation("x", "y", clock="clock", half_life=HALF_LIFE).alias("corr"),
    )

    assert_matches(
        bigger["variance"].to_list(),
        [None if v is None else v * 9.0 for v in plain["variance"].to_list()],
    )
    assert_matches(
        bigger["cov"].to_list(),
        [None if v is None else v * 3.0 for v in plain["cov"].to_list()],
    )
    assert_matches(bigger["corr"].to_list(), plain["corr"].to_list())


def test_a_column_correlates_perfectly_with_itself(series):
    out = series.select(
        pf.timeseries.ewm_correlation("x", "x", clock="clock", half_life=HALF_LIFE).alias("out"),
        pf.timeseries.rolling_correlation("x", "x", clock="clock", window=WINDOW).alias("rolling"),
    )

    for column in ("out", "rolling"):
        values = [value for value in out[column].to_list() if value is not None]
        assert values
        assert all(abs(value - 1.0) < 1e-9 for value in values)


def test_covariance_does_not_depend_on_which_column_is_which(series):
    one = series.select(
        pf.timeseries.ewm_covariance("x", "y", clock="clock", half_life=HALF_LIFE).alias("out")
    )
    other = series.select(
        pf.timeseries.ewm_covariance("y", "x", clock="clock", half_life=HALF_LIFE).alias("out")
    )

    assert_matches(one["out"].to_list(), other["out"].to_list())


def test_covariance_with_itself_is_its_variance(series):
    covariance = series.select(
        pf.timeseries.ewm_covariance("x", "x", clock="clock", half_life=HALF_LIFE).alias("out")
    )
    variance = series.select(
        pf.timeseries.ewm_variance("x", clock="clock", half_life=HALF_LIFE).alias("out")
    )

    assert_matches(covariance["out"].to_list(), variance["out"].to_list())


def test_large_offsets_do_not_swamp_a_small_spread(series):
    offset = series.with_columns((pl.col("x") + 1e9).alias("x"), (pl.col("y") + 1e9).alias("y"))

    plain = series.select(
        pf.timeseries.rolling_covariance("x", "y", clock="clock", window=WINDOW).alias("out")
    )
    shifted = offset.select(
        pf.timeseries.rolling_covariance("x", "y", clock="clock", window=WINDOW).alias("out")
    )

    assert_matches(shifted["out"].to_list(), plain["out"].to_list(), 1e-6)


# -------------------------------------------------------------- namespace


def test_the_expression_namespace_matches_the_function(series):
    through_function = series.select(
        pf.timeseries.ewm_correlation("x", "y", clock="clock", half_life=HALF_LIFE).alias("out")
    )
    through_namespace = series.select(
        pl.col("x").faer.ewm_correlation("y", clock="clock", half_life=HALF_LIFE).alias("out")  # type: ignore[attr-defined]
    )

    assert through_function.equals(through_namespace)


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("rolling_sum", {"window": WINDOW}),
        ("rolling_mean", {"window": WINDOW}),
        ("rolling_variance", {"window": WINDOW}),
        ("ewm_sum", {"half_life": HALF_LIFE}),
        ("ewm_mean", {"half_life": HALF_LIFE}),
        ("ewm_variance", {"half_life": HALF_LIFE}),
    ],
)
def test_every_univariate_namespace_method_matches_its_function(series, method, arguments):
    through_function = series.select(
        getattr(pf.timeseries, method)("x", clock="clock", **arguments).alias("out")
    )
    through_namespace = series.select(
        getattr(pl.col("x").faer, method)(clock="clock", **arguments).alias("out")  # type: ignore[attr-defined]
    )

    assert through_function.equals(through_namespace)


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("rolling_covariance", {"window": WINDOW}),
        ("rolling_correlation", {"window": WINDOW}),
        ("ewm_covariance", {"half_life": HALF_LIFE}),
        ("ewm_correlation", {"half_life": HALF_LIFE}),
    ],
)
def test_every_bivariate_namespace_method_matches_its_function(series, method, arguments):
    through_function = series.select(
        getattr(pf.timeseries, method)("x", "y", clock="clock", **arguments).alias("out")
    )
    through_namespace = series.select(
        getattr(pl.col("x").faer, method)("y", clock="clock", **arguments).alias("out")  # type: ignore[attr-defined]
    )

    assert through_function.equals(through_namespace)
