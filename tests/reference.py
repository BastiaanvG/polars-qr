"""Reference implementations of the timeseries statistics.

Written to be read rather than to be fast: one loop, one observation at a time, no state
that is not obvious. They are the oracle the compiled versions are checked against, so
where they disagree the reference is what says which one is wrong.
"""

import math


def decay(distance, half_life):
    return 2.0 ** (-distance / half_life) if distance > 0 else 1.0


def _weights(values, clock, half_life):
    """The weight every observation carries at each row."""
    for row in range(len(values)):
        weights = []
        for earlier in range(row + 1):
            if values[earlier] is None:
                weights.append(0.0)
            else:
                weights.append(decay(clock[row] - clock[earlier], half_life))
        yield weights


def _valid_count(values, row):
    return sum(1 for value in values[: row + 1] if value is not None)


def ewm_sum(values, clock, half_life, min_samples=1):
    out: list[float | None] = []
    for row, weights in enumerate(_weights(values, clock, half_life)):
        if _valid_count(values, row) < min_samples:
            out.append(None)
            continue
        out.append(
            sum(
                weight * value
                for weight, value in zip(weights, values[: len(weights)], strict=True)
                if value is not None
            )
        )
    return out


def ewm_mean(values, clock, half_life, min_samples=1):
    out: list[float | None] = []
    for row, weights in enumerate(_weights(values, clock, half_life)):
        if _valid_count(values, row) < min_samples:
            out.append(None)
            continue
        total = sum(weights)
        if total <= 0:
            out.append(None)
            continue
        out.append(
            sum(
                weight * value
                for weight, value in zip(weights, values[: len(weights)], strict=True)
                if value is not None
            )
            / total
        )
    return out


def _weighted_moments(values, weights):
    total = sum(weights)
    if total <= 0:
        return None
    mean = (
        sum(w * v for w, v in zip(weights, values[: len(weights)], strict=True) if v is not None)
        / total
    )
    m2 = sum(
        w * (v - mean) ** 2
        for w, v in zip(weights, values[: len(weights)], strict=True)
        if v is not None
    )
    return total, sum(w * w for w in weights), mean, m2


def ewm_variance(values, clock, half_life, min_samples=1, bias=False):
    out: list[float | None] = []
    for row, weights in enumerate(_weights(values, clock, half_life)):
        moments = _weighted_moments(values, weights)
        if _valid_count(values, row) < min_samples or moments is None:
            out.append(None)
            continue
        total, squared, _, m2 = moments
        denominator = total if bias else total - squared / total
        out.append(m2 / denominator if denominator > 0 else None)
    return out


def _pair_weights(x, y, clock, half_life, row):
    weights = []
    for earlier in range(row + 1):
        if x[earlier] is None or y[earlier] is None:
            weights.append(0.0)
        else:
            weights.append(decay(clock[row] - clock[earlier], half_life))
    return weights


def _joint_count(x, y, row):
    return sum(
        1
        for a, b in zip(x[: row + 1], y[: row + 1], strict=True)
        if a is not None and b is not None
    )


def _cross(x, y, weights):
    total = sum(weights)
    if total <= 0:
        return None
    mean_x = sum(w * v for w, v in zip(weights, x[: len(weights)], strict=True) if w > 0) / total
    mean_y = sum(w * v for w, v in zip(weights, y[: len(weights)], strict=True) if w > 0) / total
    cross = sum(
        w * (a - mean_x) * (b - mean_y)
        for w, a, b in zip(weights, x[: len(weights)], y[: len(weights)], strict=True)
        if w > 0
    )
    m2_x = sum(
        w * (a - mean_x) ** 2 for w, a in zip(weights, x[: len(weights)], strict=True) if w > 0
    )
    m2_y = sum(
        w * (b - mean_y) ** 2 for w, b in zip(weights, y[: len(weights)], strict=True) if w > 0
    )
    return total, sum(w * w for w in weights), cross, m2_x, m2_y


def ewm_covariance(x, y, clock, half_life, min_samples=1, bias=False):
    out: list[float | None] = []
    for row in range(len(x)):
        weights = _pair_weights(x, y, clock, half_life, row)
        moments = _cross(x, y, weights)
        if _joint_count(x, y, row) < min_samples or moments is None:
            out.append(None)
            continue
        total, squared, cross, _, _ = moments
        denominator = total if bias else total - squared / total
        out.append(cross / denominator if denominator > 0 else None)
    return out


def ewm_correlation(x, y, clock, half_life, min_samples=1):
    out: list[float | None] = []
    for row in range(len(x)):
        weights = _pair_weights(x, y, clock, half_life, row)
        moments = _cross(x, y, weights)
        if _joint_count(x, y, row) < min_samples or moments is None:
            out.append(None)
            continue
        _, _, cross, m2_x, m2_y = moments
        if m2_x <= 0 or m2_y <= 0:
            out.append(None)
            continue
        out.append(cross / math.sqrt(m2_x * m2_y))
    return out


def _window(clock, row, window):
    """The rows inside the trailing window `(clock[row] - window, clock[row]]`."""
    return [earlier for earlier in range(row + 1) if clock[row] - clock[earlier] < window]


def _kept(values, clock, row, window):
    return [earlier for earlier in _window(clock, row, window) if values[earlier] is not None]


def _report(kept, clock, row, min_samples, min_clock_span):
    if len(kept) < max(min_samples, 1):
        return False
    if min_clock_span is None:
        return True
    return bool(kept) and clock[row] - clock[kept[0]] >= min_clock_span


def rolling_sum(values, clock, window, min_samples=1, min_clock_span=None):
    out: list[float | None] = []
    for row in range(len(values)):
        kept = _kept(values, clock, row, window)
        if not _report(kept, clock, row, min_samples, min_clock_span):
            out.append(None)
            continue
        out.append(sum(values[index] for index in kept))
    return out


def rolling_mean(values, clock, window, min_samples=1, min_clock_span=None):
    out: list[float | None] = []
    for row in range(len(values)):
        kept = _kept(values, clock, row, window)
        if not _report(kept, clock, row, min_samples, min_clock_span):
            out.append(None)
            continue
        out.append(sum(values[index] for index in kept) / len(kept))
    return out


def rolling_variance(values, clock, window, min_samples=1, min_clock_span=None, ddof=1):
    out: list[float | None] = []
    for row in range(len(values)):
        kept = _kept(values, clock, row, window)
        if not _report(kept, clock, row, min_samples, min_clock_span) or len(kept) - ddof <= 0:
            out.append(None)
            continue
        sample = [values[index] for index in kept]
        mean = sum(sample) / len(sample)
        out.append(sum((value - mean) ** 2 for value in sample) / (len(sample) - ddof))
    return out


def _joint_kept(x, y, clock, row, window):
    return [
        earlier
        for earlier in _window(clock, row, window)
        if x[earlier] is not None and y[earlier] is not None
    ]


def rolling_covariance(x, y, clock, window, min_samples=1, min_clock_span=None, ddof=1):
    out: list[float | None] = []
    for row in range(len(x)):
        kept = _joint_kept(x, y, clock, row, window)
        if not _report(kept, clock, row, min_samples, min_clock_span) or len(kept) - ddof <= 0:
            out.append(None)
            continue
        first = [x[index] for index in kept]
        second = [y[index] for index in kept]
        mean_x = sum(first) / len(first)
        mean_y = sum(second) / len(second)
        cross = sum((a - mean_x) * (b - mean_y) for a, b in zip(first, second, strict=True))
        out.append(cross / (len(kept) - ddof))
    return out


def rolling_correlation(x, y, clock, window, min_samples=1, min_clock_span=None):
    out: list[float | None] = []
    for row in range(len(x)):
        kept = _joint_kept(x, y, clock, row, window)
        if not _report(kept, clock, row, min_samples, min_clock_span):
            out.append(None)
            continue
        first = [x[index] for index in kept]
        second = [y[index] for index in kept]
        mean_x = sum(first) / len(first)
        mean_y = sum(second) / len(second)
        cross = sum((a - mean_x) * (b - mean_y) for a, b in zip(first, second, strict=True))
        m2_x = sum((a - mean_x) ** 2 for a in first)
        m2_y = sum((b - mean_y) ** 2 for b in second)
        if m2_x <= 0 or m2_y <= 0:
            out.append(None)
            continue
        out.append(cross / math.sqrt(m2_x * m2_y))
    return out
