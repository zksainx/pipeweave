from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from .util import percentile


def absolute_percentage_errors(actual: Sequence[float], predicted: Sequence[float]) -> list[float]:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lengths differ")
    errors: list[float] = []
    for truth, estimate in zip(actual, predicted, strict=True):
        if truth == 0:
            raise ValueError("MAPE is undefined for zero ground truth")
        errors.append(abs(estimate - truth) / abs(truth) * 100.0)
    return errors


def latency_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    errors = absolute_percentage_errors(actual, predicted)
    pairs = list(zip(actual, predicted, strict=True))
    signed = [(estimate - truth) / truth * 100.0 for truth, estimate in pairs]
    absolute_sum = sum(abs(estimate - truth) for truth, estimate in pairs)
    return {
        "count": float(len(errors)),
        "mape": statistics.fmean(errors),
        "wmape": absolute_sum / sum(abs(value) for value in actual) * 100.0,
        "median_ape": statistics.median(errors),
        "p90_ape": percentile(errors, 0.90),
        "max_ape": max(errors),
        "signed_bias": statistics.fmean(signed),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0 + 1.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def ranking_metrics(
    actual_latency: Sequence[float], predicted_latency: Sequence[float], *, top_k: int = 5
) -> dict[str, Any]:
    if len(actual_latency) != len(predicted_latency):
        raise ValueError("actual and predicted lengths differ")
    if not actual_latency:
        raise ValueError("ranking metrics require at least one candidate")
    top_k = min(top_k, len(actual_latency))
    actual_order = sorted(range(len(actual_latency)), key=lambda index: actual_latency[index])
    predicted_order = sorted(
        range(len(predicted_latency)), key=lambda index: predicted_latency[index]
    )
    actual_top = set(actual_order[:top_k])
    predicted_top = set(predicted_order[:top_k])
    selected = predicted_order[0]
    best = actual_order[0]
    return {
        "candidate_count": len(actual_latency),
        "spearman": _pearson(_average_ranks(actual_latency), _average_ranks(predicted_latency)),
        "top_k": top_k,
        "top_k_recall": len(actual_top & predicted_top) / top_k,
        "selected_index": selected,
        "best_index": best,
        "regret_percent": (actual_latency[selected] / actual_latency[best] - 1.0) * 100.0,
    }
