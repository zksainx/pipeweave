from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from typing import Any

from .util import percentile


def summarize_samples(samples: Sequence[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("at least one timing sample is required")
    values = [float(value) for value in samples]
    median = statistics.median(values)
    absolute_deviations = [abs(value - median) for value in values]
    mean = statistics.fmean(values)
    standard_deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": float(len(values)),
        "mean": mean,
        "median": median,
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "mad": statistics.median(absolute_deviations),
        "cv": standard_deviation / mean if mean else 0.0,
        "min": min(values),
        "max": max(values),
    }


def benchmark_cuda_event(
    function: Callable[[], None],
    *,
    warmup: int,
    samples: int,
    target_sample_us: float,
) -> dict[str, Any]:
    import torch

    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    rough_us = max(float(start.elapsed_time(end)) * 1000.0, 0.1)
    repetitions = max(1, min(10_000, math.ceil(target_sample_us / rough_us)))

    raw_samples: list[float] = []
    for _ in range(samples):
        start.record()
        for _ in range(repetitions):
            function()
        end.record()
        end.synchronize()
        raw_samples.append(float(start.elapsed_time(end)) * 1000.0 / repetitions)

    return {
        "mode": "eager_event",
        "unit": "us",
        "repetitions_per_sample": repetitions,
        "raw_samples": raw_samples,
        "summary": summarize_samples(raw_samples),
    }


def benchmark_cuda_graph(
    function: Callable[[], None],
    *,
    warmup: int,
    samples: int,
    target_sample_us: float,
) -> dict[str, Any]:
    import torch

    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()

    result = benchmark_cuda_event(
        graph.replay,
        warmup=max(2, warmup // 2),
        samples=samples,
        target_sample_us=target_sample_us,
    )
    result["mode"] = "cuda_graph_event"
    return result


def benchmark_cuda_graph_batched(
    function: Callable[[], None],
    *,
    warmup: int,
    samples: int,
    target_sample_us: float,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    if batch_size < 2:
        raise ValueError("CUDA Graph batch_size must be at least two")
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(batch_size):
            function()

    result = benchmark_cuda_event(
        graph.replay,
        warmup=max(2, warmup // 2),
        samples=samples,
        target_sample_us=target_sample_us,
    )
    raw_samples = [value / batch_size for value in result["raw_samples"]]
    result.update(
        {
            "mode": "cuda_graph_batched_event",
            "raw_samples": raw_samples,
            "summary": summarize_samples(raw_samples),
            "kernel_nodes_per_graph": batch_size,
            "label_semantics": "steady-state same-stream throughput per kernel node",
        }
    )
    return result


def compare_tensors(actual: Any, expected: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    import torch

    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    denominator = expected_float.abs().clamp_min(1e-8)
    max_absolute = float(difference.max().item())
    max_relative = float((difference / denominator).max().item())
    passed = bool(torch.allclose(actual_float, expected_float, atol=atol, rtol=rtol))
    return {
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": max_absolute,
        "max_rel_error": max_relative,
    }
