#!/usr/bin/env python3
"""GEMM feature calculator for FP8 tensor cores."""

from pipes import (
    GemmFeatures,
    GemmProblemConfig,
    HardwareSpec,
    MemoryPipe,
    TensorPipe,
)
from utils import ceil_div


def gemm_fp8_calculator(problem: GemmProblemConfig, hardware: HardwareSpec) -> GemmFeatures:
    """Compute the FP8 GEMM feature vector.

    Args:
        problem: Problem configuration describing matrix shapes, tiling, and launch grid.
        hardware: Hardware capabilities that parameterize the analytical model.

    Returns:
        Aggregated GEMM features estimated for the provided problem and hardware.
    """

    if hardware.num_sms <= 0:
        raise ValueError("Hardware spec must have num_sms > 0")

    m_tiles = ceil_div(problem.m, problem.tile_m)
    n_tiles = ceil_div(problem.n, problem.tile_n)
    k_tiles = ceil_div(problem.k, problem.tile_k)

    m_padded = m_tiles * problem.tile_m
    n_padded = n_tiles * problem.tile_n
    k_padded = k_tiles * problem.tile_k

    tile_count = m_tiles * n_tiles
    if tile_count <= 0:
        raise ValueError("Tile count must be positive")

    tile_flops = 2.0 * problem.tile_m * problem.tile_n * k_padded
    flops = 2.0 * m_padded * n_padded * k_padded

    num_waves = ceil_div(tile_count, hardware.num_sms)
    sm_max_flops = tile_flops * num_waves

    overall_cycle = flops / hardware.tc_fp8 / hardware.num_sms
    sm_max_cycle = sm_max_flops / hardware.tc_fp8

    global_in_flight = (
        problem.data_size_bytes
        * (m_padded * k_padded + n_padded * k_padded)
        / 1024.0
    )
    global_cycle = (
        global_in_flight
        / hardware.mem_bandwidth
        / (1024.0**2)
        * hardware.sm_freq
        * 1e6
    )
    local_cycle = (
        global_in_flight
        / hardware.l2_cache_bandwidth
        / (1024.0**2)
        * hardware.sm_freq
        * 1e6
    )

    sm_max_in_flight = (
        problem.data_size_bytes
        * (problem.tile_m * k_padded + problem.tile_n * k_padded)
        / 1024.0
        * num_waves
    )

    sm_max_global_cycle = (
        sm_max_in_flight
        / (hardware.mem_bandwidth / hardware.num_sms)
        / (1024.0**2)
        * hardware.sm_freq
        * 1e6
    )
    sm_max_local_cycle = (
        sm_max_in_flight
        / (hardware.l2_cache_bandwidth / hardware.num_sms)
        / (1024.0**2)
        * hardware.sm_freq
        * 1e6
    )
    sm_max_shared_cycle = (
        sm_max_in_flight
        * 1024.0
        / hardware.shared_memory_bandwidth
    )

    tensor_pipe = TensorPipe(
        all_ops=flops,
        all_cycle=overall_cycle,
        sm_max_ops=sm_max_flops,
        sm_max_cycle=sm_max_cycle,
    )

    memory_pipe = MemoryPipe(
        global_in_flight=global_in_flight,
        global_cycle=global_cycle,
        local_cycle=local_cycle,
        sm_max_in_flight=sm_max_in_flight,
        sm_max_global_cycle=sm_max_global_cycle,
        sm_max_shared_cycle=sm_max_shared_cycle,
        sm_max_local_cycle=sm_max_local_cycle,
    )

    return GemmFeatures(
        tensor_pipe=tensor_pipe,
        memory_pipe=memory_pipe,
        hardware=hardware,
    )
