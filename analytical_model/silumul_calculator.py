#!/usr/bin/env python3
"""SiLU* instruction calculator with uniform CTA workload."""

from typing import List

from pipes import (
    FmaPipe,
    HardwareSpec,
    MemoryPipe,
    SiluMulFeatures,
    SiluMulProblemConfig,
    XuPipe,
)
from utils import ceil_div


def schedule_uniform_ctas(total_ctas: int, num_sms: int) -> List[int]:
    if num_sms <= 0:
        raise ValueError("Number of SMs must be positive")
    base = total_ctas // num_sms
    remainder = total_ctas % num_sms
    return [base + 1 if i < remainder else base for i in range(num_sms)]


def calculate_silumul_ops(dim: int, block_size: int, vec_size: int) -> tuple[float, float]:
    """Calculate FMA and XU ops per CTA for SiLU and Mul kernel.

    Args:
        dim: Hidden dimension (d in kernel)
        block_size: Number of threads per CTA
        vec_size: Vectorization size (8 for bf16/fp16)

    Returns:
        Tuple of (fma_ops_per_cta, xu_ops_per_cta)
    """
    FMA_OPS_PER_ELEMENT = 4 * 1.71875
    XU_OPS_PER_ELEMENT = 2 * 1.1875  

    num_warps = ceil_div(block_size, 32)

    # Main loop iterations
    total_vec_iterations = dim // vec_size
    rounds = ceil_div(total_vec_iterations, block_size)
    elements_per_thread = rounds * vec_size

    # Ops per warp in main loop
    ops_per_warp_fma = elements_per_thread * FMA_OPS_PER_ELEMENT
    ops_per_warp_xu = elements_per_thread * XU_OPS_PER_ELEMENT

    # Main loop ops per CTA (all warps)
    fma_main = ops_per_warp_fma * num_warps
    xu_main = ops_per_warp_xu * num_warps

    # Remaining elements handling
    remaining_elements = dim % (block_size * vec_size)
    if remaining_elements > 0:
        participating_threads = min(remaining_elements, block_size)
        fma_remaining = participating_threads * FMA_OPS_PER_ELEMENT
        xu_remaining = participating_threads * XU_OPS_PER_ELEMENT
    else:
        fma_remaining = 0
        xu_remaining = 0

    # Total ops per CTA
    fma_per_cta = fma_main + fma_remaining
    xu_per_cta = xu_main + xu_remaining

    return fma_per_cta, xu_per_cta


def calculate_memory_pipe(
    bytes_per_cta: float,
    total_ctas: int,
    ctas_per_sm: List[int],
    hardware: HardwareSpec,
) -> MemoryPipe:
    if hardware.mem_bandwidth <= 0 or hardware.l2_cache_bandwidth <= 0 or hardware.shared_memory_bandwidth <= 0:
        raise ValueError("Hardware bandwidth values must be positive")

    total_bytes = bytes_per_cta * total_ctas
    global_in_flight_kb = total_bytes / 1024.0
    sm_max_ctas = max(ctas_per_sm) if ctas_per_sm else 0
    sm_max_in_flight_kb = (bytes_per_cta * sm_max_ctas) / 1024.0

    global_cycle = (
        global_in_flight_kb
        / hardware.mem_bandwidth
        / (1024.0 ** 2)
        * hardware.sm_freq
        * 1e6
    )
    local_cycle = (
        global_in_flight_kb
        / hardware.l2_cache_bandwidth
        / (1024.0 ** 2)
        * hardware.sm_freq
        * 1e6
    )

    per_sm_mem_bandwidth = hardware.mem_bandwidth / hardware.num_sms
    per_sm_l2_bandwidth = hardware.l2_cache_bandwidth / hardware.num_sms

    sm_max_global_cycle = (
        sm_max_in_flight_kb
        / per_sm_mem_bandwidth
        / (1024.0 ** 2)
        * hardware.sm_freq
        * 1e6
    )
    sm_max_local_cycle = (
        sm_max_in_flight_kb
        / per_sm_l2_bandwidth
        / (1024.0 ** 2)
        * hardware.sm_freq
        * 1e6
    )
    sm_max_shared_cycle = (
        sm_max_in_flight_kb
        * 1024.0
        / hardware.shared_memory_bandwidth
    )

    return MemoryPipe(
        global_in_flight=global_in_flight_kb,
        global_cycle=global_cycle,
        local_cycle=local_cycle,
        sm_max_in_flight=sm_max_in_flight_kb,
        sm_max_global_cycle=sm_max_global_cycle,
        sm_max_shared_cycle=sm_max_shared_cycle,
        sm_max_local_cycle=sm_max_local_cycle,
    )


def silu_mul_calculator(problem: SiluMulProblemConfig, hardware: HardwareSpec) -> SiluMulFeatures:
    if hardware.tc_bf16 <= 0 or hardware.xu_fp32 <= 0 or hardware.fma_fp32 <= 0:
        raise ValueError("Hardware throughput values must be positive")
    if hardware.num_sms <= 0:
        raise ValueError("Hardware num_sms must be positive")

    dim = problem.dim
    vec_size = 8
    block_size = min(1024, dim // vec_size)
    grid = (problem.seq_len, 1, 1)
    total_ctas = problem.seq_len

    fma_per_cta, xu_per_cta = calculate_silumul_ops(dim, block_size, vec_size)
    total_fma = fma_per_cta * total_ctas
    total_xu = xu_per_cta * total_ctas

    ctas_per_sm = schedule_uniform_ctas(total_ctas, hardware.num_sms)
    sm_max_ctas = max(ctas_per_sm) if ctas_per_sm else 0
    sm_max_fma = fma_per_cta * sm_max_ctas
    sm_max_xu = xu_per_cta * sm_max_ctas

    fma_pipe = FmaPipe(
        all_ops=total_fma,
        all_cycle=total_fma / hardware.fma_fp32 / hardware.num_sms,
        sm_max_ops=sm_max_fma,
        sm_max_cycle=sm_max_fma / hardware.fma_fp32,
    )
    xu_pipe = XuPipe(
        all_ops=total_xu,
        all_cycle=total_xu / hardware.xu_fp32 / hardware.num_sms,
        sm_max_ops=sm_max_xu,
        sm_max_cycle=sm_max_xu / hardware.xu_fp32,
    )

    bytes_per_cta = dim * problem.dtype_size # read
    memory_pipe = calculate_memory_pipe(bytes_per_cta, total_ctas, ctas_per_sm, hardware)

    return SiluMulFeatures(
        fma_pipe=fma_pipe,
        xu_pipe=xu_pipe,
        memory_pipe=memory_pipe,
        hardware=hardware,
    )


