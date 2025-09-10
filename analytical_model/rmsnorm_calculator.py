#!/usr/bin/env python3


from typing import List

from pipes import (
    FmaPipe,
    HardwareSpec,
    MemoryPipe,
    RmsNormFeatures,
    RmsNormProblemConfig,
    XuPipe,
)
from utils import ceil_div, gcd


def schedule_uniform_ctas(total_ctas: int, num_sms: int) -> List[int]:
    if num_sms <= 0:
        raise ValueError("Number of SMs must be positive")
    base = total_ctas // num_sms
    remainder = total_ctas % num_sms
    return [base + 1 if i < remainder else base for i in range(num_sms)]


def calculate_rmsnorm_ops(
    dim: int,
    vec_size: int,
    block_size: int,
    num_warps: int,
    rounds: int,
    warp_size: int,
) -> tuple[float, float]:
    phase1_fadd_per_warp = rounds * vec_size
    phase1_ffma_per_warp = phase1_fadd_per_warp
    phase1_fadd_cta = phase1_fadd_per_warp * num_warps
    phase1_ffma_cta = phase1_ffma_per_warp * num_warps

    reduction_rounds = warp_size.bit_length() - 1
    warp_internal_fadd = reduction_rounds * num_warps
    cross_warp_fadd = reduction_rounds
    phase2_fadd_cta = warp_internal_fadd + cross_warp_fadd

    phase3_fadd = rounds * vec_size * num_warps
    phase3_fmul = 2 * rounds * vec_size * num_warps
    phase3_ffma = num_warps
    phase3_mufu = 2 * num_warps

    init_mufu = num_warps

    fadd_per_cta = phase1_fadd_cta + phase2_fadd_cta + phase3_fadd
    fmul_per_cta = phase3_fmul
    ffma_per_cta = phase1_ffma_cta + phase3_ffma
    mufu_per_cta = phase3_mufu + init_mufu

    remaining_elements = dim % (block_size * vec_size)
    if remaining_elements > 0:
        participating_threads = min(remaining_elements, block_size)
        fma_rem_ops = participating_threads * 3
        xu_rem_ops = participating_threads
        fadd_per_cta += ceil_div(fma_rem_ops, warp_size)
        mufu_per_cta += ceil_div(xu_rem_ops, warp_size)
    
    fma_per_cta = (fadd_per_cta + fmul_per_cta + ffma_per_cta) * 1.616838
    xu_per_cta = mufu_per_cta * 1.666667
    return fma_per_cta, xu_per_cta


def calculate_memory_pipe(
    bytes_per_cta: float,
    total_ctas: int,
    ctas_per_sm: List[int],
    hardware: HardwareSpec,
) -> MemoryPipe:
    if hardware.mem_bandwidth <= 0:
        raise ValueError("Hardware mem_bandwidth must be positive")
    if hardware.l2_cache_bandwidth <= 0:
        raise ValueError("Hardware l2_cache_bandwidth must be positive")
    if hardware.shared_memory_bandwidth <= 0:
        raise ValueError("Hardware shared_memory_bandwidth must be positive")

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


def rmsnorm_calculator(problem: RmsNormProblemConfig, hardware: HardwareSpec) -> RmsNormFeatures:
    if hardware.tc_bf16 <= 0:
        raise ValueError("Hardware tc_bf16 throughput must be positive")
    if hardware.xu_fp32 <= 0:
        raise ValueError("Hardware xu_fp32 throughput must be positive")
    if hardware.fma_fp32 <= 0:
        raise ValueError("Hardware fma_fp32 throughput must be positive")
    if hardware.num_sms <= 0:
        raise ValueError("Hardware num_sms must be positive")

    dtype_size = problem.dtype_size
    vec_size = gcd(16 // dtype_size, problem.dim)
    threads = max(1, problem.dim // vec_size)
    block_size = min(1024, threads)
    if block_size <= 0:
        block_size = 1
    num_warps = ceil_div(block_size, 32)
    num_threads = block_size
    rounds = ceil_div(problem.dim, vec_size * num_threads)

    fma_per_cta, xu_per_cta = calculate_rmsnorm_ops(
        dim=problem.dim,
        vec_size=vec_size,
        block_size=block_size,
        num_warps=num_warps,
        rounds=rounds,
        warp_size=32,
    )

    total_ctas = problem.batch_size
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

    bytes_per_cta = problem.dim * problem.dtype_size  # read
    memory_pipe = calculate_memory_pipe(bytes_per_cta, total_ctas, ctas_per_sm, hardware)

    return RmsNormFeatures(
        fma_pipe=fma_pipe,
        xu_pipe=xu_pipe,
        memory_pipe=memory_pipe,
        hardware=hardware
    )

