#!/usr/bin/env python3
"""
FlashInfer BatchPrefill FA3 (Hopper Architecture) KernelTraits Parameter Calculator V2

Unified calculator for both paged and ragged KV cache storage layouts optimized for FA3/Hopper.
Supports variable-length sequences and includes operation statistics analysis.

Key Features:
- FA3/Hopper optimization: Uses getCTATileSize() logic for SM90+ architectures
- Paged and Ragged KV Cache: Supports both storage layouts
- Variable-length sequences: Handles batches with different Q/KV sequence lengths
- Operation statistics: Computes XU and MMA operation counts across SMs
- MinHeap scheduling: Simulates FA3 task distribution

Source code references:
- attention/hopper/prefill_sm90.cuh:503-521 (getCTATileSize function)
- attention/hopper/kernel_traits.cuh (AttentionKernelTraits structure)
- attention/hopper/mainloop.cuh:140-148 (get_num_kv_tiles)
- attention/scheduler.cuh:868-1017 (PrefillSM90Plan - MinHeap scheduling)
"""

import json
from pathlib import Path
from typing import List, Any, Tuple, Optional
import heapq

from pipes import FaFeatures, FaProblemConfig, HardwareSpec, MemoryPipe, TensorPipe, XuPipe
from utils import ceil_div


def calculate_memory_pipe(
    cta_tile_q: int,
    cta_tile_kv: int,
    head_dim: int,
    dtype_q_size: int,
    dtype_kv_size: int,
    sm_task_iterations: List[List[int]],
    hardware: HardwareSpec,
) -> MemoryPipe:
    """Aggregate memory pipeline metrics based on SM task distribution."""

    if hardware.num_sms <= 0:
        raise ValueError("Hardware spec must have num_sms > 0")
    if hardware.sm_freq <= 0:
        raise ValueError("Hardware spec must have sm_freq > 0")
    if hardware.mem_bandwidth <= 0:
        raise ValueError("Hardware spec must have mem_bandwidth > 0")
    if hardware.l2_cache_bandwidth <= 0:
        raise ValueError("Hardware spec must have l2_cache_bandwidth > 0")
    if hardware.shared_memory_bandwidth <= 0:
        raise ValueError("Hardware spec must have shared_memory_bandwidth > 0")

    q_tile_bytes = cta_tile_q * head_dim * dtype_q_size
    kv_tile_bytes = cta_tile_kv * head_dim * dtype_kv_size

    total_bytes = 0.0
    per_sm_bytes: List[float] = []

    for sm_tasks in sm_task_iterations:
        sm_bytes = 0.0
        for iterations in sm_tasks:
            task_bytes = q_tile_bytes + iterations * kv_tile_bytes
            sm_bytes += task_bytes
            total_bytes += task_bytes
        per_sm_bytes.append(sm_bytes)

    global_in_flight_kb = total_bytes / 1024.0
    sm_max_in_flight_kb = max(per_sm_bytes) / 1024.0 if per_sm_bytes else 0.0

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


class MinHeap:
    """MinHeap for FA3 scheduling simulation.

    Based on include/flashinfer/attention/heap.h
    Tracks (cost, sm_id) pairs to distribute tasks evenly across SMs.
    """

    def __init__(self, num_sm: int):
        """Initialize heap with num_sm elements, all with cost 0."""
        self.heap = [(0, i) for i in range(num_sm)]
        heapq.heapify(self.heap)

    def pop(self) -> Tuple[int, int]:
        """Pop the SM with minimum cost.

        Returns:
            Tuple of (sm_id, cost)
        """
        cost, sm_id = heapq.heappop(self.heap)
        return sm_id, cost

    def insert(self, sm_cost: Tuple[int, int]):
        """Insert (sm_id, cost) back into heap.

        Args:
            sm_cost: Tuple of (sm_id, new_cost)
        """
        sm_id, cost = sm_cost
        heapq.heappush(self.heap, (cost, sm_id))


def calculate_fa3_ops(cta_q: int, cta_kv: int, head_dim: int, iterations: int) -> Tuple[int, int]:
    """Calculate MMA and XU operation counts for FA3.

    Based on FA3 source code analysis.

    Args:
        cta_q: CTA Q tile size
        cta_kv: CTA KV tile size
        head_dim: Attention head dimension
        iterations: Number of KV iterations

    Returns:
        Dictionary with XU and MMA operation counts
    """
    # XU = EX2 + RCP + LG2
    total_xu = cta_q * cta_kv + (iterations - 1) * cta_q * (4 + cta_kv) + 8 * cta_q

    # MMA: 2 GEMMs per iteration, each is 2MNK FLOPs
    # Q@K^T: 2 × CTA_Q × CTA_KV × head_dim
    # P@V:   2 × CTA_Q × CTA_KV × head_dim
    total_mma = 4 * cta_q * cta_kv * head_dim * iterations

    return total_mma, total_xu


def calculate_fa3_operation_stats(
    cta_q: int,
    cta_kv: int,
    head_dim: int,
    sm_task_iterations: List[List[int]],
    hardware: HardwareSpec,
) -> Tuple[TensorPipe, XuPipe]:
    """Calculate operation statistics across all SMs and encode them as pipe data."""

    total_xu = 0
    total_mma = 0
    sm_mma_max = 0
    sm_xu_max = 0

    for sm_tasks in sm_task_iterations:
        sm_xu = 0
        sm_mma = 0
        for iterations in sm_tasks:
            mma_ops, xu_ops = calculate_fa3_ops(cta_q, cta_kv, head_dim, iterations)
            sm_xu += xu_ops
            sm_mma += mma_ops

        total_xu += sm_xu
        total_mma += sm_mma
        sm_mma_max = max(sm_mma_max, sm_mma)
        sm_xu_max = max(sm_xu_max, sm_xu)


    tensor_pipe = TensorPipe(
        all_ops=total_mma,
        all_cycle=total_mma / hardware.tc_bf16 / hardware.num_sms,
        sm_max_ops=sm_mma_max,
        sm_max_cycle=sm_mma_max / hardware.tc_bf16,
    )
    xu_pipe = XuPipe(
        all_ops=total_xu,
        all_cycle=total_xu / hardware.xu_fp32 / hardware.num_sms,
        sm_max_ops=sm_xu_max,
        sm_max_cycle=sm_xu_max / hardware.xu_fp32,
    )

    return tensor_pipe, xu_pipe


def FA3GetCTATileSize(head_dim_qk: int, head_dim_vo: int, layout: str, causal: bool = True) -> Tuple[int, int]:
    """Get CTA tile size for FA3/Hopper architecture.

    Source code references:
        - Ragged: hopper/prefill_sm90.cuh:552 (uses getCTATileSize() logic)
        - Paged: hopper/prefill_sm90.cuh:574-608 (uses hardcoded values)

    Args:
        head_dim_qk: Q/K head dimension
        head_dim_vo: V/O head dimension
        layout: KV cache layout type ("paged" or "ragged")
        causal: Whether using causal mask (only affects ragged layout)

    Returns:
        Tuple of (CTA_Q, CTA_KV)

    Note:
        - Ragged: Uses getCTATileSize() logic (prefill_sm90.cuh:503-521)
        - Paged: Uses hardcoded values optimized for paging, ignores causal parameter
    """
    if layout == "ragged":
        # Ragged layout: use getCTATileSize() logic
        if head_dim_qk == head_dim_vo:
            if head_dim_qk == 64:
                return (192, 128)
            elif head_dim_qk == 128:
                if causal:
                    return (128, 128)
                else:
                    return (128, 192)
            else:  # head_dim >= 256
                return (128, 64)
        else:
            # Special case for DeepSeek-like models with different QK/VO dimensions
            # NOTE: This is a hack for deepseek prefill (QK=192, VO=128)
            return (128, 128)
    elif layout == "paged":
        # Paged layout: use hardcoded values (prefill_sm90.cuh:574-608)
        # Note: Ignores causal parameter
        if head_dim_qk == head_dim_vo:
            if head_dim_qk == 64:
                return (192, 96)
            elif head_dim_qk == 128:
                return (128, 96)
            else:  # head_dim >= 256
                return (128, 32)
        else:
            # DeepSeek models not supported in paged layout
            raise ValueError(f"Paged layout does not support different QK/VO dimensions: {head_dim_qk} != {head_dim_vo}")
    else:
        raise ValueError(f"Invalid layout: {layout}. Must be 'paged' or 'ragged'")



def fa3_scheduler(
    batch_size: int,
    q_lengths: List[int],
    kv_lengths: List[int],
    num_sm: int,
    cta_tile_q: int,
    cta_tile_kv: int,
    causal: bool,
    num_qo_heads: int,
    same_schedule_for_all_heads: bool
) -> List[List[int]]:
    """Simulate FA3 MinHeap scheduling to get SM-level iteration distribution.

    Based on scheduler.cuh:913-946.

    Args:
        batch_size: Number of requests
        q_lengths: Query lengths for each request
        kv_lengths: KV lengths for each request
        num_sm: Number of SMs
        cta_tile_q: CTA_Q tile size
        cta_tile_kv: CTA_KV tile size
        causal: Whether using causal masking
        num_qo_heads: Number of QO heads
        same_schedule_for_all_heads: Whether all heads share same schedule

    Returns:
        List of task iterations for each SM
    """
    heap = MinHeap(num_sm)
    sm_task_iterations = [[] for _ in range(num_sm)]

    # Sort requests by KV length (descending), matching scheduler.cuh:900-901
    idx_qo_kv_len = [(i, q_lengths[i], kv_lengths[i]) for i in range(batch_size)]
    idx_qo_kv_len.sort(key=lambda x: x[2], reverse=True)

    num_heads_to_schedule = 1 if same_schedule_for_all_heads else num_qo_heads

    for qo_head_idx in range(num_heads_to_schedule):
        for idx, qo_len, kv_len in idx_qo_kv_len:
            num_q_tiles = ceil_div(qo_len, cta_tile_q)

            for q_tile_idx in range(num_q_tiles - 1, -1, -1):
                sm_id, cost = heap.pop()

                # Calculate num_kv_tiles matching mainloop.cuh:140-152
                num_kv_tiles = ceil_div(kv_len, cta_tile_kv)
                if causal:
                    num_kv_tiles = min(num_kv_tiles,
                                      ceil_div((q_tile_idx + 1) * cta_tile_q + kv_len - qo_len, cta_tile_kv))

                # Only track tasks that actually execute (num_kv_tiles > 0)
                if num_kv_tiles > 0:
                    sm_task_iterations[sm_id].append(num_kv_tiles)

                # Calculate task cost
                if causal:
                    effective_kv_len = max(min((q_tile_idx + 1) * cta_tile_q + kv_len - qo_len, kv_len), 0)
                    task_cost = cta_tile_q * effective_kv_len
                else:
                    task_cost = cta_tile_q * kv_len

                heap.insert((sm_id, cost + task_cost))

    return sm_task_iterations


def calculate_fa3_params(problem: FaProblemConfig, hardware: HardwareSpec) -> FaFeatures:
    """Calculate FA3 KernelTraits parameters for variable-length sequences."""

    batch_size = problem.batch_size
    q_lengths = problem.q_lengths
    kv_lengths = problem.kv_lengths
    num_qo_heads = problem.num_qo_heads
    num_kv_heads = problem.num_kv_heads
    head_dim = problem.head_dim
    layout = problem.layout
    causal = problem.causal
    num_sm = hardware.num_sms

    if len(q_lengths) != batch_size:
        raise ValueError(
            f"Number of q_lengths ({len(q_lengths)}) must match batch_size ({batch_size})"
        )
    if len(kv_lengths) != batch_size:
        raise ValueError(
            f"Number of kv_lengths ({len(kv_lengths)}) must match batch_size ({batch_size})"
        )
    if hardware.tc_bf16 <= 0:
        raise ValueError("Hardware spec must have positive tc_bf16 throughput")
    if hardware.xu_fp32 <= 0:
        raise ValueError("Hardware spec must have positive xu_fp32 throughput")

    head_dim_qk = head_dim
    head_dim_vo = head_dim
    cta_q, cta_kv = FA3GetCTATileSize(head_dim_qk, head_dim_vo, layout, causal)

    num_warps = ((cta_q // 64) + 1) * 4
    num_threads = num_warps * 32
    dtype_q_size = problem.data_size_q
    dtype_kv_size = problem.data_size_kv

    total_num_rows = sum(q_lengths)
    max_num_works_per_head = ceil_div(total_num_rows, cta_q) + batch_size - 1
    same_schedule_for_all_heads = max_num_works_per_head > 8192

    if same_schedule_for_all_heads:
        scheduler_type = "BatchPrefillTileScheduler"
    else:
        scheduler_type = "BatchPrefillPersistentTileScheduler"

    if scheduler_type == "BatchPrefillPersistentTileScheduler":
        grid_dim = (num_sm, 1, 1)
    else:
        grid_dim = (num_sm, num_kv_heads, 1)
    block_dim = (num_threads, 1, 1)

    sm_iteration_distribution = fa3_scheduler(
        batch_size=batch_size,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        num_sm=num_sm,
        cta_tile_q=cta_q,
        cta_tile_kv=cta_kv,
        causal=causal,
        num_qo_heads=num_qo_heads,
        same_schedule_for_all_heads=same_schedule_for_all_heads
    )

    tensor_pipe, xu_pipe = calculate_fa3_operation_stats(
        cta_q=cta_q,
        cta_kv=cta_kv,
        head_dim=head_dim_qk,
        sm_task_iterations=sm_iteration_distribution,
        hardware=hardware,
    )

    memory_pipe = calculate_memory_pipe(
        cta_tile_q=cta_q,
        cta_tile_kv=cta_kv,
        head_dim=head_dim_qk,
        dtype_q_size=dtype_q_size,
        dtype_kv_size=dtype_kv_size,
        sm_task_iterations=sm_iteration_distribution,
        hardware=hardware,
    )

    return FaFeatures(
        tensor_pipe=tensor_pipe,
        xu_pipe=xu_pipe,
        memory_pipe=memory_pipe,
        hardware=hardware,
    )

