#!/usr/bin/env python3
"""
FlashInfer CUTLASS Blackwell (SM100) KernelTraits Parameter Calculator

Analytical model for CUTLASS-based Flash Attention on Blackwell architecture.
Supports variable-length sequences with cost-based bucket scheduling.

Key Features:
- Blackwell/SM100 optimization: CUTLASS-based FMHA implementation
- Cost-based bucket scheduling: Balances work across SMs using min-cost allocation
- Variable-length sequences: Handles batches with different Q/KV sequence lengths
- TMEM support: Tensor Memory operations merged into unified calculation
- Operation statistics: Computes XU and MMA operation counts across SMs

Source code references:
- attention/blackwell/plan.cuh:65-145 (plan_kernel - cost-based bucket scheduling)
- attention/blackwell/fmha_cutlass_sm100.cuh (FwdRunner structure)
- attention/blackwell/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp (TmemAllocation)
"""

import heapq
from typing import List, Tuple

from pipes import FaFeatures, FaProblemConfig, HardwareSpec, MemoryPipe, TensorPipe, XuPipe
from utils import ceil_div


def FACutlassDetermineCTATileSize(head_dim: int, causal: bool = True) -> Tuple[int, int]:
    """Determine CTA tile sizes for CUTLASS Blackwell implementation.

    Based on typical CUTLASS tile configurations and Blackwell architecture.

    Args:
        head_dim: Attention head dimension
        causal: Whether using causal mask

    Returns:
        Tuple of (CTA_Q, CTA_KV) tile sizes

    Note:
        These are conservative estimates based on typical CUTLASS configurations.
        Actual tile sizes may vary based on specific kernel instantiation.
    """
    if head_dim == 64:
        # Smaller head dimension allows larger tiles
        return (128, 128)
    elif head_dim == 128:
        # Most common configuration
        if causal:
            return (128, 64)
        else:
            return (128, 96)
    elif head_dim == 192:
        # DeepSeek-style models
        return (128, 64)
    else:  # head_dim >= 256
        # Larger head dimensions require smaller KV tiles
        return (128, 32)


class CostBasedBucketScheduler:
    """Cost-based bucket scheduler for Blackwell CUTLASS implementation.

    Based on flashinfer/attention/blackwell/plan.cuh:65-145.

    This scheduler distributes work across buckets (SMs) by:
    1. Tracking cost for each bucket
    2. Assigning each task to the bucket with minimum cost
    3. Updating bucket cost after assignment

    Args:
        num_buckets: Number of buckets (typically equals num_sms)
    """

    def __init__(self, num_buckets: int):
        self.num_buckets = num_buckets
        # Min-heap of (cost, bucket_id)
        self.heap = [(0.0, i) for i in range(num_buckets)]
        heapq.heapify(self.heap)

    def get_min_cost_bucket(self) -> Tuple[int, float]:
        """Get the bucket with minimum cost.

        Returns:
            Tuple of (bucket_id, current_cost)
        """
        cost, bucket_id = heapq.heappop(self.heap)
        return bucket_id, cost

    def update_bucket_cost(self, bucket_id: int, new_cost: float):
        """Update bucket cost after task assignment.

        Args:
            bucket_id: Bucket to update
            new_cost: New total cost for this bucket
        """
        heapq.heappush(self.heap, (new_cost, bucket_id))


def fa_cutlass_scheduler(
    batch_size: int,
    q_lengths: List[int],
    kv_lengths: List[int],
    num_qo_heads: int,
    num_kv_heads: int,
    num_buckets: int,
    cta_tile_q: int,
    causal: bool,
) -> List[List[int]]:
    """Simulate CUTLASS Blackwell cost-based bucket scheduling.

    Based on plan.cuh:65-145 logic.

    Args:
        batch_size: Number of requests
        q_lengths: Query lengths for each request
        kv_lengths: KV lengths for each request
        num_qo_heads: Number of query/output heads
        num_kv_heads: Number of KV heads
        num_buckets: Number of buckets (SMs)
        cta_tile_q: CTA_Q tile size
        causal: Whether using causal masking

    Returns:
        List of task iterations for each bucket (SM)
    """
    scheduler = CostBasedBucketScheduler(num_buckets)
    bucket_task_iterations = [[] for _ in range(num_buckets)]

    # Iterate through all heads, batches, and Q tiles
    # Following plan.cuh:80-98 iteration order
    for head_idx in range(num_qo_heads):
        for batch_idx in range(batch_size):
            qo_len = q_lengths[batch_idx]
            kv_len = kv_lengths[batch_idx]
            num_qo_tiles = ceil_div(qo_len, cta_tile_q)

            # Iterate Q tiles in reverse order (following plan.cuh:87)
            for qo_tile_idx in range(num_qo_tiles - 1, -1, -1):
                # Get bucket with minimum cost
                bucket_id, cost = scheduler.get_min_cost_bucket()

                # Calculate effective KV length for this Q tile
                if causal:
                    # From plan.cuh:93-94
                    effective_kv_len = kv_len - (num_qo_tiles - qo_tile_idx - 1) * cta_tile_q
                    effective_kv_len = max(0, effective_kv_len)
                else:
                    effective_kv_len = kv_len

                # Store the effective KV length as "iterations" for this task
                # Note: This is KV length, not tile count - we'll convert later
                if effective_kv_len > 0:
                    bucket_task_iterations[bucket_id].append(effective_kv_len)

                # Update bucket cost
                new_cost = cost + cta_tile_q * effective_kv_len
                scheduler.update_bucket_cost(bucket_id, new_cost)

    return bucket_task_iterations


def calculate_memory_pipe(
    cta_tile_q: int,
    cta_tile_kv: int,
    head_dim: int,
    dtype_q_size: int,
    dtype_kv_size: int,
    sm_task_kv_lens: List[List[int]],
    hardware: HardwareSpec,
) -> MemoryPipe:
    """Calculate memory pipeline metrics based on SM task distribution.

    Args:
        cta_tile_q: CTA Q tile size
        cta_tile_kv: CTA KV tile size
        head_dim: Attention head dimension
        dtype_q_size: Q data type size in bytes
        dtype_kv_size: KV data type size in bytes
        sm_task_kv_lens: List of KV lengths for tasks on each SM
        hardware: Hardware specification

    Returns:
        MemoryPipe with aggregated metrics
    """
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

    for sm_tasks in sm_task_kv_lens:
        sm_bytes = 0.0
        for kv_len in sm_tasks:
            # Each task loads Q tile once and KV tiles for the effective KV length
            num_kv_tiles = ceil_div(kv_len, cta_tile_kv)
            task_bytes = q_tile_bytes + num_kv_tiles * kv_tile_bytes
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


def calculate_fa_cutlass_ops(
    cta_q: int,
    cta_kv: int,
    head_dim: int,
    kv_len: int
) -> Tuple[int, int]:
    """Calculate MMA and XU operation counts for CUTLASS Blackwell.

    Similar to FA3 operation counting but adapted for CUTLASS implementation.
    TMEM operations are merged into the unified XU calculation as requested.

    Args:
        cta_q: CTA Q tile size
        cta_kv: CTA KV tile size
        head_dim: Attention head dimension
        kv_len: Effective KV length for this task

    Returns:
        Tuple of (total_mma, total_xu)
    """
    num_kv_tiles = ceil_div(kv_len, cta_kv)

    # XU operations: Similar to FA3 pattern
    # EX2 (exponential), RCP (reciprocal), LG2 (log2)
    # These include operations from softmax, normalization, and TMEM operations
    # TMEM operations merged as requested
    total_xu = (
        cta_q * cta_kv  # Initial softmax exp operations
        + (num_kv_tiles - 1) * cta_q * (4 + cta_kv)  # Accumulation and correction
        + 8 * cta_q  # Final normalization and log-sum-exp
    )

    # MMA operations: 2 GEMMs per KV tile
    # Q@K^T: 2 × CTA_Q × CTA_KV × head_dim FLOPs
    # P@V:   2 × CTA_Q × CTA_KV × head_dim FLOPs
    total_mma = 4 * cta_q * cta_kv * head_dim * num_kv_tiles

    return total_mma, total_xu


def calculate_fa_cutlass_operation_stats(
    cta_q: int,
    cta_kv: int,
    head_dim: int,
    sm_task_kv_lens: List[List[int]],
    hardware: HardwareSpec,
) -> Tuple[TensorPipe, XuPipe]:
    """Calculate operation statistics across all SMs.

    Args:
        cta_q: CTA Q tile size
        cta_kv: CTA KV tile size
        head_dim: Attention head dimension
        sm_task_kv_lens: List of KV lengths for tasks on each SM
        hardware: Hardware specification

    Returns:
        Tuple of (TensorPipe, XuPipe) with aggregated statistics
    """
    total_xu = 0
    total_mma = 0
    sm_mma_max = 0
    sm_xu_max = 0

    for sm_tasks in sm_task_kv_lens:
        sm_xu = 0
        sm_mma = 0
        for kv_len in sm_tasks:
            mma_ops, xu_ops = calculate_fa_cutlass_ops(cta_q, cta_kv, head_dim, kv_len)
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


def calculate_fa_cutlass_params(
    problem: FaProblemConfig,
    hardware: HardwareSpec
) -> FaFeatures:
    """Calculate CUTLASS Blackwell KernelTraits parameters for variable-length sequences.

    Main entry point for the FA_CUTLASS calculator.

    Args:
        problem: Problem configuration (batch size, sequence lengths, heads, etc.)
        hardware: Hardware specification (SMs, frequencies, bandwidths, etc.)

    Returns:
        FaFeatures with tensor, XU, and memory pipeline metrics

    Raises:
        ValueError: If input parameters are invalid
    """
    batch_size = problem.batch_size
    q_lengths = problem.q_lengths
    kv_lengths = problem.kv_lengths
    num_qo_heads = problem.num_qo_heads
    num_kv_heads = problem.num_kv_heads
    head_dim = problem.head_dim
    causal = problem.causal
    num_sms = hardware.num_sms

    # Validation
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

    # Determine CTA tile sizes
    cta_q, cta_kv = FACutlassDetermineCTATileSize(head_dim, causal)

    # Get data type sizes
    dtype_q_size = problem.data_size_q
    dtype_kv_size = problem.data_size_kv

    # Run cost-based bucket scheduler
    # Note: In CUTLASS Blackwell, num_buckets typically equals num_sms
    num_buckets = num_sms
    sm_task_kv_lens = fa_cutlass_scheduler(
        batch_size=batch_size,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        num_buckets=num_buckets,
        cta_tile_q=cta_q,
        causal=causal,
    )

    # Calculate operation statistics
    tensor_pipe, xu_pipe = calculate_fa_cutlass_operation_stats(
        cta_q=cta_q,
        cta_kv=cta_kv,
        head_dim=head_dim,
        sm_task_kv_lens=sm_task_kv_lens,
        hardware=hardware,
    )

    # Calculate memory pipeline metrics
    memory_pipe = calculate_memory_pipe(
        cta_tile_q=cta_q,
        cta_tile_kv=cta_kv,
        head_dim=head_dim,
        dtype_q_size=dtype_q_size,
        dtype_kv_size=dtype_kv_size,
        sm_task_kv_lens=sm_task_kv_lens,
        hardware=hardware,
    )

    return FaFeatures(
        tensor_pipe=tensor_pipe,
        xu_pipe=xu_pipe,
        memory_pipe=memory_pipe,
        hardware=hardware,
    )
