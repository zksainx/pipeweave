#!/usr/bin/env python3
"""
Flash Attention Composable Kernel (CK) Calculator

Analytical model for AMD Composable Kernel Flash Attention implementation.
Focuses on ROCm/HIP FMHA forward pass with causal masking optimization.

Key Features:
- CK tile configuration: Fixed kM0=64, dynamic kN0 based on head_dim
- Causal mask optimization: Skips upper triangle computation (~50% reduction)
- MHA forward: Batch mode with uniform sequence lengths
- Round-robin scheduling: Simple SM-level task distribution

Source code references:
- composable_kernel/include/ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp:48-52 (TileFmhaShape)
- composable_kernel/include/ck_tile/ops/fmha/block/block_masking.hpp:110-142 (GetTileRangeAlongX)
- flash-attention/csrc/flash_attn_ck/mha_fwd.cpp:233 (kM0 configuration)
- flash-attention/csrc/composable_kernel/example/ck_tile/01_fmha/fmha_fwd.cpp (example usage)
"""

from typing import List, Tuple

from pipes import FaFeatures, FaProblemConfig, HardwareSpec, MemoryPipe, TensorPipe, XuPipe
from utils import ceil_div


def FACKDetermineTileSize(head_dim: int, causal: bool = True) -> Tuple[int, int]:
    """Determine CK tile sizes (kM0, kN0) based on head dimension.

    Based on actual Composable Kernel tile configurations observed from MI308X profiling:
    - tile_fmha_shape.hpp: kM0, kN0, kK0 definitions
    - Verified against MI308X flash_attn JSON profiling data

    CK uses kM0=128 for Q sequence dimension (verified from actual kernels).
    The kN0 (K sequence dimension) is adjusted based on head_dim.

    Args:
        head_dim: Attention head dimension (64, 128, 192, 256, etc.)
        causal: Whether using causal mask (currently doesn't affect tile selection
                in observed CK kernels)

    Returns:
        Tuple of (kM0, kN0) tile sizes

    Note:
        - kM0 is consistently 128 across observed CK implementations (not 64!)
        - kN0 configuration verified from MI308X profiling:
          * head_dim=64:  kN0=64
          * head_dim=128: kN0=128
          * Larger head_dims likely use smaller kN0 for register pressure
    """
    kM0 = 128  # Fixed Q tile size in CK (verified from actual kernels)

    if head_dim <= 64:
        # Verified from MI308X: head_dim=64 uses kN0=64
        kN0 = 64
    elif head_dim == 128:
        # Verified from MI308X: head_dim=128 uses kN0=128
        kN0 = 128
    elif head_dim <= 192:
        # DeepSeek-style models with larger heads
        # Estimated: likely uses smaller kN0 due to register pressure
        kN0 = 64
    else:  # head_dim >= 256
        # Very large head dimensions need smaller K tiles to fit in registers
        kN0 = 32

    return kM0, kN0


def get_causal_tile_range(
    q_tile_idx: int,
    kM0: int,
    kN0: int,
    seqlen_q: int,
    seqlen_k: int,
    causal: bool
) -> Tuple[int, int]:
    """Calculate K tile range for a given Q tile with causal masking.

    Based on CK's GenericAttentionMask::GetTileRangeAlongX from:
    block_masking.hpp:110-142

    For causal attention, each Q tile only needs to compute attention scores
    for K tiles up to the diagonal. This function calculates the valid K tile
    range [k_tile_start, k_tile_end) for a given Q tile.

    Causal masking pattern (1=compute, *=skip):
        K tiles →
    Q   1 * * * *
    ↓   1 1 * * *
        1 1 1 * *
        1 1 1 1 *
        1 1 1 1 1

    Args:
        q_tile_idx: Index of current Q tile (0-indexed)
        kM0: Q tile size (kM0)
        kN0: K tile size (kN0)
        seqlen_q: Total Q sequence length
        seqlen_k: Total K sequence length
        causal: Whether to apply causal masking

    Returns:
        Tuple of (k_tile_start, k_tile_end) indices

    Example:
        For seqlen_q=256, seqlen_k=256, kM0=64, kN0=64, causal=True:
        - Q tile 0 [0:64]:   computes K tiles 0-1 [0:64]
        - Q tile 1 [64:128]: computes K tiles 0-2 [0:128]
        - Q tile 2 [128:192]: computes K tiles 0-3 [0:192]
        - Q tile 3 [192:256]: computes K tiles 0-4 [0:256]
    """
    num_k_tiles = ceil_div(seqlen_k, kN0)

    if not causal:
        # No masking: process all K tiles
        return 0, num_k_tiles

    # Causal masking: only process K tiles up to diagonal
    # Based on GetTileRangeAlongX logic:
    # x_end = min(i_y + YTile - 1 + x, x_total)
    # where x=1 (top-left causal), y=seqlen_q, YTile=kM0

    # Calculate the last query position in this tile
    q_end_pos = min((q_tile_idx + 1) * kM0, seqlen_q)

    # For causal attention, we can attend to all keys up to q_end_pos
    # Round up to tile boundary
    k_tile_end = ceil_div(q_end_pos, kN0)
    k_tile_end = min(k_tile_end, num_k_tiles)

    k_tile_start = 0

    return k_tile_start, k_tile_end


def fa_ck_scheduler(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_heads: int,
    num_sms: int,
    kM0: int,
    kN0: int,
    causal: bool
) -> List[List[int]]:
    """Simple round-robin scheduler for CK FMHA.

    Distributes FMHA tasks across SMs using round-robin allocation.
    Each task represents one (batch, head, q_tile) combination.

    Args:
        batch_size: Number of sequences in batch
        seqlen_q: Query sequence length (uniform across batch)
        seqlen_k: Key/Value sequence length (uniform across batch)
        num_heads: Number of attention heads
        num_sms: Number of streaming multiprocessors
        kM0: Q tile size
        kN0: K tile size
        causal: Whether using causal masking

    Returns:
        List of iteration counts for each SM, where each element is a list
        of K tile iterations for tasks assigned to that SM.

    Example:
        For batch=2, heads=4, seqlen_q=256, seqlen_k=256, kM0=64, kN0=64:
        - Total Q tiles per sequence: 256/64 = 4
        - Total tasks: 2 * 4 * 4 = 32 tasks
        - Each task processes some number of K tiles (depends on causal)
    """
    num_q_tiles = ceil_div(seqlen_q, kM0)

    # Initialize SM task lists
    sm_task_iterations = [[] for _ in range(num_sms)]
    sm_index = 0

    # Iterate over all tasks: batch × heads × q_tiles
    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            for q_tile_idx in range(num_q_tiles):
                # Calculate K tile range for this Q tile
                k_tile_start, k_tile_end = get_causal_tile_range(
                    q_tile_idx=q_tile_idx,
                    kM0=kM0,
                    kN0=kN0,
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    causal=causal
                )

                num_k_tiles = k_tile_end - k_tile_start

                # Only schedule tasks with actual work
                if num_k_tiles > 0:
                    # Assign to current SM in round-robin fashion
                    sm_task_iterations[sm_index].append(num_k_tiles)
                    sm_index = (sm_index + 1) % num_sms

    return sm_task_iterations


def calculate_ck_ops(kM0: int, kN0: int, head_dim: int, num_k_tiles: int) -> Tuple[int, int]:
    """Calculate MMA and XU operation counts for CK FMHA.

    CK FMHA performs two main GEMM operations per K tile:
    1. S = Q @ K^T: Compute attention scores
    2. O = P @ V:   Compute attention output

    Plus special function operations (exp2, log2, rcp) for softmax.

    Args:
        kM0: Q tile size
        kN0: K tile size
        head_dim: Attention head dimension
        num_k_tiles: Number of K tiles to process

    Returns:
        Tuple of (mma_ops, xu_ops)

    Operation breakdown:
    - GEMM Q@K^T: kM0 × kN0 × head_dim × 2 FLOPs per K tile
    - GEMM P@V:   kM0 × kN0 × head_dim × 2 FLOPs per K tile
    - Softmax:    exp2, max, div operations
    """
    # Each K tile iteration performs two GEMMs:
    # 1. S = Q[kM0, head_dim] @ K^T[head_dim, kN0] -> S[kM0, kN0]
    #    FLOPs = kM0 * kN0 * head_dim * 2 (multiply-add)
    # 2. O = P[kM0, kN0] @ V[kN0, head_dim] -> O[kM0, head_dim]
    #    FLOPs = kM0 * kN0 * head_dim * 2 (multiply-add)

    gemm_qk_flops_per_tile = kM0 * kN0 * head_dim * 2
    gemm_pv_flops_per_tile = kM0 * kN0 * head_dim * 2
    total_gemm_flops = (gemm_qk_flops_per_tile + gemm_pv_flops_per_tile) * num_k_tiles

    # XU operations for softmax:
    # - exp2 for attention scores: kM0 * kN0 per K tile
    # - max reduction: kM0 per K tile (finding max per row)
    # - sum reduction: kM0 per K tile (normalizing)
    # - final scale: kM0 * head_dim (scaling output)

    exp2_ops_per_tile = kM0 * kN0  # exp2 for each attention score
    max_ops_per_tile = kM0  # max reduction per Q row
    sum_ops_per_tile = kM0  # sum for normalization

    xu_ops_per_tile = exp2_ops_per_tile + max_ops_per_tile + sum_ops_per_tile
    total_xu_ops = xu_ops_per_tile * num_k_tiles

    # Add final output scaling
    total_xu_ops += kM0 * head_dim  # scale output by softmax denominator

    return total_gemm_flops, total_xu_ops


def calculate_memory_pipe(
    kM0: int,
    kN0: int,
    head_dim: int,
    dtype_q_size: int,
    dtype_kv_size: int,
    sm_task_iterations: List[List[int]],
    hardware: HardwareSpec,
) -> MemoryPipe:
    """Calculate memory pipeline metrics for CK FMHA.

    Args:
        kM0: Q tile size
        kN0: K tile size
        head_dim: Attention head dimension
        dtype_q_size: Q data type size in bytes
        dtype_kv_size: K/V data type size in bytes
        sm_task_iterations: Task iterations per SM
        hardware: Hardware specifications

    Returns:
        MemoryPipe with memory access statistics
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

    # Memory footprint per task:
    # - Q tile: kM0 × head_dim × dtype_q_size (loaded once per task)
    # - K tile: kN0 × head_dim × dtype_kv_size (loaded per K iteration)
    # - V tile: kN0 × head_dim × dtype_kv_size (loaded per K iteration)
    q_tile_bytes = kM0 * head_dim * dtype_q_size
    k_tile_bytes = kN0 * head_dim * dtype_kv_size
    v_tile_bytes = kN0 * head_dim * dtype_kv_size
    kv_tile_bytes = k_tile_bytes + v_tile_bytes

    total_bytes = 0.0
    per_sm_bytes: List[float] = []

    for sm_tasks in sm_task_iterations:
        sm_bytes = 0.0
        for num_k_tiles in sm_tasks:
            # Each task loads Q once, then K/V for each K tile iteration
            task_bytes = q_tile_bytes + num_k_tiles * kv_tile_bytes
            sm_bytes += task_bytes
            total_bytes += task_bytes
        per_sm_bytes.append(sm_bytes)

    global_in_flight_kb = total_bytes / 1024.0
    sm_max_in_flight_kb = max(per_sm_bytes) / 1024.0 if per_sm_bytes else 0.0

    # Calculate cycles for different memory hierarchies
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


def calculate_ck_operation_stats(
    kM0: int,
    kN0: int,
    head_dim: int,
    sm_task_iterations: List[List[int]],
    hardware: HardwareSpec,
) -> Tuple[TensorPipe, XuPipe]:
    """Calculate operation statistics across all SMs.

    Args:
        kM0: Q tile size
        kN0: K tile size
        head_dim: Attention head dimension
        sm_task_iterations: Task iterations per SM
        hardware: Hardware specifications

    Returns:
        Tuple of (TensorPipe, XuPipe) with operation statistics
    """
    total_xu = 0
    total_mma = 0
    sm_mma_max = 0
    sm_xu_max = 0

    for sm_tasks in sm_task_iterations:
        sm_xu = 0
        sm_mma = 0
        for num_k_tiles in sm_tasks:
            mma_ops, xu_ops = calculate_ck_ops(kM0, kN0, head_dim, num_k_tiles)
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


def calculate_fa_ck_params(problem: FaProblemConfig, hardware: HardwareSpec) -> FaFeatures:
    """Calculate CK FMHA parameters and performance metrics.

    Main entry point for CK Flash Attention analytical model.
    Assumes MHA forward with uniform sequence lengths within batch.

    Args:
        problem: Flash Attention problem configuration
        hardware: Hardware specifications

    Returns:
        FaFeatures with performance metrics

    Raises:
        ValueError: If configuration is invalid (e.g., non-uniform sequence lengths)
    """
    batch_size = problem.batch_size
    q_lengths = problem.q_lengths
    kv_lengths = problem.kv_lengths
    num_heads = problem.num_qo_heads
    head_dim = problem.head_dim
    causal = problem.causal
    num_sms = hardware.num_sms

    # Validate uniform sequence lengths (MHA requirement)
    if len(set(q_lengths)) > 1:
        raise ValueError(
            f"CK calculator requires uniform Q lengths within batch, "
            f"got {len(set(q_lengths))} unique lengths"
        )
    if len(set(kv_lengths)) > 1:
        raise ValueError(
            f"CK calculator requires uniform KV lengths within batch, "
            f"got {len(set(kv_lengths))} unique lengths"
        )

    seqlen_q = q_lengths[0]
    seqlen_k = kv_lengths[0]

    # Validate hardware specs
    if hardware.tc_bf16 <= 0:
        raise ValueError("Hardware spec must have positive tc_bf16 throughput")
    if hardware.xu_fp32 <= 0:
        raise ValueError("Hardware spec must have positive xu_fp32 throughput")

    # Determine tile sizes based on head dimension and causal masking
    kM0, kN0 = FACKDetermineTileSize(head_dim, causal)

    # Schedule tasks across SMs
    sm_task_iterations = fa_ck_scheduler(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_heads=num_heads,
        num_sms=num_sms,
        kM0=kM0,
        kN0=kN0,
        causal=causal
    )

    # Calculate memory pipeline metrics
    memory_pipe = calculate_memory_pipe(
        kM0=kM0,
        kN0=kN0,
        head_dim=head_dim,
        dtype_q_size=problem.data_size_q,
        dtype_kv_size=problem.data_size_kv,
        sm_task_iterations=sm_task_iterations,
        hardware=hardware,
    )

    # Calculate operation statistics
    tensor_pipe, xu_pipe = calculate_ck_operation_stats(
        kM0=kM0,
        kN0=kN0,
        head_dim=head_dim,
        sm_task_iterations=sm_task_iterations,
        hardware=hardware,
    )

    return FaFeatures(
        tensor_pipe=tensor_pipe,
        xu_pipe=xu_pipe,
        memory_pipe=memory_pipe,
        hardware=hardware,
    )
