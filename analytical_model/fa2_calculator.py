#!/usr/bin/env python3
"""
FlashInfer BatchPrefill FA2 KernelTraits Parameter Calculator V2

Unified calculator for both paged and ragged KV cache storage layouts.
Simplified version focusing on kernel traits calculation and SM-level operation statistics.

Key Features:
- Unified paged/ragged support: Single function for both storage layouts
- RR scheduling simulation: Distributes tasks across SMs using round-robin strategy
- Operation statistics: Computes XU and MMA operation counts per SM
- Simplified output: Essential kernel traits and performance metrics only

Source code references:
- attention/prefill.cuh:100-140 (KernelTraits structure)
- attention/prefill.cuh:55-73 (Basic functions: get_num_warps_q, get_num_warps_kv, get_num_mma_q)
- utils.cuh:303 (FA2DetermineCtaTileQ function)
- attention/scheduler.cuh:101 (PrefillBinarySearchKVChunkSize function)
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

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


@dataclass
class CTA:
    """Represents a Cooperative Thread Array (CTA/Block)."""
    id: int
    iterations: int
    original_iterations: Optional[int] = None

    def __post_init__(self):
        if self.original_iterations is None:
            self.original_iterations = self.iterations


@dataclass
class SM:
    """Represents a Streaming Multiprocessor."""
    id: int
    max_ctas_per_sm: int = 2

    def __post_init__(self):
        self.active_ctas = []

    def can_accept_cta(self) -> bool:
        return len(self.active_ctas) < self.max_ctas_per_sm

    def assign_cta(self, cta: CTA) -> bool:
        if self.can_accept_cta():
            self.active_ctas.append(cta)
            return True
        return False

    def remove_cta(self, cta: CTA):
        if cta in self.active_ctas:
            self.active_ctas.remove(cta)


def get_num_warps_q(cta_tile_q: int) -> int:
    """Calculate number of warps in Q dimension.

    Based on source code: prefill.cuh:55-61

    Args:
        cta_tile_q: CTA tile size in Q dimension

    Returns:
        Number of warps in Q dimension
    """
    return 4 if cta_tile_q > 16 else 1


def get_num_warps_kv(cta_tile_q: int) -> int:
    """Calculate number of warps in KV dimension.

    Based on source code: prefill.cuh:63-65
    Note: Source function parameter is cta_tile_kv, but cta_tile_q is passed during scheduling

    Args:
        cta_tile_q: CTA tile size in Q dimension

    Returns:
        Number of warps in KV dimension
    """
    return 4 // get_num_warps_q(cta_tile_q)


def get_num_mma_q(cta_tile_q: int) -> int:
    """Calculate number of MMA operations in Q dimension.

    Based on source code: prefill.cuh:67-73

    Args:
        cta_tile_q: CTA tile size in Q dimension

    Returns:
        Number of MMA operations in Q dimension
    """
    return 2 if cta_tile_q > 64 else 1


def FA2DetermineCtaTileQ(avg_packed_qo_len: int, head_dim: int) -> int:
    """Determine CTA tile Q size.

    Based on source code: utils.cuh
    Note: Assumes Ampere or newer architecture (compute_capability >= 8)

    Args:
        avg_packed_qo_len: Average packed QO length
        head_dim: Attention head dimension

    Returns:
        CTA_TILE_Q size
    """
    if avg_packed_qo_len > 64 and head_dim < 256:
        return 128
    else:
        if avg_packed_qo_len > 16:
            return 64
        else:
            return 16


def PrefillBinarySearchKVChunkSize(max_batch_size_if_split: int,
                                   packed_qo_len_arr: List[int],
                                   kv_len_arr: List[int],
                                   qo_chunk_size: int,
                                   min_kv_chunk_size: int = 1) -> Tuple[bool, int]:
    """Binary search for optimal KV chunk size.

    Based on source code: scheduler.cuh:101

    Args:
        max_batch_size_if_split: Maximum batch size when KV is split
        packed_qo_len_arr: Array of packed QO lengths
        kv_len_arr: Array of KV lengths
        qo_chunk_size: QO chunk size (usually cta_tile_q)
        min_kv_chunk_size: Minimum KV chunk size

    Returns:
        Tuple of (split_kv, kv_chunk_size): Whether to split KV and KV chunk size
    """
    batch_size = len(packed_qo_len_arr)
    max_kv_len = 1
    for kv_len in kv_len_arr:
        max_kv_len = max(max_kv_len, kv_len)

    low = min_kv_chunk_size
    high = max_kv_len
    min_kv_len = 1

    while low < high:
        mid = (low + high) // 2
        new_batch_size = 0
        for i in range(batch_size):
            new_batch_size += ceil_div(packed_qo_len_arr[i], qo_chunk_size) * \
                              ceil_div(max(kv_len_arr[i], min_kv_len), mid)
        if new_batch_size > max_batch_size_if_split:
            low = mid + 1
        else:
            high = mid

    return (low < max_kv_len, low)


def is_kernel_traits_invalid(num_mma_q: int, num_mma_kv: int, num_mma_d_vo: int,
                           num_warps_q: int, dtype_kv_size: int) -> bool:
    """Implementation of KTraits::IsInvalid() logic.

    Based on source code: prefill.cuh:133-140
    Note: pos_encoding_mode forced to 'kNone'

    Args:
        num_mma_q: Number of MMA operations in Q dimension
        num_mma_kv: Number of MMA operations in KV dimension
        num_mma_d_vo: Number of MMA operations in V/O dimension
        num_warps_q: Number of warps in Q dimension
        dtype_kv_size: KV data type size in bytes

    Returns:
        Whether the configuration is invalid
    """
    # Various invalid condition checks
    if num_mma_d_vo < 4:
        return True

    if num_mma_d_vo == 4 and num_mma_kv % 2 == 1:
        return True

    # # COMMENTED: pos_encoding_mode forced to 'kNone'
    # if (pos_encoding_mode == 'kRoPELlama' and num_mma_d_vo > 4 and
    #     num_mma_d_vo % (2 * num_warps_q) != 0):
    #     return True

    # dtype_qk_accum == 'float32'
    dtype_qk_accum_size = 4
    if num_mma_q * (8 * num_mma_d_vo + 2 * dtype_qk_accum_size * num_mma_kv) >= 256:
        return True

    if dtype_kv_size == 1 and num_mma_kv * 2 % num_warps_q != 0:
        return True

    # # COMMENTED: pos_encoding_mode forced to 'kNone'
    # if dtype_kv_size == 1 and pos_encoding_mode == 'kRoPELlama':
    #     return True

    return False


def dispatch_num_mma_kv(max_num_mma_kv_constraint: int, num_mma_q: int, num_mma_d_vo: int,
                       num_warps_q: int, dtype_kv_size: int) -> int:
    """Simulate DISPATCH_NUM_MMA_KV macro selection logic.

    Based on source code: prefill.cuh
    Note: pos_encoding_mode forced to 'kNone'

    Args:
        max_num_mma_kv_constraint: Maximum constraint value for NUM_MMA_KV
        num_mma_q: Number of MMA operations in Q dimension
        num_mma_d_vo: Number of MMA operations in V/O dimension
        num_warps_q: Number of warps in Q dimension
        dtype_kv_size: KV data type size in bytes

    Returns:
        Selected NUM_MMA_KV value
    """
    # Start from maximum constraint and decrement to find first valid configuration
    for num_mma_kv in range(max_num_mma_kv_constraint, 0, -1):
        if not is_kernel_traits_invalid(num_mma_q, num_mma_kv, num_mma_d_vo,
                                       num_warps_q, dtype_kv_size):
            return num_mma_kv

    # If all invalid, return 1 as fallback
    return 1


def sub_if_greater_or_zero(x: int, y: int) -> int:
    """Implementation of sub_if_greater_or_zero logic.

    Based on source code: utils.cuh:351-353

    Args:
        x: Minuend
        y: Subtrahend

    Returns:
        (x > y) ? x - y : 0
    """
    return (x - y) if x > y else 0



def create_fa2_cta_workload(
    q_lengths: List[int],
    kv_lengths: List[int],
    num_kv_heads: int,
    causal: bool,
    cta_tile_q: int,
    cta_tile_kv: int,
    gqa_group_size: int,
    kv_chunk_size: int
) -> List[CTA]:
    """Create CTA workload based on real FlashInfer calculation logic.

    This function creates the actual CTA list for scheduling,
    taking into account the grid's z-dimension (num_kv_heads).

    Args:
        q_lengths: Query sequence lengths for all requests
        kv_lengths: KV sequence lengths for all requests
        num_kv_heads: Number of KV heads (grid z-dimension)
        causal: Whether to use causal mask
        cta_tile_q: CTA_TILE_Q size
        cta_tile_kv: CTA_TILE_KV size
        gqa_group_size: GQA group size (num_qo_heads // num_kv_heads)
        kv_chunk_size: KV chunk size

    Returns:
        List of CTA objects with real iteration counts
    """
    ctas = []
    cta_id = 0
    batch_size = len(q_lengths)

    for request_idx in range(batch_size):
        qo_len = q_lengths[request_idx]
        kv_len = kv_lengths[request_idx]

        # For GQA: packed_qo_len = qo_len * gqa_group_size
        packed_qo_len = qo_len * gqa_group_size
        num_q_tiles = ceil_div(packed_qo_len, cta_tile_q)
        num_kv_chunks = ceil_div(kv_len, kv_chunk_size)

        for qo_tile_idx in range(num_q_tiles):
            for kv_chunk_idx in range(num_kv_chunks):
                chunk_start = kv_chunk_idx * kv_chunk_size
                chunk_end = min(chunk_start + kv_chunk_size, kv_len)
                actual_chunk_size = chunk_end - chunk_start

                if causal:
                    kv_upper_bound = kv_len - qo_len + ceil_div((qo_tile_idx + 1) * cta_tile_q, gqa_group_size)
                    effective_chunk_size = sub_if_greater_or_zero(kv_upper_bound, chunk_start)
                    final_chunk_size = min(actual_chunk_size, effective_chunk_size)
                else:
                    final_chunk_size = actual_chunk_size

                num_iterations = ceil_div(final_chunk_size, cta_tile_kv)

                # Create num_kv_heads identical CTAs (grid z-dimension)
                for kv_head_idx in range(num_kv_heads):
                    cta = CTA(id=cta_id, iterations=num_iterations)
                    ctas.append(cta)
                    cta_id += 1

    return ctas


class NVIDIACTASchedulerRR:
    """NVIDIA Round-Robin CTA Scheduler Implementation.

    This scheduler implements dynamic CTA scheduling with retirement and replacement.

    Args:
        num_sms: Number of streaming multiprocessors available
        max_ctas_per_sm: Maximum CTAs that can run concurrently on each SM
    """

    def __init__(self, num_sms: int, max_ctas_per_sm: int = 2):
        self.num_sms = num_sms
        self.sms = [SM(id=i, max_ctas_per_sm=max_ctas_per_sm) for i in range(num_sms)]

    def schedule_ctas(self, ctas: List[CTA]) -> List[List[int]]:
        """Schedule CTAs using dynamic scheduling with retirement and replacement.

        Args:
            ctas: List of CTAs to schedule

        Returns:
            List of task iterations for each SM
        """
        waiting_queue = ctas.copy()
        sm_task_iterations = [[] for _ in range(self.num_sms)]

        # Phase 1: Initial round-robin assignment
        sm_index = 0
        while waiting_queue and any(sm.can_accept_cta() for sm in self.sms):
            cta = waiting_queue.pop(0)
            attempts = 0
            while attempts < self.num_sms:
                if self.sms[sm_index].can_accept_cta():
                    self.sms[sm_index].assign_cta(cta)
                    break
                sm_index = (sm_index + 1) % self.num_sms
                attempts += 1
            sm_index = (sm_index + 1) % self.num_sms

        if not waiting_queue:
            for sm in self.sms:
                for cta in sm.active_ctas:
                    if cta.iterations > 0:
                        sm_task_iterations[sm.id].append(cta.iterations)
            return sm_task_iterations

        # Phase 2: Dynamic scheduling simulation
        while waiting_queue or any(len(sm.active_ctas) > 0 for sm in self.sms):
            # Simulate one timestep
            retired_ctas = []
            for sm in self.sms:
                to_retire = []
                for cta in sm.active_ctas:
                    cta.iterations -= 1
                    if cta.iterations <= 0:
                        to_retire.append(cta)
                for cta in to_retire:
                    sm.remove_cta(cta)
                    retired_ctas.append((sm.id, cta))

            # Schedule new CTAs
            retirement_sms = [sm_id for sm_id, _ in retired_ctas]
            for sm_id in retirement_sms:
                if not waiting_queue:
                    break
                if self.sms[sm_id].can_accept_cta():
                    cta = waiting_queue.pop(0)
                    self.sms[sm_id].assign_cta(cta)

        # Collect all task iterations per SM
        for sm in self.sms:
            for cta in sm.active_ctas:
                if cta.original_iterations is not None and cta.original_iterations > 0:
                    sm_task_iterations[sm.id].append(cta.original_iterations)

        # Also need to collect retired CTAs' iterations
        # We'll track this by re-running the simulation
        return self._collect_task_iterations(ctas)

    def _collect_task_iterations(self, ctas: List[CTA]) -> List[List[int]]:
        """Re-run simulation to collect all task iterations per SM."""
        # Reset scheduler
        self.sms = [SM(id=i, max_ctas_per_sm=self.sms[0].max_ctas_per_sm) for i in range(self.num_sms)]
        sm_task_iterations = [[] for _ in range(self.num_sms)]

        waiting_queue = ctas.copy()

        # Phase 1: Initial assignment and record
        sm_index = 0
        while waiting_queue and any(sm.can_accept_cta() for sm in self.sms):
            cta = waiting_queue.pop(0)
            attempts = 0
            while attempts < self.num_sms:
                if self.sms[sm_index].can_accept_cta():
                    self.sms[sm_index].assign_cta(cta)
                    if cta.original_iterations is not None and cta.original_iterations > 0:
                        sm_task_iterations[sm_index].append(cta.original_iterations)
                    break
                sm_index = (sm_index + 1) % self.num_sms
                attempts += 1
            sm_index = (sm_index + 1) % self.num_sms

        # Phase 2: Dynamic scheduling and record
        while waiting_queue or any(len(sm.active_ctas) > 0 for sm in self.sms):
            retired_ctas = []
            for sm in self.sms:
                to_retire = []
                for cta in sm.active_ctas:
                    cta.iterations -= 1
                    if cta.iterations <= 0:
                        to_retire.append(cta)
                for cta in to_retire:
                    sm.remove_cta(cta)
                    retired_ctas.append((sm.id, cta))

            retirement_sms = [sm_id for sm_id, _ in retired_ctas]
            for sm_id in retirement_sms:
                if not waiting_queue:
                    break
                if self.sms[sm_id].can_accept_cta():
                    cta = waiting_queue.pop(0)
                    self.sms[sm_id].assign_cta(cta)
                    if cta.original_iterations is not None and cta.original_iterations > 0:
                        sm_task_iterations[sm_id].append(cta.original_iterations)

        return sm_task_iterations


def calculate_fa2_ops(cta_q: int, cta_kv: int, head_dim: int, iterations: int) -> Tuple[int, int]:
    """Calculate MMA and XU operation counts for FA2.

    Based on FA2 attention mechanism analysis from prefill.cuh.

    XU operations breakdown:
    1. update_mdo_states (prefill.cuh:889, 900-907):
       - o_scale exp2: 4 ops per thread per iteration
       - s_frag exp2: 32 ops per thread per iteration
    2. OutputTransform (variant_helper.cuh:82):
       - d_rcp: 128 ops per thread (called for each output element)
    3. write lse (prefill.cuh:1562):
       - log2: one per query position

    Args:
        cta_q: CTA Q tile size
        cta_kv: CTA KV tile size
        head_dim: Attention head dimension
        iterations: Number of KV iterations

    Returns:
        Dictionary with XU and MMA operation counts
    """
    num_threads = 128  # 32 threads/warp × 4 warps_q × 1 warp_kv

    # Per iteration XU operations
    # 1. s_frag exp2: 32 ops per thread (from update_mdo_states)
    s_frag_exp2_per_iter = 32 * num_threads  # 4,096

    # 2. o_scale exp2: 4 ops per thread (from update_mdo_states)
    o_scale_exp2_per_iter = 4 * num_threads  # 512

    # Total exp2 across all iterations
    total_exp2 = (s_frag_exp2_per_iter + o_scale_exp2_per_iter) * iterations

    # Final stage XU operations
    # 3. d_rcp: 128 ops per thread (from OutputTransform, called for each output element)
    d_rcp = 128 * num_threads  # 16,384

    # 4. lse log2: one per query position
    lse_log2 = cta_q  # 128

    total_xu = total_exp2 + d_rcp + lse_log2

    # MMA operations: 2 GEMMs per iteration
    # Q@K^T: 2 × CTA_Q × CTA_KV × head_dim FLOPs
    # P@V:   2 × CTA_Q × CTA_KV × head_dim FLOPs
    # 32/33 efficiency factor
    total_mma = 4 * cta_q * cta_kv * head_dim * iterations // 32 * 33

    return total_mma, total_xu


def calculate_fa2_operation_stats(
    cta_q: int,
    cta_kv: int,
    head_dim: int,
    sm_task_iterations: List[List[int]],
    hardware: HardwareSpec,
) -> Tuple[TensorPipe, XuPipe]:
    """Calculate operation statistics across all SMs."""
    total_xu = 0
    total_mma = 0
    sm_mma_max = 0
    sm_xu_max = 0

    for sm_tasks in sm_task_iterations:
        sm_xu = 0
        sm_mma = 0
        for iterations in sm_tasks:
            mma_ops, xu_ops = calculate_fa2_ops(cta_q, cta_kv, head_dim, iterations)
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


def calculate_fa2_params(problem: FaProblemConfig, hardware: HardwareSpec) -> FaFeatures:
    """Calculate FA2 KernelTraits parameters for variable-length sequences."""

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
    if hardware.shared_memory_size is None:
        raise ValueError("Hardware spec must include shared_memory_size for FA2 calculations")
    if hardware.tc_bf16 <= 0:
        raise ValueError("Hardware spec must have positive tc_bf16 throughput")
    if hardware.xu_fp32 <= 0:
        raise ValueError("Hardware spec must have positive xu_fp32 throughput")

    page_size = 16 if layout == "paged" else 1
    gqa_group_size = num_qo_heads // num_kv_heads

    total_packed_qo_len = sum(q_len * gqa_group_size for q_len in q_lengths)
    avg_packed_qo_len = total_packed_qo_len // batch_size

    cta_tile_q = FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim)
    num_warps_q = get_num_warps_q(cta_tile_q)
    num_warps_kv = get_num_warps_kv(cta_tile_q)
    num_mma_q = get_num_mma_q(cta_tile_q)

    head_dim_qk = head_dim
    head_dim_vo = head_dim
    num_mma_d_qk = head_dim_qk // 16
    num_mma_d_vo = head_dim_vo // 16

    dtype_q_size = problem.data_size_q
    dtype_kv_size = problem.data_size_kv

    max_smem_per_sm = int(hardware.shared_memory_size*1024)

    smem_per_cta = (cta_tile_q * head_dim_qk * dtype_q_size +
                    (head_dim_qk + head_dim_vo) * 16 * num_warps_kv * dtype_kv_size)
    num_ctas_per_sm = 2 if max_smem_per_sm >= 2 * smem_per_cta else 1
    max_smem_per_threadblock = max_smem_per_sm // num_ctas_per_sm

    max_num_mma_kv_reg = 8 // num_mma_q
    max_num_mma_kv_smem = ((max_smem_per_threadblock - cta_tile_q * head_dim_qk * dtype_q_size) //
                          ((head_dim_qk + head_dim_vo) * 16 * num_warps_kv * dtype_kv_size))
    max_num_mma_kv_constraint = min(max_num_mma_kv_smem, max_num_mma_kv_reg)
    num_mma_kv = dispatch_num_mma_kv(max_num_mma_kv_constraint, num_mma_q, num_mma_d_vo,
                                    num_warps_q, dtype_kv_size)

    cta_tile_kv = num_mma_kv * num_warps_kv * 16

    num_blocks_per_sm = 2
    max_grid_size = num_blocks_per_sm * num_sm
    max_batch_size_if_split = max_grid_size // num_kv_heads
    min_kv_chunk_size = max((128 // page_size), 1)

    packed_qo_len_arr = [q_len * gqa_group_size for q_len in q_lengths]
    kv_len_arr = kv_lengths.copy()
    effective_kv_len_arr = kv_len_arr.copy()

    split_kv, kv_chunk_size = PrefillBinarySearchKVChunkSize(
        max_batch_size_if_split=max_batch_size_if_split,
        packed_qo_len_arr=packed_qo_len_arr,
        kv_len_arr=effective_kv_len_arr,
        qo_chunk_size=cta_tile_q,
        min_kv_chunk_size=min_kv_chunk_size
    )

    total_tasks_per_head = 0
    for i in range(batch_size):
        packed_qo_len_single = packed_qo_len_arr[i]
        effective_kv_len = effective_kv_len_arr[i]
        num_tiles_q = ceil_div(packed_qo_len_single, cta_tile_q)
        num_chunks_kv = ceil_div(effective_kv_len, kv_chunk_size)
        tasks_per_head = num_tiles_q * num_chunks_kv
        total_tasks_per_head += tasks_per_head

    # padded_batch_size = total_tasks_per_head

    # grid_dim = (padded_batch_size, 1, num_kv_heads)
    # block_dim = (32, num_warps_q, num_warps_kv)

    ctas = create_fa2_cta_workload(
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        num_kv_heads=num_kv_heads,
        causal=causal,
        cta_tile_q=cta_tile_q,
        cta_tile_kv=cta_tile_kv,
        gqa_group_size=gqa_group_size,
        kv_chunk_size=kv_chunk_size
    )

    scheduler = NVIDIACTASchedulerRR(num_sms=num_sm, max_ctas_per_sm=2)
    sm_iteration_distribution = scheduler.schedule_ctas(ctas)

    memory_pipe = calculate_memory_pipe(
        cta_tile_q=cta_tile_q,
        cta_tile_kv=cta_tile_kv,
        head_dim=head_dim_qk,
        dtype_q_size=dtype_q_size,
        dtype_kv_size=dtype_kv_size,
        sm_task_iterations=sm_iteration_distribution,
        hardware=hardware,
    )

    tensor_pipe, xu_pipe = calculate_fa2_operation_stats(
        cta_q=cta_tile_q,
        cta_kv=cta_tile_kv,
        head_dim=head_dim_qk,
        sm_task_iterations=sm_iteration_distribution,
        hardware=hardware,
    )

    return FaFeatures(
        tensor_pipe=tensor_pipe,
        xu_pipe=xu_pipe,
        memory_pipe=memory_pipe,
        hardware=hardware,
    )
