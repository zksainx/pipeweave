#!/usr/bin/env python3
"""Triton Fused MOE analytical calculator.

This calculator models the Triton fused MOE kernel performance based on:
- Token routing and distribution across experts
- Two-pass GEMM operations (W13: gate_up_proj, W2: down_proj)
- Uniform CTA workload scheduling

Based on the fused MOE implementation from SGLang/vLLM.
"""

from typing import List, Tuple, Dict, Any

from pipes import (
    HardwareSpec,
    MemoryPipe,
    MoeFeatures,
    MoeProblemConfig,
    TensorPipe,
)
from utils import ceil_div

def get_moe_config(
    M: int,
    E: int,
    H: int,
    smem: int = 100,
) -> Dict[str, Any]:
    """Pick Triton tiling parameters for the MoE GEMM kernel via heuristics.

    Args:
            M: Tokens per expert batch (matrix height for BM/BK decisions).
            E: Number of experts, used to break ties when M is very large.
            H: Expert hidden size (matrix width for BN/GM decisions).
            smem: Shared-memory budget in KB observed during kernel tuning.

    Returns:
            Dict with Triton tiling parameters: BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, and GROUP_SIZE_M.

    Heuristic details:
            - BLOCK_SIZE_M prefers 16/32/64/128 tiles depending on M and smem, with
                very small M choosing 16 and generous smem allowing 128 for better reuse.
            - BLOCK_SIZE_N is driven by H and smem, generally stepping up to 64 or
                128 when the hidden dimension widens or more shared memory is available.
            - BLOCK_SIZE_K shrinks from 256 to 64 as M grows or smem tightens to
                keep register pressure balanced.
            - GROUP_SIZE_M increases for large H to spread work across blocks and
                reverts to 1 when the hidden width is moderate or smem is limited.
    """

    # 1. M Heuristic for BM (BLOCK_SIZE_M)
    if M <= 128:
        BM = 16
    elif M <= 256:
        if smem <= 100:
            BM = 32
        elif smem <= 164:
            BM = 32
        else:
            BM = 32
    elif M <= 512:
        if smem <= 100:
            BM = 64
        elif smem <= 164:
            BM = 64
        else:
            BM = 64
    elif M <= 2048:
        if smem <= 100:
            BM = 128
        elif smem <= 164:
            BM = 64
        else:
            BM = 128
    else:
        if smem <= 100:
            if E < 64:
                BM = 64
            else:
                BM = 128
        elif smem <= 164:
            BM = 64
        else:
            BM = 128

    # 2. H Heuristic for BN (BLOCK_SIZE_N)
    if H <= 1536:
        if smem <= 100:
            if M <= 128:
                BN = 16
            else:
                BN = 64
        elif smem <= 164:
            BN = 128
        else:
            BN = 128
    elif H <= 3072:
        if smem <= 100:
            BN = 64
        elif smem <= 164:
            BN = 128
        else:
            BN = 128
    else:
        if smem <= 100:
            BN = 64
        elif smem <= 164:
            BN = 128
        else:
            BN = 128

    # 3. M Heuristic for BK (BLOCK_SIZE_K)
    if M <= 256:
        if smem <= 100:
            BK = 256
        elif smem <= 164:
            BK = 128
        else:
            BK = 128
    else:
        BK = 64 

    # 4. H Heuristic for GM (GROUP_SIZE_M)
    if H <= 1536:
        if smem <= 100:
            GM = 1
        elif smem <= 164:
            GM = 1
        else:
            GM = 1
    elif H <= 2048:
        if smem <= 100:
            GM = 1
        elif smem <= 164:
            GM = 1
        else:
            GM = 1
    elif H <= 3072:
        if smem <= 100:
            GM = 16
        elif smem <= 164:
            GM = 16
        else:
            GM = 1
    else:
        if smem <= 100:
            GM = 16
        elif smem <= 164:
            GM = 32
        else:
            GM = 1

    return {
        "BLOCK_SIZE_M": BM,
        "BLOCK_SIZE_N": BN,
        "BLOCK_SIZE_K": BK,
        "GROUP_SIZE_M": GM,
    }


def calculate_token_distribution_per_expert_rr(
    m: int,
    e: int,
    top_k: int,
    block_size_m: int,
) -> Tuple[List[int], List[int], int]:
    """Calculate token distribution across experts with padding using round-robin routing.

    Based on moe_align_block_size_ideal logic from the benchmark.
    Tokens are evenly distributed across experts in round-robin fashion.

    Args:
        m: Number of tokens
        e: Number of experts
        top_k: Number of experts per token
        block_size_m: Block size for alignment

    Returns:
        Tuple of (tokens_per_expert, padded_tokens_per_expert, total_padded_tokens)
    """
    total_tokens = m * top_k

    # For round-robin: tokens are evenly distributed across experts
    tokens_per_expert = [0] * e
    for i in range(total_tokens):
        expert_id = i % e
        tokens_per_expert[expert_id] += 1

    # Pad each expert's token count to block_size_m
    padded_tokens_per_expert = [
        ceil_div(count, block_size_m) * block_size_m
        for count in tokens_per_expert
    ]

    total_padded_tokens = sum(padded_tokens_per_expert)

    return tokens_per_expert, padded_tokens_per_expert, total_padded_tokens


def calculate_moe_w13_ops(
    n: int,
    h: int,
    padded_tokens: int,
    block_size_m: int,
    block_size_n: int,
) -> Tuple[int, int]:
    """Calculate tensor ops and grid size for W13 (gate_up_proj) pass.

    W13 GEMM: [M, H] @ [E, N*2, H] → [M, top_k, N*2]
    After sorting and padding, effective shape: [padded_tokens, H] @ [N*2, H]^T

    Args:
        n: MoE intermediate size per expert
        h: Hidden size
        padded_tokens: Total tokens after expert padding
        block_size_m: CTA tile size M
        block_size_n: CTA tile size N

    Returns:
        Tuple of (total_flops, grid_size)
    """
    # Output dimension for W13 is N*2 (gate + up)
    n_out = n * 2

    # Grid dimensions based on padded tokens and output size
    m_tiles = ceil_div(padded_tokens, block_size_m)
    n_tiles = ceil_div(n_out, block_size_n)

    grid_size = m_tiles * n_tiles

    # Total FLOPs: 2 * M * N * K for GEMM
    # Use padded dimensions for accurate modeling
    total_flops = 2 * padded_tokens * n_out * h

    return total_flops, grid_size


def calculate_moe_w2_ops(
    n: int,
    h: int,
    padded_tokens: int,
    block_size_m: int,
    block_size_n: int,
) -> Tuple[int, int]:
    """Calculate tensor ops and grid size for W2 (down_proj) pass.

    W2 GEMM: [M*top_k, N] @ [E, H, N] → [M, top_k, H]
    After expert processing: [padded_tokens, N] @ [H, N]^T

    Args:
        n: MoE intermediate size per expert
        h: Hidden size
        padded_tokens: Total tokens after expert padding
        block_size_m: CTA tile size M
        block_size_n: CTA tile size N

    Returns:
        Tuple of (total_flops, grid_size)
    """
    # W2 output dimension is H
    # Input is [M*top_k, N] after intermediate processing

    # Grid dimensions
    m_tiles = ceil_div(padded_tokens, block_size_m)
    n_tiles = ceil_div(h, block_size_n)

    grid_size = m_tiles * n_tiles

    # Total FLOPs: 2 * M * N * K
    total_flops = 2 * padded_tokens * h * n

    return total_flops, grid_size


def schedule_uniform_ctas(total_ctas: int, num_sms: int) -> List[int]:
    """Schedule uniform CTAs across SMs.

    For fused MOE, all CTAs process the same tile size, so workload is uniform.

    Args:
        total_ctas: Total number of CTAs to schedule
        num_sms: Number of streaming multiprocessors

    Returns:
        List of CTA counts per SM
    """
    if num_sms <= 0:
        raise ValueError("Number of SMs must be positive")

    base = total_ctas // num_sms
    remainder = total_ctas % num_sms
    return [base + 1 if i < remainder else base for i in range(num_sms)]


def calculate_memory_pipe(
    problem: MoeProblemConfig,
    block_size_m: int,
    block_size_n: int,
    grid_size: int,
    ctas_per_sm: List[int],
    hardware: HardwareSpec,
    pass_type: str,  # "w13" or "w2"
) -> MemoryPipe:
    """Calculate memory pipeline metrics for one MOE pass.

    Only counts LOAD traffic (input activations + weights).
    Total memory = sum of each CTA's load.

    Args:
        problem: MOE problem configuration
        block_size_m: CTA tile size M
        block_size_n: CTA tile size N
        grid_size: Total number of CTAs
        ctas_per_sm: CTA distribution across SMs
        hardware: Hardware specification
        pass_type: "w13" for gate_up_proj, "w2" for down_proj

    Returns:
        MemoryPipe with calculated metrics for the specified pass
    """
    if hardware.mem_bandwidth <= 0:
        raise ValueError("Hardware mem_bandwidth must be positive")
    if hardware.l2_cache_bandwidth <= 0:
        raise ValueError("Hardware l2_cache_bandwidth must be positive")
    if hardware.shared_memory_bandwidth <= 0:
        raise ValueError("Hardware shared_memory_bandwidth must be positive")

    dtype_bytes = problem.data_size_bytes

    # Calculate per-CTA load (activation tile + weight tile)
    if pass_type == "w13":
        # W13: Each CTA loads [BLOCK_SIZE_M, H] activation + [H, BLOCK_SIZE_N] weight
        # Activation tile: BLOCK_SIZE_M × H
        activation_tile_bytes = block_size_m * problem.h * dtype_bytes
        # Weight tile: H × BLOCK_SIZE_N (output is N*2, so weight reads from N*2 dimension)
        weight_tile_bytes = problem.h * block_size_n * dtype_bytes
    elif pass_type == "w2":
        # W2: Each CTA loads [BLOCK_SIZE_M, N] activation + [N, BLOCK_SIZE_N] weight
        # Activation tile: BLOCK_SIZE_M × N
        activation_tile_bytes = block_size_m * problem.n * dtype_bytes
        # Weight tile: N × BLOCK_SIZE_N (output is H)
        weight_tile_bytes = problem.n * block_size_n * dtype_bytes
    else:
        raise ValueError(f"Invalid pass_type: {pass_type}. Must be 'w13' or 'w2'")

    # Per-CTA load
    bytes_per_cta = activation_tile_bytes + weight_tile_bytes

    # Total load = sum of all CTAs
    total_bytes = bytes_per_cta * grid_size
    global_in_flight_kb = total_bytes / 1024.0

    # Per-SM memory based on CTA distribution
    sm_max_ctas = max(ctas_per_sm) if ctas_per_sm else 0
    sm_max_bytes = bytes_per_cta * sm_max_ctas
    sm_max_in_flight_kb = sm_max_bytes / 1024.0

    # Calculate cycles
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


def triton_moe_calculator(
    problem: MoeProblemConfig,
    hardware: HardwareSpec,
    pass_type: str = "w13"
) -> MoeFeatures:
    """Calculate Triton Fused MOE analytical features for ONE kernel pass.

    This calculator models a single kernel call (either W13 or W2).
    Call this function twice to model both passes separately.

    Args:
        problem: MOE problem configuration with kernel parameters
        hardware: Hardware specification
        pass_type: "w13" for gate_up_proj pass, "w2" for down_proj pass

    Returns:
        MoeFeatures with tensor and memory pipeline metrics for the specified pass
    """
    if hardware.tc_bf16 <= 0:
        raise ValueError("Hardware tc_bf16 throughput must be positive")
    if hardware.num_sms <= 0:
        raise ValueError("Hardware num_sms must be positive")
    if pass_type not in ["w13", "w2"]:
        raise ValueError(f"Invalid pass_type: {pass_type}. Must be 'w13' or 'w2'")

    # Get kernel configuration based on problem size and shared-memory budget
    if hardware.shared_memory_size is None:
        raise ValueError(
            "Hardware spec must include shared_memory_size for Triton MoE calculations"
        )

    config = get_moe_config(
        M=problem.m,
        E=problem.e,
        H=problem.h,
        smem=hardware.shared_memory_size)
    
    block_size_m = config["BLOCK_SIZE_M"]
    block_size_n = config["BLOCK_SIZE_N"]
    block_size_k = config["BLOCK_SIZE_K"]

    # Calculate token distribution across experts
    tokens_per_expert, padded_tokens_per_expert, total_padded_tokens = (
        calculate_token_distribution_per_expert_rr(
            m=problem.m,
            e=problem.e,
            top_k=problem.top_k,
            block_size_m=block_size_m,
        )
    )

    # Calculate ops based on pass type
    if pass_type == "w13":
        # W13 pass: [M, H] @ [E, N*2, H] → [M, top_k, N*2]
        total_flops, grid_size = calculate_moe_w13_ops(
            n=problem.n,
            h=problem.h,
            padded_tokens=total_padded_tokens,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
        )
    else:  # "w2"
        # W2 pass: [M*top_k, N] @ [E, H, N] → [M, top_k, H]
        total_flops, grid_size = calculate_moe_w2_ops(
            n=problem.n,
            h=problem.h,
            padded_tokens=total_padded_tokens,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
        )

    # Schedule CTAs to SMs (uniform workload)
    ctas_per_sm = schedule_uniform_ctas(grid_size, hardware.num_sms)
    sm_max_ctas = max(ctas_per_sm) if ctas_per_sm else 0

    # Calculate per-CTA FLOPs
    flops_per_cta = total_flops / grid_size if grid_size > 0 else 0
    sm_max_flops = flops_per_cta * sm_max_ctas

    # Tensor pipeline
    overall_cycle = total_flops / hardware.tc_bf16 / hardware.num_sms
    sm_max_cycle = sm_max_flops / hardware.tc_bf16

    tensor_pipe = TensorPipe(
        all_ops=total_flops,
        all_cycle=overall_cycle,
        sm_max_ops=sm_max_flops,
        sm_max_cycle=sm_max_cycle,
    )

    # Memory pipeline for this pass
    memory_pipe = calculate_memory_pipe(
        problem=problem,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        grid_size=grid_size,
        ctas_per_sm=ctas_per_sm,
        hardware=hardware,
        pass_type=pass_type,
    )

    return MoeFeatures(
        tensor_pipe=tensor_pipe,
        memory_pipe=memory_pipe,
        hardware=hardware,
    )
