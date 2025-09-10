"""Common dataclass definitions shared across analytical calculators."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class HardwareSpec:

    tc_bf16: float
    tc_fp8: float
    xu_fp32: float
    fma_fp32: float
    num_sms: int
    sm_freq: float
    mem_bandwidth: float
    l2_cache_bandwidth: float
    shared_memory_bandwidth: float
    shared_memory_size: Optional[float] = None



@dataclass
class TensorPipe:

    all_ops: float
    all_cycle: float
    sm_max_ops: float
    sm_max_cycle: float


@dataclass
class MemoryPipe:
    """Memory pipeline metrics."""

    global_in_flight: float
    global_cycle: float
    local_cycle: float
    sm_max_in_flight: float
    sm_max_global_cycle: float
    sm_max_shared_cycle: float
    sm_max_local_cycle: float


@dataclass
class XuPipe:

    all_ops: float
    all_cycle: float
    sm_max_ops: float
    sm_max_cycle: float


@dataclass
class FmaPipe:

    all_ops: float
    all_cycle: float
    sm_max_ops: float
    sm_max_cycle: float


@dataclass
class GemmProblemConfig:
    """Problem-level configuration shared across GEMM calculators."""

    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    tile_k: int
    cta_count: int
    is_split_k: bool = True
    data_size_bytes: int = 2


@dataclass
class GemmFeatures:
    """Final GEMM feature vector returned by calculator helpers."""

    tensor_pipe: TensorPipe
    memory_pipe: MemoryPipe
    hardware: HardwareSpec


@dataclass
class FaProblemConfig:
    """Problem configuration shared by FA calculators."""

    batch_size: int
    q_lengths: List[int]
    kv_lengths: List[int]
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    layout: str
    data_size_q: int
    data_size_kv: int
    data_size_o: int
    causal: bool = True


@dataclass
class FaFeatures:
    """Aggregated features for Flash Attention calculators."""

    tensor_pipe: TensorPipe
    xu_pipe: XuPipe
    memory_pipe: MemoryPipe
    hardware: HardwareSpec


@dataclass
class RmsNormProblemConfig:
    """Problem configuration for RMSNorm calculators."""

    batch_size: int
    dim: int
    dtype_size: int


@dataclass
class SiluMulProblemConfig:
    """Problem configuration for SiLU* calculations."""

    seq_len: int
    dim: int
    dtype_size: int


@dataclass
class RmsNormFeatures:
    """Aggregated features for RMSNorm calculators."""

    fma_pipe: FmaPipe
    xu_pipe: XuPipe
    memory_pipe: MemoryPipe
    hardware: HardwareSpec


@dataclass
class SiluMulFeatures:
    """Aggregated features for SiLU* calculators."""

    fma_pipe: FmaPipe
    xu_pipe: XuPipe
    memory_pipe: MemoryPipe
    hardware: HardwareSpec


@dataclass
class MoeProblemConfig:
    """Problem configuration for Triton Fused MOE calculators."""

    m: int  # Number of tokens
    e: int  # Number of experts
    top_k: int  # Number of experts per token
    h: int  # Hidden size
    n: int  # MoE intermediate size per expert
    data_size_bytes: int = 2  # Data type size (2 for bf16/fp16)


@dataclass
class MoeFeatures:
    """Aggregated features for Triton Fused MOE calculators."""

    tensor_pipe: TensorPipe
    memory_pipe: MemoryPipe
    hardware: HardwareSpec


__all__ = [
    "HardwareSpec",
    "TensorPipe",
    "MemoryPipe",
    "XuPipe",
    "FmaPipe",
    "GemmProblemConfig",
    "GemmFeatures",
    "FaProblemConfig",
    "FaFeatures",
    "RmsNormProblemConfig",
    "SiluMulProblemConfig",
    "RmsNormFeatures",
    "SiluMulFeatures",
    "MoeProblemConfig",
    "MoeFeatures",
]
