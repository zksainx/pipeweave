#!/usr/bin/env python3
"""
Workload Generator for Transformer Operators

This script generates workload specifications for transformer operators based on:
- Model configuration (hidden_size, num_heads, intermediate_size, etc.)
- Batch sequence lengths
- KV cache lengths
- Tensor parallelism size

Usage:
    Edit the parameters in the main() function and run:
    python workload_generator.py
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple


def divide(numerator: int, denominator: int) -> int:
    """
    Integer division with validation that the division is exact.

    Args:
        numerator: The number to be divided
        denominator: The number to divide by

    Returns:
        The quotient if division is exact

    Raises:
        ValueError: If the division is not exact
    """
    if numerator % denominator != 0:
        raise ValueError(
            f"Division is not exact: {numerator} is not divisible by {denominator}"
        )
    return numerator // denominator


def get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int) -> Tuple[int, int]:
    """
    Calculate the layer range for a given PP rank.

    Args:
        num_hidden_layers: Total number of hidden layers
        pp_rank: Pipeline parallel rank (0-indexed)
        pp_size: Total number of pipeline parallel ranks

    Returns:
        Tuple of (start_layer, end_layer) for this rank
    """
    layers_per_partition = num_hidden_layers // pp_size
    start_layer = pp_rank * layers_per_partition
    end_layer = start_layer + layers_per_partition

    # Last rank gets all remaining layers
    if pp_rank == pp_size - 1:
        end_layer = num_hidden_layers

    return start_layer, end_layer


def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_workload_from_csv(csv_path: str) -> Tuple[List[int], int]:
    """
    Load workload configuration from CSV file.

    Args:
        csv_path: Path to CSV file with columns: num_prefill_tokens, num_decode_tokens, output_tokens

    Returns:
        Tuple of (prefill_lengths, output_tokens)
        - prefill_lengths: List of prefill token counts (used for both q_lengths and kv_lengths)
        - output_tokens: Number of output tokens (assumes all rows have same value)
    """
    df = pd.read_csv(csv_path)

    # Extract prefill lengths (q_lengths = kv_lengths)
    prefill_lengths = df['num_prefill_tokens'].tolist()

    # Extract output tokens (use the first value, assuming all rows have same output_tokens)
    output_tokens = int(df['output_tokens'].iloc[0])

    return prefill_lengths, output_tokens


def get_head_dim(config: Dict[str, Any]) -> int:
    """Get head dimension from config, calculate if not present."""
    if 'head_dim' in config:
        return config['head_dim']
    else:
        return divide(config['hidden_size'], config['num_attention_heads'])


def generate_operators(
    q_lengths: List[int],
    kv_lengths: List[int],
    config: Dict[str, Any],
    tp_size: int,
    pp_size: int,
    attention_backend: str
) -> Dict[str, Any]:
    """
    Generate operator specifications for one iteration (prefill or decode step).
    This generates end-to-end workload, not per-rank workload.

    Args:
        q_lengths: List of query lengths for each request in the batch
        kv_lengths: List of KV cache lengths for each request in the batch
        config: Model configuration dictionary
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        attention_backend: FlashAttention backend type (e.g., 'fa2_ragged', 'fa2_paged')

    Returns:
        Dictionary containing operator specifications
    """
    # Extract model parameters
    hidden_size = config['hidden_size']
    num_heads = config['num_attention_heads']
    num_kv_heads = config['num_key_value_heads']
    intermediate_size = config['intermediate_size']
    vocab_size = config['vocab_size']
    head_dim = get_head_dim(config)
    num_layers = config['num_hidden_layers']

    # Calculate derived values
    batch_size = len(q_lengths)
    num_tokens = sum(q_lengths)

    operators = {}

    # Generate GEMM operators
    gemm_ops = []

    # 1. QKV Projection (ColumnParallelLinear)
    gemm_ops.append({
        "name": "qkv_proj",
        "type": "ColumnParallelLinear",
        "M": num_tokens,
        "N": divide((num_heads + 2 * num_kv_heads) * head_dim, tp_size),
        "K": hidden_size,
        "count": num_layers
    })

    # 2. Output Projection (RowParallelLinear)
    gemm_ops.append({
        "name": "o_proj",
        "type": "RowParallelLinear",
        "M": num_tokens,
        "N": hidden_size,
        "K": divide(num_heads * head_dim, tp_size),
        "count": num_layers
    })

    # 3. Gate and Up Projection (MergedColumnParallelLinear)
    gemm_ops.append({
        "name": "gate_up_proj",
        "type": "MergedColumnParallelLinear",
        "M": num_tokens,
        "N": divide(intermediate_size * 2, tp_size),
        "K": hidden_size,
        "count": num_layers
    })

    # 4. Down Projection (RowParallelLinear)
    gemm_ops.append({
        "name": "down_proj",
        "type": "RowParallelLinear",
        "M": num_tokens,
        "N": hidden_size,
        "K": divide(intermediate_size, tp_size),
        "count": num_layers
    })

    # 5. LM Head (torch.nn.Linear) - only last token
    gemm_ops.append({
        "name": "lm_head",
        "type": "Linear",
        "M": batch_size,
        "N": vocab_size,
        "K": hidden_size,
        "count": 1
    })

    operators["gemm"] = gemm_ops

    # Generate Attention operator
    # For nh and nkv: if kv_heads < tp_size, they are replicated, not partitioned
    if num_kv_heads >= tp_size:
        nkv_per_rank = divide(num_kv_heads, tp_size)
    else:
        nkv_per_rank = num_kv_heads

    operators["attention"] = {
        "attention_type": attention_backend,
        "bs": batch_size,
        "q_lengths": q_lengths,
        "kv_lengths": kv_lengths,
        "nh": divide(num_heads, tp_size),
        "nkv": nkv_per_rank,
        "hd": head_dim,
        "count": num_layers
    }

    # Generate RMSNorm operator
    operators["rmsnorm"] = {
        "seq": num_tokens,
        "dim": hidden_size,
        "count": 2 * num_layers + 1  # 2 per layer + 1 final 
        # Qwen3 qk norm 4*num_layers + 1
        #"count" : 4 * num_layers + 1
    }

    # Generate SiLUAndMul operator
    operators["siluandmul"] = {
        "seq": num_tokens,
        "dim": divide(intermediate_size * 2, tp_size),
        "count": num_layers
    }

    # Generate communication operators
    comm_ops = []

    # All-reduce (for TP)
    if tp_size > 1:
        # All-reduce happens after o_proj and down_proj (2 times per layer)
        # Communication volume: 2 * (tp_size - 1) / tp_size * data_size
        # Data type: bf16 (2 bytes)
        tp_data_size_bytes = num_tokens * hidden_size * 2
        # allreduce_volume_bytes = 2 * (tp_size - 1) * data_size_bytes // tp_size

        comm_ops.append({
            "name": "all_reduce",
            "type": "collective",
            "volume_bytes": tp_data_size_bytes,
            "count": 2 * num_layers  # 2 per layer (o_proj + down_proj)
        })

    # Send/Recv (for PP)
    if pp_size > 1:
        # PP creates (pp_size - 1) stage boundaries
        # Each boundary requires send/recv of hidden_states + residual
        # Data size: 2 * num_tokens * hidden_size * 2 bytes (bf16)
        pp_data_size_bytes = 2 * num_tokens * hidden_size * 2

        comm_ops.append({
            "name": "send_recv",
            "type": "p2p",
            "volume_bytes": pp_data_size_bytes,
            "count": pp_size - 1  # Number of stage boundaries
        })

    operators["communication"] = comm_ops

    return operators


def generate_workload(
    config: Dict[str, Any],
    q_lengths: List[int],
    kv_lengths: List[int],
    output_len: int,
    tp_size: int,
    pp_size: int,
    p_attn_backend: str,
    d_attn_backend: str
) -> Dict[str, Any]:
    """
    Generate complete workload specification including prefill and decode phases.

    Args:
        config: Model configuration dictionary
        q_lengths: List of query lengths for each request in the batch (prefill)
        kv_lengths: List of KV cache lengths for each request in the batch (prefill)
        output_len: Total number of iterations (1 prefill + (output_len-1) decode steps)
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        p_attn_backend: Attention backend for prefill phase (e.g., 'fa2_ragged')
        d_attn_backend: Attention backend for decode phase (e.g., 'fa2_paged')

    Returns:
        Dictionary containing complete workload specifications
    """
    # Validate input
    if len(q_lengths) != len(kv_lengths):
        raise ValueError(f"q_lengths length ({len(q_lengths)}) must match kv_lengths length ({len(kv_lengths)})")

    if output_len < 1:
        raise ValueError(f"output_len must be at least 1")

    # Extract model parameters
    hidden_size = config['hidden_size']
    num_heads = config['num_attention_heads']
    num_kv_heads = config['num_key_value_heads']
    intermediate_size = config['intermediate_size']
    vocab_size = config['vocab_size']
    head_dim = get_head_dim(config)
    num_layers = config['num_hidden_layers']

    # Get model name from config path or architectures
    model_name = config.get('_name_or_path', config.get('architectures', ['unknown'])[0])

    batch_size = len(q_lengths)

    workload = {
        "model": model_name,
        "model_params": {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
            "num_hidden_layers": num_layers
        },
        "tp_size": tp_size,
        "pp_size": pp_size,
        "iterations": []
    }

    # Generate iteration 1: prefill phase
    prefill_operators = generate_operators(q_lengths, kv_lengths, config, tp_size, pp_size, p_attn_backend)
    workload["iterations"].append({
        "iteration": 1,
        "phase": "prefill",
        "batch_size": batch_size,
        "q_lengths": q_lengths,
        "kv_lengths": kv_lengths,
        "num_tokens": sum(q_lengths),
        "operators": prefill_operators
    })

    # Generate iterations 2 to output_len: decode phases
    for step in range(1, output_len):
        # In decode, all q_lengths are 1
        decode_q_lengths = [1] * batch_size
        # KV lengths accumulate: original + step
        decode_kv_lengths = [kv + step for kv in kv_lengths]

        decode_operators = generate_operators(decode_q_lengths, decode_kv_lengths, config, tp_size, pp_size, d_attn_backend)

        workload["iterations"].append({
            "iteration": step + 1,
            "phase": "decode",
            "batch_size": batch_size,
            "q_lengths": decode_q_lengths,
            "kv_lengths": decode_kv_lengths,
            "num_tokens": batch_size,
            "operators": decode_operators
        })

    return workload


def main():
    # ============ Configuration Parameters ============
    # Model configuration file path
    config_path = 'config/Llama-3.1-70B.json'

    # Workload CSV file path
    workload_csv_path = 'workload/vllm_arxiv16.csv'

    # Tensor parallel size
    tp_size = 4

    # Pipeline parallel size
    pp_size = 2

    # Attention backend for prefill phase
    #p_attn_backend = 'fa2_ragged'
    p_attn_backend = 'fa3_ragged'

    # Attention backend for decode phase
    d_attn_backend = 'fa2_paged'
    # ==================================================

    # Validate config file exists
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    # Validate workload CSV file exists
    if not Path(workload_csv_path).exists():
        print(f"Error: Workload CSV file not found: {workload_csv_path}")
        return 1

    # Load model config
    print(f"Loading model config from: {config_path}")
    config = load_model_config(config_path)

    # Load workload from CSV
    print(f"Loading workload from: {workload_csv_path}")
    prefill_lengths, output_tokens = load_workload_from_csv(workload_csv_path)

    # Set q_lengths = kv_lengths = prefill_lengths
    q_lengths = prefill_lengths
    kv_lengths = prefill_lengths
    output_len = output_tokens

    # Generate output path from config and workload parameters
    # Extract model name from config path (e.g., 'Qwen2.5-14B' from 'config/Qwen2.5-14B.json')
    model_name = Path(config_path).stem
    # Extract dataset name from workload CSV path (e.g., 'arxiv_8' from 'workload/arxiv_8.csv')
    dataset_name = Path(workload_csv_path).stem
    # Extract attention backend name (e.g., 'fa2' from 'fa2_ragged')
    attn_backend_name = p_attn_backend.split('_')[0]
    # Generate output filename
    output_path = f'workload/{model_name}_{dataset_name}_{attn_backend_name}_tp{tp_size}_pp{pp_size}.json'

    # Generate workload
    print(f"Generating workload:")
    print(f"  Total iterations: {output_len} (1 prefill + {output_len - 1} decode)")
    print(f"  Batch size: {len(q_lengths)}")
    print(f"  Prefill num_tokens: {sum(q_lengths)}")
    print(f"  Tensor parallel size: {tp_size}")
    print(f"  Pipeline parallel size: {pp_size}")
    print(f"  Prefill attention backend: {p_attn_backend}")
    print(f"  Decode attention backend: {d_attn_backend}")

    try:
        workload = generate_workload(
            config=config,
            q_lengths=q_lengths,
            kv_lengths=kv_lengths,
            output_len=output_len,
            tp_size=tp_size,
            pp_size=pp_size,
            p_attn_backend=p_attn_backend,
            d_attn_backend=d_attn_backend
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Save to JSON file
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        json.dump(workload, f, indent=2)

    print(f"\nWorkload saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Model: {workload['model']}")
    print(f"  Layers: {workload['model_params']['num_hidden_layers']}")
    print(f"  Tensor parallel size: {workload['tp_size']}")
    print(f"  Pipeline parallel size: {workload['pp_size']}")
    print(f"  Total iterations: {len(workload['iterations'])}")
    print(f"    - Prefill: 1 iteration")
    print(f"    - Decode: {len(workload['iterations']) - 1} iterations")

    return 0


if __name__ == '__main__':
    exit(main())
