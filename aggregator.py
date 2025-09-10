#!/usr/bin/env python3
"""
Duration Aggregator for Transformer Workloads

This script predicts operator durations and end-to-end latency for transformer workloads.
It loads MLP models trained for each operator type and predicts durations based on workload
specifications and hardware characteristics.

Usage:
    python3 aggregator.py --workload workload.json \
                         --hardware "A100" \
                         --model_dir mlp_models/ \
                         --output predictions.json
"""

import json
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mlp_model import MLP

# Import analytical model calculators
sys.path.insert(0, str(Path(__file__).parent / 'analytical_model'))
from pipes import (
    GemmProblemConfig, HardwareSpec, FaProblemConfig,
    RmsNormProblemConfig, SiluMulProblemConfig
)
from gemm_8_calculator import gemm8_calculator
from gemm_9_calculator import gemm9_calculator
from fa2_calculator import calculate_fa2_params
from fa3_calculator import calculate_fa3_params
# from fa_cutlass_calculator import calculate_fa_cutlass_params
from rmsnorm_calculator import rmsnorm_calculator
from silumul_calculator import silu_mul_calculator


class DurationPredictor:
    """Predicts operator durations using trained MLP models."""

    def __init__(
        self,
        hardware_name: str,
        model_dir: str = "mlp_models",
        dataset_dir: str = "dataset",
        hardware_dir: str = "hardware",
        device: str = "cpu",
        collective_hardware_name: Optional[str] = None
    ):
        """
        Initialize the duration predictor.

        Args:
            hardware_name: Name of the hardware (e.g., 'NVIDIA A100-SXM4-80GB')
            model_dir: Directory containing trained MLP models
            dataset_dir: Directory containing operator datasets
            hardware_dir: Directory containing hardware specifications
            device: Device to run inference on ('cpu' or 'cuda')
            collective_hardware_name: Hardware name for collective communication dataset (e.g., 'A100', 'H20')
        """
        self.hardware_name = hardware_name
        self.model_dir = Path(model_dir)
        self.dataset_dir = Path(dataset_dir)
        self.hardware_dir = Path(hardware_dir)
        self.device = device
        self.collective_hardware_name = collective_hardware_name

        # Load hardware specifications
        self.hardware_spec = self._load_hardware_spec()

        # Create hardware spec object for analytical model
        self.hardware_spec_obj = self._create_hardware_spec_obj()

        # Load MLP models for each operator type
        self.models = self._load_models()

        # Load datasets for parameter lookup (only for config parameters, not features)
        self.datasets = self._load_datasets()

        # Load collective communication models (Random Forest)
        self.collective_models = self._load_collective_models()

        print(f"Initialized DurationPredictor for {hardware_name}")
        print(f"  Loaded {len(self.models)} MLP models")
        print(f"  Hardware: {self.hardware_spec['name']}")

    def _create_hardware_spec_obj(self) -> HardwareSpec:
        """Create HardwareSpec object from hardware spec dict."""
        return HardwareSpec(
            tc_fp8=self.hardware_spec['tc_fp8'],
            tc_bf16=self.hardware_spec['tc_bf16'],
            xu_fp32=self.hardware_spec['xu_fp32'],
            fma_fp32=self.hardware_spec['fma_fp32'],
            num_sms=self.hardware_spec['num_sms'],
            sm_freq=self.hardware_spec['sm_freq'],
            mem_bandwidth=self.hardware_spec['mem_bandwidth'],
            l2_cache_bandwidth=self.hardware_spec['l2_cache_bandwidth'],
            shared_memory_bandwidth=self.hardware_spec['shared_memory_bandwidth'],
            shared_memory_size=self.hardware_spec.get('shared_memory_size')
        )

    def _load_hardware_spec(self) -> Dict[str, Any]:
        """Load hardware specifications from JSON file."""
        # Try to find hardware spec file
        # First try with hardware_name as-is
        hardware_file = self.hardware_dir / f"{self.hardware_name}.json"

        if not hardware_file.exists():
            raise FileNotFoundError(f"Hardware spec not found for: {self.hardware_name}")

        with open(hardware_file, 'r') as f:
            spec_json = json.load(f)

        # Convert JSON fields (camelCase) to snake_case for internal use
        spec = {
            'name': spec_json.get('name', self.hardware_name),
            'tc_fp8': spec_json.get('tcFp8', 0),
            'tc_bf16': spec_json['tcBf16'],
            'xu_fp32': spec_json['xuFp32'],
            'fma_fp32': spec_json['FmaFp32'],
            'num_sms': spec_json['numSms'],
            'sm_freq': spec_json['smFreq'],
            'mem_bandwidth': spec_json['memBandwidth'],
            'l2_cache_bandwidth': spec_json['l2CacheBandwidth'],
            'shared_memory_bandwidth': spec_json['sharedMemoryBandwidth'],
            'shared_memory_size': spec_json.get('sharedMemorySize'),
            'architecture': spec_json.get('architecture', 'ampere')
        }

        return spec

    def _load_models(self) -> Dict[str, torch.nn.Module]:
        """Load trained MLP models for each operator type."""
        models = {}
        operator_types = ['gemm', 'attn', 'rmsnorm', 'siluandmul']

        # Feature dimensions for each operator type
        feature_dims = {
            'gemm': 11,
            'attn': 15,
            'rmsnorm': 15,
            'siluandmul': 15
        }

        for op_type in operator_types:
            # Look for model in timestamped subdirectories
            op_dir = self.model_dir / op_type
            if not op_dir.exists():
                print(f"Warning: Directory not found for {op_type}: {op_dir}")
                continue

            # Find the latest timestamped directory
            subdirs = sorted([d for d in op_dir.iterdir() if d.is_dir()])
            if not subdirs:
                print(f"Warning: No model subdirectories found for {op_type}")
                continue

            latest_dir = subdirs[-1]  # Take the latest (sorted by name which is timestamp)
            model_path = latest_dir / f"{op_type}_mlp_model.pth"

            if not model_path.exists():
                print(f"Warning: Model not found for {op_type}: {model_path}")
                continue

            # Create model with correct input dimension
            input_dim = feature_dims[op_type]
            model = MLP(input_dim=input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.1)

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(self.device)

            models[op_type] = model
            print(f"  Loaded {op_type} model from {model_path}")

        return models

    def _load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load operator datasets for parameter lookup (only GEMM needs this)."""
        datasets = {}

        # Hardware fallback mapping for GEMM data
        # Maps hardware without data to similar hardware with data
        hardware_fallback = {
            'NVIDIA RTX PRO 6000 Blackwell Server Edition': 'NVIDIA A40',
            'NVIDIA RTX A6000': 'NVIDIA A40',
            'NVIDIA L40': 'NVIDIA RTX 6000 Ada Generation',
            'NVIDIA H200': 'NVIDIA H800',
            'NVIDIA H100': 'NVIDIA H800'
        }

        # Only load GEMM dataset for config parameter lookup
        # Other operators don't need dataset - they get all params from workload
        gemm_filepath = self.dataset_dir / 'gemm_train.csv'

        if gemm_filepath.exists():
            df = pd.read_csv(gemm_filepath)

            # Try to filter by hardware using full name from hardware spec
            hardware_name = self.hardware_spec['name']
            df_hw = df[df['hardware'] == hardware_name]

            # If no data found and hardware has a fallback, use fallback hardware
            if len(df_hw) == 0 and hardware_name in hardware_fallback:
                fallback_name = hardware_fallback[hardware_name]
                print(f"  No GEMM data for {hardware_name}, using fallback: {fallback_name}")
                df_hw = df[df['hardware'] == fallback_name]

            datasets['gemm'] = df_hw
            print(f"  Loaded {len(df_hw)} GEMM samples for config lookup")
        else:
            print(f"Warning: GEMM dataset not found: {gemm_filepath}")

        return datasets

    def _load_collective_models(self) -> Dict[str, Any]:
        """Load trained Random Forest models for collective communication."""
        collective_models = {}

        if self.collective_hardware_name is None:
            print("  No collective hardware specified, skipping collective model loading")
            return collective_models

        collective_dir = self.dataset_dir / 'collective' / self.collective_hardware_name

        if not collective_dir.exists():
            raise FileNotFoundError(f"Collective model directory not found: {collective_dir}")

        # Load all_reduce and send_recv models
        for comm_type in ['all_reduce', 'send_recv']:
            model_path = collective_dir / f"{comm_type}_rf_model.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                collective_models[comm_type] = model
                print(f"  Loaded {comm_type} Random Forest model from {self.collective_hardware_name}")
            else:
                print(f"  Warning: {comm_type} model not found: {model_path}")

        return collective_models

    def interpolate_collective_time(
        self,
        comm_name: str,
        volume_bytes: int,
        tp_size: int
    ) -> Optional[float]:
        """
        Predict collective communication time using trained Random Forest model.

        Args:
            comm_name: Communication operation name (e.g., 'all_reduce', 'send_recv')
            volume_bytes: Communication volume in bytes
            tp_size: Tensor parallel size (num_workers)

        Returns:
            Communication time in milliseconds, or None if model not found
        """
        if comm_name not in self.collective_models:
            raise ValueError(f"No model for {comm_name}")

        model = self.collective_models[comm_name]

        # Prepare features: [size, num_workers]
        features = np.array([[volume_bytes, tp_size]])

        # Predict time in milliseconds
        predicted_time_ms = float(model.predict(features)[0])

        return predicted_time_ms

    def find_nearest_gemm_config(self, M: int, N: int, K: int) -> Optional[Dict[str, Any]]:
        """
        Find the nearest GEMM configuration in the dataset.

        Args:
            M, N, K: Matrix dimensions

        Returns:
            Dictionary with tile_M, tile_N, tile_K, cta_count, is_split_k
        """
        if 'gemm' not in self.datasets or len(self.datasets['gemm']) == 0:
            return None

        df = self.datasets['gemm']

        # Calculate Euclidean distance in MNK space
        distances = np.sqrt(
            (df['M'] - M) ** 2 +
            (df['N'] - N) ** 2 +
            (df['K'] - K) ** 2
        )

        # Find closest match
        nearest_idx = distances.idxmin()
        nearest_row = df.loc[nearest_idx]

        return {
            'tile_M': int(nearest_row['tile_M']),
            'tile_N': int(nearest_row['tile_N']),
            'tile_K': int(nearest_row['tile_K']),
            'cta_count': int(nearest_row['cta_count']),
            'is_split_k': bool(nearest_row['is_split_k']),
            'distance': distances[nearest_idx]
        }

    def generate_gemm_features(
        self,
        M: int,
        N: int,
        K: int,
        tile_M: int,
        tile_N: int,
        tile_K: int,
        cta_count: int,
        is_split_k: bool
    ) -> np.ndarray:
        """
        Generate features for GEMM operator using analytical model.

        Args:
            M, N, K: Matrix dimensions
            tile_M, tile_N, tile_K: Tile sizes
            cta_count: Number of CTAs
            is_split_k: Whether split-K is used

        Returns:
            Feature array of shape (11,)
        """
        # Create problem config
        problem = GemmProblemConfig(
            m=M,
            n=N,
            k=K,
            tile_m=tile_M,
            tile_n=tile_N,
            tile_k=tile_K,
            cta_count=cta_count,
            is_split_k=is_split_k,
            data_size_bytes=2  # bf16
        )

        # Select calculator based on architecture
        arch = self.hardware_spec.get('architecture', 'ampere').lower()
        if arch == 'hopper':
            calculator = gemm9_calculator
        else:  # ampere, ada
            calculator = gemm8_calculator

        # Calculate features using analytical model
        features = calculator(problem, self.hardware_spec_obj)

        # Extract features from the returned GemmFeatures object
        feature_array = np.array([
            features.tensor_pipe.all_ops,
            features.tensor_pipe.all_cycle,
            features.tensor_pipe.sm_max_ops,
            features.tensor_pipe.sm_max_cycle,
            features.memory_pipe.global_in_flight,
            features.memory_pipe.global_cycle,
            features.memory_pipe.local_cycle,
            features.memory_pipe.sm_max_in_flight,
            features.memory_pipe.sm_max_global_cycle,
            features.memory_pipe.sm_max_shared_cycle,
            features.memory_pipe.sm_max_local_cycle
        ], dtype=np.float32)

        return feature_array

    def generate_attn_features(
        self,
        bs: int,
        q_lengths: List[int],
        kv_lengths: List[int],
        nh: int,
        nkv: int,
        hd: int,
        attention_type: str
    ) -> np.ndarray:
        """
        Generate features for attention operator using analytical model.

        Args:
            bs: Batch size
            q_lengths: List of query lengths
            kv_lengths: List of KV lengths
            nh: Number of query heads
            nkv: Number of KV heads
            hd: Head dimension
            attention_type: Attention type (e.g., 'fa2_ragged', 'fa2_paged', 'fa3_ragged', 'fa3_paged')

        Returns:
            Feature array of shape (15,)
        """
        # Parse attention_type to determine calculator and layout
        attn_type_lower = attention_type.lower()

        # Determine calculator
        if 'fa2' in attn_type_lower:
            calculator = calculate_fa2_params
        elif 'fa3' in attn_type_lower:
            calculator = calculate_fa3_params
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Determine layout
        if 'paged' in attn_type_lower:
            layout = 'paged'
        elif 'ragged' in attn_type_lower:
            layout = 'ragged'
        else:
            raise ValueError(f"Unknown layout in attention type: {attention_type}")

        # Create problem config
        problem = FaProblemConfig(
            batch_size=bs,
            q_lengths=q_lengths,
            kv_lengths=kv_lengths,
            num_qo_heads=nh,
            num_kv_heads=nkv,
            head_dim=hd,
            layout=layout,
            data_size_q=2,  # bf16
            data_size_kv=2,
            data_size_o=2,
            causal=True
        )

        # Calculate features using analytical model
        features = calculator(problem, self.hardware_spec_obj)

        # Extract features from the returned FaFeatures object
        feature_array = np.array([
            features.tensor_pipe.all_ops,
            features.tensor_pipe.all_cycle,
            features.tensor_pipe.sm_max_ops,
            features.tensor_pipe.sm_max_cycle,
            features.xu_pipe.all_ops,
            features.xu_pipe.all_cycle,
            features.xu_pipe.sm_max_ops,
            features.xu_pipe.sm_max_cycle,
            features.memory_pipe.global_in_flight,
            features.memory_pipe.global_cycle,
            features.memory_pipe.local_cycle,
            features.memory_pipe.sm_max_in_flight,
            features.memory_pipe.sm_max_global_cycle,
            features.memory_pipe.sm_max_shared_cycle,
            features.memory_pipe.sm_max_local_cycle
        ], dtype=np.float32)

        return feature_array

    def generate_rmsnorm_features(self, seq: int, dim: int) -> np.ndarray:
        """
        Generate features for RMSNorm operator using analytical model.

        Args:
            seq: Sequence length (number of tokens)
            dim: Hidden dimension

        Returns:
            Feature array of shape (15,)
        """
        # Create problem config
        problem = RmsNormProblemConfig(
            batch_size=seq,
            dim=dim,
            dtype_size=2  # bf16
        )

        # Calculate features using analytical model
        features = rmsnorm_calculator(problem, self.hardware_spec_obj)

        # Extract features from the returned RmsNormFeatures object
        feature_array = np.array([
            features.fma_pipe.all_ops,
            features.fma_pipe.all_cycle,
            features.fma_pipe.sm_max_ops,
            features.fma_pipe.sm_max_cycle,
            features.xu_pipe.all_ops,
            features.xu_pipe.all_cycle,
            features.xu_pipe.sm_max_ops,
            features.xu_pipe.sm_max_cycle,
            features.memory_pipe.global_in_flight,
            features.memory_pipe.global_cycle,
            features.memory_pipe.local_cycle,
            features.memory_pipe.sm_max_in_flight,
            features.memory_pipe.sm_max_global_cycle,
            features.memory_pipe.sm_max_shared_cycle,
            features.memory_pipe.sm_max_local_cycle
        ], dtype=np.float32)

        return feature_array

    def generate_siluandmul_features(self, seq: int, dim: int) -> np.ndarray:
        """
        Generate features for SiLUAndMul operator using analytical model.

        Args:
            seq: Sequence length (number of tokens)
            dim: Hidden dimension

        Returns:
            Feature array of shape (15,)
        """
        # Create problem config
        problem = SiluMulProblemConfig(
            seq_len=seq,
            dim=dim,
            dtype_size=2  # bf16
        )

        # Calculate features using analytical model
        features = silu_mul_calculator(problem, self.hardware_spec_obj)

        # Extract features from the returned SiluMulFeatures object
        feature_array = np.array([
            features.fma_pipe.all_ops,
            features.fma_pipe.all_cycle,
            features.fma_pipe.sm_max_ops,
            features.fma_pipe.sm_max_cycle,
            features.xu_pipe.all_ops,
            features.xu_pipe.all_cycle,
            features.xu_pipe.sm_max_ops,
            features.xu_pipe.sm_max_cycle,
            features.memory_pipe.global_in_flight,
            features.memory_pipe.global_cycle,
            features.memory_pipe.local_cycle,
            features.memory_pipe.sm_max_in_flight,
            features.memory_pipe.sm_max_global_cycle,
            features.memory_pipe.sm_max_shared_cycle,
            features.memory_pipe.sm_max_local_cycle
        ], dtype=np.float32)

        return feature_array

    def predict_operator_duration(
        self,
        operator_type: str,
        features: np.ndarray,
        sum_all_cycle: float
    ) -> float:
        """
        Predict operator duration using MLP model.

        Args:
            operator_type: Type of operator ('gemm', 'attn', 'rmsnorm', 'siluandmul')
            features: Input features for the MLP
            sum_all_cycle: Sum of relevant cycle counts for duration calculation

        Returns:
            Duration in milliseconds
        """
        if operator_type not in self.models:
            raise ValueError(f"Model not loaded for operator type: {operator_type}")

        model = self.models[operator_type]

        # Apply log1p transformation
        features_log = np.log1p(features)

        # Convert to tensor
        features_tensor = torch.FloatTensor(features_log).unsqueeze(0).to(self.device)

        # Predict overall_perf
        with torch.no_grad():
            overall_perf = model(features_tensor).item()

        # Convert to duration: duration = sum_all_cycle / (overall_perf * sm_freq)
        sm_freq = self.hardware_spec['sm_freq']  # MHz
        duration_ms = sum_all_cycle / (overall_perf * sm_freq)/1e3

        return duration_ms

    def predict_workload(self, workload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict durations for entire workload.

        Args:
            workload: Workload specification from workload.json

        Returns:
            Dictionary with duration predictions
        """
        results = {
            "model": workload["model"],
            "hardware": self.hardware_spec['name'],
            "tp_size": workload["tp_size"],
            "pp_size": workload["pp_size"],
            "iterations": []
        }

        total_duration = 0.0
        prefill_duration = 0.0
        decode_durations = []

        # Process each iteration
        for iteration in workload["iterations"]:
            iter_num = iteration["iteration"]
            phase = iteration["phase"]
            operators = iteration["operators"]

            print(f"\nProcessing iteration {iter_num} ({phase})...")

            iter_result = {
                "iteration": iter_num,
                "phase": phase,
                "operator_durations": {},
                "total_duration_ms": 0.0
            }

            iter_duration = 0.0

            # Process GEMM operators
            if "gemm" in operators:
                gemm_durations = {}
                for gemm_op in operators["gemm"]:
                    name = gemm_op["name"]
                    M = gemm_op["M"]
                    N = gemm_op["N"]
                    K = gemm_op["K"]
                    count = gemm_op["count"]

                    # Find nearest config for missing parameters
                    config = self.find_nearest_gemm_config(M, N, K)
                    if config is None:
                        print(f"  Warning: Could not find config for {name}, skipping")
                        continue

                    # Generate features
                    features = self.generate_gemm_features(
                        M, N, K,
                        config['tile_M'], config['tile_N'], config['tile_K'],
                        config['cta_count'], config['is_split_k']
                    )

                    # sum_all_cycle for GEMM is tensor_all_cycle
                    sum_all_cycle = float(features[1])  # tensor_all_cycle is index 1

                    # Predict duration
                    duration = self.predict_operator_duration('gemm', features, sum_all_cycle)
                    total_op_duration = duration * count

                    gemm_durations[name] = {
                        "single_duration_ms": float(round(duration, 6)),
                        "count": int(count),
                        "total_duration_ms": float(round(total_op_duration, 6))
                    }

                    iter_duration += total_op_duration

                iter_result["operator_durations"]["gemm"] = gemm_durations

            # Process Attention operator
            if "attention" in operators:
                attn = operators["attention"]
                bs = attn["bs"]
                q_lengths = attn["q_lengths"]
                kv_lengths = attn["kv_lengths"]
                nh = attn["nh"]
                nkv = attn["nkv"]
                hd = attn["hd"]
                attention_type = attn["attention_type"]
                count = attn["count"]

                features = self.generate_attn_features(
                    bs, q_lengths, kv_lengths, nh, nkv, hd, attention_type
                )

                # sum_all_cycle for attention is tensor_all_cycle
                sum_all_cycle = float(features[1])  # tensor_all_cycle

                duration = self.predict_operator_duration('attn', features, sum_all_cycle)
                total_op_duration = duration * count

                iter_result["operator_durations"]["attention"] = {
                    "single_duration_ms": float(round(duration, 6)),
                    "count": int(count),
                    "total_duration_ms": float(round(total_op_duration, 6))
                }

                iter_duration += total_op_duration

            # Process RMSNorm operator
            if "rmsnorm" in operators:
                rmsnorm = operators["rmsnorm"]
                seq = rmsnorm["seq"]
                dim = rmsnorm["dim"]
                count = rmsnorm["count"]

                features = self.generate_rmsnorm_features(seq, dim)

                # sum_all_cycle for rmsnorm is fma_all_cycle + xu_all_cycle
                sum_all_cycle = float(features[1] + features[5])  # fma_all_cycle + xu_all_cycle

                duration = self.predict_operator_duration('rmsnorm', features, sum_all_cycle)
                total_op_duration = duration * count

                iter_result["operator_durations"]["rmsnorm"] = {
                    "single_duration_ms": float(round(duration, 6)),
                    "count": int(count),
                    "total_duration_ms": float(round(total_op_duration, 6))
                }

                iter_duration += total_op_duration

            # Process SiLUAndMul operator
            if "siluandmul" in operators:
                siluandmul = operators["siluandmul"]
                seq = siluandmul["seq"]
                dim = siluandmul["dim"]
                count = siluandmul["count"]

                features = self.generate_siluandmul_features(seq, dim)

                # sum_all_cycle for siluandmul is fma_all_cycle + xu_all_cycle
                sum_all_cycle = float(features[1] + features[5])

                duration = self.predict_operator_duration('siluandmul', features, sum_all_cycle)
                total_op_duration = duration * count

                iter_result["operator_durations"]["siluandmul"] = {
                    "single_duration_ms": float(round(duration, 6)),
                    "count": int(count),
                    "total_duration_ms": float(round(total_op_duration, 6))
                }

                iter_duration += total_op_duration

            # Process Communication operators
            if "communication" in operators and len(operators["communication"]) > 0:
                comm_durations = {}
                for comm_op in operators["communication"]:
                    name = comm_op["name"]
                    comm_type = comm_op["type"]
                    volume_bytes = comm_op["volume_bytes"]
                    count = comm_op["count"]

                    # Determine which dataset to use based on operation name
                    # 'all_reduce' uses all_reduce dataset, 'send_recv' or p2p uses send_recv dataset
                    if 'all_reduce' in name.lower():
                        dataset_name = 'all_reduce'
                    elif 'send' in name.lower() or 'recv' in name.lower() or comm_type == 'p2p':
                        dataset_name = 'send_recv'
                    else:
                        # Default to all_reduce for collective operations
                        dataset_name = 'all_reduce'

                    # Get tp_size from workload
                    tp_size = workload.get('tp_size', 1)

                    # Interpolate communication time from dataset
                    try:
                        duration = self.interpolate_collective_time(dataset_name, volume_bytes, tp_size)

                        if duration is None:
                            print(f"  ERROR: Could not interpolate time for {name}")
                            duration = 0.0
                    except Exception as e:
                        print(f"  ERROR: Failed to interpolate time for {name}: {e}")
                        duration = 0.0

                    total_op_duration = duration * count

                    comm_durations[name] = {
                        "single_duration_ms": float(round(duration, 6)),
                        "count": int(count),
                        "total_duration_ms": float(round(total_op_duration, 6))
                    }

                    iter_duration += total_op_duration

                iter_result["operator_durations"]["communication"] = comm_durations

            iter_result["total_duration_ms"] = float(round(iter_duration, 6))
            results["iterations"].append(iter_result)

            total_duration += iter_duration

            if phase == "prefill":
                prefill_duration = iter_duration
            else:
                decode_durations.append(iter_duration)

        # Compute summary statistics
        results["summary"] = {
            "prefill_duration_ms": float(round(prefill_duration, 6)),
            "decode_avg_duration_ms": float(round(np.mean(decode_durations), 6)) if decode_durations else 0.0,
            "decode_total_duration_ms": float(round(sum(decode_durations), 6)),
            "total_duration_ms": float(round(total_duration, 6))
        }

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict operator durations for transformer workloads",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--workload',
        type=str,
        required=True,
        help='Path to workload JSON file'
    )

    parser.add_argument(
        '--hardware',
        type=str,
        required=True,
        help='Hardware name (e.g., "A100")'
    )

    parser.add_argument(
        '--model_dir',
        type=str,
        default='mlp_models',
        help='Directory containing trained MLP models (default: mlp_models)'
    )

    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='dataset',
        help='Directory containing operator datasets (default: dataset)'
    )

    parser.add_argument(
        '--hardware_dir',
        type=str,
        default='hardware',
        help='Directory containing hardware specs (default: hardware)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Output JSON file path (default: predictions.json)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on (default: cpu)'
    )

    parser.add_argument(
        '--collective_hardware',
        type=str,
        default=None,
        help='Hardware name for collective communication dataset (e.g., "A100", "H20")'
    )

    args = parser.parse_args()

    # Load workload
    print(f"Loading workload from: {args.workload}")
    with open(args.workload, 'r') as f:
        workload = json.load(f)

    # Create predictor
    predictor = DurationPredictor(
        hardware_name=args.hardware,
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        hardware_dir=args.hardware_dir,
        device=args.device,
        collective_hardware_name=args.collective_hardware
    )

    # Predict durations
    print("\n" + "=" * 80)
    print("Starting duration prediction...")
    print("=" * 80)

    results = predictor.predict_workload(workload)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("Prediction Summary:")
    print("=" * 80)
    print(f"  Prefill duration: {results['summary']['prefill_duration_ms']:.3f} ms")
    print(f"  Decode avg duration: {results['summary']['decode_avg_duration_ms']:.3f} ms")
    print(f"  Total duration: {results['summary']['total_duration_ms']:.3f} ms")
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
