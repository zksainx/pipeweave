#!/usr/bin/env python3
"""
Compare vLLM prediction results with real measurements.
Only compares end-to-end (e2e) latency.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional


def parse_vllm_pred_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse vLLM prediction filename to extract metadata.
    Format: Llama-3.1-70B_vllm_{workload}_fa3_tp4_pp2_{hardware}[_suffix].json
    Example: Llama-3.1-70B_vllm_arxiv16_fa3_tp4_pp2_H800.json
    Or with suffix: Llama-3.1-70B_vllm_arxiv16_fa3_tp4_pp2_H800_linear.json
    """
    # Remove .json extension
    name = filename.replace('.json', '')

    # Remove suffix like _linear, _neusight, _roofline if present
    suffix = None
    if name.endswith('_linear'):
        suffix = 'linear'
        name = name[:-7]
    elif name.endswith('_neusight'):
        suffix = 'neusight'
        name = name[:-9]
    elif name.endswith('_roofline'):
        suffix = 'roofline'
        name = name[:-9]
    elif name.endswith('_habitat'):
        suffix = 'habitat'
        name = name[:-8]

    # Split by underscore
    parts = name.split('_')

    # Expected format: Llama-3.1-70B_vllm_{workload}_fa3_tp4_pp2_{hardware}
    # Find indices of key markers
    if 'vllm' not in parts:
        return None

    vllm_idx = parts.index('vllm')

    # Model is everything before 'vllm'
    model_parts = parts[:vllm_idx]
    model = '_'.join(model_parts)

    # Workload is after 'vllm' and before 'fa3'
    # Could be "arxiv16" or "splitwise64"
    workload_idx = vllm_idx + 1
    if workload_idx >= len(parts):
        return None
    workload = parts[workload_idx]

    # Hardware is the last part
    hardware = parts[-1]

    return {
        'model': model,
        'workload': workload,
        'hardware': hardware,
        'suffix': suffix
    }


def load_pred_data(pred_file: Path) -> Optional[float]:
    """
    Load prediction data from JSON file.
    Returns total_duration_ms (e2e latency in milliseconds).
    """
    try:
        with open(pred_file, 'r') as f:
            data = json.load(f)

        summary = data.get('summary', {})
        total_ms = summary.get('total_duration_ms', 0)
        return total_ms
    except Exception as e:
        print(f"Error loading {pred_file}: {e}")
        return None


def load_real_data(real_file: Path) -> Optional[float]:
    """
    Load real measurement data from JSON file.
    Returns avg_latency converted to milliseconds.
    """
    try:
        with open(real_file, 'r') as f:
            data = json.load(f)

        # avg_latency is in seconds, convert to milliseconds
        avg_latency_sec = data.get('avg_latency', 0)
        avg_latency_ms = avg_latency_sec * 1000
        return avg_latency_ms
    except Exception as e:
        print(f"Error loading {real_file}: {e}")
        return None


def calculate_mape(pred: float, real: float) -> float:
    """Calculate Mean Absolute Percentage Error."""
    if real == 0:
        return 0.0
    return abs(pred - real) / real * 100


def get_real_filename(workload: str, hardware: str) -> str:
    """
    Generate real data filename from workload and hardware.
    Format: vllm_{workload}_{hardware}_llama70B.json
    """
    return f"vllm_{workload}_{hardware}_llama70B.json"


def process_all_predictions(e2e_dir: Path) -> List[Dict]:
    """Process all vLLM predictions from all methods."""
    results = []

    # Define prediction directories and their method names
    pred_methods = {
        'pipeweave_pred': 'pipeweave',
        'linear_pred': 'linear',
        'neusight_pred': 'neusight',
        'roofline_pred': 'roofline',
        'habitat_pred': 'habitat'
    }

    real_dir = e2e_dir / 'real'

    for pred_dir_name, method_name in pred_methods.items():
        pred_dir = e2e_dir / pred_dir_name

        # Check if directory exists
        if not pred_dir.exists():
            print(f"\nWarning: {pred_dir_name} directory not found, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {method_name}...")
        print(f"{'='*80}")

        # Find all vLLM prediction files
        pred_files = sorted(pred_dir.glob('*vllm*.json'))
        print(f"Found {len(pred_files)} vLLM prediction files")

        for pred_file in pred_files:
            print(f"\nProcessing: {pred_file.name}")

            # Parse filename
            metadata = parse_vllm_pred_filename(pred_file.name)
            if not metadata:
                print(f"  Warning: Could not parse filename: {pred_file.name}")
                continue

            print(f"  Model: {metadata['model']}, Workload: {metadata['workload']}, "
                  f"Hardware: {metadata['hardware']}")

            # Load prediction data
            pred_e2e_ms = load_pred_data(pred_file)
            if pred_e2e_ms is None:
                print(f"  Warning: Could not load prediction data")
                continue

            # Find matching real file
            real_filename = get_real_filename(metadata['workload'], metadata['hardware'])
            real_file = real_dir / real_filename

            if not real_file.exists():
                print(f"  Warning: Real file not found: {real_filename}")
                continue

            # Load real data
            real_e2e_ms = load_real_data(real_file)
            if real_e2e_ms is None:
                print(f"  Warning: Could not load real data")
                continue

            # Calculate MAPE
            mape = calculate_mape(pred_e2e_ms, real_e2e_ms)

            results.append({
                'Method': method_name,
                'Hardware': metadata['hardware'],
                'Workload': metadata['workload'],
                'Model': metadata['model'],
                'pred_e2e_ms': round(pred_e2e_ms, 3),
                'real_e2e_ms': round(real_e2e_ms, 3),
                'MAPE_e2e(%)': round(mape, 2)
            })

            print(f"  Pred: {pred_e2e_ms:.3f}ms, Real: {real_e2e_ms:.3f}ms, "
                  f"MAPE: {mape:.2f}%")

    return results


def main():
    e2e_dir = Path('e2e')
    output_csv = e2e_dir / 'vllm_comparison.csv'

    print(f"{'='*80}")
    print("vLLM Prediction vs Real Comparison")
    print(f"{'='*80}")

    # Process all predictions
    results = process_all_predictions(e2e_dir)

    # Sort results by method, hardware, workload
    results.sort(key=lambda x: (x['Method'], x['Hardware'], x['Workload']))

    # Write to CSV
    if results:
        fieldnames = [
            'Method', 'Hardware', 'Workload', 'Model',
            'pred_e2e_ms', 'real_e2e_ms', 'MAPE_e2e(%)'
        ]

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'='*80}")
        print(f"Successfully generated vLLM comparison CSV")
        print(f"Total comparisons: {len(results)}")
        print(f"Output: {output_csv}")
        print(f"{'='*80}")

        # Print summary statistics
        print(f"\nSummary by Method:")
        methods = {}
        for result in results:
            method = result['Method']
            if method not in methods:
                methods[method] = []
            methods[method].append(result['MAPE_e2e(%)'])

        for method, mapes in sorted(methods.items()):
            avg_mape = sum(mapes) / len(mapes)
            print(f"  {method:12s}: Avg MAPE = {avg_mape:.2f}% ({len(mapes)} comparisons)")
    else:
        print(f"\nNo results generated!")


if __name__ == '__main__':
    main()
