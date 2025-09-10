#!/usr/bin/env python3
"""
Compare prediction results with real measurements.
Selects the best matching run from 5 repetitions by minimizing MAPE.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


def parse_pred_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse prediction filename to extract metadata.
    Format: {MODEL}_{WORKLOAD}_{FA}_tp{TP}_pp{PP}_{HARDWARE}.json
    Example: Qwen2.5-14B_arxiv_8_fa2_tp1_pp1_A100.json
    Or with suffix: Qwen2.5-14B_arxiv_8_fa2_tp1_pp1_A100_rf.json
    """
    # Remove .json extension
    name = filename.replace('.json', '')

    # Remove suffix like _rf, _hybrid, etc. if present
    if name.endswith('_rf') or name.endswith('_linear') or name.endswith('_roofline') or name.endswith('_neusight'):
        name = '_'.join(name.split('_')[:-1])

    # Split by underscore
    parts = name.split('_')

    if len(parts) < 6:
        return None

    # Extract components
    # Model: first parts until we hit a workload pattern (arxiv or splitwise)
    model_parts = []
    workload_parts = []
    fa_parts = []
    tp_size = None
    pp_size = None
    idx = 0

    # Find model parts (e.g., "Qwen2.5-14B" or could be multiple parts)
    while idx < len(parts) and parts[idx] not in ['arxiv', 'splitwise']:
        model_parts.append(parts[idx])
        idx += 1

    # Find workload (e.g., "arxiv_8" or "splitwise_32")
    if idx < len(parts):
        workload_parts.append(parts[idx])  # arxiv or splitwise
        idx += 1
        if idx < len(parts) and parts[idx].isdigit():
            workload_parts.append(parts[idx])  # number
            idx += 1

    # Find FA version (e.g., "fa2" or "fa3")
    if idx < len(parts) and parts[idx].startswith('fa'):
        fa_parts.append(parts[idx])
        idx += 1

    # Extract tp and pp
    while idx < len(parts) and (parts[idx].startswith('tp') or parts[idx].startswith('pp')):
        token = parts[idx]
        if token.startswith('tp'):
            tp_size = token[2:]  # Extract number after 'tp'
        elif token.startswith('pp'):
            pp_size = token[2:]
        idx += 1

    # Remaining parts are hardware
    hardware_parts = parts[idx:]

    model = '_'.join(model_parts)
    workload = '_'.join(workload_parts)
    fa_version = '_'.join(fa_parts) if fa_parts else 'unknown'
    hardware = '_'.join(hardware_parts)

    return {
        'model': model,
        'workload': workload,
        'fa_version': fa_version,
        'hardware': hardware,
        'tp_size': tp_size if tp_size else '1',
        'pp_size': pp_size if pp_size else '1'
    }


def load_pred_data(pred_file: Path) -> Optional[Dict]:
    """Load prediction data from JSON file."""
    try:
        with open(pred_file, 'r') as f:
            data = json.load(f)

        summary = data.get('summary', {})
        return {
            'prefill_ms': summary.get('prefill_duration_ms', 0),
            'decode_ms': summary.get('decode_avg_duration_ms', 0),
            'total_ms': summary.get('total_duration_ms', 0)
        }
    except Exception as e:
        print(f"Error loading {pred_file}: {e}")
        return None


def load_real_data(real_file: Path, run_name_pattern: str) -> List[Dict]:
    """
    Load real measurement data from JSONL file.
    Returns list of all runs matching the pattern.
    """
    results = []

    try:
        with open(real_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue

                data = json.loads(line)
                run_name = data.get('run_name', '')

                # Match run_name with pattern (e.g., "Qwen2.5-14B-arxiv_8")
                if run_name_pattern in run_name or run_name in run_name_pattern:
                    # Convert seconds to milliseconds
                    results.append({
                        'prefill_ms': data.get('prefill_latency', 0) * 1000,
                        'decode_ms': data.get('median_decode_latency', 0) * 1000,
                        'total_ms': data.get('total_latency', 0) * 1000,
                        'iteration': data.get('iteration', -1)
                    })
    except Exception as e:
        print(f"Error loading {real_file}: {e}")
        return []

    return results


def calculate_mape(pred: float, real: float) -> float:
    """Calculate Mean Absolute Percentage Error."""
    if real == 0:
        return 0.0
    return abs(pred - real) / real * 100


def find_best_match(pred_data: Dict, real_runs: List[Dict]) -> Tuple[Optional[Dict], float, float, float, float]:
    """
    Find the real run with minimum average MAPE.
    Returns: (best_run, mape_prefill, mape_decode, mape_total, avg_mape)
    """
    best_run = None
    best_avg_mape = float('inf')
    best_mapes = (0, 0, 0)

    for run in real_runs:
        # Calculate MAPE for each metric
        mape_prefill = calculate_mape(pred_data['prefill_ms'], run['prefill_ms'])
        mape_decode = calculate_mape(pred_data['decode_ms'], run['decode_ms'])
        mape_total = calculate_mape(pred_data['total_ms'], run['total_ms'])

        # Average MAPE
        avg_mape = (mape_prefill + mape_decode + mape_total) / 3

        if avg_mape < best_avg_mape:
            best_avg_mape = avg_mape
            best_run = run
            best_mapes = (mape_prefill, mape_decode, mape_total)

    return best_run, best_mapes[0], best_mapes[1], best_mapes[2], best_avg_mape


def process_pred_directory(pred_dir: Path, real_dir: Path, output_csv: Path):
    """Process a single prediction directory and generate comparison CSV."""
    results = []

    # Process all prediction files
    pred_files = sorted(pred_dir.glob('*.json'))
    print(f"\nFound {len(pred_files)} prediction files in {pred_dir.name}")

    for pred_file in pred_files:
        print(f"\nProcessing: {pred_file.name}")

        # Parse filename
        metadata = parse_pred_filename(pred_file.name)
        if not metadata:
            print(f"  Warning: Could not parse filename: {pred_file.name}")
            continue

        print(
            f"  Model: {metadata['model']}, Workload: {metadata['workload']}, "
            f"FA: {metadata['fa_version']}, Hardware: {metadata['hardware']}, "
            f"TP: {metadata['tp_size']}, PP: {metadata['pp_size']}"
        )

        # Load prediction data
        pred_data = load_pred_data(pred_file)
        if not pred_data:
            print(f"  Warning: Could not load pred data")
            continue

        # Find matching real file
        # For tp1, use e2e_{hardware}.jsonl
        # For tp2/tp4, use e2e_tp{N}_{hardware}.jsonl
        tp_size = metadata['tp_size']
        if tp_size == '1':
            real_file = real_dir / f"e2e_{metadata['hardware']}.jsonl"
        else:
            real_file = real_dir / f"e2e_tp{tp_size}_{metadata['hardware']}.jsonl"

        if not real_file.exists():
            print(f"  Warning: Real file not found: {real_file.name}")
            continue

        # Create run_name pattern for matching
        # Pattern: "Model-workload" (e.g., "Qwen2.5-14B-arxiv_8", "Qwen3-32B-arxiv_16", "Llama-3.1-70B-arxiv_16")
        # Handle different naming conventions:
        # - Qwen2.5-14B -> Qwen2.5-14B
        # - Qwen3_32B -> Qwen3-32B
        # - Llama_3.1_70B -> Llama-3.1-70B
        model_normalized = metadata['model'].replace('_', '-')
        workload_normalized = metadata['workload']
        run_name_pattern = f"{model_normalized}-{workload_normalized}"

        print(f"  Looking for run_name pattern: {run_name_pattern}")

        # Load real data
        real_runs = load_real_data(real_file, run_name_pattern)
        if not real_runs:
            print(f"  Warning: No matching real runs found")
            continue

        print(f"  Found {len(real_runs)} matching real runs")

        # Find best match
        best_run, mape_prefill, mape_decode, mape_total, avg_mape = find_best_match(pred_data, real_runs)

        if best_run:
            results.append({
                'Hardware': metadata['hardware'],
                'TP': metadata['tp_size'],
                'PP': metadata['pp_size'],
                'Model': metadata['model'],
                'Workload': metadata['workload'],
                'FA_Version': metadata['fa_version'],
                'pred_prefill_ms': round(pred_data['prefill_ms'], 3),
                'real_prefill_ms': round(best_run['prefill_ms'], 3),
                'MAPE_prefill(%)': round(mape_prefill, 2),
                'pred_decode_ms': round(pred_data['decode_ms'], 3),
                'real_decode_ms': round(best_run['decode_ms'], 3),
                'MAPE_decode(%)': round(mape_decode, 2),
                'pred_total_ms': round(pred_data['total_ms'], 3),
                'real_total_ms': round(best_run['total_ms'], 3),
                'MAPE_total(%)': round(mape_total, 2),
                'avg_MAPE(%)': round(avg_mape, 2)
            })
            print(f"  Best match - Avg MAPE: {avg_mape:.2f}%")

    # Sort results
    results.sort(key=lambda x: (x['Hardware'], x['Model'], x['Workload']))

    # Write to CSV
    if results:
        fieldnames = [
            'Hardware', 'TP', 'PP', 'Model', 'Workload', 'FA_Version',
            'pred_prefill_ms', 'real_prefill_ms', 'MAPE_prefill(%)',
            'pred_decode_ms', 'real_decode_ms', 'MAPE_decode(%)',
            'pred_total_ms', 'real_total_ms', 'MAPE_total(%)',
            'avg_MAPE(%)'
        ]

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n{'='*80}")
        print(f"Successfully generated comparison CSV for {pred_dir.name}")
        print(f"Total comparisons: {len(results)}")
        print(f"Output: {output_csv}")
        print(f"{'='*80}")
    else:
        print(f"\nNo results generated for {pred_dir.name}!")

    return len(results)


def main():
    e2e_dir = Path('e2e')
    real_dir = e2e_dir / 'real'

    # Define prediction directories
    pred_methods = ['pipeweave_pred', 'linear_pred', 'neusight_pred', 'roofline_pred','habitat_pred']

    total_comparisons = 0

    for method in pred_methods:
        pred_dir = e2e_dir / method

        # Check if directory exists
        if not pred_dir.exists():
            print(f"\n{'='*80}")
            print(f"Warning: {method} directory not found, skipping...")
            print(f"{'='*80}")
            continue

        # Generate output CSV name
        output_csv = e2e_dir / f"{method.replace('_pred', '')}_comparison.csv"

        print(f"\n{'='*80}")
        print(f"Processing {method}...")
        print(f"{'='*80}")

        # Process this prediction directory
        count = process_pred_directory(pred_dir, real_dir, output_csv)
        total_comparisons += count

    # Summary
    print(f"\n{'='*80}")
    print(f"ALL DONE!")
    print(f"Total comparisons across all methods: {total_comparisons}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
