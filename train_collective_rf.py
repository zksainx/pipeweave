#!/usr/bin/env python3
"""
Training script: Train a Random Forest model for each communication operator on each hardware
Input features: size, num_workers
Output target: time_stats.<operator>.mean
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path


def train_collective_operator(csv_path, output_dir):
    """
    Train a Random Forest model for a single communication operator

    Args:
        csv_path: Path to the CSV file
        output_dir: Output directory
    """
    # Read data
    print(f"Processing: {csv_path}")
    df = pd.read_csv(csv_path)

    # Get operator name (from filename)
    operator_name = Path(csv_path).stem  # e.g., 'all_reduce', 'send_recv'

    # Construct target column name
    target_col = f'time_stats.{operator_name}.mean'

    # Check if column exists
    if target_col not in df.columns:
        print(f"Warning: Column {target_col} not found in {csv_path}")
        return

    # Feature columns
    feature_cols = ['size', 'num_workers']

    # Check if feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols} in {csv_path}")
        return

    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values

    # Remove NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        print(f"Warning: No valid data in {csv_path}")
        return

    # Train Random Forest model (using all data)
    print(f"  Training Random Forest for {operator_name} with {len(X)} samples...")

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X, y)

    # Save model
    model_path = os.path.join(output_dir, f'{operator_name}_rf_model.pkl')
    joblib.dump(rf_model, model_path)
    print(f"  Model saved to: {model_path}")


def main():
    """Main function: Iterate through all hardware and operators, train models"""

    base_dir = '../pipeweave/dataset/collective'

    # Get all hardware directories
    hardware_dirs = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]

    print(f"Found {len(hardware_dirs)} hardware directories: {hardware_dirs}\n")

    # Iterate through each hardware
    for hardware in sorted(hardware_dirs):
        hardware_path = os.path.join(base_dir, hardware)
        print(f"\n{'='*60}")
        print(f"Processing Hardware: {hardware}")
        print(f"{'='*60}")

        # Get all CSV files under this hardware
        csv_files = [f for f in os.listdir(hardware_path) if f.endswith('.csv')]

        if not csv_files:
            print(f"No CSV files found in {hardware_path}")
            continue

        print(f"Found {len(csv_files)} operator CSV files: {csv_files}\n")

        # Train each operator
        for csv_file in sorted(csv_files):
            csv_path = os.path.join(hardware_path, csv_file)

            try:
                train_collective_operator(csv_path, hardware_path)
                print()
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
                import traceback
                traceback.print_exc()
                print()

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
