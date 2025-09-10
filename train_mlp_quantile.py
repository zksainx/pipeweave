#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified MLP model for operator performance prediction using Quantile Loss
Supports: GEMM, Attention, RMSNorm, SiLU*Mul, MOE operators
Target: overall_perf upper bound prediction (configurable quantile)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
from datetime import datetime

from mlp_model import MLP

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ============================================================
# FEATURE DEFINITIONS FOR EACH OPERATOR TYPE
# ============================================================

GEMM_FEATURES = [
    'tensor_all_ops', 'tensor_all_cycle', 'tensor_sm_max_ops', 'tensor_sm_max_cycle',
    'global_in_flight', 'global_cycle', 'local_cycle',
    'sm_max_in_flight', 'sm_max_global_cycle', 'sm_max_shared_cycle', 'sm_max_local_cycle',
]

ATTN_FEATURES = [
    'tensor_all_ops', 'tensor_all_cycle', 'tensor_sm_max_ops', 'tensor_sm_max_cycle',
    'xu_all_ops', 'xu_all_cycle', 'xu_sm_max_ops', 'xu_sm_max_cycle',
    'global_in_flight', 'global_cycle', 'local_cycle',
    'sm_max_in_flight', 'sm_max_global_cycle', 'sm_max_shared_cycle', 'sm_max_local_cycle',
]

EW_FEATURES = [
    'fma_all_ops', 'fma_all_cycle', 'fma_sm_max_ops', 'fma_sm_max_cycle',
    'xu_all_ops', 'xu_all_cycle', 'xu_sm_max_ops', 'xu_sm_max_cycle',
    'global_in_flight', 'global_cycle', 'local_cycle',
    'sm_max_in_flight', 'sm_max_global_cycle', 'sm_max_shared_cycle', 'sm_max_local_cycle',
]


# ============================================================
# AUTOMATIC CONFIGURATION BASED ON OPERATOR TYPE
# ============================================================

# Validate operator type
VALID_OPERATORS = ['gemm', 'attn', 'moe', 'gemm_fp8']
OPERATOR_TYPE = VALID_OPERATORS[2]  # Default: moe

if OPERATOR_TYPE not in VALID_OPERATORS:
    raise ValueError(f"Invalid OPERATOR_TYPE '{OPERATOR_TYPE}'. Must be one of: {VALID_OPERATORS}")

# Select features based on operator type
FEATURES_MAP = {
    'gemm_fp8': GEMM_FEATURES,
    'gemm': GEMM_FEATURES,
    'attn': ATTN_FEATURES,
    'rmsnorm': EW_FEATURES,
    'siluandmul': EW_FEATURES,
    'moe': GEMM_FEATURES
}
FEATURES = FEATURES_MAP[OPERATOR_TYPE]

# Set data path (only train data)
TRAIN_PATH = f'./dataset/{OPERATOR_TYPE}_train.csv'

# Target variable
TARGET = 'overall_perf'

# ============================================================
# QUANTILE LOSS CONFIGURATION
# ============================================================

# Quantile parameter for loss function
# 0.5 = median, 0.75 = 75th percentile, 0.9 = 90th percentile, etc.
QUANTILE = 0.8

# ============================================================
# PERFORMANCE ANALYSIS CONFIGURATION
# ============================================================

# Threshold for identifying underperforming kernels
# perf_diff = actual_perf - predicted_perf
# Kernels with perf_diff < THRESHOLD are considered underperforming
UNDERPERFORM_THRESHOLD = 0.0  # Default: 0.0 (actual < predicted)
                               # Set to -0.05 to catch only kernels 5% below prediction
                               # Set to 0.05 to include kernels slightly above prediction

# Top-K kernels to save per hardware type
TOPK_KERNELS = None  # Number of worst-performing kernels to save per hardware
                    # Set to None to save all underperforming kernels

# ============================================================
# OUTPUT PATHS
# ============================================================

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join('./mlp_models_quantile', OPERATOR_TYPE, timestamp)
MODEL_PATH = os.path.join(OUTPUT_DIR, f'{OPERATOR_TYPE}_mlp_model.pth')
METADATA_PATH = os.path.join(OUTPUT_DIR, 'metadata.json')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'kernel_reports')

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Device configuration - prioritize CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("=" * 60)
print("QUANTILE LOSS MLP TRAINING SYSTEM")
print("=" * 60)
print(f"Operator type: {OPERATOR_TYPE.upper()}")
print(f"Number of features: {len(FEATURES)}")
print(f"Using device: {device}")
print(f"Loss function: Quantile Loss (q={QUANTILE})")
print(f"Underperform threshold: {UNDERPERFORM_THRESHOLD}")
print(f"Top-K kernels per hardware: {TOPK_KERNELS if TOPK_KERNELS else 'All'}")
print("=" * 60)


def load_data():
    """Load training dataset only"""
    print("Loading data...")

    # Check file existence
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Train dataset size: {train_df.shape}")

    # Print hardware information
    if 'hardware' in train_df.columns:
        hardware_types = train_df['hardware'].unique()
        hardware_counts = train_df['hardware'].value_counts()
        print(f"Hardware types: {hardware_types}")
        for hw_type in hardware_types:
            print(f"  {hw_type}: {hardware_counts[hw_type]}")

    return train_df


def preprocess_data(train_df, features=FEATURES, target=TARGET):
    """
    Preprocess training data (No Validation Split)
    Only applies log1p transformation without standardization
    """
    print("Preprocessing data...")

    print(f"Features: {features}")
    print(f"Target: {target}")

    # Process train data
    X_train = train_df[features].values.astype(np.float32)
    y_train = train_df[target].values.astype(np.float32)

    # Log transform features for better learning
    # Apply log1p to handle potential zeros and large values better
    X_train = np.log1p(X_train)
    # y is already in 0-1 range, no transformation needed

    # Final training data feature statistics
    print(f"\nFinal training data feature ranges:")
    for i, feature in enumerate(features):
        print(f"  {feature}: [{X_train[:, i].min():.3f}, {X_train[:, i].max():.3f}]")
    print(f"  Target ({target}): [{y_train.min():.5f}, {y_train.max():.5f}]")

    return X_train, y_train, features


def create_data_loader(X_train, y_train, batch_size=1024):
    """Create PyTorch data loader for training"""

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)

    # Create dataset
    train_dataset = TensorDataset(X_train, y_train)

    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def train_model(model, train_loader, num_epochs, learning_rate=0.001):
    """Train the MLP model using Quantile Loss"""
    print(f"Training MLP model with Quantile Loss (q={QUANTILE})...")

    # Quantile Loss function - targets the specified percentile
    def quantile_loss(y_pred, y_true, quantile):
        """
        Quantile Loss for predicting upper bounds
        """
        errors = y_true - y_pred
        loss = torch.mean(torch.max((quantile - 1) * errors, quantile * errors))
        return loss

    criterion = lambda y_pred, y_true: quantile_loss(y_pred, y_true, quantile=QUANTILE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Training history
    train_losses = []
    best_train_loss = float('inf')

    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # Average loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Save best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
            }, MODEL_PATH)

        # Print progress
        if epoch % 50 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.6f}')

    # Load best model
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Training completed. Best training loss: {best_train_loss:.6f}")

    return model, train_losses


def generate_kernel_performance_report(model, train_df, features, threshold=0.0, topk=100):
    """
    Generate kernel performance analysis report
    Identify kernels where actual_perf < predicted_perf (underperforming)
    Based on analyze_kernel_performance.py logic

    Args:
        model: Trained MLP model
        train_df: Training dataframe
        features: List of feature names
        threshold: Performance difference threshold for identifying underperforming kernels
                  (perf_diff < threshold). Default: 0.0
        topk: Number of top worst-performing kernels to save per hardware.
              Set to None to save all. Default: 100
    """
    print("\n" + "="*60)
    print("GENERATING KERNEL PERFORMANCE REPORT")
    print("="*60)
    print(f"Threshold: {threshold}")
    print(f"Top-K: {topk if topk else 'All'}")
    print("="*60)

    # Preprocess features
    X = train_df[features].values.astype(np.float32)
    X = np.log1p(X)  # Same transformation as training

    # Get actual performance
    actual_perf = train_df[TARGET].values.astype(np.float32)

    # Predict using trained model
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy().flatten()
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 1e-9)

    # Calculate performance differences
    perf_diff = actual_perf - predictions  # Negative means actual < predicted
    rel_gap = (actual_perf - predictions) / (predictions + 1e-9)

    # Add predictions and analysis columns to dataframe
    analysis_df = train_df.copy()
    analysis_df['pred_perf'] = predictions
    analysis_df['perf_diff'] = perf_diff
    analysis_df['rel_gap'] = rel_gap

    # Identify underperforming kernels (actual < predicted)
    underperforming_mask = perf_diff < threshold
    underperforming_df = analysis_df[underperforming_mask].copy()

    print(f"Total samples: {len(analysis_df)}")
    print(f"Underperforming kernels (perf_diff < {threshold}): {len(underperforming_df)}")

    # Define key columns for reporting
    key_columns = []
    # if 'weight_type' in analysis_df.columns:
    #     key_columns.append('weight_type')

    # Add columns if they exist
    add_columns = ['weight_type','M' ,'E', 'topk', 'H', 'N', 'K', 'attention_type','bs','q_lengths','kv_lengths','nh','nkv','hd']
    for col in add_columns:
        if col in analysis_df.columns:
            key_columns.append(col)

    # Add performance columns
    if 'avg_duration' in analysis_df.columns:
        key_columns.append('avg_duration')

    key_columns.extend(['overall_perf', 'pred_perf', 'perf_diff', 'rel_gap'])

    # Generate reports by hardware
    if 'hardware' in analysis_df.columns:
        hardware_types = analysis_df['hardware'].unique()

        summary_data = []

        for hw_type in hardware_types:
            hw_mask = underperforming_df['hardware'] == hw_type
            hw_underperforming = underperforming_df[hw_mask]

            if len(hw_underperforming) == 0:
                print(f"\n{hw_type}: No underperforming kernels")
                continue

            # Sort by perf_diff (ascending, so worst cases first)
            hw_underperforming = hw_underperforming.sort_values('perf_diff', ascending=True)

            # Save top-k underperforming for this hardware
            if topk is None:
                topk_count = len(hw_underperforming)
                topk_hw = hw_underperforming
            else:
                topk_count = min(topk, len(hw_underperforming))
                topk_hw = hw_underperforming.head(topk_count)

            # Select columns that exist
            output_columns = [col for col in key_columns if col in topk_hw.columns]

            output_path = os.path.join(REPORT_DIR, f'topk_underperforming_{hw_type}.csv')
            topk_hw[output_columns].to_csv(output_path, index=False)

            print(f"\n{hw_type}: {len(hw_underperforming)} underperforming kernels")
            print(f"  Saved top-{topk_count} to: {output_path}")
            print(f"  Average perf_diff: {hw_underperforming['perf_diff'].mean():.6f}")
            print(f"  Min perf_diff: {hw_underperforming['perf_diff'].min():.6f}")
            print(f"  Average rel_gap: {hw_underperforming['rel_gap'].mean():.4f}")
            print(f"  Min rel_gap: {hw_underperforming['rel_gap'].min():.4f}")

            # Add to summary
            summary_data.append({
                'hardware': hw_type,
                'total_underperforming': len(hw_underperforming),
                'avg_perf_diff': hw_underperforming['perf_diff'].mean(),
                'min_perf_diff': hw_underperforming['perf_diff'].min(),
                'avg_rel_gap': hw_underperforming['rel_gap'].mean(),
                'min_rel_gap': hw_underperforming['rel_gap'].min(),
                'topk_count': topk_count
            })

        # Save summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(REPORT_DIR, 'summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSummary saved to: {summary_path}")
    else:
        # No hardware column, save all underperforming kernels
        if len(underperforming_df) > 0:
            underperforming_df = underperforming_df.sort_values('perf_diff', ascending=True)

            if topk is None:
                topk_count = len(underperforming_df)
                topk_df = underperforming_df
            else:
                topk_count = min(topk, len(underperforming_df))
                topk_df = underperforming_df.head(topk_count)

            output_columns = [col for col in key_columns if col in topk_df.columns]
            output_path = os.path.join(REPORT_DIR, 'topk_underperforming_all.csv')
            topk_df[output_columns].to_csv(output_path, index=False)
            print(f"\nSaved top-{topk_count} underperforming kernels to: {output_path}")

    print("="*60)


def save_metadata(features, train_df, model_params=None, threshold=0.0, topk=100):
    """Save comprehensive metadata about the model and training process"""
    print("Saving metadata...")

    # Calculate feature ranges from original data before transformation
    feature_ranges = {}
    for feature in features:
        train_values = train_df[feature].values
        feature_ranges[feature] = {
            'min': float(train_values.min()),
            'max': float(train_values.max()),
            'mean': float(train_values.mean()),
            'std': float(train_values.std())
        }

    # Prepare metadata dictionary
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'model_type': f'{OPERATOR_TYPE.upper()}_MLP_QUANTILE',
            'operator_type': OPERATOR_TYPE,
            'features': features,
            'target': TARGET,
            'num_features': len(features),
            'loss_function': f'Quantile Loss (q={QUANTILE})',
            'quantile': QUANTILE,
            'purpose': f'Upper bound prediction ({int(QUANTILE*100)}th percentile)',
            'model_parameters': model_params if model_params else {},
            'batch_norm_position': 'post_activation'
        },
        'data_info': {
            'train_samples': len(train_df),
            'train_path': TRAIN_PATH,
            'validation': 'No validation split - trained on full dataset'
        },
        'feature_ranges': feature_ranges,
        'preprocessing': {
            'transformation': 'log1p only (no standardization)',
            'note': 'This model does not use StandardScaler'
        },
        'kernel_reports': {
            'report_directory': REPORT_DIR,
            'description': 'Kernel performance analysis reports identifying underperforming kernels (actual < predicted)',
            'threshold': threshold,
            'topk_per_hardware': topk if topk else 'all',
            'note': f'Kernels with perf_diff < {threshold} are identified as underperforming',
            'files': 'See kernel_reports/ directory for CSV files'
        }
    }

    # Add hardware distribution info
    if 'hardware' in train_df.columns:
        train_hardware_counts = train_df['hardware'].value_counts().to_dict()
        metadata['data_info']['train_hardware_distribution'] = {k: int(v) for k, v in train_hardware_counts.items()}

    # Save metadata to JSON file
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved to: {METADATA_PATH}")
    return metadata


def main():
    """Main function"""
    # Output path information
    print("=" * 60)
    print(f"{OPERATOR_TYPE.upper()} MLP Training with Quantile Loss")
    print("=" * 60)
    print(f"Training data: {TRAIN_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model file: {MODEL_PATH}")
    print(f"Report directory: {REPORT_DIR}")
    print(f"Operator type: {OPERATOR_TYPE}")
    print(f"Number of features: {len(FEATURES)}")
    print(f"Loss function: Quantile Loss (q={QUANTILE})")
    print("=" * 60)

    # Check file existence
    print("File existence check:")
    print(f"Training data exists: {os.path.exists(TRAIN_PATH)}")
    print(f"Output directory exists: {os.path.exists(OUTPUT_DIR)}")
    print(f"Report directory exists: {os.path.exists(REPORT_DIR)}")
    print("=" * 60)

    # Load data
    train_df = load_data()

    # Preprocess data (no validation split)
    X_train, y_train, features = preprocess_data(train_df, FEATURES, TARGET)

    # Create data loader
    train_loader = create_data_loader(X_train, y_train, batch_size=1024)

    # Model configuration
    hidden_dims = [256, 128, 64]
    dropout_rate = 0.1

    # Create model
    model = MLP(
        input_dim=len(features),
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )

    model_params = {
        'input_dim': len(features),
        'hidden_dims': hidden_dims,
        'dropout_rate': dropout_rate,
        'total_parameters': sum(p.numel() for p in model.parameters())
    }
    print(f"Total parameters: {model_params['total_parameters']:,}")

    # Train model
    model, train_losses = train_model(
        model, train_loader, num_epochs=200, learning_rate=0.001
    )

    # Generate kernel performance analysis report
    generate_kernel_performance_report(
        model, train_df, features,
        threshold=UNDERPERFORM_THRESHOLD,
        topk=TOPK_KERNELS
    )

    # Save comprehensive metadata
    metadata = save_metadata(
        features, train_df,
        model_params=model_params,
        threshold=UNDERPERFORM_THRESHOLD,
        topk=TOPK_KERNELS
    )

    print(f"\nTraining and analysis completed.")
    print(f"All files saved to: {OUTPUT_DIR}")
    print(f"Kernel performance reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
