#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified MLP model for operator performance prediction
Supports: GEMM, Attention, RMSNorm, SiLU*Mul operators
Target: overall_perf prediction
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
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
VALID_OPERATORS = ['gemm', 'attn', 'rmsnorm', 'siluandmul','moe','gemm_fp8']
OPERATOR_TYPE = VALID_OPERATORS[5]

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

# Set data paths based on operator type
TRAIN_PATH = f'./dataset/{OPERATOR_TYPE}_train.csv'
TEST_PATH = f'./dataset/{OPERATOR_TYPE}_test.csv'

# Target variable
TARGET = 'overall_perf'

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join('./mlp_models', OPERATOR_TYPE, timestamp)
MODEL_PATH = os.path.join(OUTPUT_DIR, f'{OPERATOR_TYPE}_mlp_model.pth')
METADATA_PATH = os.path.join(OUTPUT_DIR, 'metadata.json')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device configuration - prioritize CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("=" * 60)
print("UNIFIED MLP TRAINING SYSTEM")
print("=" * 60)
print(f"Operator type: {OPERATOR_TYPE.upper()}")
print(f"Number of features: {len(FEATURES)}")
print(f"Using device: {device}")
print("=" * 60)


def load_data():
    """Load pre-split train and test datasets"""
    print("Loading data...")

    # Check file existence
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Testing data not found: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train dataset size: {train_df.shape}")
    print(f"Test dataset size: {test_df.shape}")

    # Print hardware information for both datasets
    if 'hardware' in train_df.columns:
        train_hardware_types = train_df['hardware'].unique()
        train_hardware_counts = train_df['hardware'].value_counts()
        print(f"Train hardware types: {train_hardware_types}")
        for hw_type in train_hardware_types:
            print(f"  {hw_type}: {train_hardware_counts[hw_type]}")

    if 'hardware' in test_df.columns:
        test_hardware_types = test_df['hardware'].unique()
        test_hardware_counts = test_df['hardware'].value_counts()
        print(f"Test hardware types: {test_hardware_types}")
        for hw_type in test_hardware_types:
            print(f"  {hw_type}: {test_hardware_counts[hw_type]}")

    return train_df, test_df


def preprocess_data(train_df, test_df, features=FEATURES, target=TARGET):
    """
    Preprocess pre-split train and test data (No Scaler Version)
    Only applies log1p transformation without standardization
    """
    print("Preprocessing data...")

    print(f"Features: {features}")
    print(f"Target: {target}")

    # Process train data
    X_train = train_df[features].values.astype(np.float32)
    y_train = train_df[target].values.astype(np.float32)

    # Process test data
    X_test = test_df[features].values.astype(np.float32)
    y_test = test_df[target].values.astype(np.float32)

    # Log transform features for better learning (due to large range differences)
    # Apply log1p to handle potential zeros and large values better
    X_train = np.log1p(X_train)
    X_test = np.log1p(X_test)
    # y is already in 0-1 range, no transformation needed

    # Create test data grouped by hardware for separate evaluation
    test_data_by_hardware = {}
    if 'hardware' in test_df.columns:
        hardware_types = test_df['hardware'].unique()

        for hw_type in hardware_types:
            hw_mask = test_df['hardware'] == hw_type
            hw_indices = np.where(hw_mask)[0]

            X_test_hw = X_test[hw_indices]
            y_test_hw = y_test[hw_indices]

            # Store test data by hardware for separate evaluation
            test_data_by_hardware[hw_type] = {
                'X_test': X_test_hw,
                'y_test': y_test_hw
            }

    # Split training data for validation (from train data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Final training data feature statistics
    print(f"\nFinal training data feature ranges:")
    for i, feature in enumerate(features):
        print(f"  {feature}: [{X_train[:, i].min():.3f}, {X_train[:, i].max():.3f}]")
    print(f"  Target ({target}): [{y_train.min():.5f}, {y_train.max():.5f}]")

    return X_train, X_val, X_test, y_train, y_val, y_test, features, test_data_by_hardware


def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=1024):
    """Create PyTorch data loaders"""

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    y_val = torch.FloatTensor(y_val).view(-1, 1)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, num_epochs, learning_rate=0.001):
    """Train the MLP model"""
    print("Training MLP model...")

    # MAPE Loss function (Mean Absolute Percentage Error) - relative error
    def mape_loss(y_pred, y_true):
        """Mean Absolute Percentage Error loss - relative error"""
        epsilon = 1e-9  # Small value to avoid division by zero
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon)))

    criterion = mape_loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=45, factor=0.6, min_lr=1e-6)

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 85

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

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, MODEL_PATH)
        else:
            patience_counter += 1

        # Print progress
        if epoch % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch}/{num_epochs}], '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, '
                  f'LR: {current_lr:.2e}')

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")

    return model, train_losses, val_losses


def evaluate_model(model, test_loader, features):
    """Evaluate model performance"""
    print("Evaluating model performance...")

    model.eval()
    predictions = []
    actuals = []
    test_features = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)

            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.cpu().numpy().flatten())
            test_features.extend(batch_X.cpu().numpy())

    predictions = np.array(predictions)
    predictions = np.maximum(predictions, 1e-9)  # Clip predictions to avoid division by zero
    actuals = np.array(actuals)
    test_features = np.array(test_features)

    # y values are in original scale (0-1), no inverse transform needed
    # predictions and actuals are already in correct scale

    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)

    # Calculate relative error
    rel_err = np.abs(predictions - actuals) / (actuals + 1e-8)
    mean_rel_err = np.mean(rel_err)
    median_rel_err = np.median(rel_err)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Relative Error: {mean_rel_err:.4f}")
    print(f"Median Relative Error: {median_rel_err:.4f}")

    # Calculate accuracy percentages
    accuracy_10 = np.mean(rel_err < 0.1) * 100
    accuracy_20 = np.mean(rel_err < 0.2) * 100
    accuracy_50 = np.mean(rel_err < 0.5) * 100

    print(f"Percentage with relative error <10%: {accuracy_10:.2f}%")
    print(f"Percentage with relative error <20%: {accuracy_20:.2f}%")
    print(f"Percentage with relative error <50%: {accuracy_50:.2f}%")

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_rel_err': mean_rel_err,
        'median_rel_err': median_rel_err,
        'accuracy_10': accuracy_10,
        'accuracy_20': accuracy_20,
        'accuracy_50': accuracy_50,
        'predictions': predictions,
        'actuals': actuals,
        'test_features': test_features,
    }


def evaluate_model_by_hardware(model, test_data_by_hardware):
    """Evaluate model performance separately for each hardware type"""
    print("\n" + "="*60)
    print("HARDWARE-SPECIFIC EVALUATION")
    print("="*60)

    hardware_results = {}

    for hw_type, test_data in test_data_by_hardware.items():
        print(f"\n--- Evaluating {hw_type} ---")

        # Convert to tensors
        X_test_hw = torch.FloatTensor(test_data['X_test'])
        y_test_hw = torch.FloatTensor(test_data['y_test'])

        # Create dataset and loader
        test_dataset_hw = TensorDataset(X_test_hw, y_test_hw)
        test_loader_hw = DataLoader(test_dataset_hw, batch_size=1024, shuffle=False)

        # Evaluate
        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader_hw:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)

                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(batch_y.cpu().numpy().flatten())

        predictions = np.array(predictions)
        predictions = np.maximum(predictions, 1e-9)
        actuals = np.array(actuals)

        # y values are in original scale (0-1), no inverse transform needed
        # predictions and actuals are already in correct scale

        # Calculate metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)

        # Calculate relative error
        rel_err = np.abs(predictions - actuals) / (actuals + 1e-8)
        mean_rel_err = np.mean(rel_err)
        median_rel_err = np.median(rel_err)

        # Calculate accuracy percentages
        accuracy_10 = np.mean(rel_err < 0.1) * 100
        accuracy_20 = np.mean(rel_err < 0.2) * 100
        accuracy_50 = np.mean(rel_err < 0.5) * 100

        print(f"{hw_type} Results:")
        print(f"  Test samples: {len(actuals)}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Mean Relative Error: {mean_rel_err:.4f}")
        print(f"  Median Relative Error: {median_rel_err:.4f}")
        print(f"  Accuracy <10%: {accuracy_10:.2f}%")
        print(f"  Accuracy <20%: {accuracy_20:.2f}%")
        print(f"  Accuracy <50%: {accuracy_50:.2f}%")

        hardware_results[hw_type] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_rel_err': mean_rel_err,
            'median_rel_err': median_rel_err,
            'accuracy_10': accuracy_10,
            'accuracy_20': accuracy_20,
            'accuracy_50': accuracy_50,
            'test_samples': len(actuals),
            'predictions': predictions,
            'actuals': actuals
        }

    return hardware_results


def analyze_errors(predictions, actuals, test_features, features, test_df, output_dir):
    """Analyze prediction errors and save worst cases to CSV"""
    print("Analyzing prediction errors...")

    # Calculate relative error
    rel_err = np.abs(predictions - actuals) / (actuals + 1e-8)

    # Create results DataFrame for analysis
    results_dict = {}
    for i, feature in enumerate(features):
        # Use log-transformed features as-is for analysis
        results_dict[feature] = test_features[:, i]

    results_dict['actual'] = actuals
    results_dict['predicted'] = predictions
    results_dict['rel_error'] = rel_err

    results_df = pd.DataFrame(results_dict)

    # Find the cases with largest errors
    worst_cases = results_df.sort_values('rel_error', ascending=False).head(20)
    print("\nTop 20 cases with largest errors:")

    # Display key features if they exist, otherwise show first few features
    display_cols = ['actual', 'predicted', 'rel_error']
    if len(features) > 0:
        # Show first 3-4 most important features
        key_features = features[:min(4, len(features))]
        display_cols = key_features + display_cols

    print(worst_cases[display_cols].round(4))

    # Analyze error patterns
    print(f"\nError analysis:")
    print(f"Mean relative error: {np.mean(rel_err):.4f}")
    print(f"Median relative error: {np.median(rel_err):.4f}")
    print(f"95th percentile relative error: {np.percentile(rel_err, 95):.4f}")
    print(f"Max relative error: {np.max(rel_err):.4f}")

    # Save top 100 worst cases to CSV with all original columns
    print(f"\nSaving top 100 worst cases to CSV...")

    # Get top 100 worst cases
    top_100_worst = results_df.sort_values('rel_error', ascending=False).head(100)

    # Get the indices of these worst cases in the original test_df
    # Since predictions, actuals, and test_df are all aligned by row order
    worst_indices = top_100_worst.index

    # Create a copy of the corresponding rows from original test_df
    worst_cases_full = test_df.iloc[worst_indices].copy()

    # Add prediction and error columns
    worst_cases_full['predicted'] = top_100_worst['predicted'].values
    worst_cases_full['rel_error'] = top_100_worst['rel_error'].values

    # Save to CSV
    csv_path = os.path.join(output_dir, 'worst_cases_top100.csv')
    worst_cases_full.to_csv(csv_path, index=False)
    print(f"Saved top 100 worst cases to: {csv_path}")

    return worst_cases, results_df


def save_metadata(features, train_df, test_df, overall_metrics, hardware_results=None, model_params=None):
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
            'model_type': f'{OPERATOR_TYPE.upper()}_MLP',
            'operator_type': OPERATOR_TYPE,
            'features': features,
            'target': TARGET,
            'num_features': len(features),
            'model_parameters': model_params if model_params else {},
            'batch_norm_position': 'post_activation'
        },
        'data_info': {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_path': TRAIN_PATH,
            'test_path': TEST_PATH
        },
        'feature_ranges': feature_ranges,
        'preprocessing': {
            'transformation': 'log1p only (no standardization)',
            'note': 'This model does not use StandardScaler'
        },
        'evaluation_results': {
            'overall': {
                'mae': float(overall_metrics['mae']),
                'rmse': float(overall_metrics['rmse']),
                'r2': float(overall_metrics['r2']),
                'mean_relative_error': float(overall_metrics['mean_rel_err']),
                'median_relative_error': float(overall_metrics['median_rel_err']),
                'accuracy_10_percent': float(overall_metrics['accuracy_10']),
                'accuracy_20_percent': float(overall_metrics['accuracy_20']),
                'accuracy_50_percent': float(overall_metrics['accuracy_50'])
            }
        }
    }

    # Add hardware-specific results if available
    if hardware_results:
        metadata['evaluation_results']['by_hardware'] = {}
        for hw_type, results in hardware_results.items():
            metadata['evaluation_results']['by_hardware'][hw_type] = {
                'test_samples': int(results['test_samples']),
                'mae': float(results['mae']),
                'rmse': float(results['rmse']),
                'r2': float(results['r2']),
                'mean_relative_error': float(results['mean_rel_err']),
                'median_relative_error': float(results['median_rel_err']),
                'accuracy_10_percent': float(results['accuracy_10']),
                'accuracy_20_percent': float(results['accuracy_20']),
                'accuracy_50_percent': float(results['accuracy_50'])
            }

    # Add hardware distribution info
    if 'hardware' in train_df.columns:
        train_hardware_counts = train_df['hardware'].value_counts().to_dict()
        metadata['data_info']['train_hardware_distribution'] = {k: int(v) for k, v in train_hardware_counts.items()}

    if 'hardware' in test_df.columns:
        test_hardware_counts = test_df['hardware'].value_counts().to_dict()
        metadata['data_info']['test_hardware_distribution'] = {k: int(v) for k, v in test_hardware_counts.items()}

    # Save metadata to JSON file
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved to: {METADATA_PATH}")
    return metadata


def main():
    """Main function"""
    # Output path information
    print("=" * 60)
    print(f"{OPERATOR_TYPE.upper()} MLP Training")
    print("=" * 60)
    print(f"Training data: {TRAIN_PATH}")
    print(f"Testing data: {TEST_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model file: {MODEL_PATH}")
    print(f"Operator type: {OPERATOR_TYPE}")
    print(f"Number of features: {len(FEATURES)}")
    print("=" * 60)

    # Check file existence
    print("File existence check:")
    print(f"Training data exists: {os.path.exists(TRAIN_PATH)}")
    print(f"Testing data exists: {os.path.exists(TEST_PATH)}")
    print(f"Output directory exists: {os.path.exists(OUTPUT_DIR)}")
    print("=" * 60)

    # Load data
    train_df, test_df = load_data()

    # Preprocess data
    preprocess_result = preprocess_data(train_df, test_df, FEATURES, TARGET)
    if len(preprocess_result) == 8:
        X_train, X_val, X_test, y_train, y_val, y_test, features, test_data_by_hardware = preprocess_result
    else:
        raise ValueError(f"preprocess_data returned {len(preprocess_result)} values, expected 8")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=1024
    )

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
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs=2500, learning_rate=0.001
    )

    # Evaluate model by hardware type if available
    hardware_results = None
    if test_data_by_hardware:
        print("\n" + "="*60)
        print("SEPARATE HARDWARE EVALUATION")
        print("="*60)
        hardware_results = evaluate_model_by_hardware(model, test_data_by_hardware)

    # Evaluate model (combined - overall)
    print("\n" + "="*60)
    print("COMBINED EVALUATION (ALL TEST HARDWARE)")
    print("="*60)
    metrics = evaluate_model(model, test_loader, features)

    # Analyze errors
    worst_cases, results_df = analyze_errors(
        metrics['predictions'], metrics['actuals'], metrics['test_features'], features,
        test_df, OUTPUT_DIR
    )

    # Save comprehensive metadata
    metadata = save_metadata(
        features, train_df, test_df, metrics,
        hardware_results=hardware_results, model_params=model_params
    )

    print(f"\nTraining and evaluation completed.")
    print(f"All files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
