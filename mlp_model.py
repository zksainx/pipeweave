#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP model architecture for overall_perf prediction

This module contains the shared MLP model architecture used across
different operator types (GEMM, Attention, RMSNorm, SiLUAndMul).
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for overall_perf prediction

    Architecture designed for workload characteristics:
    - Fully connected layers with ReLU, BatchNorm and Dropout
    - Post-activation BatchNorm (Linear -> ReLU -> BatchNorm -> Dropout)
    - Output layer with Sigmoid activation to ensure predictions in [0, 1] range
    """

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.1):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Input layer
        layers = []
        prev_dim = input_dim

        # Build hidden layers with post-activation BatchNorm
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer with sigmoid activation to ensure 0-1 range
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLP_v2(nn.Module):
    """
    Multi-Layer Perceptron for log_cycle prediction (v2)

    Architecture designed for workload characteristics:
    - Fully connected layers with ReLU, BatchNorm and Dropout
    - Post-activation BatchNorm (Linear -> ReLU -> BatchNorm -> Dropout)
    - Output layer WITHOUT Sigmoid activation (for unbounded regression)
    """

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.1):
        super(MLP_v2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Input layer
        layers = []
        prev_dim = input_dim

        # Build hidden layers with post-activation BatchNorm
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer without sigmoid activation for unbounded regression
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
