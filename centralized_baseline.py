#!/usr/bin/env python3
"""
Centralized baseline for comparison with federated learning results
"""

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

from models.crop_yield_model import CropYieldMLP, CropYieldDataPreprocessor
from tabular_data_utils import create_crop_yield_datasets
from regression_trainer import evaluate_regression_model

def train_centralized_model():
    """Train centralized model for baseline comparison"""
    
    # Load data
    train_dataset, test_dataset, preprocessor = create_crop_yield_datasets(
        'dataset/crop_yield_10k_records.csv', test_size=0.2, random_state=42
    )
    
    # Create model with same architecture as federated version
    model = CropYieldMLP(
        input_dim=train_dataset.features.shape[1],
        hidden_dims=[512, 256, 128, 64, 32],
        dropout_rate=0.25
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0
    )
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.003, 
        weight_decay=5e-6
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.8
    )
    
    # Training loop
    model.train()
    best_rmse = float('inf')
    patience_counter = 0
    
    for epoch in range(500):  # More epochs for centralized training
        total_loss = 0.0
        num_batches = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            
            if target.dim() > 1:
                target = target.squeeze(-1)
            if output.dim() > 1:
                output = output.squeeze(-1)
            
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            metrics = evaluate_regression_model(model, criterion, test_loader, cuda=False)
            current_rmse = metrics['rmse']
            
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_metrics = metrics
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch}: Loss: {avg_loss:.4f}, RMSE: {current_rmse:.4f}, "
                  f"R²: {metrics['r2']:.4f}, Best RMSE: {best_rmse:.4f}")
            
            # Early stopping
            if patience_counter >= 20:
                print("Early stopping triggered")
                break
    
    print("\n" + "="*60)
    print("CENTRALIZED BASELINE RESULTS")
    print("="*60)
    print(f"Best RMSE: {best_metrics['rmse']:.4f}")
    print(f"Best MAE: {best_metrics['mae']:.4f}")
    print(f"Best R²: {best_metrics['r2']:.4f}")
    print(f"Best MAPE: {best_metrics['mape']:.2f}%")
    print("="*60)
    
    return best_metrics

if __name__ == "__main__":
    train_centralized_model()
