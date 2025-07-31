import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

class CropYieldMLP(nn.Module):
    """
    Multi-Layer Perceptron for Crop Yield Regression
    Designed for tabular data with mixed categorical and numerical features
    """
    def __init__(self, input_dim=21, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3):
        super(CropYieldMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with batch normalization and dropout
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if hidden_dim > 1 else nn.Identity(),  # Avoid BN for single neurons
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate if i < len(hidden_dims) - 1 else dropout_rate/2)
            ])
            prev_dim = hidden_dim
        
        # Output layer for regression (single value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Ensure input has correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x).squeeze(-1)

class CropYieldDataPreprocessor:
    """
    Preprocessor for crop yield data handling categorical encoding and feature scaling
    """
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, df):
        """Fit the preprocessor on training data"""
        df_copy = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['Region', 'Crop_Type', 'Soil_Type']
        for col in categorical_cols:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])
            self.label_encoders[col] = le
        
        # Separate features and target
        if 'Yield_tons_per_hectare' in df_copy.columns:
            X = df_copy.drop('Yield_tons_per_hectare', axis=1)
        else:
            X = df_copy
        
        # Create interaction features
        X = self._create_interaction_features(X)
        
        # Fit scaler on numerical features (skip encoded categoricals)
        numerical_cols = ['Rainfall_mm', 'Temperature_C', 'Soil_pH', 
                         'Fertilizer_kg_per_hectare', 'Pesticide_kg_per_hectare']
        interaction_cols = [col for col in X.columns if 'interaction' in col]
        cols_to_scale = numerical_cols + interaction_cols
        
        # Only scale if columns exist
        cols_to_scale = [col for col in cols_to_scale if col in X.columns]
        
        if cols_to_scale:
            self.scaler.fit(X[cols_to_scale])
        
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        
        return self
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df_copy = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['Region', 'Crop_Type', 'Soil_Type']
        for col in categorical_cols:
            if col in df_copy.columns:
                # Handle unseen categories by assigning them to the most frequent category
                le = self.label_encoders[col]
                mask = df_copy[col].isin(le.classes_)
                df_copy.loc[~mask, col] = le.classes_[0]  # Assign to first class for unseen
                df_copy[col] = le.transform(df_copy[col])
        
        # Separate features and target
        y = None
        if 'Yield_tons_per_hectare' in df_copy.columns:
            y = df_copy['Yield_tons_per_hectare'].values
            X = df_copy.drop('Yield_tons_per_hectare', axis=1)
        else:
            X = df_copy
        
        # Create interaction features
        X = self._create_interaction_features(X)
        
        # Scale numerical features
        numerical_cols = ['Rainfall_mm', 'Temperature_C', 'Soil_pH', 
                         'Fertilizer_kg_per_hectare', 'Pesticide_kg_per_hectare']
        interaction_cols = [col for col in X.columns if 'interaction' in col]
        cols_to_scale = numerical_cols + interaction_cols
        cols_to_scale = [col for col in cols_to_scale if col in X.columns]
        
        if cols_to_scale:
            X_scaled = X.copy()
            X_scaled[cols_to_scale] = self.scaler.transform(X[cols_to_scale])
        else:
            X_scaled = X
        
        # Ensure feature order matches training
        X_scaled = X_scaled.reindex(columns=self.feature_names, fill_value=0)
        
        return X_scaled.values, y
    
    def _create_interaction_features(self, X):
        """Create meaningful interaction features"""
        X_copy = X.copy()
        
        # Key agricultural interactions
        if all(col in X_copy.columns for col in ['Rainfall_mm', 'Temperature_C']):
            X_copy['rainfall_temp_interaction'] = X_copy['Rainfall_mm'] * X_copy['Temperature_C']
        
        if all(col in X_copy.columns for col in ['Fertilizer_kg_per_hectare', 'Soil_pH']):
            X_copy['fertilizer_ph_interaction'] = X_copy['Fertilizer_kg_per_hectare'] * X_copy['Soil_pH']
        
        if all(col in X_copy.columns for col in ['Rainfall_mm', 'Fertilizer_kg_per_hectare']):
            X_copy['rainfall_fertilizer_interaction'] = X_copy['Rainfall_mm'] * X_copy['Fertilizer_kg_per_hectare']
        
        # Climate stress indicator
        if all(col in X_copy.columns for col in ['Temperature_C', 'Rainfall_mm']):
            X_copy['climate_stress'] = (X_copy['Temperature_C'] - 25) ** 2 + (X_copy['Rainfall_mm'] - 120) ** 2
        
        return X_copy

if __name__ == '__main__':
    # Test the model
    model = CropYieldMLP(input_dim=21)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(32, 21)  # Batch size 32, 21 features
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
