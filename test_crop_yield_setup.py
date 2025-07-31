#!/usr/bin/env python3
"""
Test script to verify the crop yield federated learning setup
"""

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def test_data_loading():
    """Test data loading and preprocessing"""
    print("Testing data loading and preprocessing...")
    
    try:
        from tabular_data_utils import create_crop_yield_datasets
        train_dataset, test_dataset, preprocessor = create_crop_yield_datasets(
            'dataset/crop_yield_10k_records.csv', test_size=0.2
        )
        
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Test dataset: {len(test_dataset)} samples")
        print(f"✓ Feature dimension: {train_dataset.features.shape[1]}")
        
        # Test a sample
        sample_features, sample_target = train_dataset[0]
        print(f"✓ Sample features shape: {sample_features.shape}")
        print(f"✓ Sample target: {sample_target:.3f}")
        
        return train_dataset, test_dataset, preprocessor
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None, None

def test_model():
    """Test model creation and forward pass"""
    print("\nTesting model creation...")
    
    try:
        from models.crop_yield_model import CropYieldMLP
        
        model = CropYieldMLP(input_dim=21, hidden_dims=[128, 64, 32])
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created with {total_params:,} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(10, 21)
        output = model(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return None

def test_partitioner():
    """Test data partitioning"""
    print("\nTesting data partitioning...")
    
    try:
        from tabular_data_utils import CropYieldPartitioner
        
        # Create dummy targets
        targets = torch.randn(1000)
        
        partitioner = CropYieldPartitioner(
            targets=targets,
            num_clients=5,
            partition="dirichlet",
            alpha=0.5
        )
        
        print(f"✓ Partitioner created for {len(partitioner)} clients")
        
        total_samples = 0
        for i in range(len(partitioner)):
            client_data = partitioner[i]
            total_samples += len(client_data)
            print(f"  Client {i}: {len(client_data)} samples")
        
        print(f"✓ Total distributed samples: {total_samples}/1000")
        
        return partitioner
        
    except Exception as e:
        print(f"❌ Partitioner test failed: {e}")
        return None

def test_pruning():
    """Test pruning utilities"""
    print("\nTesting pruning utilities...")
    
    try:
        from tabular_pruning_utils import TabularFedLTHPruner
        from models.crop_yield_model import CropYieldMLP
        
        model = CropYieldMLP(input_dim=21, hidden_dims=[64, 32])
        pruner = TabularFedLTHPruner(start_ratio=0.2, end_ratio=0.5)
        
        initial_sparsity = pruner.check_sparsity(model)
        print(f"✓ Initial sparsity: {100-initial_sparsity:.1f}%")
        
        pruner.unstructured_prune(model)
        final_sparsity = pruner.check_sparsity(model)
        print(f"✓ After pruning sparsity: {100-final_sparsity:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Pruning test failed: {e}")
        return False

def test_trainer():
    """Test regression trainer"""
    print("\nTesting regression trainer...")
    
    try:
        from regression_trainer import RegressionSerialTrainer, evaluate_regression_model
        from models.crop_yield_model import CropYieldMLP
        from tabular_data_utils import create_crop_yield_datasets, CropYieldPartitioner
        
        # Load small dataset
        train_dataset, test_dataset, _ = create_crop_yield_datasets(
            'dataset/crop_yield_10k_records.csv', test_size=0.8  # Use only 20% for quick test
        )
        
        # Create partitioner
        partitioner = CropYieldPartitioner(
            targets=train_dataset.targets,
            num_clients=2,
            partition="iid"
        )
        
        # Create model
        model = CropYieldMLP(input_dim=train_dataset.features.shape[1], hidden_dims=[32, 16])
        
        # Create trainer
        trainer = RegressionSerialTrainer(
            model=model,
            dataset=train_dataset,
            data_slices=partitioner,
            args={
                "batch_size": 32,
                "epochs": 1,
                "lr": 0.01,
                "weight_decay": 1e-5
            },
            cuda=False
        )
        
        print(f"✓ Trainer created")
        
        # Test evaluation
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
        criterion = torch.nn.MSELoss()
        
        metrics = evaluate_regression_model(model, criterion, test_loader, cuda=False)
        print(f"✓ Initial RMSE: {metrics['rmse']:.3f}")
        print(f"✓ Initial R²: {metrics['r2']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        return False

def test_integration():
    """Test a mini federated learning run"""
    print("\nTesting mini federated learning integration...")
    
    try:
        from fedlab.utils.aggregator import Aggregators
        from fedlab.utils.serialization import SerializationTool
        from regression_trainer import RegressionSerialTrainer, evaluate_regression_model
        from models.crop_yield_model import CropYieldMLP
        from tabular_data_utils import create_crop_yield_datasets, CropYieldPartitioner
        
        # Load dataset
        train_dataset, test_dataset, _ = create_crop_yield_datasets(
            'dataset/crop_yield_10k_records.csv', test_size=0.9  # Use only 10% for quick test
        )
        
        # Create partitioner
        partitioner = CropYieldPartitioner(
            targets=train_dataset.targets,
            num_clients=3,
            partition="iid"
        )
        
        # Create model
        model = CropYieldMLP(input_dim=train_dataset.features.shape[1], hidden_dims=[32])
        
        # Create trainer
        trainer = RegressionSerialTrainer(
            model=model,
            dataset=train_dataset,
            data_slices=partitioner,
            args={
                "batch_size": 32,
                "epochs": 1,
                "lr": 0.01
            },
            cuda=False
        )
        
        # Test federated round
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
        criterion = torch.nn.MSELoss()
        
        # Initial evaluation
        initial_metrics = evaluate_regression_model(model, criterion, test_loader, cuda=False)
        print(f"✓ Initial RMSE: {initial_metrics['rmse']:.3f}")
        
        # Run one federated round
        model_parameters = SerializationTool.serialize_model(model)
        parameters_list = trainer.local_process(
            model_data=[model_parameters],
            id_list=[0, 1, 2]  # All clients
        )
        
        # Aggregate
        aggregator = Aggregators.fedavg_aggregate
        SerializationTool.deserialize_model(model, aggregator(parameters_list))
        
        # Final evaluation
        final_metrics = evaluate_regression_model(model, criterion, test_loader, cuda=False)
        print(f"✓ Final RMSE: {final_metrics['rmse']:.3f}")
        
        improvement = initial_metrics['rmse'] - final_metrics['rmse']
        print(f"✓ RMSE change: {improvement:+.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("CROP YIELD FEDERATED LEARNING - SYSTEM TEST")
    print("="*60)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Data loading
    train_dataset, test_dataset, preprocessor = test_data_loading()
    if train_dataset is not None:
        tests_passed += 1
    
    # Test 2: Model
    model = test_model()
    if model is not None:
        tests_passed += 1
    
    # Test 3: Partitioner
    partitioner = test_partitioner()
    if partitioner is not None:
        tests_passed += 1
    
    # Test 4: Pruning
    if test_pruning():
        tests_passed += 1
    
    # Test 5: Trainer
    if test_trainer():
        tests_passed += 1
    
    # Test 6: Integration
    if test_integration():
        tests_passed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    print(f"{'='*60}")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! The system is ready for crop yield federated learning.")
        print("\nTo run the full training, use:")
        print("python fedlth_crop_yield.py --com_round 100 --total_client 8")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        
    return tests_passed == total_tests

if __name__ == "__main__":
    main()
