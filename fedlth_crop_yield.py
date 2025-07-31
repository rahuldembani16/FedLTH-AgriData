#!/usr/bin/env python3
"""
FedLTH for Crop Yield Prediction - Federated Learning with Lottery Ticket Hypothesis on Tabular Data
"""

import os
import argparse
import random
from copy import deepcopy
import torch
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import get_best_gpu

# Import our custom modules
from models.crop_yield_model import CropYieldMLP, CropYieldDataPreprocessor
from tabular_data_utils import CropYieldDataset, CropYieldPartitioner, create_crop_yield_datasets
from regression_trainer import RegressionSerialTrainer, evaluate_regression_model, RegressionMetricsTracker
from tabular_pruning_utils import TabularFedLTHPruner
from conf import *

# Configuration
parser = argparse.ArgumentParser(description="FedLTH for Crop Yield Prediction")

# Server Global Config
parser.add_argument("--total_client", type=int, default=8, 
                   help="Total number of federated clients")
parser.add_argument("--com_round", type=int, default=150, 
                   help="Total communication rounds")
parser.add_argument("--pretrained", type=str, default=None,
                   help="Path to pretrained model")

# Dataset Config
parser.add_argument("--dataset", type=str, default="crop_yield",
                   help="Dataset to use (crop_yield)")
parser.add_argument("--data_path", type=str, default="dataset/crop_yield_10k_records.csv",
                   help="Path to crop yield dataset")
parser.add_argument("--partition", type=str, default="dirichlet", 
                   choices=["iid", "dirichlet", "geographic", "crop_based"],
                   help="Data partitioning strategy")
parser.add_argument("--alpha", type=float, default=0.5,
                   help="Dirichlet concentration parameter for non-IID")

# Client Config
parser.add_argument("--sample_ratio", type=float, default=1.0,
                   help="Fraction of clients to sample each round")

# Training Config
parser.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for training")
parser.add_argument("--epochs", type=int, default=3,
                   help="Local epochs per communication round")
parser.add_argument("--lr", type=float, default=0.001,
                   help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5,
                   help="Weight decay for regularization")
parser.add_argument("--cuda", type=bool, default=True,
                   help="Use CUDA if available")

# Model Config
parser.add_argument("--hidden_dims", type=int, nargs='+', default=[256, 128, 64, 32],
                   help="Hidden layer dimensions")
parser.add_argument("--dropout_rate", type=float, default=0.3,
                   help="Dropout rate")

# Pruning Config
parser.add_argument("--enable_pruning", type=bool, default=True,
                   help="Enable pruning")
parser.add_argument("--prune_start_round", type=int, default=30,
                   help="Round to start pruning")
parser.add_argument("--prune_threshold", type=float, default=2.0,
                   help="RMSE threshold to start pruning")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parser.parse_args()

def create_model(input_dim, args):
    """Create crop yield prediction model"""
    return CropYieldMLP(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate
    )

def main():
    print("="*80)
    print("FedLTH Crop Yield Prediction - Federated Learning with Tabular Data")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Clients: {args.total_client}")
    print(f"  - Rounds: {args.com_round}")
    print(f"  - Partition: {args.partition} (alpha={args.alpha})")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Hidden dims: {args.hidden_dims}")
    print(f"  - Pruning: {'Enabled' if args.enable_pruning else 'Disabled'}")
    print("="*80)

    # Load and preprocess data
    print("Loading crop yield dataset...")
    try:
        train_dataset, test_dataset, preprocessor = create_crop_yield_datasets(
            args.data_path, test_size=0.2, random_state=42
        )
        print(f"✓ Train samples: {len(train_dataset)}")
        print(f"✓ Test samples: {len(test_dataset)}")
        print(f"✓ Feature dimension: {train_dataset.features.shape[1]}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Make sure the dataset file exists and is properly formatted.")
        return

    # Create data partitioner
    print(f"\nPartitioning data among {args.total_client} clients using {args.partition} strategy...")
    trainset_partition = CropYieldPartitioner(
        targets=train_dataset.targets,
        num_clients=args.total_client,
        partition=args.partition,
        alpha=args.alpha,
        min_samples_per_client=50,
        seed=42
    )
    
    # Print partition statistics
    print("Client data distribution:")
    for i in range(args.total_client):
        client_samples = len(trainset_partition[i])
        client_targets = train_dataset.targets[trainset_partition[i]]
        mean_yield = torch.mean(client_targets).item()
        std_yield = torch.std(client_targets).item()
        print(f"  Client {i:2d}: {client_samples:4d} samples, "
              f"mean yield: {mean_yield:.2f} ± {std_yield:.2f}")

    # Create test data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        num_workers=0
    )

    # Create model
    input_dim = train_dataset.features.shape[1]
    model = create_model(input_dim, args)
    
    print(f"\nModel architecture:")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Hidden layers: {args.hidden_dims}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained model if specified
    if args.pretrained:
        try:
            model.load_state_dict(torch.load(args.pretrained))
            print(f"✓ Loaded pretrained model from {args.pretrained}")
        except Exception as e:
            print(f"⚠️  Failed to load pretrained model: {e}")

    # Setup device
    if args.cuda and torch.cuda.is_available():
        gpu = get_best_gpu()
        print(f"✓ Using GPU: {gpu}")
        model = model.cuda(gpu)
    else:
        print("✓ Using CPU")
        args.cuda = False
        gpu = None

    # Create federated trainer
    trainer = RegressionSerialTrainer(
        model=model,
        dataset=train_dataset,
        data_slices=trainset_partition,
        args={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay
        },
        cuda=args.cuda,
        gpu=gpu
    )

    # Setup federated learning
    num_per_round = int(args.total_client * args.sample_ratio)
    aggregator = Aggregators.fedavg_aggregate
    to_select = list(range(args.total_client))
    
    # Initial evaluation
    criterion = torch.nn.MSELoss()
    initial_metrics = evaluate_regression_model(model, criterion, test_loader, args.cuda, gpu)
    print(f"\nInitial Performance:")
    print(f"  RMSE: {initial_metrics['rmse']:.4f}")
    print(f"  MAE: {initial_metrics['mae']:.4f}")
    print(f"  R²: {initial_metrics['r2']:.4f}")

    # Setup pruning
    pruner = None
    if args.enable_pruning:
        pruner = TabularFedLTHPruner(
            start_ratio=start_ratio,
            end_ratio=end_ratio,
            device='cpu' if not args.cuda else gpu,
            min_increase=min_inscrease
        )
        print(f"\nPruning configuration:")
        print(f"  - Start ratio: {start_ratio}")
        print(f"  - End ratio: {end_ratio}")
        print(f"  - Threshold RMSE: {args.prune_threshold}")

    # Training tracking
    metrics_tracker = RegressionMetricsTracker()
    best_rmse = float('inf')
    best_model_state = None
    best_model_architecture = None
    early_train_phase = True
    sp_round = 0

    print(f"\n{'='*80}")
    print("STARTING FEDERATED TRAINING")
    print(f"{'='*80}")

    # Main training loop
    for round_num in range(1, args.com_round + 1):
        # Check if we should start pruning
        current_rmse = initial_metrics['rmse'] if round_num == 1 else metrics['rmse']
        
        if early_train_phase and current_rmse <= args.prune_threshold:
            early_train_phase = False
            print(f"✓ RMSE reached threshold ({args.prune_threshold:.3f}), pruning phase begins")

        # Select clients for this round
        if early_train_phase or not args.enable_pruning:
            # Use all clients in early phase
            selection = to_select
        else:
            # Sample clients for pruning phase
            selection = random.sample(to_select, num_per_round)

        # Serialize model for distribution
        model_parameters = SerializationTool.serialize_model(model)
        
        # Local training on selected clients
        parameters_list = trainer.local_process(
            model_data=[model_parameters],
            id_list=selection
        )
        
        # Aggregate updates
        SerializationTool.deserialize_model(model, aggregator(parameters_list))

        # Evaluate model
        metrics = evaluate_regression_model(model, criterion, test_loader, args.cuda, gpu)
        metrics_tracker.update(metrics)
        metrics_tracker.print_current_metrics(round_num, metrics)

        # Track best performance (save both state and architecture info)
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_model_state = deepcopy(model.state_dict())
            # Save architecture info for reconstruction
            best_model_architecture = {
                'input_dim': input_dim,
                'hidden_dims': [layer.out_features for name, layer in model.named_modules() 
                               if isinstance(layer, torch.nn.Linear) and 'network' in name][:-1],  # Exclude output layer
                'dropout_rate': args.dropout_rate
            }

        # Apply pruning if enabled and conditions are met
        if (args.enable_pruning and not early_train_phase and 
            round_num >= args.prune_start_round and sp_round < max_sp_rounds):
            
            if round_num % prune_step == 0:
                print(f"\n🔧 Applying unstructured pruning (round {sp_round+1}/{max_sp_rounds})...")
                pruner.unstructured_prune(model, skip_first_layer=True, skip_last_layer=True)
                weight_with_mask = deepcopy(model.state_dict())
                pruner.remove_prune(model, skip_first_layer=True, skip_last_layer=True)
                
                current_sparsity = pruner.check_sparsity(model)
                print(f"Current sparsity: {100-current_sparsity:.1f}%")

            # Check if we should do structured pruning
            if pruner.ratio >= pruner.end_ratio and pruner.unpruned_flag:
                print(f"\n🔧 Applying structured pruning...")
                
                # Create training loader for structured pruning
                train_loader_full = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
                )
                
                new_model, final_sparsity = pruner.structured_prune(
                    model, weight_with_mask, train_loader_full, criterion, output_dim=1
                )
                
                if new_model is not model:  # If a new model was created
                    model = new_model
                    if args.cuda:
                        model = model.cuda(gpu)
                    
                    print(f"✓ Structured pruning complete, final sparsity: {100*final_sparsity:.1f}%")
                    
                    # Reinitialize trainer with new model
                    trainer = RegressionSerialTrainer(
                        model=model,
                        dataset=train_dataset,
                        data_slices=trainset_partition,
                        args={
                            "batch_size": args.batch_size,
                            "epochs": args.epochs,
                            "lr": args.lr,
                            "weight_decay": args.weight_decay
                        },
                        cuda=args.cuda,
                        gpu=gpu
                    )
                    
                    # Reset pruning parameters
                    pruner.ratio = pruner.start_ratio
                    pruner.unpruned_flag = False
                    sp_round += 1

        # Early stopping check
        if round_num > 50 and len(metrics_tracker.history['rmse']) > 10:
            recent_rmse = metrics_tracker.history['rmse'][-10:]
            if max(recent_rmse) - min(recent_rmse) < 0.01:  # Convergence check
                print(f"\n⚠️  Training converged at round {round_num}")
                break

    print(f"\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}")
    
    # Create the best model with correct architecture and load its state
    if best_model_architecture is not None:
        print("Reconstructing best model with correct architecture...")
        best_model = CropYieldMLP(
            input_dim=best_model_architecture['input_dim'],
            hidden_dims=best_model_architecture['hidden_dims'],
            dropout_rate=best_model_architecture['dropout_rate']
        )
        if args.cuda:
            best_model = best_model.cuda(gpu)
        
        try:
            best_model.load_state_dict(best_model_state)
            final_metrics = evaluate_regression_model(best_model, criterion, test_loader, args.cuda, gpu)
            model = best_model  # Use best model for saving
        except Exception as e:
            print(f"⚠️  Could not load best model state: {e}")
            print("Using current model instead.")
            final_metrics = evaluate_regression_model(model, criterion, test_loader, args.cuda, gpu)
    else:
        print("Using current model (no architecture changes occurred)")
        final_metrics = evaluate_regression_model(model, criterion, test_loader, args.cuda, gpu)
    
    print(f"Final Results:")
    print(f"  Best RMSE: {best_rmse:.4f}")
    print(f"  Final RMSE: {final_metrics['rmse']:.4f}")
    print(f"  Final MAE: {final_metrics['mae']:.4f}")
    print(f"  Final R²: {final_metrics['r2']:.4f}")
    print(f"  Final MAPE: {final_metrics['mape']:.2f}%")

    # Print training summary
    metrics_tracker.print_summary()

    # Save model and results
    save_dir = f"saved_models/crop_yield_fedlth"
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = f"{save_dir}/model_clients{args.total_client}_rounds{args.com_round}.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save preprocessor
    import pickle
    preprocessor_path = f"{save_dir}/preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Preprocessor saved to: {preprocessor_path}")
    print(f"✓ Training completed successfully!")

if __name__ == "__main__":
    main()
