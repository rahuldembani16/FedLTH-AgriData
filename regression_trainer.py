from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.utils.dataset.functional import split_indices
from fedlab.utils.functional import partition_report
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
from fedlab.utils import Logger, SerializationTool
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionSerialTrainer(SerialClientTrainer):
    """
    Modified SerialClientTrainer for regression tasks (crop yield prediction)
    Supports tabular data and regression-specific metrics
    """

    def __init__(self,
                 model,
                 dataset,
                 data_slices,
                 logger=None,
                 cuda=False,
                 gpu=None,
                 args=None) -> None:

        super(RegressionSerialTrainer, self).__init__(model=model,
                                                     num_clients=len(data_slices),
                                                     cuda=cuda)
        self.gpu = gpu
        self.dataset = dataset
        self.data_slices = data_slices
        self.args = args
        self.logger = logger if logger is not None else Logger()
        self._LOGGER = self.logger

    def local_process(self, id_list, model_data):
        """Train local model with different dataset according to client id in ``id_list``."""
        self.param_list = []
        model_parameters = model_data[0]

        self._LOGGER.info(
            "Local training with client id list: {}".format(id_list))
        
        for idx in id_list:
            self._LOGGER.info(
                "Starting training procedure of client [{}]".format(idx))

            data_loader = self._get_dataloader(client_id=idx)
            self._train_alone(model_parameters=model_parameters,
                              train_loader=data_loader)
            self.param_list.append(self.model_parameters)
        return self.param_list

    def _get_dataloader(self, client_id):
        """Return a training dataloader for specific client"""
        batch_size = self.args["batch_size"]

        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(indices=self.data_slices[client_id]),
            batch_size=batch_size,
            shuffle=False,  # SubsetRandomSampler handles shuffling
            num_workers=0,  # Reduced for stability
            pin_memory=self.cuda,
            drop_last=True  # Drop last incomplete batch to avoid BN issues
        )

        return train_loader

    def _train_alone(self, model_parameters, train_loader):
        """Single round of local training for one client (regression version)"""
        epochs, lr = self.args["epochs"], self.args["lr"]
        
        # Deserialize model parameters
        SerializationTool.deserialize_model(self._model, model_parameters)
        
        # Use MSE loss for regression
        criterion = torch.nn.MSELoss()
        
        # Use Adam optimizer with weight decay for better generalization
        optimizer = torch.optim.Adam(
            self._model.parameters(), 
            lr=lr, 
            weight_decay=self.args.get("weight_decay", 1e-5)
        )
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.8
        )
        
        self._model.train()
        
        # Handle batch normalization for small batches
        for module in self._model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.track_running_stats = False  # Disable running stats for small batches

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data = data.cuda(self.gpu)
                    target = target.cuda(self.gpu)

                # Forward pass
                optimizer.zero_grad()
                output = self.model(data)
                
                # Ensure target and output have compatible shapes
                if target.dim() > 1:
                    target = target.squeeze(-1)
                if output.dim() > 1:
                    output = output.squeeze(-1)
                
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Update learning rate
            avg_loss = total_loss / max(num_batches, 1)
            scheduler.step(avg_loss)

        # Serialize and store model parameters
        self.set_model(SerializationTool.serialize_model(self._model))
        return self.model_parameters

def evaluate_regression_model(model, criterion, test_loader, cuda=False, gpu=None):
    """
    Evaluate regression model with appropriate metrics
    """
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data = data.cuda(gpu)
                target = target.cuda(gpu)
            
            output = model(data)
            
            # Ensure compatible shapes
            if target.dim() > 1:
                target = target.squeeze(-1)
            if output.dim() > 1:
                output = output.squeeze(-1)
            
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # Store predictions and targets for metrics calculation
            if cuda:
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
            else:
                predictions.extend(output.numpy())
                targets.extend(target.numpy())
    
    # Calculate regression metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # Calculate MAPE (Mean Absolute Percentage Error) safely
    non_zero_mask = targets != 0
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    avg_loss = test_loss / len(test_loader)
    
    metrics = {
        'loss': avg_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    return metrics

class RegressionMetricsTracker:
    """Track and display regression metrics during training"""
    
    def __init__(self):
        self.history = {
            'loss': [],
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': [],
            'mape': []
        }
    
    def update(self, metrics):
        """Update metrics history"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best_metrics(self):
        """Get best metrics achieved so far"""
        if not self.history['loss']:
            return None
        
        best_idx = np.argmin(self.history['loss'])
        best_metrics = {}
        for key, values in self.history.items():
            if values:
                best_metrics[f'best_{key}'] = values[best_idx]
        
        return best_metrics
    
    def print_current_metrics(self, round_num, metrics):
        """Print current round metrics in a formatted way"""
        print(f"ROUND {round_num}: Loss: {metrics['loss']:.4f}, RMSE: {metrics['rmse']:.4f}, "
              f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.2f}%")
    
    def print_summary(self):
        """Print training summary"""
        if not self.history['loss']:
            print("No metrics recorded.")
            return
        
        best_metrics = self.get_best_metrics()
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Best Loss: {best_metrics['best_loss']:.4f}")
        print(f"Best RMSE: {best_metrics['best_rmse']:.4f}")
        print(f"Best MAE: {best_metrics['best_mae']:.4f}")
        print(f"Best R²: {best_metrics['best_r2']:.4f}")
        print(f"Best MAPE: {best_metrics['best_mape']:.2f}%")
        print("="*60)

if __name__ == '__main__':
    print("Regression training utilities loaded successfully!")
